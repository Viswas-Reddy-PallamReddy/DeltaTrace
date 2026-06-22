"""End-to-end correctness: every committed version must reconstruct exactly."""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import assert_equivalent, build_versions

from deltatrace import DeltaRepo
from deltatrace.storage import StoreError


def _make_repo(tmp_path, **kwargs):
    return DeltaRepo.init(tmp_path / "store", primary_key="id", overwrite=True, **kwargs)


def test_roundtrip_all_versions(tmp_path):
    keys, versions, expected = build_versions()
    repo = _make_repo(tmp_path)
    for df, msg in versions:
        repo.commit(df, message=msg)

    assert repo.head == len(versions)
    for v in range(1, len(versions) + 1):
        got = repo.checkout(v)
        assert_equivalent(got, expected[v], keys)


def test_reopen_persists_head_and_key(tmp_path):
    keys, versions, expected = build_versions()
    repo = _make_repo(tmp_path)
    for df, msg in versions[:3]:
        repo.commit(df, message=msg)

    reopened = DeltaRepo.open(tmp_path / "store")
    assert reopened.head == 3
    assert reopened.primary_key == ["id"]
    assert_equivalent(reopened.checkout(3), expected[3], keys)

    # continue committing on the reopened repo
    reopened.commit(versions[3][0], message=versions[3][1])
    assert reopened.head == 4
    assert_equivalent(reopened.checkout(4), expected[4], keys)


def test_head_default_checkout(tmp_path):
    keys, versions, expected = build_versions()
    repo = _make_repo(tmp_path)
    for df, msg in versions:
        repo.commit(df, message=msg)
    assert_equivalent(repo.checkout(), expected[len(versions)], keys)


def test_commit_stats(tmp_path):
    _, versions, _ = build_versions()
    repo = _make_repo(tmp_path)
    repo.commit(versions[0][0])
    r2 = repo.commit(versions[1][0])  # appended 6,7
    assert r2.stats["added"] == 2 and r2.stats["deleted"] == 0
    r3 = repo.commit(versions[2][0])  # deleted 2,4
    assert r3.stats["deleted"] == 2
    r4 = repo.commit(versions[3][0])  # updated 1,6
    assert r4.stats["updated"] == 2
    r5 = repo.commit(versions[4][0])  # add duration
    assert r5.stats["columns_added"] == 1
    r6 = repo.commit(versions[5][0])  # drop vendor
    assert r6.stats["columns_removed"] == 1
    r7 = repo.commit(versions[6][0])  # dtype change
    assert r7.stats["types_changed"] == 1


def test_log_records_every_commit(tmp_path):
    _, versions, _ = build_versions()
    repo = _make_repo(tmp_path)
    for df, msg in versions:
        repo.commit(df, message=msg)
    log = repo.log()
    assert [e["version"] for e in log] == list(range(1, len(versions) + 1))
    assert log[1]["message"] == "append rows 6,7"


def test_composite_primary_key(tmp_path):
    repo = DeltaRepo.init(tmp_path / "ck", primary_key=["region", "id"], overwrite=True)
    a = pd.DataFrame({"region": ["x", "x", "y"], "id": [1, 2, 1], "val": [10, 20, 30]})
    b = a.copy()
    b.loc[b["id"] == 2, "val"] = 99           # update (x,2)
    b = pd.concat([b, pd.DataFrame({"region": ["y"], "id": [2], "val": [40]})], ignore_index=True)
    repo.commit(a)
    res = repo.commit(b)
    assert res.stats["updated"] == 1 and res.stats["added"] == 1
    assert_equivalent(repo.checkout(2), b, ["region", "id"])


def test_duplicate_primary_key_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    dup = pd.DataFrame({"id": [1, 1], "tip": [1.0, 2.0]})
    with pytest.raises(ValueError):
        repo.commit(dup)


def test_missing_primary_key_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    with pytest.raises(KeyError):
        repo.commit(pd.DataFrame({"tip": [1.0]}))


def test_init_existing_without_overwrite(tmp_path):
    DeltaRepo.init(tmp_path / "s", primary_key="id")
    with pytest.raises(StoreError):
        DeltaRepo.init(tmp_path / "s", primary_key="id")


def test_checkout_out_of_range(tmp_path):
    repo = _make_repo(tmp_path)
    repo.commit(build_versions()[1][0][0])
    with pytest.raises(StoreError):
        repo.checkout(99)


def test_diff_between_versions(tmp_path):
    _, versions, _ = build_versions()
    repo = _make_repo(tmp_path)
    for df, msg in versions:
        repo.commit(df, message=msg)
    d = repo.diff(1, 2)
    assert d.rows_added == 2 and d.rows_deleted == 0
    d51 = repo.diff(4, 5)
    assert d51.columns_added == ["duration"]
    d67 = repo.diff(6, 7)
    assert d67.type_changes and d67.type_changes[0]["column"] == "duration"


def test_audit_writes_cell_log(tmp_path):
    _, versions, _ = build_versions()
    repo = _make_repo(tmp_path)
    repo.commit(versions[0][0])
    repo.commit(versions[1][0])
    repo.commit(versions[2][0])
    repo.commit(versions[3][0], audit=True)  # updates ids 1 and 6
    audit_path = tmp_path / "store" / "versions" / "v4" / "updated_cells.parquet"
    assert audit_path.exists()
    cells = pd.read_parquet(audit_path)
    assert set(cells["column"]) <= {"tip", "surcharge"}
    assert len(cells) >= 2
