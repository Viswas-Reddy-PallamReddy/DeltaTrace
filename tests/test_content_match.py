"""Keyless (content-matched) versioning + the error-confinement guarantee.

These tests exercise :class:`deltatrace.ContentMatchIdentity`, the strategy that
recovers row identity for tables with **no primary key**. The headline property
under test is *error confinement*: no matter how the matcher is (mis)configured,
every committed version still reconstructs exactly. Match quality only ever moves
storage and provenance, never correctness.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from deltatrace import ContentMatchIdentity, DeltaRepo


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def _as_multiset(df: pd.DataFrame, columns):
    sub = df[list(columns)]
    return sorted(tuple(rec) for rec in sub.itertuples(index=False, name=None))


def assert_same_rows(got: pd.DataFrame, expected: pd.DataFrame):
    """Keyless reconstruction fidelity = multiset equality of rows."""
    assert set(got.columns) == set(expected.columns), (
        f"columns differ: {set(got.columns)} != {set(expected.columns)}"
    )
    cols = list(expected.columns)
    assert _as_multiset(got, cols) == _as_multiset(expected, cols)


def _dir_size(path) -> int:
    return sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(path)
        for f in fs
    )


def build_keyless_chain(seed: int = 0):
    """A keyless 'people' table that is appended to, edited, and pruned.

    There is deliberately **no id column**. A hidden positional key is tracked
    only so the test can assert exactness; it is never given to the repo.
    """
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"],
            "city": ["NY", "LA", "SF", "NY", "LA", "SF", "NY", "LA"],
            "salary": [50, 60, 70, 80, 90, 55, 65, 75],
        }
    )
    versions = [base.copy()]
    cur = base.copy()
    for v in range(2, 7):
        cur = cur.copy().reset_index(drop=True)
        # update salary of two rows (the keyless-hard case)
        idx = rng.choice(len(cur), 2, replace=False)
        cur.loc[idx, "salary"] = cur.loc[idx, "salary"] + rng.integers(1, 5, 2)
        # delete one, insert one
        cur = cur.drop(cur.index[rng.integers(0, len(cur))]).reset_index(drop=True)
        cur = pd.concat(
            [cur, pd.DataFrame({"name": [f"New{v}"], "city": ["SF"], "salary": [40 + v]})],
            ignore_index=True,
        )
        versions.append(cur.copy())
    return versions


# --------------------------------------------------------------------------- #
# core correctness                                                            #
# --------------------------------------------------------------------------- #
def test_keyless_roundtrip_exact(tmp_path):
    versions = build_keyless_chain()
    repo = DeltaRepo.init(
        tmp_path / "k",
        identity=ContentMatchIdentity(block_on=["city"]),
        overwrite=True,
    )
    for i, df in enumerate(versions):
        repo.commit(df, f"v{i + 1}")
    assert repo.head == len(versions)
    assert repo.keyless is True
    for i, df in enumerate(versions):
        assert_same_rows(repo.checkout(i + 1), df)


def test_base_only_keyless(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    repo = DeltaRepo.init(tmp_path / "b", identity=ContentMatchIdentity(), overwrite=True)
    repo.commit(df)
    assert_same_rows(repo.checkout(1), df)


def test_pure_insert_and_pure_delete(tmp_path):
    repo = DeltaRepo.init(tmp_path / "id", identity=ContentMatchIdentity(), overwrite=True)
    v1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    v2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})  # +1 row
    v3 = pd.DataFrame({"a": [2], "b": ["y"]})                   # delete 2 rows
    for df in (v1, v2, v3):
        repo.commit(df)
    assert_same_rows(repo.checkout(1), v1)
    assert_same_rows(repo.checkout(2), v2)
    assert_same_rows(repo.checkout(3), v3)


# --------------------------------------------------------------------------- #
# the headline guarantee: error confinement                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.5, 0.9, 1.0])
def test_error_confinement_any_threshold(tmp_path, threshold):
    """Reconstruction is exact for *every* matcher threshold.

    threshold=0.0 makes the matcher accept almost any pairing (deliberately wrong
    matches); threshold=1.0 makes it accept none (every edit becomes
    delete+insert). Both extremes -- and everything between -- must still
    reconstruct every version exactly.
    """
    versions = build_keyless_chain(seed=3)
    repo = DeltaRepo.init(
        tmp_path / f"t{int(threshold * 100)}",
        identity=ContentMatchIdentity(block_on=["city"], row_threshold=threshold),
        overwrite=True,
    )
    for df in versions:
        repo.commit(df)
    for i, df in enumerate(versions):
        assert_same_rows(repo.checkout(i + 1), df)


def test_exact_only_still_reconstructs(tmp_path):
    """The hash-only baseline (no fuzzy pass) is also always exact."""
    versions = build_keyless_chain(seed=5)
    repo = DeltaRepo.init(
        tmp_path / "exact",
        identity=ContentMatchIdentity(block_on=["city"], exact_only=True),
        overwrite=True,
    )
    for df in versions:
        repo.commit(df)
    for i, df in enumerate(versions):
        assert_same_rows(repo.checkout(i + 1), df)


# --------------------------------------------------------------------------- #
# the contribution: fuzzy recovers updates -> smaller, better provenance      #
# --------------------------------------------------------------------------- #
def _wide_chain(seed=11, n=400, ncol=10, versions=6):
    rng = np.random.default_rng(seed)
    cols = {f"f{j}": rng.normal(0, 1, n).round(4) for j in range(ncol)}
    cols["cat"] = rng.choice(list("abcd"), n)
    base = pd.DataFrame(cols)
    chain = [base.copy()]
    cur = base.copy()
    for _ in range(versions - 1):
        cur = cur.copy().reset_index(drop=True)
        idx = rng.choice(len(cur), max(1, int(0.05 * len(cur))), replace=False)
        cur.loc[idx, "f0"] = (cur.loc[idx, "f0"] + rng.normal(0, 1, len(idx))).round(4)
        chain.append(cur.copy())
    return chain


def test_fuzzy_smaller_than_hash_only(tmp_path):
    chain = _wide_chain()

    def storage(exact_only):
        root = tmp_path / ("exact" if exact_only else "fuzzy")
        repo = DeltaRepo.init(
            root,
            identity=ContentMatchIdentity(block_on=["cat"], exact_only=exact_only),
            overwrite=True,
        )
        for df in chain:
            repo.commit(df)
        for i, df in enumerate(chain):
            assert_same_rows(repo.checkout(i + 1), df)
        return _dir_size(root)

    s_exact = storage(True)
    s_fuzzy = storage(False)
    # recovering identity (update vs delete+insert) must not cost storage
    assert s_fuzzy < s_exact


def test_fuzzy_records_updates_not_delete_insert(tmp_path):
    v1 = pd.DataFrame(
        {"name": ["Alice", "Bob", "Carol"], "city": ["NY", "LA", "SF"], "salary": [10, 20, 30]}
    )
    v2 = v1.copy()
    v2.loc[v2["name"] == "Bob", "salary"] = 25  # one cell edit, no key

    repo = DeltaRepo.init(tmp_path / "u", identity=ContentMatchIdentity(block_on=["city"]), overwrite=True)
    repo.commit(v1)
    res = repo.commit(v2)
    # the edit is seen as an update, not an add + delete
    assert res.stats["updated"] == 1
    assert res.stats["added"] == 0
    assert res.stats["deleted"] == 0
    prov = repo.explain_last()
    assert (prov["method"] == "fuzzy").sum() == 1
    assert_same_rows(repo.checkout(2), v2)


def test_exact_only_turns_update_into_delete_insert(tmp_path):
    v1 = pd.DataFrame(
        {"name": ["Alice", "Bob", "Carol"], "city": ["NY", "LA", "SF"], "salary": [10, 20, 30]}
    )
    v2 = v1.copy()
    v2.loc[v2["name"] == "Bob", "salary"] = 25

    repo = DeltaRepo.init(
        tmp_path / "x", identity=ContentMatchIdentity(block_on=["city"], exact_only=True), overwrite=True
    )
    repo.commit(v1)
    res = repo.commit(v2)
    # with no identity recovery, the edit looks like a brand-new row + a deletion
    assert res.stats["added"] == 1
    assert res.stats["deleted"] == 1
    assert res.stats["updated"] == 0
    assert_same_rows(repo.checkout(2), v2)  # still exact


# --------------------------------------------------------------------------- #
# persistence                                                                 #
# --------------------------------------------------------------------------- #
def test_content_match_config_roundtrip(tmp_path):
    versions = build_keyless_chain(seed=9)
    repo = DeltaRepo.init(
        tmp_path / "p",
        identity=ContentMatchIdentity(block_on=["city"], row_threshold=0.7, num_rel_tol=0.1),
        overwrite=True,
    )
    for df in versions[:3]:
        repo.commit(df)

    reopened = DeltaRepo.open(tmp_path / "p")
    assert reopened.keyless is True
    assert isinstance(reopened.identity, ContentMatchIdentity)
    assert reopened.identity.block_on == ["city"]
    assert reopened.identity._sim.row_threshold == 0.7
    assert_same_rows(reopened.checkout(3), versions[2])

    # can keep committing on the reopened keyless repo
    reopened.commit(versions[3])
    assert_same_rows(reopened.checkout(4), versions[3])


def test_init_rejects_both_key_and_identity(tmp_path):
    with pytest.raises(ValueError):
        DeltaRepo.init(
            tmp_path / "bad", primary_key="id", identity=ContentMatchIdentity()
        )


def test_init_requires_one_of_key_or_identity(tmp_path):
    with pytest.raises(ValueError):
        DeltaRepo.init(tmp_path / "none")


def test_duplicate_rows_are_handled(tmp_path):
    """Identical rows (no key) must round-trip with correct multiplicity."""
    v1 = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    v2 = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})  # one dup -> new
    repo = DeltaRepo.init(tmp_path / "dup", identity=ContentMatchIdentity(), overwrite=True)
    repo.commit(v1)
    repo.commit(v2)
    assert_same_rows(repo.checkout(1), v1)
    assert_same_rows(repo.checkout(2), v2)
