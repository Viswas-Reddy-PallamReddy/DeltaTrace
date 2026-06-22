"""Snapshots must match full replay and must short-circuit reconstruction."""

from __future__ import annotations

import shutil

from conftest import assert_equivalent, build_versions

from deltatrace import DeltaRepo


def _full_repo(tmp_path, **kwargs):
    keys, versions, expected = build_versions()
    repo = DeltaRepo.init(tmp_path / "s", primary_key="id", overwrite=True, **kwargs)
    for df, msg in versions:
        repo.commit(df, message=msg)
    return repo, keys, expected


def test_snapshot_matches_replay(tmp_path):
    repo, keys, expected = _full_repo(tmp_path)
    replay = repo.checkout(8)
    repo.snapshot(8)
    assert (tmp_path / "s" / "snapshots" / "v8.parquet").exists()
    assert_equivalent(repo.checkout(8), replay, keys)
    assert_equivalent(repo.checkout(8), expected[8], keys)


def test_snapshot_short_circuits_replay(tmp_path):
    """After snapshotting v8, deleting all deltas must not break checkout(8)."""
    repo, keys, expected = _full_repo(tmp_path)
    repo.snapshot(8)
    shutil.rmtree(tmp_path / "s" / "base")
    shutil.rmtree(tmp_path / "s" / "versions")
    assert_equivalent(repo.checkout(8), expected[8], keys)


def test_auto_snapshot_interval(tmp_path):
    repo, keys, expected = _full_repo(tmp_path, snapshot_interval=3)
    assert (tmp_path / "s" / "snapshots" / "v3.parquet").exists()
    assert (tmp_path / "s" / "snapshots" / "v6.parquet").exists()
    for v in range(1, 9):
        assert_equivalent(repo.checkout(v), expected[v], keys)
