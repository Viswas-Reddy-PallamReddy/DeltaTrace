"""Unit tests for the pure identity and diffing primitives."""

from __future__ import annotations

import pandas as pd
import pytest

from deltatrace.diffing import changed_row_ids, column_diff, row_diff, type_changes
from deltatrace.identity import RID, compute_row_ids, normalize_keys, with_row_ids


def test_normalize_keys():
    assert normalize_keys("id") == ["id"]
    assert normalize_keys(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        normalize_keys([])
    with pytest.raises(ValueError):
        normalize_keys(["a", "a"])


def test_compute_row_ids_single():
    rid = compute_row_ids(pd.DataFrame({"id": [1, 2, 3]}), ["id"])
    assert rid.tolist() == ["1", "2", "3"]
    assert rid.name == RID


def test_compute_row_ids_composite_unique():
    df = pd.DataFrame({"r": ["x", "y"], "id": [1, 1]})
    assert compute_row_ids(df, ["r", "id"]).nunique() == 2


def test_compute_row_ids_rejects_duplicates():
    with pytest.raises(ValueError):
        compute_row_ids(pd.DataFrame({"id": [1, 1]}), ["id"])


def test_compute_row_ids_rejects_nulls():
    with pytest.raises(ValueError):
        compute_row_ids(pd.DataFrame({"id": [1, None]}), ["id"])


def test_compute_row_ids_missing_column():
    with pytest.raises(KeyError):
        compute_row_ids(pd.DataFrame({"x": [1]}), ["id"])


def test_row_diff():
    d = row_diff(["1", "2", "3"], ["2", "3", "4"])
    assert d.appended == {"4"}
    assert d.deleted == {"1"}
    assert d.unchanged == {"2", "3"}


def test_column_diff_ignores_rid():
    c = column_diff(["id", "a", RID], ["id", "b"])
    assert c["added"] == ["b"]
    assert c["removed"] == ["a"]
    assert c["common"] == ["id"]


def test_type_changes():
    a = pd.DataFrame({"x": pd.Series([1], dtype="int64")})
    b = pd.DataFrame({"x": pd.Series([1], dtype="int32")})
    ch = type_changes(a, b, ["x"])
    assert ch == [{"column": "x", "old_type": "int64", "new_type": "int32"}]


def test_changed_row_ids():
    old = with_row_ids(pd.DataFrame({"id": [1, 2, 3], "v": [10, 20, 30]}), ["id"])
    new = with_row_ids(pd.DataFrame({"id": [1, 2, 3], "v": [10, 99, 30]}), ["id"])
    assert changed_row_ids(old, new, {"1", "2", "3"}, ["v"]) == ["2"]


def test_changed_row_ids_nan_aware():
    old = with_row_ids(pd.DataFrame({"id": [1, 2], "v": [None, 1.0]}), ["id"])
    new = with_row_ids(pd.DataFrame({"id": [1, 2], "v": [None, 1.0]}), ["id"])
    assert changed_row_ids(old, new, {"1", "2"}, ["v"]) == []
