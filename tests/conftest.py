"""Shared fixtures: a synthetic 8-version dataset exercising every delta type.

This mirrors the progression in the original DeltaTrace notebook (append,
delete, update, add column, drop column, dtype change, delete) but uses small
self-contained data so the suite needs no external parquet files.
"""

from __future__ import annotations

import pandas as pd
import pytest


def build_versions():
    keys = ["id"]

    v1 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "vendor": ["A", "B", "A", "C", "B"],
            "tip": [1.0, 2.0, 0.5, 3.0, 1.5],
            "surcharge": [0.3, 0.3, 0.3, 0.3, 0.3],
        }
    )

    # v2: append two rows
    v2 = pd.concat(
        [
            v1,
            pd.DataFrame(
                {"id": [6, 7], "vendor": ["C", "A"], "tip": [2.5, 0.0], "surcharge": [0.3, 0.3]}
            ),
        ],
        ignore_index=True,
    )

    # v3: delete rows 2 and 4
    v3 = v2[~v2["id"].isin([2, 4])].reset_index(drop=True)

    # v4: update existing rows
    v4 = v3.copy()
    v4.loc[v4["id"] == 1, "tip"] = 9.9
    v4.loc[v4["id"] == 6, "surcharge"] = 1.0

    # v5: add a column
    v5 = v4.copy()
    v5["duration"] = v5["id"] * 10

    # v6: drop a column
    v6 = v5.drop(columns=["vendor"])

    # v7: change a column's dtype
    v7 = v6.copy()
    v7["duration"] = v7["duration"].astype("int32")

    # v8: delete another row
    v8 = v7[v7["id"] != 3].reset_index(drop=True)

    versions = [
        (v1, "initial load"),
        (v2, "append rows 6,7"),
        (v3, "delete rows 2,4"),
        (v4, "update tip/surcharge"),
        (v5, "add duration column"),
        (v6, "drop vendor column"),
        (v7, "shrink duration dtype"),
        (v8, "delete row 3"),
    ]
    expected = {i + 1: df for i, (df, _) in enumerate(versions)}
    return keys, versions, expected


@pytest.fixture
def versions_dataset():
    return build_versions()


def assert_equivalent(got: pd.DataFrame, expected: pd.DataFrame, keys):
    """Order-insensitive frame comparison keyed by the primary key."""
    assert set(got.columns) == set(expected.columns), (
        f"columns differ: {set(got.columns)} != {set(expected.columns)}"
    )
    g = got.sort_values(keys).reset_index(drop=True)
    e = expected.sort_values(keys).reset_index(drop=True)[g.columns.tolist()]
    pd.testing.assert_frame_equal(g, e, check_dtype=True)
