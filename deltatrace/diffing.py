"""Pure diffing primitives shared by the repository and the test-suite.

Every function here is side-effect free and operates on DataFrames that already
carry the internal :data:`~deltatrace.identity.RID` column. Keeping them pure
makes the engine easy to reason about and to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

import pandas as pd

from .identity import RID


@dataclass
class RowDiff:
    appended: Set[str] = field(default_factory=set)
    deleted: Set[str] = field(default_factory=set)
    unchanged: Set[str] = field(default_factory=set)


def row_diff(old_ids: Sequence[str], new_ids: Sequence[str]) -> RowDiff:
    """Set-difference the row ids of two versions."""
    old, new = set(old_ids), set(new_ids)
    return RowDiff(appended=new - old, deleted=old - new, unchanged=old & new)


def column_diff(old_cols: Sequence[str], new_cols: Sequence[str]) -> Dict[str, List[str]]:
    """Compare column *sets* (order-insensitive), ignoring the internal RID."""
    old = [c for c in old_cols if c != RID]
    new = [c for c in new_cols if c != RID]
    old_set, new_set = set(old), set(new)
    return {
        "added": [c for c in new if c not in old_set],
        "removed": [c for c in old if c not in new_set],
        "common": [c for c in new if c in old_set],
    }


def type_changes(
    old_df: pd.DataFrame, new_df: pd.DataFrame, common_cols: Sequence[str]
) -> List[Dict[str, str]]:
    """Detect dtype changes for columns present in both versions."""
    changes: List[Dict[str, str]] = []
    for col in common_cols:
        old_t, new_t = str(old_df[col].dtype), str(new_df[col].dtype)
        if old_t != new_t:
            changes.append({"column": col, "old_type": old_t, "new_type": new_t})
    return changes


def changed_row_ids(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    common_ids: Set[str],
    compare_cols: Sequence[str],
) -> List[str]:
    """Return the ids (from ``common_ids``) whose values differ in ``compare_cols``.

    Comparison is NaN-aware: two NaNs count as equal. Columns that exist in only
    one version are *not* compared here (they are handled as column add/remove).
    """
    cols = [c for c in compare_cols if c != RID]
    if not common_ids or not cols:
        return []

    old = old_df[old_df[RID].isin(common_ids)].set_index(RID)
    new = new_df[new_df[RID].isin(common_ids)].set_index(RID)
    # Align both frames on the same ids/order so positional compare is valid.
    idx = old.index
    new = new.reindex(idx)

    differs = pd.Series(False, index=idx)
    for col in cols:
        a = old[col]
        b = new[col]
        ne = (a.values != b.values) & ~(pd.isna(a.values) & pd.isna(b.values))
        differs = differs | pd.Series(ne, index=idx)
    return [str(i) for i in idx[differs.values]]


def cell_level_changes(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    changed_ids: Sequence[str],
    compare_cols: Sequence[str],
) -> pd.DataFrame:
    """Build a (row_id, column, old_val, new_val) audit log for changed rows.

    This is an *optional* provenance artifact (the spirit of the prototype's
    "triplet" output). It is never used during reconstruction.
    """
    cols = [c for c in compare_cols if c != RID]
    changed = list(changed_ids)
    if not changed or not cols:
        return pd.DataFrame(columns=[RID, "column", "old_val", "new_val"])

    old = old_df[old_df[RID].isin(changed)].set_index(RID)
    new = new_df[new_df[RID].isin(changed)].set_index(RID)
    new = new.reindex(old.index)

    records: List[Dict[str, object]] = []
    for col in cols:
        a, b = old[col], new[col]
        ne = (a.values != b.values) & ~(pd.isna(a.values) & pd.isna(b.values))
        for rid in old.index[ne]:
            records.append(
                {
                    RID: rid,
                    "column": col,
                    "old_val": old.at[rid, col],
                    "new_val": new.at[rid, col],
                }
            )
    return pd.DataFrame(records, columns=[RID, "column", "old_val", "new_val"])
