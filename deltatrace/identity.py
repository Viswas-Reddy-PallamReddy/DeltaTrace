"""Row identity for DeltaTrace.

A version-control system for tabular data needs a *stable* way to recognise
"the same row" across versions. The original prototype assigned a random UUID
to every row, which only works if you keep passing the *same* DataFrame object
forward. That falls apart the moment you commit a freshly-loaded DataFrame.

DeltaTrace instead derives a deterministic row id from a user-declared
**primary key**. The same logical row therefore maps to the same id in every
version, no matter where the DataFrame came from. This is the same idea that
real lakehouse engines (Delta Lake, Iceberg) rely on for MERGE/upsert.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import pandas as pd

# Internal column name used to carry the derived row id around. Chosen to be
# unlikely to collide with a real user column.
RID = "__dt_rid__"

# Separator used when concatenating composite-key values into a single id.
_KEY_SEP = "\x1f"  # ASCII unit separator


def normalize_keys(primary_key: Union[str, Sequence[str]]) -> List[str]:
    """Normalise the ``primary_key`` argument into a list of column names."""
    if isinstance(primary_key, str):
        keys = [primary_key]
    else:
        keys = list(primary_key)
    if not keys:
        raise ValueError("primary_key must name at least one column")
    if len(set(keys)) != len(keys):
        raise ValueError(f"primary_key contains duplicate columns: {keys}")
    return keys


def compute_row_ids(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    """Return a string Series of row ids derived from the primary key columns.

    Raises if a key column is missing, contains nulls, or is not unique.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"primary key column(s) missing from data: {missing}")

    key_frame = df[keys]
    if bool(key_frame.isna().any().any()):
        raise ValueError("primary key columns must not contain null values")

    if len(keys) == 1:
        rid = key_frame[keys[0]].astype(str)
    else:
        rid = key_frame.astype(str).agg(_KEY_SEP.join, axis=1)

    rid = rid.reset_index(drop=True)
    rid.name = RID

    if bool(rid.duplicated().any()):
        example = rid[rid.duplicated()].iloc[0]
        raise ValueError(
            f"primary key {keys} is not unique (duplicate id: {example!r})"
        )
    return rid


def with_row_ids(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Return a copy of ``df`` (index reset) with the internal RID column added."""
    out = df.reset_index(drop=True).copy()
    out[RID] = compute_row_ids(out, keys)
    return out
