"""Content-based row matching for keyless tabular data.

Most real datasets have **no reliable primary key**, yet a version-control system
still has to decide which row in a new version is "the same row" as one in the
previous version. This module implements that decision as a two-phase pipeline:

1. **Exact pass** -- canonicalise every row and match rows whose canonical content
   is byte-for-byte identical. This is an ``O(n)`` hash join and handles the bulk
   of unchanged rows for free.
2. **Residual pass** -- the rows left over (genuine inserts, deletes, and *updated*
   rows whose content changed) are matched with a tolerance-aware, typed
   similarity score, restricted to *blocks* so the comparison stays cheap.

The output is a one-to-one assignment between new rows and parent rows. Anything
unmatched on the new side is an insert; anything unmatched on the parent side is a
delete.

Design rule -- **a hash is an accelerator, not an identity.** Exact hashing alone
turns every cell-level edit into a delete + insert (it cannot see that a row was
merely updated). The residual pass is what recovers those updates. Crucially, the
*correctness* of reconstruction never depends on the quality of this matching: a
wrong guess only costs storage, never fidelity (see :mod:`deltatrace.identity` and
the error-confinement tests).

Everything here is pure and dependency-light (stdlib :mod:`difflib` for fuzzy
string comparison); the only third-party import is pandas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_NULL_TOKEN = "\x00__dt_null__\x00"
_FIELD_SEP = "\x1f"  # ASCII unit separator


# --------------------------------------------------------------------------- #
# Canonicalisation                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class CanonConfig:
    """How raw cell values are normalised *for matching only*.

    Canonicalisation never touches the values that get stored -- it only affects
    which rows are considered "the same" during hashing and similarity scoring.
    Imperfect canonicalisation therefore costs efficiency, never correctness.
    """

    round_decimals: Optional[int] = 6
    case_insensitive: bool = False
    strip_strings: bool = True


def _canon_scalar(value, cfg: CanonConfig) -> str:
    """Return a stable string token for a single cell value."""
    if value is None:
        return _NULL_TOKEN
    if isinstance(value, float):
        if np.isnan(value):
            return _NULL_TOKEN
        if cfg.round_decimals is not None:
            value = round(value, cfg.round_decimals)
        return repr(float(value))
    try:
        if pd.isna(value):
            return _NULL_TOKEN
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        f = float(value)
        if np.isnan(f):
            return _NULL_TOKEN
        if cfg.round_decimals is not None:
            f = round(f, cfg.round_decimals)
        return repr(f)
    s = str(value)
    if cfg.strip_strings:
        s = s.strip()
    if cfg.case_insensitive:
        s = s.casefold()
    return s


def canon_frame(df: pd.DataFrame, columns: Sequence[str], cfg: CanonConfig) -> List[Tuple[str, ...]]:
    """Return one canonical tuple per row of ``df`` over ``columns``."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return [tuple() for _ in range(len(df))]
    canon_cols = [df[c].map(lambda v: _canon_scalar(v, cfg)).to_numpy() for c in cols]
    return list(zip(*canon_cols)) if canon_cols else [tuple() for _ in range(len(df))]


def row_hash(canon_row: Tuple[str, ...]) -> str:
    """Stable digest of a canonical row tuple."""
    joined = _FIELD_SEP.join(canon_row)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()


# --------------------------------------------------------------------------- #
# Similarity (residual pass)                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """Parameters for the tolerance-aware residual similarity score."""

    row_threshold: float = 0.6
    num_rel_tol: float = 0.05  # within 5% (relative to column scale) counts as equal
    string_soft: bool = True   # use difflib ratio for strings instead of hard equality


def _column_kinds(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for c in columns:
        dt = df[c].dtype
        if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt):
            kinds[c] = "numeric"
        else:
            kinds[c] = "categorical"
    return kinds


def _column_scales(df: pd.DataFrame, numeric_cols: Sequence[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        std = float(s.std(skipna=True)) if len(s) else 0.0
        if not np.isfinite(std) or std == 0.0:
            rng = float(s.max(skipna=True) - s.min(skipna=True)) if len(s) else 0.0
            std = rng if (np.isfinite(rng) and rng > 0) else 1.0
        scales[c] = std if std > 0 else 1.0
    return scales


def _cell_similarity(a, b, kind: str, scale: float, cfg: SimConfig) -> float:
    a_na = a is None or (isinstance(a, float) and np.isnan(a)) or (_safe_isna(a))
    b_na = b is None or (isinstance(b, float) and np.isnan(b)) or (_safe_isna(b))
    if a_na and b_na:
        return 1.0
    if a_na or b_na:
        return 0.0
    if kind == "numeric":
        try:
            diff = abs(float(a) - float(b))
        except (TypeError, ValueError):
            return 1.0 if str(a) == str(b) else 0.0
        if diff <= cfg.num_rel_tol * scale:
            return 1.0
        return float(max(0.0, 1.0 - diff / (scale if scale > 0 else 1.0)))
    # categorical / string
    sa, sb = str(a), str(b)
    if sa == sb:
        return 1.0
    if cfg.string_soft:
        return float(SequenceMatcher(None, sa, sb).ratio())
    return 0.0


def _safe_isna(v) -> bool:
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def row_similarity(
    new_row: Dict[str, object],
    parent_row: Dict[str, object],
    columns: Sequence[str],
    kinds: Dict[str, str],
    scales: Dict[str, float],
    cfg: SimConfig,
) -> float:
    """Mean per-column similarity in ``[0, 1]`` between two rows."""
    if not columns:
        return 0.0
    total = 0.0
    for c in columns:
        total += _cell_similarity(
            new_row.get(c), parent_row.get(c), kinds[c], scales.get(c, 1.0), cfg
        )
    return total / len(columns)


# --------------------------------------------------------------------------- #
# Match result                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class MatchResult:
    """Outcome of matching a new version against its parent.

    ``pairs`` maps a new-row positional index to the parent RID it inherits.
    ``confidence`` carries the score behind each decision (1.0 for exact matches).
    ``method`` records how each pair was found ("exact" or "fuzzy").
    """

    pairs: Dict[int, str] = field(default_factory=dict)
    confidence: Dict[int, float] = field(default_factory=dict)
    method: Dict[int, str] = field(default_factory=dict)
    inserts: List[int] = field(default_factory=list)        # new-row indices, no match
    deletes: List[str] = field(default_factory=list)        # parent RIDs, unmatched

    @property
    def n_exact(self) -> int:
        return sum(1 for m in self.method.values() if m == "exact")

    @property
    def n_fuzzy(self) -> int:
        return sum(1 for m in self.method.values() if m == "fuzzy")


# --------------------------------------------------------------------------- #
# The matcher                                                                 #
# --------------------------------------------------------------------------- #
def match_versions(
    new_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    parent_rids: Sequence[str],
    *,
    compare_on: Optional[Sequence[str]] = None,
    block_on: Optional[Sequence[str]] = None,
    exact_only: bool = False,
    canon: Optional[CanonConfig] = None,
    sim: Optional[SimConfig] = None,
    max_block_pairs: int = 2_000_000,
) -> MatchResult:
    """Match ``new_df`` rows to ``parent_df`` rows (carrying ``parent_rids``).

    Parameters
    ----------
    compare_on:
        Columns used for content comparison. Defaults to the columns common to
        both frames.
    block_on:
        Columns whose (canonical) values define candidate blocks for the residual
        fuzzy pass. Rows only compete to match within the same block, keeping the
        pass cheap. ``None`` means a single global block (capped by
        ``max_block_pairs``).
    exact_only:
        If True, skip the residual fuzzy pass entirely. This reproduces the
        behaviour of a keyless system that has nothing but content hashing: every
        updated row degrades into a delete + insert. Used as the baseline.
    """
    canon = canon or CanonConfig()
    sim = sim or SimConfig()

    common = [c for c in new_df.columns if c in set(parent_df.columns)]
    compare_cols = [c for c in (compare_on or common) if c in common]

    result = MatchResult()
    n_new = len(new_df)
    n_parent = len(parent_df)
    if n_parent == 0:
        result.inserts = list(range(n_new))
        return result
    if n_new == 0:
        result.deletes = list(parent_rids)
        return result

    # ---- phase 1: exact, multiplicity-aware hash join --------------------- #
    new_canon = canon_frame(new_df, compare_cols, canon)
    parent_canon = canon_frame(parent_df, compare_cols, canon)
    new_hashes = [row_hash(t) for t in new_canon]
    parent_hashes = [row_hash(t) for t in parent_canon]

    parent_buckets: Dict[str, List[int]] = {}
    for pi, h in enumerate(parent_hashes):
        parent_buckets.setdefault(h, []).append(pi)

    new_matched = [False] * n_new
    parent_matched = [False] * n_parent
    for ni, h in enumerate(new_hashes):
        bucket = parent_buckets.get(h)
        if bucket:
            pi = bucket.pop()
            new_matched[ni] = True
            parent_matched[pi] = True
            result.pairs[ni] = parent_rids[pi]
            result.confidence[ni] = 1.0
            result.method[ni] = "exact"

    residual_new = [ni for ni in range(n_new) if not new_matched[ni]]
    residual_parent = [pi for pi in range(n_parent) if not parent_matched[pi]]

    if exact_only or not residual_new or not residual_parent:
        result.inserts = list(residual_new)
        result.deletes = [parent_rids[pi] for pi in residual_parent]
        return result

    # ---- phase 2: residual fuzzy match within blocks ---------------------- #
    kinds = _column_kinds(new_df, compare_cols)
    numeric_cols = [c for c in compare_cols if kinds[c] == "numeric"]
    scales = _column_scales(pd.concat([new_df[numeric_cols], parent_df[numeric_cols]],
                                      ignore_index=True), numeric_cols) if numeric_cols else {}

    block_cols = [c for c in (block_on or []) if c in compare_cols]
    new_block_keys = _block_keys(new_df, residual_new, block_cols, canon)
    parent_block_keys = _block_keys(parent_df, residual_parent, block_cols, canon)

    parent_by_block: Dict[Tuple[str, ...], List[int]] = {}
    for pi in residual_parent:
        parent_by_block.setdefault(parent_block_keys[pi], []).append(pi)

    new_records = new_df.to_dict("records")
    parent_records = parent_df.to_dict("records")

    claimed_parent: set = set()
    for ni in residual_new:
        candidates = parent_by_block.get(new_block_keys[ni], [])
        if not candidates:
            continue
        if len(candidates) > max_block_pairs:
            candidates = candidates[:max_block_pairs]
        best_pi = -1
        best_score = sim.row_threshold
        nrow = new_records[ni]
        for pi in candidates:
            if pi in claimed_parent:
                continue
            score = row_similarity(nrow, parent_records[pi], compare_cols, kinds, scales, sim)
            if score >= best_score:
                best_score = score
                best_pi = pi
        if best_pi >= 0:
            claimed_parent.add(best_pi)
            result.pairs[ni] = parent_rids[best_pi]
            result.confidence[ni] = float(best_score)
            result.method[ni] = "fuzzy"
            parent_matched[best_pi] = True

    result.inserts = [ni for ni in residual_new if ni not in result.pairs]
    result.deletes = [parent_rids[pi] for pi in range(n_parent) if not parent_matched[pi]]
    return result


def _block_keys(
    df: pd.DataFrame,
    indices: Sequence[int],
    block_cols: Sequence[str],
    canon: CanonConfig,
) -> Dict[int, Tuple[str, ...]]:
    """Map each row index to its block key (empty tuple = single global block)."""
    if not block_cols:
        return {i: tuple() for i in indices}
    sub = df.iloc[list(indices)] if indices else df.iloc[[]]
    canon_rows = canon_frame(sub, block_cols, canon)
    return {idx: canon_rows[pos] for pos, idx in enumerate(indices)}
