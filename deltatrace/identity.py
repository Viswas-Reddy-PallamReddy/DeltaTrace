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

import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from .matching import CanonConfig, MatchResult, SimConfig, match_versions

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


# --------------------------------------------------------------------------- #
# Identity strategies                                                         #
# --------------------------------------------------------------------------- #
# The DeltaTrace engine is deliberately *identity-agnostic*: it only needs each
# version's DataFrame to arrive with a consistent RID column. An identity
# strategy is the pluggable component that decides what "the same row" means.
#
#   * PrimaryKeyIdentity   -- a key is declared; identity is derived from it
#                             (deterministic, stateless, exact).
#   * ContentMatchIdentity -- there is no key; identity is *recovered* by
#                             matching each version against the previous one.
#
# Whatever the strategy guesses, reconstruction stays exact (error confinement):
# a wrong guess only changes which RID carries a row's values, never the set of
# values that gets materialised. The strategy therefore trades storage and
# provenance quality, never correctness.


class IdentityStrategy(ABC):
    """Assigns a stable RID column to each committed version."""

    @property
    def user_keys(self) -> Optional[List[str]]:
        """User-facing key columns to sort reconstructions by, if any."""
        return None

    @property
    def requires_parent(self) -> bool:
        """Whether :meth:`assign` needs the materialised parent frame."""
        return False

    @abstractmethod
    def assign(self, df: pd.DataFrame, parent: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Return a copy of ``df`` (index reset) carrying the RID column.

        ``parent`` is the fully materialised previous version (with its RID
        column) or ``None`` for the very first commit.
        """

    @abstractmethod
    def to_config(self) -> Dict:
        """Serialise this strategy to a JSON-friendly dict (for the store)."""

    def explain(self) -> Optional[pd.DataFrame]:
        """Optional provenance for the most recent :meth:`assign` call."""
        return None


class PrimaryKeyIdentity(IdentityStrategy):
    """Identity derived from a declared primary key (the exact, keyed case)."""

    def __init__(self, primary_key: Union[str, Sequence[str]]):
        self._keys = normalize_keys(primary_key)

    @property
    def keys(self) -> List[str]:
        return list(self._keys)

    @property
    def user_keys(self) -> Optional[List[str]]:
        return list(self._keys)

    def assign(self, df: pd.DataFrame, parent: Optional[pd.DataFrame]) -> pd.DataFrame:
        return with_row_ids(df, self._keys)

    def to_config(self) -> Dict:
        return {"strategy": "primary_key", "keys": list(self._keys)}

    @classmethod
    def from_config(cls, cfg: Dict) -> "PrimaryKeyIdentity":
        return cls(list(cfg["keys"]))


class ContentMatchIdentity(IdentityStrategy):
    """Identity *recovered* from row content for keyless tables.

    Each new version is matched against its parent with the two-phase pipeline in
    :mod:`deltatrace.matching` (exact hash join, then a tolerance-aware residual
    pass). Matched rows inherit their parent's RID -- so an edited row is stored
    as a small update instead of a delete + insert -- while genuinely new rows are
    issued a fresh, globally unique RID.

    Set ``exact_only=True`` to disable the residual pass; the strategy then behaves
    like a keyless system that has nothing but content hashing (every update
    degrades into delete + insert). This is the honest baseline the benchmark
    compares against.
    """

    def __init__(
        self,
        *,
        compare_on: Optional[Sequence[str]] = None,
        block_on: Optional[Sequence[str]] = None,
        exact_only: bool = False,
        row_threshold: float = 0.6,
        num_rel_tol: float = 0.05,
        round_decimals: Optional[int] = 6,
        case_insensitive: bool = False,
        strip_strings: bool = True,
        string_soft: bool = True,
    ):
        self.compare_on = list(compare_on) if compare_on is not None else None
        self.block_on = list(block_on) if block_on is not None else None
        self.exact_only = bool(exact_only)
        self._canon = CanonConfig(
            round_decimals=round_decimals,
            case_insensitive=case_insensitive,
            strip_strings=strip_strings,
        )
        self._sim = SimConfig(
            row_threshold=row_threshold,
            num_rel_tol=num_rel_tol,
            string_soft=string_soft,
        )
        self._last: Optional[pd.DataFrame] = None

    # -- assignment -----------------------------------------------------
    @property
    def requires_parent(self) -> bool:
        return True

    @staticmethod
    def _fresh_rid() -> str:
        # 16 hex chars (~64 bits) is collision-safe well past billions of rows
        # while keeping the stored identity column compact -- the RID is carried
        # on every persisted row, so its width is a real storage cost.
        return uuid.uuid4().hex[:16]

    def assign(self, df: pd.DataFrame, parent: Optional[pd.DataFrame]) -> pd.DataFrame:
        out = df.reset_index(drop=True).copy()
        n = len(out)

        if parent is None or RID not in getattr(parent, "columns", []):
            rids = [self._fresh_rid() for _ in range(n)]
            out[RID] = pd.Series(rids, index=out.index, dtype="object")
            self._last = self._provenance(rids, {}, {}, set(range(n)))
            return out

        parent_rids = [str(r) for r in parent[RID].tolist()]
        parent_data = parent.drop(columns=[RID])
        result: MatchResult = match_versions(
            out,
            parent_data,
            parent_rids,
            compare_on=self.compare_on,
            block_on=self.block_on,
            exact_only=self.exact_only,
            canon=self._canon,
            sim=self._sim,
        )

        rids: List[str] = [""] * n
        for ni, prid in result.pairs.items():
            rids[ni] = prid
        insert_set = set(result.inserts)
        for ni in range(n):
            if rids[ni] == "":
                rids[ni] = self._fresh_rid()
                insert_set.add(ni)

        out[RID] = pd.Series(rids, index=out.index, dtype="object")
        self._last = self._provenance(rids, result.method, result.confidence, insert_set)
        return out

    def _provenance(self, rids, method, confidence, insert_set) -> pd.DataFrame:
        rows = []
        for ni, rid in enumerate(rids):
            if ni in insert_set:
                m, conf = "insert", float("nan")
            else:
                m, conf = method.get(ni, "exact"), confidence.get(ni, 1.0)
            rows.append({RID: rid, "row": ni, "method": m, "confidence": conf})
        return pd.DataFrame(rows, columns=[RID, "row", "method", "confidence"])

    def explain(self) -> Optional[pd.DataFrame]:
        return self._last

    # -- config ---------------------------------------------------------
    def to_config(self) -> Dict:
        return {
            "strategy": "content_match",
            "compare_on": self.compare_on,
            "block_on": self.block_on,
            "exact_only": self.exact_only,
            "row_threshold": self._sim.row_threshold,
            "num_rel_tol": self._sim.num_rel_tol,
            "string_soft": self._sim.string_soft,
            "round_decimals": self._canon.round_decimals,
            "case_insensitive": self._canon.case_insensitive,
            "strip_strings": self._canon.strip_strings,
        }

    @classmethod
    def from_config(cls, cfg: Dict) -> "ContentMatchIdentity":
        return cls(
            compare_on=cfg.get("compare_on"),
            block_on=cfg.get("block_on"),
            exact_only=cfg.get("exact_only", False),
            row_threshold=cfg.get("row_threshold", 0.6),
            num_rel_tol=cfg.get("num_rel_tol", 0.05),
            round_decimals=cfg.get("round_decimals", 6),
            case_insensitive=cfg.get("case_insensitive", False),
            strip_strings=cfg.get("strip_strings", True),
            string_soft=cfg.get("string_soft", True),
        )


def make_identity(cfg: Dict) -> IdentityStrategy:
    """Reconstruct an :class:`IdentityStrategy` from its stored config."""
    strategy = cfg.get("strategy")
    if strategy == "primary_key":
        return PrimaryKeyIdentity.from_config(cfg)
    if strategy == "content_match":
        return ContentMatchIdentity.from_config(cfg)
    raise ValueError(f"unknown identity strategy: {strategy!r}")
