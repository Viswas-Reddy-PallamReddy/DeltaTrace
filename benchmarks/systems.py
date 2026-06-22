"""Run four versioning systems on identical keyless payloads and measure them.

Every system is handed the *same* sequence of keyless DataFrames (no ground-truth
key). We measure on-disk storage and verify exact reconstruction:

* ``naive``     -- write every version out in full (parquet snapshot per version);
* ``hash-only`` -- DeltaTrace with ``ContentMatchIdentity(exact_only=True)``;
* ``deltatrace``-- DeltaTrace with full exact + fuzzy matching;
* ``oracle``    -- DeltaTrace fed the hidden true key as identity (upper bound).

All four reconstruct every version exactly; the difference is purely *storage*
and identity/provenance quality.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from deltatrace import ContentMatchIdentity, DeltaRepo
from deltatrace.identity import RID, IdentityStrategy

from .synth import Version


# --------------------------------------------------------------------------- #
# Oracle: identity taken from the hidden ground-truth key (best case).
# --------------------------------------------------------------------------- #
class OracleIdentity(IdentityStrategy):
    """Assigns RID = the hidden true key. Never stored as a user column.

    The true key is hashed to the *same width* DeltaTrace uses for its synthetic
    RIDs, so the storage gap against DeltaTrace reflects only identity-recovery
    quality -- not how compact the id happens to be.
    """

    def __init__(self, true_keys: Sequence[np.ndarray]):
        self._seq = [np.asarray(k) for k in true_keys]
        self._i = 0

    @staticmethod
    def _rid(key) -> str:
        return hashlib.blake2b(str(key).encode(), digest_size=8).hexdigest()

    def assign(self, df: pd.DataFrame, parent: Optional[pd.DataFrame]) -> pd.DataFrame:
        out = df.reset_index(drop=True).copy()
        keys = self._seq[self._i]
        self._i += 1
        out[RID] = pd.Series([self._rid(k) for k in keys], index=out.index, dtype="object")
        return out

    def to_config(self) -> Dict:  # never reopened, but init serialises it
        return {"strategy": "oracle"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _multiset(df: pd.DataFrame) -> Counter:
    """Order-independent fingerprint of a frame's rows (NaN-safe, exact)."""
    cols = sorted(df.columns)
    rows = []
    for tup in df[cols].itertuples(index=False, name=None):
        norm = tuple("\x00NaN" if (isinstance(v, float) and v != v) else v for v in tup)
        rows.append(norm)
    return Counter(rows)


def _identity_for(system: str, spec, true_keys):
    if system == "hash-only":
        return ContentMatchIdentity(
            block_on=spec.block_on, exact_only=True, round_decimals=6
        )
    if system == "deltatrace":
        return ContentMatchIdentity(
            block_on=spec.block_on,
            exact_only=False,
            row_threshold=0.55,
            num_rel_tol=0.25,
            round_decimals=6,
        )
    if system == "oracle":
        return OracleIdentity(true_keys)
    raise ValueError(system)


@dataclass
class SystemResult:
    system: str
    bytes: int
    versions: int
    reconstruct_ok: bool


def run_system(system: str, chain: Sequence[Version], spec, workdir: Path) -> SystemResult:
    payloads: List[pd.DataFrame] = [v.payload for v in chain]
    true_keys = [v.true_key for v in chain]
    root = workdir / system
    if root.exists():
        shutil.rmtree(root)

    if system == "naive":
        root.mkdir(parents=True)
        for i, p in enumerate(payloads, 1):
            p.to_parquet(root / f"v{i}.parquet", index=False)
        # naive trivially round-trips: each file *is* the version
        ok = all(
            _multiset(pd.read_parquet(root / f"v{i}.parquet")) == _multiset(p)
            for i, p in enumerate(payloads, 1)
        )
        return SystemResult(system, _dir_size(root), len(payloads), ok)

    identity = _identity_for(system, spec, true_keys)
    repo = DeltaRepo.init(root, identity=identity, overwrite=True)
    for p in payloads:
        repo.commit(p)

    ok = True
    for i, p in enumerate(payloads, 1):
        got = repo.checkout(i)
        if _multiset(got) != _multiset(p):
            ok = False
            break
    return SystemResult(system, _dir_size(root), len(payloads), ok)


def run_systems(
    chain: Sequence[Version],
    spec,
    systems: Sequence[str] = ("naive", "hash-only", "deltatrace", "oracle"),
    workdir: Optional[Path] = None,
) -> List[SystemResult]:
    tmp = workdir or Path(tempfile.mkdtemp(prefix="dt_bench_"))
    out = [run_system(s, chain, spec, tmp) for s in systems]
    if workdir is None:
        shutil.rmtree(tmp, ignore_errors=True)
    return out
