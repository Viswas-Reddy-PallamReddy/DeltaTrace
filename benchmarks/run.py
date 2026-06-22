"""Orchestrate the full benchmark: datasets x systems -> tidy results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .datasets import DATASETS, load
from .metrics import identity_quality
from .synth import Version, make_chain
from .systems import run_systems


@dataclass
class DatasetRun:
    name: str
    chain: List[Version]
    storage: pd.DataFrame   # one row per system
    identity: pd.DataFrame  # one row per matching system


def run_dataset(
    name: str,
    *,
    n_versions: int = 6,
    seed: int = 0,
    systems: Sequence[str] = ("naive", "hash-only", "deltatrace", "oracle"),
) -> DatasetRun:
    spec = DATASETS[name]
    df = load(name, seed=seed)
    chain = make_chain(
        df,
        update_cols=spec.update_cols,
        n_versions=n_versions,
        seed=seed,
    )

    rows = run_systems(chain, spec, systems=systems)
    base = next((r.bytes for r in rows if r.system == "naive"), None)
    storage = pd.DataFrame(
        [
            {
                "dataset": name,
                "system": r.system,
                "bytes": r.bytes,
                "kb": round(r.bytes / 1024, 1),
                "vs_naive": round(r.bytes / base, 3) if base else float("nan"),
                "saved_pct": round(100 * (1 - r.bytes / base), 1) if base else float("nan"),
                "reconstruct_ok": r.reconstruct_ok,
            }
            for r in rows
        ]
    )

    id_rows = []
    for sysname in ("hash-only", "deltatrace"):
        if sysname in systems:
            s = identity_quality(chain, spec, system=sysname)
            id_rows.append(
                {
                    "dataset": name,
                    "system": sysname,
                    "precision": round(s.precision, 3),
                    "recall": round(s.recall, 3),
                    "f1": round(s.f1, 3),
                    "tp": s.tp,
                    "fp": s.fp,
                    "fn": s.fn,
                }
            )
    identity = pd.DataFrame(id_rows)
    return DatasetRun(name, chain, storage, identity)


def run_all(
    names: Optional[Sequence[str]] = None,
    *,
    n_versions: int = 6,
    seed: int = 0,
) -> Dict[str, DatasetRun]:
    names = list(names or DATASETS.keys())
    return {n: run_dataset(n, n_versions=n_versions, seed=seed) for n in names}


def storage_table(runs: Dict[str, DatasetRun]) -> pd.DataFrame:
    return pd.concat([r.storage for r in runs.values()], ignore_index=True)


def identity_table(runs: Dict[str, DatasetRun]) -> pd.DataFrame:
    return pd.concat([r.identity for r in runs.values()], ignore_index=True)
