"""Synthesise a realistic version chain from a single static dataset.

A public dataset is just one snapshot, so we manufacture a believable *history*:
starting from a subset of the rows, each new version edits some rows, inserts a
few, and deletes a few. Every row carries a hidden ground-truth key (``__tk__``)
so the benchmark can (a) run an oracle and (b) score how well each system
recovers identity. The keyless systems never see ``__tk__`` -- it is stripped
from the payload they are handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

TRUE_KEY = "__tk__"


@dataclass
class Version:
    """One snapshot in the chain."""

    payload: pd.DataFrame      # what a versioning system sees (no __tk__)
    true_key: np.ndarray       # ground-truth identity per row (aligned to payload)


def _perturb_cell(value, rng: np.random.Generator):
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if value != value:  # NaN
            return value
        mag = abs(float(value))
        delta = rng.normal(0, 0.1 * mag + 1.0)
        new = float(value) + delta
        return round(new, 4) if isinstance(value, (float, np.floating)) else int(round(new))
    return value


def make_chain(
    df: pd.DataFrame,
    *,
    update_cols: Sequence[str],
    n_versions: int = 6,
    start_frac: float = 0.7,
    update_frac: float = 0.06,
    insert_frac: float = 0.02,
    delete_frac: float = 0.02,
    seed: int = 0,
) -> List[Version]:
    """Build an ``n_versions`` chain with edits, inserts and deletes.

    The mutations are tracked by a hidden key so identity is known exactly.
    """
    rng = np.random.default_rng(seed)
    universe = df.reset_index(drop=True).copy()
    universe[TRUE_KEY] = np.arange(len(universe))
    edit_cols = [c for c in update_cols if c in universe.columns] or [
        c for c in universe.columns if c != TRUE_KEY
    ][:1]

    n_start = max(2, int(start_frac * len(universe)))
    order = rng.permutation(len(universe))
    active = universe.iloc[order[:n_start]].copy().reset_index(drop=True)
    pool = list(order[n_start:])  # row indices available for future inserts

    chain: List[Version] = []
    for v in range(n_versions):
        payload = active.drop(columns=[TRUE_KEY]).reset_index(drop=True)
        chain.append(Version(payload=payload, true_key=active[TRUE_KEY].to_numpy().copy()))
        if v == n_versions - 1:
            break

        active = active.reset_index(drop=True)

        # ---- updates: perturb a few cells of existing rows (key unchanged) ----
        n_upd = max(1, int(update_frac * len(active)))
        upd_rows = rng.choice(len(active), min(n_upd, len(active)), replace=False)
        for ri in upd_rows:
            col = edit_cols[rng.integers(0, len(edit_cols))]
            active.at[ri, col] = _perturb_cell(active.at[ri, col], rng)

        # ---- deletes ---------------------------------------------------------
        n_del = int(delete_frac * len(active))
        if n_del:
            del_rows = rng.choice(len(active), min(n_del, len(active) - 1), replace=False)
            active = active.drop(index=del_rows).reset_index(drop=True)

        # ---- inserts (draw fresh rows from the untouched pool) ---------------
        n_ins = int(insert_frac * len(universe))
        if n_ins and pool:
            take = pool[: min(n_ins, len(pool))]
            pool = pool[len(take):]
            new_rows = universe.iloc[take].copy()
            active = pd.concat([active, new_rows], ignore_index=True)

    return chain


def ground_truth_pairs(chain: Sequence[Version]):
    """Yield ``(parent_keys, child_keys)`` for each consecutive transition."""
    for prev, cur in zip(chain, chain[1:]):
        yield prev.true_key, cur.true_key
