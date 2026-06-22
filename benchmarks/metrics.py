"""Score how well each system *recovers row identity* against the hidden key.

We replay the ground-truth chain and, for every parent->child transition, ask the
matcher (in either ``hash-only`` or full ``deltatrace`` mode) to link child rows
back to parent rows. A prediction is scored against the hidden true key.

Duplicate-content caveat: when two parent rows are byte-identical but carry
different hidden keys, *which* one a survivor "came from" is genuinely
undecidable. We therefore treat a prediction as correct if it lands on the right
hidden key **or** on a parent row whose content is identical to the right one.
That removes duplicate noise without flattering anybody -- it is applied to every
system equally.

The headline that survives every caveat: ``hash-only`` cannot link an *edited*
row to its parent at all (its content no longer hashes to anything in the
parent), so its recall collapses exactly where real data churns -- updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from deltatrace.matching import CanonConfig, SimConfig, match_versions

from .synth import Version


def _content_map(df: pd.DataFrame, keys: np.ndarray) -> Dict[str, tuple]:
    cols = sorted(df.columns)
    out: Dict[str, tuple] = {}
    for rid, tup in zip(keys, df[cols].itertuples(index=False, name=None)):
        out[f"{rid}"] = tuple(
            "\x00NaN" if (isinstance(v, float) and v != v) else v for v in tup
        )
    return out


@dataclass
class IdentityScore:
    system: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def identity_quality(
    chain: Sequence[Version],
    spec,
    *,
    system: str = "deltatrace",
) -> IdentityScore:
    """Precision / recall / F1 of identity recovery for one system on a chain."""
    exact_only = system == "hash-only"
    canon = CanonConfig(round_decimals=6)
    sim = SimConfig(row_threshold=0.55, num_rel_tol=0.25)

    tp = fp = fn = 0
    for prev, cur in zip(chain, chain[1:]):
        parent_keys = [f"{k}" for k in prev.true_key]
        parent_content = _content_map(prev.payload, prev.true_key)
        parent_key_set = set(parent_keys)

        res = match_versions(
            cur.payload,
            prev.payload,
            parent_keys,
            block_on=spec.block_on,
            exact_only=exact_only,
            canon=canon,
            sim=sim,
        )

        for ni in range(len(cur.payload)):
            true_parent = f"{cur.true_key[ni]}"
            gold = true_parent in parent_key_set  # survivor (vs genuine insert)
            pred_parent = res.pairs.get(ni)
            pred = pred_parent is not None

            if gold and pred:
                same = pred_parent == true_parent or (
                    parent_content.get(pred_parent) == parent_content.get(true_parent)
                )
                if same:
                    tp += 1
                else:
                    fp += 1
                    fn += 1
            elif gold and not pred:
                fn += 1
            elif (not gold) and pred:
                fp += 1
            # not gold and not pred -> correct insert, ignored

    return IdentityScore(system, tp, fp, fn)
