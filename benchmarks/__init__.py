"""Benchmark harness: does keyless content-matching actually beat the baselines?

This package compares four ways of versioning a *keyless* table on famous public
datasets that have no natural primary key (iris, penguins, titanic, diamonds):

* **naive**      -- store every version in full (what a folder of ``data_v*.csv``
  files does);
* **hash-only**  -- the best a keyless system can do with content hashing alone:
  every edited row becomes a delete + a brand-new insert;
* **DeltaTrace** -- our exact-hash + tolerance-aware fuzzy matching, which
  recovers row identity and stores edits as cell-level deltas;
* **oracle**     -- identity taken from a hidden ground-truth key (the best any
  matcher could possibly do).

The point the benchmark makes: hash-only inflates storage and destroys update
provenance the moment rows are edited, while DeltaTrace stays close to the oracle
-- *without ever sacrificing exact reconstruction* (error confinement).
"""

from .datasets import DATASETS, DatasetSpec, load
from .metrics import identity_quality
from .run import (
    DatasetRun,
    identity_table,
    run_all,
    run_dataset,
    storage_table,
)
from .synth import make_chain

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "load",
    "make_chain",
    "identity_quality",
    "run_dataset",
    "run_all",
    "DatasetRun",
    "storage_table",
    "identity_table",
]
