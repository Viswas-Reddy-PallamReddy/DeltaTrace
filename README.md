# DeltaTrace

**Git-style version control for tabular data.** DeltaTrace records a sequence of
versions of a DataFrame, storing only the *delta* between consecutive versions
(new / changed rows, deleted rows, added / dropped columns, dtype changes) plus
a small metadata file. Any historical version can be reconstructed exactly by
replaying deltas along the parent chain — optionally short-circuited by a
snapshot.

Think of it as a tiny, dependency-light cousin of Delta Lake / DVC for a single
table: **commit → log → diff → checkout (time-travel)**, backed by Parquet.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)

---

## Why

A dataset is rarely static — rows get appended, values get corrected, columns
get added or retyped. Keeping a full copy per version wastes space, and a pile
of `data_v3_final_FINAL.parquet` files loses the *what changed and why*. Git
solves this for code with content-addressed deltas; DeltaTrace applies the same
idea to a table, keyed by a **primary key** so "the same row" is well-defined
across versions.

## Features

- **Delta commits** — each version stores only what changed from its parent.
- **Exact time-travel** — `checkout(v)` reconstructs any past version; a test
  suite asserts `reconstruct(v) == original(v)` for every version.
- **Row + column + type tracking** — appends, deletes, cell updates, added /
  dropped columns and dtype changes are all captured.
- **Snapshots** — cache a full materialisation of a version to skip replay;
  optional auto-snapshot every *N* commits.
- **Append-only log & diffs** — `log()` and `diff(a, b)` for auditability.
- **Primary-key row identity** — works on any freshly-loaded DataFrame, not just
  one you mutate in place.
- **CLI + Python API**, Parquet storage, only `pandas` + `pyarrow` required.

## Install

```bash
git clone https://github.com/Viswas-Reddy-PallamReddy/DeltaTrace.git
cd DeltaTrace
pip install -e .          # or: pip install -e ".[dev]" to run the tests
```

## Quickstart (Python)

```python
import pandas as pd
from deltatrace import DeltaRepo

repo = DeltaRepo.init("my_store", primary_key="id", overwrite=True)

df = pd.DataFrame({"id": [1, 2, 3], "city": ["A", "B", "C"], "pop": [10, 20, 30]})
repo.commit(df, "initial load")

df2 = df.copy()
df2.loc[df2.id == 2, "pop"] = 999                                   # update a row
df2 = pd.concat([df2, pd.DataFrame({"id": [4], "city": ["D"], "pop": [40]})],
                ignore_index=True)                                  # append a row
repo.commit(df2, "bump city B, add city D")

repo.diff(1, 2)        # -> +1/-0/~1 rows; cols +[] -[]; types []
repo.checkout(1)       # -> exactly the original v1 DataFrame
```

## Quickstart (CLI)

```bash
deltatrace init   my_store --primary-key id
deltatrace commit my_store january.parquet -m "january load"
deltatrace commit my_store february.parquet -m "february load"
deltatrace log    my_store
deltatrace diff   my_store 1 2
deltatrace checkout my_store --version 1 --out v1.parquet
deltatrace snapshot my_store --version 2
```

Run the bundled demo:

```bash
python examples/demo.py
```

## How it works

### Storage layout

```
my_store/
  deltatrace.json          repo config: primary_key, head, format version
  base/v1.parquet          full materialisation of version 1
  versions/v<N>/
    metadata.json          version, parent, schema, message, stats, components
    upserts.parquet        full rows (carried columns) for new / changed ids
    deleted_ids.json       row ids removed in this version
    added_columns.parquet  values for columns introduced in this version
    removed_columns.json   names of columns dropped in this version
    updated_cells.parquet  optional (row_id, column, old, new) audit log
  snapshots/v<N>.parquet   optional full materialisation (reconstruction cache)
  logs/history.jsonl       append-only commit log (one JSON object per line)
```

### Row identity

Every row is identified by a deterministic id derived from the user-declared
**primary key** (single or composite). The same logical row therefore maps to
the same id in every version, so diffing two arbitrary DataFrames is just set
algebra on ids — no fragile positional alignment, and no need to keep mutating
one in-memory object.

### The delta model

A commit diffs the new DataFrame against the current head and decomposes the
change into independent components:

| change            | stored as                                  |
|-------------------|--------------------------------------------|
| appended rows     | full rows in `upserts.parquet`             |
| updated rows      | full rows in `upserts.parquet`             |
| deleted rows      | id list in `deleted_ids.json`              |
| added columns     | id → value map in `added_columns.parquet`  |
| dropped columns   | name list in `removed_columns.json`        |
| dtype changes     | recorded in `metadata.json` schema         |

Crucially, row-level and column-level changes are kept **separate**: adding a
column does not rewrite every row, and updating a few rows does not rewrite the
schema. Row-change detection compares only the columns carried over from the
parent, so a pure column-add commits as a cheap column delta rather than a
full-table rewrite.

### Reconstruction (`checkout`)

`checkout(v)` walks the parent chain and replays deltas:

```
reconstruct(v):
    if snapshot(v) exists:        return snapshot(v)        # short-circuit
    if v == 1:                    return base/v1
    frame = reconstruct(parent(v))                          # recurse
    frame = drop removed columns
    frame = remove deleted ids
    frame = upsert changed / new rows by id
    frame = attach added columns by id
    return cast(frame, schema(v))
```

Cost is `O(depth × rows)` in the worst case; a snapshot collapses it to a single
read, trading storage for reconstruction speed. `snapshot_interval=N` automates
this every *N* commits.

### Correctness

`tests/test_repo_roundtrip.py` builds an 8-version dataset that exercises *every*
delta type (append, delete, update, add column, drop column, dtype change,
delete) and asserts each version reconstructs exactly. `tests/test_snapshot.py`
additionally deletes every base/delta file after snapshotting and proves
`checkout` still succeeds purely from the snapshot.

```bash
pip install -e ".[dev]"
pytest -q          # 27 passed
```

## Design notes & limitations (honest)

- **In-memory engine.** Diffing and reconstruction run in pandas; this targets
  datasets that fit in memory, not billion-row tables. The on-disk format is the
  interesting part and would port to a chunked/streaming executor.
- **Snapshots are the scaling lever.** Long history chains make `checkout` walk
  many deltas; snapshots (manual or `snapshot_interval`) bound that cost.
- **Linear history.** One head, no branching/merge yet — a natural next step
  given ids and per-version metadata are already in place.
- **Primary key required.** Identity is key-based by design; there is no
  content-hash fallback (intentional — silent identity is worse than an explicit
  key).
- **Single-writer.** No concurrency control; intended for batch/offline use.

## Provenance

DeltaTrace began as a Colab notebook prototype (kept at
[`notebooks/prototype.ipynb`](notebooks/prototype.ipynb)) that demonstrated the
core idea on NYC-taxi Parquet data. This package generalises that prototype into
an installable library + CLI with key-based identity, automatic diff-on-commit,
separated row/column deltas, an append-only log, and a regression test suite.

## License

MIT — see [LICENSE](LICENSE).
