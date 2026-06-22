"""Runnable demo for DeltaTrace.

    python examples/demo.py

Versions a small table through updates, appends and schema changes, then shows
the log, a diff, time-travel checkout, and snapshot-accelerated reconstruction.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pandas as pd

from deltatrace import DeltaRepo


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="deltatrace_demo_"))
    repo = DeltaRepo.init(tmp / "store", primary_key="id", overwrite=True)

    df1 = pd.DataFrame({"id": [1, 2, 3], "city": ["A", "B", "C"], "pop": [10, 20, 30]})
    print(repo.commit(df1, "v1: initial load"))

    df2 = df1.copy()
    df2.loc[df2["id"] == 2, "pop"] = 999  # update an existing row
    df2 = pd.concat(  # append a new row
        [df2, pd.DataFrame({"id": [4], "city": ["D"], "pop": [40]})], ignore_index=True
    )
    print(repo.commit(df2, "v2: bump city B, add city D"))

    df3 = df2.drop(columns=["city"])  # drop a column ...
    df3["density"] = df3["pop"] / 100  # ... and add one
    print(repo.commit(df3, "v3: drop city, add density"))

    print("\n-- log --")
    for e in repo.log():
        print(f"  v{e['version']}: {e['message']}  {e['stats']}")

    print("\n-- diff v1 -> v3 --")
    print(" ", repo.diff(1, 3))

    print("\n-- checkout v1 (time-travel) --")
    print(repo.checkout(1).to_string(index=False))

    # Snapshot acceleration: cache a full materialisation so checkout skips replay.
    t0 = time.perf_counter()
    repo.checkout(3)
    t_replay = time.perf_counter() - t0
    repo.snapshot(3)
    t0 = time.perf_counter()
    repo.checkout(3)
    t_snapshot = time.perf_counter() - t0
    print(f"\ncheckout(v3): replay={t_replay * 1e3:.2f} ms  snapshot={t_snapshot * 1e3:.2f} ms")
    print(f"store written to {tmp / 'store'}")


if __name__ == "__main__":
    main()
