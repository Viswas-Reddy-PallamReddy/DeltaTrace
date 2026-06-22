"""Command-line interface for DeltaTrace.

    deltatrace init   <root> --primary-key id [--snapshot-interval N] [--overwrite]
    deltatrace commit <root> data.parquet -m "message" [--audit]
    deltatrace log    <root>
    deltatrace checkout <root> [--version N] [--out out.parquet]
    deltatrace diff   <root> <v1> <v2>
    deltatrace snapshot <root> [--version N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

from . import __version__
from .repo import DeltaRepo
from .storage import StoreError


def _read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    raise SystemExit(f"unsupported input format: {suffix!r} (use .parquet or .csv)")


def _write_table(df: pd.DataFrame, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix in (".csv", ".txt"):
        df.to_csv(path, index=False)
    else:
        raise SystemExit(f"unsupported output format: {suffix!r} (use .parquet or .csv)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltatrace", description=__doc__)
    parser.add_argument("--version", action="version", version=f"deltatrace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create a new repository")
    p_init.add_argument("root")
    p_init.add_argument("--primary-key", "-k", required=True, nargs="+")
    p_init.add_argument("--snapshot-interval", type=int, default=None)
    p_init.add_argument("--overwrite", action="store_true")

    p_commit = sub.add_parser("commit", help="commit a data file as the next version")
    p_commit.add_argument("root")
    p_commit.add_argument("data")
    p_commit.add_argument("--message", "-m", default="")
    p_commit.add_argument("--audit", action="store_true", help="also write a cell-level change log")

    p_log = sub.add_parser("log", help="show the commit history")
    p_log.add_argument("root")

    p_checkout = sub.add_parser("checkout", help="materialise a version")
    p_checkout.add_argument("root")
    p_checkout.add_argument("--version", "-v", type=int, default=None)
    p_checkout.add_argument("--out", "-o", default=None)

    p_diff = sub.add_parser("diff", help="diff two versions")
    p_diff.add_argument("root")
    p_diff.add_argument("from_version", type=int)
    p_diff.add_argument("to_version", type=int)

    p_snap = sub.add_parser("snapshot", help="cache a full materialisation of a version")
    p_snap.add_argument("root")
    p_snap.add_argument("--version", "-v", type=int, default=None)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            repo = DeltaRepo.init(
                args.root,
                args.primary_key,
                snapshot_interval=args.snapshot_interval,
                overwrite=args.overwrite,
            )
            print(f"initialised empty DeltaTrace repo at {args.root} "
                  f"(primary key: {', '.join(repo.primary_key)})")

        elif args.command == "commit":
            repo = DeltaRepo.open(args.root)
            result = repo.commit(_read_table(args.data), message=args.message, audit=args.audit)
            print(result)

        elif args.command == "log":
            repo = DeltaRepo.open(args.root)
            for entry in repo.log():
                s = entry["stats"]
                print(
                    f"v{entry['version']:<3} {entry['timestamp']}  "
                    f"+{s['added']}/-{s['deleted']}/~{s['updated']} rows  "
                    f"{entry['message']}"
                )

        elif args.command == "checkout":
            repo = DeltaRepo.open(args.root)
            df = repo.checkout(args.version)
            if args.out:
                _write_table(df, args.out)
                print(f"wrote {len(df)} rows x {df.shape[1]} cols to {args.out}")
            else:
                version = args.version if args.version is not None else repo.head
                print(f"# v{version}: {len(df)} rows x {df.shape[1]} cols")
                print(df.head(10).to_string(index=False))

        elif args.command == "diff":
            repo = DeltaRepo.open(args.root)
            print(repo.diff(args.from_version, args.to_version))

        elif args.command == "snapshot":
            repo = DeltaRepo.open(args.root)
            path = repo.snapshot(args.version)
            print(f"wrote snapshot {path}")

    except StoreError as exc:
        raise SystemExit(f"error: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
