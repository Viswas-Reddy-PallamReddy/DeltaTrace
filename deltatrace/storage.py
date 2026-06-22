"""Filesystem layout and (de)serialisation helpers for a DeltaTrace store.

Store layout::

    <root>/
      deltatrace.json          repo config: primary_key, head, format_version
      base/v1.parquet          full materialisation of version 1
      versions/v<N>/
        metadata.json          version, parent, schema, message, stats, components
        upserts.parquet        full rows (carried columns) for new/changed ids
        deleted_ids.json       row ids removed in this version
        added_columns.parquet  values for columns introduced in this version
        removed_columns.json   names of columns dropped in this version
        updated_cells.parquet  optional (row_id, column, old, new) audit log
      snapshots/v<N>.parquet   optional full materialisation (reconstruction cache)
      logs/history.jsonl       append-only commit log (one JSON object per line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

CONFIG_NAME = "deltatrace.json"
FORMAT_VERSION = 1


class StoreError(RuntimeError):
    """Raised for store-level problems (missing repo, bad version, ...)."""


class Store:
    """Thin wrapper over the on-disk directory structure of a repository."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # -- directories ----------------------------------------------------
    @property
    def base_dir(self) -> Path:
        return self.root / "base"

    @property
    def versions_dir(self) -> Path:
        return self.root / "versions"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def version_dir(self, version: int) -> Path:
        return self.versions_dir / f"v{version}"

    def base_path(self) -> Path:
        return self.base_dir / "v1.parquet"

    def snapshot_path(self, version: int) -> Path:
        return self.snapshots_dir / f"v{version}.parquet"

    def history_path(self) -> Path:
        return self.logs_dir / "history.jsonl"

    def config_path(self) -> Path:
        return self.root / CONFIG_NAME

    # -- lifecycle ------------------------------------------------------
    def scaffold(self) -> None:
        for d in (self.base_dir, self.versions_dir, self.snapshots_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)
        if not self.history_path().exists():
            self.history_path().write_text("", encoding="utf-8")

    def exists(self) -> bool:
        return self.config_path().exists()

    # -- config ---------------------------------------------------------
    def write_config(self, config: Dict) -> None:
        self.config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")

    def read_config(self) -> Dict:
        if not self.exists():
            raise StoreError(f"no DeltaTrace repository at {self.root}")
        return json.loads(self.config_path().read_text(encoding="utf-8"))

    # -- parquet / json io ---------------------------------------------
    @staticmethod
    def write_parquet(df: Optional[pd.DataFrame], path: Path) -> bool:
        """Write ``df`` to parquet. Returns False (writes nothing) if empty."""
        if df is None or df.empty:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return True

    @staticmethod
    def read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def write_json(data, path: Path) -> bool:
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return True

    @staticmethod
    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    # -- metadata / history --------------------------------------------
    def write_metadata(self, version: int, meta: Dict) -> None:
        vdir = self.version_dir(version)
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

    def read_metadata(self, version: int) -> Dict:
        path = self.version_dir(version) / "metadata.json"
        if not path.exists():
            raise StoreError(f"metadata.json missing for v{version}")
        return json.loads(path.read_text(encoding="utf-8"))

    def append_history(self, entry: Dict) -> None:
        with self.history_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def read_history(self) -> List[Dict]:
        path = self.history_path()
        if not path.exists():
            return []
        lines: Iterable[str] = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
