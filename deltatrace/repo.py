"""The DeltaTrace repository: git-style version control for tabular data.

A :class:`DeltaRepo` records a sequence of versions of a DataFrame. Each commit
stores only the *delta* from its parent (new/changed rows, deleted ids, added or
dropped columns, dtype changes) plus a small metadata file. Any version can be
materialised again by walking the parent chain and replaying the deltas
(:meth:`checkout`), optionally short-circuited by a snapshot.

Example
-------
>>> repo = DeltaRepo.init("/tmp/store", primary_key="id", overwrite=True)
>>> repo.commit(df_v1, message="initial load")
>>> repo.commit(df_v2, message="add rows")
>>> later = repo.checkout(1)        # exactly reproduces df_v1
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from .diffing import (
    cell_level_changes,
    changed_row_ids,
    column_diff,
    row_diff,
    type_changes,
)
from .identity import RID, normalize_keys, with_row_ids
from .storage import FORMAT_VERSION, Store, StoreError

__all__ = ["DeltaRepo", "CommitResult", "DiffResult", "StoreError"]


@dataclass
class CommitResult:
    version: int
    parent: int
    message: str
    stats: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        s = self.stats
        return (
            f"v{self.version} (parent v{self.parent}): "
            f"+{s.get('added', 0)} rows, -{s.get('deleted', 0)} rows, "
            f"~{s.get('updated', 0)} rows, "
            f"+{s.get('columns_added', 0)}/-{s.get('columns_removed', 0)} cols, "
            f"{s.get('types_changed', 0)} type change(s)"
        )


@dataclass
class DiffResult:
    from_version: int
    to_version: int
    rows_added: int
    rows_deleted: int
    rows_updated: int
    columns_added: List[str] = field(default_factory=list)
    columns_removed: List[str] = field(default_factory=list)
    type_changes: List[Dict[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"diff v{self.from_version} -> v{self.to_version}: "
            f"+{self.rows_added}/-{self.rows_deleted}/~{self.rows_updated} rows; "
            f"cols +{self.columns_added or '[]'} -{self.columns_removed or '[]'}; "
            f"types {self.type_changes or '[]'}"
        )


class DeltaRepo:
    """A versioned store for a single logical table, keyed by a primary key."""

    def __init__(self, store: Store, primary_key: List[str], head: int,
                 snapshot_interval: Optional[int]):
        self.store = store
        self._keys = primary_key
        self._head = head
        self._snapshot_interval = snapshot_interval

    # -- construction ---------------------------------------------------
    @classmethod
    def init(
        cls,
        root: Union[str, Path],
        primary_key: Union[str, Sequence[str]],
        *,
        snapshot_interval: Optional[int] = None,
        overwrite: bool = False,
    ) -> "DeltaRepo":
        keys = normalize_keys(primary_key)
        store = Store(root)
        if store.exists():
            if not overwrite:
                raise StoreError(
                    f"a DeltaTrace repository already exists at {store.root} "
                    "(pass overwrite=True to replace it)"
                )
            shutil.rmtree(store.root)
        store.scaffold()
        store.write_config(
            {
                "format_version": FORMAT_VERSION,
                "primary_key": keys,
                "head": 0,
                "snapshot_interval": snapshot_interval,
            }
        )
        return cls(store, keys, head=0, snapshot_interval=snapshot_interval)

    @classmethod
    def open(cls, root: Union[str, Path]) -> "DeltaRepo":
        store = Store(root)
        cfg = store.read_config()
        return cls(
            store,
            list(cfg["primary_key"]),
            head=int(cfg.get("head", 0)),
            snapshot_interval=cfg.get("snapshot_interval"),
        )

    # -- introspection --------------------------------------------------
    @property
    def primary_key(self) -> List[str]:
        return list(self._keys)

    @property
    def keys(self) -> List[str]:
        return list(self._keys)

    @property
    def head(self) -> int:
        return self._head

    @property
    def versions(self) -> List[int]:
        return list(range(1, self._head + 1))

    def __len__(self) -> int:
        return self._head

    def __repr__(self) -> str:
        return (
            f"DeltaRepo(root={str(self.store.root)!r}, "
            f"primary_key={self._keys}, head={self._head})"
        )

    # -- commit ---------------------------------------------------------
    def commit(self, df: pd.DataFrame, message: str = "", *, audit: bool = False) -> CommitResult:
        """Record ``df`` as the next version, storing only its delta from head."""
        vdf = with_row_ids(df, self._keys)
        user_cols = [c for c in vdf.columns if c != RID]
        schema = {
            "columns": user_cols,
            "dtypes": {c: str(vdf[c].dtype) for c in user_cols},
        }

        if self._head == 0:
            return self._commit_base(vdf, user_cols, schema, message)

        parent = self._head
        new_version = parent + 1
        head_frame = self._materialize(parent)
        parent_cols = [c for c in head_frame.columns if c != RID]

        cdiff = column_diff(parent_cols, user_cols)
        added_cols, removed_cols, common_cols = (
            cdiff["added"], cdiff["removed"], cdiff["common"],
        )

        rdiff = row_diff(head_frame[RID].tolist(), vdf[RID].tolist())
        changed = changed_row_ids(head_frame, vdf, rdiff.unchanged, common_cols)

        upsert_ids = set(rdiff.appended) | set(changed)
        carried = [c for c in user_cols if c in set(common_cols)]
        upserts = (
            vdf[vdf[RID].isin(upsert_ids)][[RID] + carried].copy()
            if upsert_ids else None
        )

        added_columns = None
        if added_cols:
            sub = vdf[[RID] + added_cols].copy()
            keep = sub[added_cols].notna().any(axis=1) | sub[RID].isin(rdiff.appended)
            sub = sub[keep]
            added_columns = sub if not sub.empty else None

        tchanges = type_changes(head_frame, vdf, common_cols)

        vdir = self.store.version_dir(new_version)
        vdir.mkdir(parents=True, exist_ok=True)
        components: Dict[str, str] = {}
        if self.store.write_parquet(upserts, vdir / "upserts.parquet"):
            components["upserts"] = "upserts.parquet"
        if self.store.write_json(sorted(rdiff.deleted), vdir / "deleted_ids.json"):
            components["deleted_ids"] = "deleted_ids.json"
        if self.store.write_parquet(added_columns, vdir / "added_columns.parquet"):
            components["added_columns"] = "added_columns.parquet"
        if self.store.write_json(removed_cols, vdir / "removed_columns.json"):
            components["removed_columns"] = "removed_columns.json"
        if audit:
            cells = cell_level_changes(head_frame, vdf, changed, common_cols)
            if self.store.write_parquet(cells, vdir / "updated_cells.parquet"):
                components["updated_cells"] = "updated_cells.parquet"

        stats = {
            "added": len(rdiff.appended),
            "deleted": len(rdiff.deleted),
            "updated": len(changed),
            "columns_added": len(added_cols),
            "columns_removed": len(removed_cols),
            "types_changed": len(tchanges),
        }
        meta = self._build_meta(
            new_version, parent, schema, message, stats, components, tchanges
        )
        self.store.write_metadata(new_version, meta)
        self.store.append_history(self._history_entry(meta))
        self._set_head(new_version)

        if self._snapshot_interval and new_version % self._snapshot_interval == 0:
            self.snapshot(new_version)
        return CommitResult(new_version, parent, message, stats)

    def _commit_base(self, vdf, user_cols, schema, message) -> CommitResult:
        self.store.write_parquet(vdf[[RID] + user_cols], self.store.base_path())
        stats = {
            "added": len(vdf), "deleted": 0, "updated": 0,
            "columns_added": len(user_cols), "columns_removed": 0, "types_changed": 0,
        }
        meta = self._build_meta(
            1, 0, schema, message, stats, {"base": "base/v1.parquet"}, []
        )
        self.store.write_metadata(1, meta)
        self.store.append_history(self._history_entry(meta))
        self._set_head(1)
        return CommitResult(1, 0, message, stats)

    # -- reconstruction -------------------------------------------------
    def checkout(self, version: Optional[int] = None) -> pd.DataFrame:
        """Return the materialised DataFrame for ``version`` (default: head)."""
        version = self._resolve_version(version)
        df = self._materialize(version)
        if RID in df.columns:
            df = df.drop(columns=[RID])
        return df.sort_values(by=self._keys, kind="stable").reset_index(drop=True)

    # Backwards-friendly alias mirroring the prototype's vocabulary.
    reconstruct = checkout

    def _materialize(self, version: int) -> pd.DataFrame:
        snap = self.store.snapshot_path(version)
        if snap.exists():
            return self.store.read_parquet(snap)
        meta = self.store.read_metadata(version)
        if version == 1 or meta.get("parent") in (0, None):
            base = self.store.read_parquet(self.store.base_path())
            return self._finalize_schema(base, meta["schema"])
        parent_frame = self._materialize(int(meta["parent"]))
        return self._apply_delta(parent_frame, version, meta)

    def _apply_delta(self, parent_frame: pd.DataFrame, version: int, meta: Dict) -> pd.DataFrame:
        schema = meta["schema"]
        target_cols = schema["columns"]
        vdir = self.store.version_dir(version)
        df = parent_frame.copy()

        rc_path = vdir / "removed_columns.json"
        if rc_path.exists():
            removed = self.store.read_json(rc_path)
            df = df.drop(columns=[c for c in removed if c in df.columns], errors="ignore")

        di_path = vdir / "deleted_ids.json"
        if di_path.exists():
            deleted = set(self.store.read_json(di_path))
            df = df[~df[RID].isin(deleted)]

        up_path = vdir / "upserts.parquet"
        if up_path.exists():
            up = self.store.read_parquet(up_path)
            df = df[~df[RID].isin(set(up[RID]))]
            df = pd.concat([df, up], ignore_index=True)

        ac_path = vdir / "added_columns.parquet"
        if ac_path.exists():
            add = self.store.read_parquet(ac_path).set_index(RID)
            for col in add.columns:
                df[col] = df[RID].map(add[col])

        for col in target_cols:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[[RID] + target_cols]
        return self._finalize_schema(df, schema)

    def _finalize_schema(self, df: pd.DataFrame, schema: Dict) -> pd.DataFrame:
        target_cols = schema["columns"]
        cols = ([RID] if RID in df.columns else []) + target_cols
        df = df.reindex(columns=cols)
        for col, dtype in schema["dtypes"].items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (TypeError, ValueError):
                    pass
        return df.reset_index(drop=True)

    # -- snapshots ------------------------------------------------------
    def snapshot(self, version: Optional[int] = None) -> Path:
        """Materialise ``version`` fully and cache it to short-circuit replay."""
        version = self._resolve_version(version)
        df = self._materialize(version)
        self.store.snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self.store.snapshot_path(version)
        df.to_parquet(path, index=False)
        return path

    # -- history / diff -------------------------------------------------
    def log(self) -> List[Dict]:
        """Return the append-only commit log, newest last."""
        return self.store.read_history()

    def diff(self, from_version: int, to_version: int) -> DiffResult:
        a_v = self._resolve_version(from_version)
        b_v = self._resolve_version(to_version)
        a = self._materialize(a_v)
        b = self._materialize(b_v)
        rdiff = row_diff(a[RID].tolist(), b[RID].tolist())
        cdiff = column_diff(
            [c for c in a.columns if c != RID],
            [c for c in b.columns if c != RID],
        )
        changed = changed_row_ids(a, b, rdiff.unchanged, cdiff["common"])
        return DiffResult(
            from_version=a_v,
            to_version=b_v,
            rows_added=len(rdiff.appended),
            rows_deleted=len(rdiff.deleted),
            rows_updated=len(changed),
            columns_added=cdiff["added"],
            columns_removed=cdiff["removed"],
            type_changes=type_changes(a, b, cdiff["common"]),
        )

    # -- internals ------------------------------------------------------
    def _resolve_version(self, version: Optional[int]) -> int:
        if version is None:
            version = self._head
        if self._head == 0:
            raise StoreError("repository is empty; commit something first")
        if not (1 <= version <= self._head):
            raise StoreError(f"version v{version} out of range (head is v{self._head})")
        return int(version)

    def _set_head(self, version: int) -> None:
        self._head = version
        cfg = self.store.read_config()
        cfg["head"] = version
        self.store.write_config(cfg)

    @staticmethod
    def _build_meta(version, parent, schema, message, stats, components, type_changes=None) -> Dict:
        return {
            "version": version,
            "parent": parent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "schema": schema,
            "stats": stats,
            "components": components,
            "type_changes": type_changes or [],
        }

    @staticmethod
    def _history_entry(meta: Dict) -> Dict:
        return {
            "version": meta["version"],
            "parent": meta["parent"],
            "timestamp": meta["timestamp"],
            "message": meta["message"],
            "stats": meta["stats"],
        }
