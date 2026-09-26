# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Durable intents and recovery for closed project edits."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from .asset_storage import lichtfeld_home
from .credential_storage import FileBackend
from .portal_account import _locked_sidecar

_log = logging.getLogger(__name__)
_store_locks = {}
_store_locks_guard = threading.Lock()


class ProjectOperationFailure(RuntimeError):
    def __init__(self, reason, record):
        super().__init__(reason)
        self.record = record


class ProjectOperations:
    def __init__(self, io, root=None):
        self.io = io
        self.root = Path(root or lichtfeld_home() / "data" / "asset_library").resolve()
        self.path = self.root / "contents.json"
        with _store_locks_guard:
            self._lock = _store_locks.setdefault(self.path, threading.RLock())

    def _read(self):
        raw = FileBackend(self.path).read()
        if raw is None:
            return {}
        data = json.loads(raw)
        if data.get("version") != 1 or not isinstance(data.get("operations"), dict):
            raise ValueError(f"The Contents recovery record is invalid: {self.path}")
        for identifier, row in data["operations"].items():
            if not isinstance(row, dict) or row.get("id") != identifier or not all(
                    isinstance(row.get(key), str) for key in ("path", "asset_id", "title", "status")):
                raise ValueError(f"The Contents recovery entry is invalid: {self.path}")
        return data["operations"]

    def _put(self, row):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, _locked_sidecar(self.root / "contents.lock"):
            rows = self._read()
            rows[row["id"]] = row.copy()
            finished = [key for key, value in rows.items() if value["status"] in ("completed", "failed")]
            for key in finished[:-256]:
                del rows[key]
            FileBackend(self.path).write(json.dumps(
                {"version": 1, "operations": rows}, ensure_ascii=False).encode("utf-8"))

    def run(self, identifier, asset, title, operation, *, backup=True, metadata=None):
        path = str(Path(asset["path"]).expanduser().absolute())
        if asset.get("operation_path", str(Path(path).resolve())) != str(Path(path).resolve()):
            raise ValueError("The project path changed. Refresh Projects and try again.")
        if not backup:
            if asset.get("status") != "REPAIR_ONLY":
                card = self.io.inspect_project_card(path)
                if str(card.project_uuid) != asset.get("project_uuid", asset["id"]):
                    raise ValueError("The project identity changed. Refresh Projects and try again.")
            return operation(), {}
        row = dict(id=identifier, asset_id=asset["id"], project_uuid=asset.get("project_uuid", asset["id"]),
                   path=path, title=title,
                   status="preparing", input_commit="", backup_path="")
        row.update(metadata or {})
        result = None

        def guarded():
            nonlocal result
            card = self.io.inspect_project_card(path)
            row["input_commit"] = str(card.commit_uuid)
            self._put(row)
            if backup:
                row["backup_path"] = str(self.io.backup_project_file(path))
            row["status"] = "running"
            self._put(row)
            result = operation()
            row["output_commit"] = str(self.io.inspect_project_card(path).commit_uuid)
            row["status"] = "completed"
            self._put(row)

        try:
            self.io.run_project_operation(path, row["project_uuid"], str(asset.get("commit_uuid") or ""), guarded)
        except Exception as exc:
            _log.exception("Project operation failed operation=%s path=%s", title, path)
            row["reason"] = str(exc)
            self._recover(row)
            self._put(row)
            raise ProjectOperationFailure(str(exc), row) from exc
        return result, row

    def _recover(self, row):
        path = row["path"]
        backup = row.get("backup_path")

        def guarded():
            if backup:
                backup_path = Path(backup).resolve()
                if not backup_path.is_relative_to((lichtfeld_home() / "data" / "backups" / "contents").resolve()):
                    raise ValueError("The Contents recovery copy is outside the app backup folder")
                card = self.io.inspect_project_card(path)
                if str(card.commit_uuid) != row["input_commit"]:
                    self.io.restore_project_backup(path, backup_path, row.get("project_uuid", row["asset_id"]),
                                                   str(card.commit_uuid))
                row["reason"] = row.get("reason") or "The interrupted edit was rolled back."
            else:
                row["reason"] = row.get("reason") or "The edit was interrupted before a change was saved."
            row["status"] = "failed"

        try:
            if row["status"] in ("preparing", "running") and row.get("input_commit"):
                self.io.run_project_operation(path, row.get("project_uuid", row["asset_id"]), "", guarded)
            else:
                row["status"] = "failed"
        except Exception as exc:
            if "open for writing in another" in str(exc):
                _log.info("Project recovery deferred while its writer is active path=%s: %s", path, exc)
                return
            _log.exception("Recover project operation failed operation=%s path=%s backup=%s", row["title"], path, backup)
            row.update(status="failed", reason=str(exc))

    def recover(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock, _locked_sidecar(self.root / "contents.lock"):
            rows = self._read()
        for row in rows.values():
            if row["status"] in ("preparing", "running"):
                self._recover(row)
                self._put(row)
        return rows
