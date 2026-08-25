# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Asset Manager dialog for .licht-only watched directories."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Optional

import lichtfeld as lf

from . import rml_widgets
from .asset_watch import WatchScanResult, scan_watch_directories
from .types import Panel
from .ui import RuntimeState

_log = logging.getLogger(__name__)
_watch_dirs_dialog_panel = None

__lfs_panel_classes__ = ["WatchDirsDialogPanel"]
__lfs_panel_ids__ = ["lfs.watch_dirs_dialog"]


def open_watch_dirs_dialog(
    index: Any,
    folder_id: str,
    on_catalog_changed: Optional[Callable[[], None]] = None,
) -> bool:
    """Open the watched-directory editor for an Asset Manager folder."""
    global _watch_dirs_dialog_panel
    if index is None or not folder_id:
        return False
    if _watch_dirs_dialog_panel is None:
        try:
            lf.register_class(WatchDirsDialogPanel)
        except Exception:
            _log.warning("Failed to register watched-directory panel", exc_info=True)
            return False
    return bool(
        _watch_dirs_dialog_panel
        and _watch_dirs_dialog_panel.show(index, folder_id, on_catalog_changed)
    )


class WatchDirsDialogPanel(Panel):
    """Edit directories recursively scanned for .licht project files."""

    id = "lfs.watch_dirs_dialog"
    label = "Watched Directories"
    space = lf.ui.PanelSpace.FLOATING
    order = 14
    options = {lf.ui.PanelOption.DEFAULT_CLOSED}
    template = "rmlui/watch_dirs_dialog.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (520, 0)
    update_policy = "dirty"

    def __init__(self):
        global _watch_dirs_dialog_panel
        _watch_dirs_dialog_panel = self
        self._handle = None
        self._index = None
        self._folder_id = ""
        self._folder_name = ""
        self._watch_dirs: list[str] = []
        self._on_catalog_changed: Optional[Callable[[], None]] = None
        self._reactive_unsubscribers = []
        self._scan_active = False
        self._scan_done = False
        self._scan_status = ""
        self._scan_generation = 0
        self._scan_cancel_event: Optional[threading.Event] = None
        self._scan_thread: Optional[threading.Thread] = None

    def show(
        self,
        index: Any,
        folder_id: str,
        on_catalog_changed: Optional[Callable[[], None]] = None,
    ) -> bool:
        folder = index.folders.get(folder_id)
        if folder is None:
            return False
        self._cancel_active_scan()
        self._index = index
        self._folder_id = folder_id
        self._folder_name = str(folder.get("name") or "")
        self._watch_dirs = list(index.get_watch_dirs(folder_id))
        self._on_catalog_changed = on_catalog_changed
        self._scan_generation += 1
        self._reset_scan_state()
        lf.ui.set_panel_enabled(self.id, True)
        self._dirty_model()
        return True

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("watch_dirs_dialog")
        if model is None:
            return
        model.bind_func("panel_label", self._panel_label)
        model.bind_func("subtitle", lambda: lf.ui.tr("watch_dirs.subtitle"))
        model.bind_func(
            "add_directory_label", lambda: lf.ui.tr("watch_dirs.add_directory")
        )
        model.bind_func("remove_label", lambda: lf.ui.tr("common.remove"))
        model.bind_func("empty_label", lambda: lf.ui.tr("watch_dirs.empty"))
        model.bind_func("cancel_label", lambda: lf.ui.tr("common.cancel"))
        model.bind_func("save_scan_label", self._save_scan_label)
        model.bind_func("is_empty", lambda: not self._watch_dirs)
        model.bind_func("scan_active", lambda: self._scan_active)
        model.bind_func("scan_status", lambda: self._scan_status)
        model.bind_func("scan_status_visible", lambda: bool(self._scan_status))
        model.bind_record_list("watch_dirs_list")
        model.bind_event("on_browse_add", self._on_browse_add)
        model.bind_event("on_remove_dir", self._on_remove_dir)
        model.bind_event("on_cancel", self._on_cancel)
        model.bind_event("on_save", self._on_save)
        self._handle = model.get_handle()
        self._dirty_model()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._reactive_unsubscribers = [
            RuntimeState.language_generation.subscribe(
                lambda _value: self._request_model_update()
            )
        ]

    def on_unmount(self, doc):
        self._scan_generation += 1
        self._cancel_active_scan()
        for unsubscribe in self._reactive_unsubscribers:
            try:
                unsubscribe()
            except Exception:
                pass
        self._reactive_unsubscribers = []
        self._handle = None
        doc.remove_data_model("watch_dirs_dialog")

    def _panel_label(self) -> str:
        title = lf.ui.tr("watch_dirs.title")
        return f"{title} — {self._folder_name}" if self._folder_name else title

    def _request_model_update(self) -> None:
        if self._handle:
            rml_widgets.request_model_update(self._handle)

    def _dirty_model(self) -> None:
        if not self._handle:
            return
        self._handle.update_record_list(
            "watch_dirs_list", [{"path": path} for path in self._watch_dirs]
        )
        self._handle.dirty_all()

    def _reset_scan_state(self) -> None:
        self._scan_active = False
        self._scan_done = False
        self._scan_status = ""

    def _cancel_active_scan(self) -> None:
        if self._scan_cancel_event is not None:
            self._scan_cancel_event.set()
        thread = self._scan_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join()
        self._scan_cancel_event = None
        self._scan_thread = None
        self._scan_active = False

    def _save_scan_label(self) -> str:
        if self._scan_active:
            return lf.ui.tr("watch_dirs.scanning")
        if self._scan_done:
            return lf.ui.tr("watch_dirs.done")
        return lf.ui.tr("watch_dirs.save_scan")

    def _on_browse_add(self, _handle=None, _ev=None, _args=None) -> None:
        start_dir = self._watch_dirs[-1] if self._watch_dirs else ""
        path = lf.ui.open_folder_dialog(lf.ui.tr("watch_dirs.title"), start_dir)
        if not path:
            return
        normalized = os.path.abspath(os.path.expanduser(str(path)))
        key = os.path.normcase(normalized)
        if any(os.path.normcase(existing) == key for existing in self._watch_dirs):
            return
        self._watch_dirs.append(normalized)
        self._reset_scan_state()
        self._dirty_model()

    def _on_remove_dir(self, _handle=None, _ev=None, args=None) -> None:
        if not args:
            return
        try:
            index = int(args[0])
        except (TypeError, ValueError):
            return
        if 0 <= index < len(self._watch_dirs):
            self._watch_dirs.pop(index)
            self._reset_scan_state()
            self._dirty_model()

    def _on_cancel(self, _handle=None, _ev=None, _args=None) -> None:
        self._scan_generation += 1
        self._cancel_active_scan()
        lf.ui.set_panel_enabled(self.id, False)

    def _on_save(self, _handle=None, _ev=None, _args=None) -> None:
        if self._scan_active:
            return
        if self._scan_done:
            lf.ui.set_panel_enabled(self.id, False)
            return
        index = self._index
        folder_id = self._folder_id
        directories = list(self._watch_dirs)
        if index is None or not folder_id:
            return
        if not index.set_watch_dirs(folder_id, directories):
            _log.error("Failed to save watched directories for folder %s", folder_id)
            return

        on_catalog_changed = self._on_catalog_changed
        if not directories:
            lf.ui.set_panel_enabled(self.id, False)
            self._schedule_catalog_refresh(on_catalog_changed)
            return

        self._scan_active = True
        self._scan_status = lf.ui.tr("watch_dirs.scanning")
        self._scan_generation += 1
        scan_generation = self._scan_generation
        cancel_event = threading.Event()
        self._scan_cancel_event = cancel_event
        self._dirty_model()
        thread = threading.Thread(
            target=self._scan_worker,
            args=(
                index,
                folder_id,
                directories,
                on_catalog_changed,
                scan_generation,
                cancel_event,
            ),
            daemon=True,
            name="AssetManagerLichtWatchScan",
        )
        self._scan_thread = thread
        thread.start()

    def _scan_worker(
        self,
        index: Any,
        folder_id: str,
        directories: list[str],
        on_catalog_changed: Optional[Callable[[], None]],
        scan_generation: int,
        cancel_event: threading.Event,
    ) -> None:
        try:
            result = scan_watch_directories(
                index,
                folder_id,
                directories,
                cancel_event,
            )
        except Exception:
            _log.error("Watched-directory scan failed", exc_info=True)
            result = WatchScanResult(failed=1)
        self._log_scan_result(result)

        def _finish() -> None:
            if self._scan_generation == scan_generation and not result.cancelled:
                self._scan_thread = None
                self._scan_cancel_event = None
                self._scan_active = False
                self._scan_done = True
                self._scan_status = lf.ui.tr("watch_dirs.scan_complete_summary").format(
                    discovered=result.discovered,
                    added=result.added,
                )
                self._dirty_model()
                if on_catalog_changed is not None:
                    on_catalog_changed()

        self._schedule_on_ui_thread(_finish)

    @staticmethod
    def _log_scan_result(result: WatchScanResult) -> None:
        _log.info(
            "Watched-directory scan finished: discovered=%d added=%d existing=%d failed=%d",
            result.discovered,
            result.added,
            result.already_cataloged,
            result.failed,
        )

    @staticmethod
    def _schedule_catalog_refresh(
        callback: Optional[Callable[[], None]],
    ) -> None:
        if callback is None:
            return
        WatchDirsDialogPanel._schedule_on_ui_thread(callback)

    @staticmethod
    def _schedule_on_ui_thread(callback: Callable[[], None]) -> None:
        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if not callable(scheduler):
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
        if callable(scheduler):
            try:
                scheduler(callback)
                return
            except Exception:
                _log.warning("Failed to schedule Asset Manager refresh", exc_info=True)
        callback()
