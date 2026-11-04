# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained UI: stable actions and independent current/browsed session state."""
from html import escape
import os
import time
from concurrent.futures import ThreadPoolExecutor
import lichtfeld as lf
from .transport import discover
from .viewer import Viewer
from .presentation import title, can_open, option_label, preview_heading, preview_quality, preview_cost, TERMINAL, _tr

_viewer = None
_discovery = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trainer-discovery")


class TrainingSessionsPanel(lf.ui.Panel):
    id = "lfs.training_sessions"
    label = "external_training.panel"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 35
    template = "rmlui/external_training.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    update_policy = "interval"
    update_interval_ms = 500

    def __init__(self):
        self.sessions = []
        self.scan = _discovery.submit(discover)
        self.records = []
        self.next_scan = 0
        self.selected_id = None
        self.loaded = False
        self.scan_error = ""
        self.error = ""
        self.browser_open = False
        self.current_details = False
        self.selected_details = False
        self.doc = None
        self.cache = {}
        self.rendering = False
        self.pending_scroll = None

    def on_update(self, doc):
        if self.scan and self.scan.done():
            try:
                self.records = [item for item in self.scan.result() if item["pid"] != os.getpid()]
                self.scan_error = ""
                self.loaded = True
            except Exception as error:
                self.scan_error = str(error)
            self.scan = None
            self.next_scan = time.monotonic() + 2
        browse_needed = _viewer is None or self.browser_open or _viewer.stop.is_set()
        if browse_needed and self.scan is None and time.monotonic() >= self.next_scan:
            self.scan = _discovery.submit(discover)
        changed = self.render()
        if self.pending_scroll and self.doc is not None:
            if self.pending_scroll == "browser":
                self.doc.get_element_by_id("et-browse-toggle").scroll_into_view(True)
            else:
                self.doc.get_element_by_id("content-wrap").scroll_top = 0
            self.pending_scroll = None
            changed = True
        return changed

    def connect(self, selected):
        global _viewer
        if not can_open(selected):
            self.error = _tr("external_training.not_available_choose")
            return
        replacement = None
        try:
            replacement = Viewer(selected["directory"])
            replacement.session_info = dict(selected)
            if _viewer:
                if (_viewer.session_info["id"] == selected["id"] and _viewer.snapshot_in_scene()
                        and _viewer.display_arrays is not None):
                    replacement.display_arrays = _viewer.display_arrays
                    replacement.display_snapshot = _viewer.display_snapshot
                    replacement.shown_iteration = _viewer.shown_iteration
                    replacement.display_geometry = _viewer.display_geometry
                    replacement.display_bounds = _viewer.display_bounds
                    replacement.cameras_visible = _viewer.cameras_visible
                    replacement.camera_count = _viewer.camera_count
                if _viewer.session_info["id"] == selected["id"]:
                    _viewer.close(update_status=False)  # The replacement now owns this retained node's status.
                else:
                    _viewer.discard()
                    # Removing the old node advances the scene generation. Capture
                    # the new anchor guard after this intentional local mutation.
                    replacement.scene = lf.get_scene()
            _viewer = replacement
            self.error = ""
            self.browser_open = False
            self.current_details = False
        except Exception as error:
            if replacement is not None:
                replacement.close()
            self.error = str(error)

    def current_record(self):
        identifier = _viewer.session_info["id"] if _viewer else None
        record = next((item for item in self.records if item["id"] == identifier), {})
        if _viewer and not _viewer.stop.is_set():
            status = _viewer.reader.status
            if status.get('id') == identifier:
                # The existing reader validates identity and liveness. Monitoring
                # this trainer does not need another full registry scan.
                record = {**_viewer.session_info, **record, **status,
                          'active': status.get('liveness') == 'responsive' and status.get('state') not in TERMINAL}
        return record

    def selected_record(self):
        return next((item for item in self.sessions if item["id"] == self.selected_id), {})

    @staticmethod
    def details(record):
        location = record.get("project") or record.get("dataset") or record.get("job_root") or ""
        identity = _tr("external_training.identity_details", pid=str(record.get('pid', '?')), session=record.get('id', '')[:8])
        return f"{location}\n{identity}".strip()

    def view_model(self):
        viewer = _viewer
        current_id = viewer.session_info["id"] if viewer else None
        current = self.current_record()
        self.sessions = [item for item in self.records if item["id"] != current_id
                         and item.get("active")]
        if self.sessions and not any(item["id"] == self.selected_id for item in self.sessions):
            self.selected_id = self.sessions[0]["id"]
        selected = self.selected_record()
        available = can_open(selected)
        active_count = sum(bool(item.get("active")) for item in self.records if item["id"] != current_id)
        texts = {"browse-toggle": (_tr("external_training.browse_hide", count=active_count) if self.browser_open else _tr("external_training.browse_show")),
                 "available-title": _tr("external_training.available"),
                 "selected-details": self.details(selected),
                 "switch-help": _tr("external_training.switch_help"),
                 "scan-error": _tr("external_training.scan_error", error=self.scan_error),
                 "connect-error": _tr("external_training.connect_error", error=self.error)}
        visible = {"intro": viewer is None, "current": viewer is not None,
                   "available-title": viewer is None, "browse-toggle": viewer is not None,
                   "browser": viewer is None or self.browser_open, "choices": bool(self.sessions),
                   "empty": not self.sessions, "unavailable": bool(selected) and not available,
                   "selected-details": bool(selected) and self.selected_details,
                   "scan-error": bool(self.scan_error), "connect-error": bool(self.error),
                   "switch-help": bool(viewer and not viewer.stop.is_set() and available)}
        enabled = {"open": available, "refresh": True}
        if not self.loaded:
            texts["empty"] = _tr("external_training.list_unavailable" if self.scan_error else "external_training.searching")
        else:
            texts["empty"] = _tr("external_training.none_other" if viewer else "external_training.none")
        texts["open"] = _tr("external_training.switch" if viewer and not viewer.stop.is_set() else "external_training.connect")
        if viewer:
            status = viewer.reader.status
            stopped, snapshot, error = viewer.stop.is_set(), viewer.snapshot_in_scene(), viewer.reader.error or status.get("error", "")
            info = {**viewer.session_info, **current, **status}
            step, total = status.get("iteration", 0), status.get("iterations", 0)
            texts.update({"current-name": title(info),
                "current-state": preview_heading(status, stopped, error, snapshot),
                "training-step": _tr("external_training.training_step_total", step=step, total=total) if total else _tr("external_training.training_step", step=step),
                "preview-step": _tr("external_training.preview_step", step=viewer.shown_iteration),
                "preview-quality": preview_quality(viewer.display_snapshot),
                "preview-cost": preview_cost(status),
                "saved-project": _tr("external_training.saved_project", path=str(status.get("saved_project", ""))),
                "current-error": str(error), "copy-error": viewer.copy_error,
                "cameras": _tr("external_training.cameras_hide" if viewer.cameras_visible else "external_training.cameras_show", count=viewer.camera_count),
                "camera-error": viewer.camera_error or status.get("geometry_error", ""),
                "current-details": self.details(info),
                "reconnect": _tr("external_training.reconnect")})
            message = ""
            if stopped:
                message = _tr("external_training.retained_readonly" if snapshot else "external_training.no_snapshot")
            elif status.get("state") == "disconnected":
                message = _tr("external_training.trainer_unavailable_retained" if snapshot else "external_training.trainer_unavailable_empty")
            elif not snapshot and not error:
                message = _tr("external_training.ended_no_snapshot" if status.get("state") in TERMINAL else
                              "external_training.paused_waiting" if status.get("state") == "paused" else "external_training.waiting_first")
            if status.get("state") in TERMINAL and snapshot and not stopped:
                message = _tr("external_training.ended_retained")
            texts["current-message"] = message
            visible.update({"current-message": bool(message), "training-step": bool(total or step),
                "preview-step": snapshot, "preview-quality": snapshot,
                "preview-cost": snapshot and bool(preview_cost(status)),
                "saved-project": bool(status.get("saved_project")), "copy": snapshot, "copy-busy": snapshot and viewer.copy_busy,
                "copy-error": bool(viewer.copy_error), "disconnect": not stopped,
                "reconnect": stopped and can_open(current), "discard": True,
                "reconnect-missing": stopped and self.loaded and not can_open(current),
                "current-details": self.current_details, "current-error": bool(error),
                "cameras": bool(viewer.camera_count), "camera-error": bool(viewer.camera_error or status.get("geometry_error")),
                "frame": snapshot,
                "retry-help": bool(error) and not stopped})
            enabled["copy"] = snapshot and not viewer.copy_busy
        return {"text": texts, "visible": visible, "enabled": enabled, "checked": {},
                "options": [(item["id"], option_label(item)) for item in self.sessions]}

    def on_mount(self, doc):
        self.doc = doc
        self.cache.clear()
        def listen(identifier, event="click"):
            element = doc.get_element_by_id("et-" + identifier)
            if element is None:
                raise RuntimeError("External training UI element missing: " + identifier)
            element.add_event_listener(event, lambda ev: self.dispatch(identifier, ev))
        for identifier in ("copy", "disconnect", "discard", "reconnect", "open", "refresh", "browse-toggle", "current-details-toggle", "selected-details-toggle", "cameras", "frame"):
            listen(identifier)
        listen("choice", "change")
        self.render()

    def on_unmount(self, doc):
        self.doc = None
        self.cache.clear()

    def dispatch(self, identifier, event):
        global _viewer
        if self.rendering:
            return
        if identifier == "choice":
            self.selected_id = event.get_parameter("value", "")
            self.selected_details = False
        elif identifier == "open":
            self.connect(self.selected_record())
        elif identifier == "reconnect":
            self.connect(self.current_record())
        elif identifier == "copy" and _viewer and _viewer.snapshot_in_scene() and not _viewer.copy_busy:
            _viewer.create_local_copy()
        elif identifier == "discard" and _viewer:
            try:
                _viewer.discard()
                _viewer = None
                self.error = ""
            except Exception as error:
                self.error = str(error)
        elif identifier == "disconnect" and _viewer:
            _viewer.close()
            self.next_scan = 0
        elif identifier == "cameras" and _viewer:
            _viewer.toggle_cameras()
        elif identifier == "frame" and _viewer:
            _viewer.frame_preview()
        elif identifier == "refresh" and self.scan is None:
            self.scan = _discovery.submit(discover)
        elif identifier == "browse-toggle":
            self.browser_open = not self.browser_open
            if self.browser_open:
                self.next_scan = 0
        elif identifier == "current-details-toggle":
            self.current_details = not self.current_details
        elif identifier == "selected-details-toggle":
            self.selected_details = not self.selected_details
        self.render()
        if self.doc is not None:
            if identifier == "browse-toggle" and self.browser_open:
                self.pending_scroll = "browser"
            elif identifier in {"open", "reconnect", "browse-toggle"} and not self.error:
                self.pending_scroll = "top"
        lf.ui.request_redraw()

    def render(self):
        if self.doc is None or self.rendering:
            return False
        self.rendering = True
        changed = False
        try:
            model = self.view_model()
            for kind in ("text", "visible", "enabled", "checked"):
                for identifier, value in model[kind].items():
                    key = (kind, identifier)
                    if self.cache.get(key) == value:
                        continue
                    element = self.doc.get_element_by_id("et-" + identifier)
                    if element is None:
                        raise RuntimeError("External training UI element missing: " + identifier)
                    self.cache[key] = value
                    if kind == "text": element.set_text(value)
                    elif kind == "visible": element.set_class("hidden", not value)
                    elif kind == "checked":
                        if value: element.set_attribute("checked", "checked")
                        else: element.remove_attribute("checked")
                    elif value: element.remove_attribute("disabled")
                    else: element.set_attribute("disabled", "")
                    changed = True
            name = self.doc.get_element_by_id("et-current-name")
            current_name = model['text'].get('current-name', '')
            if self.cache.get(('title', 'current-name')) != current_name:
                name.set_attribute('title', current_name)
                self.cache[('title', 'current-name')] = current_name
                changed = True
            options = [(identifier, label, identifier == self.selected_id) for identifier, label in model["options"]]
            if self.cache.get("options") != options:
                self.cache["options"] = options
                markup = "".join(f'<option value="{escape(identifier, quote=True)}"' + (' selected="selected"' if selected else '') + f'>{escape(label)}</option>' for identifier, label, selected in options)
                self.doc.get_element_by_id("et-choice").set_inner_rml(markup)
                changed = True
        finally:
            self.rendering = False
        return changed


def register():
    lf.register_class(TrainingSessionsPanel)
