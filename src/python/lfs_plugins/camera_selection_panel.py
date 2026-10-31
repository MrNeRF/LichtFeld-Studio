# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera selection controls opened from the Scene Graph."""

import lichtfeld as lf
from .panels import panel_class
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .types import Panel
from .ui import RuntimeState, PanelStateBinding, native_value as _native_store_value

SCRUB_FIELD_DEFS = {
    "percentage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int),
    "overlap": ScrubFieldSpec(0, 100, 1, "%d", data_type=int),
    "coverage": ScrubFieldSpec(0, 100, 1, "%d", data_type=int),
}
OPERATION_LABEL_KEYS = {
    "random": "scene.select_random_cameras",
    "every_n": "scene.select_every_n_cameras",
    "distinct": "scene.select_distinct_cameras",
    "cropbox_visibility": "scene.select_cropbox_cameras",
}


@panel_class("camera_selection")
class CameraSelectionPanel(Panel):
    def __init__(self):
        self._handle = None
        self._operation = "random"
        self._percentage, self._every_n = 25, 2
        self._overlap, self._coverage, self._cropbox_id = 65, 40, ""
        self._local_error = ""
        self._task_error = ""
        self._running = False
        self._cropbox_records = []
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS, self._get_scrub_value, self._set_scrub_value)
        self._reactive_binding = PanelStateBinding()

    def on_bind_model(self, ctx):
        self._cropbox_records = None
        model = ctx.create_data_model("camera_selection")
        if model is None:
            return
        model.bind_func("panel_label", self._panel_label)
        for name, operation in (("is_random", "random"), ("is_every_n", "every_n"),
                                ("is_distinct", "distinct"), ("is_cropbox", "cropbox_visibility")):
            model.bind_func(name, lambda operation=operation: self._operation == operation)
        model.bind("percentage", lambda: str(self._percentage), lambda value: self._set_int("_percentage", value, 0, 100))
        model.bind("every_n", lambda: str(self._every_n), lambda value: self._set_int("_every_n", value, 1, 1000000))
        model.bind("overlap", lambda: str(self._overlap), lambda value: self._set_int("_overlap", value, 0, 100))
        model.bind("coverage", lambda: str(self._coverage), lambda value: self._set_int("_coverage", value, 0, 100))
        model.bind("cropbox_id", lambda: self._cropbox_id, self._set_cropbox)
        model.bind_record_list("cropboxes")
        model.bind_func("message", lambda: self._local_error or self._task_error)
        model.bind_func("running", lambda: self._running)
        model.bind_event("apply", self._apply)
        model.bind_event("cancel", self._cancel)
        self._handle = model.get_handle()
        self._refresh_cropboxes()

    def on_mount(self, doc):
        self._scrub_fields.mount(doc)
        self._refresh()
        self._refresh_cropboxes()
        self._sync_task_state(force=True)
        self._reactive_binding.set_handle(self._handle).watch(
            RuntimeState.scene_generation, RuntimeState.camera_selection_state,
            RuntimeState.language_generation, dirty="*")

    def on_update(self, doc):
        changed = self._refresh()
        changed |= self._refresh_cropboxes()
        changed |= self._sync_task_state(force=False)
        changed |= self._scrub_fields.sync_all()
        return changed

    def on_unmount(self, doc):
        self._reactive_binding.close()
        self._scrub_fields.unmount()
        doc.remove_data_model("camera_selection")
        self._handle = None

    def _dirty(self, *names):
        if self._handle:
            for name in names:
                self._handle.dirty(name)

    def _panel_label(self):
        return lf.ui.tr(OPERATION_LABEL_KEYS.get(self._operation, "scene.camera_selection"))

    def _refresh(self):
        operation = lf.ui.get_camera_selection_operation() or "random"
        if operation == self._operation:
            return False
        self._operation, self._local_error = operation, ""
        self._dirty("panel_label", "is_random", "is_every_n", "is_distinct", "is_cropbox", "cropboxes", "message")
        return True

    def _set_int(self, attribute, value, minimum, maximum):
        try:
            setattr(self, attribute, max(minimum, min(maximum, int(float(str(value).replace(",", "").strip())))))
            self._dirty(attribute[1:])
        except (TypeError, ValueError):
            pass

    def _get_scrub_value(self, prop):
        return float(getattr(self, f"_{prop}"))

    def _set_scrub_value(self, prop, value):
        self._set_int(f"_{prop}", value, 0, 100)

    def _set_cropbox(self, value):
        self._cropbox_id = str(value or "")
        self._dirty("cropbox_id")

    def _scene(self):
        return lf.get_scene()

    def _refresh_cropboxes(self):
        scene = self._scene()
        records = [] if scene is None else [
            {"id": str(node.id), "name": node.name}
            for node in scene.get_nodes(lf.scene.NodeType.CROPBOX)
        ]
        if records == self._cropbox_records:
            return False
        self._cropbox_records = records
        if self._handle:
            self._handle.update_record_list("cropboxes", records)
        if self._cropbox_id and not any(record["id"] == self._cropbox_id for record in records):
            self._cropbox_id = ""
            self._dirty("cropbox_id")
        if len(records) == 1 and not self._cropbox_id:
            self._cropbox_id = records[0]["id"]
            self._dirty("cropbox_id")
        return True

    def _task_state(self):
        state = _native_store_value("camera_selection_state", None)
        return state if isinstance(state, dict) else lf.camera_selection_status()

    def _sync_task_state(self, force=False):
        state = self._task_state()
        running = bool(state.get("active", False))
        error = str(state.get("error", "") or "")
        if not force and running == self._running and error == self._task_error:
            return False
        self._running, self._task_error = running, error
        self._dirty("running", "message")
        return True

    def _apply(self, *_):
        try:
            if self._operation == "cropbox_visibility" and not self._cropbox_id:
                raise RuntimeError(lf.ui.tr("scene.camera_selection_cropbox_required"))
            lf.camera_selection_start(self._operation, self._percentage, self._every_n, self._overlap,
                                      self._coverage, int(self._cropbox_id or 0))
        except Exception as error:
            self._local_error = str(error)
            self._dirty("message")
            return
        self._local_error = ""
        self._sync_task_state(force=True)

    def _cancel(self, *_):
        if self._task_state().get("active", False):
            lf.camera_selection_cancel()
            return
        lf.ui.set_panel_enabled(self.id, False)
