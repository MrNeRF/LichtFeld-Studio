# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contextual viewport controls; native occupancy and meshes stay in place."""

import lichtfeld as lf

from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import RuntimeState


_SPECS = {
    "collision_walk_height": ScrubFieldSpec(.05, 3, .01, "%.2f m"),
    "collision_walk_radius": ScrubFieldSpec(.01, .9, .005, "%.3f m"),
    "collision_walk_speed": ScrubFieldSpec(.01, 10, .025, "%.2f m/s"),
    "collision_walk_step": ScrubFieldSpec(0, 1, .005, "%.3f m"),
    "collision_preview_opacity": ScrubFieldSpec(0, 100, 1, "%.0f%%"),
    "collision_preview_cutaway": ScrubFieldSpec(0, 100, .1, "%.1f%%"),
}


class CollisionControlsController:
    def __init__(self):
        self._handle = None
        self._state = {}
        self._context = None
        self._signature = None
        self._error = ""
        self._targets = []
        self.visible = False
        self._scrubs = ScrubFieldController(_SPECS, self._get_value, self._set_value)

    def bind_model(self, model):
        model.bind_func("collision_controls_walk", lambda: bool(self._state.get("walking")))
        model.bind_func("collision_controls_editable", lambda: not self._state.get("target", {}).get("locked", False))
        model.bind_func("collision_controls_error", lambda: self._error)
        model.bind_func("collision_controls_status", self._status)
        model.bind("collision_controls_target", lambda: (self._state.get("target") or {}).get("uuid", ""), self._select_target)
        model.bind_record_list("collision_controls_targets")
        for prop in _SPECS:
            model.bind(prop, lambda p=prop: str(self._get_value(p)), lambda value, p=prop: self._set_value(p, float(value)))
        self._handle = model.get_handle()

    def mount(self, doc):
        self._context = None
        self._signature = None
        self._targets = []
        self.visible = False
        self._scrubs.mount(doc)

    def unmount(self):
        self._scrubs.unmount()
        self._handle = None
        self.visible = False
        self._context = None
        self._signature = None

    def update(self, doc):
        getter = getattr(lf, "get_collision_controls", None)
        state = getter() if callable(getter) else {}
        # Normalize optional target once; the rest of the card can use a mapping.
        state["target"] = state.get("target") or {}
        context = "walk" if state.get("walking") else state["target"].get("uuid")
        if context != self._context:
            # Cancel drafts before changing their target. A draft must never be
            # committed to a newly selected collider or a different camera mode.
            for prop in _SPECS:
                self._scrubs.prepare_reset(prop)
            self._context = context
            self._error = ""
        self._state = state
        self.visible = bool(context)
        wrap = doc.get_element_by_id("collision-controls-block")
        if wrap:
            wrap.set_class("hidden", not self.visible)
        signature = (context, repr(state), self._error, RuntimeState.language_generation.value)
        if signature == self._signature:
            return None
        self._signature = signature
        targets = [{"uuid": t["uuid"], "name": t["name"]} for t in state.get("targets", [])]
        if targets != self._targets:
            self._targets = targets
            if self._handle:
                self._handle.update_record_list("collision_controls_targets", targets)
        height = float(state.get("height", 1.8))
        radius = float(state.get("radius", .2))
        self._scrubs.set_spec("collision_walk_height", ScrubFieldSpec(max(.05, 2 * radius), max(3, height), .01, "%.2f m"))
        self._scrubs.set_spec("collision_walk_radius", ScrubFieldSpec(.01, height / 2, .005, "%.3f m"))
        self._scrubs.set_spec("collision_walk_step", ScrubFieldSpec(0, height, .005, "%.3f m"))
        self._scrubs.sync_all()
        if self._handle:
            for field in (*_SPECS, "collision_controls_walk", "collision_controls_editable", "collision_controls_target", "collision_controls_error", "collision_controls_status"):
                self._handle.dirty(field)
        return "context"

    def _get_value(self, prop):
        if prop.startswith("collision_walk_"):
            key = prop.removeprefix("collision_walk_")
            return float(self._state.get(key, {"height": 1.8, "radius": .2, "speed": 2.5, "step": .25}[key]))
        key = prop.removeprefix("collision_preview_")
        return float((self._state.get("target") or {}).get(key, 1)) * 100

    def _set_value(self, prop, value):
        try:
            if prop.startswith("collision_walk_"):
                current = lf.get_collision_controls()
                values = {key: current[key] for key in ("height", "radius", "speed", "step")}
                values[prop.removeprefix("collision_walk_")] = value
                lf.set_walk_settings(**values)
                self._state.update(values)
            else:
                target = self._state.get("target") or {}
                if not target or target.get("locked"):
                    raise RuntimeError(lf.ui.tr("collision.controls_locked"))
                current = lf.get_collision_controls()
                target = next((t for t in current.get("targets", []) if t["uuid"] == target["uuid"]), None)
                if not target:
                    raise RuntimeError(lf.ui.tr("collision.export_select"))
                key = prop.removeprefix("collision_preview_")
                values = {"opacity": target["opacity"], "cutaway": target["cutaway"]}
                values[key] = value / 100
                lf.set_collision_preview(target["uuid"], **values)
                self._state["target"].update(values)
            self._error = ""
        except (RuntimeError, TypeError, ValueError) as exc:
            # Native validation leaves the actual body/display unchanged.
            self._error = str(exc)
        self._signature = None
        if self._handle:
            self._handle.dirty("collision_controls_error")

    def _select_target(self, uuid):
        # Resolve the UUID against current state, rather than a stale display name.
        state = lf.get_collision_controls()
        target = next((t for t in state.get("targets", []) if t["uuid"] == uuid), None)
        if target:
            lf.select_node(target["name"])

    def _status(self):
        target = self._state.get("target") or {}
        if self._state.get("walking"):
            return ""
        if target.get("locked"):
            return lf.ui.tr("collision.controls_locked")
        if not target.get("visible", True):
            return lf.ui.tr("collision.controls_hidden")
        return ""
