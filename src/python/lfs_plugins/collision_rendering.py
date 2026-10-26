# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared collision controls; all work uses the native observable job."""
from _lfs_path_io import Path
import lichtfeld as lf
from math import isclose
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import invalidate_panel

WALK_FIELDS = ("height", "radius", "step", "speed")
SPECS = {
    "size": ScrubFieldSpec(.000001, 10, .001, "%.6g"),
    "opacity": ScrubFieldSpec(0, .99, .01, "%.3f"),
    "area": ScrubFieldSpec(0, 1000000, .01, "%.4g"),
    "height": ScrubFieldSpec(.0001, 1000, .01, "%.4g"),
    "radius": ScrubFieldSpec(.00001, 500, .01, "%.4g"),
    "step": ScrubFieldSpec(0, 1000, .01, "%.4g"),
    "speed": ScrubFieldSpec(.00001, 1000, .1, "%.4g"),
}
ERRORS = {
    "No collision nodes are loaded": "collision.walk_no_collision",
    "No Walk support below the camera": "collision.walk_no_support",
    "Walk capsule does not fit at the camera": "collision.walk_blocked",
    "The edited Walk capsule does not fit": "collision.walk_blocked",
    "Walk capsule overlaps collision geometry at its current position": "collision.walk_blocked",
    "Walk support is no longer present": "collision.walk_support_lost",
    "Walk dimensions and speed must be finite and positive": "collision.walk_invalid_settings",
    "Invalid Walk capsule or step height": "collision.walk_invalid_settings",
    "Collision source is no longer available": "collision.source_unavailable",
    "Collision source changed during generation; generate it again": "collision.source_changed",
    "Collision source transform must be invertible": "collision.source_transform_invalid",
}
STAGES = {
    "Preparing collision geometry": "gallery.transfer.phase.preparing",
    "Filtering collision sources": "collision.cleanup",
    "Voxelizing collision": "collision.generate",
    "Sorting collision records": "gallery.transfer.phase.processing",
    "Assembling occupancy": "gallery.transfer.phase.processing",
    "Preparing voxel export": "runtime.export_preparing_data",
    "Packing voxel tree": "collision.export",
    "Writing voxel binary": "collision.export",
    "Publishing voxel pair": "collision.export",
    "Cancelling": "common.cancel",
}


class CollisionRendering:
    export_mode = False
    export_target = None
    popup_handle = None

    def __init__(self, popup=False):
        self.popup = popup
        self.handle = None
        self.values = dict(size=.05, opacity=.1, area=0., automatic=False, cleanup=True)
        self.path = "collision.voxel.json"
        self.error = ""
        self.state = {}
        self.scrubs = ScrubFieldController(
            {f"collision_{key}": spec for key, spec in SPECS.items()},
            lambda prop: self.values[prop.removeprefix("collision_")],
            self.set_value, commit_value=self.commit,
            on_error=self.show_error,
        )

    def mount(self, doc):
        self.update()
        self.scrubs.mount(doc)

    def show_error(self, message):
        self.error = self.localized_error(str(message))
        self.invalidate()

    @staticmethod
    def localized_error(message):
        key = ERRORS.get(message)
        return lf.ui.tr(key) if key else message

    def invalidate(self):
        invalidate_panel(self.handle, "*")

    def set_value(self, prop, value):
        self.values[prop.removeprefix("collision_")] = value
        self.invalidate()

    def commit(self, prop, value):
        key = prop.removeprefix("collision_")
        if key in WALK_FIELDS:
            def apply():
                settings = list(lf.get_collision_state()["walk_settings"])
                settings[WALK_FIELDS.index(key)] = value
                lf.set_walk_settings(*settings)
            self.run(apply)
            self.update()
            self.invalidate()

    def bind(self, model):
        self.handle = model.get_handle()
        self.update()
        for key in (*SPECS, "automatic", "cleanup"):
            model.bind(f"collision_{key}", lambda k=key: self.values[k],
                       lambda value, k=key: self.set_value(f"collision_{k}", value))
        model.bind_func("collision_resolved_size", self.resolved_size)
        model.bind_func("collision_can_generate", lambda: bool(self.state.get("source_name")) and not self.state.get("active"))
        model.bind_func("collision_source", lambda: self.state.get("source_name", ""))
        model.bind_func("collision_has_source", lambda: bool(self.state.get("source_name")))
        model.bind_func("collision_has_collision", lambda: self.state.get("has_collision", False))
        model.bind_func("collision_active", lambda: self.state.get("active", False))
        model.bind_func("collision_target", lambda: self.target()[1])
        model.bind_func("collision_exportable", lambda: bool(self.target()[0]) and not self.state.get("active", False))
        model.bind_func("collision_progress", lambda: self.state.get("progress", 0))
        model.bind_func("collision_status", lambda: f'{lf.ui.tr(STAGES.get(self.state.get("stage"), "training_panel.running"))} {self.state.get("progress", 0):.0%}')
        error = lambda: self.error or self.localized_error(self.state.get("error" if self.popup else "walk_error", ""))
        model.bind_func("collision_error", error)
        model.bind_func("collision_has_error", lambda: bool(error()))
        model.bind_event("collision_generate", lambda *_: self.run(lambda: lf.generate_collision(
            *self.generation_settings())))
        model.bind_event("collision_cancel", lambda *_: self.run(lf.cancel_collision))
        model.bind_event("collision_export", lambda *_: self.run(self.export))
        model.bind_event("collision_close", lambda *_: lf.ui.set_panel_enabled("lfs.collision", False))

    def generation_settings(self):
        return tuple(self.values[key] for key in ("size", "opacity", "automatic", "cleanup", "area"))

    def resolved_size(self):
        requested = self.generation_settings()
        captured = self.state.get("generation_settings", ())
        size = self.state.get("resolved_voxel_size", 0)
        if size > 0 and len(captured) == len(requested) and all(isclose(a, b, rel_tol=1e-6) for a, b in zip(requested, captured)):
            return f"{size:.6g}"
        return lf.ui.tr("collision.size_pending")

    def run(self, action):
        try:
            self.error = ""
            action()
        except (RuntimeError, ValueError) as exc:
            self.show_error(exc)
        self.invalidate()

    def target(self):
        return self.export_target if self.popup and self.export_mode and self.export_target else (
            self.state.get("target_uuid", ""), self.state.get("target_name", ""))

    def export(self, *_):
        uuid, _ = self.target()
        if not uuid or self.state.get("active"):
            return
        destination, confirmed = lf.ui.save_json_file_dialog(self.path, with_overwrite_confirmation=True)
        if not destination:
            return
        path = Path(destination)
        self.path = destination
        action = lambda: self.run(lambda: lf.export_collision(uuid, str(path)))
        # Native approval covers the pair; a BIN-only or post-suffix collision still needs approval.
        if not confirmed and (path.exists() or path.with_suffix(".bin").exists()):
            label = lf.ui.tr("collision.export")
            lf.ui.confirm_dialog(
                label, f'{lf.ui.tr("collision.replace_pair")}\n{path}\n{path.with_suffix(".bin")}',
                [lf.ui.tr("common.cancel"), label],
                lambda button: action() if button == label else None,
            )
        else:
            action()

    def update(self):
        state = lf.get_collision_state()
        changed = state != self.state
        self.state = state
        for key, value in zip(WALK_FIELDS, state.get("walk_settings", ())):
            if self.scrubs._active_prop != f"collision_{key}":
                self.values[key] = value
        changed |= self.scrubs.sync_all()
        if changed:
            self.invalidate()
        return changed


def open_collision(export=False):
    CollisionRendering.export_mode = export
    state = lf.get_collision_state() if export else {}
    CollisionRendering.export_target = (state.get("target_uuid", ""), state.get("target_name", ""))
    lf.ui.set_panel_enabled("lfs.collision", True)
    invalidate_panel(CollisionRendering.popup_handle, "*")
