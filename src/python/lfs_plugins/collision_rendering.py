# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared collision controls; all work uses the native observable job."""
from _lfs_path_io import Path
import lichtfeld as lf
from math import isclose
from . import rml_widgets as w
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
STAGES = {
    "Preparing collision": "gallery.transfer.phase.preparing",
    "Preparing collision geometry": "gallery.transfer.phase.preparing",
    "Filtering collision sources": "collision.cleanup",
    "Voxelizing collision": "collision.generate",
    "Sorting collision records": "gallery.transfer.phase.processing",
    "Assembling occupancy": "gallery.transfer.phase.processing",
    "Preparing voxel export": "runtime.export_preparing_data",
    "Packing voxel tree": "export.export",
    "Writing voxel binary": "export.export",
    "Publishing voxel pair": "export.export",
    "Publishing voxel metadata": "export.export",
    "Collision ready": "status.ready",
    "Cancelling": "common.cancel",
}


class CollisionRendering:
    def __init__(self, popup=False):
        self.popup = popup
        self.export_target = None
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
        for group, keys in (("size", ("size",)), ("generator", ("opacity", "area")), ("walk", ("height", "radius", "step", "speed"))):
            container = doc.get_element_by_id(f"collision-{group}-fields")
            if container is not None:
                for key in keys:
                    spec = SPECS[key]
                    localized_key = {"size": "main_panel.voxel_size", "speed": "ui.speed"}.get(key, "collision." + key)
                    row = w.slider(container, key, label=lf.ui.tr(localized_key),
                                   min=spec.min_value, max=spec.max_value, step=spec.step, value=self.values[key])
                    row.query_selector(".setting-row__label-col").set_attribute("data-lfs-i18n", localized_key)
                    row.query_selector("input").set_attribute("data-value", f"collision_{key}")
                    if group != "walk":
                        row.set_attribute("data-tooltip", "tooltip.collision." + ("voxel_size" if key == "size" else key))
        self.scrubs.mount(doc)

    def show_error(self, message):
        self.error = self.localized_error(message)
        invalidate_panel(self.handle, "*")

    @staticmethod
    def localized_error(message):
        key = getattr(message, "message_key", "")
        if not key:
            return str(message)
        details = getattr(message, "message_details", "")
        return lf.ui.tr(key) + (f"\n{details}" if details else "")

    def set_value(self, prop, value):
        self.values[prop.removeprefix("collision_")] = value
        invalidate_panel(self.handle, "*")

    def commit(self, prop, value):
        key = prop.removeprefix("collision_")
        if key in WALK_FIELDS:
            def apply():
                settings = list(lf.get_collision_state()["walk_settings"])
                settings[WALK_FIELDS.index(key)] = value
                lf.set_walk_settings(*settings)
            self.run(apply)
            self.update()

    def bind(self, model):
        self.handle = model.get_handle()
        self.update()
        model.bind_func("collision_has_collision", lambda: self.state.get("has_collision", False))
        error = lambda: self.error or self.localized_error(self.state.get("error" if self.popup else "walk_error", ""))
        model.bind_func("collision_error", error)
        model.bind_func("collision_has_error", lambda: bool(error()))
        if not self.popup:
            return
        for key in ("automatic", "cleanup"):
            model.bind(f"collision_{key}", lambda k=key: self.values[k],
                       lambda value, k=key: self.set_value(f"collision_{k}", value))
        model.bind_func("collision_resolved_size", self.resolved_size)
        model.bind_func("collision_can_generate", lambda: bool(self.state.get("source_name")) and not self.state.get("active"))
        model.bind_func("collision_source", lambda: self.state.get("source_name", ""))
        model.bind_func("collision_active", lambda: self.state.get("active", False))
        model.bind_func("collision_progress", lambda: self.state.get("progress", 0))
        model.bind_func("collision_has_source", lambda: bool(self.state.get("source_name")))
        model.bind_func("collision_target", lambda: self.target()[1])
        model.bind_func("collision_exportable", lambda: bool(self.target()[0]) and not self.state.get("active", False))
        model.bind_func("collision_status", lambda: f'{lf.ui.tr(STAGES.get(self.state.get("stage"), "training_panel.running"))} {self.state.get("progress", 0):.0%}')
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
        invalidate_panel(self.handle, "*")

    def target(self):
        return self.export_target if self.export_target is not None else (
            self.state.get("target_uuid", ""), self.state.get("target_name", ""))

    def export(self, *_):
        uuid, _ = self.target()
        if not uuid or self.state.get("active"):
            return
        destination, confirmed = lf.ui.save_json_file_dialog_with_confirmation(self.path)
        if not destination:
            return
        path = Path(destination)
        self.path = destination
        action = lambda: self.run(lambda: lf.export_collision(uuid, str(path)))
        # Native approval covers the pair; a BIN-only or post-suffix collision still needs approval.
        if not confirmed and (path.exists() or path.with_suffix(".bin").exists()):
            label = lf.ui.tr("export.export")
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
            invalidate_panel(self.handle, "*")
        return changed


def open_collision(export=False):
    panel = lf.ui.get_panel_object("lfs.collision")
    if panel is None:
        raise RuntimeError("Collision panel is not registered")
    panel.show(export)
