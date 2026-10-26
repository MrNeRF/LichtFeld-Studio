# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared collision controls; all work uses the native observable job."""
from _lfs_path_io import Path
import lichtfeld as lf
from math import isclose
from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import invalidate_panel, RuntimeState, PanelStateBinding
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["CollisionPanel"]
__lfs_panel_ids__ = ["lfs.collision"]

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


@panel_class("collision")
class CollisionPanel(Panel):
    def __init__(self):
        self.export_target = None
        self.handle = None
        self.values = dict(size=.05, opacity=.1, area=0., automatic=False, cleanup=True, error="")
        self.path = "collision.voxel.json"
        self.state = {}
        self.scrubs = ScrubFieldController(
            SPECS, self.values.__getitem__, self.set_value, commit_value=self.commit,
            on_error=lambda error: self.set_value("error", self.localized_error(error)),
        )

    def mount(self, doc):
        self.update()
        for group, keys in (("size", ("size",)), ("generator", ("opacity", "area")), ("walk", WALK_FIELDS)):
            container = doc.get_element_by_id(f"collision-{group}-fields")
            if container is not None:
                for key in keys:
                    spec = SPECS[key]
                    localized_key = {"size": "main_panel.voxel_size", "speed": "ui.speed"}.get(key, "collision." + key)
                    row = w.slider(container, key, label=lf.ui.tr(localized_key),
                                   min=spec.min_value, max=spec.max_value, step=spec.step, value=self.values[key])
                    row.query_selector(".setting-row__label-col").set_attribute("data-lfs-i18n", localized_key)
                    row.query_selector("input").set_attribute("data-value", key)
                    if group != "walk":
                        row.set_attribute("data-tooltip", "tooltip.collision." + ("voxel_size" if key == "size" else key))
        self.scrubs.mount(doc)

    def show(self, export=False):
        state = lf.get_collision_state() if export else None
        self.export_target = None if state is None else (
            state.get("target_uuid", ""), state.get("target_name", ""))
        invalidate_panel(self.handle, "*")
        lf.ui.set_panel_enabled("lfs.collision", True)

    @staticmethod
    def localized_error(message):
        key = getattr(message, "message_key", "")
        if not key:
            return str(message)
        details = getattr(message, "message_details", "")
        return lf.ui.tr(key) + (f"\n{details}" if details else "")

    def set_value(self, prop, value):
        self.values[prop] = value
        invalidate_panel(self.handle, "*")

    def commit(self, key, value):
        if key in WALK_FIELDS:
            self.run(lambda: lf.set_walk_settings(*(
                value if field == key else setting for field, setting in
                zip(WALK_FIELDS, lf.get_collision_state()["walk_settings"]))))
            self.update()

    def bind(self, model, popup=False):
        self.handle = model.get_handle()
        self.update()
        for key, field, default in (("collision_has_collision", "has_collision", False), ("collision_source", "source_name", ""), ("collision_active", "active", False), ("collision_progress", "progress", 0)):
            model.bind_func(key, lambda k=field, d=default: self.state.get(k, d))
        model.bind_func("collision_error", lambda: self.values["error"] or self.localized_error(self.state.get("error" if popup else "walk_error", "")))
        if not popup:
            return
        model.bind_func("panel_label", lambda: lf.ui.tr(
            "export.export" if self.export_target is not None else "collision.title"))
        model.bind_func("export_mode", lambda: self.export_target is not None)
        for key in ("automatic", "cleanup"):
            model.bind(f"collision_{key}", lambda k=key: self.values[k],
                       lambda value, k=key: self.set_value(k, value))
        model.bind_func("collision_resolved_size", self.resolved_size)
        model.bind_func("collision_target", lambda: self.export_target[1] if self.export_target else "")
        model.bind_func("collision_exportable", lambda: bool(self.export_target and self.export_target[0]) and not self.state.get("active", False))
        model.bind_func("collision_status", lambda: f'{self.state.get("stage") or lf.ui.tr("training_panel.running")} {self.state.get("progress", 0):.0%}')
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
        return lf.ui.tr("projects.gallery.review.estimate_pending")

    def run(self, action):
        try:
            self.values["error"] = ""
            action()
        except (RuntimeError, ValueError) as exc:
            self.values["error"] = self.localized_error(exc)
        invalidate_panel(self.handle, "*")

    def export(self, *_):
        uuid, _ = self.export_target or ("", "")
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

    def update(self, doc=None):
        state = lf.get_collision_state()
        changed = state != self.state
        self.state = state
        for key, value in zip(WALK_FIELDS, state.get("walk_settings", ())):
            if self.scrubs._active_prop != key:
                self.values[key] = value
        return self.scrubs.sync_all() or changed

    def on_bind_model(self, ctx):
        self.bind(ctx.create_data_model("collision"), popup=True)

    def on_mount(self, doc):
        super().on_mount(doc)
        self.mount(doc)
        self.binding = PanelStateBinding(self.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.viewport_toolbar_generation,
            refresh=self.update, dirty="*").watch(RuntimeState.language_generation, dirty="*")

    on_update = update

    def on_unmount(self, doc):
        self.binding.close()
        self.scrubs.unmount()
        self.handle = None
        doc.remove_data_model("collision")
