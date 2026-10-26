# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generator and export views hosted by the standard floating panel."""
import lichtfeld as lf
from .collision_rendering import CollisionRendering
from .panels import panel_class
from .types import Panel
from .ui import RuntimeState, PanelStateBinding, invalidate_panel

__lfs_panel_classes__ = ["CollisionPanel"]
__lfs_panel_ids__ = ["lfs.collision"]


@panel_class("collision")
class CollisionPanel(Panel):
    def __init__(self):
        self.controls = CollisionRendering(popup=True)
        self.binding = PanelStateBinding()

    def show(self, export=False):
        state = lf.get_collision_state() if export else None
        self.controls.export_target = None if state is None else (
            state.get("target_uuid", ""), state.get("target_name", ""))
        invalidate_panel(self.controls.handle, "*")
        lf.ui.set_panel_enabled(self.id, True)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("collision")
        model.bind_func("panel_label", lambda: lf.ui.tr(
            "export.export" if self.controls.export_target is not None else "collision.title"))
        model.bind_func("export_mode", lambda: self.controls.export_target is not None)
        self.controls.bind(model)

    def on_mount(self, doc):
        super().on_mount(doc)
        self.controls.mount(doc)
        self.binding.set_handle(self.controls.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.collision_state, RuntimeState.viewport_toolbar_generation,
            RuntimeState.language_generation, dirty="*")

    def on_update(self, doc):
        return self.controls.update()

    def on_unmount(self, doc):
        self.binding.close()
        self.controls.scrubs.unmount()
        self.controls.handle = None
        doc.remove_data_model("collision")
