# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generator and export views hosted by the standard floating panel."""
from .collision_rendering import CollisionRendering
from .panels import panel_class
from .types import Panel
from .ui import RuntimeState, PanelStateBinding

__lfs_panel_classes__ = ["CollisionPanel"]
__lfs_panel_ids__ = ["lfs.collision"]


@panel_class("collision")
class CollisionPanel(CollisionRendering, Panel):
    def on_bind_model(self, ctx):
        self.bind(ctx.create_data_model("collision"), popup=True)

    def on_mount(self, doc):
        super().on_mount(doc)
        self.mount(doc)
        self.binding = PanelStateBinding(self.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.viewport_toolbar_generation,
            refresh=self.update, dirty="*").watch(RuntimeState.language_generation, dirty="*")

    on_update = CollisionRendering.update

    def on_unmount(self, doc):
        self.binding.close()
        self.scrubs.unmount()
        self.handle = None
        doc.remove_data_model("collision")
