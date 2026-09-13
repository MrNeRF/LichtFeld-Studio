# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility redirect for saved layouts and older Gallery commands."""
import lichtfeld as lf
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["GalleryPanel"]
__lfs_panel_ids__ = ["lfs.gallery"]

@panel_class("gallery")
class GalleryPanel(Panel):
    def on_bind_model(self, ctx):
        model = ctx.create_data_model("gallery_redirect")
        if model:
            model.bind_func("notice", lambda: lf.ui.tr("asset_manager.gallery.info.redirect"))
            model.bind_func("panel_label", lambda: lf.ui.tr("asset_manager.gallery.sidebar.title"))

    def focus_project(self, path=None):
        lf.ui.set_panel_enabled("lfs.asset_manager", True)
        panel = lf.ui.get_panel_object("lfs.asset_manager")
        if panel is not None:
            panel.focus_gallery(path)
        lf.ui.set_panel_enabled("lfs.gallery", False)

    def on_mount(self, doc):
        super().on_mount(doc)
        self.focus_project()

    def on_update(self, doc):
        return False
