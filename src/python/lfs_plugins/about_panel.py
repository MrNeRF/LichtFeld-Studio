# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""About panel showing application info and build details."""

import lichtfeld as lf
from .types import Panel

__lfs_panel_classes__ = ["AboutPanel"]
__lfs_panel_ids__ = ["lfs.about"]


class AboutPanel(Panel):
    """Floating panel displaying application information."""

    id = "lfs.about"
    label = "About"
    space = lf.ui.PanelSpace.FLOATING
    order = 100
    template = "rmlui/about.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (400, 0)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("about")
        if model is None:
            return

        bi = lf.build_info

        model.bind_func("panel_label", lambda: "@tr:about.title")

        model.bind_func("version", lambda: bi.version)
        model.bind_func("commit", lambda: bi.commit)
        model.bind_func("build_type", lambda: bi.build_type)
        model.bind_func("platform", lambda: bi.platform)
        model.bind_func("repo_url", lambda: bi.repo_url)
        model.bind_func("website_url", lambda: bi.website_url)

        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)

        repo_el = doc.get_element_by_id("link-repo")
        if repo_el:
            repo_el.add_event_listener("click", lambda _ev: lf.ui.open_url(lf.build_info.repo_url))

        website_el = doc.get_element_by_id("link-website")
        if website_el:
            website_el.add_event_listener("click", lambda _ev: lf.ui.open_url(lf.build_info.website_url))
