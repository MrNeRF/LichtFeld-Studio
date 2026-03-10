# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""About panel showing application info and build details."""

import lichtfeld as lf
from .types import RmlPanel


class AboutPanel(RmlPanel):
    """Floating panel displaying application information."""

    idname = "lfs.about"
    label = "About"
    space = "FLOATING"
    order = 100
    rml_template = "rmlui/about.rml"
    rml_height_mode = "content"
    initial_width = 400

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
        model.bind_func("cuda_gl_interop",
                         lambda: "@tr:about.interop.enabled" if bi.cuda_gl_interop
                         else "@tr:about.interop.disabled")
        model.bind_func("repo_url", lambda: bi.repo_url)
        model.bind_func("website_url", lambda: bi.website_url)

        self._handle = model.get_handle()

    def on_load(self, doc):
        super().on_load(doc)

        repo_el = doc.get_element_by_id("link-repo")
        if repo_el:
            repo_el.add_event_listener("click", lambda _ev: lf.ui.open_url(lf.build_info.repo_url))

        website_el = doc.get_element_by_id("link-website")
        if website_el:
            website_el.add_event_listener("click", lambda _ev: lf.ui.open_url(lf.build_info.website_url))
