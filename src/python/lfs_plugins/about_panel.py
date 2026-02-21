# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""About panel showing application info and build details."""

import lichtfeld as lf
from .types import Panel
from .windows.layout_utils import center_content


class AboutPanel(Panel):
    """Floating panel displaying application information."""

    idname = "lfs.about"
    label = "About"
    space = "FLOATING"
    order = 100
    options = {"DEFAULT_CLOSED"}

    TITLE_COLOR = (0.3, 0.7, 1.0, 1.0)
    LABEL_COLOR = (0.6, 0.6, 0.6, 1.0)
    LINK_COLOR = (0.4, 0.8, 1.0, 1.0)

    def draw(self, layout):
        tr = lf.ui.tr

        layout.text_colored(tr("about.title"), self.TITLE_COLOR)
        layout.spacing()
        layout.separator()
        layout.spacing()

        layout.text_wrapped(tr("about.description"))
        layout.spacing()
        layout.spacing()

        layout.text_colored(tr("about.build_info"), self.LABEL_COLOR)
        layout.spacing()

        if layout.begin_table("build_info", 2):
            layout.table_setup_column(tr("about.build_info.property"), 140)
            layout.table_setup_column(tr("about.build_info.value"))

            self._table_row(layout, tr("about.build_info.version"), lf.build_info.version)
            self._table_row(layout, tr("about.build_info.commit"), lf.build_info.commit)
            self._table_row(layout, tr("about.build_info.build_type"), lf.build_info.build_type)
            self._table_row(layout, tr("about.build_info.platform"), lf.build_info.platform)
            interop_str = tr("about.interop.enabled") if lf.build_info.cuda_gl_interop else tr("about.interop.disabled")
            self._table_row(layout, tr("about.build_info.cuda_gl_interop"), interop_str)

            layout.end_table()

        layout.spacing()
        layout.spacing()

        layout.text_colored(tr("about.links"), self.LABEL_COLOR)
        layout.spacing()

        layout.label(tr("about.repository"))
        layout.same_line()
        layout.text_colored(lf.build_info.repo_url, self.LINK_COLOR)
        if layout.is_item_hovered():
            layout.set_mouse_cursor_hand()
        if layout.is_item_clicked():
            lf.ui.open_url(lf.build_info.repo_url)

        layout.label(tr("about.website"))
        layout.same_line()
        layout.text_colored(lf.build_info.website_url, self.LINK_COLOR)
        if layout.is_item_hovered():
            layout.set_mouse_cursor_hand()
        if layout.is_item_clicked():
            lf.ui.open_url(lf.build_info.website_url)

        layout.spacing()
        layout.separator()
        layout.spacing()

        separator = tr("about.separator")
        footer_text = tr("about.authors") + separator + tr("about.license")
        text_w, _ = layout.calc_text_size(footer_text)
        center_content(layout, text_w)
        layout.text_colored(tr("about.authors"), self.LABEL_COLOR)
        layout.same_line()
        layout.text_colored(separator, (0.4, 0.4, 0.4, 1.0))
        layout.same_line()
        layout.text_colored(tr("about.license"), self.LABEL_COLOR)

    def _table_row(self, layout, label: str, value: str):
        layout.table_next_row()
        layout.table_next_column()
        layout.text_colored(label, self.LABEL_COLOR)
        layout.table_next_column()
        layout.label(value)

