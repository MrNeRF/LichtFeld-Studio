# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Getting Started panel with tutorial videos and documentation links."""

import lichtfeld as lf
from .types import Panel


class GettingStartedPanel(Panel):
    """Floating panel displaying tutorial videos and documentation."""

    idname = "lfs.getting_started"
    label = "Getting Started"
    space = "FLOATING"
    order = 99
    options = {"DEFAULT_CLOSED"}

    TITLE_COLOR = (0.3, 0.7, 1.0, 1.0)
    LINK_COLOR = (0.4, 0.8, 1.0, 1.0)
    LABEL_COLOR = (0.6, 0.6, 0.6, 1.0)

    CARD_WIDTH = 160
    CARD_HEIGHT = 90
    CARD_SPACING = 16
    CARDS_PER_ROW = 3

    VIDEO_CARDS = [
        ("getting_started.video_intro", "b1Olu_IU1sM", "https://www.youtube.com/watch?v=b1Olu_IU1sM"),
        ("getting_started.video_latest", "zWIzBHRc-60", "https://www.youtube.com/watch?v=zWIzBHRc-60"),
        ("getting_started.video_masks", "956qR8N3Xk4", "https://www.youtube.com/watch?v=956qR8N3Xk4"),
        ("getting_started.video_reality_scan", "JWmkhTlbDvg", "https://www.youtube.com/watch?v=JWmkhTlbDvg"),
        ("getting_started.video_colmap", "-3TBbukYN00", "https://www.youtube.com/watch?v=-3TBbukYN00"),
        ("getting_started.video_lichtfeld", "aX8MTlr9Ypc", "https://www.youtube.com/watch?v=aX8MTlr9Ypc"),
    ]

    WIKI_URL = "https://github.com/MrNeRF/LichtFeld-Studio/wiki"

    def __init__(self):
        self._thumbnails_requested = set()

    def draw(self, layout):
        tr = lf.ui.tr

        lf.ui.process_thumbnails()

        scale = layout.get_dpi_scale()
        card_w = self.CARD_WIDTH * scale
        card_h = self.CARD_HEIGHT * scale
        spacing = self.CARD_SPACING * scale

        layout.text_colored(tr("getting_started.title"), self.TITLE_COLOR)
        layout.spacing()
        layout.separator()
        layout.spacing()

        layout.text_wrapped(tr("getting_started.description"))
        layout.spacing()
        layout.spacing()

        for i, (title_key, video_id, url) in enumerate(self.VIDEO_CARDS):
            self._draw_video_card(layout, tr, title_key, video_id, url, card_w, card_h)

            if (i % self.CARDS_PER_ROW) < (self.CARDS_PER_ROW - 1):
                layout.same_line(spacing=spacing)

        layout.spacing()
        layout.spacing()
        layout.separator()
        layout.spacing()

        layout.text_colored(tr("getting_started.wiki_section"), self.LABEL_COLOR)
        layout.spacing()

        layout.label(tr("getting_started.wiki_label"))
        layout.same_line()
        layout.text_colored(self.WIKI_URL, self.LINK_COLOR)
        if layout.is_item_hovered():
            layout.set_mouse_cursor_hand()
        if layout.is_item_clicked():
            lf.ui.open_url(self.WIKI_URL)

    def _draw_video_card(self, layout, tr, title_key: str, video_id: str, url: str, width: float, height: float):
        title = tr(title_key)

        if video_id not in self._thumbnails_requested:
            lf.ui.request_thumbnail(video_id)
            self._thumbnails_requested.add(video_id)

        layout.begin_group()

        if lf.ui.is_thumbnail_ready(video_id):
            texture_id = lf.ui.get_thumbnail_texture(video_id)
            if texture_id > 0:
                if layout.image_button(f"thumb_{video_id}", texture_id, (width, height)):
                    lf.ui.open_url(url)
            else:
                if layout.button(f"##btn_{video_id}", (width, height)):
                    lf.ui.open_url(url)
        else:
            if layout.button(tr("getting_started.loading") + f"##btn_{video_id}", (width, height)):
                lf.ui.open_url(url)

        if layout.is_item_hovered():
            layout.set_mouse_cursor_hand()
            layout.set_tooltip(tr("getting_started.watch_tooltip").format(title=title))

        layout.text_colored(title, self.LABEL_COLOR)

        layout.end_group()

