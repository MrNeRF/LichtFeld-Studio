# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Getting Started panel with tutorial videos and documentation links."""

import os
import threading

import lichtfeld as lf
from .types import RmlPanel

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lichtfeld-studio", "thumbnails")


def _extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None


def _download_thumbnail(video_id, on_done):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{video_id}.jpg")
    if os.path.exists(path):
        on_done(video_id, path)
        return
    try:
        from urllib.request import urlopen
        data = urlopen(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", timeout=5).read()
        with open(path, "wb") as f:
            f.write(data)
        on_done(video_id, path)
    except Exception:
        pass


class GettingStartedPanel(RmlPanel):
    """Floating panel displaying tutorial videos and documentation."""

    idname = "lfs.getting_started"
    label = "Getting Started"
    space = "FLOATING"
    order = 99
    rml_template = "rmlui/getting_started.rml"
    rml_height_mode = "content"
    initial_width = 560

    WIKI_URL = "https://github.com/MrNeRF/LichtFeld-Studio/wiki"

    VIDEO_CARDS = [
        ("card-intro", "getting_started.video_intro", "https://www.youtube.com/watch?v=b1Olu_IU1sM"),
        ("card-latest", "getting_started.video_latest", "https://www.youtube.com/watch?v=zWIzBHRc-60"),
        ("card-masks", "getting_started.video_masks", "https://www.youtube.com/watch?v=956qR8N3Xk4"),
        ("card-reality-scan", "getting_started.video_reality_scan", "https://www.youtube.com/watch?v=JWmkhTlbDvg"),
        ("card-colmap", "getting_started.video_colmap", "https://www.youtube.com/watch?v=-3TBbukYN00"),
        ("card-lichtfeld", "getting_started.video_lichtfeld", "https://www.youtube.com/watch?v=aX8MTlr9Ypc"),
    ]

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("getting_started")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("getting_started.title"))
        model.bind_func("title", lambda: tr("getting_started.title"))
        model.bind_func("description", lambda: tr("getting_started.description"))
        model.bind_func("wiki_section", lambda: tr("getting_started.wiki_section"))

        for _elem_id, title_key, _url in self.VIDEO_CARDS:
            binding_name = title_key.split(".")[-1]
            model.bind_func(binding_name, lambda k=title_key: tr(k))

        self._handle = model.get_handle()

    def on_load(self, doc):
        super().on_load(doc)

        self._ready_lock = threading.Lock()
        self._ready_queue = []
        self._thumb_card_map = {}

        for elem_id, _title_key, url in self.VIDEO_CARDS:
            el = doc.get_element_by_id(elem_id)
            if el:
                el.add_event_listener("click", lambda _ev, u=url: lf.ui.open_url(u))

            vid = _extract_video_id(url)
            if vid:
                self._thumb_card_map[vid] = elem_id
                threading.Thread(target=_download_thumbnail,
                                 args=(vid, self._on_thumb_ready),
                                 daemon=True).start()

        wiki_section = doc.get_element_by_id("wiki-section")
        if wiki_section:
            wiki_section.add_event_listener("click", lambda _ev: lf.ui.open_url(self.WIKI_URL))

    def _on_thumb_ready(self, video_id, path):
        with self._ready_lock:
            self._ready_queue.append((video_id, path))

    def on_update(self, doc):
        if not hasattr(self, "_ready_lock"):
            return

        with self._ready_lock:
            batch = list(self._ready_queue)
            self._ready_queue.clear()

        for video_id, path in batch:
            elem_id = self._thumb_card_map.get(video_id)
            if not elem_id:
                continue
            card = doc.get_element_by_id(elem_id)
            if not card:
                continue
            body = card.query_selector(".card-body")
            if body:
                body.set_property("decorator", f"image({path})")
