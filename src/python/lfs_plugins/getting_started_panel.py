# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Getting Started panel with tutorial videos and documentation links."""

import os
import threading
from urllib.parse import parse_qs, quote, urlparse

import lichtfeld as lf
from .http import urlopen
from .types import Panel

__lfs_panel_classes__ = ["GettingStartedPanel"]
__lfs_panel_ids__ = ["lfs.getting_started"]

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lichtfeld-studio", "thumbnails")
_RML_PATH_SAFE_CHARS = "/:._-~"


def _encode_rml_path(path):
    return quote(str(path), safe=_RML_PATH_SAFE_CHARS)


def _extract_video_id(url):
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/").split("/", 1)[0] or None

    if host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in ("embed", "shorts"):
            return parts[1]

    return None


def _download_thumbnail(video_id, on_done):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{video_id}.jpg")
    if os.path.exists(path):
        on_done(video_id, path)
        return
    try:
        data = urlopen(f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg", timeout=5).read()
        with open(path, "wb") as f:
            f.write(data)
        on_done(video_id, path)
    except Exception:
        pass


class GettingStartedPanel(Panel):
    """Floating panel displaying tutorial videos and documentation."""

    id = "lfs.getting_started"
    label = "Getting Started"
    space = lf.ui.PanelSpace.FLOATING
    order = 99
    template = "rmlui/getting_started.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (560, 0)
    update_interval_ms = 100

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("getting_started")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("getting_started.title"))
        model.bind_event("open_url", self._on_open_url)

    def on_mount(self, doc):
        super().on_mount(doc)

        self._ready_lock = threading.Lock()
        self._ready_queue = []
        self._thumb_card_map = {}

        for card in doc.query_selector_all(".video-card"):
            url = card.get_attribute("data-url", "").strip()
            if not url:
                continue

            vid = _extract_video_id(url)
            elem_id = card.get_attribute("id", "") or card.id()
            if vid and elem_id:
                self._thumb_card_map[vid] = elem_id
                threading.Thread(target=_download_thumbnail,
                                 args=(vid, self._on_thumb_ready),
                                 daemon=True).start()

    def _on_open_url(self, _handle, event, _args):
        target = event.current_target()
        if target is None:
            return

        url = target.get_attribute("data-url", "").strip()
        if url:
            lf.ui.open_url(url)

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
                body.set_property("decorator", f"image({_encode_rml_path(path)})")
