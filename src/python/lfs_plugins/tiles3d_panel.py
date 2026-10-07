# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Debug window for view-dependent 3D Tiles streaming."""

import lichtfeld as lf
from .types import Panel
from .panels import panel_class

__lfs_panel_classes__ = ["Tiles3dPanel"]
__lfs_panel_ids__ = ["lfs.tiles3d"]


def _tr(key: str, **values) -> str:
    text = lf.ui.tr(key)
    return text.format(**values) if values else text


def _count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _gib(value: int) -> str:
    return f"{value / (1 << 30):.2f}"


@panel_class("tiles3d")
class Tiles3dPanel(Panel):
    """On-demand streaming controls and statistics for a 3D Tiles tileset."""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("tiles3d")
        if model is None:
            return
        model.bind_func("panel_label", lambda: "@tr:tiles3d.title")
        self._handle = model.get_handle()

    def poll(self, _context):
        return lf.get_tiles_mode() is not None

    def draw(self, ui):
        mode = lf.get_tiles_mode()
        if mode is None:
            return
        mode_label = _tr("tiles3d.mode_stream") if mode == "stream" else _tr("tiles3d.mode_flat")
        ui.label(_tr("tiles3d.mode", mode=mode_label))

        settings = lf.get_tiles_settings()
        stats = lf.get_tiles_stats()
        if not settings or stats is None:
            # Loaded flat (no streaming): the mode line above is all there is to show.
            ui.text_disabled(_tr("tiles3d.flat_help"))
            return

        ui.separator()
        changed, value = ui.slider_float(_tr("tiles3d.cache_fraction"), settings["cache_fraction"], 0.0, 1.0)
        if changed:
            lf.set_tiles_settings(cache_fraction=value)
        ui.text_disabled(_tr("tiles3d.cache_fraction_help"))

        changed, value = ui.slider_float(_tr("tiles3d.max_sse"), settings["max_sse"], 1.0, 64.0)
        if changed:
            lf.set_tiles_settings(max_sse=value)
        ui.text_disabled(_tr("tiles3d.max_sse_help"))

        changed, value = ui.slider_int(_tr("tiles3d.num_load_workers"), settings["num_load_workers"], 0, 16)
        if changed:
            lf.set_tiles_settings(num_load_workers=value)
        ui.text_disabled(_tr("tiles3d.num_load_workers_help"))

        changed, value = ui.checkbox(_tr("tiles3d.cull"), settings["cull"])
        if changed:
            lf.set_tiles_settings(cull=value)
        ui.text_disabled(_tr("tiles3d.cull_help"))

        changed, value = ui.checkbox(_tr("tiles3d.freeze"), settings["freeze"])
        if changed:
            lf.set_tiles_settings(freeze=value)
        ui.text_disabled(_tr("tiles3d.freeze_help"))

        ui.separator()
        ui.heading(_tr("tiles3d.statistics"))
        ui.label(_tr("tiles3d.stats_tiles", drawn=stats["drawn_tiles"], cached=stats["cached_tiles"],
                     loading=stats["loading_tiles"], failed=stats["failed_tiles"], total=stats["tiles"]))
        if stats.get("skipped_contents", 0):
            ui.text_disabled(_tr("tiles3d.stats_skipped", count=stats["skipped_contents"]))
        ui.label(_tr("tiles3d.stats_splats", drawn=_count(stats["drawn_splats"]),
                     full=_count(stats["full_detail_splats"])))
        ui.label(_tr("tiles3d.stats_memory", cache=_gib(stats["cache_bytes"]), drawn=_gib(stats["drawn_bytes"]),
                     limit=_gib(stats["cache_limit_bytes"]), total=_gib(stats["gpu_total_bytes"])))
        ui.label(_tr("tiles3d.stats_sse", sse=f"{stats['max_sse']:.1f}"))
        ui.label(_tr("tiles3d.stats_workers", workers=stats["load_workers"]))
        ui.label(_tr("tiles3d.stats_build", ms=f"{stats['build_ms']:.0f}"))
