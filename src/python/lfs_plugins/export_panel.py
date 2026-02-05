# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export panel for exporting scene nodes."""

from typing import Set
from enum import IntEnum

import lichtfeld as lf
from .types import Panel


class ExportFormat(IntEnum):
    PLY = 0
    SOG = 1
    SPZ = 2
    HTML_VIEWER = 3


class ExportPanel(Panel):
    """Export panel - floating window for scene export."""

    idname = "lfs.export"
    label = "Export"
    space = "FLOATING"
    order = 10
    options = {"DEFAULT_CLOSED"}

    def __init__(self):
        self._format = ExportFormat.PLY
        self._selected_nodes: Set[str] = set()
        self._export_sh_degree = 3
        self._initialized = False

    def draw(self, layout):
        tr = lf.ui.tr
        splat_nodes = self._get_splat_nodes()

        if not self._initialized and splat_nodes:
            self._selected_nodes = {node.name for node in splat_nodes}
            self._export_sh_degree = 3
            self._initialized = True

        # Format selection
        layout.text_colored(tr("export_dialog.format"), (0.6, 0.6, 0.6, 1.0))
        layout.spacing()

        format_idx = int(self._format)
        changed, format_idx = layout.radio_button(tr("export.format.ply_standard"), format_idx, 0)
        if changed:
            self._format = ExportFormat.PLY
        changed, format_idx = layout.radio_button(tr("export.format.sog_supersplat"), format_idx, 1)
        if changed:
            self._format = ExportFormat.SOG
        changed, format_idx = layout.radio_button(tr("export.format.spz_niantic"), format_idx, 2)
        if changed:
            self._format = ExportFormat.SPZ
        changed, format_idx = layout.radio_button(tr("export.format.html_viewer"), format_idx, 3)
        if changed:
            self._format = ExportFormat.HTML_VIEWER

        layout.spacing()
        layout.spacing()

        # Model selection
        layout.text_colored(tr("export_dialog.models"), (0.6, 0.6, 0.6, 1.0))
        layout.spacing()

        if not splat_nodes:
            layout.text_colored(tr("export_dialog.no_models"), (0.6, 0.6, 0.6, 1.0))
        else:
            if layout.small_button(tr("export.all")):
                self._selected_nodes = {node.name for node in splat_nodes}
            layout.same_line()
            if layout.small_button(tr("export.none")):
                self._selected_nodes.clear()

            layout.spacing()

            for node in splat_nodes:
                selected = node.name in self._selected_nodes
                _, new_selected = layout.checkbox(node.name, selected)
                if new_selected != selected:
                    if new_selected:
                        self._selected_nodes.add(node.name)
                    else:
                        self._selected_nodes.discard(node.name)
                layout.same_line()
                layout.text_colored(f"({node.gaussian_count})", (0.6, 0.6, 0.6, 1.0))

        layout.spacing()
        layout.spacing()

        # SH Degree selection
        layout.text_colored(tr("export_dialog.sh_degree"), (0.6, 0.6, 0.6, 1.0))
        layout.spacing()
        _, self._export_sh_degree = layout.slider_int("##sh_degree", self._export_sh_degree, 0, 3)

        layout.spacing()
        layout.separator()
        layout.spacing()

        # Export button
        can_export = len(self._selected_nodes) > 0
        if not can_export:
            layout.text_colored(tr("export.select_at_least_one"), (0.9, 0.3, 0.3, 1.0))
            layout.spacing()
            layout.begin_disabled()

        label = tr("export_dialog.export_merged") if len(self._selected_nodes) > 1 else tr("export.export")
        if layout.button_styled(label, "primary", (130, 28)):
            self._do_export()

        if not can_export:
            layout.end_disabled()

        layout.same_line()
        if layout.button(tr("export.cancel"), (80, 28)):
            lf.ui.set_panel_enabled("lfs.export", False)

    def _get_splat_nodes(self):
        nodes = []
        try:
            scene = lf.get_scene()
            if scene is None:
                return nodes
            for node in scene.get_nodes():
                if node.type == lf.scene.NodeType.SPLAT and node.gaussian_count > 0:
                    nodes.append(node)
        except Exception:
            pass
        return nodes

    def _do_export(self):
        default_name = list(self._selected_nodes)[0] if self._selected_nodes else "export"
        path = None

        if self._format == ExportFormat.PLY:
            path = lf.ui.save_ply_file_dialog(f"{default_name}.ply")
        elif self._format == ExportFormat.SOG:
            path = lf.ui.save_sog_file_dialog(f"{default_name}.sog")
        elif self._format == ExportFormat.SPZ:
            path = lf.ui.save_spz_file_dialog(f"{default_name}.spz")
        elif self._format == ExportFormat.HTML_VIEWER:
            path = lf.ui.save_html_file_dialog(f"{default_name}.html")

        if path:
            lf.export_scene(int(self._format), path, list(self._selected_nodes), self._export_sh_degree)
            lf.ui.set_panel_enabled("lfs.export", False)
            self._initialized = False


