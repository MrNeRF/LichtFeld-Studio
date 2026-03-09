# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export panel for exporting scene nodes."""

import html
from typing import Set
from enum import IntEnum

import lichtfeld as lf
from .types import RmlPanel


class ExportFormat(IntEnum):
    PLY = 0
    SOG = 1
    SPZ = 2
    HTML_VIEWER = 3


FORMAT_INFO = (
    (ExportFormat.PLY, "export.format.ply_standard"),
    (ExportFormat.SOG, "export.format.sog_supersplat"),
    (ExportFormat.SPZ, "export.format.spz_niantic"),
    (ExportFormat.HTML_VIEWER, "export.format.html_viewer"),
)


def _xml_unescape(text):
    return html.unescape(text or "")


class ExportPanel(RmlPanel):
    idname = "lfs.export"
    label = "Export"
    space = "FLOATING"
    order = 10
    rml_template = "rmlui/export_panel.rml"
    rml_height_mode = "content"
    initial_width = 320
    update_interval_ms = 100

    def __init__(self):
        self._format = ExportFormat.PLY
        self._selected_nodes: Set[str] = set()
        self._export_sh_degree = 3
        self._selection_seeded = False
        self._handle = None
        self._last_node_key = None
        self._last_lang = ""
        self._exporting = False
        self._last_progress = -1.0
        self._progress_value = "0"
        self._has_models = False
        self._cached_export_state = {}

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("export")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("export.export"))
        model.bind_func("format_label", lambda: tr("export_dialog.format"))
        model.bind_func("models_label", lambda: tr("export_dialog.models"))
        model.bind_func("sh_degree_label", lambda: tr("export_dialog.sh_degree"))
        model.bind_func("all_label", lambda: tr("export.all"))
        model.bind_func("none_label", lambda: tr("export.none"))
        model.bind_func("export_label", self._get_export_label)
        model.bind_func("cancel_label", lambda: tr("export.cancel"))
        model.bind_func("select_at_least_one", lambda: tr("export.select_at_least_one"))
        model.bind_func("no_models_label", lambda: tr("export_dialog.no_models"))
        model.bind_func("show_no_models", lambda: not self._has_models)
        model.bind_func("can_export", lambda: bool(self._selected_nodes))
        model.bind_func("progress_value", lambda: self._progress_value)

        model.bind(
            "sh_degree",
            lambda: str(self._export_sh_degree),
            self._set_sh_degree,
        )

        model.bind_func("show_form", lambda: not self._exporting)
        model.bind_func("show_progress", lambda: self._exporting)
        model.bind_func("progress_title", self._get_progress_title)
        model.bind_func("progress_pct", self._get_progress_pct)
        model.bind_func("progress_stage", self._get_progress_stage)

        model.bind_event("do_export", self._on_export)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("do_cancel_export", self._on_cancel_export)
        model.bind_record_list("formats")
        model.bind_record_list("models")

        self._handle = model.get_handle()

    def _set_sh_degree(self, v):
        try:
            degree = max(0, min(3, int(float(v))))
        except (ValueError, TypeError):
            return

        if degree == self._export_sh_degree:
            return

        self._export_sh_degree = degree
        self._dirty_model("sh_degree")

    # ── Lifecycle ─────────────────────────────────────────────

    def on_load(self, doc):
        super().on_load(doc)
        self._exporting = False
        self._last_progress = -1.0
        self._cached_export_state = {}
        self._selection_seeded = False
        self._last_node_key = None
        self._last_lang = lf.ui.get_current_language()

        format_list = doc.get_element_by_id("format-list")
        if format_list:
            format_list.add_event_listener("click", self._on_format_click)

        btn_all = doc.get_element_by_id("btn-select-all")
        if btn_all:
            btn_all.add_event_listener("click", self._on_select_all)

        btn_none = doc.get_element_by_id("btn-select-none")
        if btn_none:
            btn_none.add_event_listener("click", self._on_select_none)

        model_list = doc.get_element_by_id("model-list")
        if model_list:
            model_list.add_event_listener("change", self._on_model_toggle)
            model_list.add_event_listener("click", self._on_model_toggle)

        self._rebuild_format_records()
        self._rebuild_model_records(self._get_splat_nodes())

    def on_update(self, doc):
        if self._exporting:
            return self._update_export_progress()

        if self._last_progress >= 0.0:
            self._last_progress = -1.0
            self._progress_value = "0"
            self._dirty_model("show_form", "show_progress")
            return True

        dirty = False
        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._dirty_model()
            self._rebuild_format_records()
            self._last_node_key = None
            dirty = True

        nodes = self._get_splat_nodes()
        node_key = tuple((n.name, n.gaussian_count) for n in nodes)

        if self._sync_selection(nodes):
            self._rebuild_model_records(nodes)
            self._dirty_model("export_label", "can_export")
            dirty = True

        if node_key != self._last_node_key:
            self._last_node_key = node_key
            self._rebuild_model_records(nodes)
            self._dirty_model("show_no_models", "can_export")
            dirty = True

        return dirty

    def on_scene_changed(self, doc):
        self._last_node_key = None

    # ── Helpers ──────────────────────────────────────────────

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _get_export_label(self):
        tr = lf.ui.tr
        if len(self._selected_nodes) > 1:
            return tr("export_dialog.export_merged")
        return tr("export.export")

    def _sync_selection(self, nodes):
        node_names = {node.name for node in nodes}

        if not node_names:
            changed = bool(self._selected_nodes) or self._selection_seeded
            self._selected_nodes.clear()
            self._selection_seeded = False
            return changed

        if not self._selection_seeded:
            self._selected_nodes = node_names
            self._export_sh_degree = 3
            self._selection_seeded = True
            self._dirty_model("sh_degree")
            return True

        selected_nodes = self._selected_nodes & node_names
        if selected_nodes != self._selected_nodes:
            self._selected_nodes = selected_nodes
            return True

        return False

    def _find_ancestor_with_attribute(self, element, attribute, stop=None):
        while element is not None and element != stop:
            if element.has_attribute(attribute):
                return element
            element = element.parent()
        return None

    def _get_checkbox_from_event(self, event):
        container = event.current_target()
        target = self._find_ancestor_with_attribute(event.target(), "data-node-name", container)
        if target is None:
            return None, None

        checkbox = target
        if checkbox.tag_name != "input" or checkbox.get_attribute("type", "") != "checkbox":
            checkbox = target.query_selector('input[type="checkbox"]')
        if checkbox is None:
            return None, None

        node_name = _xml_unescape(checkbox.get_attribute("data-node-name", ""))
        if not node_name:
            return None, None

        return checkbox, node_name

    # ── Retained model updates ────────────────────────────────

    def _rebuild_format_records(self):
        if not self._handle:
            return
        tr = lf.ui.tr
        self._handle.update_record_list(
            "formats",
            [
                {
                    "index": str(int(fmt)),
                    "label": tr(key),
                    "selected": fmt == self._format,
                }
                for fmt, key in FORMAT_INFO
            ],
        )

    def _rebuild_model_records(self, nodes):
        if not self._handle:
            return
        self._handle.update_record_list(
            "models",
            [
                {
                    "name": node.name,
                    "selected": node.name in self._selected_nodes,
                    "count_text": f"({node.gaussian_count})",
                }
                for node in nodes
            ],
        )
        self._has_models = bool(nodes)

    # ── Event handlers ────────────────────────────────────────

    def _on_format_click(self, ev):
        container = ev.current_target()
        target = self._find_ancestor_with_attribute(ev.target(), "data-format-idx", container)
        if target is None:
            return

        try:
            new_format = ExportFormat(int(target.get_attribute("data-format-idx", "")))
        except ValueError:
            return

        if new_format == self._format:
            return

        self._format = new_format
        self._rebuild_format_records()

    def _on_model_toggle(self, ev):
        checkbox, node_name = self._get_checkbox_from_event(ev)
        if checkbox is None:
            return

        if checkbox.has_attribute("checked"):
            self._selected_nodes.add(node_name)
        else:
            self._selected_nodes.discard(node_name)

        self._rebuild_model_records(self._get_splat_nodes())
        self._dirty_model("can_export", "export_label")

    def _on_select_all(self, _ev):
        nodes = self._get_splat_nodes()
        self._selected_nodes = {node.name for node in nodes}
        self._rebuild_model_records(nodes)
        self._dirty_model("can_export", "export_label")

    def _on_select_none(self, _ev):
        self._selected_nodes.clear()
        self._rebuild_model_records(self._get_splat_nodes())
        self._dirty_model("can_export", "export_label")

    def _on_export(self, _handle, _ev, _args):
        if not self._selected_nodes:
            return
        self._do_export()

    def _on_cancel(self, _handle, _ev, _args):
        if self._exporting:
            lf.ui.cancel_export()
        lf.ui.set_panel_enabled("lfs.export", False)

    def _on_cancel_export(self, _handle, _ev, _args):
        if self._exporting:
            lf.ui.cancel_export()

    # ── Export logic ──────────────────────────────────────────

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

    def _get_selected_node_names(self):
        selected = []
        for node in self._get_splat_nodes():
            if node.name in self._selected_nodes:
                selected.append(node.name)
        return selected

    def _get_save_path(self, default_name):
        if self._format == ExportFormat.PLY:
            return lf.ui.save_ply_file_dialog(f"{default_name}.ply")
        if self._format == ExportFormat.SOG:
            return lf.ui.save_sog_file_dialog(f"{default_name}.sog")
        if self._format == ExportFormat.SPZ:
            return lf.ui.save_spz_file_dialog(f"{default_name}.spz")
        if self._format == ExportFormat.HTML_VIEWER:
            return lf.ui.save_html_file_dialog(f"{default_name}.html")
        return None

    def _do_export(self):
        selected_nodes = self._get_selected_node_names()
        if not selected_nodes:
            self._dirty_model("can_export")
            return

        default_name = selected_nodes[0]
        path = self._get_save_path(default_name)

        if path:
            lf.export_scene(int(self._format), path, selected_nodes, self._export_sh_degree)
            self._exporting = True
            self._last_progress = -1.0
            self._progress_value = "0"
            self._dirty_model("show_form", "show_progress", "progress_title",
                              "progress_pct", "progress_stage", "progress_value")

    # ── Progress helpers ─────────────────────────────────────

    def _get_progress_title(self):
        fmt = self._cached_export_state.get("format", "file")
        return lf.ui.tr("progress.exporting").replace("%s", fmt)

    def _get_progress_pct(self):
        return f"{self._cached_export_state.get('progress', 0.0) * 100:.0f}%"

    def _get_progress_stage(self):
        return self._cached_export_state.get("stage", "")

    def _update_export_progress(self):
        state = lf.ui.get_export_state()
        self._cached_export_state = state
        if not state.get("active", False):
            self._exporting = False
            self._selection_seeded = False
            lf.ui.set_panel_enabled("lfs.export", False)
            return True

        progress = state.get("progress", 0.0)
        if progress != self._last_progress:
            self._last_progress = progress
            self._progress_value = str(progress)
            self._dirty_model("progress_value", "progress_pct", "progress_stage")
            return True

        return False
