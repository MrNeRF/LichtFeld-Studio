# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export panel for exporting scene nodes."""

import html
from typing import Set
from enum import IntEnum

import lichtfeld as lf
from . import rml_widgets
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .types import Panel

__lfs_panel_classes__ = ["ExportPanel"]
__lfs_panel_ids__ = ["lfs.export"]


class ExportFormat(IntEnum):
    PLY = 0
    SOG = 1
    SPZ = 2
    HTML_VIEWER = 3
    USD = 4
    NUREC_USDZ = 5
    RAD = 6


FORMAT_INFO = (
    (ExportFormat.PLY, "export.format.ply_standard"),
    (ExportFormat.SOG, "export.format.sog_supersplat"),
    (ExportFormat.SPZ, "export.format.spz_niantic"),
    (ExportFormat.RAD, "export.format.rad_random_access"),
    (ExportFormat.USD, "export.format.usd_openusd"),
    (ExportFormat.NUREC_USDZ, "export.format.usdz_nurec"),
    (ExportFormat.HTML_VIEWER, "export.format.html_viewer"),
)

SCRUB_FIELD_DEFS = {
    "sh_degree": ScrubFieldSpec(0.0, 3.0, 1.0, "%d", data_type=int),
}


def _xml_unescape(text):
    return html.unescape(text or "")


class ExportPanel(Panel):
    id = "lfs.export"
    label = "Export"
    space = lf.ui.PanelSpace.FLOATING
    order = 10
    template = "rmlui/export_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (320, 0)
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
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS,
            self._get_scrub_value,
            self._set_scrub_value,
        )
        # RAD export settings
        self._rad_flip_y = False  # Y-flip checkbox (off by default)
        self._rad_customize_lod = False
        self._rad_lod_list: list[int] = [100]  # Default to 100%
        self._rad_new_lod_str = "100"
        self._rad_lod_collapsed = True  # Whether LOD levels section is collapsed
        self._doc = None  # Document reference for DOM access

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("export")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("export.export"))
        model.bind_func("export_label", self._get_export_label)
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

        # RAD export settings bindings
        model.bind_func("show_rad_settings", lambda: self._format == ExportFormat.RAD)
        model.bind_func("rad_flip_y", lambda: self._rad_flip_y)
        model.bind_event("toggle_rad_flip_y", self._on_toggle_rad_flip_y)
        model.bind_func("rad_customize_lod", lambda: self._rad_customize_lod)
        model.bind_record_list("rad_lod_list")  # Record list for editable values
        model.bind(
            "rad_new_lod_str",
            lambda: self._rad_new_lod_str,
            self._set_rad_new_lod_str,
        )
        model.bind_event("toggle_rad_customize_lod", self._on_toggle_rad_customize_lod)
        model.bind_event("add_rad_lod", self._on_add_rad_lod)
        model.bind_event("remove_rad_lod", self._on_remove_rad_lod)
        model.bind_event("num_step_rad_lod", self._on_num_step_rad_lod)
        model.bind_event("update_lod_value", self._on_update_lod_value)
        model.bind_event("toggle_section", self._on_toggle_section)

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

    def _set_rad_new_lod_str(self, v):
        self._rad_new_lod_str = v if v else ""

    def _update_rad_lod_list(self):
        """Update the RAD LOD record list in the data model."""
        if self._handle:
            self._handle.update_record_list(
                "rad_lod_list", [{"value": str(lod)} for lod in self._rad_lod_list]
            )

    def _on_toggle_rad_flip_y(self, _handle, _ev, _args):
        self._rad_flip_y = not self._rad_flip_y
        self._dirty_model("rad_flip_y")

    def _on_toggle_rad_customize_lod(self, _handle, _ev, _args):
        self._rad_customize_lod = not self._rad_customize_lod
        self._update_rad_lod_list()
        self._dirty_model("rad_customize_lod")

    def _on_toggle_section(self, _handle, _event, args):
        """Toggle collapsible section visibility."""
        if not args:
            return
        section = str(args[0])
        if section != "rad_lod":
            return

        if not self._doc:
            return

        # Find the arrow and content elements
        arrow = self._doc.get_element_by_id("arrow-rad-lod")
        content = self._doc.get_element_by_id("sec-rad-lod")
        header = self._doc.get_element_by_id("hdr-rad-lod")

        if content is None:
            return

        # Toggle the collapsed state
        expanding = self._rad_lod_collapsed
        self._rad_lod_collapsed = not expanding

        # Animate the section toggle
        rml_widgets.animate_section_toggle(
            content, expanding, arrow, header_element=header
        )

    def _sync_rad_lod_section_state(self):
        """Initialize the RAD LOD section visual state."""
        if not self._doc:
            return
        header = self._doc.get_element_by_id("hdr-rad-lod")
        arrow = self._doc.get_element_by_id("arrow-rad-lod")
        content = self._doc.get_element_by_id("sec-rad-lod")
        if content:
            rml_widgets.sync_section_state(
                content, not self._rad_lod_collapsed, header, arrow
            )

    def _on_add_rad_lod(self, _handle, _ev, _args):
        try:
            value = int(self._rad_new_lod_str)
        except (ValueError, TypeError):
            return
        # Clamp to 1-100 range
        value = max(1, min(100, value))
        # Avoid duplicates
        if value not in self._rad_lod_list:
            self._rad_lod_list.append(value)
            self._rad_lod_list.sort()
            self._update_rad_lod_list()

    def _on_remove_rad_lod(self, _handle, _ev, args):
        try:
            index = int(args[0]) if args else -1
        except (ValueError, TypeError):
            return
        if 0 <= index < len(self._rad_lod_list):
            self._rad_lod_list.pop(index)
            # Ensure at least one LOD remains (default to 100 if empty)
            if not self._rad_lod_list:
                self._rad_lod_list = [100]
            self._update_rad_lod_list()

    def _on_update_lod_value(self, _handle, event, _args):
        """Update a specific LOD value when edited."""
        # Get the index from the parent row element
        target = event.current_target()
        if target is None:
            return

        # Find the parent row element with data-lod-index
        parent = target.parent_node()
        if parent is None:
            return

        try:
            index = int(parent.get_attribute("data-lod-index", "-1"))
        except (ValueError, TypeError):
            return

        if index < 0 or index >= len(self._rad_lod_list):
            return

        # Get the new value from the input
        try:
            new_value_str = target.get_attribute("value", "")
            new_value = int(new_value_str)
        except (ValueError, TypeError):
            return

        # Clamp to 1-100 range (values outside this range are not allowed)
        original_value = self._rad_lod_list[index]
        clamped_value = max(1, min(100, new_value))

        # Check for duplicates (excluding the current index)
        if (
            clamped_value in self._rad_lod_list
            and self._rad_lod_list.index(clamped_value) != index
        ):
            # Duplicate found - revert to original value
            self._update_rad_lod_list()
            return

        self._rad_lod_list[index] = clamped_value
        self._rad_lod_list.sort()
        self._update_rad_lod_list()

    def _on_num_step_rad_lod(self, _handle, _ev, args):
        try:
            delta = int(args[0]) if args else 0
        except (ValueError, TypeError):
            return
        try:
            current = int(self._rad_new_lod_str)
        except (ValueError, TypeError):
            current = 100
        new_value = max(1, min(100, current + delta))
        self._rad_new_lod_str = str(new_value)
        self._dirty_model("rad_new_lod_str")

    def _get_scrub_value(self, prop):
        del prop
        return self._export_sh_degree

    def _set_scrub_value(self, prop, value):
        del prop
        self._set_sh_degree(value)

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._exporting = False
        self._last_progress = -1.0
        self._cached_export_state = {}
        self._selection_seeded = False
        self._last_node_key = None
        self._last_lang = lf.ui.get_current_language()

        export_form = doc.get_element_by_id("export-form")
        if export_form:
            export_form.add_event_listener("submit", self._on_export_submit)

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
        self._update_rad_lod_list()
        self._sync_rad_lod_section_state()
        self._scrub_fields.mount(doc)

    def on_update(self, doc):
        if self._exporting:
            dirty = self._update_export_progress()
            dirty |= self._scrub_fields.sync_all()
            return dirty

        if self._last_progress >= 0.0:
            self._last_progress = -1.0
            self._progress_value = "0"
            self._dirty_model("show_form", "show_progress")
            self._scrub_fields.sync_all()
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

        dirty |= self._scrub_fields.sync_all()
        return dirty

    def on_scene_changed(self, doc):
        self._last_node_key = None

    def on_unmount(self, doc):
        doc.remove_data_model("export")
        self._handle = None
        self._doc = None
        self._scrub_fields.unmount()

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

    def _get_checkbox_from_event(self, event):
        container = event.current_target()
        target = rml_widgets.find_ancestor_with_attribute(
            event.target(), "data-node-name", container
        )
        if target is None:
            return None, None

        checkbox = target
        if (
            checkbox.tag_name != "input"
            or checkbox.get_attribute("type", "") != "checkbox"
        ):
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
        target = rml_widgets.find_ancestor_with_attribute(
            ev.target(), "data-format-idx", container
        )
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
        # Dirty RAD settings visibility when format changes
        self._dirty_model(
            "show_rad_settings", "rad_flip_y", "rad_customize_lod", "no_rad_lod"
        )

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

    def _on_export_submit(self, ev):
        if self._selected_nodes:
            self._do_export()
        ev.stop_propagation()

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
            return lf.ui.save_ply_file_dialog(default_name)
        if self._format == ExportFormat.SOG:
            return lf.ui.save_sog_file_dialog(default_name)
        if self._format == ExportFormat.SPZ:
            return lf.ui.save_spz_file_dialog(default_name)
        if self._format == ExportFormat.USD:
            return lf.ui.save_usd_file_dialog(default_name)
        if self._format == ExportFormat.NUREC_USDZ:
            return lf.ui.save_usdz_file_dialog(default_name)
        if self._format == ExportFormat.HTML_VIEWER:
            return lf.ui.save_html_file_dialog(default_name)
        if self._format == ExportFormat.RAD:
            return lf.ui.save_rad_file_dialog(default_name)
        return None

    def _do_export(self):
        selected_nodes = self._get_selected_node_names()
        if not selected_nodes:
            self._dirty_model("can_export")
            return

        default_name = selected_nodes[0]
        path = self._get_save_path(default_name)

        if path:
            # Prepare RAD LOD settings if applicable
            rad_lod_ratios = None
            if self._format == ExportFormat.RAD and self._rad_customize_lod:
                # Convert percentages to ratios (e.g., 100 -> 1.0, 50 -> 0.5)
                rad_lod_ratios = [lod / 100.0 for lod in self._rad_lod_list]

            lf.export_scene(
                int(self._format),
                path,
                selected_nodes,
                self._export_sh_degree,
                rad_lod_ratios=rad_lod_ratios,
                rad_flip_y=self._rad_flip_y,
            )
            self._exporting = True
            self._last_progress = -1.0
            self._progress_value = "0"
            self._dirty_model(
                "show_form",
                "show_progress",
                "progress_title",
                "progress_pct",
                "progress_stage",
                "progress_value",
            )

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
