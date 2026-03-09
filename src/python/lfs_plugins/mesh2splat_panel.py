# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained RmlUI panel for mesh-to-splat conversion."""

from __future__ import annotations

from typing import Iterable

import lichtfeld as lf

from .types import RmlPanel


class Mesh2SplatPanel(RmlPanel):
    """Floating retained panel for mesh-to-splat conversion."""

    idname = "native.mesh2splat"
    label = "Mesh to Splat"
    space = "FLOATING"
    order = 12
    rml_template = "rmlui/mesh2splat_panel.rml"
    rml_height_mode = "content"
    initial_width = 420
    update_interval_ms = 100

    _RESOLUTION_OPTIONS = (128, 256, 512, 1024, 2048, 4096)
    _MIN_RESOLUTION = 16

    def __init__(self):
        self._handle = None
        self._selected_mesh_name = ""
        self._quality = 0.5
        self._resolution_index = 3
        self._gaussian_scale = 0.65
        self._has_initial_conversion = False
        self._has_meshes = False
        self._last_mesh_key = None
        self._last_lang = ""
        self._last_progress_value = "0"
        self._last_progress_stage = ""
        self._last_active = False
        self._error_text = ""

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("mesh2splat")
        if model is None:
            return

        tr = lf.ui.tr

        model.bind_func("panel_label", lambda: tr("mesh2splat.title"))
        model.bind_func("source_mesh_label", lambda: tr("mesh2splat.source_mesh"))
        model.bind_func("gaussian_scale_label", lambda: tr("mesh2splat.gaussian_scale"))
        model.bind_func("gaussian_scale_help", lambda: tr("mesh2splat.tooltip_gaussian_scale"))
        model.bind_func("sampling_density_header", lambda: tr("mesh2splat.sampling_density_header"))
        model.bind_func("sampling_density_label", lambda: tr("mesh2splat.sampling_density"))
        model.bind_func("sampling_density_help", lambda: tr("mesh2splat.tooltip_quality"))
        model.bind_func("max_resolution_label", lambda: tr("mesh2splat.max_resolution"))
        model.bind_func("max_resolution_help", lambda: tr("mesh2splat.tooltip_max_resolution"))
        model.bind_func("effective_resolution_label", lambda: tr("mesh2splat.effective_resolution"))
        model.bind_func("convert_label", lambda: tr("mesh2splat.convert"))
        model.bind_func("no_meshes_label", lambda: tr("mesh2splat.no_meshes"))

        model.bind_func("gaussian_scale_text", lambda: f"{self._gaussian_scale:.2f}")
        model.bind_func("quality_text", lambda: f"{self._quality:.2f}")
        model.bind_func("effective_resolution", lambda: str(self._compute_resolution_target()))
        model.bind_func("show_no_meshes", lambda: not self._has_meshes)
        model.bind_func("can_convert", self._can_convert)
        model.bind_func("show_progress", lambda: self._last_active)
        model.bind_func("progress_value", lambda: self._last_progress_value)
        model.bind_func("progress_pct", self._progress_pct)
        model.bind_func("progress_stage", lambda: self._last_progress_stage)
        model.bind_func("show_error", lambda: bool(self._error_text))
        model.bind_func("error_text", lambda: self._error_text)

        model.bind("gaussian_scale", lambda: f"{self._gaussian_scale:.2f}", self._set_gaussian_scale)
        model.bind("quality", lambda: f"{self._quality:.2f}", self._set_quality)

        model.bind_event("do_convert", self._on_convert)
        model.bind_record_list("meshes")
        model.bind_record_list("resolutions")

        self._handle = model.get_handle()

    def on_load(self, doc):
        super().on_load(doc)

        mesh_list = doc.get_element_by_id("mesh-list")
        if mesh_list:
            mesh_list.add_event_listener("click", self._on_mesh_click)

        resolution_list = doc.get_element_by_id("resolution-list")
        if resolution_list:
            resolution_list.add_event_listener("click", self._on_resolution_click)

        gaussian_slider = doc.get_element_by_id("gaussian-scale-slider")
        if gaussian_slider:
            gaussian_slider.add_event_listener("change", self._on_parameter_commit)

        quality_slider = doc.get_element_by_id("quality-slider")
        if quality_slider:
            quality_slider.add_event_listener("change", self._on_parameter_commit)

        self._last_lang = lf.ui.get_current_language()
        self._last_mesh_key = None
        self._refresh_scene_state(force=True)
        self._sync_conversion_state(force=True)

    def on_update(self, doc):
        del doc
        dirty = False

        current_lang = lf.ui.get_current_language()
        if current_lang != self._last_lang:
            self._last_lang = current_lang
            self._rebuild_resolution_records()
            self._last_mesh_key = None
            self._dirty_model()
            dirty = True

        if self._refresh_scene_state(force=False):
            dirty = True

        if self._sync_conversion_state(force=False):
            dirty = True

        return dirty

    def on_scene_changed(self, doc):
        del doc
        self._last_mesh_key = None

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _scene_mesh_nodes(self):
        scene = lf.get_scene()
        if scene is None:
            return []

        try:
            nodes = list(scene.get_nodes(lf.scene.NodeType.MESH))
        except TypeError:
            nodes = list(scene.get_nodes())

        mesh_nodes = []
        for node in nodes:
            if getattr(node, "type", None) != lf.scene.NodeType.MESH:
                continue
            try:
                mesh_info = node.mesh()
            except TypeError:
                mesh_info = getattr(node, "mesh", None)
                if callable(mesh_info):
                    mesh_info = mesh_info()
            if mesh_info is not None:
                mesh_nodes.append((node, mesh_info))
        return mesh_nodes

    def _refresh_scene_state(self, force: bool) -> bool:
        mesh_nodes = self._scene_mesh_nodes()
        mesh_key = tuple(
            (
                node.name,
                int(getattr(mesh_info, "vertex_count", 0)),
                int(getattr(mesh_info, "face_count", 0)),
            )
            for node, mesh_info in mesh_nodes
        )

        changed = force or mesh_key != self._last_mesh_key
        if not changed:
            return False

        self._last_mesh_key = mesh_key
        self._has_meshes = bool(mesh_nodes)
        self._sync_selected_mesh(mesh_nodes)
        self._rebuild_mesh_records(mesh_nodes)
        self._rebuild_resolution_records()
        self._dirty_model(
            "show_no_meshes",
            "can_convert",
            "effective_resolution",
            "error_text",
            "show_error",
        )
        return True

    def _sync_selected_mesh(self, mesh_nodes: Iterable[tuple[object, object]]):
        mesh_nodes = list(mesh_nodes)
        valid_names = {node.name for node, _mesh_info in mesh_nodes}
        if self._selected_mesh_name in valid_names:
            return
        if mesh_nodes:
            self._selected_mesh_name = mesh_nodes[0][0].name
            return
        self._selected_mesh_name = ""

    def _mesh_stats_text(self, mesh_info) -> str:
        vertex_count = int(getattr(mesh_info, "vertex_count", 0))
        face_count = int(getattr(mesh_info, "face_count", 0))
        return f"{vertex_count}v / {face_count}f"

    def _rebuild_mesh_records(self, mesh_nodes):
        if not self._handle:
            return
        self._handle.update_record_list(
            "meshes",
            [
                {
                    "name": node.name,
                    "selected": node.name == self._selected_mesh_name,
                    "stats_text": self._mesh_stats_text(mesh_info),
                }
                for node, mesh_info in mesh_nodes
            ],
        )

    def _rebuild_resolution_records(self):
        if not self._handle:
            return
        self._handle.update_record_list(
            "resolutions",
            [
                {
                    "index": str(index),
                    "label": str(value),
                    "selected": index == self._resolution_index,
                }
                for index, value in enumerate(self._RESOLUTION_OPTIONS)
            ],
        )

    def _compute_resolution_target(self) -> int:
        max_res = self._RESOLUTION_OPTIONS[self._resolution_index]
        return int(self._MIN_RESOLUTION + self._quality * (max_res - self._MIN_RESOLUTION))

    def _can_convert(self) -> bool:
        return bool(self._selected_mesh_name and self._has_meshes and not self._last_active)

    def _set_gaussian_scale(self, value):
        try:
            next_value = max(0.1, min(2.0, float(value)))
        except (TypeError, ValueError):
            return
        if abs(next_value - self._gaussian_scale) < 1e-6:
            return
        self._gaussian_scale = next_value
        self._dirty_model("gaussian_scale", "gaussian_scale_text")

    def _set_quality(self, value):
        try:
            next_value = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        if abs(next_value - self._quality) < 1e-6:
            return
        self._quality = next_value
        self._dirty_model("quality", "quality_text", "effective_resolution")

    def _progress_pct(self) -> str:
        try:
            return f"{int(round(float(self._last_progress_value) * 100.0))}%"
        except (TypeError, ValueError):
            return "0%"

    def _sync_conversion_state(self, force: bool) -> bool:
        active = bool(lf.is_mesh2splat_active())
        progress_value = f"{max(0.0, min(1.0, float(lf.get_mesh2splat_progress()))):.4f}".rstrip("0").rstrip(".")
        if not progress_value:
            progress_value = "0"
        stage = str(getattr(lf, "get_mesh2splat_stage", lambda: "")() or "")
        error_text = ""
        error_text = str(lf.get_mesh2splat_error() or "")

        changed = force or (
            active != self._last_active or
            progress_value != self._last_progress_value or
            stage != self._last_progress_stage or
            error_text != self._error_text
        )
        if not changed:
            return False

        self._last_active = active
        self._last_progress_value = progress_value
        self._last_progress_stage = stage
        self._error_text = error_text
        self._dirty_model(
            "can_convert",
            "show_progress",
            "progress_value",
            "progress_pct",
            "progress_stage",
            "show_error",
            "error_text",
        )
        return True

    def _find_ancestor_with_attribute(self, element, attribute, stop=None):
        while element is not None and element != stop:
            if element.has_attribute(attribute):
                return element
            element = element.parent()
        return None

    def _on_mesh_click(self, event):
        container = event.current_target()
        target = self._find_ancestor_with_attribute(event.target(), "data-mesh-name", container)
        if target is None:
            return

        mesh_name = target.get_attribute("data-mesh-name", "")
        if not mesh_name or mesh_name == self._selected_mesh_name:
            return

        self._selected_mesh_name = mesh_name
        self._rebuild_mesh_records(self._scene_mesh_nodes())
        self._dirty_model("can_convert")

    def _on_resolution_click(self, event):
        container = event.current_target()
        target = self._find_ancestor_with_attribute(event.target(), "data-resolution-index", container)
        if target is None:
            return

        try:
            next_index = int(target.get_attribute("data-resolution-index", "-1"))
        except (TypeError, ValueError):
            return

        if next_index < 0 or next_index >= len(self._RESOLUTION_OPTIONS):
            return
        if next_index == self._resolution_index:
            return

        self._resolution_index = next_index
        self._rebuild_resolution_records()
        self._dirty_model("effective_resolution")
        self._request_reconvert_if_needed()

    def _on_parameter_commit(self, _event=None):
        self._request_reconvert_if_needed()

    def _request_reconvert_if_needed(self):
        if self._has_initial_conversion and not lf.is_mesh2splat_active():
            self._start_conversion()

    def _start_conversion(self):
        if not self._selected_mesh_name:
            return

        self._error_text = ""
        self._dirty_model("show_error", "error_text")

        try:
            lf.mesh_to_splat(
                self._selected_mesh_name,
                sigma=self._gaussian_scale,
                quality=self._quality,
                max_resolution=self._RESOLUTION_OPTIONS[self._resolution_index],
            )
            self._has_initial_conversion = True
            self._sync_conversion_state(force=True)
        except Exception as exc:
            self._error_text = str(exc)
            self._dirty_model("show_error", "error_text")

    def _on_convert(self, _handle=None, _ev=None, _args=None):
        if not self._can_convert():
            return
        self._start_conversion()
