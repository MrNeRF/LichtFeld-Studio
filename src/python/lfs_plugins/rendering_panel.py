# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rendering panel - main tab for rendering settings."""

import math
import os

import lichtfeld as lf

from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .transform_controls import TransformControlsController
from .types import Panel

__lfs_panel_classes__ = ["RenderingPanel"]
__lfs_panel_ids__ = ["lfs.rendering"]


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


def _tr_fallback(key: str, fallback: str) -> str:
    result = lf.ui.tr(key)
    if result and result != key:
        return result
    return fallback


def _theme():
    return lf.ui.theme()


def _theme_vignette():
    theme = _theme()
    return theme.vignette if theme else None


def _set_theme_vignette_style(*, intensity=None, radius=None, softness=None):
    vignette = _theme_vignette()
    if vignette is None:
        return
    lf.ui.set_theme_vignette_style(
        float(vignette.intensity if intensity is None else intensity),
        float(vignette.radius if radius is None else radius),
        float(vignette.softness if softness is None else softness),
    )


SENSOR_HALF_HEIGHT_MM = 12.0
DEFAULT_SIMPLIFY_TARGET_RATIO = 0.5
DEFAULT_SIMPLIFY_KNN_K = 16
DEFAULT_SIMPLIFY_MERGE_CAP = 0.5
DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD = 0.1
MAX_SIMPLIFY_KNN_K = 64

BOOL_PROPS = [
    "show_coord_axes", "show_pivot", "show_grid", "show_camera_frustums",
    "point_cloud_mode", "desaturate_unselected", "desaturate_cropping", "hide_outside_depth_box",
    "equirectangular", "gut", "mip_filter",
    "mesh_wireframe", "mesh_backface_culling", "mesh_shadow_enabled",
    "apply_appearance_correction", "ppisp_vignette_enabled",
]

SLIDER_PROPS = [
    "axes_size", "grid_opacity", "camera_frustum_scale", "voxel_size",
    "focal_length_mm", "render_scale", "environment_exposure", "environment_rotation_degrees",
    "mesh_wireframe_width", "mesh_light_intensity", "mesh_ambient",
    "ppisp_exposure", "ppisp_vignette_strength", "ppisp_gamma_multiplier",
    "ppisp_gamma_red", "ppisp_gamma_green", "ppisp_gamma_blue",
    "ppisp_crf_toe", "ppisp_crf_shoulder",
]

SCRUB_FIELD_DEFS = {
    "axes_size": ScrubFieldSpec(0.5, 10.0, 0.01, "%.3f"),
    "grid_opacity": ScrubFieldSpec(0.0, 1.0, 0.01, "%.3f"),
    "camera_frustum_scale": ScrubFieldSpec(0.01, 10.0, 0.01, "%.3f"),
    "voxel_size": ScrubFieldSpec(0.001, 0.1, 0.001, "%.3f"),
    "focal_length_mm": ScrubFieldSpec(10.0, 200.0, 0.1, "%.1f"),
    "render_scale": ScrubFieldSpec(0.25, 1.0, 0.01, "%.2f"),
    "environment_exposure": ScrubFieldSpec(-6.0, 6.0, 0.01, "%.2f"),
    "environment_rotation_degrees": ScrubFieldSpec(-180.0, 180.0, 0.1, "%.1f"),
    "mesh_wireframe_width": ScrubFieldSpec(0.5, 5.0, 0.01, "%.2f"),
    "mesh_light_intensity": ScrubFieldSpec(0.0, 5.0, 0.01, "%.2f"),
    "mesh_ambient": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
    "ppisp_exposure": ScrubFieldSpec(-3.0, 3.0, 0.01, "%.2f"),
    "ppisp_vignette_strength": ScrubFieldSpec(0.0, 2.0, 0.01, "%.2f"),
    "ppisp_gamma_multiplier": ScrubFieldSpec(0.5, 2.5, 0.01, "%.2f"),
    "ppisp_gamma_red": ScrubFieldSpec(-0.5, 0.5, 0.01, "%.2f"),
    "ppisp_gamma_green": ScrubFieldSpec(-0.5, 0.5, 0.01, "%.2f"),
    "ppisp_gamma_blue": ScrubFieldSpec(-0.5, 0.5, 0.01, "%.2f"),
    "ppisp_crf_toe": ScrubFieldSpec(-1.0, 1.0, 0.01, "%.2f"),
    "ppisp_crf_shoulder": ScrubFieldSpec(-1.0, 1.0, 0.01, "%.2f"),
    "theme_vignette_intensity": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
    "theme_vignette_radius": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
    "theme_vignette_softness": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
    "simplify_target": ScrubFieldSpec(1.0, 1.0, 1.0, "%d", data_type=int),
    "simplify_knn_k": ScrubFieldSpec(1.0, float(MAX_SIMPLIFY_KNN_K), 1.0, "%d", data_type=int),
    "simplify_merge_cap": ScrubFieldSpec(0.01, 0.5, 0.01, "%.2f"),
    "simplify_opacity_prune_threshold": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
}

SELECT_PROPS = [
    "grid_plane", "sh_degree", "camera_metrics_mode", "mesh_shadow_resolution",
]

ENVIRONMENT_PRESET_PATHS = (
    "environments/kloofendal_48d_partly_cloudy_puresky_1k.hdr",
    "environments/alps_field_1k.hdr",
)
CUSTOM_ENVIRONMENT_PRESET_VALUE = "__custom__"
DEFAULT_ENVIRONMENT_PRESET_INDEX = 0

CHROM_FLOAT_PROPS = [
    "ppisp_color_red_x", "ppisp_color_red_y",
    "ppisp_color_green_x", "ppisp_color_green_y",
    "ppisp_color_blue_x", "ppisp_color_blue_y",
    "ppisp_wb_temperature", "ppisp_wb_tint",
]

COLOR_PROPS = [
    "background_color",
    "selection_color_committed", "selection_color_preview",
    "selection_color_center_marker",
    "mesh_wireframe_color",
]

SECTION_NAMES = (
    "transform",
    "viewport",
    "camera",
    "simplify",
    "selection",
    "mesh",
    "post_process",
    "ppisp_crf",
)

LOCALE_KEY = {
    "show_coord_axes": "main_panel.show_coord_axes",
    "show_pivot": "main_panel.show_pivot",
    "show_grid": "main_panel.show_grid",
    "show_camera_frustums": "main_panel.camera_frustums",
    "point_cloud_mode": "main_panel.point_cloud_mode",
    "desaturate_unselected": "main_panel.desaturate_unselected",
    "desaturate_cropping": "main_panel.desaturate_cropping",
    "hide_outside_depth_box": "main_panel.hide_outside_depth_box",
    "equirectangular": "main_panel.equirectangular",
    "gut": "main_panel.gut_mode",
    "mip_filter": "main_panel.mip_filter",
    "axes_size": "main_panel.axes_size",
    "grid_opacity": "main_panel.grid_opacity",
    "focal_length_mm": "main_panel.focal_length",
    "render_scale": "main_panel.render_scale",
    "camera_metrics_mode": "main_panel.camera_metrics",
    "sh_degree": "main_panel.sh_degree",
    "grid_plane": "main_panel.plane",
    "background_color": "main_panel.color",
    "selection_color_committed": "main_panel.committed",
    "selection_color_preview": "main_panel.preview",
    "selection_color_center_marker": "main_panel.center_marker",
    "mesh_wireframe": "main_panel.mesh_wireframe",
    "mesh_wireframe_color": "main_panel.mesh_wireframe_color",
    "mesh_wireframe_width": "main_panel.mesh_wireframe_width",
    "mesh_light_intensity": "main_panel.mesh_light_intensity",
    "mesh_ambient": "main_panel.mesh_ambient",
    "mesh_backface_culling": "main_panel.mesh_backface_culling",
    "mesh_shadow_enabled": "main_panel.mesh_shadow_enabled",
    "mesh_shadow_resolution": "main_panel.mesh_shadow_resolution",
    "camera_frustum_scale": "main_panel.camera_frustum_scale",
    "voxel_size": "main_panel.voxel_size",
    "apply_appearance_correction": "main_panel.appearance_correction",
    "ppisp_mode": "main_panel.ppisp_mode",
    "ppisp_exposure": "main_panel.ppisp_exposure",
    "ppisp_vignette_enabled": "main_panel.ppisp_vignette",
    "ppisp_vignette_strength": "main_panel.ppisp_vignette",
    "ppisp_gamma_multiplier": "main_panel.ppisp_gamma",
    "ppisp_gamma_red": "main_panel.ppisp_gamma_red",
    "ppisp_gamma_green": "main_panel.ppisp_gamma_green",
    "ppisp_gamma_blue": "main_panel.ppisp_gamma_blue",
    "ppisp_crf_toe": "main_panel.ppisp_crf_toe",
    "ppisp_crf_shoulder": "main_panel.ppisp_crf_shoulder",
}


def _prop_label(prop_id):
    key = LOCALE_KEY.get(prop_id)
    if key:
        label = lf.ui.tr(key)
        if label:
            return _entry_label(label)
    s = lf.get_render_settings()
    if s:
        info = s.prop_info(prop_id)
        return _entry_label(info.get("name", prop_id))
    return _entry_label(prop_id)


def _entry_label(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ":"
    return text if text.endswith(":") else f"{text}:"


def _color_to_hex(c):
    return f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"


def _hex_to_color(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)
    except ValueError:
        return None


class RenderingPanel(Panel):
    id = "lfs.rendering"
    label = "Rendering"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 10
    template = "rmlui/rendering.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 16

    def __init__(self):
        self._handle = None
        self._transform_controls = TransformControlsController()
        self._color_edit_prop = None
        self._collapsed = {"selection", "mesh", "post_process", "ppisp_crf"}
        self._popup_el = None
        self._doc = None
        self._picker_click_handled = False
        self._last_swatch_colors = {}
        self._last_panel_label = ""
        self._simplify_target_count = 0
        self._simplify_target_touched = False
        self._simplify_knn_k = DEFAULT_SIMPLIFY_KNN_K
        self._simplify_knn_k_touched = False
        self._simplify_merge_cap = DEFAULT_SIMPLIFY_MERGE_CAP
        self._simplify_opacity_prune_threshold = DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD
        self._simplify_source_name = ""
        self._simplify_original_count = 0
        self._simplify_task_active = False
        self._simplify_progress_value = "0"
        self._simplify_progress_stage = ""
        self._simplify_error_text = ""
        self._last_environment_state = None
        self._last_custom_environment_map_path = ""
        self._escape_revert = w.EscapeRevertController()
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    def _sync_panel_label(self):
        label = tr("window.rendering")
        if not label or label == self._last_panel_label:
            return
        if lf.ui.set_panel_label(self.id, label):
            self._last_panel_label = label

    def on_mount(self, doc):
        self._doc = doc
        self._sync_panel_label()
        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)
        for el in doc.query_selector_all("input.color-hex"):
            w.bind_select_all_on_focus(el)
            data_value = el.get_attribute("data-value", "")
            if data_value.endswith("_hex"):
                prop_id = data_value[:-4]
                self._escape_revert.bind(
                    el,
                    data_value,
                    lambda p=prop_id: self._capture_color_snapshot(p),
                    lambda snapshot, p=prop_id: self._restore_color_snapshot(p, snapshot),
                )
        self._refresh_simplify_source(force=True)
        self._scrub_fields.mount(doc)
        self._transform_controls.mount(doc)
        self._sync_section_states()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("rendering")
        if model is None:
            return

        s = lf.get_render_settings

        self._transform_controls.bind_model(model)

        for prop_id in BOOL_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: getattr(s(), p, False),
                       lambda v, p=prop_id: setattr(s(), p, v) if s() else None)

        for prop_id in SLIDER_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: float(getattr(s(), p, 0.0)),
                       lambda v, p=prop_id: setattr(s(), p, float(v)) if s() else None)

        for prop_id in SELECT_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: str(getattr(s(), p, "")),
                       lambda v, p=prop_id: setattr(s(), p, v) if s() else None)

        model.bind("environment_mode",
                   lambda: str(getattr(s(), "environment_mode", "")),
                   lambda v: self._set_environment_mode(v))
        model.bind("environment_map_preset",
                   self._get_environment_map_preset,
                   self._set_environment_map_preset)
        model.bind_func("environment_map_is_custom", self._environment_map_is_custom)
        model.bind_func("environment_map_display_name", self._environment_map_display_name)
        model.bind_func("environment_map_has_custom_option", self._environment_map_has_custom_option)
        model.bind_func("environment_map_last_custom_display_name", self._environment_map_last_custom_display_name)

        model.bind("ppisp_mode",
                    lambda: str(getattr(s(), "ppisp_mode", "")),
                    lambda v: self._set_ppisp_mode(v))

        model.bind_func("environment_enabled",
                        lambda: s() is not None and getattr(s(), "environment_mode", "") == "EQUIRECTANGULAR")

        all_props = BOOL_PROPS + SLIDER_PROPS + SELECT_PROPS + [
            "environment_mode", "environment_map_path", "ppisp_mode"
        ] + COLOR_PROPS
        for prop_id in all_props:
            model.bind_func(f"label_{prop_id}", lambda p=prop_id: _prop_label(p))

        for prop_id in COLOR_PROPS:
            model.bind_func(f"{prop_id}_r",
                            lambda p=prop_id: f"R:{int(getattr(s(), p, (0,0,0))[0]*255):>3d}")
            model.bind_func(f"{prop_id}_g",
                            lambda p=prop_id: f"G:{int(getattr(s(), p, (0,0,0))[1]*255):>3d}")
            model.bind_func(f"{prop_id}_b",
                            lambda p=prop_id: f"B:{int(getattr(s(), p, (0,0,0))[2]*255):>3d}")
            model.bind(f"{prop_id}_hex",
                       lambda p=prop_id: _color_to_hex(getattr(s(), p, (0,0,0))),
                       lambda v, p=prop_id: self._set_color_hex(p, v))

        for prop_id in CHROM_FLOAT_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: float(getattr(s(), p, 0.0)),
                       lambda v, p=prop_id: setattr(s(), p, float(v)) if s() else None)

        model.bind("simplify_target",
                   lambda: str(self._compute_simplify_target_count()),
                   lambda v: self._set_simplify_target_count(v))
        model.bind("simplify_knn_k",
                   lambda: str(self._compute_simplify_knn_k()),
                   lambda v: self._set_simplify_knn_k(v))
        model.bind("simplify_merge_cap",
                   lambda: f"{self._compute_simplify_merge_cap():.2f}",
                   lambda v: self._set_simplify_merge_cap(v))
        model.bind("simplify_opacity_prune_threshold",
                   lambda: f"{self._compute_simplify_opacity_prune_threshold():.2f}",
                   lambda v: self._set_simplify_opacity_prune_threshold(v))

        model.bind_func("ppisp_auto",
                         lambda: s() is not None and getattr(s(), "ppisp_mode", "") != "MANUAL")

        model.bind_func("label_panel_title",
                         lambda: lf.ui.tr("rendering") or "Rendering")
        model.bind_func("label_hdr_viewport",
                         lambda: "Viewport")
        model.bind_func("label_hdr_camera",
                         lambda: "Camera & Projection")
        model.bind_func("label_hdr_simplify",
                         lambda: _tr_fallback("rendering_panel.section_simplify", "Splat Simplify"))
        model.bind_func("label_hdr_selection",
                         lambda: "Selection & Overlays")
        model.bind_func("label_hdr_mesh",
                         lambda: lf.ui.tr("main_panel.mesh") or "Mesh")
        model.bind_func("label_hdr_post_process",
                         lambda: "Post Processing")
        model.bind_func("label_environment_map_browse",
                         lambda: lf.ui.tr("common.browse") or "Browse...")
        model.bind_func("label_simplify_source",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_source", "Source")))
        model.bind_func("label_simplify_select_source",
                         lambda: _tr_fallback("rendering_panel.simplify_select_source", "Select a splat node"))
        model.bind_func("label_simplify_target",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_target", "Target")))
        model.bind_func("label_simplify_target_stat",
                         lambda: _tr_fallback("rendering_panel.simplify_target", "Target"))
        model.bind_func("label_simplify_knn_k",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_knn_k", "kNN K")))
        model.bind_func("label_simplify_merge_cap",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_merge_cap", "Merge Cap")))
        model.bind_func("label_simplify_opacity_prune",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_opacity_prune", "Opacity Prune")))
        model.bind_func("label_simplify_original",
                         lambda: _tr_fallback("rendering_panel.simplify_original", "Original"))
        model.bind_func("label_simplify_output",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_output", "Output")))
        model.bind_func("label_simplify_apply",
                         lambda: _tr_fallback("common.apply", "Apply"))
        model.bind_func("label_simplify_cancel",
                         lambda: _tr_fallback("common.cancel", "Cancel"))
        model.bind_func("label_ppisp_color_balance",
                         lambda: _entry_label(
                             lf.ui.tr("main_panel.ppisp_color_balance") or "Color Correction"))
        model.bind_func("label_ppisp_crf",
                         lambda: lf.ui.tr("main_panel.ppisp_crf_advanced") or "CRF")

        model.bind_func("fov_display", self._compute_fov)

        model.bind_func("picker_r",
                         lambda: float(getattr(s(), self._color_edit_prop, (0, 0, 0))[0])
                         if self._color_edit_prop and s() else 0.0)
        model.bind_func("picker_g",
                         lambda: float(getattr(s(), self._color_edit_prop, (0, 0, 0))[1])
                         if self._color_edit_prop and s() else 0.0)
        model.bind_func("picker_b",
                         lambda: float(getattr(s(), self._color_edit_prop, (0, 0, 0))[2])
                         if self._color_edit_prop and s() else 0.0)

        model.bind_func("is_windows", lambda: lf.ui.is_windows_platform())
        model.bind_func("label_console",
                         lambda: lf.ui.tr("main_panel.console") or "Console")
        model.bind_func("simplify_has_source", lambda: bool(self._simplify_source_name))
        model.bind_func("simplify_source_name", lambda: self._simplify_source_name)
        model.bind_func("simplify_original_count", lambda: f"{self._simplify_original_count:,}")
        model.bind_func("simplify_target_count", lambda: f"{self._compute_simplify_target_count():,}")
        model.bind_func("simplify_output_name", self._simplify_output_name)
        model.bind_func("simplify_can_apply", self._can_run_simplify)
        model.bind_func("simplify_show_progress", lambda: self._simplify_task_active)
        model.bind_func("simplify_progress_value", lambda: self._simplify_progress_value)
        model.bind_func("simplify_progress_pct", self._simplify_progress_pct)
        model.bind_func("simplify_progress_stage", lambda: self._simplify_progress_stage)
        model.bind_func("simplify_show_error", lambda: bool(self._simplify_error_text))
        model.bind_func("simplify_error_text", lambda: self._simplify_error_text)

        model.bind("theme_vignette_enabled",
                   lambda: bool((vignette := _theme_vignette()) and vignette.enabled),
                   lambda v: lf.ui.set_theme_vignette_enabled(bool(v)))
        model.bind("theme_vignette_intensity",
                   lambda: float(vignette.intensity) if (vignette := _theme_vignette()) else 0.3,
                   lambda v: lf.ui.set_theme_vignette_intensity(float(v)))
        model.bind("theme_vignette_radius",
                   lambda: float(vignette.radius) if (vignette := _theme_vignette()) else 0.75,
                   lambda v: _set_theme_vignette_style(radius=float(v)))
        model.bind("theme_vignette_softness",
                   lambda: float(vignette.softness) if (vignette := _theme_vignette()) else 0.45,
                   lambda v: _set_theme_vignette_style(softness=float(v)))
        model.bind_func("label_theme_vignette_enabled",
                         lambda: _entry_label(lf.ui.tr("main_panel.theme_vignette") or "Vignette"))
        model.bind_func("label_theme_vignette_intensity",
                         lambda: _entry_label(
                             lf.ui.tr("main_panel.theme_vignette_intensity") or "Intensity"))
        model.bind_func("label_theme_vignette_radius",
                         lambda: _entry_label(
                             lf.ui.tr("main_panel.theme_vignette_radius") or "Radius"))
        model.bind_func("label_theme_vignette_softness",
                         lambda: _entry_label(
                             lf.ui.tr("main_panel.theme_vignette_softness") or "Softness"))

        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("color_click", self._on_color_click)
        model.bind_event("chrom_change", self._on_chrom_change)
        model.bind_event("picker_change", self._on_picker_change)
        model.bind_event("simplify_apply", self._on_simplify_apply)
        model.bind_event("simplify_cancel", self._on_simplify_cancel)
        model.bind_event("toggle_console",
                         lambda h, e, a: lf.ui.toggle_system_console())
        model.bind_event("browse_environment_map", self._on_browse_environment_map)

        self._handle = model.get_handle()
        self._sync_panel_label()

    def on_update(self, doc):
        self._sync_panel_label()
        s = lf.get_render_settings()
        if not s:
            return False

        dirty = False
        dirty |= self._transform_controls.update(doc)
        dirty |= self._sync_environment_state()
        for prop_id in COLOR_PROPS:
            val = getattr(s, prop_id)
            key = (prop_id, int(val[0] * 255), int(val[1] * 255), int(val[2] * 255))
            if key == self._last_swatch_colors.get(prop_id):
                continue
            self._last_swatch_colors[prop_id] = key
            swatch = doc.get_element_by_id(f"swatch-{prop_id}")
            if swatch:
                swatch.set_property("background-color", f"rgb({key[1]},{key[2]},{key[3]})")
                dirty = True
        dirty |= self._refresh_simplify_source(force=False)
        dirty |= self._sync_simplify_task_state(force=False)
        dirty |= self._scrub_fields.sync_all()
        return dirty

    def _environment_state_snapshot(self):
        settings = lf.get_render_settings()
        if not settings:
            return None
        return (
            str(getattr(settings, "environment_mode", "")),
            str(getattr(settings, "environment_map_path", "")),
        )

    def _dirty_environment_bindings(self):
        if not self._handle:
            return
        self._handle.dirty_all()

    def _sync_environment_state(self):
        settings = lf.get_render_settings()
        current_path = str(getattr(settings, "environment_map_path", "") or "") if settings else ""
        if current_path and current_path not in ENVIRONMENT_PRESET_PATHS:
            self._remember_custom_environment_map(current_path)
        state = self._environment_state_snapshot()
        if state == self._last_environment_state:
            return False
        self._last_environment_state = state
        self._dirty_environment_bindings()
        return True

    def _set_environment_mode(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        settings.environment_mode = value
        self._sync_environment_state()

    def _environment_map_is_custom(self):
        settings = lf.get_render_settings()
        if not settings:
            return False
        current_path = str(getattr(settings, "environment_map_path", "") or "")
        return bool(current_path) and current_path not in ENVIRONMENT_PRESET_PATHS

    def _environment_map_display_name(self):
        settings = lf.get_render_settings()
        if not settings:
            return ""
        current_path = str(getattr(settings, "environment_map_path", "") or "")
        if not current_path or current_path in ENVIRONMENT_PRESET_PATHS:
            return ""
        display_name = os.path.basename(current_path)
        return display_name or current_path

    def _environment_map_has_custom_option(self):
        return bool(self._last_custom_environment_map_path)

    def _environment_map_last_custom_display_name(self):
        if not self._last_custom_environment_map_path:
            return ""
        display_name = os.path.basename(self._last_custom_environment_map_path)
        return display_name or self._last_custom_environment_map_path

    def _remember_custom_environment_map(self, path):
        normalized = str(path or "")
        if not normalized or normalized in ENVIRONMENT_PRESET_PATHS:
            return
        self._last_custom_environment_map_path = normalized

    def _environment_map_dialog_start_dir(self):
        settings = lf.get_render_settings()
        if not settings:
            return ""
        current_path = str(getattr(settings, "environment_map_path", "") or "")
        if not current_path:
            return ""
        if os.path.isabs(current_path):
            return os.path.dirname(current_path)
        return ""

    def _open_environment_map_dialog(self, start_dir):
        for module in (getattr(lf, "ui", None), lf):
            if module is None:
                continue
            for name in ("open_environment_map_dialog", "open_image_file_dialog", "open_image_dialog"):
                dialog = getattr(module, name, None)
                if callable(dialog):
                    return dialog(start_dir)
        lf.log.warn("No environment-map file dialog is available in the current Python bindings")
        return ""

    def _browse_environment_map(self):
        settings = lf.get_render_settings()
        if not settings:
            return
        selected_path = self._open_environment_map_dialog(self._environment_map_dialog_start_dir())
        if not selected_path:
            self._dirty_environment_bindings()
            return
        if os.path.splitext(selected_path)[1].lower() not in {".hdr", ".exr"}:
            lf.log.warn(f"Ignoring unsupported environment map selection: {selected_path}")
            self._dirty_environment_bindings()
            return
        self._remember_custom_environment_map(selected_path)
        settings.environment_map_path = selected_path
        self._sync_environment_state()

    def _get_environment_map_preset(self):
        settings = lf.get_render_settings()
        if not settings:
            return str(DEFAULT_ENVIRONMENT_PRESET_INDEX)
        current_path = str(getattr(settings, "environment_map_path", "") or "")
        try:
            return str(ENVIRONMENT_PRESET_PATHS.index(current_path))
        except ValueError:
            if current_path:
                return CUSTOM_ENVIRONMENT_PRESET_VALUE
            return str(DEFAULT_ENVIRONMENT_PRESET_INDEX)

    def _set_environment_map_preset(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        value = str(value or "")
        if value == CUSTOM_ENVIRONMENT_PRESET_VALUE:
            if self._environment_map_is_custom():
                self._dirty_environment_bindings()
                return
            if self._last_custom_environment_map_path:
                settings.environment_map_path = self._last_custom_environment_map_path
                self._sync_environment_state()
            else:
                self._dirty_environment_bindings()
            return
        try:
            preset_index = int(value)
        except (TypeError, ValueError):
            if value:
                settings.environment_map_path = value
                self._sync_environment_state()
            else:
                self._dirty_environment_bindings()
            return
        if preset_index < 0 or preset_index >= len(ENVIRONMENT_PRESET_PATHS):
            self._dirty_environment_bindings()
            return
        preset_path = ENVIRONMENT_PRESET_PATHS[preset_index]
        settings.environment_map_path = preset_path
        self._sync_environment_state()

    def _on_browse_environment_map(self, handle, event, args):
        del handle, event, args
        self._browse_environment_map()

    def on_scene_changed(self, doc):
        del doc
        self._transform_controls.scene_changed()
        if self._handle:
            self._handle.dirty_all()

    def on_unmount(self, doc):
        doc.remove_data_model("rendering")
        self._handle = None
        self._transform_controls.unmount()
        self._popup_el = None
        self._doc = None
        self._escape_revert.clear()
        self._scrub_fields.unmount()

    def _get_scrub_value(self, prop):
        if prop == "simplify_target":
            return float(self._compute_simplify_target_count())
        if prop == "simplify_knn_k":
            return float(self._compute_simplify_knn_k())
        if prop == "simplify_merge_cap":
            return self._compute_simplify_merge_cap()
        if prop == "simplify_opacity_prune_threshold":
            return self._compute_simplify_opacity_prune_threshold()
        if prop == "theme_vignette_intensity":
            theme = _theme()
            return float(theme.vignette.intensity) if theme else 0.3
        if prop == "theme_vignette_radius":
            vignette = _theme_vignette()
            return float(vignette.radius) if vignette else 0.75
        if prop == "theme_vignette_softness":
            vignette = _theme_vignette()
            return float(vignette.softness) if vignette else 0.45
        settings = lf.get_render_settings()
        if not settings:
            spec = SCRUB_FIELD_DEFS[prop]
            return spec.min_value
        return float(getattr(settings, prop, 0.0))

    def _set_scrub_value(self, prop, value):
        if prop == "simplify_target":
            self._set_simplify_target_count(value)
            return
        if prop == "simplify_knn_k":
            self._set_simplify_knn_k(value)
            return
        if prop == "simplify_merge_cap":
            self._set_simplify_merge_cap(value)
            return
        if prop == "simplify_opacity_prune_threshold":
            self._set_simplify_opacity_prune_threshold(value)
            return
        if prop == "theme_vignette_intensity":
            lf.ui.set_theme_vignette_intensity(float(value))
            if self._handle:
                self._handle.dirty(prop)
            return
        if prop == "theme_vignette_radius":
            _set_theme_vignette_style(radius=float(value))
            if self._handle:
                self._handle.dirty(prop)
            return
        if prop == "theme_vignette_softness":
            _set_theme_vignette_style(softness=float(value))
            if self._handle:
                self._handle.dirty(prop)
            return
        settings = lf.get_render_settings()
        if not settings:
            return
        setattr(settings, prop, float(value))
        if self._handle:
            self._handle.dirty(prop)
            if prop == "focal_length_mm":
                self._handle.dirty("fov_display")

    def _set_color_hex(self, prop_id, hex_val):
        s = lf.get_render_settings()
        if not s:
            return
        color = _hex_to_color(hex_val)
        if color:
            setattr(s, prop_id, color)

    def _capture_color_snapshot(self, prop_id):
        settings = lf.get_render_settings()
        if not settings:
            return (0.0, 0.0, 0.0)
        return tuple(getattr(settings, prop_id, (0.0, 0.0, 0.0)))

    def _restore_color_snapshot(self, prop_id, snapshot):
        settings = lf.get_render_settings()
        if not settings:
            return
        setattr(settings, prop_id, tuple(snapshot or (0.0, 0.0, 0.0)))
        if self._handle:
            self._handle.dirty_all()

    def _compute_fov(self):
        s = lf.get_render_settings()
        view = lf.get_current_view()
        if not s or not view or view.width <= 0 or view.height <= 0:
            return ""
        focal_mm = s.focal_length_mm
        vfov = 2.0 * math.degrees(math.atan(SENSOR_HALF_HEIGHT_MM / focal_mm))
        aspect = view.width / view.height
        hfov = 2.0 * math.degrees(math.atan(aspect * math.tan(math.radians(vfov * 0.5))))
        fmt = lf.ui.tr("rendering_panel.fov_format")
        if fmt:
            return fmt.format(hfov=hfov, vfov=vfov)
        return f"H:{hfov:.1f}\u00b0 V:{vfov:.1f}\u00b0"

    def _get_section_elements(self, name):
        if not self._doc:
            return None, None, None
        dom_name = name.replace("_", "-")
        header = self._doc.get_element_by_id(f"hdr-{dom_name}")
        arrow = self._doc.get_element_by_id(f"arrow-{dom_name}")
        content = self._doc.get_element_by_id(f"sec-{dom_name}")
        return header, arrow, content

    def _sync_section_states(self):
        for name in SECTION_NAMES:
            header, arrow, content = self._get_section_elements(name)
            if content:
                w.sync_section_state(content, name not in self._collapsed, header, arrow)

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        header, arrow, content = self._get_section_elements(name)
        if content:
            w.animate_section_toggle(content, expanding, arrow, header_element=header)

    def _on_color_click(self, handle, event, args):
        if not args or not self._popup_el:
            return
        self._picker_click_handled = True
        prop_id = str(args[0])
        if self._color_edit_prop == prop_id:
            self._hide_picker()
            return
        self._color_edit_prop = prop_id
        mx = int(float(event.get_parameter("mouse_x", "0")))
        my = int(float(event.get_parameter("mouse_y", "0")))
        left = max(0, mx - 210)
        self._popup_el.set_property("left", f"{left}px")
        self._popup_el.set_property("top", f"{my + 2}px")
        self._popup_el.set_class("visible", True)
        handle.dirty("picker_r")
        handle.dirty("picker_g")
        handle.dirty("picker_b")

    def _on_picker_change(self, handle, event, args):
        s = lf.get_render_settings()
        if not s or not event or not self._color_edit_prop:
            return
        r = float(event.get_parameter("red", "0"))
        g = float(event.get_parameter("green", "0"))
        b = float(event.get_parameter("blue", "0"))
        prop = self._color_edit_prop
        setattr(s, prop, (r, g, b))
        handle.dirty(f"{prop}_r")
        handle.dirty(f"{prop}_g")
        handle.dirty(f"{prop}_b")
        handle.dirty(f"{prop}_hex")

    def _on_popup_click(self, event):
        event.stop_propagation()

    def _on_body_click(self, event):
        if self._picker_click_handled:
            self._picker_click_handled = False
            return
        self._hide_picker()

    def _hide_picker(self):
        self._color_edit_prop = None
        if self._popup_el:
            self._popup_el.set_class("visible", False)

    def _set_ppisp_mode(self, v):
        s = lf.get_render_settings()
        if s:
            setattr(s, "ppisp_mode", v)
        if self._handle:
            self._handle.dirty("ppisp_auto")

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _active_splat_node(self):
        scene = getattr(lf, "get_scene", lambda: None)()
        if scene is None:
            return None, "", 0

        selected_name = str(getattr(lf, "get_selected_node_name", lambda: "")() or "")
        if not selected_name:
            return None, "", 0

        node = scene.get_node(selected_name)
        if node is None:
            return None, "", 0

        node_type_enum = getattr(getattr(lf, "scene", None), "NodeType", None)
        if node_type_enum is not None and getattr(node, "type", None) != node_type_enum.SPLAT:
            return None, "", 0

        try:
            splat = node.splat_data()
        except Exception:
            splat = None
        if splat is None:
            return None, "", 0

        try:
            count = int(splat.visible_count())
        except Exception:
            count = int(getattr(node, "gaussian_count", 0))
        return node, selected_name, count

    def _refresh_simplify_source(self, force: bool) -> bool:
        _node, source_name, source_count = self._active_splat_node()
        changed = force or source_name != self._simplify_source_name or source_count != self._simplify_original_count
        if not changed:
            return False

        self._simplify_source_name = source_name
        self._simplify_original_count = source_count
        if source_count > 0:
            if self._simplify_target_touched and self._simplify_target_count > 0:
                self._simplify_target_count = self._clamp_simplify_target_count(self._simplify_target_count, source_count)
            else:
                self._simplify_target_count = self._default_simplify_target_count(source_count)
            if self._simplify_knn_k_touched and self._simplify_knn_k > 0:
                self._simplify_knn_k = self._clamp_simplify_knn_k(self._simplify_knn_k, source_count)
            else:
                self._simplify_knn_k = self._default_simplify_knn_k(source_count)
        elif not self._simplify_target_touched:
            self._simplify_target_count = 0
        if source_count <= 0 and not self._simplify_knn_k_touched:
            self._simplify_knn_k = DEFAULT_SIMPLIFY_KNN_K
        self._sync_simplify_scrub_spec()
        self._dirty_model(
            "simplify_has_source",
            "simplify_source_name",
            "simplify_original_count",
            "simplify_target",
            "simplify_target_count",
            "simplify_knn_k",
            "simplify_merge_cap",
            "simplify_opacity_prune_threshold",
            "simplify_output_name",
            "simplify_can_apply",
        )
        return True

    def _sync_simplify_scrub_spec(self):
        max_value = max(0, int(self._simplify_original_count))
        target_spec = ScrubFieldSpec(
            1.0 if max_value > 0 else 0.0,
            float(max_value),
            1.0,
            "%d",
            data_type=int,
        )
        knn_max = float(self._compute_simplify_knn_k_max())
        knn_spec = ScrubFieldSpec(1.0, knn_max, 1.0, "%d", data_type=int)
        self._scrub_fields.set_spec("simplify_target", target_spec)
        self._scrub_fields.set_spec("simplify_knn_k", knn_spec)

    def _default_simplify_target_count(self, original_count=None) -> int:
        source_count = self._simplify_original_count if original_count is None else int(original_count)
        if source_count <= 0:
            return 0
        return max(1, min(source_count, int(math.ceil(source_count * DEFAULT_SIMPLIFY_TARGET_RATIO))))

    def _clamp_simplify_target_count(self, value, original_count=None):
        try:
            parsed = int(round(float(str(value).strip().replace(",", "").replace("_", ""))))
        except (TypeError, ValueError):
            return None
        clamped = max(1, parsed)
        max_count = self._simplify_original_count if original_count is None else int(original_count)
        if max_count > 0:
            clamped = min(clamped, max_count)
        return clamped

    def _compute_simplify_target_count(self) -> int:
        if self._simplify_original_count <= 0:
            return 0
        if self._simplify_target_count <= 0:
            return self._default_simplify_target_count()
        clamped = self._clamp_simplify_target_count(self._simplify_target_count)
        if clamped is None:
            return self._default_simplify_target_count()
        return clamped

    def _compute_simplify_ratio(self) -> float:
        if self._simplify_original_count <= 0:
            return 0.0
        return float(self._compute_simplify_target_count()) / float(self._simplify_original_count)

    def _compute_simplify_knn_k_max(self, original_count=None) -> int:
        source_count = self._simplify_original_count if original_count is None else int(original_count)
        if source_count <= 1:
            return 1
        return max(1, min(MAX_SIMPLIFY_KNN_K, source_count - 1))

    def _default_simplify_knn_k(self, original_count=None) -> int:
        clamped = self._clamp_simplify_knn_k(DEFAULT_SIMPLIFY_KNN_K, original_count)
        return 1 if clamped is None else clamped

    def _clamp_simplify_knn_k(self, value, original_count=None):
        try:
            parsed = int(round(float(str(value).strip().replace(",", "").replace("_", ""))))
        except (TypeError, ValueError):
            return None
        return max(1, min(parsed, self._compute_simplify_knn_k_max(original_count)))

    def _compute_simplify_knn_k(self, original_count=None) -> int:
        clamped = self._clamp_simplify_knn_k(self._simplify_knn_k, original_count)
        if clamped is not None:
            return clamped
        return self._default_simplify_knn_k(original_count)

    def _set_simplify_knn_k(self, value):
        next_value = self._clamp_simplify_knn_k(value)
        if next_value is None or next_value == self._simplify_knn_k:
            return
        self._simplify_knn_k = next_value
        self._simplify_knn_k_touched = True
        self._dirty_model("simplify_knn_k")

    def _clamp_simplify_merge_cap(self, value):
        try:
            parsed = float(str(value).strip().replace(",", "").replace("_", ""))
        except (TypeError, ValueError):
            return None
        return max(0.01, min(parsed, 0.5))

    def _compute_simplify_merge_cap(self) -> float:
        clamped = self._clamp_simplify_merge_cap(self._simplify_merge_cap)
        return DEFAULT_SIMPLIFY_MERGE_CAP if clamped is None else clamped

    def _set_simplify_merge_cap(self, value):
        next_value = self._clamp_simplify_merge_cap(value)
        if next_value is None or math.isclose(next_value, self._simplify_merge_cap, abs_tol=1.0e-9):
            return
        self._simplify_merge_cap = next_value
        self._dirty_model("simplify_merge_cap")

    def _clamp_simplify_opacity_prune_threshold(self, value):
        try:
            parsed = float(str(value).strip().replace(",", "").replace("_", ""))
        except (TypeError, ValueError):
            return None
        return max(0.0, min(parsed, 1.0))

    def _compute_simplify_opacity_prune_threshold(self) -> float:
        clamped = self._clamp_simplify_opacity_prune_threshold(self._simplify_opacity_prune_threshold)
        return DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD if clamped is None else clamped

    def _set_simplify_opacity_prune_threshold(self, value):
        next_value = self._clamp_simplify_opacity_prune_threshold(value)
        if next_value is None or math.isclose(next_value, self._simplify_opacity_prune_threshold, abs_tol=1.0e-9):
            return
        self._simplify_opacity_prune_threshold = next_value
        self._dirty_model("simplify_opacity_prune_threshold")

    def _simplify_output_name(self) -> str:
        if not self._simplify_source_name:
            return ""
        return f"{self._simplify_source_name}_{self._compute_simplify_target_count()}"

    def _can_run_simplify(self) -> bool:
        return bool(
            self._simplify_source_name
            and self._simplify_original_count > 0
            and self._compute_simplify_target_count() > 0
            and not self._simplify_task_active
        )

    def _set_simplify_target_count(self, value):
        next_value = self._clamp_simplify_target_count(value)
        if next_value is None:
            return
        if next_value == self._simplify_target_count and self._simplify_target_touched:
            return
        self._simplify_target_count = next_value
        self._simplify_target_touched = True
        self._dirty_model("simplify_target", "simplify_target_count", "simplify_output_name")

    def _simplify_progress_pct(self) -> str:
        try:
            return f"{int(round(float(self._simplify_progress_value) * 100.0))}%"
        except (TypeError, ValueError):
            return "0%"

    def _sync_simplify_task_state(self, force: bool) -> bool:
        active = bool(getattr(lf, "is_splat_simplify_active", lambda: False)())
        progress = max(0.0, min(1.0, float(getattr(lf, "get_splat_simplify_progress", lambda: 0.0)())))
        progress_value = f"{progress:.4f}".rstrip("0").rstrip(".") or "0"
        stage = str(getattr(lf, "get_splat_simplify_stage", lambda: "")() or "")
        error_text = str(getattr(lf, "get_splat_simplify_error", lambda: "")() or "")

        changed = force or (
            active != self._simplify_task_active or
            progress_value != self._simplify_progress_value or
            stage != self._simplify_progress_stage or
            error_text != self._simplify_error_text
        )
        if not changed:
            return False

        self._simplify_task_active = active
        self._simplify_progress_value = progress_value
        self._simplify_progress_stage = stage
        self._simplify_error_text = error_text
        self._dirty_model(
            "simplify_can_apply",
            "simplify_show_progress",
            "simplify_progress_value",
            "simplify_progress_pct",
            "simplify_progress_stage",
            "simplify_show_error",
            "simplify_error_text",
        )
        return True

    def _start_simplify(self):
        if not self._can_run_simplify():
            return
        self._simplify_error_text = ""
        self._dirty_model("simplify_show_error", "simplify_error_text")
        lf.simplify_splats(
            self._simplify_source_name,
            ratio=self._compute_simplify_ratio(),
            knn_k=self._compute_simplify_knn_k(),
            merge_cap=self._compute_simplify_merge_cap(),
            opacity_prune_threshold=self._compute_simplify_opacity_prune_threshold(),
        )
        self._sync_simplify_task_state(force=True)

    def _on_simplify_apply(self, _handle=None, _ev=None, _args=None):
        self._start_simplify()

    def _on_simplify_cancel(self, _handle=None, _ev=None, _args=None):
        cancel = getattr(lf, "cancel_splat_simplify", None)
        if cancel is not None:
            cancel()

    def _on_chrom_change(self, handle, event, args):
        s = lf.get_render_settings()
        if not s or not event:
            return
        mapping = {
            "red_x": "ppisp_color_red_x",
            "red_y": "ppisp_color_red_y",
            "green_x": "ppisp_color_green_x",
            "green_y": "ppisp_color_green_y",
            "blue_x": "ppisp_color_blue_x",
            "blue_y": "ppisp_color_blue_y",
            "wb_temp": "ppisp_wb_temperature",
            "wb_tint": "ppisp_wb_tint",
        }
        for param_key, prop_name in mapping.items():
            val = event.get_parameter(param_key, "")
            if val:
                setattr(s, prop_name, float(val))
        for prop_name in mapping.values():
            handle.dirty(prop_name)
