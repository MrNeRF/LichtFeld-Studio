# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rendering panel - main tab for rendering settings."""

import math
import os

import lichtfeld as lf

from . import rml_widgets as w
from .default_reset import LabelDefaultReset, values_match
from .panels import panel_class
from .rendering_defaults import confirm_apply_defaults
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .collision_rendering import CollisionRendering
from .selection_groups import GROUPS_LIST_ID, SelectionGroupsSection
from .types import Panel
from .selected_splat import active_splat_node
from .ui import RuntimeState, PanelStateBinding

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




def _graphics_capabilities():
    query = getattr(lf, "get_graphics_capabilities", None)
    if query is None:
        return {}
    try:
        return query() or {}
    except RuntimeError:
        return {}


def _mesh_wireframe_supported():
    return bool(_graphics_capabilities().get("mesh_wireframe", False))


def _mesh_wide_lines_supported():
    return bool(_graphics_capabilities().get("wide_lines", False))




def _set_theme_vignette_style(*, radius=None, softness=None):
    vignette = _theme_vignette()
    if vignette is None:
        return
    lf.ui.set_theme_vignette_style(
        float(vignette.intensity),
        float(vignette.radius if radius is None else radius),
        float(vignette.softness if softness is None else softness),
    )


SENSOR_HALF_HEIGHT_MM = 12.0
LOD_BUDGET_MIN = 1
LOD_BUDGET_FALLBACK_MAX = 5_000_000
LOD_BUDGET_HARD_MAX = 500_000_000

BOOL_PROPS = [
    "selection_outline_holes",
    "show_coord_axes", "show_pivot", "show_grid", "show_camera_frustums",
    "point_cloud_mode", "selection_outline", "desaturate_unselected", "desaturate_cropping",
    "equirectangular", "mip_filter",
    "mesh_wireframe", "mesh_backface_culling", "mesh_shadow_enabled",
    "apply_appearance_correction", "ppisp_vignette_enabled",
    "lod_enabled", "lod_debug_mode",
]

SLIDER_PROPS = [
    "stochastic_max_frames",
    "selection_outline_opacity", "selection_outline_width", "selection_outline_threshold",
    "axes_size", "grid_opacity", "collision_preview_opacity", "camera_frustum_scale", "voxel_size",
    "focal_length_mm", "render_scale", "color_exposure", "environment_exposure", "environment_rotation_degrees",
    "mesh_wireframe_width", "mesh_light_intensity", "mesh_ambient",
    "ppisp_exposure", "ppisp_vignette_strength", "ppisp_gamma_multiplier",
    "ppisp_gamma_red", "ppisp_gamma_green", "ppisp_gamma_blue",
    "ppisp_crf_toe", "ppisp_crf_shoulder",
    "lod_render_scale", "lod_cone_foveation", "lod_cone_inner_degrees", "lod_cone_outer_degrees",
    "lod_page_pool_splats", "lod_pool_vram_fraction", "lod_fade_frames",
]

SCRUB_FIELD_DEFS = {
    "stochastic_max_frames": ScrubFieldSpec(1, 1024, 1, "%d", data_type=int),
    "selection_outline_opacity": ScrubFieldSpec(0, 1, 0.01, "%.2f"),
    "selection_outline_width": ScrubFieldSpec(1, 12, 0.01, "%.2f"),
    "selection_outline_threshold": ScrubFieldSpec(0.05, 0.95, 0.01, "%.2f"),

    "axes_size": ScrubFieldSpec(0.5, 10.0, 0.01, "%.3f"),
    "collision_preview_opacity": ScrubFieldSpec(0, 1, .01, "%.2f"),
    "grid_opacity": ScrubFieldSpec(0.0, 1.0, 0.01, "%.3f"),
    "camera_frustum_scale": ScrubFieldSpec(0.01, 10.0, 0.01, "%.3f"),
    "voxel_size": ScrubFieldSpec(0.001, 0.1, 0.001, "%.3f"),
    "focal_length_mm": ScrubFieldSpec(10.0, 200.0, 0.1, "%.1f"),
    "render_scale": ScrubFieldSpec(0.25, 1.0, 0.01, "%.2f"),
    "color_exposure": ScrubFieldSpec(0.1, 8.0, 0.01, "%.2f"),
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
    "lod_max_splats": ScrubFieldSpec(
        float(LOD_BUDGET_MIN),
        float(LOD_BUDGET_FALLBACK_MAX),
        10_000.0,
        "%d",
        data_type=int,
    ),
    "lod_render_scale": ScrubFieldSpec(0.1, 5.0, 0.1, "%.1f"),
    "lod_page_pool_splats": ScrubFieldSpec(0.0, 100_000_000.0, 1_000_000.0, "%d", data_type=int),
    "lod_pool_vram_fraction": ScrubFieldSpec(0.05, 0.9, 0.05, "%.2f"),
    "lod_fade_frames": ScrubFieldSpec(0.0, 60.0, 1.0, "%d", data_type=int),
    "lod_cone_foveation": ScrubFieldSpec(0.1, 2.0, 0.1, "%.1f"),
    "lod_cone_inner_degrees": ScrubFieldSpec(0.0, 180.0, 1.0, "%.0f"),
    "lod_cone_outer_degrees": ScrubFieldSpec(0.0, 180.0, 1.0, "%.0f"),
}

SELECT_PROPS = [
    "grid_plane", "sh_degree", "raster_backend", "camera_metrics_mode", "mesh_shadow_resolution",
    "depth_filter_viz_mode",
    "color_tonemapping",
    "transparency_mode",
]
RASTER_BACKENDS = {"3dgs", "3dgut"}

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

BACKGROUND_COLOR_PROP = "background_color"

COLOR_PROPS = [
    "selection_outline_color",
    BACKGROUND_COLOR_PROP,
    "selection_color_preview",
    "selection_color_center_marker",
    "mesh_wireframe_color",
]

SECTION_NAMES = (
    "viewport",
    "quality",
    "quality_advanced",
    "environment",
    "camera",
    "lod",
    "lod_advanced",
    "selection",
    "mesh",
    "collision",
    "post_process",
    "ppisp_crf",
)

LOCALE_KEY = {
    "selection_outline_opacity": "main_panel.selection_outline_opacity",
    "selection_outline_width": "main_panel.selection_outline_width",
    "selection_outline_threshold": "main_panel.selection_outline_threshold",
    "selection_outline_holes": "main_panel.selection_outline_holes",
    "selection_outline_color": "main_panel.selection_outline_color",

    "show_coord_axes": "main_panel.show_coord_axes",
    "show_pivot": "main_panel.show_pivot",
    "show_grid": "main_panel.show_grid",
    "show_camera_frustums": "main_panel.camera_frustums",
    "point_cloud_mode": "main_panel.point_cloud_mode",
    "selection_outline": "main_panel.selection_outline",
    "desaturate_unselected": "main_panel.desaturate_unselected",
    "desaturate_cropping": "main_panel.desaturate_cropping",
    "depth_filter_viz_mode": "main_panel.depth_filter_viz_mode",
    "equirectangular": "main_panel.equirectangular",
    "raster_backend": "main_panel.raster_backend",
    "transparency_mode": "main_panel.transparency_mode",
    "stochastic_max_frames": "main_panel.stochastic_max_frames",
    "mip_filter": "main_panel.mip_filter",
    "axes_size": "main_panel.axes_size",
    "grid_opacity": "main_panel.grid_opacity",
    "focal_length_mm": "main_panel.focal_length",
    "render_scale": "main_panel.render_scale",
    "environment_mode": "main_panel.environment",
    "environment_map_path": "main_panel.environment_map_path",
    "color_exposure": "main_panel.color_exposure",
    "color_tonemapping": "main_panel.color_tonemapping",
    "environment_exposure": "main_panel.environment_exposure",
    "environment_rotation_degrees": "main_panel.environment_rotation",
    "camera_metrics_mode": "main_panel.camera_metrics",
    "sh_degree": "main_panel.sh_degree",
    "grid_plane": "main_panel.plane",
    "background_color": "main_panel.color",
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
    "lod_enabled": "rendering_panel.lod_enabled",
    "lod_debug_mode": "rendering_panel.lod_debug_mode",
    "lod_max_splats": "rendering_panel.lod_max_splats",
    "lod_page_pool_splats": "rendering_panel.lod_page_pool_splats",
    "lod_pool_vram_fraction": "rendering_panel.lod_pool_vram_fraction",
    "lod_fade_frames": "rendering_panel.lod_fade_frames",
    "lod_render_scale": "rendering_panel.lod_render_scale",
    "lod_cone_foveation": "rendering_panel.lod_cone_foveation",
    "lod_cone_inner_degrees": "rendering_panel.lod_cone_inner_degrees",
    "lod_cone_outer_degrees": "rendering_panel.lod_cone_outer_degrees",
}


def _prop_label(prop_id):
    key = LOCALE_KEY.get(prop_id)
    if key:
        label = lf.ui.tr(key)
        if label and label != key:
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
    return text if text.endswith((":", "：")) else f"{text}:"


def _normalize_raster_backend(value):
    backend = str(value or "").lower()
    return backend if backend in RASTER_BACKENDS else "3dgs"


COLOR_CHANNELS = (
    ("r", 0),
    ("g", 1),
    ("b", 2),
)


def _color_channel_field(prop_id, suffix):
    return f"{prop_id}_{suffix}"


RENDERING_INITIALLY_COLLAPSED = {
    "viewport", "environment", "quality_advanced", "lod_advanced",
    "lod",
    "selection",
    "mesh",
    "post_process",
    "ppisp_crf",
}

# Border box of the color picker popup at 100% UI scale: 160dp picker,
# 4dp padding and a 1dp border on each side. The horizontal offset is 210.
_COLOR_PICKER_POPUP_HEIGHT = 170


def color_picker_popup_top(mouse_y, panel_height, popup_height, gap=2):
    """Keep the popup inside the panel, opening above the cursor when needed."""
    top = int(mouse_y) + gap
    if panel_height <= 0 or popup_height <= 0 or top + popup_height <= panel_height:
        return top
    above = int(mouse_y) - int(popup_height) - gap
    if above >= 0:
        return above
    return max(0, int(panel_height) - int(popup_height))


@panel_class("rendering")
class RenderingPanel(Panel):
    id = "lfs.rendering"
    label = "window.rendering"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 10
    template = "rmlui/rendering.rml"
    height_mode = lf.ui.PanelHeightMode.FILL
    update_policy = "dirty"

    def __init__(self):
        self._handle = None
        self._color_edit_prop = None
        self._color_edit_group = None
        self._selection_groups = SelectionGroupsSection(self._show_group_color_picker)
        self._collapsed = set(RENDERING_INITIALLY_COLLAPSED)
        self._popup_el = None
        self._doc = None
        self._picker_click_handled = False
        self._last_swatch_colors = {}
        self._color_text_bufs = {}
        self._last_environment_state = None
        self._last_projection_state = None
        self._last_custom_environment_map_path = ""
        self._last_lod_budget = 0
        self._last_lod_budget_slider_max = 0
        self._chrom_edit_pending = False
        self._render_slider_pending = False
        self._escape_revert = w.EscapeRevertController()
        self._default_reset = LabelDefaultReset(self._reset_default, self._apply_reset)
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS,
            self._get_scrub_value,
            self._set_scrub_value,
            commit_value=self._commit_render_setting,
        )
        self._reactive_binding = PanelStateBinding()
        self._collision = CollisionRendering()

    def capture_chrome(self):
        return {"version": 2, "collapsed": sorted(self._collapsed)}

    def apply_chrome(self, payload):
        self._collapsed = set(RENDERING_INITIALLY_COLLAPSED)
        if isinstance(payload, dict):
            collapsed = payload.get("collapsed")
            if isinstance(collapsed, (list, tuple)):
                self._collapsed = {str(name) for name in collapsed}
                if payload.get("version", 1) < 2:
                    self._collapsed.update({"environment", "quality_advanced", "lod_advanced"})
        if self._handle:
            self._handle.dirty_all()

    def on_mount(self, doc):
        self._doc = doc
        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)
            body.add_event_listener("mouseup", self._on_body_mouseup)
        for el in doc.query_selector_all("input.color-hex"):
            w.bind_select_all_on_focus(el)
            data_value = el.get_attribute("data-value", "")
            if not data_value.endswith("_hex"):
                continue
            prop_id = data_value[:-4]
            if prop_id not in COLOR_PROPS:
                continue
            self._escape_revert.bind(
                el,
                data_value,
                lambda p=prop_id: self._capture_color_snapshot(p),
                lambda snapshot, p=prop_id: self._restore_color_snapshot(p, snapshot),
            )
            el.add_event_listener("change", self._on_color_text_change)
            el.add_event_listener("blur", self._on_color_text_blur)
        for el in doc.query_selector_all("input.color-channel"):
            w.bind_select_all_on_focus(el)
            data_value = el.get_attribute("data-value", "")
            channel = self._split_color_channel_field(data_value)
            if channel is None:
                continue
            prop_id, _channel_index = channel
            self._escape_revert.bind(
                el,
                data_value,
                lambda p=prop_id: self._capture_color_snapshot(p),
                lambda snapshot, p=prop_id: self._restore_color_snapshot(p, snapshot),
            )
            el.add_event_listener("change", self._on_color_text_change)
            el.add_event_listener("blur", self._on_color_text_blur)
        self._scrub_fields.mount(doc)
        self._collision.mount(doc)
        self._default_reset.mount(doc)
        self._selection_groups.mount(doc)
        self._sync_section_states()
        self._subscribe_reactive_state()
        self._request_reactive_update()

    def _subscribe_reactive_state(self):
        if self._reactive_binding.active:
            return

        native_signals = (
            RuntimeState.scene_generation,
            RuntimeState.selection_generation,
            RuntimeState.active_tool,
            RuntimeState.transform_space,
            RuntimeState.pivot_mode,
            RuntimeState.language_generation,
            RuntimeState.viewport_toolbar_generation,
            RuntimeState.collision_state,
        )
        self._reactive_binding.set_handle(self._handle).watch(*native_signals)
        self._reactive_binding.watch(RuntimeState.render_settings_generation, dirty="*")

    def _unsubscribe_reactive_state(self):
        self._reactive_binding.close()

    def _request_reactive_update(self):
        if self._handle:
            w.request_model_update(self._handle)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("rendering")
        if model is None:
            return

        s = lf.get_render_settings
        self._collision.bind(model)
        model.bind("navigation_up_idx", self._navigation_up_index, self._set_navigation_up_index)
        model.bind_event("align_navigation_grid", self._align_navigation_grid)

        for prop_id in BOOL_PROPS:
            if prop_id == "equirectangular":
                model.bind(prop_id,
                           lambda p=prop_id: getattr(s(), p, False),
                           lambda v: self._set_equirectangular(v))
            elif prop_id == "lod_debug_mode":
                model.bind(prop_id,
                           lambda: getattr(s(), "lod_debug_colors", False),
                           lambda v: self._set_render_property("lod_debug_colors", v))
            elif prop_id == "mesh_wireframe":
                model.bind(prop_id,
                           lambda: _mesh_wireframe_supported() and getattr(s(), "mesh_wireframe", False),
                           lambda v, p=prop_id: self._set_render_property(p, bool(v))
                           if _mesh_wireframe_supported() else None)
            else:
                model.bind(prop_id,
                           lambda p=prop_id: getattr(s(), p, False),
                           lambda v, p=prop_id: self._set_render_property(p, v))

        for prop_id in SLIDER_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: float(getattr(s(), p, 0.0)),
                       lambda v, p=prop_id: self._set_render_slider(p, v))

        model.bind("lod_max_splats",
                   lambda: float(self._current_lod_budget()),
                   lambda v: self._set_lod_budget(v))

        for prop_id in SELECT_PROPS:
            if prop_id == "raster_backend":
                model.bind(prop_id,
                           lambda p=prop_id: _normalize_raster_backend(getattr(s(), p, "")),
                           lambda v: self._set_raster_backend(v))
            elif prop_id == "depth_filter_viz_mode":
                model.bind(prop_id,
                           lambda: str(int(getattr(s(), "depth_filter_viz_mode", 1) or 0)),
                           lambda v: self._set_depth_filter_viz_mode(v))
            else:
                model.bind(prop_id,
                           lambda p=prop_id: str(getattr(s(), p, "")),
                           lambda v, p=prop_id: self._set_render_property(p, v))

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

        model.bind_func("mesh_wireframe_supported", _mesh_wireframe_supported)
        model.bind_func("mesh_wide_lines_supported", _mesh_wide_lines_supported)

        all_props = BOOL_PROPS + SLIDER_PROPS + SELECT_PROPS + [
            "environment_mode", "environment_map_path", "ppisp_mode"
        ] + COLOR_PROPS
        for prop_id in all_props:
            model.bind_func(f"label_{prop_id}", lambda p=prop_id: _prop_label(p))
        model.bind_func("label_lod_max_splats", lambda: _prop_label("lod_max_splats"))

        for prop_id in COLOR_PROPS:
            if prop_id == BACKGROUND_COLOR_PROP:
                for suffix, channel_index in COLOR_CHANNELS:
                    field = _color_channel_field(prop_id, suffix)
                    self._color_text_bufs[field] = None

                    def getter(f=field, p=prop_id, idx=channel_index):
                        return self._color_text_value(
                            f,
                            lambda: w.color_channel_text(
                                getattr(s(), p, (0, 0, 0)), idx
                            ),
                        )

                    model.bind(
                        field,
                        getter,
                        lambda v, f=field: self._set_color_text_buf(f, v),
                    )
                hex_field = f"{prop_id}_hex"
                self._color_text_bufs[hex_field] = None
                model.bind(
                    hex_field,
                    lambda f=hex_field, p=prop_id: self._color_text_value(
                        f, lambda: w.color_to_hex(getattr(s(), p, (0, 0, 0)))
                    ),
                    lambda v, f=hex_field: self._set_color_text_buf(f, v),
                )
                continue

            model.bind_func(
                f"{prop_id}_r",
                lambda p=prop_id: w.color_component_label(
                    "R", getattr(s(), p, (0, 0, 0)), 0
                ),
            )
            model.bind_func(
                f"{prop_id}_g",
                lambda p=prop_id: w.color_component_label(
                    "G", getattr(s(), p, (0, 0, 0)), 1
                ),
            )
            model.bind_func(
                f"{prop_id}_b",
                lambda p=prop_id: w.color_component_label(
                    "B", getattr(s(), p, (0, 0, 0)), 2
                ),
            )
            hex_field = f"{prop_id}_hex"
            self._color_text_bufs[hex_field] = None
            model.bind(
                hex_field,
                lambda f=hex_field, p=prop_id: self._color_text_value(
                    f, lambda: w.color_to_hex(getattr(s(), p, (0, 0, 0)))
                ),
                lambda v, f=hex_field: self._set_color_text_buf(f, v),
            )

        for prop_id in CHROM_FLOAT_PROPS:
            model.bind(prop_id,
                       lambda p=prop_id: float(getattr(s(), p, 0.0)),
                       lambda v, p=prop_id: self._set_render_slider(p, v))

        model.bind_func("ppisp_auto",
                         lambda: s() is not None and getattr(s(), "ppisp_mode", "") != "MANUAL")

        model.bind_func("label_hdr_quality", lambda: tr("rendering_panel.section_quality"))
        model.bind_func("label_hdr_environment", lambda: tr("rendering_panel.section_environment"))
        model.bind_func("label_hdr_advanced", lambda: tr("rendering_panel.advanced"))
        model.bind_func("label_hdr_viewport",
                         lambda: _tr_fallback("rendering_panel.section_viewport", "Viewport & Guides"))
        model.bind_func("label_hdr_camera",
                         lambda: _tr_fallback("rendering_panel.section_camera", "Camera & Projection"))
        model.bind_func("label_hdr_lod",
                         lambda: _tr_fallback("rendering_panel.section_lod", "LOD"))
        model.bind_func("label_hdr_selection",
                         lambda: _tr_fallback("rendering_panel.section_selection", "Selection & Overlays"))
        model.bind_func("label_hdr_mesh",
                         lambda: lf.ui.tr("main_panel.mesh") or "Mesh")
        model.bind_func("label_hdr_post_process",
                         lambda: _tr_fallback("rendering_panel.section_post_process", "Color & Appearance"))
        model.bind_func("label_environment_map_browse",
                         lambda: lf.ui.tr("common.browse") or "Browse...")
        model.bind_func("label_ppisp_color_balance",
                         lambda: _entry_label(
                             lf.ui.tr("main_panel.ppisp_color_balance") or "Color Correction"))
        model.bind_func("label_ppisp_crf",
                         lambda: lf.ui.tr("main_panel.ppisp_crf_advanced") or "CRF")

        model.bind_func("fov_display", self._compute_fov)
        model.bind_func("show_fov_controls", self._show_fov_controls)

        model.bind_func("picker_r", lambda: float(self._picker_color()[0]))
        model.bind_func("picker_g", lambda: float(self._picker_color()[1]))
        model.bind_func("picker_b", lambda: float(self._picker_color()[2]))
        model.bind_func(
            "editing_background_color",
            lambda: self._color_edit_prop == BACKGROUND_COLOR_PROP,
        )

        model.bind_func("is_windows", lambda: lf.ui.is_windows_platform())
        model.bind_event("save_defaults", lambda *_: lf.ui.capture_rendering_defaults())
        model.bind_event("apply_defaults", self._confirm_apply_defaults)
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("color_click", self._on_color_click)
        model.bind_event("chrom_change", self._on_chrom_change)
        model.bind_event("picker_change", self._on_picker_change)
        model.bind_event("browse_environment_map", self._on_browse_environment_map)
        self._selection_groups.bind(model)

        self._handle = model.get_handle()
        self._selection_groups.attach(self._handle)

    def _navigation_up_index(self):
        return "1" if lf.get_navigation_up_axis() == "data_z" else "0"

    def _set_navigation_up_index(self, value):
        axis = {"0": "legacy_y", "1": "data_z"}.get(str(value))
        if axis is not None:
            lf.set_navigation_up_axis(axis)
            if self._handle:
                self._handle.dirty("navigation_up_idx")

    def _align_navigation_grid(self, *_):
        lf.set_navigation_up_axis(lf.get_navigation_up_axis(), align_grid=True)
        if self._handle:
            self._handle.dirty("grid_plane")

    def _commit_render_setting(self, prop, value):
        if prop in SLIDER_PROPS + ["lod_max_splats"] and prop in SCRUB_FIELD_DEFS:
            self._mark_view_setting_commit()

    @staticmethod
    def _mark_view_setting_commit():
        lf.ui.mark_project_session_chapter_committed("VIEW")

    def _set_render_property(self, prop, value):
        settings = lf.get_render_settings()
        if settings is None:
            return
        if getattr(settings, prop, object()) == value:
            return
        settings._set_without_session_dirty(prop, value)
        self._mark_view_setting_commit()

    def _set_render_slider(self, prop, value):
        settings = lf.get_render_settings()
        if settings is None:
            return
        typed_value = int(round(float(value))) if prop == "stochastic_max_frames" else float(value)
        settings._set_without_session_dirty(prop, typed_value)
        self._render_slider_pending = True

    def on_update(self, doc):
        s = lf.get_render_settings()
        if not s:
            return False

        dirty = self._collision.update()
        if not getattr(s, "selection_outline", False) and self._color_edit_prop == "selection_outline_color":
            self._hide_picker()
            dirty = True
        dirty |= self._sync_environment_state()
        dirty |= self._sync_projection_state()
        dirty |= self._sync_lod_budget()
        for prop_id in COLOR_PROPS:
            val = getattr(s, prop_id)
            key = (
                prop_id,
                w.color_channel_byte(val, 0),
                w.color_channel_byte(val, 1),
                w.color_channel_byte(val, 2),
            )
            if key == self._last_swatch_colors.get(prop_id):
                continue
            self._last_swatch_colors[prop_id] = key
            swatch = doc.get_element_by_id(f"swatch-{prop_id}")
            if swatch:
                swatch.set_property("background-color", f"rgb({key[1]},{key[2]},{key[3]})")
                dirty = True
            self._dirty_color_bindings(prop_id)
        dirty |= self._scrub_fields.sync_all()
        dirty |= self._selection_groups.sync(visible=self._selection_section_expanded())
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
        if self._doc and current_path and current_path not in ENVIRONMENT_PRESET_PATHS:
            select = self._doc.get_element_by_id("environment-map-preset")
            if select and select.get_attribute("value", "") == CUSTOM_ENVIRONMENT_PRESET_VALUE:
                # RmlUi caches the closed label even when the selected option's
                # text changes. Reapply the value to refresh that copy on update.
                select.set_attribute("value", CUSTOM_ENVIRONMENT_PRESET_VALUE)
        return True

    def _projection_state_snapshot(self):
        settings = lf.get_render_settings()
        if not settings:
            return None
        return (
            _normalize_raster_backend(getattr(settings, "raster_backend", "")),
            bool(getattr(settings, "equirectangular", False)),
        )

    def _dirty_projection_bindings(self):
        self._dirty_model("raster_backend", "equirectangular", "show_fov_controls")

    def _show_fov_controls(self):
        settings = lf.get_render_settings()
        return not bool(getattr(settings, "equirectangular", False)) if settings else True

    def _sync_projection_state(self):
        state = self._projection_state_snapshot()
        if state == self._last_projection_state:
            return False
        self._last_projection_state = state
        self._dirty_projection_bindings()
        return True

    def _set_environment_mode(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        settings._set_without_session_dirty("environment_mode", value)
        self._mark_view_setting_commit()
        self._sync_environment_state()

    def _set_raster_backend(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        backend = _normalize_raster_backend(value)
        settings._set_without_session_dirty("raster_backend", backend)
        self._mark_view_setting_commit()
        self._sync_projection_state()

    def _set_depth_filter_viz_mode(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        try:
            mode = int(value)
        except (TypeError, ValueError):
            return
        settings.depth_filter_viz_mode = max(0, min(2, mode))
        self._dirty_model("depth_filter_viz_mode")

    def _set_equirectangular(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        settings._set_without_session_dirty("equirectangular", bool(value))
        self._mark_view_setting_commit()
        self._sync_projection_state()

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
        settings._set_without_session_dirty("environment_map_path", selected_path)
        self._mark_view_setting_commit()
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
                settings._set_without_session_dirty(
                    "environment_map_path", self._last_custom_environment_map_path
                )
                self._mark_view_setting_commit()
                self._sync_environment_state()
            else:
                self._dirty_environment_bindings()
            return
        try:
            preset_index = int(value)
        except (TypeError, ValueError):
            if value:
                settings._set_without_session_dirty("environment_map_path", value)
                self._mark_view_setting_commit()
                self._sync_environment_state()
            else:
                self._dirty_environment_bindings()
            return
        if preset_index < 0 or preset_index >= len(ENVIRONMENT_PRESET_PATHS):
            self._dirty_environment_bindings()
            return
        preset_path = ENVIRONMENT_PRESET_PATHS[preset_index]
        settings._set_without_session_dirty("environment_map_path", preset_path)
        self._mark_view_setting_commit()
        self._sync_environment_state()

    def _on_browse_environment_map(self, handle, event, args):
        del handle, event, args
        self._browse_environment_map()

    def on_scene_changed(self, doc):
        del doc
        if not self._handle:
            return False

        dirty = False
        self._selection_groups.invalidate()
        dirty |= self._selection_groups.sync(visible=self._selection_section_expanded())
        return dirty

    def on_unmount(self, doc):
        self._unsubscribe_reactive_state()
        self._selection_groups.unmount()
        doc.remove_data_model("rendering")
        self._handle = None
        self._popup_el = None
        self._doc = None
        self._escape_revert.clear()
        self._scrub_fields.unmount()
        self._collision.scrubs.unmount()
        self._collision.handle = None
        self._default_reset.unmount()

    def _reset_default(self, prop):
        settings = lf.get_render_settings()
        if settings is None or prop not in BOOL_PROPS + SLIDER_PROPS + SELECT_PROPS + COLOR_PROPS + ["lod_max_splats", "environment_mode", "ppisp_mode"]:
            return None
        actual = "lod_debug_colors" if prop == "lod_debug_mode" else prop
        info = settings.prop_info(actual)
        return None if info.get("readonly") else info.get("builtin_default")

    def _apply_reset(self, prop, value):
        settings = lf.get_render_settings()
        if settings is None:
            return False
        actual = "lod_debug_colors" if prop == "lod_debug_mode" else prop
        self._scrub_fields.prepare_reset(prop)
        if prop == "equirectangular":
            self._set_equirectangular(value)
        elif prop == "raster_backend":
            self._set_raster_backend(value)
        elif prop == "environment_mode":
            self._set_environment_mode(value)
        elif prop == "ppisp_mode":
            self._set_ppisp_mode(value)
        elif prop == "lod_max_splats":
            self._set_render_property(prop, int(value))
        elif prop == "mesh_wireframe" and not _mesh_wireframe_supported():
            return False
        else:
            self._set_render_property(actual, value)
        # Discard stale text so blur cannot restore the value preceding this reset.
        if prop in COLOR_PROPS:
            self._sync_color_text_bufs(prop)
            self._escape_revert.rebase([f"{prop}_hex"] + [f"{prop}_{suffix}" for suffix, _ in COLOR_CHANNELS])
        if self._handle:
            self._handle.dirty_all()
        self._scrub_fields.sync_all()
        return values_match(getattr(lf.get_render_settings(), actual), value)

    def _get_scrub_value(self, prop):
        if prop == "lod_max_splats":
            return float(self._current_lod_budget())
        settings = lf.get_render_settings()
        if not settings:
            spec = SCRUB_FIELD_DEFS[prop]
            return spec.min_value
        return float(getattr(settings, prop, 0.0))

    def _set_scrub_value(self, prop, value):
        if prop == "lod_max_splats":
            self._set_lod_budget(value)
            return
        settings = lf.get_render_settings()
        if not settings:
            return
        typed_value = int(round(float(value))) if prop == "stochastic_max_frames" else float(value)
        settings._set_without_session_dirty(prop, typed_value)
        if self._handle:
            self._handle.dirty(prop)
            if prop == "focal_length_mm":
                self._handle.dirty("fov_display")

    def _set_lod_budget(self, value):
        settings = lf.get_render_settings()
        if not settings:
            return
        try:
            parsed = int(round(float(str(value).strip().replace(",", "").replace("_", ""))))
        except (TypeError, ValueError):
            return
        budget = max(LOD_BUDGET_MIN, min(self._lod_budget_slider_max(), parsed))
        settings._set_without_session_dirty("lod_max_splats", budget)
        self._render_slider_pending = True
        self._dirty_model("lod_max_splats")

    def _color_text_value(self, field, canonical):
        if self._color_text_bufs.get(field) is None:
            self._color_text_bufs[field] = canonical()
        return self._color_text_bufs[field]

    def _set_color_text_buf(self, field, value):
        self._color_text_bufs[field] = str(value)

    def _color_prop_from_hex_field(self, field):
        if not field.endswith("_hex"):
            return None
        prop_id = field[:-4]
        return prop_id if prop_id in COLOR_PROPS else None

    def _canonical_color_text_value(self, field):
        settings = lf.get_render_settings()
        if not settings:
            return ""
        channel = self._split_color_channel_field(field)
        if channel is not None:
            prop_id, channel_index = channel
            return w.color_channel_text(
                getattr(settings, prop_id, (0.0, 0.0, 0.0)), channel_index
            )
        prop_id = self._color_prop_from_hex_field(field)
        if prop_id is not None:
            return w.color_to_hex(getattr(settings, prop_id, (0.0, 0.0, 0.0)))
        return None

    def _sync_color_text_bufs(self, prop_id):
        settings = lf.get_render_settings()
        if not settings:
            return
        color = getattr(settings, prop_id, (0.0, 0.0, 0.0))
        hex_field = f"{prop_id}_hex"
        if hex_field in self._color_text_bufs:
            self._color_text_bufs[hex_field] = w.color_to_hex(color)
        if prop_id == BACKGROUND_COLOR_PROP:
            for suffix, channel_index in COLOR_CHANNELS:
                field = _color_channel_field(prop_id, suffix)
                self._color_text_bufs[field] = w.color_channel_text(color, channel_index)

    def _commit_color_text_field(self, field):
        buf_val = self._color_text_bufs.get(field)
        updated = False
        if buf_val is not None and str(buf_val).strip():
            channel = self._split_color_channel_field(field)
            if channel is not None:
                prop_id, channel_index = channel
                updated = self._set_color_channel(prop_id, channel_index, buf_val)
            else:
                prop_id = self._color_prop_from_hex_field(field)
                if prop_id is not None:
                    updated = self._set_color_hex(prop_id, buf_val)

        canonical = self._canonical_color_text_value(field)
        if canonical is None:
            return
        self._color_text_bufs[field] = canonical
        if updated:
            self._mark_view_setting_commit()
            channel = self._split_color_channel_field(field)
            prop_id = (
                channel[0]
                if channel is not None
                else self._color_prop_from_hex_field(field)
            )
            if prop_id is not None:
                self._dirty_color_bindings(prop_id)
        elif self._handle:
            self._handle.dirty(field)

    def _set_color_hex(self, prop_id, hex_val):
        s = lf.get_render_settings()
        if not s:
            return False
        color = w.hex_to_color(hex_val)
        if color is not None:
            s._set_without_session_dirty(prop_id, w.normalize_color(color))
            return True
        return False

    def _set_color_channel(self, prop_id, channel_index, val_str):
        s = lf.get_render_settings()
        if not s:
            return False
        parsed = w.parse_color_channel(val_str)
        if parsed is None:
            return False
        color = list(w.normalize_color(getattr(s, prop_id, (0.0, 0.0, 0.0))))
        color[channel_index] = parsed
        s._set_without_session_dirty(prop_id, tuple(color))
        return True

    def _capture_color_snapshot(self, prop_id):
        settings = lf.get_render_settings()
        if not settings:
            return (0.0, 0.0, 0.0)
        return tuple(getattr(settings, prop_id, (0.0, 0.0, 0.0)))

    def _restore_color_snapshot(self, prop_id, snapshot):
        settings = lf.get_render_settings()
        if not settings:
            return
        settings._set_without_session_dirty(
            prop_id, w.normalize_color(snapshot or (0.0, 0.0, 0.0))
        )
        self._dirty_color_bindings(prop_id)

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
        if name == "selection" and expanding:
            self._selection_groups.sync()

    def _selection_section_expanded(self):
        return "selection" not in self._collapsed

    def reveal_selection_groups(self):
        """Expand Selection & Overlays and scroll the group list into view."""
        if not self._selection_section_expanded():
            self._collapsed.discard("selection")
            self._sync_section_states()
            self._selection_groups.sync()
        groups_list = self._doc.get_element_by_id(GROUPS_LIST_ID) if self._doc else None
        if groups_list:
            groups_list.scroll_into_view(False)
        return True

    def _picker_color(self):
        if self._color_edit_group is not None:
            return self._selection_groups.group_color(self._color_edit_group) or (0.0, 0.0, 0.0)
        s = lf.get_render_settings()
        if not self._color_edit_prop or not s:
            return (0.0, 0.0, 0.0)
        return getattr(s, self._color_edit_prop, (0.0, 0.0, 0.0))

    def _open_picker(self, event):
        self._picker_click_handled = True
        mx = int(float(event.get_parameter("mouse_x", "0")))
        my = int(float(event.get_parameter("mouse_y", "0")))
        left = max(0, mx - 210)
        scale = float(lf.ui.get_ui_scale() or 1.0)
        if scale <= 0:
            scale = 1.0
        panel_height = 0.0
        if self._doc is not None:
            body = self._doc.get_element_by_id("body")
            if body is not None:
                panel_height = float(body.client_height)
        top = color_picker_popup_top(my, panel_height, _COLOR_PICKER_POPUP_HEIGHT * scale)
        self._popup_el.set_property("left", f"{left}px")
        self._popup_el.set_property("top", f"{top}px")
        self._popup_el.set_class("visible", True)
        if self._handle:
            self._handle.dirty("picker_r")
            self._handle.dirty("picker_g")
            self._handle.dirty("picker_b")
            self._handle.dirty("editing_background_color")

    def _on_color_click(self, handle, event, args):
        del handle
        if not args or not self._popup_el:
            return
        prop_id = str(args[0])
        if self._color_edit_prop == prop_id:
            self._picker_click_handled = True
            self._hide_picker()
            return
        self._color_edit_group = None
        self._color_edit_prop = prop_id
        self._open_picker(event)

    def _show_group_color_picker(self, gid, event):
        if not self._popup_el:
            return
        if self._color_edit_group == gid:
            self._picker_click_handled = True
            self._hide_picker()
            return
        self._color_edit_prop = None
        self._color_edit_group = gid
        self._open_picker(event)

    def _on_picker_change(self, handle, event, args):
        if not event:
            return
        r = float(event.get_parameter("red", "0"))
        g = float(event.get_parameter("green", "0"))
        b = float(event.get_parameter("blue", "0"))
        if self._color_edit_group is not None:
            self._selection_groups.set_group_color(self._color_edit_group, (r, g, b))
            return
        s = lf.get_render_settings()
        if not s or not self._color_edit_prop:
            return
        prop = self._color_edit_prop
        s._set_without_session_dirty(prop, w.normalize_color((r, g, b)))
        self._dirty_color_bindings(prop, handle)

    def _on_body_mouseup(self, _event):
        if (self._color_edit_prop is not None or self._chrom_edit_pending or
                self._render_slider_pending):
            self._mark_view_setting_commit()
            self._chrom_edit_pending = False
            self._render_slider_pending = False

    def _on_color_text_change(self, event):
        if not event.get_bool_parameter("linebreak", False):
            return
        target = event.current_target()
        if target is None:
            return
        self._commit_color_text_field(target.get_attribute("data-value", ""))

    def _on_color_text_blur(self, event):
        target = event.current_target()
        if target is None:
            return
        self._commit_color_text_field(target.get_attribute("data-value", ""))

    def _on_popup_click(self, event):
        event.stop_propagation()

    def _on_body_click(self, event):
        if self._picker_click_handled:
            self._picker_click_handled = False
            return
        self._hide_picker()

    def _hide_picker(self):
        self._color_edit_prop = None
        self._color_edit_group = None
        if self._popup_el:
            self._popup_el.set_class("visible", False)
        if self._handle:
            self._handle.dirty("editing_background_color")

    def _set_ppisp_mode(self, v):
        s = lf.get_render_settings()
        if s:
            s._set_without_session_dirty("ppisp_mode", v)
            self._mark_view_setting_commit()
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

    def _dirty_color_bindings(self, prop_id, handle=None):
        self._sync_color_text_bufs(prop_id)
        target = handle or self._handle
        if not target:
            return
        for suffix, _channel_index in COLOR_CHANNELS:
            target.dirty(_color_channel_field(prop_id, suffix))
        target.dirty(f"{prop_id}_hex")

    def _split_color_channel_field(self, field):
        for suffix, channel_index in COLOR_CHANNELS:
            suffix_text = f"_{suffix}"
            if field.endswith(suffix_text):
                prop_id = field[: -len(suffix_text)]
                if prop_id == BACKGROUND_COLOR_PROP:
                    return prop_id, channel_index
        return None

    def _current_lod_budget(self) -> int:
        settings = lf.get_render_settings()
        if not settings:
            return LOD_BUDGET_MIN
        return max(LOD_BUDGET_MIN, int(getattr(settings, "lod_max_splats", LOD_BUDGET_FALLBACK_MAX)))

    def _lod_budget_slider_max(self) -> int:
        stats = getattr(lf, "get_lod_stats", lambda: None)()
        if stats:
            full_quality = int(stats.get("full_quality_splats", 0) or 0)
            if full_quality > 0:
                return max(LOD_BUDGET_MIN, min(LOD_BUDGET_HARD_MAX, full_quality))
            model_splats = int(stats.get("model_splats", 0) or 0)
            if model_splats > 0:
                return max(LOD_BUDGET_MIN, min(LOD_BUDGET_HARD_MAX, model_splats))
        _node, _name, active_count = self._active_splat_node()
        if active_count:
            return max(LOD_BUDGET_MIN, min(LOD_BUDGET_HARD_MAX, int(active_count)))
        return LOD_BUDGET_FALLBACK_MAX

    @staticmethod
    def _lod_budget_step(max_budget: int) -> float:
        if max_budget >= 1_000_000:
            return 10_000.0
        if max_budget >= 100_000:
            return 1_000.0
        if max_budget >= 10_000:
            return 100.0
        return 1.0

    def _sync_lod_budget_scrub_spec(self, max_budget: int) -> bool:
        spec = ScrubFieldSpec(
            float(LOD_BUDGET_MIN),
            float(max(LOD_BUDGET_MIN, max_budget)),
            self._lod_budget_step(max_budget),
            "%d",
            data_type=int,
        )
        return self._scrub_fields.set_spec("lod_max_splats", spec)

    def _active_splat_node(self):
        return active_splat_node()


















    def _sync_lod_budget(self) -> bool:
        budget = self._current_lod_budget()
        slider_max = self._lod_budget_slider_max()
        spec_changed = self._sync_lod_budget_scrub_spec(slider_max)
        if budget > slider_max:
            self._set_lod_budget(slider_max)
            budget = slider_max
        changed = budget != self._last_lod_budget or slider_max != self._last_lod_budget_slider_max
        if not changed:
            return spec_changed
        self._last_lod_budget = budget
        self._last_lod_budget_slider_max = slider_max
        self._dirty_model("lod_max_splats")
        return True





    def _on_chrom_change(self, handle, event, args):
        s = lf.get_render_settings()
        if not s or not event:
            return
        self._chrom_edit_pending = True
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
                s._set_without_session_dirty(prop_name, float(val))
        for prop_name in mapping.values():
            handle.dirty(prop_name)

    def _confirm_apply_defaults(self, *_):
        def refresh():
            if self._handle:
                self._handle.dirty_all()
        confirm_apply_defaults(refresh)
