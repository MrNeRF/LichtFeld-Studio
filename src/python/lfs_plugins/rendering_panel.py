# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rendering panel - main tab for rendering settings."""

import math

import lichtfeld as lf

from .types import RmlPanel

SENSOR_HALF_HEIGHT_MM = 12.0

BOOL_PROPS = [
    "show_coord_axes", "show_pivot", "show_grid", "show_camera_frustums",
    "point_cloud_mode", "desaturate_unselected", "desaturate_cropping",
    "equirectangular", "gut", "mip_filter",
    "mesh_wireframe", "mesh_backface_culling", "mesh_shadow_enabled",
    "apply_appearance_correction", "ppisp_vignette_enabled",
]

SLIDER_PROPS = [
    "axes_size", "grid_opacity", "camera_frustum_scale", "voxel_size",
    "focal_length_mm", "render_scale",
    "mesh_wireframe_width", "mesh_light_intensity", "mesh_ambient",
    "ppisp_exposure", "ppisp_vignette_strength", "ppisp_gamma_multiplier",
    "ppisp_gamma_red", "ppisp_gamma_green", "ppisp_gamma_blue",
    "ppisp_crf_toe", "ppisp_crf_shoulder",
]

SELECT_PROPS = [
    "grid_plane", "sh_degree", "mesh_shadow_resolution",
]

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

LOCALE_KEY = {
    "show_coord_axes": "main_panel.show_coord_axes",
    "show_pivot": "main_panel.show_pivot",
    "show_grid": "main_panel.show_grid",
    "show_camera_frustums": "main_panel.camera_frustums",
    "point_cloud_mode": "main_panel.point_cloud_mode",
    "desaturate_unselected": "main_panel.desaturate_unselected",
    "desaturate_cropping": "main_panel.desaturate_cropping",
    "equirectangular": "main_panel.equirectangular",
    "gut": "main_panel.gut_mode",
    "mip_filter": "main_panel.mip_filter",
    "axes_size": "main_panel.axes_size",
    "grid_opacity": "main_panel.grid_opacity",
    "focal_length_mm": "main_panel.focal_length",
    "render_scale": "main_panel.render_scale",
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
            return label
    s = lf.get_render_settings()
    if s:
        info = s.prop_info(prop_id)
        return info.get("name", prop_id)
    return prop_id


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


class RenderingPanel(RmlPanel):
    idname = "lfs.rendering"
    label = "Rendering"
    space = "MAIN_PANEL_TAB"
    order = 10
    rml_template = "rmlui/rendering.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._handle = None
        self._color_edit_prop = None
        self._collapsed = set()
        self._popup_el = None
        self._doc = None
        self._picker_click_handled = False
        self._last_swatch_colors = {}

    def on_load(self, doc):
        self._doc = doc
        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("rendering")
        if model is None:
            return

        s = lf.get_render_settings

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

        model.bind("ppisp_mode",
                    lambda: str(getattr(s(), "ppisp_mode", "")),
                    lambda v: self._set_ppisp_mode(v))

        all_props = BOOL_PROPS + SLIDER_PROPS + SELECT_PROPS + ["ppisp_mode"] + COLOR_PROPS
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

        model.bind_func("ppisp_auto",
                         lambda: s() is not None and getattr(s(), "ppisp_mode", "") != "MANUAL")

        model.bind_func("label_hdr_selection_colors",
                         lambda: lf.ui.tr("main_panel.selection_colors") or "Selection Colors")
        model.bind_func("label_hdr_mesh",
                         lambda: lf.ui.tr("main_panel.mesh") or "Mesh")
        model.bind_func("label_ppisp_color_balance",
                         lambda: lf.ui.tr("main_panel.ppisp_color_balance") or "Color Correction")
        model.bind_func("label_ppisp_crf",
                         lambda: lf.ui.tr("main_panel.ppisp_crf_advanced") or "CRF")

        for sec in ["selection_colors", "mesh", "ppisp_crf"]:
            model.bind(f"sec_{sec}_collapsed",
                       lambda n=sec: n in self._collapsed)
            model.bind_func(f"sec_{sec}_arrow",
                            lambda n=sec: "\u25B6" if n in self._collapsed else "\u25BC")

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

        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("color_click", self._on_color_click)
        model.bind_event("chrom_change", self._on_chrom_change)
        model.bind_event("picker_change", self._on_picker_change)
        model.bind_event("toggle_console",
                         lambda h, e, a: lf.ui.toggle_system_console())

        self._handle = model.get_handle()

    def on_update(self, doc):
        s = lf.get_render_settings()
        if not s:
            return

        for prop_id in COLOR_PROPS:
            val = getattr(s, prop_id)
            key = (prop_id, int(val[0] * 255), int(val[1] * 255), int(val[2] * 255))
            if key == self._last_swatch_colors.get(prop_id):
                continue
            self._last_swatch_colors[prop_id] = key
            swatch = doc.get_element_by_id(f"swatch-{prop_id}")
            if swatch:
                swatch.set_property("background-color", f"rgb({key[1]},{key[2]},{key[3]})")

    def on_scene_changed(self, doc):
        if self._handle:
            self._handle.dirty_all()

    def on_unload(self, doc):
        doc.remove_data_model("rendering")
        self._handle = None
        self._popup_el = None
        self._doc = None

    def _set_color_hex(self, prop_id, hex_val):
        s = lf.get_render_settings()
        if not s:
            return
        color = _hex_to_color(hex_val)
        if color:
            setattr(s, prop_id, color)

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

    def _on_toggle_section(self, handle, event, args):
        if not args:
            return
        name = str(args[0])
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        handle.dirty(f"sec_{name}_collapsed")
        handle.dirty(f"sec_{name}_arrow")

    def _on_color_click(self, handle, event, args):
        if not args or not self._popup_el:
            return
        self._picker_click_handled = True
        prop_id = str(args[0])
        if self._color_edit_prop == prop_id:
            self._hide_picker()
            return
        self._color_edit_prop = prop_id
        mx = event.get_parameter("mouse_x", "0")
        my = event.get_parameter("mouse_y", "0")
        self._popup_el.set_property("left", f"{mx}px")
        self._popup_el.set_property("top", f"{int(float(my)) + 2}px")
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

