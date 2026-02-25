# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rendering panel - main tab for rendering settings."""

import math

import lichtfeld as lf

from .types import RmlPanel
from .sequencer_section import draw_sequencer_section

SENSOR_HALF_HEIGHT_MM = 12.0

BOOL_PROPS = [
    "show_coord_axes", "show_pivot", "show_grid", "show_camera_frustums",
    "point_cloud_mode", "desaturate_unselected", "desaturate_cropping",
    "equirectangular", "gut", "mip_filter",
    "mesh_wireframe", "mesh_backface_culling", "mesh_shadow_enabled",
]

SLIDER_PROPS = [
    "axes_size", "grid_opacity", "camera_frustum_scale", "voxel_size",
    "focal_length_mm", "render_scale",
    "mesh_wireframe_width", "mesh_light_intensity", "mesh_ambient",
]

SELECT_PROPS = [
    "grid_plane", "sh_degree", "mesh_shadow_resolution",
]

COLOR_PROPS = [
    "background_color",
    "selection_color_committed", "selection_color_preview",
    "selection_color_center_marker",
    "mesh_wireframe_color",
]

DEP_MAP = {
    "show_coord_axes": "dep-show_coord_axes",
    "show_grid": "dep-show_grid",
    "show_camera_frustums": "dep-show_camera_frustums",
    "point_cloud_mode": "dep-point_cloud_mode",
    "mesh_wireframe": "dep-mesh_wireframe",
    "mesh_shadow_enabled": "dep-mesh_shadow_enabled",
}


def _color_to_hex(c):
    r = int(c[0] * 255)
    g = int(c[1] * 255)
    b = int(c[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_color(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
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
        self._els = {}
        self._sections = {}
        self._hex_visible = set()
        self._slider_user_vals = {}
        self._color_edit_prop = None
        self._color_picker_needs_pos = False

    def on_load(self, doc):
        settings = lf.get_render_settings()
        if not settings:
            return

        tr = lf.ui.tr

        for prop_id in BOOL_PROPS:
            info = settings.prop_info(prop_id)
            text_el = doc.get_element_by_id(f"text-{prop_id}")
            if text_el:
                text_el.set_inner_rml(info.get("name", prop_id))

        for prop_id in SLIDER_PROPS:
            info = settings.prop_info(prop_id)
            label_el = doc.get_element_by_id(f"label-{prop_id}")
            if label_el:
                label_el.set_inner_rml(info.get("name", prop_id))

        for prop_id in SELECT_PROPS:
            info = settings.prop_info(prop_id)
            label_el = doc.get_element_by_id(f"label-{prop_id}")
            if label_el:
                label_el.set_inner_rml(info.get("name", prop_id))

        for prop_id in COLOR_PROPS:
            info = settings.prop_info(prop_id)
            label_el = doc.get_element_by_id(f"label-{prop_id}")
            if label_el:
                label_el.set_inner_rml(info.get("name", prop_id))

        hdr_sel = doc.get_element_by_id("text-hdr-selection_colors")
        if hdr_sel:
            hdr_sel.set_inner_rml(tr("main_panel.selection_colors"))

        hdr_mesh = doc.get_element_by_id("text-hdr-mesh")
        if hdr_mesh:
            hdr_mesh.set_inner_rml(tr("main_panel.mesh"))

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("change", self._on_change)
            body.add_event_listener("click", self._on_click)

    def on_update(self, doc):
        settings = lf.get_render_settings()
        if not settings:
            return

        for prop_id in BOOL_PROPS:
            el = doc.get_element_by_id(f"cb-{prop_id}")
            if el:
                val = getattr(settings, prop_id)
                if val:
                    el.set_attribute("checked", "")
                else:
                    el.remove_attribute("checked")

            if prop_id in DEP_MAP:
                dep = doc.get_element_by_id(DEP_MAP[prop_id])
                if dep:
                    val = getattr(settings, prop_id)
                    if val:
                        dep.set_class("hidden", False)
                    else:
                        dep.set_class("hidden", True)

        for prop_id in SLIDER_PROPS:
            val = getattr(settings, prop_id)
            user_val = self._slider_user_vals.get(prop_id)
            if user_val is None or abs(val - user_val) > 1e-6:
                el = doc.get_element_by_id(f"slider-{prop_id}")
                if el:
                    el.set_attribute("value", f"{val:.4g}")
                self._slider_user_vals.pop(prop_id, None)
            val_el = doc.get_element_by_id(f"val-{prop_id}")
            if val_el:
                val_el.set_inner_rml(f"{val:.3f}")

        for prop_id in SELECT_PROPS:
            el = doc.get_element_by_id(f"sel-{prop_id}")
            if el:
                val = getattr(settings, prop_id)
                el.set_attribute("value", str(val))

        for prop_id in COLOR_PROPS:
            val = getattr(settings, prop_id)
            r = int(val[0] * 255)
            g = int(val[1] * 255)
            b = int(val[2] * 255)

            r_el = doc.get_element_by_id(f"rc-{prop_id}")
            if r_el:
                r_el.set_inner_rml(f"R:{r:>3d}")
            g_el = doc.get_element_by_id(f"gc-{prop_id}")
            if g_el:
                g_el.set_inner_rml(f"G:{g:>3d}")
            b_el = doc.get_element_by_id(f"bc-{prop_id}")
            if b_el:
                b_el.set_inner_rml(f"B:{b:>3d}")

            swatch = doc.get_element_by_id(f"swatch-{prop_id}")
            if swatch:
                swatch.set_property("background-color", f"rgb({r},{g},{b})")

            hex_el = doc.get_element_by_id(f"hex-{prop_id}")
            if hex_el:
                hex_el.set_attribute("value", _color_to_hex(val))

        view = lf.get_current_view()
        fov_el = doc.get_element_by_id("fov-display")
        if fov_el:
            if view and view.width > 0 and view.height > 0:
                focal_mm = settings.focal_length_mm
                vfov = 2.0 * math.degrees(math.atan(SENSOR_HALF_HEIGHT_MM / focal_mm))
                aspect = view.width / view.height
                hfov = 2.0 * math.degrees(
                    math.atan(aspect * math.tan(math.radians(vfov * 0.5)))
                )
                fov_el.set_inner_rml(
                    lf.ui.tr("rendering_panel.fov_format").format(hfov=hfov, vfov=vfov)
                )
            else:
                fov_el.set_inner_rml("")

    def draw_imgui(self, layout):
        settings = lf.get_render_settings()
        if not settings:
            return

        tr = lf.ui.tr

        if self._color_picker_needs_pos:
            layout.set_next_window_pos(layout.get_mouse_pos())
            layout.open_popup("##color_picker")
            self._color_picker_needs_pos = False

        if self._color_edit_prop and layout.begin_popup("##color_picker"):
            prop_id = self._color_edit_prop
            val = getattr(settings, prop_id)
            changed, new_color = layout.color_picker3("##picker", val)
            if changed:
                setattr(settings, prop_id, new_color)
            layout.end_popup()
        elif self._color_edit_prop:
            self._color_edit_prop = None

        layout.prop(settings, "apply_appearance_correction")
        if settings.apply_appearance_correction:
            layout.indent()
            layout.prop(settings, "ppisp_mode")

            is_manual = settings.ppisp_mode == "MANUAL"
            if not is_manual:
                layout.begin_disabled()

            layout.prop(settings, "ppisp_exposure")

            layout.prop(settings, "ppisp_vignette_enabled")
            if settings.ppisp_vignette_enabled:
                layout.same_line()
                layout.prop(settings, "ppisp_vignette_strength")

            changed, values = layout.chromaticity_diagram(
                tr("main_panel.ppisp_color_balance"),
                settings.ppisp_color_red_x,
                settings.ppisp_color_red_y,
                settings.ppisp_color_green_x,
                settings.ppisp_color_green_y,
                settings.ppisp_color_blue_x,
                settings.ppisp_color_blue_y,
                settings.ppisp_wb_temperature,
                settings.ppisp_wb_tint,
            )
            if changed:
                settings.ppisp_color_red_x = values[0]
                settings.ppisp_color_red_y = values[1]
                settings.ppisp_color_green_x = values[2]
                settings.ppisp_color_green_y = values[3]
                settings.ppisp_color_blue_x = values[4]
                settings.ppisp_color_blue_y = values[5]
                settings.ppisp_wb_temperature = values[6]
                settings.ppisp_wb_tint = values[7]

            layout.prop(settings, "ppisp_gamma_multiplier")

            if layout.collapsing_header(tr("main_panel.ppisp_crf_advanced")):
                layout.crf_curve_preview(
                    "##crf_preview",
                    settings.ppisp_gamma_multiplier,
                    settings.ppisp_crf_toe,
                    settings.ppisp_crf_shoulder,
                    settings.ppisp_gamma_red,
                    settings.ppisp_gamma_green,
                    settings.ppisp_gamma_blue,
                )
                layout.prop(settings, "ppisp_gamma_red")
                layout.prop(settings, "ppisp_gamma_green")
                layout.prop(settings, "ppisp_gamma_blue")
                layout.prop(settings, "ppisp_crf_toe")
                layout.prop(settings, "ppisp_crf_shoulder")

            if not is_manual:
                layout.end_disabled()
            layout.unindent()

        if lf.ui.is_sequencer_visible():
            layout.separator()
            draw_sequencer_section(layout)

        layout.separator()
        lf.ui.invoke_hooks("rendering", "selection_groups", True)
        lf.ui.invoke_hooks("rendering", "selection_groups", False)

        layout.separator()
        lf.ui.draw_tools_section()
        lf.ui.draw_console_button()

    def _on_change(self, event):
        settings = lf.get_render_settings()
        if not settings:
            return

        target = event.target()
        if not target:
            return

        prop_id = target.get_attribute("data-prop")
        if not prop_id:
            return

        tag = target.get_attribute("id") or ""

        if tag.startswith("cb-"):
            val = target.has_attribute("checked")
            setattr(settings, prop_id, val)
        elif tag.startswith("slider-"):
            try:
                val = float(target.get_attribute("value"))
                setattr(settings, prop_id, val)
                self._slider_user_vals[prop_id] = val
            except (ValueError, TypeError):
                pass
        elif tag.startswith("sel-"):
            val = target.get_attribute("value")
            if val is not None:
                setattr(settings, prop_id, val)
        elif tag.startswith("hex-"):
            hex_val = target.get_attribute("value")
            if hex_val:
                color = _hex_to_color(hex_val)
                if color:
                    setattr(settings, prop_id, color)

    def _on_click(self, event):
        target = event.target()
        if not target:
            return

        body = event.current_target()

        el = target
        while el:
            cls = el.get_class_names()

            if "section-header" in cls:
                section_name = el.get_attribute("data-section")
                if section_name and body:
                    sec_el = body.get_element_by_id(f"sec-{section_name}")
                    if sec_el:
                        is_collapsed = sec_el.is_class_set("collapsed")
                        sec_el.set_class("collapsed", not is_collapsed)
                        arrow = el.query_selector(".section-arrow")
                        if arrow:
                            arrow.set_inner_rml(
                                "\u25BC" if is_collapsed else "\u25B6"
                            )
                return

            if "color-swatch" in cls:
                prop_id = el.get_attribute("data-prop")
                if prop_id:
                    if self._color_edit_prop == prop_id:
                        self._color_edit_prop = None
                    else:
                        self._color_edit_prop = prop_id
                        self._color_picker_needs_pos = True
                return

            el = el.parent()
