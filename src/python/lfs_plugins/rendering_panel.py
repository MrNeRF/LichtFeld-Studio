# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rendering panel - main tab for rendering settings."""

import math

import lichtfeld as lf

from .types import Panel
from .sequencer_section import draw_sequencer_section

SENSOR_HALF_HEIGHT_MM = 12.0


class RenderingPanel(Panel):
    idname = "lfs.rendering"
    label = "Rendering"
    space = "MAIN_PANEL_TAB"
    order = 10

    def draw(self, layout):
        self._draw_settings(layout)

        if lf.ui.is_sequencer_visible():
            layout.separator()
            draw_sequencer_section(layout)

        layout.separator()
        lf.ui.invoke_hooks("rendering", "selection_groups", True)
        lf.ui.invoke_hooks("rendering", "selection_groups", False)

        layout.separator()
        lf.ui.draw_tools_section()
        lf.ui.draw_console_button()

    def _draw_settings(self, layout):
        tr = lf.ui.tr
        settings = lf.get_render_settings()
        if not settings:
            return

        layout.prop(settings, "background_color")

        layout.prop(settings, "show_coord_axes")
        if settings.show_coord_axes:
            layout.indent()
            layout.prop(settings, "axes_size")
            layout.unindent()

        layout.prop(settings, "show_pivot")

        layout.prop(settings, "show_grid")
        if settings.show_grid:
            layout.indent()
            layout.prop(settings, "grid_plane")
            layout.prop(settings, "grid_opacity")
            layout.unindent()

        layout.prop(settings, "show_camera_frustums")
        if settings.show_camera_frustums:
            layout.indent()
            layout.prop(settings, "camera_frustum_scale")
            layout.unindent()

        layout.prop(settings, "point_cloud_mode")
        if settings.point_cloud_mode:
            layout.indent()
            layout.prop(settings, "voxel_size")
            layout.unindent()

        layout.separator()
        if layout.collapsing_header(tr("main_panel.selection_colors")):
            layout.prop(settings, "selection_color_committed")
            layout.prop(settings, "selection_color_preview")
            layout.prop(settings, "selection_color_center_marker")

        layout.prop(settings, "desaturate_unselected")
        layout.prop(settings, "desaturate_cropping")

        layout.separator()
        layout.prop(settings, "focal_length_mm")
        view = lf.get_current_view()
        if view and view.width > 0 and view.height > 0:
            focal_mm = settings.focal_length_mm
            vfov = 2.0 * math.degrees(math.atan(SENSOR_HALF_HEIGHT_MM / focal_mm))
            aspect = view.width / view.height
            hfov = 2.0 * math.degrees(math.atan(aspect * math.tan(math.radians(vfov * 0.5))))
            layout.text_disabled(tr("rendering_panel.fov_format").format(hfov=hfov, vfov=vfov))
        layout.prop(settings, "sh_degree")
        layout.prop(settings, "equirectangular")
        layout.prop(settings, "gut")

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

        layout.prop(settings, "mip_filter")
        layout.prop(settings, "render_scale")

        layout.separator()
        if layout.collapsing_header(tr("main_panel.mesh")):
            layout.prop(settings, "mesh_wireframe")
            if settings.mesh_wireframe:
                layout.indent()
                layout.prop(settings, "mesh_wireframe_color")
                layout.prop(settings, "mesh_wireframe_width")
                layout.unindent()
            layout.prop(settings, "mesh_light_intensity")
            layout.prop(settings, "mesh_ambient")
            layout.prop(settings, "mesh_backface_culling")
