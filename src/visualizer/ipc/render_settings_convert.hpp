/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "ipc/view_context.hpp"
#include "rendering/rendering_types.hpp"

namespace lfs::vis {

    namespace detail {
        inline std::array<float, 3> to_array(const glm::vec3& v) { return {v.x, v.y, v.z}; }
        inline glm::vec3 to_vec3(const std::array<float, 3>& a) { return {a[0], a[1], a[2]}; }
        inline std::array<float, 4> to_array(const glm::quat& q) { return {q.w, q.x, q.y, q.z}; }
        inline glm::quat to_quat(const std::array<float, 4>& a) { return {a[0], a[1], a[2], a[3]}; }
    } // namespace detail

    inline RenderSettingsProxy to_proxy(const RenderSettings& s) {
        RenderSettingsProxy p;
        p.focal_length_mm = s.focal_length_mm;
        p.scaling_modifier = s.scaling_modifier;
        p.antialiasing = s.antialiasing;
        p.mip_filter = s.mip_filter;
        p.sh_degree = s.sh_degree;
        p.render_scale = s.render_scale;
        p.camera_metrics_mode = static_cast<int>(s.camera_metrics_mode);
        p.show_crop_box = s.show_crop_box;
        p.use_crop_box = s.use_crop_box;
        p.show_ellipsoid = s.show_ellipsoid;
        p.use_ellipsoid = s.use_ellipsoid;
        p.desaturate_unselected = s.desaturate_unselected;
        p.desaturate_cropping = s.desaturate_cropping;
        p.hide_outside_depth_box = s.hide_outside_depth_box;
        p.crop_filter_for_selection = s.crop_filter_for_selection;
        p.apply_appearance_correction = s.apply_appearance_correction;
        p.ppisp_mode = static_cast<int>(s.ppisp_mode);
        p.ppisp = s.ppisp_overrides;
        p.background_color = detail::to_array(s.background_color);
        p.environment_mode = static_cast<int>(s.environment_mode);
        p.environment_map_path = s.environment_map_path;
        p.environment_exposure = s.environment_exposure;
        p.environment_rotation_degrees = s.environment_rotation_degrees;
        p.show_coord_axes = s.show_coord_axes;
        p.axes_size = s.axes_size;
        p.axes_visibility = s.axes_visibility;
        p.show_grid = s.show_grid;
        p.grid_plane = s.grid_plane;
        p.grid_opacity = s.grid_opacity;
        p.point_cloud_mode = s.point_cloud_mode;
        p.voxel_size = s.voxel_size;
        p.show_rings = s.show_rings;
        p.ring_width = s.ring_width;
        p.show_center_markers = s.show_center_markers;
        p.show_camera_frustums = s.show_camera_frustums;
        p.camera_frustum_scale = s.camera_frustum_scale;
        p.train_camera_color = detail::to_array(s.train_camera_color);
        p.eval_camera_color = detail::to_array(s.eval_camera_color);
        p.show_pivot = s.show_pivot;
        p.split_view_mode = static_cast<int>(s.split_view_mode);
        p.split_position = s.split_position;
        p.raster_backend = static_cast<int>(s.raster_backend);
        p.gut = s.raster_backend == lfs::rendering::GaussianRasterBackend::Gut;
        p.equirectangular = s.equirectangular;
        p.orthographic = s.orthographic;
        p.ortho_scale = s.ortho_scale;
        p.selection_color_committed = detail::to_array(s.selection_color_committed);
        p.selection_color_preview = detail::to_array(s.selection_color_preview);
        p.selection_color_center_marker = detail::to_array(s.selection_color_center_marker);
        p.depth_clip_enabled = s.depth_clip_enabled;
        p.depth_clip_far = s.depth_clip_far;
        p.mesh_wireframe = s.mesh_wireframe;
        p.mesh_wireframe_color = detail::to_array(s.mesh_wireframe_color);
        p.mesh_wireframe_width = s.mesh_wireframe_width;
        p.mesh_light_dir = detail::to_array(s.mesh_light_dir);
        p.mesh_light_intensity = s.mesh_light_intensity;
        p.mesh_ambient = s.mesh_ambient;
        p.mesh_backface_culling = s.mesh_backface_culling;
        p.mesh_shadow_enabled = s.mesh_shadow_enabled;
        p.mesh_shadow_resolution = s.mesh_shadow_resolution;
        p.depth_filter_enabled = s.depth_filter_enabled;
        p.depth_filter_min = detail::to_array(s.depth_filter_min);
        p.depth_filter_max = detail::to_array(s.depth_filter_max);
        p.depth_filter_rotation = detail::to_array(s.depth_filter_transform.getRotation());
        p.depth_filter_translation = detail::to_array(s.depth_filter_transform.getTranslation());
        return p;
    }

    inline void apply_proxy(RenderSettings& s, const RenderSettingsProxy& p) {
        s.focal_length_mm = p.focal_length_mm;
        s.scaling_modifier = p.scaling_modifier;
        s.antialiasing = p.antialiasing;
        s.mip_filter = p.mip_filter;
        s.sh_degree = p.sh_degree;
        s.render_scale = p.render_scale;
        s.camera_metrics_mode = static_cast<RenderSettings::CameraMetricsMode>(p.camera_metrics_mode);
        s.show_crop_box = p.show_crop_box;
        s.use_crop_box = p.use_crop_box;
        s.show_ellipsoid = p.show_ellipsoid;
        s.use_ellipsoid = p.use_ellipsoid;
        s.desaturate_unselected = p.desaturate_unselected;
        s.desaturate_cropping = p.desaturate_cropping;
        s.hide_outside_depth_box = p.hide_outside_depth_box;
        s.crop_filter_for_selection = p.crop_filter_for_selection;
        s.apply_appearance_correction = p.apply_appearance_correction;
        s.ppisp_mode = static_cast<RenderSettings::PPISPMode>(p.ppisp_mode);
        s.ppisp_overrides = p.ppisp;
        s.background_color = detail::to_vec3(p.background_color);
        s.environment_mode = static_cast<EnvironmentBackgroundMode>(p.environment_mode);
        s.environment_map_path = p.environment_map_path;
        s.environment_exposure = p.environment_exposure;
        s.environment_rotation_degrees = p.environment_rotation_degrees;
        s.show_coord_axes = p.show_coord_axes;
        s.axes_size = p.axes_size;
        s.axes_visibility = p.axes_visibility;
        s.show_grid = p.show_grid;
        s.grid_plane = p.grid_plane;
        s.grid_opacity = p.grid_opacity;
        s.point_cloud_mode = p.point_cloud_mode;
        s.voxel_size = p.voxel_size;
        s.show_rings = p.show_rings;
        s.ring_width = p.ring_width;
        s.show_center_markers = p.show_center_markers;
        s.show_camera_frustums = p.show_camera_frustums;
        s.camera_frustum_scale = p.camera_frustum_scale;
        s.train_camera_color = detail::to_vec3(p.train_camera_color);
        s.eval_camera_color = detail::to_vec3(p.eval_camera_color);
        s.show_pivot = p.show_pivot;
        s.split_view_mode = static_cast<SplitViewMode>(p.split_view_mode);
        s.split_position = p.split_position;
        s.raster_backend = static_cast<lfs::rendering::GaussianRasterBackend>(p.raster_backend);
        if (p.gut && s.raster_backend == lfs::rendering::GaussianRasterBackend::FastGs) {
            s.raster_backend = lfs::rendering::GaussianRasterBackend::Gut;
        }
        s.gut = s.raster_backend == lfs::rendering::GaussianRasterBackend::Gut;
        s.equirectangular = p.equirectangular;
        s.orthographic = p.orthographic;
        s.ortho_scale = p.ortho_scale;
        s.selection_color_committed = detail::to_vec3(p.selection_color_committed);
        s.selection_color_preview = detail::to_vec3(p.selection_color_preview);
        s.selection_color_center_marker = detail::to_vec3(p.selection_color_center_marker);
        s.depth_clip_enabled = p.depth_clip_enabled;
        s.depth_clip_far = p.depth_clip_far;
        s.mesh_wireframe = p.mesh_wireframe;
        s.mesh_wireframe_color = detail::to_vec3(p.mesh_wireframe_color);
        s.mesh_wireframe_width = p.mesh_wireframe_width;
        s.mesh_light_dir = detail::to_vec3(p.mesh_light_dir);
        s.mesh_light_intensity = p.mesh_light_intensity;
        s.mesh_ambient = p.mesh_ambient;
        s.mesh_backface_culling = p.mesh_backface_culling;
        s.mesh_shadow_enabled = p.mesh_shadow_enabled;
        s.mesh_shadow_resolution = p.mesh_shadow_resolution;
        s.depth_filter_enabled = p.depth_filter_enabled;
        s.depth_filter_min = detail::to_vec3(p.depth_filter_min);
        s.depth_filter_max = detail::to_vec3(p.depth_filter_max);
        s.depth_filter_transform =
            lfs::geometry::EuclideanTransform(detail::to_quat(p.depth_filter_rotation),
                                              detail::to_vec3(p.depth_filter_translation));
    }

} // namespace lfs::vis
