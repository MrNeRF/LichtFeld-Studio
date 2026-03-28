/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_rasterizer_tensor.hpp"
#include "core/logger.hpp"
#include "rasterization_api_tensor.h"
#include <algorithm>

namespace lfs::rendering {

    namespace {
        [[nodiscard]] int resolve_render_sh_degree(
            const lfs::core::SplatData& gaussian_model,
            const int requested_sh_degree) {
            const int fallback_sh_degree = gaussian_model.get_active_sh_degree();
            const int target_sh_degree = requested_sh_degree >= 0 ? requested_sh_degree : fallback_sh_degree;
            return std::clamp(target_sh_degree, 0, gaussian_model.get_max_sh_degree());
        }
    } // namespace

    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        const int sh_degree_override,
        bool show_rings,
        float ring_width,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const Tensor* selection_mask,
        Tensor* screen_positions_out,
        bool cursor_active,
        float cursor_x,
        float cursor_y,
        float cursor_radius,
        bool preview_selection_add_mode,
        Tensor* preview_selection_out,
        bool cursor_saturation_preview,
        float cursor_saturation_amount,
        bool show_center_markers,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        bool crop_inverse,
        bool crop_desaturate,
        int crop_parent_node_index,
        const Tensor* ellipsoid_transform,
        const Tensor* ellipsoid_radii,
        bool ellipsoid_inverse,
        bool ellipsoid_desaturate,
        int ellipsoid_parent_node_index,
        const Tensor* view_volume_transform,
        const Tensor* view_volume_min,
        const Tensor* view_volume_max,
        bool view_volume_cull,
        const Tensor* deleted_mask,
        unsigned long long* hovered_depth_id,
        int focused_gaussian_id,
        float far_plane,
        const std::vector<bool>& emphasized_node_mask,
        bool dim_non_emphasized,
        const std::vector<bool>& node_visibility_mask,
        float emphasis_flash_intensity,
        bool orthographic,
        float ortho_scale,
        bool mip_filter) {

        // Get camera parameters
        const float fx = viewpoint_camera.focal_x();
        const float fy = viewpoint_camera.focal_y();
        const float cx = viewpoint_camera.center_x();
        const float cy = viewpoint_camera.center_y();

        const int sh_degree = resolve_render_sh_degree(gaussian_model, sh_degree_override);
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float NEAR_PLANE = 0.01f;

        const auto& w2c = viewpoint_camera.world_view_transform();
        const auto& cam_pos = viewpoint_camera.cam_position();

        // Get model data
        const auto& means = gaussian_model.means_raw();
        const auto& scales_raw = gaussian_model.scaling_raw();
        const auto& rotations_raw = gaussian_model.rotation_raw();
        const auto& opacities_raw = gaussian_model.opacity_raw();
        const auto& sh0 = gaussian_model.sh0_raw();
        const auto& shN = gaussian_model.shN_raw();

        if (lfs::core::Logger::get().is_enabled(lfs::core::LogLevel::Debug)) {
            constexpr int DEBUG_LOG_INTERVAL = 300;
            static int frame_count = 0;
            if (frame_count++ % DEBUG_LOG_INTERVAL == 0) {
                const auto means_cpu = means.cpu();
                const auto scales_cpu = scales_raw.cpu();
                const auto opacities_cpu = opacities_raw.cpu();
                const auto cam_cpu = cam_pos.cpu();

                LOG_DEBUG("Rasterizer stats (frame {}): {} gaussians", frame_count, means.size(0));
                LOG_DEBUG("  means x:[{:.3f},{:.3f}] y:[{:.3f},{:.3f}] z:[{:.3f},{:.3f}]",
                          means_cpu.slice(1, 0, 1).min().item(), means_cpu.slice(1, 0, 1).max().item(),
                          means_cpu.slice(1, 1, 2).min().item(), means_cpu.slice(1, 1, 2).max().item(),
                          means_cpu.slice(1, 2, 3).min().item(), means_cpu.slice(1, 2, 3).max().item());
                LOG_DEBUG("  scales log:[{:.3f},{:.3f}] exp:[{:.6f},{:.3f}]",
                          scales_cpu.min().item(), scales_cpu.max().item(),
                          scales_cpu.exp().min().item(), scales_cpu.exp().max().item());
                LOG_DEBUG("  opacity logit:[{:.3f},{:.3f}] sigmoid:[{:.4f},{:.4f}]",
                          opacities_cpu.min().item(), opacities_cpu.max().item(),
                          opacities_cpu.sigmoid().min().item(), opacities_cpu.sigmoid().max().item());
                LOG_DEBUG("  camera:[{:.3f},{:.3f},{:.3f}]",
                          cam_cpu.ptr<float>()[0], cam_cpu.ptr<float>()[1], cam_cpu.ptr<float>()[2]);
            }
        }

        // Get deleted mask (use passed parameter or from model)
        const Tensor* actual_deleted_mask = deleted_mask;
        if (!actual_deleted_mask && gaussian_model.has_deleted_mask()) {
            actual_deleted_mask = &gaussian_model.deleted();
        }

        // Call the tensor-based forward wrapper
        auto [image, alpha, depth] = forward_wrapper_tensor(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh0,
            shN,
            w2c,
            cam_pos,
            active_sh_bases,
            viewpoint_camera.camera_width(),
            viewpoint_camera.camera_height(),
            fx,
            fy,
            cx,
            cy,
            NEAR_PLANE,
            far_plane,
            show_rings,
            ring_width,
            model_transforms,
            transform_indices,
            selection_mask,
            screen_positions_out,
            cursor_active,
            cursor_x,
            cursor_y,
            cursor_radius,
            preview_selection_add_mode,
            preview_selection_out,
            cursor_saturation_preview,
            cursor_saturation_amount,
            show_center_markers,
            crop_box_transform,
            crop_box_min,
            crop_box_max,
            crop_inverse,
            crop_desaturate,
            crop_parent_node_index,
            ellipsoid_transform,
            ellipsoid_radii,
            ellipsoid_inverse,
            ellipsoid_desaturate,
            ellipsoid_parent_node_index,
            view_volume_transform,
            view_volume_min,
            view_volume_max,
            view_volume_cull,
            actual_deleted_mask,
            hovered_depth_id,
            focused_gaussian_id,
            emphasized_node_mask,
            dim_non_emphasized,
            node_visibility_mask,
            emphasis_flash_intensity,
            orthographic,
            ortho_scale,
            mip_filter);

        // Blend background in-place: image += (1 - alpha) * bg
        Tensor bg = bg_color.unsqueeze(1).unsqueeze(2); // [3, 1, 1] view
        alpha.mul_(-1.0f).add_(1.0f);                   // alpha → (1 - alpha) in-place
        image.add_(alpha * bg);                         // 1 broadcast temp [3,H,W]
        image.clamp_(0.0f, 1.0f);

        LOG_TRACE("Tensor rasterization completed: {}x{}",
                  viewpoint_camera.camera_width(),
                  viewpoint_camera.camera_height());

        return {std::move(image), std::move(depth)};
    }

    DualRasterizeTensorOutput rasterize_tensor_pair(
        const std::array<lfs::core::Camera, 2>& viewpoint_cameras,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        const DualRasterizeTensorRequest& request) {

        const int sh_degree = resolve_render_sh_degree(gaussian_model, request.sh_degree_override);
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);
        constexpr float NEAR_PLANE = 0.01f;

        const auto& means = gaussian_model.means_raw();
        const auto& scales_raw = gaussian_model.scaling_raw();
        const auto& rotations_raw = gaussian_model.rotation_raw();
        const auto& opacities_raw = gaussian_model.opacity_raw();
        const auto& sh0 = gaussian_model.sh0_raw();
        const auto& shN = gaussian_model.shN_raw();

        const Tensor* actual_deleted_mask = request.deleted_mask;
        if (!actual_deleted_mask && gaussian_model.has_deleted_mask()) {
            actual_deleted_mask = &gaussian_model.deleted();
        }

        std::array<ForwardWrapperTensorViewState, 2> views;
        for (size_t i = 0; i < views.size(); ++i) {
            views[i] = ForwardWrapperTensorViewState{
                .w2c = viewpoint_cameras[i].world_view_transform(),
                .cam_position = viewpoint_cameras[i].cam_position(),
                .width = viewpoint_cameras[i].camera_width(),
                .height = viewpoint_cameras[i].camera_height(),
                .focal_x = viewpoint_cameras[i].focal_x(),
                .focal_y = viewpoint_cameras[i].focal_y(),
                .center_x = viewpoint_cameras[i].center_x(),
                .center_y = viewpoint_cameras[i].center_y(),
                .cursor_active = request.view_states[i].cursor_active,
                .cursor_x = request.view_states[i].cursor_x,
                .cursor_y = request.view_states[i].cursor_y,
                .cursor_radius = request.view_states[i].cursor_radius,
                .cursor_saturation_preview = request.view_states[i].cursor_saturation_preview,
                .cursor_saturation_amount = request.view_states[i].cursor_saturation_amount,
                .hovered_depth_id = request.view_states[i].hovered_depth_id,
                .focused_gaussian_id = request.view_states[i].focused_gaussian_id};
        }

        ForwardWrapperTensorSharedParams shared{
            .active_sh_bases = active_sh_bases,
            .near_plane = NEAR_PLANE,
            .far_plane = request.far_plane,
            .show_rings = request.show_rings,
            .ring_width = request.ring_width,
            .model_transforms = request.model_transforms,
            .transform_indices = request.transform_indices,
            .selection_mask = request.selection_mask,
            .preview_selection_add_mode = request.preview_selection_add_mode,
            .preview_selection_out = request.preview_selection_out,
            .show_center_markers = request.show_center_markers,
            .crop_box_transform = request.crop_box_transform,
            .crop_box_min = request.crop_box_min,
            .crop_box_max = request.crop_box_max,
            .crop_inverse = request.crop_inverse,
            .crop_desaturate = request.crop_desaturate,
            .crop_parent_node_index = request.crop_parent_node_index,
            .ellipsoid_transform = request.ellipsoid_transform,
            .ellipsoid_radii = request.ellipsoid_radii,
            .ellipsoid_inverse = request.ellipsoid_inverse,
            .ellipsoid_desaturate = request.ellipsoid_desaturate,
            .ellipsoid_parent_node_index = request.ellipsoid_parent_node_index,
            .view_volume_transform = request.view_volume_transform,
            .view_volume_min = request.view_volume_min,
            .view_volume_max = request.view_volume_max,
            .view_volume_cull = request.view_volume_cull,
            .deleted_mask = actual_deleted_mask,
            .emphasized_node_mask = &request.emphasized_node_mask,
            .dim_non_emphasized = request.dim_non_emphasized,
            .node_visibility_mask = &request.node_visibility_mask,
            .emphasis_flash_intensity = request.emphasis_flash_intensity,
            .orthographic = request.orthographic,
            .ortho_scale = request.ortho_scale,
            .mip_filter = request.mip_filter};

        auto outputs = forward_wrapper_tensor_dual(
            means,
            scales_raw,
            rotations_raw,
            opacities_raw,
            sh0,
            shN,
            views,
            shared);

        DualRasterizeTensorOutput result;
        Tensor bg = bg_color.unsqueeze(1).unsqueeze(2);
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i].alpha.mul_(-1.0f).add_(1.0f);
            outputs[i].image.add_(outputs[i].alpha * bg);
            outputs[i].image.clamp_(0.0f, 1.0f);
            result.images[i] = std::move(outputs[i].image);
            result.depths[i] = std::move(outputs[i].depth);
        }

        return result;
    }

    GutRenderOutput gut_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const Tensor& bg_color,
        const int sh_degree_override,
        const float scaling_modifier,
        const GutCameraModel camera_model,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const std::vector<bool>& node_visibility_mask) {

        const int width = camera.camera_width();
        const int height = camera.camera_height();
        const int sh_degree = resolve_render_sh_degree(model, sh_degree_override);

        const auto& w2c = camera.world_view_transform();

        // Equirectangular uses image dimensions in K; pinhole uses focal lengths
        const std::vector<float> K_data = (camera_model == GutCameraModel::EQUIRECTANGULAR)
                                              ? std::vector<float>{static_cast<float>(width), 0.0f, 0.0f,
                                                                   0.0f, static_cast<float>(height), 0.0f,
                                                                   0.0f, 0.0f, 1.0f}
                                              : std::vector<float>{camera.focal_x(), 0.0f, camera.center_x(),
                                                                   0.0f, camera.focal_y(), camera.center_y(),
                                                                   0.0f, 0.0f, 1.0f};
        const Tensor K = Tensor::from_vector(K_data, {3, 3}, lfs::core::Device::CPU).cuda();

        auto [image, alpha, depth] = forward_gut_tensor(
            model.means_raw(),
            model.scaling_raw(),
            model.rotation_raw(),
            model.opacity_raw(),
            model.sh0_raw(),
            model.shN_raw(),
            w2c, K,
            sh_degree, width, height,
            camera_model,
            nullptr, nullptr, &bg_color,
            model_transforms,
            transform_indices, node_visibility_mask);

        return {std::move(image), std::move(depth)};
    }

} // namespace lfs::rendering
