/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gs_rasterizer_tensor.hpp"
#include "core/logger.hpp"
#include "rasterization_api_tensor.h"
#include <glm/glm.hpp>

namespace lfs::rendering {

    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        bool show_rings,
        float ring_width,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const Tensor* selection_mask,
        Tensor* screen_positions_out,
        bool brush_active,
        float brush_x,
        float brush_y,
        float brush_radius,
        bool brush_add_mode,
        Tensor* brush_selection_out,
        bool brush_saturation_mode,
        float brush_saturation_amount,
        bool selection_mode_rings,
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
        const Tensor* depth_filter_transform,
        const Tensor* depth_filter_min,
        const Tensor* depth_filter_max,
        const Tensor* deleted_mask,
        unsigned long long* hovered_depth_id,
        int highlight_gaussian_id,
        float far_plane,
        const std::vector<bool>& selected_node_mask,
        bool desaturate_unselected,
        const std::vector<bool>& node_visibility_mask,
        float selection_flash_intensity,
        bool orthographic,
        float ortho_scale,
        bool mip_filter) {

        // Get camera parameters
        const float fx = viewpoint_camera.focal_x();
        const float fy = viewpoint_camera.focal_y();
        const float cx = viewpoint_camera.center_x();
        const float cy = viewpoint_camera.center_y();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float NEAR_PLANE = 0.01f;

        // Build world-to-camera transform matrix [4, 4]
        // w2c = [R | t]
        //       [0 | 1]
        const auto& R = viewpoint_camera.R(); // [3, 3] rotation
        const auto& T = viewpoint_camera.T(); // [3] translation

        // Create w2c matrix [4, 4] on CPU
        std::vector<float> w2c_data(16, 0.0f);

        // Copy rotation (first 3x3)
        auto R_cpu = R.cpu();
        const float* R_ptr = R_cpu.ptr<float>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                w2c_data[i * 4 + j] = R_ptr[i * 3 + j];
            }
        }

        // Copy translation (last column of first 3 rows)
        auto T_cpu = T.cpu();
        const float* T_ptr = T_cpu.ptr<float>();
        w2c_data[0 * 4 + 3] = T_ptr[0];
        w2c_data[1 * 4 + 3] = T_ptr[1];
        w2c_data[2 * 4 + 3] = T_ptr[2];

        // Set last row [0, 0, 0, 1]
        w2c_data[3 * 4 + 3] = 1.0f;

        Tensor w2c = Tensor::from_vector(w2c_data, {4, 4}, lfs::core::Device::CPU).cuda();

        // Camera position is -R^T @ t
        // We can compute this from the existing R and T
        auto R_t = R.transpose(0, 1);                 // R^T
        auto T_expanded = T.unsqueeze(1);             // [3, 1]
        auto cam_pos = -R_t.mm(T_expanded).squeeze(); // [3]

        // Get model data
        const auto& means = gaussian_model.means_raw();
        const auto& scales_raw = gaussian_model.scaling_raw();
        const auto& rotations_raw = gaussian_model.rotation_raw();
        const auto& opacities_raw = gaussian_model.opacity_raw();
        const auto& sh0 = gaussian_model.sh0_raw();
        const auto& shN = gaussian_model.shN_raw();

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
            brush_active,
            brush_x,
            brush_y,
            brush_radius,
            brush_add_mode,
            brush_selection_out,
            brush_saturation_mode,
            brush_saturation_amount,
            selection_mode_rings,
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
            depth_filter_transform,
            depth_filter_min,
            depth_filter_max,
            actual_deleted_mask,
            hovered_depth_id,
            highlight_gaussian_id,
            selected_node_mask,
            desaturate_unselected,
            node_visibility_mask,
            selection_flash_intensity,
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

    GutRenderOutput gut_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const Tensor& bg_color,
        const float scaling_modifier,
        const GutCameraModel camera_model,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const std::vector<bool>& node_visibility_mask) {

        const int width = camera.camera_width();
        const int height = camera.camera_height();
        const int sh_degree = model.get_active_sh_degree();

        // Build w2c matrix [4,4] = [R|t; 0|1]
        const auto R_cpu = camera.R().cpu();
        const auto T_cpu = camera.T().cpu();
        const float* R_ptr = R_cpu.ptr<float>();
        const float* T_ptr = T_cpu.ptr<float>();

        std::vector<float> w2c_data(16, 0.0f);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                w2c_data[i * 4 + j] = R_ptr[i * 3 + j];
            }
            w2c_data[i * 4 + 3] = T_ptr[i];
        }
        w2c_data[15] = 1.0f;
        const Tensor w2c = Tensor::from_vector(w2c_data, {4, 4}, lfs::core::Device::CPU).cuda();

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
