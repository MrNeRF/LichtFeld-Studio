/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "gsplat_forward.h"
#include <tuple>
#include <vector>

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    /**
     * @brief Forward rasterization with custom Tensor types (libtorch-free)
     *
     * @param means Gaussian means [N, 3]
     * @param scales_raw Gaussian scales in log-space [N, 3]
     * @param rotations_raw Gaussian rotations (unnormalized quaternions) [N, 4]
     * @param opacities_raw Gaussian opacities in logit-space [N, 1]
     * @param sh_coefficients_0 SH coefficients degree 0 [N, 1, 3]
     * @param sh_coefficients_rest SH coefficients degree 1+ [N, (degree+1)²-1, 3]
     * @param w2c World-to-camera transform matrix [4, 4]
     * @param cam_position Camera position in world space [3]
     * @param active_sh_bases Number of active SH bases (degree+1)²
     * @param width Image width in pixels
     * @param height Image height in pixels
     * @param focal_x Focal length in x direction
     * @param focal_y Focal length in y direction
     * @param center_x Principal point x coordinate
     * @param center_y Principal point y coordinate
     * @param near_plane Near clipping plane
     * @param far_plane Far clipping plane
     * @param show_rings Enable ring mode visualization
     * @param ring_width Width of the ring band
     * @param model_transforms Array of 4x4 transforms [num_transforms, 4, 4]
     * @param transform_indices Per-Gaussian transform index [N]
     * @param selection_mask Per-Gaussian selection mask [N] (uint8, 1=selected/yellow)
     * @param screen_positions_out Optional output: screen positions [N, 2] for brush tool
     *
     * @return Tuple of (rendered_image [3, H, W], alpha_map [1, H, W], depth_map [1, H, W])
     */
    std::tuple<Tensor, Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool show_rings = false,
        const float ring_width = 0.01f,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const Tensor* selection_mask = nullptr,
        Tensor* screen_positions_out = nullptr,
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        Tensor* brush_selection_out = nullptr,
        bool brush_saturation_mode = false,
        float brush_saturation_amount = 0.0f,
        bool selection_mode_rings = false,
        bool show_center_markers = false,
        const Tensor* crop_box_transform = nullptr,
        const Tensor* crop_box_min = nullptr,
        const Tensor* crop_box_max = nullptr,
        bool crop_inverse = false,
        bool crop_desaturate = false,
        const Tensor* depth_filter_transform = nullptr,
        const Tensor* depth_filter_min = nullptr,
        const Tensor* depth_filter_max = nullptr,
        const Tensor* deleted_mask = nullptr,
        unsigned long long* hovered_depth_id = nullptr,
        int highlight_gaussian_id = -1,
        const std::vector<bool>& selected_node_mask = {},
        bool desaturate_unselected = false,
        float selection_flash_intensity = 0.0f,
        bool orthographic = false,
        float ortho_scale = 1.0f,
        bool mip_filter = false);

    /**
     * @brief Select Gaussians within brush radius using GPU
     *
     * @param screen_positions Screen positions [N, 2] from render
     * @param mouse_x Mouse X in image coords
     * @param mouse_y Mouse Y in image coords
     * @param radius Brush radius in pixels
     * @param selection_out Output selection mask [N] uint8
     */
    void brush_select_tensor(
        const Tensor& screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        Tensor& selection_out);

    // Select Gaussians inside rectangle on GPU (sets true for points inside)
    void rect_select_tensor(
        const Tensor& screen_positions,
        float x0, float y0, float x1, float y1,
        Tensor& selection_out);

    // Select Gaussians inside rectangle with add/remove mode
    void rect_select_mode_tensor(
        const Tensor& screen_positions,
        float x0, float y0, float x1, float y1,
        Tensor& selection_out,
        bool add_mode);

    // Select Gaussians inside polygon on GPU
    void polygon_select_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out);

    // Select Gaussians inside polygon with add/remove mode
    void polygon_select_mode_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out,
        bool add_mode);

    /**
     * @brief Apply selection to group mask on GPU
     *
     * @param cumulative_selection Bool mask of selected gaussians [N]
     * @param existing_mask Current group mask [N] (uint8, 0=unselected, 1-255=group)
     * @param output_mask Output group mask [N] (uint8)
     * @param group_id Group ID to assign (1-255)
     * @param locked_groups Bitmask of locked groups (256 bits = 32 bytes)
     * @param add_mode If true, add to group; if false, remove from group
     * @param transform_indices Per-gaussian node index [N] (int32), nullptr to skip filtering
     * @param target_node_index Only apply to gaussians with this node index, -1 to apply to all
     */
    void apply_selection_group_tensor(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const uint32_t* locked_groups,
        bool add_mode,
        const Tensor* transform_indices = nullptr,
        int target_node_index = -1);

    // Apply selection with node mask (supports groups)
    // replace_mode: if true, clears active group first, then applies new selection
    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const uint32_t* locked_groups,
        bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        bool replace_mode = false);

    // Filter selection by single node index
    void filter_selection_by_node(
        Tensor& selection,
        const Tensor& transform_indices,
        int target_node_index);

    // Filter selection by node mask (supports groups)
    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes);

    // GUT forward rasterization - returns (image [3,H,W], alpha [1,H,W], depth [1,H,W])
    std::tuple<Tensor, Tensor, Tensor>
    forward_gut_tensor(
        const Tensor& means,         // [N, 3]
        const Tensor& scales_raw,    // [N, 3] log-space
        const Tensor& rotations_raw, // [N, 4] unnormalized
        const Tensor& opacities_raw, // [N, 1] logit-space
        const Tensor& sh0,           // [N, 1, 3]
        const Tensor& sh_rest,       // [N, K-1, 3]
        const Tensor& w2c,           // [4, 4]
        const Tensor& K,             // [3, 3]
        int sh_degree,
        int width,
        int height,
        GutCameraModel camera_model = GutCameraModel::PINHOLE,
        const Tensor* radial_coeffs = nullptr,
        const Tensor* tangential_coeffs = nullptr,
        const Tensor* background = nullptr);

} // namespace lfs::rendering
