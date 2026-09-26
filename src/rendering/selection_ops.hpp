/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "core/tensor_spatial.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfs::rendering {

    using lfs::core::Tensor;

    // Screen positions below this on either axis are the projection's invalid marker.
    constexpr float kInvalidScreenPositionThreshold = -1000.0f;

    void set_selection_element(Tensor& selection, int index, bool value);

    [[nodiscard]] Tensor project_screen_positions_tensor(
        const Tensor& means,
        int width,
        int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        float pixel_focal_x,
        float pixel_focal_y,
        float center_x,
        float center_y,
        core::PointProjectionModel camera_model,
        float ortho_scale,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const std::vector<bool>& node_visibility_mask);
    [[nodiscard]] int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        float x,
        float y,
        float radius);

    void brush_select_tensor(
        const Tensor& screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        Tensor& selection_out);
    // Union of disks. Iterates samples; does not allocate an N x stroke matrix.
    void brush_select_disks_tensor(
        const Tensor& screen_positions,
        const std::vector<float>& disk_xy,
        float radius,
        Tensor& selection_out);

    void rect_select_tensor(
        const Tensor& screen_positions,
        float x0,
        float y0,
        float x1,
        float y1,
        Tensor& selection_out);

    void polygon_select_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out);

    // Lock flags are a dense Bool tensor with one entry per group (256), on
    // the selection backend. An invalid tensor means that no group is locked.
    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const Tensor& locked_groups,
        bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        bool replace_mode = false);

    void apply_selection_group_indexed_tensor_mask(
        const Tensor& visible_selection,
        const Tensor& visible_indices,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const Tensor& locked_groups,
        bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        bool replace_mode = false);

    void count_selection_groups_async(const Tensor& selection_mask,
                                      Tensor& counts_scratch);
    void merge_selection_mask_or(Tensor& accumulated_mask, const Tensor& delta_mask);

    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes);

    void filter_selection_by_crop(
        Tensor& selection,
        const Tensor& means,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        bool crop_inverse,
        const Tensor* ellipsoid_transform,
        const Tensor* ellipsoid_radii,
        bool ellipsoid_inverse,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr);

    void filter_selection_by_screen_window(
        Tensor& selection,
        const Tensor& means,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        core::PointProjectionModel camera_model,
        int width,
        int height,
        float pixel_focal_x,
        float pixel_focal_y,
        float center_x,
        float center_y,
        float ortho_scale,
        float near_depth,
        float far_depth,
        float scale_x,
        float scale_y,
        float offset_x,
        float offset_y,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr);

} // namespace lfs::rendering
