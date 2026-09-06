/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace lfs::rendering {

    using lfs::core::Tensor;

    struct SelectionGroupCountResult {
        std::array<size_t, 256> group_counts{};
        size_t changed_count = 0;
    };

    struct SelectionGroupDeltaResult {
        std::array<int32_t, 256> group_deltas{};
        size_t changed_count = 0;
    };

    // Screen positions below this on either axis are the projection's invalid marker.
    constexpr float kInvalidScreenPositionThreshold = -1000.0f;

    enum class ScreenWindowCameraModel : std::uint32_t {
        Pinhole = 0,
        Orthographic = 1,
        Equirectangular = 3,
    };

    void brush_select(
        const float2* screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        uint8_t* selection_out,
        int n_primitives);

    void polygon_select(
        const float2* positions,
        const float2* polygon,
        int num_vertices,
        bool* selection,
        int n_primitives);

    void set_selection_element(bool* selection, int index, bool value);
    [[nodiscard]] Tensor project_screen_positions_tensor(
        const Tensor& means,
        int width,
        int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        float pixel_focal_x,
        float pixel_focal_y,
        ScreenWindowCameraModel camera_model,
        float ortho_scale);
    [[nodiscard]] Tensor project_screen_positions_tensor(
        const Tensor& means,
        int width,
        int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        float pixel_focal_x,
        float pixel_focal_y,
        ScreenWindowCameraModel camera_model,
        float ortho_scale,
        const Tensor* model_transforms);
    [[nodiscard]] Tensor project_screen_positions_tensor(
        const Tensor& means,
        int width,
        int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        float pixel_focal_x,
        float pixel_focal_y,
        ScreenWindowCameraModel camera_model,
        float ortho_scale,
        const Tensor* model_transforms,
        const Tensor* transform_indices);
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
        ScreenWindowCameraModel camera_model,
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

    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const uint32_t* locked_groups,
        bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        bool replace_mode = false,
        Tensor* group_counts_scratch = nullptr);

    void apply_selection_group_indexed_tensor_mask(
        const Tensor& visible_selection,
        const Tensor& visible_indices,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const uint32_t* locked_groups,
        bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        bool replace_mode = false);

    [[nodiscard]] std::array<size_t, 256> count_selection_groups(
        const Tensor& selection_mask,
        Tensor& counts_scratch);
    // Queue the group histogram and, optionally, a pinned-host copy. Neither
    // function waits for the CUDA stream; callers that need the values can
    // poll/synchronize their own event later.
    void count_selection_groups_async(const Tensor& selection_mask,
                                      Tensor& counts_scratch);
    void enqueue_selection_group_count_read(const Tensor& counts_scratch,
                                            int* pinned_host_counts,
                                            cudaEvent_t ready_event);
    [[nodiscard]] SelectionGroupCountResult read_selection_group_count_result(
        const Tensor& counts_scratch);
    [[nodiscard]] SelectionGroupDeltaResult read_selection_group_delta_result(
        const Tensor& counts_scratch);
    [[nodiscard]] std::array<size_t, 256> read_selection_group_counts(
        const Tensor& counts_scratch);

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
        ScreenWindowCameraModel camera_model,
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

    namespace config {
        void setSelectionGroupColor(int group_id, float3 color);
        void setSelectionPreviewColor(float3 color);
    } // namespace config

} // namespace lfs::rendering
