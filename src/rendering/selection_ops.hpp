/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace lfs::rendering {

    using lfs::core::Tensor;

    void brush_select(
        const float2* screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        uint8_t* selection_out,
        int n_primitives);

    void rect_select(
        const float2* positions,
        float x0,
        float y0,
        float x1,
        float y1,
        bool* selection,
        int n_primitives);

    void rect_select_mode(
        const float2* positions,
        float x0,
        float y0,
        float x1,
        float y1,
        bool* selection,
        int n_primitives,
        bool add_mode);

    void polygon_select(
        const float2* positions,
        const float2* polygon,
        int num_vertices,
        bool* selection,
        int n_primitives);

    void polygon_select_mode(
        const float2* positions,
        const float2* polygon,
        int num_vertices,
        bool* selection,
        int n_primitives,
        bool add_mode);

    void set_selection_element(bool* selection, int index, bool value);

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

    void rect_select_mode_tensor(
        const Tensor& screen_positions,
        float x0,
        float y0,
        float x1,
        float y1,
        Tensor& selection_out,
        bool add_mode);

    void polygon_select_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out);

    void polygon_select_mode_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out,
        bool add_mode);

    void apply_selection_group_tensor(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        uint8_t group_id,
        const uint32_t* locked_groups,
        bool add_mode,
        const Tensor* transform_indices = nullptr,
        int target_node_index = -1);

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

    void filter_selection_by_node(
        Tensor& selection,
        const Tensor& transform_indices,
        int target_node_index);

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

    namespace config {
        void setSelectionGroupColor(int group_id, float3 color);
        void setSelectionPreviewColor(float3 color);
    } // namespace config

} // namespace lfs::rendering
