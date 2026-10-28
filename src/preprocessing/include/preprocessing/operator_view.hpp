/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include <array>

namespace lfs::preprocessing {

    // Equidistant fisheye approximation. Angles describe the complete image circle.
    // Rotation columns are the virtual view's right, down and forward directions.
    struct OperatorView {
        int width = 0, height = 0, size = 1024;
        float center_x = 0.5f, center_y = 0.5f, radius_x = 0.5f, radius_y = 0.5f;
        float source_half_angle = 0, view_half_angle = 0;
        float rotation[9]{};
    };

    OperatorView make_operator_view(int width, int height, float operator_x, float operator_y,
                                    std::array<float, 4> circle, float source_fov_degrees = 200,
                                    float view_fov_degrees = 160, int size = 1024);

    // Pixel coordinates use pixel centres (integer coordinates identify pixel centres).
    // The third return value is one for a valid correspondence and zero outside either circle.
    std::array<float, 3> operator_view_coordinate(const OperatorView& view, float x, float y,
                                                  bool inverse);

    // Float32 CUDA NCHW. The inverse maps virtual-view logits back to the original lens.
    core::Tensor render_operator_view(const core::Tensor& image, const OperatorView& view);
    core::Tensor restore_operator_mask(const core::Tensor& logits, const OperatorView& view);

} // namespace lfs::preprocessing
