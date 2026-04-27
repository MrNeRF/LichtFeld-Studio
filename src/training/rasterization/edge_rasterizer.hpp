/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "optimizer/render_output.hpp"
#include <expected>
#include <stdexcept>
#include <string>

namespace lfs::training {
    std::expected<RenderOutput, std::string> edge_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        const lfs::core::Tensor& pixel_weights,
        bool mip_filter = false);

    inline RenderOutput edge_rasterize(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        const lfs::core::Tensor& pixel_weights,
        bool mip_filter = false) {
        auto result = edge_rasterize_forward(viewpoint_camera, gaussian_model, pixel_weights, mip_filter);
        if (!result) {
            throw std::runtime_error(result.error());
        }
        return *result;
    }
} // namespace lfs::training
