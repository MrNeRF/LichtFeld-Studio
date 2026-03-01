/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::training {

    struct RenderOutput {
        lfs::core::Tensor image;      // [..., channels, H, W]
        lfs::core::Tensor alpha;      // [..., C, H, W, 1]
        lfs::core::Tensor depth;      // [..., C, H, W, 1] - accumulated or expected depth
        lfs::core::Tensor means2d;    // [..., C, N, 2]
        lfs::core::Tensor depths;     // [..., N] - per-gaussian depths
        lfs::core::Tensor radii;      // [..., N]
        lfs::core::Tensor visibility; // [..., N]
        lfs::core::Tensor edges_score;
        int width;
        int height;
    };

    enum class RenderMode {
        RGB = 0,
        D = 1,
        ED = 2,
        RGB_D = 3,
        RGB_ED = 4
    };

    inline RenderMode render_mode_from_string(const std::string& mode) {
        if (mode == "RGB")
            return RenderMode::RGB;
        else if (mode == "D")
            return RenderMode::D;
        else if (mode == "ED")
            return RenderMode::ED;
        else if (mode == "RGB_D")
            return RenderMode::RGB_D;
        else if (mode == "RGB_ED")
            return RenderMode::RGB_ED;
        else
            throw std::runtime_error("Invalid render mode: " + mode);
    }

    // Alias for backward compatibility
    inline RenderMode stringToRenderMode(const std::string& mode) {
        return render_mode_from_string(mode);
    }

} // namespace lfs::training
