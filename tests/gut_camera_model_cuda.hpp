/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>
#include <vector>

namespace gut_camera_model_test {

    struct ThinPrismCamera {
        std::array<float, 2> focal;
        std::array<float, 2> principal;
        std::array<int, 2> resolution;
        std::array<float, 4> radial;     // k1, k2, k3, k4
        std::array<float, 4> thin_prism; // p1, p2, sx1, sy1
    };

    // Image points [x, y, valid] of camera rays [x, y, z] through GUT's thin prism fisheye model.
    std::vector<float> project(const ThinPrismCamera& camera, const std::vector<float>& rays);
    // Camera rays [x, y, z, valid] of image points [x, y] through the same model.
    std::vector<float> unproject(const ThinPrismCamera& camera, const std::vector<float>& points);

} // namespace gut_camera_model_test
