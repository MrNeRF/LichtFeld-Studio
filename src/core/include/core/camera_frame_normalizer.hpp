/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace lfs::core {

    struct CameraPoseForScale {
        std::array<double, 3> center{};
        std::array<double, 3> up{0.0, 1.0, 0.0};
    };

    [[nodiscard]] inline std::array<double, 3> camera_up_from_cv_rotation(const float* rotation) {
        return {-static_cast<double>(rotation[3]), -static_cast<double>(rotation[4]),
                -static_cast<double>(rotation[5])};
    }

    [[nodiscard]] inline double camera_frame_scale(const std::vector<CameraPoseForScale>& poses) {
        if (poses.empty())
            return 1.0;
        std::array<double, 3> up{}, center{};
        for (const auto& pose : poses) {
            for (int d = 0; d < 3; ++d) {
                up[d] += pose.up[d];
                center[d] += pose.center[d];
            }
        }
        const double norm = std::sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2]);
        for (double& value : up)
            value /= std::max(norm, 1.0e-12);
        for (double& value : center)
            value /= static_cast<double>(poses.size());

        std::array<double, 3> axis{up[1], -up[0], 0.0};
        const double sine = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1]);
        const double cosine = up[2];
        std::array<double, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (sine > 1.0e-12) {
            for (double& value : axis)
                value /= sine;
            const double c = 1.0 - cosine;
            rotation = {cosine + axis[0] * axis[0] * c,
                        axis[0] * axis[1] * c - axis[2] * sine,
                        axis[0] * axis[2] * c + axis[1] * sine,
                        axis[1] * axis[0] * c + axis[2] * sine,
                        cosine + axis[1] * axis[1] * c,
                        axis[1] * axis[2] * c - axis[0] * sine,
                        axis[2] * axis[0] * c - axis[1] * sine,
                        axis[2] * axis[1] * c + axis[0] * sine,
                        cosine + axis[2] * axis[2] * c};
        } else if (cosine < 0.0) {
            rotation[4] = -1.0;
            rotation[8] = -1.0;
        }
        double scale = 0.0;
        for (const auto& pose : poses) {
            std::array<double, 3> delta{};
            for (int d = 0; d < 3; ++d)
                delta[d] = pose.center[d] - center[d];
            for (int row = 0; row < 3; ++row) {
                const double aligned = rotation[row * 3] * delta[0] + rotation[row * 3 + 1] * delta[1] +
                                       rotation[row * 3 + 2] * delta[2];
                scale = std::max(scale, std::abs(aligned));
            }
        }
        return std::max(scale, 1.0e-12);
    }

} // namespace lfs::core
