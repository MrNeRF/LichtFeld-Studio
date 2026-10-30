/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <array>
#include <cstdint>
#include <limits>
namespace lfs::core {
    enum class PointRegion2DKind : uint32_t { Disk,
                                              Rectangle,
                                              Polygon,
                                              Disks };

    struct PointRegion2D {
        PointRegion2DKind kind = PointRegion2DKind::Disk;
        float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
        float radius = 1;
        float minimum_coordinate = -std::numeric_limits<float>::infinity();
    };

    enum class PointProjectionModel : uint32_t {
        Pinhole = 0,
        Orthographic = 1,
        Equirectangular = 3,
    };

    struct PointProjection {
        std::array<float, 9> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<float, 3> translation{};
        float focal_x = 1;
        float focal_y = 1;
        float center_x = 0;
        float center_y = 0;
        float ortho_scale = 1;
        int32_t width = 1;
        int32_t height = 1;
        PointProjectionModel model = PointProjectionModel::Pinhole;
        static constexpr float near_distance = 1e-6f;
        float invalid_value = -1e8f;
    };
} // namespace lfs::core
