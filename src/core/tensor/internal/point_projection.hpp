/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_spatial.hpp"
#include "point_math.hpp"

namespace lfs::core::internal {
    LFS_POINT_HD inline void projectPoint(const float* points, float* output, size_t i,
                                          const PointProjection& p, const float* transforms,
                                          size_t transform_count, const int32_t* indices,
                                          const uint8_t* visibility, size_t visibility_count) {
        output[2 * i] = output[2 * i + 1] = p.invalid_value;
        int32_t id = indices ? indices[i] : 0;
        if (visibility && indices && (id < 0 || size_t(id) >= visibility_count || !visibility[id]))
            return;
        float x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
        if (transforms && transform_count) {
            id = id < 0 ? 0 : size_t(id) >= transform_count ? int32_t(transform_count - 1)
                                                            : id;
            const float* m = transforms + size_t(id) * 16;
            const float tx = roundedAdd(pointDot3(m[0], x, m[1], y, m[2], z), m[3]);
            const float ty = roundedAdd(pointDot3(m[4], x, m[5], y, m[6], z), m[7]);
            const float tz = roundedAdd(pointDot3(m[8], x, m[9], y, m[10], z), m[11]);
            x = tx;
            y = ty;
            z = tz;
        }
        const float dx = roundedSub(x, p.translation[0]);
        const float dy = roundedSub(y, p.translation[1]);
        const float dz = roundedSub(z, p.translation[2]);
        const auto& r = p.rotation;
        const float vx = pointDot3(r[0], dx, r[1], dy, r[2], dz);
        const float vy = -pointDot3(r[3], dx, r[4], dy, r[5], dz);
        const float vz = -pointDot3(r[6], dx, r[7], dy, r[8], dz);
        if (!pointFinite(vx) || !pointFinite(vy) || !pointFinite(vz))
            return;
        if (p.model == PointProjectionModel::Equirectangular) {
            const float length = roundedSqrt(pointDot3(vx, vx, vy, vy, vz, vz));
            if (!pointFinite(length) || length <= p.near_distance)
                return;
            constexpr float pi = 3.14159265358979323846f;
            output[2 * i] = roundedMul(roundedAdd(roundedDiv(atan2f(roundedDiv(vx, length), roundedDiv(vz, length)), 2.f * pi), .5f), float(p.width));
            output[2 * i + 1] = roundedMul(roundedAdd(roundedDiv(asinf(fminf(fmaxf(roundedDiv(vy, length), -1.f), 1.f)), pi), .5f), float(p.height));
        } else if (vz > p.near_distance) {
            if (p.model == PointProjectionModel::Orthographic) {
                if (!pointFinite(p.ortho_scale) || p.ortho_scale <= 0)
                    return;
                output[2 * i] = roundedAdd(roundedMul(vx, p.ortho_scale), p.center_x);
                output[2 * i + 1] = roundedAdd(roundedMul(vy, p.ortho_scale), p.center_y);
            } else {
                output[2 * i] = roundedAdd(roundedDiv(roundedMul(vx, p.focal_x), vz), p.center_x);
                output[2 * i + 1] = roundedAdd(roundedDiv(roundedMul(vy, p.focal_y), vz), p.center_y);
            }
        }
    }
} // namespace lfs::core::internal
