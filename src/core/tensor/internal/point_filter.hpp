/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#define LFS_FILTER_NODES             1u
#define LFS_FILTER_BOX               2u
#define LFS_FILTER_ELLIPSOID         4u
#define LFS_FILTER_WINDOW            8u
#define LFS_FILTER_INVERSE_BOX       16u
#define LFS_FILTER_INVERSE_ELLIPSOID 32u
#define LFS_FILTER_GEOMETRY          (LFS_FILTER_BOX | LFS_FILTER_ELLIPSOID | LFS_FILTER_WINDOW)

#ifndef LFS_POINT_FILTER_FLAGS_ONLY
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_filters.hpp"
#include "point_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace lfs::core::internal {
    enum PointFilterInput : size_t {
        FilterPoints,
        FilterTransforms,
        FilterIndices,
        FilterAllowed,
        FilterBoxTransform,
        FilterBoxMin,
        FilterBoxMax,
        FilterEllipsoidTransform,
        FilterEllipsoidRadii,
        FilterInputCount
    };

    struct PointFilterArgs {
        uint8_t* mask;
        const float *points, *transforms, *box_transform, *box_min, *box_max, *ellipsoid_transform, *ellipsoid_radii;
        const int32_t* indices;
        const uint8_t* allowed;
        uint32_t count, transform_count, allowed_count, flags;
        float rotation[9], translation[3];
        float focal_x, focal_y, center_x, center_y, ortho_scale;
        float width, height, near_depth, far_depth, scale_x, scale_y, offset_x, offset_y;
        PointProjectionModel model;
    };

    inline PointFilterArgs pointFilterArgs(const PointFilterWindow& window) {
        const auto& v = window.projection;
        PointFilterArgs p{};
        std::copy(v.rotation.begin(), v.rotation.end(), p.rotation);
        std::copy(v.translation.begin(), v.translation.end(), p.translation);
        p.focal_x = v.focal_x;
        p.focal_y = v.focal_y;
        p.center_x = v.center_x;
        p.center_y = v.center_y;
        p.ortho_scale = v.ortho_scale;
        p.width = v.width;
        p.height = v.height;
        p.near_depth = window.near_depth;
        p.far_depth = window.far_depth;
        p.scale_x = window.scale_x;
        p.scale_y = window.scale_y;
        p.offset_x = window.offset_x;
        p.offset_y = window.offset_y;
        p.model = v.model;
        return p;
    }

    template <class Fetch>
    inline void loadPointFilterInputs(PointFilterArgs& args, uint8_t* mask, Fetch&& fetch, uint32_t count,
                                      uint32_t transform_count, uint32_t allowed_count, uint32_t flags) {
        args.mask = mask;
        args.points = fetch.template operator()<float>(FilterPoints);
        args.transforms = fetch.template operator()<float>(FilterTransforms);
        args.indices = fetch.template operator()<int32_t>(FilterIndices);
        args.allowed = fetch.template operator()<uint8_t>(FilterAllowed);
        args.box_transform = fetch.template operator()<float>(FilterBoxTransform);
        args.box_min = fetch.template operator()<float>(FilterBoxMin);
        args.box_max = fetch.template operator()<float>(FilterBoxMax);
        args.ellipsoid_transform = fetch.template operator()<float>(FilterEllipsoidTransform);
        args.ellipsoid_radii = fetch.template operator()<float>(FilterEllipsoidRadii);
        args.count = count;
        args.transform_count = transform_count;
        args.allowed_count = allowed_count;
        args.flags = flags;
    }

    LFS_POINT_HD inline bool pointPassesFilter(const PointFilterArgs& p, uint32_t i) {
        const int32_t index = p.indices ? p.indices[i] : 0;
        if ((p.flags & LFS_FILTER_NODES) && (index < 0 || uint32_t(index) >= p.allowed_count || !p.allowed[index]))
            return false;
        if (!(p.flags & LFS_FILTER_GEOMETRY))
            return true;
        float x = p.points[size_t(i) * 3], y = p.points[size_t(i) * 3 + 1], z = p.points[size_t(i) * 3 + 2];
        if (p.transforms) {
            const uint32_t id = index < 0                              ? 0
                                : uint32_t(index) >= p.transform_count ? p.transform_count - 1
                                                                       : uint32_t(index);
            const float* m = p.transforms + size_t(id) * 16;
            const float tx = roundedAdd(pointDot3(m[0], x, m[1], y, m[2], z), m[3]);
            const float ty = roundedAdd(pointDot3(m[4], x, m[5], y, m[6], z), m[7]);
            const float tz = roundedAdd(pointDot3(m[8], x, m[9], y, m[10], z), m[11]);
            x = tx;
            y = ty;
            z = tz;
        }
        if (p.flags & LFS_FILTER_BOX) {
            const float* c = p.box_transform;
            const float lx = roundedAdd(pointDot3(c[0], x, c[4], y, c[8], z), c[12]);
            const float ly = roundedAdd(pointDot3(c[1], x, c[5], y, c[9], z), c[13]);
            const float lz = roundedAdd(pointDot3(c[2], x, c[6], y, c[10], z), c[14]);
            const bool inside = lx >= p.box_min[0] && lx <= p.box_max[0] && ly >= p.box_min[1] && ly <= p.box_max[1] &&
                                lz >= p.box_min[2] && lz <= p.box_max[2];
            if (inside == bool(p.flags & LFS_FILTER_INVERSE_BOX))
                return false;
        }
        if (p.flags & LFS_FILTER_ELLIPSOID) {
            const float* e = p.ellipsoid_transform;
            const float* r = p.ellipsoid_radii;
            const float lx = roundedAdd(pointDot3(e[0], x, e[4], y, e[8], z), e[12]);
            const float ly = roundedAdd(pointDot3(e[1], x, e[5], y, e[9], z), e[13]);
            const float lz = roundedAdd(pointDot3(e[2], x, e[6], y, e[10], z), e[14]);
            const float norm = roundedAdd(roundedAdd(roundedDiv(roundedMul(lx, lx), roundedMul(r[0], r[0])),
                                                     roundedDiv(roundedMul(ly, ly), roundedMul(r[1], r[1]))),
                                          roundedDiv(roundedMul(lz, lz), roundedMul(r[2], r[2])));
            if ((norm <= 1.f) == bool(p.flags & LFS_FILTER_INVERSE_ELLIPSOID))
                return false;
        }
        if (!(p.flags & LFS_FILTER_WINDOW))
            return true;
        const float dx = roundedSub(x, p.translation[0]), dy = roundedSub(y, p.translation[1]),
                    dz = roundedSub(z, p.translation[2]);
        const float vx = pointDot3(p.rotation[0], dx, p.rotation[1], dy, p.rotation[2], dz);
        const float vy = -(pointDot3(p.rotation[3], dx, p.rotation[4], dy, p.rotation[5], dz));
        const float vz = -(pointDot3(p.rotation[6], dx, p.rotation[7], dy, p.rotation[8], dz));
        float px = 0, py = 0, depth = vz;
        if (p.model == PointProjectionModel::Pinhole) {
            px = roundedAdd(roundedDiv(roundedMul(vx, p.focal_x), vz), p.center_x);
            py = roundedAdd(roundedDiv(roundedMul(vy, p.focal_y), vz), p.center_y);
        } else if (p.model == PointProjectionModel::Orthographic) {
            px = roundedAdd(roundedMul(vx, p.ortho_scale), .5f * p.width);
            py = roundedAdd(roundedMul(vy, p.ortho_scale), .5f * p.height);
        } else {
            const float len = roundedSqrt(pointDot3(vx, vx, vy, vy, vz, vz));
            if (len <= 1.e-6f || !pointFinite(len))
                depth = -1;
            else {
                constexpr float pi = 3.14159265358979323846f;
                px = roundedMul(roundedAdd(roundedDiv(atan2f(roundedDiv(vx, len), roundedDiv(vz, len)), 2.f * pi), .5f),
                                p.width);
                py = roundedMul(roundedAdd(roundedDiv(asinf(fminf(fmaxf(roundedDiv(vy, len), -1.f), 1.f)), pi), .5f),
                                p.height);
                depth = len;
            }
        }
        const float hw = roundedMul(roundedMul(.5f, p.scale_x), p.width),
                    hh = roundedMul(roundedMul(.5f, p.scale_y), p.height);
        const float cx = roundedAdd(.5f * p.width, roundedMul(p.offset_x, roundedSub(.5f * p.width, hw))),
                    cy = roundedAdd(.5f * p.height, roundedMul(p.offset_y, roundedSub(.5f * p.height, hh)));
        return fabsf(roundedSub(px, cx)) <= hw && fabsf(roundedSub(py, cy)) <= hh && depth >= p.near_depth &&
               depth <= p.far_depth && depth > 0;
    }
} // namespace lfs::core::internal
#endif
