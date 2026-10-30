/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/device/point_filter.hpp"
#include "core/device/point_filter_flags.hpp"
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_filters.hpp"
#include "point_math.hpp"
#include <algorithm>
#include <array>
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

    // Window block of the GPU filters: rotation, translation, focal, center,
    // ortho scale, size, half extents, window center and depth range.
    inline std::array<float, 25> pointFilterWindowBlock(const PointFilterWindow& window) {
        const auto& v = window.projection;
        const float width = v.width, height = v.height;
        const float hw = .5f * window.scale_x * width, hh = .5f * window.scale_y * height;
        std::array<float, 25> block;
        std::copy(v.rotation.begin(), v.rotation.end(), block.begin());
        std::copy(v.translation.begin(), v.translation.end(), block.begin() + 9);
        const std::array values{v.focal_x,
                                v.focal_y,
                                v.center_x,
                                v.center_y,
                                v.ortho_scale,
                                width,
                                height,
                                hw,
                                hh,
                                .5f * width + window.offset_x * (.5f * width - hw),
                                .5f * height + window.offset_y * (.5f * height - hh),
                                window.near_depth,
                                window.far_depth};
        std::copy(values.begin(), values.end(), block.begin() + 12);
        return block;
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

} // namespace lfs::core::internal
