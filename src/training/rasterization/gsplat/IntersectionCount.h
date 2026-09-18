/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>

namespace gsplat_lfs {

    inline constexpr int64_t kMaxIntersectionCount = std::numeric_limits<int32_t>::max();

    // Host-only validation before growing pair buffers or accepting a speculative fill.
    inline lfs::Status validate_intersection_count(const int64_t count) {
        if (count >= 0 && count <= kMaxIntersectionCount) {
            return {};
        }
        if (count < 0) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Internal,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "gsplat produced an invalid negative intersection count.",
                .detail = std::format("gsplat intersection count {} is negative", count),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        const auto message = std::format(
            "gsplat intersection count {} is outside the supported range [0, {}]. "
            "Projected splats cover too many image tiles. "
            "Restart with a higher dataset resize_factor (try 2 or 4). "
            "If splats grow too large during training, try scale_reg=0.005-0.01 "
            "and scaling_lr/scaling_lr_end=0.0005-0.001. "
            "These training adjustments cannot prevent an initial-frame failure. "
            "init_scaling does not affect MRNF initialization.",
            count, kMaxIntersectionCount);
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::BoundsViolation,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message = message,
            .detail = message,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

    // VMM rounds physical capacity up; keep sentinel padding representable in
    // signed offsets. The caller validates the exact count after a speculative
    // fill and replays if needed, so no accepted frame loses real intersections.
    inline constexpr size_t intersection_sort_capacity(const size_t capacity) {
        return std::min(capacity, static_cast<size_t>(kMaxIntersectionCount));
    }

} // namespace gsplat_lfs
