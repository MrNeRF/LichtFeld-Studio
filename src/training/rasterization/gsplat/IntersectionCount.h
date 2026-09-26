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

    // Aggregate counts stay int64: oversized ranges are subdivided into whole tiles.
    // A negative GPU count still indicates an internal error.
    inline lfs::Status validate_intersection_count(const int64_t count) {
        if (count >= 0) {
            return {};
        }
        return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Rendering,
            .user_message = "gsplat produced an invalid negative intersection count.",
            .detail = std::format("gsplat intersection count {} is negative", count),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

    // VMM rounds physical capacity up; keep sentinel padding representable in
    // signed offsets AND within this batch budget. The caller subdivides ranges
    // above budget and replays undersized speculative fills before accepting them.
    inline constexpr size_t intersection_sort_capacity(const size_t capacity, const size_t batch_budget) {
        return std::min({capacity, batch_budget, static_cast<size_t>(kMaxIntersectionCount)});
    }

} // namespace gsplat_lfs
