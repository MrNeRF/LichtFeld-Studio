/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis {

    // Stable view identity. Zero is invalid. Allocated IDs are monotonic and
    // are never reused after close, failed allocation, or restore.
    using ViewId = std::uint64_t;
    inline constexpr ViewId kInvalidViewId = 0;

    // Separate tree-node identity. Same integer width as ViewId, not a UUID,
    // and not interchangeable with view identity at the API boundary.
    using LayoutNodeId = std::uint64_t;
    inline constexpr LayoutNodeId kInvalidLayoutNodeId = 0;

    [[nodiscard]] constexpr bool isValidViewId(const ViewId id) noexcept {
        return id != kInvalidViewId;
    }

    [[nodiscard]] constexpr bool isValidLayoutNodeId(const LayoutNodeId id) noexcept {
        return id != kInvalidLayoutNodeId;
    }

} // namespace lfs::vis
