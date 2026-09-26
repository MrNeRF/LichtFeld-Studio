/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "diagnostics/export.hpp"
#include "diagnostics/vram_owner_model.hpp"

#include <deque>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::diagnostics {

    struct VramTimelinePoint {
        std::int64_t epoch_ms = 0;
        int iteration = 0;
        std::size_t splats = 0;
        std::array<std::int64_t, kVramOwnerCount> bytes{};
        std::size_t process_bytes = 0;
        std::size_t device_bytes = 0;
        std::size_t capacity_bytes = 0;
        std::uint16_t resolution_seconds = 1;
        std::string role = "sample";
    };

    [[nodiscard]] LFS_DIAGNOSTICS_API bool isVramProcessSpike(std::size_t previous,
                                                              std::size_t current);
    [[nodiscard]] LFS_DIAGNOSTICS_API std::size_t vramTimelineAxisMax(
        std::span<const VramTimelinePoint> points, bool device_scale = false);

    class LFS_DIAGNOSTICS_API VramTimeline {
    public:
        void push(const VramProfilerSnapshot& snapshot, std::int64_t epoch_ms);
        [[nodiscard]] std::vector<VramTimelinePoint> points() const;
        [[nodiscard]] const std::deque<VramMarker>& markers() const noexcept { return markers_; }
        [[nodiscard]] std::string csv() const;
        void clear();

    private:
        struct Bin {
            VramTimelinePoint first;
            VramTimelinePoint last;
            VramTimelinePoint low;
            VramTimelinePoint high;
            std::int64_t slot = 0;
        };
        std::deque<VramTimelinePoint> recent_;
        std::deque<Bin> older_;
        std::deque<VramMarker> markers_;
        std::vector<VramOwnerRow> previous_rows_;
        std::unordered_map<std::string, std::size_t> previous_raw_rows_;
        std::uint64_t last_profiler_marker_id_ = 0;
    };

} // namespace lfs::diagnostics
