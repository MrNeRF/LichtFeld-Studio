/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace lfs::core {

    struct TrainingChurnMetricsSnapshot {
        std::uint64_t trim_calls = 0;
        std::uint64_t trim_time_us = 0;
        std::uint64_t arena_decommit_events = 0;
        std::uint64_t arena_decommit_time_us = 0;
        std::uint64_t arena_recommit_events = 0;
        std::uint64_t arena_recommit_time_us = 0;
        std::uint64_t child_alloc_events = 0;
        std::uint64_t child_alloc_time_us = 0;
        std::uint64_t child_free_events = 0;
        std::uint64_t child_free_time_us = 0;
    };

    class LFS_CORE_API TrainingChurnMetrics {
    public:
        static TrainingChurnMetrics& instance();

        void record_trim(std::uint64_t elapsed_us) noexcept;
        void record_arena_decommit(std::uint64_t elapsed_us) noexcept;
        void record_arena_recommit(std::uint64_t elapsed_us) noexcept;
        void record_child_alloc(std::uint64_t elapsed_us) noexcept;
        void record_child_free(std::uint64_t elapsed_us) noexcept;

        [[nodiscard]] TrainingChurnMetricsSnapshot snapshot() const noexcept;

    private:
        std::atomic<std::uint64_t> trim_calls_{0};
        std::atomic<std::uint64_t> trim_time_us_{0};
        std::atomic<std::uint64_t> arena_decommit_events_{0};
        std::atomic<std::uint64_t> arena_decommit_time_us_{0};
        std::atomic<std::uint64_t> arena_recommit_events_{0};
        std::atomic<std::uint64_t> arena_recommit_time_us_{0};
        std::atomic<std::uint64_t> child_alloc_events_{0};
        std::atomic<std::uint64_t> child_alloc_time_us_{0};
        std::atomic<std::uint64_t> child_free_events_{0};
        std::atomic<std::uint64_t> child_free_time_us_{0};
    };

} // namespace lfs::core
