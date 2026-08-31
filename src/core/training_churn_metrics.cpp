/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/training_churn_metrics.hpp"

namespace lfs::core {

    TrainingChurnMetrics& TrainingChurnMetrics::instance() {
        static TrainingChurnMetrics metrics;
        return metrics;
    }

    void TrainingChurnMetrics::record_trim(const std::uint64_t elapsed_us) noexcept {
        trim_calls_.fetch_add(1, std::memory_order_relaxed);
        trim_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    }

    void TrainingChurnMetrics::record_arena_decommit(const std::uint64_t elapsed_us) noexcept {
        arena_decommit_events_.fetch_add(1, std::memory_order_relaxed);
        arena_decommit_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    }

    void TrainingChurnMetrics::record_arena_recommit(const std::uint64_t elapsed_us) noexcept {
        arena_recommit_events_.fetch_add(1, std::memory_order_relaxed);
        arena_recommit_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    }

    void TrainingChurnMetrics::record_child_alloc(const std::uint64_t elapsed_us) noexcept {
        child_alloc_events_.fetch_add(1, std::memory_order_relaxed);
        child_alloc_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    }

    void TrainingChurnMetrics::record_child_free(const std::uint64_t elapsed_us) noexcept {
        child_free_events_.fetch_add(1, std::memory_order_relaxed);
        child_free_time_us_.fetch_add(elapsed_us, std::memory_order_relaxed);
    }

    TrainingChurnMetricsSnapshot TrainingChurnMetrics::snapshot() const noexcept {
        return {
            trim_calls_.load(std::memory_order_relaxed),
            trim_time_us_.load(std::memory_order_relaxed),
            arena_decommit_events_.load(std::memory_order_relaxed),
            arena_decommit_time_us_.load(std::memory_order_relaxed),
            arena_recommit_events_.load(std::memory_order_relaxed),
            arena_recommit_time_us_.load(std::memory_order_relaxed),
            child_alloc_events_.load(std::memory_order_relaxed),
            child_alloc_time_us_.load(std::memory_order_relaxed),
            child_free_events_.load(std::memory_order_relaxed),
            child_free_time_us_.load(std::memory_order_relaxed)};
    }

} // namespace lfs::core
