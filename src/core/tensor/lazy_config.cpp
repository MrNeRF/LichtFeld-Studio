/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/lazy_config.hpp"
#include <atomic>

namespace lfs::core::internal {

    namespace {

        struct LazyTelemetryCounters {
            std::atomic<uint64_t> expr_nodes_created{0};
            std::atomic<uint64_t> materializations{0};
            std::atomic<uint64_t> kernel_launches{0};
            std::atomic<uint64_t> allocated_bytes{0};
            std::atomic<uint64_t> expr_nodes_live{0};
            std::atomic<uint64_t> tensor_mappings_live{0};
            std::atomic<uint64_t> expr_nodes_peak{0};
            std::atomic<uint64_t> tensor_mappings_peak{0};
            std::atomic<uint64_t> expr_nodes_dropped{0};
            std::atomic<uint64_t> expr_node_limit{0};
        };

        LazyTelemetryCounters& telemetry_counters() {
            static LazyTelemetryCounters counters;
            return counters;
        }

        void record_relaxed_max(std::atomic<uint64_t>& target, const uint64_t value) {
            uint64_t observed = target.load(std::memory_order_relaxed);
            while (observed < value &&
                   !target.compare_exchange_weak(
                       observed, value, std::memory_order_relaxed, std::memory_order_relaxed)) {
            }
        }

    } // namespace

    void reset_lazy_telemetry() {
        auto& t = telemetry_counters();
        t.expr_nodes_created.store(0, std::memory_order_relaxed);
        t.materializations.store(0, std::memory_order_relaxed);
        t.kernel_launches.store(0, std::memory_order_relaxed);
        t.allocated_bytes.store(0, std::memory_order_relaxed);
        t.expr_nodes_peak.store(t.expr_nodes_live.load(std::memory_order_relaxed),
                                std::memory_order_relaxed);
        t.tensor_mappings_peak.store(t.tensor_mappings_live.load(std::memory_order_relaxed),
                                     std::memory_order_relaxed);
        t.expr_nodes_dropped.store(0, std::memory_order_relaxed);
    }

    LazyTelemetrySnapshot lazy_telemetry_snapshot() {
        const auto& t = telemetry_counters();
        return LazyTelemetrySnapshot{
            t.expr_nodes_created.load(std::memory_order_relaxed),
            t.materializations.load(std::memory_order_relaxed),
            t.kernel_launches.load(std::memory_order_relaxed),
            t.allocated_bytes.load(std::memory_order_relaxed),
            t.expr_nodes_live.load(std::memory_order_relaxed),
            t.tensor_mappings_live.load(std::memory_order_relaxed),
            t.expr_nodes_peak.load(std::memory_order_relaxed),
            t.tensor_mappings_peak.load(std::memory_order_relaxed),
            t.expr_nodes_dropped.load(std::memory_order_relaxed),
            t.expr_node_limit.load(std::memory_order_relaxed)};
    }

    void telemetry_record_expr_node(uint64_t count) {
        telemetry_counters().expr_nodes_created.fetch_add(count, std::memory_order_relaxed);
    }

    void telemetry_record_expr_node_drop(const uint64_t count) {
        telemetry_counters().expr_nodes_dropped.fetch_add(count, std::memory_order_relaxed);
    }

    void telemetry_publish_expr_registry_counts(const uint64_t live_nodes,
                                                const uint64_t live_tensor_mappings,
                                                const uint64_t node_limit) {
        auto& t = telemetry_counters();
        t.expr_nodes_live.store(live_nodes, std::memory_order_relaxed);
        t.tensor_mappings_live.store(live_tensor_mappings, std::memory_order_relaxed);
        t.expr_node_limit.store(node_limit, std::memory_order_relaxed);
        record_relaxed_max(t.expr_nodes_peak, live_nodes);
        record_relaxed_max(t.tensor_mappings_peak, live_tensor_mappings);
    }

    void telemetry_record_materialization(uint64_t bytes) {
        auto& t = telemetry_counters();
        t.materializations.fetch_add(1, std::memory_order_relaxed);
        t.allocated_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }

    void telemetry_record_kernel_launch(uint64_t count) {
        telemetry_counters().kernel_launches.fetch_add(count, std::memory_order_relaxed);
    }

} // namespace lfs::core::internal
