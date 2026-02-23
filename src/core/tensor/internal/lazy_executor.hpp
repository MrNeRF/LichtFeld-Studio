/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {

    class Tensor;

    namespace internal {

        enum class LazyPointwiseOpKind : uint8_t {
            AddScalar = 0,
            SubScalar = 1,
            MulScalar = 2,
            DivScalar = 3,
            // 4-9 reserved for future scalar ops
            Abs = 10,
            Neg = 11,
            Exp = 12,
            Log = 13,
            Sqrt = 14,
            Sigmoid = 15,
            Relu = 16,
            Square = 17,
            Tanh = 18,
            Rsqrt = 19,
            Sign = 20,
            Reciprocal = 21,
            Floor = 22,
            Ceil = 23,
            Round = 24
        };

        struct LazyPointwiseOp {
            LazyPointwiseOpKind kind = LazyPointwiseOpKind::AddScalar;
            float scalar = 0.0f;
        };

        struct LazyPlanNodeDebug {
            uint64_t node_id = 0;
            std::string op_name;
            std::vector<uint64_t> input_ids;
        };

        struct LazyExecutionPlanDebug {
            bool planner_enabled = false;
            bool has_root = false;
            uint64_t root_node_id = 0;
            std::vector<LazyPlanNodeDebug> topo_nodes;
        };

        struct LazyExecutorDiagnosticsSnapshot {
            uint64_t planned_nodes = 0;
            uint64_t executed_nodes = 0;
            uint64_t cache_hits = 0;
            uint64_t cache_misses = 0;
            uint64_t root_fallbacks = 0;
            uint64_t fused_launches = 0;
            uint64_t max_registry_entries = 0;
            uint64_t max_context_cache_entries = 0;
        };

        // Build topological execution plan metadata for a root tensor.
        LFS_CORE_API LazyExecutionPlanDebug lazy_planner_build_plan_for_tensor(const Tensor& output);

        // Execute planned deferred nodes and resolve the root tensor materialization.
        LFS_CORE_API Tensor lazy_planner_execute_plan_for_tensor(const Tensor& output,
                                                                 std::function<Tensor()> materializer);

        // Register/de-register deferred-node materializers for topo-driven planner execution.
        LFS_CORE_API void lazy_executor_register_deferred_materializer(uint64_t node_id,
                                                                       std::function<Tensor()> materializer,
                                                                       std::weak_ptr<void> owner);
        LFS_CORE_API void lazy_executor_register_pointwise_fusion_op(uint64_t node_id,
                                                                     uint64_t parent_node_id,
                                                                     const Tensor& source_tensor,
                                                                     LazyPointwiseOp op,
                                                                     std::weak_ptr<void> owner);
        LFS_CORE_API void lazy_executor_unregister_deferred_materializer(uint64_t node_id);
        LFS_CORE_API size_t lazy_executor_registered_node_count_for_testing();
        LFS_CORE_API void lazy_executor_clear_registry_for_testing();

        // Per-execution node-materialization cache (active only inside planner execution).
        LFS_CORE_API bool lazy_executor_context_active();
        LFS_CORE_API bool lazy_executor_lookup_cached_materialization(uint64_t node_id, Tensor& materialized);
        LFS_CORE_API void lazy_executor_cache_materialization(uint64_t node_id, const Tensor& materialized);

        // PR6 diagnostics helpers for planner/executor validation.
        LFS_CORE_API void lazy_executor_reset_diagnostics_for_testing();
        LFS_CORE_API LazyExecutorDiagnosticsSnapshot lazy_executor_diagnostics_snapshot_for_testing();

        // Optional runtime diagnostics dump gate (env + testing override).
        LFS_CORE_API void lazy_executor_set_debug_dump_override_for_testing(std::optional<bool> enabled);
        LFS_CORE_API void lazy_executor_clear_debug_dump_cache_for_testing();
        LFS_CORE_API bool lazy_executor_debug_dump_enabled_for_testing();

        // Pointwise fusion scaffold gate (on by default in lazy mode).
        LFS_CORE_API void lazy_executor_set_pointwise_fusion_override_for_testing(std::optional<bool> enabled);
        LFS_CORE_API bool lazy_executor_pointwise_fusion_enabled_for_testing();

    } // namespace internal

} // namespace lfs::core
