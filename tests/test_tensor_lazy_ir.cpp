/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyRuntimeGuard {
    public:
        explicit LazyRuntimeGuard(LazyMode mode) {
            internal::set_lazy_mode_override_for_testing(mode);
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_clear_debug_dump_cache_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }

        ~LazyRuntimeGuard() {
            internal::set_lazy_mode_override_for_testing(std::nullopt);
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
            internal::lazy_executor_clear_debug_dump_cache_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }
    };

    bool has_cuda_device() {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

} // namespace

TEST(TensorLazyIrTest, OffModeSkipsGraphTracking) {
    LazyRuntimeGuard guard(LazyMode::Off);

    auto a = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto b = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto c = a.add(b);

    EXPECT_FALSE(c.has_lazy_expr());
    EXPECT_EQ(c.lazy_expr_id(), 0u);
    EXPECT_FALSE(c.lazy_expr_info().has_value());

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(snapshot.expr_nodes_created, 0u);
    EXPECT_EQ(snapshot.eager_fallbacks, 0u);
}

TEST(TensorLazyIrTest, OffModePlannerSkeletonDisabled) {
    LazyRuntimeGuard guard(LazyMode::Off);

    auto tensor = Tensor::ones({4}, Device::CPU, DataType::Float32);
    const auto plan = internal::lazy_planner_build_plan_for_tensor(tensor);
    EXPECT_FALSE(plan.planner_enabled);
    EXPECT_FALSE(plan.has_root);
    EXPECT_TRUE(plan.topo_nodes.empty());
}

TEST(TensorLazyIrTest, ShadowModeBuildsGraphMetadata) {
    LazyRuntimeGuard guard(LazyMode::Shadow);

    auto a = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto b = Tensor::full({8}, 3.0f, Device::CPU, DataType::Float32);
    auto c = a.add(b).mul(2.0f);

    ASSERT_TRUE(c.has_lazy_expr());
    EXPECT_GT(c.lazy_expr_id(), 0u);

    const auto info = c.lazy_expr_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_TRUE(info->op_kind == internal::LazyOpKind::Unary ||
                info->op_kind == internal::LazyOpKind::ScalarUnary);
    EXPECT_EQ(info->input_ids.size(), 1u);

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.expr_nodes_created, 4u);
    EXPECT_EQ(snapshot.eager_fallbacks, 0u);
}

TEST(TensorLazyIrTest, OnModeDefersUntilBoundaryAndTracksFallback) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto a = Tensor::ones({16}, Device::CPU, DataType::Float32);
    auto b = Tensor::ones({16}, Device::CPU, DataType::Float32);
    auto c = a.add(b);

    EXPECT_TRUE(c.has_lazy_expr());
    EXPECT_GT(c.lazy_expr_id(), 0u);

    const auto info = c.lazy_expr_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->op_kind, internal::LazyOpKind::Deferred);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(before_boundary.expr_nodes_created, 1u);
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    // Pointer access is a hard boundary and must trigger materialization.
    const float* ptr = c.ptr<float>();
    ASSERT_NE(ptr, nullptr);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_GE(after_boundary.eager_fallback_host_read, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
    EXPECT_GE(after_boundary.expr_nodes_created, before_boundary.expr_nodes_created);
}

TEST(TensorLazyIrTest, OnModePlannerSkeletonBuildsPlanForDeferredRoot) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f).mul(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto plan = internal::lazy_planner_build_plan_for_tensor(deferred);
    EXPECT_TRUE(plan.planner_enabled);
    EXPECT_TRUE(plan.has_root);
    EXPECT_GT(plan.root_node_id, 0u);
    ASSERT_FALSE(plan.topo_nodes.empty());
    EXPECT_EQ(plan.topo_nodes.back().node_id, plan.root_node_id);
}

TEST(TensorLazyIrTest, OnModePlannerTopologicalOrderRespectsDependencies) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    std::vector<int> axes = {1, 0};
    auto chained = base.permute(std::span<const int>(axes)).slice(1, 0, 2);
    ASSERT_TRUE(chained.has_lazy_expr());

    const auto plan = internal::lazy_planner_build_plan_for_tensor(chained);
    ASSERT_TRUE(plan.planner_enabled);
    ASSERT_TRUE(plan.has_root);
    ASSERT_GE(plan.topo_nodes.size(), 3u);
    ASSERT_EQ(plan.topo_nodes.back().node_id, plan.root_node_id);

    std::unordered_map<uint64_t, size_t> node_index;
    for (size_t i = 0; i < plan.topo_nodes.size(); ++i) {
        node_index[plan.topo_nodes[i].node_id] = i;
    }

    for (size_t i = 0; i < plan.topo_nodes.size(); ++i) {
        for (uint64_t input_id : plan.topo_nodes[i].input_ids) {
            const auto it = node_index.find(input_id);
            ASSERT_NE(it, node_index.end());
            EXPECT_LT(it->second, i);
        }
    }
}

TEST(TensorLazyIrTest, OnModePlannerTopologyIsDeterministic) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    std::vector<int> axes = {1, 0};
    auto chained = base.permute(std::span<const int>(axes)).slice(1, 0, 2);

    const auto plan_a = internal::lazy_planner_build_plan_for_tensor(chained);
    const auto plan_b = internal::lazy_planner_build_plan_for_tensor(chained);

    ASSERT_EQ(plan_a.planner_enabled, plan_b.planner_enabled);
    ASSERT_EQ(plan_a.has_root, plan_b.has_root);
    ASSERT_EQ(plan_a.root_node_id, plan_b.root_node_id);
    ASSERT_EQ(plan_a.topo_nodes.size(), plan_b.topo_nodes.size());
    for (size_t i = 0; i < plan_a.topo_nodes.size(); ++i) {
        EXPECT_EQ(plan_a.topo_nodes[i].node_id, plan_b.topo_nodes[i].node_id);
        EXPECT_EQ(plan_a.topo_nodes[i].op_name, plan_b.topo_nodes[i].op_name);
        EXPECT_EQ(plan_a.topo_nodes[i].input_ids, plan_b.topo_nodes[i].input_ids);
    }
}

TEST(TensorLazyIrTest, OnModePlannerSharedSubgraphVisibleAcrossPlans) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(base.has_lazy_expr());
    const uint64_t base_id = base.lazy_expr_id();
    ASSERT_GT(base_id, 0u);

    std::vector<int> axes = {1, 0};
    auto branch_a = base.permute(std::span<const int>(axes));
    auto branch_b = base.slice(0, 0, 2);

    const auto plan_a = internal::lazy_planner_build_plan_for_tensor(branch_a);
    const auto plan_b = internal::lazy_planner_build_plan_for_tensor(branch_b);

    ASSERT_TRUE(plan_a.has_root);
    ASSERT_TRUE(plan_b.has_root);

    auto contains_node = [](const internal::LazyExecutionPlanDebug& plan, uint64_t node_id) {
        for (const auto& node : plan.topo_nodes) {
            if (node.node_id == node_id) {
                return true;
            }
        }
        return false;
    };

    EXPECT_TRUE(contains_node(plan_a, base_id));
    EXPECT_TRUE(contains_node(plan_b, base_id));
    EXPECT_EQ(plan_a.topo_nodes.back().node_id, plan_a.root_node_id);
    EXPECT_EQ(plan_b.topo_nodes.back().node_id, plan_b.root_node_id);
}

TEST(TensorLazyIrTest, OnModePlannerExecutorCachesSharedSubgraphWithinMaterialization) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto measure_fallback_delta = [](const Tensor& tensor) {
        const auto before_boundary = Tensor::lazy_telemetry_snapshot();
        const auto values = tensor.to_vector();
        const auto after_boundary = Tensor::lazy_telemetry_snapshot();
        return std::pair<std::vector<float>, uint64_t>{
            values, after_boundary.eager_fallbacks - before_boundary.eager_fallbacks};
    };

    Tensor::reset_lazy_telemetry();
    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    auto branch_a = base.slice(0, 0, 2);
    auto branch_b = base.slice(0, 0, 2);
    const auto shared = branch_a.add(branch_b);
    ASSERT_TRUE(shared.has_lazy_expr());
    auto [shared_values, shared_fallback_delta] = measure_fallback_delta(shared);
    ASSERT_EQ(shared_values.size(), 6u);
    for (float value : shared_values) {
        EXPECT_FLOAT_EQ(value, 4.0f);
    }

    Tensor::reset_lazy_telemetry();
    auto split_left = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f).slice(0, 0, 2);
    auto split_right = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f).slice(0, 0, 2);
    const auto split = split_left.add(split_right);
    auto [split_values, split_fallback_delta] = measure_fallback_delta(split);
    ASSERT_EQ(split_values.size(), 6u);
    for (float value : split_values) {
        EXPECT_FLOAT_EQ(value, 4.0f);
    }

    EXPECT_LT(shared_fallback_delta, split_fallback_delta);
}

TEST(TensorLazyIrTest, OnModePlannerDiagnosticsCaptureFanOutExecution) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto base = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f);
    auto left = base.mul(2.0f).add(3.0f);
    auto right = base.sub(4.0f).abs();
    auto fanout = left.add(right);
    ASSERT_TRUE(fanout.has_lazy_expr());

    const auto values = fanout.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 9.0f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.planned_nodes, 0u);
    EXPECT_GT(diagnostics.executed_nodes, 0u);
    EXPECT_GT(diagnostics.cache_hits, 0u);
    EXPECT_GT(diagnostics.cache_misses, 0u);
    EXPECT_LE(diagnostics.root_fallbacks, diagnostics.executed_nodes);
}

TEST(TensorLazyIrTest, OnModeRepeatedBoundaryAddsNoPlannerDiagnosticsAfterMaterialization) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({8}, Device::CPU, DataType::Float32).add(2.0f).mul(4.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());
    internal::lazy_executor_reset_diagnostics_for_testing();

    const auto first = deferred.to_vector();
    ASSERT_EQ(first.size(), 8u);
    const auto after_first = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(after_first.planned_nodes, 0u);
    EXPECT_GT(after_first.executed_nodes, 0u);

    const auto second = deferred.to_vector();
    ASSERT_EQ(second.size(), 8u);
    const auto after_second = internal::lazy_executor_diagnostics_snapshot_for_testing();

    EXPECT_EQ(after_second.planned_nodes, after_first.planned_nodes);
    EXPECT_EQ(after_second.executed_nodes, after_first.executed_nodes);
    EXPECT_EQ(after_second.cache_hits, after_first.cache_hits);
    EXPECT_EQ(after_second.cache_misses, after_first.cache_misses);
    EXPECT_EQ(after_second.root_fallbacks, after_first.root_fallbacks);
}

TEST(TensorLazyIrTest, OnModePlannerDiagnosticsTrackRootFallbackWhenPlanHasNoRoot) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto eager = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto result = internal::lazy_planner_execute_plan_for_tensor(
        eager,
        [eager]() { return eager; });

    const auto values = result.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_EQ(diagnostics.planned_nodes, 0u);
    EXPECT_EQ(diagnostics.executed_nodes, 0u);
    EXPECT_EQ(diagnostics.cache_hits, 0u);
    EXPECT_EQ(diagnostics.cache_misses, 0u);
    EXPECT_EQ(diagnostics.root_fallbacks, 1u);
}

TEST(TensorLazyIrTest, OnModePlannerDebugDumpOverrideControlsFlag) {
    LazyRuntimeGuard guard(LazyMode::On);

    internal::lazy_executor_set_debug_dump_override_for_testing(false);
    EXPECT_FALSE(internal::lazy_executor_debug_dump_enabled_for_testing());

    internal::lazy_executor_set_debug_dump_override_for_testing(true);
    EXPECT_TRUE(internal::lazy_executor_debug_dump_enabled_for_testing());

    internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
    internal::lazy_executor_set_debug_dump_override_for_testing(false);
    EXPECT_FALSE(internal::lazy_executor_debug_dump_enabled_for_testing());
}

TEST(TensorLazyIrTest, OnModePointwiseFusionOverrideControlsFlag) {
    LazyRuntimeGuard guard(LazyMode::On);

    internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
    EXPECT_FALSE(internal::lazy_executor_pointwise_fusion_enabled_for_testing());

    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    EXPECT_TRUE(internal::lazy_executor_pointwise_fusion_enabled_for_testing());

    internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
    EXPECT_TRUE(internal::lazy_executor_pointwise_fusion_enabled_for_testing());
}

TEST(TensorLazyIrTest, OnModePointwiseFusionReducesLaunchesWithParity) {
    struct RunResult {
        std::vector<float> values;
        internal::LazyExecutorDiagnosticsSnapshot diagnostics;
    };

    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::ones({4096}, Device::CPU, DataType::Float32);
        auto step1 = x.add(1.5f);
        auto step2 = step1.mul(2.0f);
        auto step3 = step2.sub(3.0f);
        auto y = step3.div(4.0f);

        RunResult result;
        result.values = y.to_vector();
        result.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return result;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-6f);
    }

    EXPECT_GT(unfused.diagnostics.executed_nodes, 0u);
    EXPECT_GT(fused.diagnostics.executed_nodes, 0u);
    EXPECT_LT(fused.diagnostics.executed_nodes, unfused.diagnostics.executed_nodes);
    EXPECT_EQ(unfused.diagnostics.fused_launches, 0u);
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeOverheadBenchmarkGuardrailVsOffMode) {
    auto run_benchmark = [](LazyMode mode) {
        LazyRuntimeGuard guard(mode);

        constexpr int warmup_iters = 6;
        constexpr int timed_iters = 24;
        constexpr size_t numel = 1u << 17;
        volatile float sink = 0.0f;

        for (int i = 0; i < warmup_iters; ++i) {
            auto x = Tensor::ones({numel}, Device::CPU, DataType::Float32);
            auto y = x.add(1.0f).mul(0.5f).sub(0.25f).abs().sqrt().exp();
            const auto values = y.to_vector();
            sink += values[0];
        }

        internal::lazy_executor_reset_diagnostics_for_testing();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < timed_iters; ++i) {
            auto x = Tensor::ones({numel}, Device::CPU, DataType::Float32);
            auto y = x.add(1.0f).mul(0.5f).sub(0.25f).abs().sqrt().exp();
            const auto values = y.to_vector();
            sink += values[0];
        }
        const auto end = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        const double avg_us = static_cast<double>(us) / static_cast<double>(timed_iters);

        struct Result {
            double avg_us = 0.0;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result result;
        result.avg_us = avg_us;
        result.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();

        EXPECT_GT(sink, 0.0f);
        return result;
    };

    const auto off = run_benchmark(LazyMode::Off);
    const auto on = run_benchmark(LazyMode::On);

    ASSERT_GT(off.avg_us, 0.0);
    ASSERT_GT(on.avg_us, 0.0);
    const double ratio = on.avg_us / off.avg_us;
    std::cout << "[lazy-overhead] off_avg_us=" << off.avg_us
              << " on_avg_us=" << on.avg_us
              << " ratio=" << ratio << std::endl;

    // Wide guardrail to catch catastrophic regressions while tolerating host variance.
    EXPECT_LT(ratio, 3.0);
    EXPECT_GT(on.diagnostics.planned_nodes, 0u);
    EXPECT_GT(on.diagnostics.executed_nodes, 0u);
}

TEST(TensorLazyIrTest, OnModeRegistryGrowthGuardrailInLongCreateDropLoop) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_reset_diagnostics_for_testing();

    constexpr int iterations = 1024;
    for (int i = 0; i < iterations; ++i) {
        {
            auto deferred = Tensor::ones({32}, Device::CPU, DataType::Float32).add(1.0f).mul(2.0f);
            ASSERT_TRUE(deferred.has_lazy_expr());
        }
        if ((i % 64) == 0) {
            (void)internal::lazy_executor_registered_node_count_for_testing();
        }
    }

    EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.max_registry_entries, 0u);
    EXPECT_LE(diagnostics.max_registry_entries, 16u);
}

TEST(TensorLazyIrTest, OnModeContextCacheGrowthGuardrailBoundedByPlannedNodes) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto base = Tensor::ones({64}, Device::CPU, DataType::Float32).add(1.0f);
    auto a = base.mul(2.0f).add(3.0f).sqrt();
    auto b = base.sub(4.0f).abs().exp();
    auto c = a.add(b).mul(0.5f);
    auto values = c.to_vector();
    ASSERT_EQ(values.size(), 64u);

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.planned_nodes, 0u);
    EXPECT_GT(diagnostics.max_context_cache_entries, 0u);
    EXPECT_LE(diagnostics.max_context_cache_entries, diagnostics.planned_nodes);
}

TEST(TensorLazyIrTest, OnModePlannerRegistryPrunesAfterMaterialization) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f).mul(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());
    ASSERT_GT(internal::lazy_executor_registered_node_count_for_testing(), 0u);

    const auto values = deferred.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 4.0f);
    }

    EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
}

TEST(TensorLazyIrTest, OnModePlannerRegistryPrunesExpiredUnmaterializedDeferredNodes) {
    LazyRuntimeGuard guard(LazyMode::On);

    {
        auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(1.0f).mul(2.0f);
        ASSERT_TRUE(deferred.has_lazy_expr());
        EXPECT_GT(internal::lazy_executor_registered_node_count_for_testing(), 0u);
    }

    EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
}

TEST(TensorLazyIrTest, OnModeRepeatedBoundaryAfterMaterializationHasNoAdditionalFallback) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({8}, Device::CPU, DataType::Float32).add(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before = Tensor::lazy_telemetry_snapshot();
    const auto first = deferred.to_vector();
    ASSERT_EQ(first.size(), 8u);

    const auto after_first = Tensor::lazy_telemetry_snapshot();
    const uint64_t first_delta = after_first.eager_fallbacks - before.eager_fallbacks;
    EXPECT_GT(first_delta, 0u);

    const auto second = deferred.to_vector();
    ASSERT_EQ(second.size(), 8u);
    const auto after_second = Tensor::lazy_telemetry_snapshot();
    const uint64_t second_delta = after_second.eager_fallbacks - after_first.eager_fallbacks;
    EXPECT_EQ(second_delta, 0u);
}

TEST(TensorLazyIrTest, OnModeMixedTransferThenHostBoundaryMaterializesOnlyOnce) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device is required for mixed boundary test";
    }

    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(2.0f).mul(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before = Tensor::lazy_telemetry_snapshot();
    auto gpu = deferred.to(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);

    const auto after_transfer = Tensor::lazy_telemetry_snapshot();
    EXPECT_GT(after_transfer.eager_fallbacks - before.eager_fallbacks, 0u);
    EXPECT_GT(after_transfer.eager_fallback_device_transfer - before.eager_fallback_device_transfer, 0u);

    const auto host_values = deferred.to_vector();
    ASSERT_EQ(host_values.size(), 4u);
    for (float value : host_values) {
        EXPECT_FLOAT_EQ(value, 9.0f);
    }

    const auto after_host_read = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(after_host_read.eager_fallbacks, after_transfer.eager_fallbacks);
    EXPECT_EQ(after_host_read.eager_fallback_host_read, after_transfer.eager_fallback_host_read);
}

TEST(TensorLazyIrTest, OnModeKeepsDeferredThroughViewChain) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32);
    std::vector<int> axes = {1, 0};
    auto view_chain = base.add(2.0f)
                          .reshape({3, 2})
                          .permute(std::span<const int>(axes))
                          .slice(1, 0, 2);

    EXPECT_TRUE(view_chain.has_lazy_expr());
    const auto info = view_chain.lazy_expr_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->op_kind, internal::LazyOpKind::Deferred);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    const auto values = view_chain.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 3.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_GE(after_boundary.expr_nodes_created, before_boundary.expr_nodes_created);
}

TEST(TensorLazyIrTest, OnModeHostReadBoundaryMaterializes) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(4.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    const auto values = deferred.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 5.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_GE(after_boundary.eager_fallback_host_read, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeDeviceTransferBoundaryMaterializes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device is required for transfer boundary test";
    }

    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    auto gpu = deferred.to(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);
    auto back_to_cpu = gpu.to(Device::CPU);
    const auto values = back_to_cpu.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_GE(after_boundary.eager_fallback_device_transfer, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeMutationBoundaryMaterializes) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(7.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.zero_();
    const auto values = deferred.to_vector();
    ASSERT_EQ(values.size(), 5u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeInteropPointerBoundaryMaterializes) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({3}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    const void* storage = deferred.storage_ptr();
    const float* data = deferred.ptr<float>();
    ASSERT_NE(storage, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_FLOAT_EQ(data[0], 3.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_GE(after_boundary.eager_fallback_interop, 1u);
}

TEST(TensorLazyIrTest, OnModeContiguousBoundaryMaterializesAsOther) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    auto contiguous = deferred.contiguous();
    const auto values = contiguous.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
    EXPECT_GE(after_boundary.eager_fallback_other, 1u);
}

TEST(TensorLazyIrTest, OnModeDtypeBoundaryMaterializesAsOther) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    auto as_int = deferred.to(DataType::Int32);
    const auto values = as_int.to_vector_int();
    ASSERT_EQ(values.size(), 5u);
    for (int value : values) {
        EXPECT_EQ(value, 3);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
    EXPECT_GE(after_boundary.eager_fallback_other, 1u);
}

TEST(TensorLazyIrTest, OnModeReserveBoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.reserve(8);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeNestedBoundaryPreservesOuterReason) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred_bool = Tensor::ones({4}, Device::CPU, DataType::Float32).gt(0.5f);
    ASSERT_TRUE(deferred_bool.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    const auto values = deferred_bool.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_GE(after_boundary.eager_fallback_host_read, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_mutation, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_other, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleBoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<int>{1, 4}, {2}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_(indices, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 6u);
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 9.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 3.0f);
    EXPECT_FLOAT_EQ(result[4], 8.0f);
    EXPECT_FLOAT_EQ(result[5], 3.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleInt64BoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{1.0f, -1.0f}, {2}, Device::CPU)
                       .to(DataType::Int64);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_(indices, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 6u);
    EXPECT_FLOAT_EQ(result[0], 3.0f);
    EXPECT_FLOAT_EQ(result[1], 9.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 3.0f);
    EXPECT_FLOAT_EQ(result[4], 3.0f);
    EXPECT_FLOAT_EQ(result[5], 8.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleEmptyIndicesBoundaryMaterializesAsMutationNoOp) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::empty({0}, Device::CPU, DataType::Int64);
    auto values = Tensor::empty({0}, Device::CPU, DataType::Float32);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_(indices, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimBoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({3, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<int>{0, 2, 1}, {3}, Device::CPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{1, 0, 2}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{5.0f, 6.0f, 7.0f}, {3}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_({row_idx, col_idx}, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 9u);
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 5.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 2.0f);
    EXPECT_FLOAT_EQ(result[5], 7.0f);
    EXPECT_FLOAT_EQ(result[6], 6.0f);
    EXPECT_FLOAT_EQ(result[7], 2.0f);
    EXPECT_FLOAT_EQ(result[8], 2.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimInt64BoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({3, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<float>{0.0f, 2.0f, 1.0f}, {3}, Device::CPU)
                       .to(DataType::Int64);
    auto col_idx = Tensor::from_vector(std::vector<float>{1.0f, 0.0f, 2.0f}, {3}, Device::CPU)
                       .to(DataType::Int64);
    auto values = Tensor::from_vector(std::vector<float>{5.0f, 6.0f, 7.0f}, {3}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_({row_idx, col_idx}, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 9u);
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 5.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 2.0f);
    EXPECT_FLOAT_EQ(result[5], 7.0f);
    EXPECT_FLOAT_EQ(result[6], 6.0f);
    EXPECT_FLOAT_EQ(result[7], 2.0f);
    EXPECT_FLOAT_EQ(result[8], 2.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimMismatchBoundaryMaterializesAsMutationNoOp) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<int>{0, 1}, {2}, Device::CPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{1}, {1}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_put_({row_idx, col_idx}, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeIndexAddEdgeBoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{-1.0f, 1.0f}, {2}, Device::CPU)
                       .to(DataType::Int64);
    auto src = Tensor::from_vector(std::vector<float>{4.0f, 6.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.index_add_(0, indices, src);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 5u);
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 6.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeAppendGatherNon1DIndexBoundaryMaterializesAsMutationNoOp) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{0.0f, 1.0f}, {1, 2}, Device::CPU)
                       .to(DataType::Int64);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.append_gather(indices);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeAppendGatherEdgeBoundaryMaterializesAsMutation) {
    LazyRuntimeGuard guard(LazyMode::On);

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    // Edge path: append_gather requires reserved capacity and will return early.
    // It should still materialize deferred state as a mutation boundary first.
    auto indices = Tensor::from_vector(std::vector<float>{0.0f}, {1}, Device::CPU)
                       .to(DataType::Int64);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(before_boundary.eager_fallbacks, 0u);

    deferred.append_gather(indices);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.eager_fallbacks, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_host_read, 0u);
    EXPECT_EQ(after_boundary.eager_fallback_device_transfer, 0u);
    EXPECT_GE(after_boundary.eager_fallback_mutation, 1u);
    EXPECT_EQ(after_boundary.eager_fallback_interop, 0u);
}

TEST(TensorLazyIrTest, OnModeGpuPointwiseFusionValueParity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::ones({4096}, Device::CUDA, DataType::Float32);
    auto y = x.add(3.0f).mul(2.0f).sub(1.0f).div(4.0f);

    auto cpu_result = y.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu_result.size(), 4096u);

    for (size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_NEAR(cpu_result[i], 1.75f, 1e-5f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeGpuPointwiseFusionReducesLaunches) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    struct RunResult {
        std::vector<float> values;
        internal::LazyExecutorDiagnosticsSnapshot diagnostics;
    };

    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::ones({2048}, Device::CUDA, DataType::Float32);
        auto step1 = x.add(1.5f);
        auto step2 = step1.mul(2.0f);
        auto step3 = step2.sub(3.0f);
        auto y = step3.div(4.0f);

        RunResult result;
        result.values = y.to(Device::CPU).to_vector();
        result.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return result;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }

    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
    EXPECT_GT(unfused.diagnostics.executed_nodes, 0u);
    EXPECT_LT(fused.diagnostics.executed_nodes, unfused.diagnostics.executed_nodes);
}

TEST(TensorLazyIrTest, OnModeCpuAffineFoldMatchesExpected) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::full({256}, 5.0f, Device::CPU, DataType::Float32);
    auto y = x.add(3.0f).mul(2.0f).sub(1.0f).div(4.0f);

    auto result = y.to_vector();
    ASSERT_EQ(result.size(), 256u);

    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], 3.75f, 1e-6f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeGpuAffineFoldIdentityIsCorrect) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({512}, 42.0f, Device::CUDA, DataType::Float32);
    auto y = x.mul(1.0f).add(0.0f).mul(1.0f).add(0.0f);

    auto cpu_result = y.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu_result.size(), 512u);

    for (size_t i = 0; i < cpu_result.size(); ++i) {
        EXPECT_FLOAT_EQ(cpu_result[i], 42.0f);
    }
}

TEST(TensorLazyIrTest, OnModeAffineFoldDivZeroProducesNonFinite) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto y = x.div(0.0f);

    auto result = y.to_vector();
    ASSERT_EQ(result.size(), 8u);

    // Affine fold: a=1/0=inf, b=0/0=NaN → fma(inf,1,NaN)=NaN. IEEE 754 non-finite.
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FALSE(std::isfinite(result[i]));
    }
}

TEST(TensorLazyIrTest, OnModeCpuPureUnaryChainFusesWithParity) {
    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::full({1024}, 4.0f, Device::CPU, DataType::Float32);
        auto y = x.abs().sqrt().exp();

        struct Result {
            std::vector<float> values;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result r;
        r.values = y.to_vector();
        r.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return r;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeCpuMixedChainFusesWithParity) {
    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::full({1024}, -3.0f, Device::CPU, DataType::Float32);
        auto y = x.add(1.0f).mul(2.0f).abs().sigmoid();

        struct Result {
            std::vector<float> values;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result r;
        r.values = y.to_vector();
        r.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return r;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeGpuPureUnaryChainFusesWithParity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::full({2048}, 4.0f, Device::CUDA, DataType::Float32);
        auto y = x.abs().sqrt().exp();

        struct Result {
            std::vector<float> values;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result r;
        r.values = y.to(Device::CPU).to_vector();
        r.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return r;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeGpuMixedChainFusesWithParity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::full({2048}, -3.0f, Device::CUDA, DataType::Float32);
        auto y = x.add(1.0f).mul(2.0f).abs().sigmoid();

        struct Result {
            std::vector<float> values;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result r;
        r.values = y.to(Device::CPU).to_vector();
        r.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return r;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeMixedChainReducesLaunchCount) {
    const auto run_chain = [](bool fusion_enabled) {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(fusion_enabled);
        internal::lazy_executor_reset_diagnostics_for_testing();

        auto x = Tensor::full({512}, 2.0f, Device::CPU, DataType::Float32);
        auto step1 = x.mul(3.0f);
        auto step2 = step1.abs();
        auto step3 = step2.add(1.0f);
        auto step4 = step3.mul(0.5f);
        auto y = step4.sigmoid();

        struct Result {
            std::vector<float> values;
            internal::LazyExecutorDiagnosticsSnapshot diagnostics;
        };
        Result r;
        r.values = y.to_vector();
        r.diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        return r;
    };

    const auto fused = run_chain(true);
    const auto unfused = run_chain(false);

    ASSERT_EQ(fused.values.size(), unfused.values.size());
    for (size_t i = 0; i < fused.values.size(); ++i) {
        EXPECT_NEAR(fused.values[i], unfused.values[i], 1e-5f);
    }

    EXPECT_GT(unfused.diagnostics.executed_nodes, 0u);
    EXPECT_GT(fused.diagnostics.fused_launches, 0u);
    EXPECT_LT(fused.diagnostics.executed_nodes, unfused.diagnostics.executed_nodes);
}

TEST(TensorLazyIrTest, OnModeSingleUnaryOpFuses) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::full({256}, -2.0f, Device::CPU, DataType::Float32);
    auto y = x.sigmoid();

    auto result = y.to_vector();
    ASSERT_EQ(result.size(), 256u);

    const float expected = 1.0f / (1.0f + std::exp(2.0f));
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], expected, 1e-6f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeInterleavedScalarUnaryChainFuses) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::full({512}, -3.0f, Device::CPU, DataType::Float32);
    auto y = x.mul(2.0f).abs().add(1.0f).sigmoid();

    auto fused_result = y.to_vector();
    ASSERT_EQ(fused_result.size(), 512u);

    // Manual: mul(2) -> -6, abs -> 6, add(1) -> 7, sigmoid(7)
    const float expected = 1.0f / (1.0f + std::exp(-7.0f));
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], expected, 1e-5f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.fused_launches, 0u);
}

TEST(TensorLazyIrTest, OnModeChainLengthBoundaryDoesNotCrash) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    // 16 ops: should fuse (at the limit)
    auto x = Tensor::full({64}, 1.0f, Device::CPU, DataType::Float32);
    auto y = x;
    for (int i = 0; i < 16; ++i) {
        y = y.abs();
    }
    auto result16 = y.to_vector();
    ASSERT_EQ(result16.size(), 64u);
    for (float v : result16) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }

    // 17 ops: falls back gracefully (no crash, values still correct)
    auto z = Tensor::full({64}, 1.0f, Device::CPU, DataType::Float32);
    auto w = z;
    for (int i = 0; i < 17; ++i) {
        w = w.abs();
    }
    auto result17 = w.to_vector();
    ASSERT_EQ(result17.size(), 64u);
    for (float v : result17) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

// ============= Fused Transform-Reduce Tests =============

TEST(TensorLazyRuntimeTest, FusedReduceSumCPU) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({1024}, 3.0f, Device::CPU, DataType::Float32);
    auto fused = x.add(1.0f).mul(2.0f).sum();

    auto unfused_x = Tensor::full({1024}, 3.0f, Device::CPU, DataType::Float32);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
    auto unfused = unfused_x.add(1.0f).mul(2.0f).sum();

    auto fused_val = fused.to_vector();
    auto unfused_val = unfused.to_vector();
    ASSERT_EQ(fused_val.size(), 1u);
    ASSERT_EQ(unfused_val.size(), 1u);
    EXPECT_NEAR(fused_val[0], unfused_val[0], 1e-2f);
}

TEST(TensorLazyRuntimeTest, FusedReduceSumGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    // Unfused reference
    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({4096}, 3.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mul(2.0f).sum();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    // Fused
    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        internal::lazy_executor_reset_diagnostics_for_testing();
        auto x = Tensor::full({4096}, 3.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mul(2.0f).sum();
        fused_val = result.to(Device::CPU).to_vector()[0];

        const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        EXPECT_GT(diagnostics.fused_launches, 0u);
    }

    EXPECT_NEAR(fused_val, unfused_val, unfused_val * 1e-5f);
}

TEST(TensorLazyRuntimeTest, FusedReduceMeanGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 5.0f, Device::CUDA, DataType::Float32);
        auto result = x.mul(2.0f).abs().mean();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::full({2048}, 5.0f, Device::CUDA, DataType::Float32);
        auto result = x.mul(2.0f).abs().mean();
        fused_val = result.to(Device::CPU).to_vector()[0];
    }

    EXPECT_NEAR(fused_val, unfused_val, std::abs(unfused_val) * 1e-5f);
}

TEST(TensorLazyRuntimeTest, FusedReduceMaxGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA);
        auto result = x.add(-1.0f).max();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA);
        auto result = x.add(-1.0f).max();
        fused_val = result.to(Device::CPU).to_vector()[0];
    }

    EXPECT_NEAR(fused_val, unfused_val, 1e-5f);
}

TEST(TensorLazyRuntimeTest, FusedReduceMinGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.sub(0.5f).sigmoid().min();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::full({2048}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.sub(0.5f).sigmoid().min();
        fused_val = result.to(Device::CPU).to_vector()[0];
    }

    EXPECT_NEAR(fused_val, unfused_val, std::abs(unfused_val) * 1e-5f);
}

TEST(TensorLazyRuntimeTest, FusedReduceUnaryChainGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().exp().sum();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        internal::lazy_executor_reset_diagnostics_for_testing();
        auto x = Tensor::full({2048}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().exp().sum();
        fused_val = result.to(Device::CPU).to_vector()[0];

        const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        EXPECT_GT(diagnostics.fused_launches, 0u);
    }

    EXPECT_NEAR(fused_val, unfused_val, std::abs(unfused_val) * 1e-5f);
}

TEST(TensorLazyRuntimeTest, FusedReduceDiagnosticsGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    const auto before = internal::lazy_executor_diagnostics_snapshot_for_testing();

    auto x = Tensor::full({4096}, 1.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).mul(2.0f).sum();
    auto val = result.to(Device::CPU).to_vector();

    const auto after = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(after.fused_launches - before.fused_launches, 0u);
}

TEST(TensorLazyRuntimeTest, NonFullReduceFallback) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
    auto partial = x.add(1.0f).sum({0});

    auto cpu_result = partial.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu_result.size(), 64u);
    for (float v : cpu_result) {
        EXPECT_NEAR(v, 32.0f * 3.0f, 1e-3f);
    }
}

// ============= PR9: Reduction Integration Tests =============

TEST(TensorLazyRuntimeTest, FusedReduceProdGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    float unfused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({512}, 1.01f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        internal::lazy_executor_reset_diagnostics_for_testing();
        auto x = Tensor::full({512}, 1.01f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod();
        fused_val = result.to(Device::CPU).to_vector()[0];

        const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        EXPECT_GT(diagnostics.fused_launches, 0u);
    }

    EXPECT_NEAR(fused_val, unfused_val, std::abs(unfused_val) * 1e-4f);
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceSumGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).sum({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        internal::lazy_executor_reset_diagnostics_for_testing();
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).sum({1});
        fused_result = result.to(Device::CPU).to_vector();

        const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        EXPECT_GT(diagnostics.fused_reduce_launches, 0u);
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    ASSERT_EQ(fused_result.size(), 32u);
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], std::abs(unfused_result[i]) * 1e-5f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceMeanGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mean({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mean({1});
        fused_result = result.to(Device::CPU).to_vector();
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], std::abs(unfused_result[i]) * 1e-5f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceMaxGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.add(1.0f).max({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.add(1.0f).max({1});
        fused_result = result.to(Device::CPU).to_vector();
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], 1e-3f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceMinGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.mul(0.5f).min({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.mul(0.5f).min({1});
        fused_result = result.to(Device::CPU).to_vector();
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], 1e-3f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceProdGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({8, 16}, 1.05f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::full({8, 16}, 1.05f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod({1});
        fused_result = result.to(Device::CPU).to_vector();
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], std::abs(unfused_result[i]) * 1e-4f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceKeepdimGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).sum({1}, true);

    ASSERT_EQ(result.shape().rank(), 2u);
    EXPECT_EQ(result.shape()[0], 32u);
    EXPECT_EQ(result.shape()[1], 1u);

    auto cpu_result = result.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu_result.size(), 32u);
    for (float v : cpu_result) {
        EXPECT_NEAR(v, 64.0f * 3.0f, 1e-2f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceUnaryChainGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    std::vector<float> unfused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({16, 128}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().sum({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyRuntimeGuard guard(LazyMode::On);
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
        auto x = Tensor::full({16, 128}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().sum({1});
        fused_result = result.to(Device::CPU).to_vector();
    }

    ASSERT_EQ(fused_result.size(), unfused_result.size());
    for (size_t i = 0; i < fused_result.size(); ++i) {
        EXPECT_NEAR(fused_result[i], unfused_result[i], std::abs(unfused_result[i]) * 1e-5f);
    }
}

TEST(TensorLazyRuntimeTest, FusedSegmentedReduceDiagnosticsGPU) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    const auto before = internal::lazy_executor_diagnostics_snapshot_for_testing();

    auto x = Tensor::full({32, 64}, 1.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).sum({1});
    auto val = result.to(Device::CPU).to_vector();

    const auto after = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(after.fused_reduce_launches - before.fused_reduce_launches, 0u);
    EXPECT_GT(after.fused_launches - before.fused_launches, 0u);
}

TEST(TensorLazyRuntimeTest, NonLastDimReduceFallback) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).sum({0});

    auto cpu_result = result.to(Device::CPU).to_vector();
    ASSERT_EQ(cpu_result.size(), 64u);
    for (float v : cpu_result) {
        EXPECT_NEAR(v, 32.0f * 3.0f, 1e-2f);
    }

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_EQ(diagnostics.fused_reduce_launches, 0u);
}

TEST(TensorLazyIrTest, LazyReduceIRNodeRecorded) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({4096}, 2.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).sum();

    ASSERT_TRUE(result.has_lazy_expr());
    const auto info = result.lazy_expr_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->op_kind, internal::LazyOpKind::Reduce);
}

TEST(TensorLazyRuntimeTest, ReduceParityAcrossModes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    struct TestCase {
        std::vector<size_t> shape;
        std::vector<int> axes;
        bool keepdim;
    };

    std::vector<TestCase> cases = {
        {{64}, {}, false},
        {{32, 64}, {1}, false},
        {{32, 64}, {1}, true},
        {{8, 16, 32}, {2}, false},
        {{128}, {0}, false},
    };

    auto reduce_op_list = {ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min, ReduceOp::Prod};

    for (const auto& tc : cases) {
        for (auto op : reduce_op_list) {
            std::vector<float> off_result;
            {
                LazyRuntimeGuard guard(LazyMode::Off);
                auto x = Tensor::full(TensorShape(tc.shape), 1.5f, Device::CUDA, DataType::Float32);
                auto result = x.add(0.5f).reduce(op, {tc.axes, tc.keepdim});
                off_result = result.to(Device::CPU).to_vector();
            }

            std::vector<float> on_result;
            {
                LazyRuntimeGuard guard(LazyMode::On);
                internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
                auto x = Tensor::full(TensorShape(tc.shape), 1.5f, Device::CUDA, DataType::Float32);
                auto result = x.add(0.5f).reduce(op, {tc.axes, tc.keepdim});
                on_result = result.to(Device::CPU).to_vector();
            }

            ASSERT_EQ(off_result.size(), on_result.size())
                << "Shape/axes/keepdim mismatch for op=" << static_cast<int>(op);
            for (size_t i = 0; i < off_result.size(); ++i) {
                const float tol = std::max(std::abs(off_result[i]) * 1e-4f, 1e-5f);
                EXPECT_NEAR(on_result[i], off_result[i], tol)
                    << "op=" << static_cast<int>(op)
                    << " shape_rank=" << tc.shape.size()
                    << " i=" << i;
            }
        }
    }
}

// ============= PR14: Size Heuristic Tests =============

TEST(TensorLazyIrTest, SizeHeuristicSkipsDeferralForTinyTensors) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    // 8 floats = 32 bytes, well below 4096 threshold
    auto tiny = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f);

    auto values = tiny.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.eager_fallback_size_heuristic, 1u);
}

TEST(TensorLazyIrTest, SizeHeuristicAllowsDeferralForLargeTensors) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    // 100K floats = 400KB, well above 4096 threshold — deferred
    const auto before = Tensor::lazy_telemetry_snapshot();
    auto large = Tensor::ones({100000}, Device::CPU, DataType::Float32).add(1.0f);

    const auto after = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(after.eager_fallback_size_heuristic - before.eager_fallback_size_heuristic, 0u);

    auto values = large.to_vector();
    ASSERT_EQ(values.size(), 100000u);
    EXPECT_FLOAT_EQ(values[0], 2.0f);
}

TEST(TensorLazyIrTest, SizeHeuristicDisabledByOverride) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(false);

    // With heuristic disabled, even tiny tensors should be deferred
    auto tiny = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f);

    auto values = tiny.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(snapshot.eager_fallback_size_heuristic, 0u);
}

TEST(TensorLazyIrTest, SizeHeuristicCustomThreshold) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);
    internal::lazy_executor_set_size_threshold_override_for_testing(size_t{256});

    EXPECT_EQ(internal::lazy_executor_size_heuristic_threshold(), 256u);

    // 64 floats = 256 bytes → at boundary → should defer (>= threshold)
    const auto before_boundary = Tensor::lazy_telemetry_snapshot();
    auto at_boundary = Tensor::ones({64}, Device::CPU, DataType::Float32).add(1.0f);
    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(after_boundary.eager_fallback_size_heuristic - before_boundary.eager_fallback_size_heuristic, 0u);
    EXPECT_FLOAT_EQ(at_boundary.to_vector()[0], 2.0f);

    // 63 floats = 252 bytes → below threshold → heuristic triggers eager
    const auto before_below = Tensor::lazy_telemetry_snapshot();
    auto below = Tensor::ones({63}, Device::CPU, DataType::Float32).add(1.0f);
    const auto after_below = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_below.eager_fallback_size_heuristic - before_below.eager_fallback_size_heuristic, 1u);
    EXPECT_FLOAT_EQ(below.to_vector()[0], 2.0f);
}

TEST(TensorLazyIrTest, SizeHeuristicSmallTensorCorrectness) {
    LazyRuntimeGuard guard(LazyMode::On);
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    // Chained ops on small tensor — heuristic triggers eager at each step
    auto a = Tensor::full({4}, 3.0f, Device::CPU, DataType::Float32);
    auto b = a.add(2.0f).mul(0.5f);

    auto values = b.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.5f);
    }

    const auto snapshot = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(snapshot.eager_fallback_size_heuristic, 2u);
}
