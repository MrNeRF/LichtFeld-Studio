/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

using namespace lfs::core;

namespace {

    class LazyTestGuard {
    public:
        LazyTestGuard() {
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

        ~LazyTestGuard() {
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

TEST(TensorLazyIrTest, OnModeDefersUntilBoundaryAndMaterializes) {
    LazyTestGuard guard;

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

    const float* ptr = c.ptr<float>();
    ASSERT_NE(ptr, nullptr);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.expr_nodes_created, before_boundary.expr_nodes_created);
}

TEST(TensorLazyIrTest, OnModePlannerSkeletonBuildsPlanForDeferredRoot) {
    LazyTestGuard guard;

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
    LazyTestGuard guard;

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
    LazyTestGuard guard;

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

TEST(TensorLazyIrTest, OnModePlannerTopologicalOrderRespectsPointwiseDependencies) {
    LazyTestGuard guard;

    auto a = Tensor::ones({32}, Device::CPU, DataType::Float32).add(1.0f);
    auto b = a.mul(2.0f);
    auto c = b.abs();
    auto d = c.sub(3.0f);

    ASSERT_TRUE(a.has_lazy_expr());
    ASSERT_TRUE(b.has_lazy_expr());
    ASSERT_TRUE(c.has_lazy_expr());
    ASSERT_TRUE(d.has_lazy_expr());

    const uint64_t a_id = a.lazy_expr_id();
    const uint64_t b_id = b.lazy_expr_id();
    const uint64_t c_id = c.lazy_expr_id();
    const uint64_t d_id = d.lazy_expr_id();
    ASSERT_NE(a_id, 0u);
    ASSERT_NE(b_id, 0u);
    ASSERT_NE(c_id, 0u);
    ASSERT_NE(d_id, 0u);

    const auto plan = internal::lazy_planner_build_plan_for_tensor(d);
    ASSERT_TRUE(plan.has_root);
    ASSERT_EQ(plan.root_node_id, d_id);

    std::unordered_map<uint64_t, size_t> node_index;
    std::unordered_map<uint64_t, std::vector<uint64_t>> node_inputs;
    for (size_t i = 0; i < plan.topo_nodes.size(); ++i) {
        node_index[plan.topo_nodes[i].node_id] = i;
        node_inputs[plan.topo_nodes[i].node_id] = plan.topo_nodes[i].input_ids;
    }

    ASSERT_TRUE(node_index.count(a_id));
    ASSERT_TRUE(node_index.count(b_id));
    ASSERT_TRUE(node_index.count(c_id));
    ASSERT_TRUE(node_index.count(d_id));

    EXPECT_LT(node_index[a_id], node_index[b_id]);
    EXPECT_LT(node_index[b_id], node_index[c_id]);
    EXPECT_LT(node_index[c_id], node_index[d_id]);

    ASSERT_EQ(node_inputs[b_id].size(), 1u);
    ASSERT_EQ(node_inputs[c_id].size(), 1u);
    ASSERT_EQ(node_inputs[d_id].size(), 1u);
    EXPECT_EQ(node_inputs[b_id][0], a_id);
    EXPECT_EQ(node_inputs[c_id][0], b_id);
    EXPECT_EQ(node_inputs[d_id][0], c_id);
}

TEST(TensorLazyIrTest, OnModePlannerSharedSubgraphVisibleAcrossPlans) {
    LazyTestGuard guard;

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
    LazyTestGuard guard;

    auto measure_materialization_delta = [](const Tensor& tensor) {
        const auto before_boundary = Tensor::lazy_telemetry_snapshot();
        const auto values = tensor.to_vector();
        const auto after_boundary = Tensor::lazy_telemetry_snapshot();
        return std::pair<std::vector<float>, uint64_t>{
            values, after_boundary.materializations - before_boundary.materializations};
    };

    Tensor::reset_lazy_telemetry();
    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    auto branch_a = base.slice(0, 0, 2);
    auto branch_b = base.slice(0, 0, 2);
    const auto shared = branch_a.add(branch_b);
    ASSERT_TRUE(shared.has_lazy_expr());
    auto [shared_values, shared_mat_delta] = measure_materialization_delta(shared);
    ASSERT_EQ(shared_values.size(), 6u);
    for (float value : shared_values) {
        EXPECT_FLOAT_EQ(value, 4.0f);
    }

    Tensor::reset_lazy_telemetry();
    auto split_left = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f).slice(0, 0, 2);
    auto split_right = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f).slice(0, 0, 2);
    const auto split = split_left.add(split_right);
    auto [split_values, split_mat_delta] = measure_materialization_delta(split);
    ASSERT_EQ(split_values.size(), 6u);
    for (float value : split_values) {
        EXPECT_FLOAT_EQ(value, 4.0f);
    }

    EXPECT_LE(shared_mat_delta, split_mat_delta);
}

TEST(TensorLazyIrTest, OnModePlannerDiagnosticsCaptureFanOutExecution) {
    LazyTestGuard guard;
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
    LazyTestGuard guard;

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
    LazyTestGuard guard;
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
    LazyTestGuard guard;

    internal::lazy_executor_set_debug_dump_override_for_testing(false);
    EXPECT_FALSE(internal::lazy_executor_debug_dump_enabled_for_testing());

    internal::lazy_executor_set_debug_dump_override_for_testing(true);
    EXPECT_TRUE(internal::lazy_executor_debug_dump_enabled_for_testing());

    internal::lazy_executor_set_debug_dump_override_for_testing(std::nullopt);
    internal::lazy_executor_set_debug_dump_override_for_testing(false);
    EXPECT_FALSE(internal::lazy_executor_debug_dump_enabled_for_testing());
}

TEST(TensorLazyIrTest, OnModePointwiseFusionOverrideControlsFlag) {
    LazyTestGuard guard;

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
        LazyTestGuard guard;
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

TEST(TensorLazyIrTest, OnModeRegistryGrowthGuardrailInLongCreateDropLoop) {
    LazyTestGuard guard;
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
    LazyTestGuard guard;
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
    LazyTestGuard guard;

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
    LazyTestGuard guard;

    {
        auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(1.0f).mul(2.0f);
        ASSERT_TRUE(deferred.has_lazy_expr());
        EXPECT_GT(internal::lazy_executor_registered_node_count_for_testing(), 0u);
    }

    EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
}

TEST(TensorLazyIrTest, OnModeRepeatedBoundaryAfterMaterializationHasNoAdditionalMaterialization) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({8}, Device::CPU, DataType::Float32).add(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before = Tensor::lazy_telemetry_snapshot();
    const auto first = deferred.to_vector();
    ASSERT_EQ(first.size(), 8u);

    const auto after_first = Tensor::lazy_telemetry_snapshot();
    const uint64_t first_delta = after_first.materializations - before.materializations;
    EXPECT_GT(first_delta, 0u);

    const auto second = deferred.to_vector();
    ASSERT_EQ(second.size(), 8u);
    const auto after_second = Tensor::lazy_telemetry_snapshot();
    const uint64_t second_delta = after_second.materializations - after_first.materializations;
    EXPECT_EQ(second_delta, 0u);
}

TEST(TensorLazyIrTest, OnModeMixedTransferThenHostBoundaryMaterializesOnlyOnce) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device is required for mixed boundary test";
    }

    LazyTestGuard guard;

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(2.0f).mul(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before = Tensor::lazy_telemetry_snapshot();
    auto gpu = deferred.to(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);

    const auto after_transfer = Tensor::lazy_telemetry_snapshot();
    EXPECT_GT(after_transfer.materializations - before.materializations, 0u);

    const auto host_values = deferred.to_vector();
    ASSERT_EQ(host_values.size(), 4u);
    for (float value : host_values) {
        EXPECT_FLOAT_EQ(value, 9.0f);
    }

    const auto after_host_read = Tensor::lazy_telemetry_snapshot();
    EXPECT_EQ(after_host_read.materializations, after_transfer.materializations);
}

TEST(TensorLazyIrTest, OnModeKeepsDeferredThroughViewChain) {
    LazyTestGuard guard;

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

    const auto values = view_chain.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 3.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, 1u);
    EXPECT_GE(after_boundary.expr_nodes_created, before_boundary.expr_nodes_created);
}

TEST(TensorLazyIrTest, OnModeHostReadBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(4.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    const auto values = deferred.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 5.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeDeviceTransferBoundaryMaterializes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device is required for transfer boundary test";
    }

    LazyTestGuard guard;

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    auto gpu = deferred.to(Device::CUDA);
    EXPECT_EQ(gpu.device(), Device::CUDA);
    auto back_to_cpu = gpu.to(Device::CPU);
    const auto values = back_to_cpu.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeMutationBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(7.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.zero_();
    const auto values = deferred.to_vector();
    ASSERT_EQ(values.size(), 5u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 0.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeInteropPointerBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({3}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    const void* storage = deferred.storage_ptr();
    const float* data = deferred.ptr<float>();
    ASSERT_NE(storage, nullptr);
    ASSERT_NE(data, nullptr);
    EXPECT_FLOAT_EQ(data[0], 3.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeContiguousBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    auto contiguous = deferred.contiguous();
    const auto values = contiguous.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeDtypeBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    auto as_int = deferred.to(DataType::Int32);
    const auto values = as_int.to_vector_int();
    ASSERT_EQ(values.size(), 5u);
    for (int value : values) {
        EXPECT_EQ(value, 3);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeReserveBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(3.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.reserve(8);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeNestedBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred_bool = Tensor::ones({4}, Device::CPU, DataType::Float32).gt(0.5f);
    ASSERT_TRUE(deferred_bool.has_lazy_expr());

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    const auto values = deferred_bool.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float value : values) {
        EXPECT_FLOAT_EQ(value, 1.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<int>{1, 4}, {2}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

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
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleInt64BoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({6}, Device::CPU, DataType::Float32).add(2.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{1.0f, -1.0f}, {2}, Device::CPU)
                       .to(DataType::Int64);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

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
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutSingleEmptyIndicesBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({4}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::empty({0}, Device::CPU, DataType::Int64);
    auto values = Tensor::empty({0}, Device::CPU, DataType::Float32);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.index_put_(indices, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({3, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<int>{0, 2, 1}, {3}, Device::CPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{1, 0, 2}, {3}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{5.0f, 6.0f, 7.0f}, {3}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

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
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimInt64BoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({3, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<float>{0.0f, 2.0f, 1.0f}, {3}, Device::CPU)
                       .to(DataType::Int64);
    auto col_idx = Tensor::from_vector(std::vector<float>{1.0f, 0.0f, 2.0f}, {3}, Device::CPU)
                       .to(DataType::Int64);
    auto values = Tensor::from_vector(std::vector<float>{5.0f, 6.0f, 7.0f}, {3}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

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
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexPutMultiDimMismatchBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto row_idx = Tensor::from_vector(std::vector<int>{0, 1}, {2}, Device::CPU);
    auto col_idx = Tensor::from_vector(std::vector<int>{1}, {1}, Device::CPU);
    auto values = Tensor::from_vector(std::vector<float>{9.0f, 8.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.index_put_({row_idx, col_idx}, values);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeIndexAddEdgeBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({5}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{-1.0f, 1.0f}, {2}, Device::CPU)
                       .to(DataType::Int64);
    auto src = Tensor::from_vector(std::vector<float>{4.0f, 6.0f}, {2}, Device::CPU);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.index_add_(0, indices, src);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 5u);
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[1], 8.0f);
    EXPECT_FLOAT_EQ(result[2], 2.0f);
    EXPECT_FLOAT_EQ(result[3], 2.0f);
    EXPECT_FLOAT_EQ(result[4], 6.0f);

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeAppendGatherNon1DIndexBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{0.0f, 1.0f}, {1, 2}, Device::CPU)
                       .to(DataType::Int64);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.append_gather(indices);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeAppendGatherEdgeBoundaryMaterializes) {
    LazyTestGuard guard;

    auto deferred = Tensor::ones({2, 2}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(deferred.has_lazy_expr());

    auto indices = Tensor::from_vector(std::vector<float>{0.0f}, {1}, Device::CPU)
                       .to(DataType::Int64);

    const auto before_boundary = Tensor::lazy_telemetry_snapshot();

    deferred.append_gather(indices);
    const auto result = deferred.to_vector();

    ASSERT_EQ(result.size(), 4u);
    for (float value : result) {
        EXPECT_FLOAT_EQ(value, 2.0f);
    }

    const auto after_boundary = Tensor::lazy_telemetry_snapshot();
    EXPECT_GE(after_boundary.materializations, before_boundary.materializations + 1);
}

TEST(TensorLazyIrTest, OnModeGpuPointwiseFusionValueParity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    LazyTestGuard guard;
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
        LazyTestGuard guard;
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
    LazyTestGuard guard;
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

    LazyTestGuard guard;
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
    LazyTestGuard guard;
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
        LazyTestGuard guard;
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
        LazyTestGuard guard;
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
        LazyTestGuard guard;
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
        LazyTestGuard guard;
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
        LazyTestGuard guard;
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
    LazyTestGuard guard;
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
    LazyTestGuard guard;
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
    LazyTestGuard guard;
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
    LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({4096}, 3.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mul(2.0f).sum();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    // Fused
    float fused_val;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 5.0f, Device::CUDA, DataType::Float32);
        auto result = x.mul(2.0f).abs().mean();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA);
        auto result = x.add(-1.0f).max();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.sub(0.5f).sigmoid().min();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({2048}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().exp().sum();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyTestGuard guard;
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

    LazyTestGuard guard;
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

    LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({512}, 1.01f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod();
        unfused_val = result.to(Device::CPU).to_vector()[0];
    }

    float fused_val;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).sum({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({32, 64}, 2.0f, Device::CUDA, DataType::Float32);
        auto result = x.add(1.0f).mean({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.add(1.0f).max({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::arange(0.0f, 2048.0f, 1.0f).to(Device::CUDA).reshape({32, 64});
        auto result = x.mul(0.5f).min({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({8, 16}, 1.05f, Device::CUDA, DataType::Float32);
        auto result = x.add(0.01f).prod({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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

    LazyTestGuard guard;
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
        LazyTestGuard guard;
        internal::lazy_executor_set_pointwise_fusion_override_for_testing(false);
        auto x = Tensor::full({16, 128}, 4.0f, Device::CUDA, DataType::Float32);
        auto result = x.abs().sqrt().sum({1});
        unfused_result = result.to(Device::CPU).to_vector();
    }

    std::vector<float> fused_result;
    {
        LazyTestGuard guard;
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

    LazyTestGuard guard;
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

    LazyTestGuard guard;
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

    LazyTestGuard guard;
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::full({4096}, 2.0f, Device::CUDA, DataType::Float32);
    auto result = x.add(1.0f).sum();

    ASSERT_TRUE(result.has_lazy_expr());
    const auto info = result.lazy_expr_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->op_kind, internal::LazyOpKind::Reduce);
}

// ============= PR14: Size Heuristic Tests =============

TEST(TensorLazyIrTest, SizeHeuristicSkipsDeferralForTinyTensors) {
    LazyTestGuard guard;
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    // 8 floats = 32 bytes, well below 4096 threshold
    auto tiny = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f);

    auto values = tiny.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }
}

TEST(TensorLazyIrTest, SizeHeuristicAllowsDeferralForLargeTensors) {
    LazyTestGuard guard;
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    // 100K floats = 400KB, well above 4096 threshold -- deferred
    auto large = Tensor::ones({100000}, Device::CPU, DataType::Float32).add(1.0f);

    auto values = large.to_vector();
    ASSERT_EQ(values.size(), 100000u);
    EXPECT_FLOAT_EQ(values[0], 2.0f);
}

TEST(TensorLazyIrTest, SizeHeuristicDisabledByOverride) {
    LazyTestGuard guard;
    internal::lazy_executor_set_size_heuristic_override_for_testing(false);

    auto tiny = Tensor::ones({8}, Device::CPU, DataType::Float32).add(1.0f);

    auto values = tiny.to_vector();
    ASSERT_EQ(values.size(), 8u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }
}

TEST(TensorLazyIrTest, SizeHeuristicCustomThreshold) {
    LazyTestGuard guard;
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);
    internal::lazy_executor_set_size_threshold_override_for_testing(size_t{256});

    EXPECT_EQ(internal::lazy_executor_size_heuristic_threshold(), 256u);

    // 64 floats = 256 bytes at boundary, should defer (>= threshold)
    auto at_boundary = Tensor::ones({64}, Device::CPU, DataType::Float32).add(1.0f);
    EXPECT_FLOAT_EQ(at_boundary.to_vector()[0], 2.0f);

    // 63 floats = 252 bytes below threshold
    auto below = Tensor::ones({63}, Device::CPU, DataType::Float32).add(1.0f);
    EXPECT_FLOAT_EQ(below.to_vector()[0], 2.0f);
}

TEST(TensorLazyIrTest, SizeHeuristicSmallTensorCorrectness) {
    LazyTestGuard guard;
    internal::lazy_executor_set_size_heuristic_override_for_testing(true);

    auto a = Tensor::full({4}, 3.0f, Device::CPU, DataType::Float32);
    auto b = a.add(2.0f).mul(0.5f);

    auto values = b.to_vector();
    ASSERT_EQ(values.size(), 4u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.5f);
    }
}
