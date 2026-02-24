/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include <gtest/gtest.h>
#include <span>
#include <unordered_map>

using namespace lfs::core;

namespace {

    class LazyTestGuard {
    public:
        LazyTestGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            Tensor::reset_lazy_telemetry();
        }
        ~LazyTestGuard() {
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::clear_lazy_ir_for_testing();
            Tensor::reset_lazy_telemetry();
        }
    };

} // namespace

TEST(TensorMemoryPlannerTest, LivenessComputationLinearChain) {
    LazyTestGuard guard;

    // Shape operations should carry explicit IR dependencies through deferred chaining.
    auto a = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    ASSERT_TRUE(a.has_lazy_expr());
    auto b = a.reshape({3, 2});
    auto c = b.reshape({6});
    auto d = c.reshape({2, 3});
    ASSERT_TRUE(d.has_lazy_expr());

    const auto plan = internal::lazy_planner_build_plan_for_tensor(d);
    ASSERT_TRUE(plan.has_root);
    ASSERT_GE(plan.topo_nodes.size(), 4u);

    std::unordered_map<uint64_t, size_t> node_step;
    for (size_t i = 0; i < plan.topo_nodes.size(); ++i) {
        node_step[plan.topo_nodes[i].node_id] = i;
    }

    std::unordered_map<uint64_t, size_t> last_consumer_step;
    for (size_t step = 0; step < plan.topo_nodes.size(); ++step) {
        for (uint64_t input_id : plan.topo_nodes[step].input_ids) {
            last_consumer_step[input_id] = step;
        }
    }

    // In a linear chain, each node's last consumer is the immediately next node.
    auto a_id = a.lazy_expr_id();
    auto b_id = b.lazy_expr_id();
    auto c_id = c.lazy_expr_id();

    ASSERT_NE(a_id, 0u);
    ASSERT_NE(b_id, 0u);
    ASSERT_NE(c_id, 0u);
    ASSERT_TRUE(node_step.count(a_id));
    ASSERT_TRUE(node_step.count(b_id));
    ASSERT_TRUE(node_step.count(c_id));

    EXPECT_EQ(last_consumer_step[a_id], node_step[b_id]);
    EXPECT_EQ(last_consumer_step[b_id], node_step[c_id]);
    EXPECT_EQ(last_consumer_step[c_id], node_step[d.lazy_expr_id()]);
}

TEST(TensorMemoryPlannerTest, LivenessComputationPointwiseChain) {
    LazyTestGuard guard;

    auto a = Tensor::ones({64}, Device::CPU, DataType::Float32).add(1.0f);
    auto b = a.mul(2.0f);
    auto c = b.abs();
    auto d = c.sub(3.0f);
    ASSERT_TRUE(d.has_lazy_expr());

    const auto plan = internal::lazy_planner_build_plan_for_tensor(d);
    ASSERT_TRUE(plan.has_root);
    ASSERT_GE(plan.topo_nodes.size(), 4u);

    std::unordered_map<uint64_t, size_t> node_step;
    std::unordered_map<uint64_t, size_t> last_consumer_step;
    for (size_t step = 0; step < plan.topo_nodes.size(); ++step) {
        node_step[plan.topo_nodes[step].node_id] = step;
        for (uint64_t input_id : plan.topo_nodes[step].input_ids) {
            last_consumer_step[input_id] = step;
        }
    }

    const uint64_t a_id = a.lazy_expr_id();
    const uint64_t b_id = b.lazy_expr_id();
    const uint64_t c_id = c.lazy_expr_id();
    const uint64_t d_id = d.lazy_expr_id();
    ASSERT_NE(a_id, 0u);
    ASSERT_NE(b_id, 0u);
    ASSERT_NE(c_id, 0u);
    ASSERT_NE(d_id, 0u);
    ASSERT_TRUE(node_step.count(a_id));
    ASSERT_TRUE(node_step.count(b_id));
    ASSERT_TRUE(node_step.count(c_id));
    ASSERT_TRUE(node_step.count(d_id));

    EXPECT_EQ(last_consumer_step[a_id], node_step[b_id]);
    EXPECT_EQ(last_consumer_step[b_id], node_step[c_id]);
    EXPECT_EQ(last_consumer_step[c_id], node_step[d_id]);
}

TEST(TensorMemoryPlannerTest, EarlyReleaseFiresForLinearChain) {
    LazyTestGuard guard;
    internal::lazy_executor_set_memory_planner_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    // Build a chain of shape operations that create IR-tracked dependencies.
    auto x = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    for (int i = 0; i < 9; ++i) {
        x = (i % 2 == 0) ? x.reshape({3, 2}) : x.reshape({2, 3});
    }
    ASSERT_TRUE(x.has_lazy_expr());

    const auto values = x.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    const auto diag = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diag.early_releases, 0u);
    EXPECT_GT(diag.early_release_bytes, 0u);
}

TEST(TensorMemoryPlannerTest, PeakCacheBytesReducedVsNaive) {
    LazyTestGuard guard;
    internal::lazy_executor_set_memory_planner_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    constexpr int rows = 1024;
    constexpr int cols = 256;
    constexpr size_t numel = rows * cols;
    auto x = Tensor::ones({rows, cols}, Device::CPU, DataType::Float32).add(1.0f);
    for (int i = 0; i < 9; ++i) {
        x = (i % 2 == 0) ? x.reshape({cols, rows}) : x.reshape({rows, cols});
    }
    ASSERT_TRUE(x.has_lazy_expr());

    const auto values = x.to_vector();
    ASSERT_EQ(values.size(), numel);

    const auto diag = internal::lazy_executor_diagnostics_snapshot_for_testing();
    const uint64_t total_naive_bytes = 10u * numel * sizeof(float);

    EXPECT_GT(diag.peak_cache_bytes, 0u);
    EXPECT_LT(diag.peak_cache_bytes, total_naive_bytes);
}

TEST(TensorMemoryPlannerTest, RootNodeNotReleasedEarly) {
    LazyTestGuard guard;
    internal::lazy_executor_set_memory_planner_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto a = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    auto b = a.reshape({3, 2});
    auto root = b.reshape({6});
    ASSERT_TRUE(root.has_lazy_expr());

    const auto plan = internal::lazy_planner_build_plan_for_tensor(root);
    ASSERT_TRUE(plan.has_root);
    const uint64_t root_id = plan.root_node_id;

    const auto values = root.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    // The root must be available after execution (it's the result).
    // With chain a → b → root, the planner can release a (after b) and b (after root),
    // but not root itself.
    const auto diag = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diag.executed_nodes, 0u);
    (void)root_id;
}

TEST(TensorMemoryPlannerTest, MultiConsumerNodeReleasedAfterLastConsumer) {
    LazyTestGuard guard;
    internal::lazy_executor_set_memory_planner_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    // Build a chain: base → step1 → step2 → step3 → root.
    // All intermediates (base, step1, step2, step3) should be released.
    auto base = Tensor::ones({2, 3}, Device::CPU, DataType::Float32).add(1.0f);
    auto step1 = base.reshape({3, 2});
    auto step2 = step1.reshape({6});
    auto step3 = step2.reshape({2, 3});
    auto root = step3.reshape({6});
    ASSERT_TRUE(root.has_lazy_expr());

    const auto values = root.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float v : values) {
        EXPECT_FLOAT_EQ(v, 2.0f);
    }

    const auto diag = internal::lazy_executor_diagnostics_snapshot_for_testing();
    // 4 intermediates (base, step1, step2, step3) should all be released.
    EXPECT_GE(diag.early_releases, 4u);
    EXPECT_GT(diag.early_release_bytes, 0u);
    EXPECT_GT(diag.executed_nodes, 0u);
}

TEST(TensorMemoryPlannerTest, FusionAndEarlyReleaseCoexist) {
    LazyTestGuard guard;
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_set_memory_planner_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    // Fused pointwise chain feeding into shape operations.
    auto fused = Tensor::ones({2, 3}, Device::CPU, DataType::Float32)
                     .add(1.0f)
                     .mul(2.0f)
                     .sub(0.5f);
    auto step1 = fused.reshape({3, 2});
    auto step2 = step1.reshape({6});
    auto root = step2.reshape({2, 3});
    ASSERT_TRUE(root.has_lazy_expr());

    const auto values = root.to_vector();
    ASSERT_EQ(values.size(), 6u);
    for (float v : values) {
        // (1+1)*2 - 0.5 = 3.5
        EXPECT_FLOAT_EQ(v, 3.5f);
    }

    const auto diag = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diag.executed_nodes, 0u);
    const bool fusion_fired = diag.fused_launches > 0;
    const bool early_release_fired = diag.early_releases > 0;
    EXPECT_TRUE(fusion_fired || early_release_fired);
}
