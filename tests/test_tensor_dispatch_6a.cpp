/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 6A host-dispatch relief: 6A.5a / 6A.2 / 6A.3.

#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace lfs::core;

namespace {

    class Dispatch6AGuard {
    public:
        Dispatch6AGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_node_limit_override_for_testing(std::nullopt);
            internal::lazy_ir_set_active_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }

        ~Dispatch6AGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_ir_set_node_limit_override_for_testing(std::nullopt);
            internal::lazy_ir_set_active_for_testing(std::nullopt);
            Tensor::reset_lazy_telemetry();
        }
    };

    bool has_cuda_device() {
        int device_count = 0;
        const auto status = cudaGetDeviceCount(&device_count);
        return status == cudaSuccess && device_count > 0;
    }

} // namespace

// ---------------------------------------------------------------------------
// 6A.5a — has_lazy_expr() from local deferred state, not the global IR mutex map
// ---------------------------------------------------------------------------

TEST(TensorDispatch6A, HasLazyExprReadsLocalDeferredStateOnly) {
    Dispatch6AGuard guard;

    // Force IR on so eager binaries still record debug nodes into the map.
    internal::lazy_ir_set_active_for_testing(true);

    auto a = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto b = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto eager = a.add(b);

    ASSERT_TRUE(eager.is_valid());
    EXPECT_FALSE(eager.is_deferred());
    // Eager results must not report lazy expr via the IR map after 6A.5a.
    EXPECT_FALSE(eager.has_lazy_expr());
    // IR introspection still works when IR recording is enabled.
    EXPECT_GT(eager.lazy_expr_id(), 0u);
    EXPECT_TRUE(internal::tensor_has_lazy_expr(eager));

    // Deferred tensors report has_lazy_expr from local state alone.
    auto deferred = a.add(1.0f).mul(2.0f);
    ASSERT_TRUE(deferred.is_deferred());
    EXPECT_TRUE(deferred.has_lazy_expr());
    EXPECT_TRUE(deferred.is_valid());
}

TEST(TensorDispatch6A, HasLazyExprMatchesIsDeferredWhenIrOff) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto a = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto b = Tensor::ones({8}, Device::CPU, DataType::Float32);
    auto eager = a.add(b);
    EXPECT_FALSE(eager.is_deferred());
    EXPECT_FALSE(eager.has_lazy_expr());
    EXPECT_EQ(eager.lazy_expr_id(), 0u);

    auto deferred = a.add(1.0f).mul(2.0f);
    EXPECT_TRUE(deferred.is_deferred());
    EXPECT_TRUE(deferred.has_lazy_expr());
    // Deferred still gets a fusion node id even with IR "off".
    EXPECT_GT(deferred.lazy_expr_id(), 0u);
}

// ---------------------------------------------------------------------------
// 6A.2 — IR recording opt-in; fusion still works with IR off
// ---------------------------------------------------------------------------

TEST(TensorDispatch6A, LazyIrDefaultOffAndOptIn) {
    Dispatch6AGuard guard;
    // Testing override cleared → production default is OFF.
    internal::lazy_ir_set_active_for_testing(std::nullopt);
    EXPECT_FALSE(internal::lazy_ir_active());

    internal::lazy_ir_set_active_for_testing(true);
    EXPECT_TRUE(internal::lazy_ir_active());

    internal::lazy_ir_set_active_for_testing(false);
    EXPECT_FALSE(internal::lazy_ir_active());
}

TEST(TensorDispatch6A, EagerBinaryDoesNotRecordWhenIrOff) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    const auto before = Tensor::lazy_telemetry_snapshot();
    auto a = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto b = Tensor::ones({4}, Device::CPU, DataType::Float32);
    auto c = a.add(b);
    ASSERT_TRUE(c.is_valid());
    EXPECT_FALSE(internal::tensor_has_lazy_expr(c));
    EXPECT_EQ(c.lazy_expr_id(), 0u);
    const auto after = Tensor::lazy_telemetry_snapshot();
    // No eager IR node should have been created.
    EXPECT_EQ(after.expr_nodes_created, before.expr_nodes_created);
}

TEST(TensorDispatch6A, UnaryReduceFusesWithIrOff) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    Dispatch6AGuard guard;
    // Production-like: IR off, fusion on.
    internal::lazy_ir_set_active_for_testing(false);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);
    internal::lazy_executor_reset_diagnostics_for_testing();

    auto x = Tensor::ones({4096}, Device::CUDA, DataType::Float32);
    // abs is fusable unary; full reduce should consume the pointwise fusion.
    auto result = x.abs().sum();
    const float value = result.item<float>();
    EXPECT_NEAR(value, 4096.0f, 1e-2f);

    const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
    EXPECT_GT(diagnostics.fused_launches, 0u)
        << "unary→reduce must still fuse when lazy IR recording is off";
}

// Microbench: IR on vs IR off for tight eager-add loops (host path cost).
// Prints Mops/s; does not assert a ratio (machine-dependent).
TEST(TensorDispatch6A, MicrobenchEagerAddIrOnVsOff) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    Dispatch6AGuard guard;
    constexpr int kIters = 20000;

    auto run = [&](bool ir_on, const TensorShape& shape) -> double {
        internal::lazy_ir_set_active_for_testing(ir_on);
        internal::clear_lazy_ir_for_testing();
        auto a = Tensor::ones(shape, Device::CUDA, DataType::Float32);
        auto b = Tensor::ones(shape, Device::CUDA, DataType::Float32);
        // Warmup
        for (int i = 0; i < 100; ++i) {
            (void)a.add(b);
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) {
            (void)a.add(b);
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return static_cast<double>(kIters) / sec;
    };

    for (const auto& shape : {TensorShape({1}), TensorShape({4096})}) {
        const double on = run(true, shape);
        const double off = run(false, shape);
        std::cout << "MICROBENCH 6A.2 shape=" << shape.str()
                  << " IR_on=" << on << " ops/s"
                  << " IR_off=" << off << " ops/s"
                  << " delta=" << ((off - on) / on * 100.0) << "%\n";
        EXPECT_GT(on, 0.0);
        EXPECT_GT(off, 0.0);
    }
}

// ---------------------------------------------------------------------------
// 6A.3 — Contiguous same-shape same-dtype binary fast path correctness
// ---------------------------------------------------------------------------

TEST(TensorDispatch6A, BinaryFastPathFloat32Cpu) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f}, {4}, Device::CPU);

    auto sum = a.add(b);
    auto prod = a.mul(b);
    auto diff = a.sub(b);

    ASSERT_EQ(sum.to_vector(), (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
    ASSERT_EQ(prod.to_vector(), (std::vector<float>{10.0f, 40.0f, 90.0f, 160.0f}));
    ASSERT_EQ(diff.to_vector(), (std::vector<float>{-9.0f, -18.0f, -27.0f, -36.0f}));
}

TEST(TensorDispatch6A, BinaryFastPathEmptyAndScalarShapes) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto empty_a = Tensor::empty({0}, Device::CPU, DataType::Float32);
    auto empty_b = Tensor::empty({0}, Device::CPU, DataType::Float32);
    auto empty_sum = empty_a.add(empty_b);
    EXPECT_TRUE(empty_sum.is_valid());
    EXPECT_EQ(empty_sum.numel(), 0u);

    auto s1 = Tensor::from_vector({3.0f}, {1}, Device::CPU);
    auto s2 = Tensor::from_vector({4.0f}, {1}, Device::CPU);
    auto ssum = s1.add(s2);
    ASSERT_EQ(ssum.numel(), 1u);
    EXPECT_FLOAT_EQ(ssum.item<float>(), 7.0f);
}

TEST(TensorDispatch6A, BinaryFastPathInt32) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto a = Tensor::from_vector(std::vector<int>{1, 2, 3}, {3}, Device::CPU);
    auto b = Tensor::from_vector(std::vector<int>{4, 5, 6}, {3}, Device::CPU);
    auto sum = a.add(b);
    auto vals = sum.to_vector_int();
    ASSERT_EQ(vals, (std::vector<int>{5, 7, 9}));
}

TEST(TensorDispatch6A, BinaryFastPathInt64) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    // from_vector has no int64 overload; promote from int32.
    auto a = Tensor::from_vector(std::vector<int>{10, 20, 30}, {3}, Device::CPU)
                 .to(DataType::Int64);
    auto b = Tensor::from_vector(std::vector<int>{1, 2, 3}, {3}, Device::CPU)
                 .to(DataType::Int64);
    auto sum = a.add(b);
    EXPECT_EQ(sum.dtype(), DataType::Int64);
    auto vals = sum.to_vector_int64();
    ASSERT_EQ(vals, (std::vector<int64_t>{11, 22, 33}));
}

TEST(TensorDispatch6A, BinaryFastPathFloat32Cuda) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto a = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA);
    auto b = Tensor::from_vector({10.0f, 20.0f, 30.0f, 40.0f}, {4}, Device::CUDA);
    auto sum = a.add(b).cpu();
    auto prod = a.mul(b).cpu();
    ASSERT_EQ(sum.to_vector(), (std::vector<float>{11.0f, 22.0f, 33.0f, 44.0f}));
    ASSERT_EQ(prod.to_vector(), (std::vector<float>{10.0f, 40.0f, 90.0f, 160.0f}));
}

TEST(TensorDispatch6A, BinaryFastPathFloat16Cuda) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    auto a_f = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA);
    auto b_f = Tensor::from_vector({0.5f, 1.5f, 2.5f, 3.5f}, {4}, Device::CUDA);
    auto a = a_f.to(DataType::Float16);
    auto b = b_f.to(DataType::Float16);
    auto sum = a.add(b).to(DataType::Float32).cpu();
    auto vals = sum.to_vector();
    ASSERT_EQ(vals.size(), 4u);
    EXPECT_NEAR(vals[0], 1.5f, 1e-2f);
    EXPECT_NEAR(vals[1], 3.5f, 1e-2f);
    EXPECT_NEAR(vals[2], 5.5f, 1e-2f);
    EXPECT_NEAR(vals[3], 7.5f, 1e-2f);
}

TEST(TensorDispatch6A, BinaryPromotionAndBroadcastStillWork) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    // Promotion: int + float → float
    auto ai = Tensor::from_vector(std::vector<int>{1, 2}, {2}, Device::CPU);
    auto af = Tensor::from_vector({0.5f, 1.5f}, {2}, Device::CPU);
    auto promoted = ai.add(af);
    EXPECT_EQ(promoted.dtype(), DataType::Float32);
    auto pv = promoted.to_vector();
    ASSERT_EQ(pv.size(), 2u);
    EXPECT_NEAR(pv[0], 1.5f, 1e-5f);
    EXPECT_NEAR(pv[1], 3.5f, 1e-5f);

    // Broadcast: [2,1] + [1,3]
    auto left = Tensor::from_vector({1.0f, 2.0f}, {2, 1}, Device::CPU);
    auto right = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {1, 3}, Device::CPU);
    auto bc = left.add(right);
    EXPECT_EQ(bc.shape().str(), "[2, 3]");
    auto bv = bc.to_vector();
    ASSERT_EQ(bv.size(), 6u);
    EXPECT_FLOAT_EQ(bv[0], 11.0f);
    EXPECT_FLOAT_EQ(bv[1], 21.0f);
    EXPECT_FLOAT_EQ(bv[2], 31.0f);
    EXPECT_FLOAT_EQ(bv[3], 12.0f);
    EXPECT_FLOAT_EQ(bv[4], 22.0f);
    EXPECT_FLOAT_EQ(bv[5], 32.0f);
}

// Microbench: host dispatch cost for contiguous same-shape binary add/mul (6A.3).
// Prints ns/op; does not assert a ratio (machine-dependent).
TEST(TensorDispatch6A, MicrobenchBinaryFastPathNsPerOp) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required";
    }

    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);
    constexpr int kIters = 50000;

    auto measure_ns = [&](const TensorShape& shape, bool use_mul) -> double {
        auto a = Tensor::ones(shape, Device::CUDA, DataType::Float32);
        auto b = Tensor::ones(shape, Device::CUDA, DataType::Float32);
        for (int i = 0; i < 200; ++i) {
            if (use_mul) {
                (void)a.mul(b);
            } else {
                (void)a.add(b);
            }
        }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kIters; ++i) {
            if (use_mul) {
                (void)a.mul(b);
            } else {
                (void)a.add(b);
            }
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return (sec * 1e9) / static_cast<double>(kIters);
    };

    for (const auto& shape : {TensorShape({1}), TensorShape({4096})}) {
        const double add_ns = measure_ns(shape, false);
        const double mul_ns = measure_ns(shape, true);
        std::cout << "MICROBENCH 6A.3 shape=" << shape.str()
                  << " add=" << add_ns << " ns/op"
                  << " mul=" << mul_ns << " ns/op\n";
        EXPECT_GT(add_ns, 0.0);
        EXPECT_GT(mul_ns, 0.0);
    }
}

// ---------------------------------------------------------------------------
// 6A.1 — Share TensorState on Tensor copy (stream/name/lazy/capacity on shared impl)
// ---------------------------------------------------------------------------

TEST(TensorHandle6A, CopySharesNameAndTrackedState) {
    Dispatch6AGuard guard;

    auto a = Tensor::ones({8}, Device::CPU, DataType::Float32);
    a.set_name("alpha");
    a.set_tracked(true);

    auto b = a; // shallow copy — must share TensorState
    EXPECT_EQ(b.name(), "alpha");
    EXPECT_TRUE(b.is_tracked());

    // Mutating metadata through either handle must be visible on both.
    b.set_name("beta");
    EXPECT_EQ(a.name(), "beta") << "6A.1: copy must share TensorState (name)";
    a.set_tracked(false);
    EXPECT_FALSE(b.is_tracked()) << "6A.1: copy must share TensorState (tracked)";
}

TEST(TensorHandle6A, CopySharesCapacityMetadata) {
    Dispatch6AGuard guard;

    auto a = Tensor::ones({4, 3}, Device::CPU, DataType::Float32);
    // reserve grows capacity on dim0; shared state means both handles see it.
    a.reserve(16);
    ASSERT_GE(a.capacity(), 16u);

    auto b = a;
    EXPECT_EQ(b.capacity(), a.capacity());
    EXPECT_EQ(b.logical_size(), a.logical_size());

    b.reserve(32);
    EXPECT_EQ(a.capacity(), b.capacity()) << "6A.1: capacity must live on shared TensorState";
    EXPECT_GE(a.capacity(), 32u);
}

TEST(TensorHandle6A, EmptyDefaultTensorCopyIsSafe) {
    Dispatch6AGuard guard;

    Tensor empty;
    EXPECT_FALSE(empty.is_valid());
    EXPECT_EQ(empty.capacity(), 0u);
    EXPECT_EQ(empty.stream(), nullptr);
    EXPECT_TRUE(empty.name().empty());

    Tensor copy = empty;
    EXPECT_FALSE(copy.is_valid());
    EXPECT_EQ(copy.capacity(), 0u);
    EXPECT_EQ(copy.stream(), nullptr);

    Tensor assigned;
    assigned = empty;
    EXPECT_FALSE(assigned.is_valid());
}

TEST(TensorHandle6A, DeferredCopyMaterializesIndependentlyOnHandle) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);
    internal::lazy_executor_set_pointwise_fusion_override_for_testing(true);

    auto x = Tensor::ones({16}, Device::CPU, DataType::Float32);
    // Build a deferred fusion chain (unary→unary).
    auto deferred = x.add(1.0f).mul(2.0f);
    ASSERT_TRUE(deferred.is_deferred());

    auto sibling = deferred; // shares lazy state after 6A.1
    EXPECT_TRUE(sibling.is_deferred());

    // Materialize through the original handle (16 elements — use to_vector, not item).
    const auto vals0 = deferred.to_vector();
    ASSERT_EQ(vals0.size(), 16u);
    EXPECT_NEAR(vals0[0], 4.0f, 1e-5f); // (1+1)*2
    EXPECT_FALSE(deferred.is_deferred());
    EXPECT_TRUE(deferred.is_valid());

    // Sibling must still be able to materialize (or already see published result).
    const auto vals1 = sibling.to_vector();
    ASSERT_EQ(vals1.size(), 16u);
    EXPECT_NEAR(vals1[0], 4.0f, 1e-5f);
    EXPECT_FALSE(sibling.is_deferred());
    EXPECT_TRUE(sibling.is_valid());
}

// ---------------------------------------------------------------------------
// 6A.4 — Inline small-vector shapes/strides (no heap for rank ≤ 8)
// ---------------------------------------------------------------------------

TEST(TensorHandle6A, ShapeStridesRankedNoHeapSemantics) {
    Dispatch6AGuard guard;

    const TensorShape shape({2, 3, 4});
    EXPECT_EQ(shape.rank(), 3u);
    EXPECT_EQ(shape.elements(), 24u);
    EXPECT_EQ(shape[0], 2u);
    EXPECT_EQ(shape[1], 3u);
    EXPECT_EQ(shape[2], 4u);

    const auto strides = shape.strides();
    EXPECT_EQ(strides.size(), 3u);
    EXPECT_EQ(strides[0], 12u);
    EXPECT_EQ(strides[1], 4u);
    EXPECT_EQ(strides[2], 1u);

    // dims() must be usable as span (broadcast helpers).
    EXPECT_TRUE(broadcast::can_broadcast(shape.dims(), shape.dims()));

    auto t = Tensor::ones({2, 3, 4}, Device::CPU, DataType::Float32);
    EXPECT_EQ(t.strides().size(), 3u);
    EXPECT_EQ(t.strides()[0], 12u);
    EXPECT_EQ(t.strides()[1], 4u);
    EXPECT_EQ(t.strides()[2], 1u);

    // Copy must keep shape/strides identical without depending on heap vectors.
    auto c = t;
    EXPECT_EQ(c.shape(), t.shape());
    EXPECT_EQ(c.strides().size(), t.strides().size());
    for (size_t i = 0; i < t.strides().size(); ++i) {
        EXPECT_EQ(c.strides()[i], t.strides()[i]);
    }
}

// Microbench: Tensor copy cost (6A.1) + small eager add host path (6A.1+6A.4).
TEST(TensorHandle6A, MicrobenchTensorCopyAndPerOpNs) {
    Dispatch6AGuard guard;
    internal::lazy_ir_set_active_for_testing(false);

    constexpr int kCopyIters = 200000;
    constexpr int kOpIters = 50000;

    auto measure_copy_ns = [&](const Tensor& src) -> double {
        for (int i = 0; i < 1000; ++i) {
            Tensor tmp = src;
            (void)tmp;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < kCopyIters; ++i) {
            Tensor tmp = src;
            (void)tmp;
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return (sec * 1e9) / static_cast<double>(kCopyIters);
    };

    // CPU tensor copy (pure host path — isolates TensorState deep-copy cost).
    auto cpu = Tensor::ones({64}, Device::CPU, DataType::Float32);
    const double cpu_copy_ns = measure_copy_ns(cpu);
    std::cout << "MICROBENCH 6A.1 Tensor_copy_cpu_ns=" << cpu_copy_ns << "\n";
    EXPECT_GT(cpu_copy_ns, 0.0);

    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device required for CUDA copy + per-op microbench";
    }

    auto cuda = Tensor::ones({64}, Device::CUDA, DataType::Float32);
    const double cuda_copy_ns = measure_copy_ns(cuda);
    std::cout << "MICROBENCH 6A.1 Tensor_copy_cuda_ns=" << cuda_copy_ns << "\n";
    EXPECT_GT(cuda_copy_ns, 0.0);

    auto a = Tensor::ones({16}, Device::CUDA, DataType::Float32);
    auto b = Tensor::ones({16}, Device::CUDA, DataType::Float32);
    for (int i = 0; i < 200; ++i) {
        (void)a.add(b);
    }
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kOpIters; ++i) {
        (void)a.add(b);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double op_ns =
        (std::chrono::duration<double>(t1 - t0).count() * 1e9) / static_cast<double>(kOpIters);
    std::cout << "MICROBENCH 6A.1+6A.4 eager_add_16_ns_per_op=" << op_ns << "\n";
    EXPECT_GT(op_ns, 0.0);
}
