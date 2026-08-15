/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    SplatData create_adam_test_splat(size_t n_points) {
        auto means = Tensor::randn({n_points, 3}, Device::CUDA);
        auto sh0 = Tensor::randn({n_points, 1, 3}, Device::CUDA);
        auto shN = Tensor::zeros({n_points, 0, 3}, Device::CUDA); // sh-degree 0
        auto scaling = Tensor::randn({n_points, 3}, Device::CUDA);
        auto rotation = Tensor::randn({n_points, 4}, Device::CUDA);
        auto opacity = Tensor::randn({n_points, 1}, Device::CUDA);
        return SplatData(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    // Strip reserved capacity: replace moment/scale tensors with exact-size copies
    // so capacity()==0 and state.capacity==0, forcing the next grow onto the slow path.
    void force_zero_capacity(AdamParamState& state) {
        auto strip = [](Tensor& t) {
            if (!t.is_valid() || t.numel() == 0) {
                return;
            }
            // clone/empty without reserve → capacity 0
            auto exact = Tensor::empty(t.shape(), t.device(), t.dtype());
            exact.copy_from(t);
            t = std::move(exact);
        };
        strip(state.exp_avg);
        strip(state.joint_bounds);
        if (state.grad.is_valid()) {
            strip(state.grad);
        }
        state.capacity = 0;
    }

} // namespace

TEST(AdamCapacityInvariant, SlowPathReReservesSoSecondGrowIsFast) {
    constexpr size_t n0 = 16;
    constexpr size_t n_grow = 4;

    auto splat = create_adam_test_splat(n0);
    AdamConfig cfg;
    cfg.lr = 1e-3f;
    cfg.growth_factor = 1.5f;
    cfg.initial_capacity = 0; // do not pre-reserve a large cap
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();

    auto* state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(state, nullptr);
    force_zero_capacity(*state);
    ASSERT_EQ(state->capacity, 0u);
    ASSERT_EQ(state->exp_avg.capacity(), 0u);

    AdamOptimizer::reset_slow_path_grow_count();
    ASSERT_EQ(AdamOptimizer::slow_path_grow_count(), 0u);

    // Grow 1: must hit slow path (capacity=0) and then re-reserve.
    // Also grow the parameter tensor so shapes stay consistent with state.size.
    {
        // Give the param tensors headroom so only the optimizer state is under test.
        splat.means().reserve(n0 + 4 * n_grow);
        splat.sh0().reserve(n0 + 4 * n_grow);
        splat.scaling_raw().reserve(n0 + 4 * n_grow);
        splat.rotation_raw().reserve(n0 + 4 * n_grow);
        splat.opacity_raw().reserve(n0 + 4 * n_grow);

        splat.means().append_zeros(n_grow);
        splat.sh0().append_zeros(n_grow);
        splat.scaling_raw().append_zeros(n_grow);
        splat.rotation_raw().append_zeros(n_grow);
        splat.opacity_raw().append_zeros(n_grow);
        opt.extend_state_for_new_params(ParamType::Means, n_grow);
    }

    // Slow path must have fired once and restored capacity >= size.
    EXPECT_EQ(AdamOptimizer::slow_path_grow_count(), 1u)
        << "first grow with capacity=0 must take the slow path exactly once";
    state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(state, nullptr);
    EXPECT_GE(state->capacity, state->size)
        << "after slow path, capacity invariant must be restored";
    EXPECT_GT(state->exp_avg.capacity(), 0u)
        << "moment tensors must be re-reserved after slow path";
    EXPECT_GE(state->exp_avg.capacity(), state->size);

    const size_t size_after_first = state->size;
    const size_t cap_after_first = state->capacity;
    ASSERT_GE(cap_after_first, size_after_first + n_grow)
        << "re-reserve must leave headroom for a second grow of n_grow=" << n_grow
        << " (size=" << size_after_first << " cap=" << cap_after_first << ")";

    // Grow 2: must use fast path — alloc counter delta small (append only, no cat realloc).
    Tensor::trim_memory_pool();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto snap = alloc_counter::snapshot();
    const uint64_t slow_before = AdamOptimizer::slow_path_grow_count();

    if (splat.means().capacity() >= splat.means().shape()[0] + n_grow) {
        splat.means().append_zeros(n_grow);
        splat.sh0().append_zeros(n_grow);
        splat.scaling_raw().append_zeros(n_grow);
        splat.rotation_raw().append_zeros(n_grow);
        splat.opacity_raw().append_zeros(n_grow);
        opt.extend_state_for_new_params(ParamType::Means, n_grow);
    } else {
        FAIL() << "param capacity insufficient for second grow setup";
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto alloc_delta = alloc_counter::delta_since(snap);
    const uint64_t slow_after = AdamOptimizer::slow_path_grow_count();

    EXPECT_EQ(slow_after, slow_before)
        << "second grow must not re-enter the slow path after re-reserve";
    EXPECT_EQ(slow_after, 1u)
        << "telemetry counter must remain exactly 1 after the second (fast) grow";

    // Fast path does in-place append_zeros — zero or near-zero driver allocs.
    EXPECT_LE(alloc_delta, 2u)
        << "fast-path second grow must not cat/realloc (alloc_delta=" << alloc_delta << ")";

    state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->size, size_after_first + n_grow);
    EXPECT_GE(state->capacity, state->size);
}

TEST(AdamCapacityInvariant, SlowPathGatherAlsoRestoresCapacity) {
    constexpr size_t n0 = 16;
    constexpr size_t n_grow = 4;

    auto splat = create_adam_test_splat(n0);
    AdamConfig cfg;
    cfg.growth_factor = 1.5f;
    cfg.initial_capacity = 0;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();

    auto* state = opt.get_state_mutable(ParamType::Scaling);
    ASSERT_NE(state, nullptr);
    force_zero_capacity(*state);

    AdamOptimizer::reset_slow_path_grow_count();

    // Pre-reserve params so only state is forced slow.
    splat.means().reserve(n0 + 8 * n_grow);
    splat.sh0().reserve(n0 + 8 * n_grow);
    splat.scaling_raw().reserve(n0 + 8 * n_grow);
    splat.rotation_raw().reserve(n0 + 8 * n_grow);
    splat.opacity_raw().reserve(n0 + 8 * n_grow);

    auto indices = Tensor::arange(0.0f, static_cast<float>(n_grow), 1.0f)
                       .to(DataType::Int32)
                       .to(Device::CUDA);
    // extend_state_by_gather expects param already grown; grow param first then state.
    splat.scaling_raw().append_gather(indices);
    opt.extend_state_by_gather(ParamType::Scaling, indices);

    EXPECT_EQ(AdamOptimizer::slow_path_grow_count(), 1u);
    state = opt.get_state_mutable(ParamType::Scaling);
    ASSERT_NE(state, nullptr);
    EXPECT_GE(state->capacity, state->size);
    EXPECT_GT(state->exp_avg.capacity(), 0u);

    // Second gather grow — fast.
    const uint64_t slow_before = AdamOptimizer::slow_path_grow_count();
    const auto snap = alloc_counter::snapshot();
    splat.scaling_raw().append_gather(indices);
    opt.extend_state_by_gather(ParamType::Scaling, indices);
    EXPECT_EQ(AdamOptimizer::slow_path_grow_count(), slow_before);
    EXPECT_LE(alloc_counter::delta_since(snap), 2u);
    EXPECT_GE(state->capacity, state->size);
}
