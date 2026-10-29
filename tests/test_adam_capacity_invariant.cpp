/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <sstream>
#include <tuple>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    SplatData create_adam_test_splat(size_t n_points, int degree = 0) {
        auto means = Tensor::randn({n_points, 3}, Device::GPU);
        auto sh0 = Tensor::randn({n_points, 1, 3}, Device::GPU);
        auto shN = Tensor::full({n_points, sh_rest_coefficients_for_degree(degree), 3}, 0.05f, Device::GPU);
        auto scaling = Tensor::randn({n_points, 3}, Device::GPU);
        auto rotation = Tensor::randn({n_points, 4}, Device::GPU);
        auto opacity = Tensor::randn({n_points, 1}, Device::GPU);
        return SplatData(degree, means, sh0, shN, scaling, rotation, opacity, 1.0f);
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

class AdamCapacityInvariant : public lfs::test::CudaBackendTest {};

TEST_F(AdamCapacityInvariant, SlowPathReReservesSoSecondGrowIsFast) {
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

TEST_F(AdamCapacityInvariant, SlowPathGatherAlsoRestoresCapacity) {
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
                       .to(Device::GPU);
    // The model row count follows means, which a densify grow extends first.
    splat.means().append_gather(indices);
    opt.add_new_params_gather(ParamType::Scaling, indices);

    EXPECT_EQ(AdamOptimizer::slow_path_grow_count(), 1u);
    state = opt.get_state_mutable(ParamType::Scaling);
    ASSERT_NE(state, nullptr);
    EXPECT_GE(state->capacity, state->size);
    EXPECT_GT(state->exp_avg.capacity(), 0u);

    // Second gather grow — fast.
    const uint64_t slow_before = AdamOptimizer::slow_path_grow_count();
    const auto snap = alloc_counter::snapshot();
    splat.means().append_gather(indices);
    opt.add_new_params_gather(ParamType::Scaling, indices);
    EXPECT_EQ(AdamOptimizer::slow_path_grow_count(), slow_before);
    EXPECT_LE(alloc_counter::delta_since(snap), 2u);
    EXPECT_GE(state->capacity, state->size);
}

TEST_F(AdamCapacityInvariant, SlowPathGrowPreservesPackedMoments) {
    constexpr size_t n0 = 16;
    constexpr size_t n_grow = 4;

    auto splat = create_adam_test_splat(n0);
    AdamConfig cfg;
    cfg.growth_factor = 1.5f;
    cfg.initial_capacity = 0;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();

    auto* state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(state, nullptr);
    ASSERT_TRUE(state->exp_avg.is_valid());
    std::vector<std::uint8_t> packed_before(state->exp_avg.numel(), 0xA5);
    packed_before[0] = 0x11;
    packed_before[1] = 0x22;
    packed_before.back() = 0xFE;
    ASSERT_EQ(cudaMemcpy(state->exp_avg.data_ptr(), packed_before.data(),
                         packed_before.size(), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    force_zero_capacity(*state);
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

    state = opt.get_state_mutable(ParamType::Means);
    ASSERT_NE(state, nullptr);
    ASSERT_TRUE(state->exp_avg.is_valid());
    ASSERT_GE(state->exp_avg.numel(), packed_before.size());
    std::vector<std::uint8_t> packed_after(packed_before.size());
    ASSERT_EQ(cudaMemcpy(packed_after.data(), state->exp_avg.data_ptr(),
                         packed_after.size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(packed_after, packed_before)
        << "slow-path realloc must copy existing packed moments";
}

namespace {

    std::vector<uint8_t> adam_tensor_bytes(const Tensor& tensor) {
        if (!tensor.is_valid() || tensor.numel() == 0)
            return {};
        const auto host = tensor.cpu().contiguous();
        const auto* bytes = static_cast<const uint8_t*>(host.data_ptr());
        return {bytes, bytes + host.bytes()};
    }

    void expect_sh_capacity(const AdamParamState& state, size_t n, size_t cap, uint32_t rest) {
        const size_t cells = sh_swizzled_float_count(n, rest);
        const size_t cap_cells = sh_swizzled_float_count(cap, rest);
        EXPECT_EQ(state.size, cells);
        EXPECT_EQ(state.capacity, cap_cells);
        EXPECT_EQ(state.exp_avg.numel(), joint_adam::packed_bytes(cells, 8));
        EXPECT_EQ(state.exp_avg.capacity(), joint_adam::packed_bytes(cap_cells, 8));
        ASSERT_TRUE(state.grad.is_valid());
        EXPECT_EQ(state.grad.numel(), cells);
        EXPECT_EQ(state.grad.dtype(), DataType::Float32);
        EXPECT_EQ(std::max(state.grad.capacity(), state.grad.numel()), cap_cells);
        EXPECT_EQ(state.joint_bounds.shape()[0], joint_adam::n_bounds_for_prims(n));
        EXPECT_EQ(std::max(state.joint_bounds.capacity(), state.joint_bounds.shape()[0]),
                  joint_adam::n_bounds_for_prims(sh_swizzled_padded_n(cap)));
    }

} // namespace

class AdamShCapacity : public lfs::test::CudaBackendTest,
                       public ::testing::WithParamInterface<std::tuple<int, bool, bool, bool>> {};

TEST_P(AdamShCapacity, GrowthReserveAndResumePreserveUnitsAndValues) {
    using ParamType = lfs::training::ParamType;
    const auto [degree, gather, grad_before, slow] = GetParam();
    constexpr size_t n0 = 253;
    constexpr size_t added = 10;
    constexpr size_t n = n0 + added;
    constexpr size_t reserved = 769;
    const uint32_t rest = sh_rest_coefficients_for_degree(degree);
    auto splat = create_adam_test_splat(n0, degree);
    splat.reserve_capacity(reserved);
    AdamConfig cfg;
    cfg.initial_capacity = slow ? n0 + 1 : reserved;
    cfg.growth_factor = 1.5f;
    AdamOptimizer opt(splat, cfg);
    opt.allocate_gradients();
    auto* state = opt.get_state_mutable(ParamType::ShN);
    ASSERT_NE(state, nullptr);
    const auto param_before = adam_tensor_bytes(splat.shN_canonical());
    if (degree > 0) {
        state->step_count = 17;
        std::vector<uint8_t> codes(state->exp_avg.numel(), 91);
        // The state was zero-filled on a nonblocking tensor stream; legacy-stream copies do not wait for it.
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(state->exp_avg.data_ptr(), codes.data(), codes.size(), cudaMemcpyHostToDevice), cudaSuccess);
        state->joint_bounds.copy_from(Tensor::from_vector({-0.5f, 0.75f, 0.f, 20.f}, {1, 4}, Device::GPU));
        if (grad_before)
            opt.get_grad(ParamType::ShN).fill_(0.125f);
    }
    const auto packed_before = adam_tensor_bytes(state->exp_avg);
    const auto bounds_before = adam_tensor_bytes(state->joint_bounds);
    auto indices = Tensor::arange(0.f, static_cast<float>(added), 1.f).to(DataType::Int32).gpu();
    for (const auto type : {ParamType::Means, ParamType::Sh0, ParamType::Scaling,
                            ParamType::Rotation, ParamType::Opacity}) {
        opt.add_new_params_gather(type, indices);
    }
    if (gather) {
        opt.add_new_params_gather(ParamType::ShN, indices);
    } else {
        if (degree > 0) {
            auto& param = splat.shN();
            const size_t new_cells = sh_swizzled_float_count(n, rest);
            param.reserve(sh_swizzled_float_count(reserved, rest));
            param.append_zeros(new_cells - param.numel());
        }
        opt.extend_state_for_new_params(ParamType::ShN, added);
    }
    ASSERT_EQ(splat.size(), n);
    if (degree == 0) {
        EXPECT_EQ(state->size, 0u);
        EXPECT_EQ(state->capacity, 0u);
        EXPECT_FALSE(state->exp_avg.is_valid());
        EXPECT_FALSE(state->joint_bounds.is_valid());
        opt.reserve_capacity(reserved + 256);
        std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
        opt.serialize(stream);
        AdamOptimizer resumed(splat, cfg);
        ASSERT_NO_THROW(resumed.deserialize(stream));
        EXPECT_EQ(resumed.get_state(ParamType::ShN), nullptr);
        return;
    }
    opt.get_grad(ParamType::ShN);
    const size_t expected_cap = slow ? static_cast<size_t>(n * cfg.growth_factor) : reserved;
    expect_sh_capacity(*state, n, expected_cap, rest);
    EXPECT_EQ(state->step_count, 17);
    auto packed_after = adam_tensor_bytes(state->exp_avg);
    auto bounds_after = state->joint_bounds.cpu();
    const auto grad_after = state->grad.cpu();
    EXPECT_EQ(std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(bounds_after.data_ptr()),
                                   reinterpret_cast<const uint8_t*>(bounds_after.data_ptr()) + bounds_before.size()),
              bounds_before);
    auto old_bounds = Tensor::from_vector({-0.5f, 0.75f, 0.f, 20.f}, {4}, Device::CPU);
    for (size_t prim = 0; prim < n0; ++prim) {
        for (uint32_t cell = 0; cell < sh_float4_slots_for_rest(rest) * 4; ++cell) {
            const size_t offset = sh_swizzled_index(prim, cell / 4, rest) * 4 + cell % 4;
            EXPECT_EQ(grad_after.ptr<float>()[offset], grad_before ? 0.125f : 0.f);
            EXPECT_EQ(packed_after[offset * 2], packed_before[offset * 2]);
            EXPECT_EQ(packed_after[offset * 2 + 1], packed_before[offset * 2 + 1]);
            float m0, v0, m1, v1;
            const auto* mm = old_bounds.ptr<float>();
            joint_adam::Codec8::decode_g1g2(packed_before.data(), offset, mm[0], mm[1], mm[2], mm[3], m0, v0);
            mm = bounds_after.ptr<float>() + (prim / 256) * 4;
            joint_adam::Codec8::decode_g1g2(packed_after.data(), offset, mm[0], mm[1], mm[2], mm[3], m1, v1);
            EXPECT_EQ(m1, m0);
            EXPECT_EQ(v1, v0);
        }
    }
    const auto param_after = adam_tensor_bytes(splat.shN_canonical());
    ASSERT_GE(param_after.size(), param_before.size());
    EXPECT_TRUE(std::equal(param_before.begin(), param_before.end(), param_after.begin()));
    const auto fused = opt.prepare_fastgs_fused_adam(1001);
    const auto& fused_shN = fused.groups[static_cast<size_t>(lfs::gpu_ops::AdamSlot::ShN)];
    EXPECT_TRUE(fused_shN.enabled);
    EXPECT_EQ(static_cast<size_t>(fused_shN.primitives), n);
    expect_sh_capacity(*state, n, expected_cap, rest);
    EXPECT_EQ(state->step_count, 17);
    EXPECT_EQ(adam_tensor_bytes(state->exp_avg), packed_after);
    const auto grad_before_reserve = adam_tensor_bytes(state->grad);
    const auto bounds_before_reserve = adam_tensor_bytes(state->joint_bounds);
    constexpr size_t larger_cap = 1281;
    opt.reserve_capacity(larger_cap);
    expect_sh_capacity(*state, n, larger_cap, rest);
    EXPECT_EQ(adam_tensor_bytes(state->grad), grad_before_reserve);
    EXPECT_EQ(adam_tensor_bytes(state->exp_avg), packed_after);
    EXPECT_EQ(adam_tensor_bytes(state->joint_bounds), bounds_before_reserve);
    EXPECT_EQ(adam_tensor_bytes(splat.shN_canonical()), param_after);

    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    const size_t capacity_before_save = state->capacity;
    state->capacity = state->exp_avg.capacity();
    opt.serialize(stream);
    state->capacity = capacity_before_save;
    AdamOptimizer resumed(splat, cfg);
    ASSERT_NO_THROW(resumed.deserialize(stream));
    auto* loaded = resumed.get_state_mutable(ParamType::ShN);
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->capacity, sh_swizzled_float_count(n, rest));
    EXPECT_EQ(loaded->size, state->size);
    EXPECT_EQ(loaded->step_count, 17);
    EXPECT_EQ(adam_tensor_bytes(loaded->exp_avg), packed_after);
    EXPECT_EQ(adam_tensor_bytes(loaded->joint_bounds), bounds_before_reserve);
    if (grad_before) {
        resumed.get_grad(ParamType::ShN);
        EXPECT_EQ(std::max(loaded->grad.capacity(), loaded->grad.numel()), loaded->size);
    }
    resumed.reserve_capacity(larger_cap);
    resumed.get_grad(ParamType::ShN);
    expect_sh_capacity(*loaded, n, larger_cap, rest);
    EXPECT_EQ(adam_tensor_bytes(loaded->exp_avg), packed_after);
    EXPECT_EQ(adam_tensor_bytes(loaded->joint_bounds), bounds_before_reserve);
    EXPECT_EQ(adam_tensor_bytes(splat.shN_canonical()), param_after);
}

INSTANTIATE_TEST_SUITE_P(DegreesAndGrowthPaths, AdamShCapacity,
                         ::testing::Combine(::testing::Values(0, 1, 2, 3), ::testing::Bool(),
                                            ::testing::Bool(), ::testing::Bool()));
