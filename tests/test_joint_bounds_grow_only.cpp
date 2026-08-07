/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * WO-X — joint Adam bounds must be grow-only across densify-like growth.
 *
 * Pre-fix: every extend/compact path did Tensor::zeros([nb,4]) → one driver
 * alloc per param group per densify event (~0.07 of the 0.13 allocs/iter drift).
 * Post-fix: second growth within pre-reserved capacity does zero driver allocs.
 */

#include "core/alloc_counter.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <array>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;

namespace {

    SplatData make_splat(const size_t n) {
        auto means = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({n, size_t{15}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({n, size_t{3}}, Device::CUDA, DataType::Float32);
        auto rotation = Tensor::zeros({n, size_t{4}}, Device::CUDA, DataType::Float32);
        {
            auto cpu = rotation.cpu();
            auto* r = cpu.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                r[i * 4] = 1.0f;
            }
            rotation = cpu.to(Device::CUDA);
        }
        auto opacity = Tensor::zeros({n, size_t{1}}, Device::CUDA, DataType::Float32);
        return SplatData(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

} // namespace

TEST(JointBoundsGrowOnly, EnsureWithinCapacityIsAllocFree) {
    joint_adam::set_joint_codec_enabled_for_testing(true);

    Tensor bounds;
    // First call may allocate (capacity for 1024 prims = 4 bounds rows).
    ensure_joint_bounds_capacity(bounds, /*n_prims=*/256, /*capacity_prims=*/1024,
                                 Device::CUDA, /*zero_all=*/false);
    ASSERT_TRUE(bounds.is_valid());
    ASSERT_GE(bounds.capacity(), 4u);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto snap = alloc_counter::snapshot();
    // Grow logical N within capacity: 256 → 512 → 768 → 1024 (still 4 rows).
    for (size_t n : {512u, 768u, 1024u}) {
        ensure_joint_bounds_capacity(bounds, n, 1024, Device::CUDA, false);
    }
    // Compact-style zero-all must also reuse storage.
    ensure_joint_bounds_capacity(bounds, 512, 1024, Device::CUDA, /*zero_all=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_EQ(delta, 0u) << "grow-only joint_bounds must not driver-alloc within capacity";

    joint_adam::set_joint_codec_enabled_for_testing(std::nullopt);
}

TEST(JointBoundsGrowOnly, MultiParamCompactZeroReusesCapacity) {
    joint_adam::set_joint_codec_enabled_for_testing(true);

    // Six joint param groups (means/sh0/shN/scale/rot/opacity) each keep a
    // bounds table. Compact used to Tensor::zeros every refine (= 6 driver
    // allocs). Grow-only zero_all must be free after first allocate.
    std::array<Tensor, 6> bounds{};
    constexpr size_t kCap = 500000;
    for (auto& b : bounds) {
        ensure_joint_bounds_capacity(b, /*n_prims=*/100000, kCap, Device::CUDA, false);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto snap = alloc_counter::snapshot();
    // Simulate 15 densify refine events (bonsai 2000-iter schedule ≈ 15).
    for (int refine = 0; refine < 15; ++refine) {
        const size_t n = 100000 + static_cast<size_t>(refine) * 20000;
        for (auto& b : bounds) {
            ensure_joint_bounds_capacity(b, n, kCap, Device::CUDA, /*zero_all=*/true);
        }
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    // Pre-fix: 6 × 15 = 90 driver allocs. Post-fix: 0.
    EXPECT_EQ(delta, 0u) << "15 densify compact zero_all rounds must not driver-alloc; got "
                         << delta;

    joint_adam::set_joint_codec_enabled_for_testing(std::nullopt);
}
