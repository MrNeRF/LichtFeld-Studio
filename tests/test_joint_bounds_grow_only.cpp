/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Joint Adam bounds must be grow-only across densification-like growth. Reusing
 * pre-reserved capacity must not allocate from the driver.
 */

#include "core/alloc_counter.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <array>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;

TEST(JointBoundsGrowOnly, EnsureWithinCapacityIsAllocFree) {
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
}

TEST(JointBoundsGrowOnly, MultiParamCompactZeroReusesCapacity) {
    // Six joint parameter groups each retain a bounds table. zero_all must reuse
    // those allocations after the initial reservation.
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
    EXPECT_EQ(delta, 0u) << "15 densify compact zero_all rounds must not driver-alloc; got "
                         << delta;
}
