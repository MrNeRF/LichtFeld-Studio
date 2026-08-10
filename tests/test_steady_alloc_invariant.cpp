/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/perf_bench.hpp"
#include "training/optimizer/adam_optimizer.hpp"

#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;
using namespace lfs::training;
namespace {

    constexpr double kSteadyAllocBudget = 0.06;

} // namespace

TEST(PerfBenchPeakCover, UsesOneSnapshotAndLiveIoBytes) {
    lfs::diagnostics::VramProfilerSnapshot snapshot;
    snapshot.process.cuda_pool_bucket_cache_bytes = 11;
    snapshot.process.cuda_pool_bucket_live_waste_bytes = 22;
    snapshot.process.exportable_splat_bytes = 33;
    snapshot.rows = {
        {.scope = "io.nvimagecodec", .live_bytes = 40, .peak_bytes = 400},
        {.scope = "io.image_loader", .live_bytes = 50, .peak_bytes = 500},
        {.scope = "train", .label = "not_io", .live_bytes = 60, .peak_bytes = 600},
    };

    const auto sample =
        lfs::training::detail::collect_perf_peak_cover_sample(snapshot);
    EXPECT_EQ(sample.pool_bucket_cache_bytes, 11u);
    EXPECT_EQ(sample.pool_bucket_live_waste_bytes, 22u);
    EXPECT_EQ(sample.exportable_splat_bytes, 33u);
    EXPECT_EQ(sample.io_external_bytes, 90u)
        << "historical per-row peaks are not a concurrent I/O cover";
}

TEST(SteadyAllocInvariant, JointDensifySteadyLoopWithinBudget) {
    alloc_counter::reset_site_counts();

    // Fifteen refinements across six bounds tables must remain within the
    // 0.06-allocation average over 1800 steady iterations.
    constexpr size_t kCap = 500000;
    constexpr int kRefines = 15; // bonsai 2000-iter densify count
    constexpr int kSteadySteps = 1800;
    std::array<Tensor, 6> bounds{};
    for (auto& b : bounds) {
        ensure_joint_bounds_capacity(b, 50000, kCap, Device::CUDA, false);
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto snap = alloc_counter::snapshot();
    for (int r = 0; r < kRefines; ++r) {
        alloc_counter::ScopedSite densify("densify");
        const size_t n = 50000 + static_cast<size_t>(r) * 25000;
        for (auto& b : bounds) {
            ensure_joint_bounds_capacity(b, n, kCap, Device::CUDA, false);
            ensure_joint_bounds_capacity(b, n, kCap, Device::CUDA, true);
        }
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto delta = alloc_counter::delta_since(snap);
    const double allocs_per_iter =
        static_cast<double>(delta) / static_cast<double>(kSteadySteps);

    EXPECT_LE(allocs_per_iter, kSteadyAllocBudget)
        << "steady densify allocs/iter=" << allocs_per_iter
        << " total_delta=" << delta;
    EXPECT_EQ(delta, 0u);
}

TEST(AllocCounterSiteTags, RecordSiteIncrementsPerSite) {
    alloc_counter::reset_site_counts();
    const auto before = alloc_counter::site_count(alloc_counter::Site::ZerosDirect);
    alloc_counter::record_site(alloc_counter::Site::ZerosDirect, 3);
    EXPECT_EQ(alloc_counter::site_count(alloc_counter::Site::ZerosDirect), before + 3u);
    EXPECT_STREQ(alloc_counter::site_name(alloc_counter::Site::PoolBucket), "pool_bucket");

    {
        alloc_counter::ScopedSite a("densify");
        EXPECT_STREQ(alloc_counter::current_logical_site(), "densify");
        {
            alloc_counter::ScopedSite b("joint_bounds");
            EXPECT_STREQ(alloc_counter::current_logical_site(), "joint_bounds");
        }
        EXPECT_STREQ(alloc_counter::current_logical_site(), "densify");
    }
    EXPECT_STREQ(alloc_counter::current_logical_site(), "");
}
