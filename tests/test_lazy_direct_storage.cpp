/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "core/lazy_direct_storage.hpp"
#include "core/memory_pressure.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/strategies/strategy_utils.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace lfs::core;

namespace {
    constexpr size_t kMiB = size_t{1} << 20;

    size_t free_vram() {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        EXPECT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
        return free_bytes;
    }

    // Device memory taken outside the tensor pool, so the pool keeps nothing
    // cached for later tests.
    struct VramHog {
        void* ptr = nullptr;
        explicit VramHog(const size_t bytes) {
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
                ptr = nullptr;
                (void)cudaGetLastError();
            }
        }
        ~VramHog() {
            if (ptr) {
                cudaFree(ptr);
            }
        }
        VramHog(const VramHog&) = delete;
        VramHog& operator=(const VramHog&) = delete;
    };

    SplatData make_splat(const size_t n) {
        std::vector<float> rotation(n * 4, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            rotation[i * 4] = 1.0f;
        }
        return SplatData(3,
                         Tensor::zeros({n, 3}, Device::CUDA),
                         Tensor::zeros({n, 3}, Device::CUDA),
                         Tensor::zeros({n, 48}, Device::CUDA),
                         Tensor::full({n, 3}, -2.0f, Device::CUDA),
                         Tensor::from_vector(rotation, TensorShape({n, 4}), Device::CUDA),
                         Tensor::zeros({n, 1}, Device::CUDA),
                         1.0f);
    }
} // namespace

// A max_cap-sized reservation must not take device memory for rows that do
// not exist yet.
TEST(LazyDirectStorage, ReservationCommitsOnlyLiveRows) {
    const size_t live_before = Tensor::cuda_direct_storage_live_bytes();
    const size_t free_before = free_vram();

    // 1.5 GiB of capacity, 12 KiB of rows.
    auto t = Tensor::zeros_direct({1024, 3}, size_t{128} * kMiB, Device::CUDA);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_LT(Tensor::cuda_direct_storage_live_bytes() - live_before, 64 * kMiB);
    EXPECT_LT(free_before - std::min(free_before, free_vram()), 64 * kMiB);
    EXPECT_EQ(t.capacity(), size_t{128} * kMiB);
    EXPECT_LT(t.committed_rows(), t.capacity());
    EXPECT_GE(t.committed_rows(), size_t{1024});
    EXPECT_FLOAT_EQ(t.sum_scalar(), 0.0f);
}

// Pool tensors (the loaded point cloud) take their max_cap headroom through
// reserve; it must not commit the headroom either.
TEST(LazyDirectStorage, ReserveKeepsDataAndCommitsOnlyLiveRows) {
    std::vector<float> rows(3000);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = static_cast<float>(i);
    }
    auto t = Tensor::from_vector(rows, TensorShape({1000, 3}), Device::CUDA);
    const size_t live_before = Tensor::cuda_direct_storage_live_bytes();

    t.reserve(size_t{64} * kMiB);

    EXPECT_EQ(t.capacity(), size_t{64} * kMiB);
    EXPECT_LT(t.committed_rows(), t.capacity());
    EXPECT_LT(Tensor::cuda_direct_storage_live_bytes() - live_before, 64 * kMiB);
    EXPECT_EQ(t.cpu().to_vector(), rows);
    t.append_zeros(size_t{16} * kMiB);
    EXPECT_GE(t.committed_rows(), size_t{16} * kMiB + 1000);
    EXPECT_FLOAT_EQ(t.slice(0, 1000, t.shape()[0]).sum_scalar(), 0.0f);
}

TEST(LazyDirectStorage, AppendZerosGrowsInPlace) {
    auto t = Tensor::zeros_direct({1000, 3}, 64 * kMiB, Device::CUDA);
    t.fill_(1.0f);
    const void* base = t.data_ptr();
    const size_t committed = t.committed_rows();

    const size_t grow = 20 * kMiB;
    t.append_zeros(grow);

    EXPECT_EQ(t.data_ptr(), base);
    EXPECT_EQ(t.shape()[0], 1000 + grow);
    EXPECT_GE(t.committed_rows(), 1000 + grow);
    EXPECT_GT(t.committed_rows(), committed);
    EXPECT_FLOAT_EQ(t.sum_scalar(), 3000.0f);
    EXPECT_FLOAT_EQ(t.slice(0, 0, 1000).sum_scalar(), 3000.0f);
}

TEST(LazyDirectStorage, AppendGatherGrowsInPlace) {
    std::vector<float> rows(4096 * 4);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = static_cast<float>(i / 4);
    }
    auto src = Tensor::from_vector(rows, TensorShape({4096, 4}), Device::CUDA);
    auto t = Tensor::zeros_direct({4096, 4}, 32 * kMiB, Device::CUDA);
    t.copy_from(src);
    const void* base = t.data_ptr();

    // Duplicate every row many times so the gather crosses the committed prefix.
    std::vector<int> indices(8 * kMiB);
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(i % 4096);
    }
    t.append_gather(Tensor::from_vector(indices, TensorShape({indices.size()}), Device::CUDA));

    EXPECT_EQ(t.data_ptr(), base);
    ASSERT_EQ(t.shape()[0], 4096 + indices.size());
    const auto tail = t.slice(0, t.shape()[0] - 4096, t.shape()[0]).cpu().to_vector();
    const size_t first_index = (indices.size() - 4096) % 4096;
    for (size_t r = 0; r < 4096; ++r) {
        ASSERT_FLOAT_EQ(tail[r * 4], static_cast<float>((first_index + r) % 4096));
    }
}

TEST(LazyDirectStorage, InPlaceCatCommitsTheGrownRows) {
    auto t = Tensor::zeros_direct({16, 2}, 16 * kMiB, Device::CUDA);
    t.fill_(2.0f);
    const auto extra = Tensor::full({8 * kMiB, 2}, 3.0f, Device::CUDA);

    const auto joined = Tensor::cat({t, extra}, 0);

    EXPECT_EQ(joined.data_ptr(), t.data_ptr());
    ASSERT_EQ(joined.shape()[0], 16 + 8 * kMiB);
    EXPECT_FLOAT_EQ(joined.slice(0, 0, 16).sum_scalar(), 64.0f);
    const auto tail = joined.slice(0, joined.shape()[0] - 1024, joined.shape()[0]).cpu().to_vector();
    for (const float v : tail) {
        ASSERT_FLOAT_EQ(v, 3.0f);
    }
}

// The Adam HEAL path advances optimizer size over reserved bytes without an
// append; it relies on the tail being backed and zero.
TEST(LazyDirectStorage, CommitRowsBacksZeroedTail) {
    auto t = Tensor::zeros_direct({16}, 512 * kMiB, Device::CUDA, DataType::UInt8);
    const size_t rows = 96 * kMiB;
    t.commit_rows(rows);
    ASSERT_GE(t.committed_rows(), rows);

    std::vector<unsigned char> host(4096, 0xab);
    const auto* tail = static_cast<const unsigned char*>(t.data_ptr()) + rows - host.size();
    ASSERT_EQ(cudaStreamSynchronize(t.stream()), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(host.data(), tail, host.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    for (const unsigned char byte : host) {
        ASSERT_EQ(byte, 0);
    }
}

// Pages committed ahead of the rows go back to the device under pressure; the
// rows keep their values and growth maps zeroed pages again.
TEST(LazyDirectStorage, ReleaseAheadKeepsRowsAndGrowsAgain) {
    constexpr size_t kRowBytes = 4 * sizeof(float);
    constexpr size_t kPaddingRows = (4 * kMiB) / kRowBytes;
    auto t = Tensor::zeros_direct({16, 4}, size_t{64} << 20, Device::CUDA);
    t.fill_(1.0f);
    t.append_zeros(size_t{16} << 20);
    const size_t rows = t.shape()[0];
    ASSERT_GE(t.committed_rows() - rows, (16 * kMiB) / kRowBytes) << "no pages committed ahead";

    EXPECT_GE(release_lazy_direct_storage_ahead(), 16 * kMiB);
    EXPECT_LE(t.committed_rows() - rows, kPaddingRows);
    EXPECT_FLOAT_EQ(t.slice(0, 0, 16).sum_scalar(), 64.0f);

    t.append_zeros(size_t{1} << 20);
    EXPECT_GE(t.committed_rows(), t.shape()[0]);
    EXPECT_FLOAT_EQ(t.sum_scalar(), 64.0f);
}

TEST(LazyDirectStorage, SmallReservationIsFullyBacked) {
    auto t = Tensor::zeros_direct({10, 3}, 1000, Device::CUDA);
    EXPECT_EQ(t.committed_rows(), t.capacity());
    t.append_zeros(990);
    EXPECT_FLOAT_EQ(t.sum_scalar(), 0.0f);
}

TEST(VramGrowthLimit, ShrinksWithFreeMemory) {
    // Each Gaussian is also charged its share of the rasterizer arena earlier
    // tests left committed; enough of them keep that share small.
    auto splat = make_splat(4000000);
    lfs::training::initialize_gaussians(splat, 0);
    const size_t before = lfs::training::vram_growth_limit(splat, nullptr);
    ASSERT_GT(before, size_t{0});

    const VramHog hog(size_t{1} << 30);
    if (!hog.ptr) {
        GTEST_SKIP() << "not enough free VRAM";
    }
    const size_t after = lfs::training::vram_growth_limit(splat, nullptr);

    EXPECT_LT(after, before);
    // 1 GiB at well under 1 KiB per Gaussian is more than a million Gaussians.
    EXPECT_GT(before - after, size_t{1000000});
}

// Growth must stop before the rasterizer runs out of room, not fail in it.
TEST(VramGrowthLimit, NoGrowthWhenOnlyTheReserveIsFree) {
    auto splat = make_splat(100000);
    lfs::training::initialize_gaussians(splat, 0);
    const size_t free_bytes = free_vram();
    const size_t keep = 512 * kMiB;
    ASSERT_GT(free_bytes, keep);
    const VramHog hog(free_bytes - keep);
    if (!hog.ptr) {
        GTEST_SKIP() << "free VRAM changed while filling it";
    }

    EXPECT_EQ(lfs::training::vram_growth_limit(splat, nullptr), size_t{0});
    EXPECT_EQ(lfs::training::clamp_growth_to_vram(5000, splat, nullptr), size_t{0});
}

// Commit-ahead pages serve only their own tensor; near the limit growth must
// get them back rather than count them as free.
TEST(VramGrowthLimit, HandsBackCommitAheadNearTheLimit) {
    auto splat = make_splat(100000);
    lfs::training::initialize_gaussians(splat, 0);
    constexpr size_t kRowBytes = 4 * sizeof(float);
    auto t = Tensor::zeros_direct({16, 4}, size_t{64} << 20, Device::CUDA);
    t.append_zeros(size_t{32} << 20);
    ASSERT_GE(t.committed_rows() - t.shape()[0], (32 * kMiB) / kRowBytes) << "no pages committed ahead";

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    // Leave less room than the pages committed ahead.
    const size_t keep = 256 * kMiB + RasterizerMemoryArena::growth_headroom_bytes(total_bytes) + 32 * kMiB;
    std::vector<std::unique_ptr<VramHog>> fill;
    for (size_t chunk = 256 * kMiB; chunk >= 2 * kMiB;) {
        if (free_vram() < keep + chunk) {
            chunk /= 2;
            continue;
        }
        fill.push_back(std::make_unique<VramHog>(chunk));
        if (!fill.back()->ptr) {
            fill.pop_back();
            chunk /= 2;
        }
    }

    (void)lfs::training::vram_growth_limit(splat, nullptr);
    EXPECT_LE(t.committed_rows() - t.shape()[0], (4 * kMiB) / kRowBytes);
}

// A refine that briefly takes VRAM makes later growth keep that much free,
// scaled to the grown model.
TEST(VramGrowthLimit, PlansForTheMeasuredRefineTransient) {
    auto splat = make_splat(1000000);
    lfs::training::initialize_gaussians(splat, 0);
    Tensor::trim_memory_pool();
    const size_t before = lfs::training::vram_growth_limit(splat, nullptr);
    {
        const lfs::training::RefineVramWindow window(splat);
        auto transient = Tensor::empty({size_t{512} * kMiB}, Device::CUDA, DataType::UInt8);
        transient.zero_();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    Tensor::trim_memory_pool();

    // The 256 MiB growth reserve covers the first part of any refine.
    EXPECT_GE(lfs::training::refine_transient_bytes_per_gaussian(), size_t{256} * kMiB / 1000000);
    EXPECT_GE(lfs::training::tensor_pool_reserved_high_bytes(), size_t{512} * kMiB);
    EXPECT_LT(lfs::training::vram_growth_limit(splat, nullptr), before);
}

// A refine that fit only after pressure relief planned too little: growth
// keeps more free from then on, as after a recovered forward OOM.
TEST(VramGrowthLimit, RefineThatNeededReliefRaisesTheReserve) {
    auto splat = make_splat(4000000);
    lfs::training::initialize_gaussians(splat, 0);
    auto& pressure = lfs::core::MemoryPressureCoordinator::instance();
    const auto refine = [&](const bool short_of_vram) {
        {
            const lfs::training::RefineVramWindow window(splat);
            // The same transient in both refines keeps their plans equal, so
            // only the reserve can tell them apart.
            auto transient = Tensor::empty({size_t{512} * kMiB}, Device::CUDA, DataType::UInt8);
            transient.zero_();
            EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            if (short_of_vram) {
                // An open frame keeps the arena out of the relief, so only the
                // reserve can lower the limit.
                auto& arena = GlobalArenaManager::instance().get_arena();
                const auto frame = arena.begin_frame(nullptr);
                pressure.set_free_memory_probe([](lfs::core::MemoryDomain) -> size_t { return 0; });
                (void)pressure.run_episode(lfs::core::AllocationFailure{
                    .domain = lfs::core::MemoryDomain::CudaDevice,
                    .requested_bytes = kMiB,
                });
                pressure.set_free_memory_probe({});
                arena.end_frame(frame, nullptr);
            }
        }
        Tensor::trim_memory_pool();
        return lfs::training::vram_growth_limit(splat, nullptr);
    };
    const size_t after_plain_refine = refine(false);
    const size_t after_relieved_refine = refine(true);
    lfs::training::initialize_gaussians(splat, 0);
    EXPECT_LT(after_relieved_refine, after_plain_refine);
}

// A refine short of VRAM may take the idle rasterizer arena, but that memory
// is owed back to the next frame: growth must not spend it.
TEST(VramGrowthLimit, ArenaReleasedUnderPressureStaysBudgeted) {
    auto& arena = GlobalArenaManager::instance().get_arena();
    const auto grow_arena = [&arena](const size_t bytes) {
        const auto frame = arena.begin_frame(nullptr);
        const bool grown = arena.get_allocator(frame, "test.arena_grow")(bytes) != nullptr;
        arena.end_frame(frame, nullptr);
        return grown;
    };
    if (!grow_arena(256 * kMiB)) {
        GTEST_SKIP() << "not enough free VRAM";
    }
    ASSERT_GE(arena.get_memory_info().arena_capacity, 256 * kMiB);
    {
        const auto held = arena.begin_frame(nullptr);
        EXPECT_EQ(arena.release_idle_for_pressure(), size_t{0}) << "released a frame in use";
        arena.end_frame(held, nullptr);
    }

    auto splat = make_splat(100000);
    lfs::training::initialize_gaussians(splat, 0);
    const size_t before = lfs::training::vram_growth_limit(splat, nullptr);
    const size_t released = arena.release_idle_for_pressure();
    EXPECT_GE(released, 256 * kMiB);
    EXPECT_GE(arena.get_memory_info().recommit_bytes, released);
    // 256 MiB is far more than 100k Gaussians.
    EXPECT_LT(lfs::training::vram_growth_limit(splat, nullptr), before + 100000);

    // A lighter view leaves part of it owed; the earlier high-water stays.
    {
        const auto frame = arena.begin_frame(nullptr);
        ASSERT_NE(arena.get_allocator(frame, "test.arena_light")(64 * kMiB), nullptr);
        arena.end_frame(frame, nullptr);
    }
    EXPECT_GE(arena.get_memory_info().recommit_bytes, 128 * kMiB);
    EXPECT_GE(arena.get_memory_info().peak_usage, 256 * kMiB);
    ASSERT_TRUE(grow_arena(arena.get_memory_info().peak_usage));
    EXPECT_EQ(arena.get_memory_info().recommit_bytes, size_t{0});
}

// Growth right after a refine took the idle arena must not commit ahead into
// the memory the next frame takes back.
TEST(LazyDirectStorage, NoCommitAheadIntoArenaOwedMemory) {
    auto& arena = GlobalArenaManager::instance().get_arena();
    {
        const auto frame = arena.begin_frame(nullptr);
        const bool grown = arena.get_allocator(frame, "test.arena_grow")(512 * kMiB) != nullptr;
        arena.end_frame(frame, nullptr);
        if (!grown) {
            GTEST_SKIP() << "not enough free VRAM";
        }
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    // Free VRAM beside the owed arena is just short of what committing ahead
    // keeps clear; counting the owed arena as free would pass that check.
    const size_t keep = total_bytes / 8 + 2 * RasterizerMemoryArena::growth_headroom_bytes(total_bytes) - 64 * kMiB;
    std::vector<std::unique_ptr<VramHog>> fill;
    for (size_t chunk = 256 * kMiB; chunk >= 2 * kMiB;) {
        if (free_vram() < keep + chunk) {
            chunk /= 2;
            continue;
        }
        fill.push_back(std::make_unique<VramHog>(chunk));
        if (!fill.back()->ptr) {
            fill.pop_back();
            chunk /= 2;
        }
    }
    ASSERT_GE(arena.release_idle_for_pressure(), 512 * kMiB);

    auto t = Tensor::zeros_direct({16, 4}, 64 * kMiB, Device::CUDA);
    t.append_zeros(size_t{8} << 20);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    // The live rows plus the padding granule, rounded to a commit granule.
    EXPECT_LE(t.committed_rows(), (size_t{8} << 20) + 16 + (4 * kMiB) / (4 * sizeof(float)));

    fill.clear();
    const auto frame = arena.begin_frame(nullptr);
    EXPECT_NE(arena.get_allocator(frame, "test.arena_recommit")(arena.get_memory_info().peak_usage), nullptr);
    arena.end_frame(frame, nullptr);
    EXPECT_EQ(arena.get_memory_info().recommit_bytes, size_t{0});
}
