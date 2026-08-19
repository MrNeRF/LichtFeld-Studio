/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/exportable_storage.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/training_setup.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

using namespace lfs::core;

namespace {

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    void fill_device_pattern(void* device_ptr, std::size_t floats, float base) {
        std::vector<float> host(floats);
        for (std::size_t i = 0; i < floats; ++i) {
            host[i] = base + static_cast<float>(i);
        }
        ASSERT_EQ(cudaMemcpy(device_ptr, host.data(), floats * sizeof(float), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    void expect_device_pattern(const void* device_ptr, std::size_t floats, float base) {
        std::vector<float> host(floats);
        ASSERT_EQ(cudaMemcpy(host.data(), device_ptr, floats * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (std::size_t i = 0; i < floats; ++i) {
            EXPECT_FLOAT_EQ(host[i], base + static_cast<float>(i)) << "index " << i;
        }
    }

} // namespace

TEST(ExportableStorageTest, ImmediateDestroyLeavesCudaUsable) {
    require_cuda();

    constexpr std::size_t BLOCK_BYTES = 1 << 20;
    auto block_result = allocateExportableDeviceBlock(BLOCK_BYTES, 0, false);
    if (!block_result) {
        FAIL() << block_result.error();
    }

    auto block = std::move(*block_result);
    ASSERT_NE(block, nullptr);
    ASSERT_NE(block->device_ptr, nullptr);
    block.reset();

    constexpr std::size_t PROBE_BYTES = 4096;
    void* probe = nullptr;
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMalloc(&probe, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "allocating unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaMemset(probe, 0xa5, PROBE_BYTES),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "writing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaDeviceSynchronize(),
        reinterpret_cast<uintptr_t>(probe),
        0,
        PROBE_BYTES,
        "synchronizing unrelated CUDA probe dst={} src={} bytes={}",
        probe,
        static_cast<const void*>(nullptr),
        PROBE_BYTES);
    LFS_CUDA_CHECK_MSG_ARGS(
        cudaFree(probe),
        0,
        reinterpret_cast<uintptr_t>(probe),
        PROBE_BYTES,
        "freeing unrelated CUDA probe dst={} src={} bytes={}",
        static_cast<const void*>(nullptr),
        probe,
        PROBE_BYTES);
}

// ---------------------------------------------------------------------------
// exportable splat block grows with live N
// ---------------------------------------------------------------------------

TEST(SplatExportableStorageTest, CreateTracksLiveCapacityNotMaxCap) {
    require_cuda();

    constexpr std::size_t kLive = 1024;
    constexpr std::size_t kMaxCap = 5'000'000;
    constexpr int kShDegree = 3;

    const std::size_t live_bytes = SplatExportableStorage::layoutBytes(kLive, kShDegree);
    const std::size_t max_bytes = SplatExportableStorage::layoutBytes(kMaxCap, kShDegree);
    ASSERT_GT(max_bytes, live_bytes);
    ASSERT_GT(max_bytes - live_bytes, 100ull << 20); // hundreds of MiB at 5M SH3

    auto storage_result = SplatExportableStorage::create(kLive, kShDegree, 0, kMaxCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.valid());
    EXPECT_EQ(storage.capacity(), kLive);
    EXPECT_EQ(storage.reservedCapacity(), kMaxCap);
    EXPECT_GE(storage.block->size, live_bytes);
    // Committed physical must track live N, not max_cap.
    EXPECT_LT(storage.block->size, max_bytes / 4);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, storage.block->size);
    EXPECT_LT(snap.process.exportable_splat_bytes, max_bytes / 4);
}

TEST(SplatExportableStorageTest, LayoutBytesRejectsRegionSizeOverflow) {
    constexpr std::size_t kPerPrimitiveBytes = 3 * sizeof(float);
    const std::size_t overflowing_capacity =
        std::numeric_limits<std::size_t>::max() / kPerPrimitiveBytes + 1;

    try {
        (void)SplatExportableStorage::layoutBytes(overflowing_capacity, /*sh_degree=*/0);
        FAIL() << "overflowing exportable region size was accepted";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string_view(error.what()).find("exportable splat region"),
                  std::string_view::npos);
    }
}

TEST(SplatExportableStorageTest, GrowPreservesDataAndTracksBytes) {
    require_cuda();

    constexpr std::size_t kInitial = 256;
    constexpr std::size_t kGrown = 512;
    constexpr std::size_t kReserve = 4096;
    constexpr int kShDegree = 3;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.valid());
    const void* const stable_ptr = storage.block->device_ptr;
    const std::size_t bytes_before = storage.block->size;
    const auto gen_before = storage.generation();

    // Write a known pattern into the means region (first kInitial * 3 floats).
    constexpr std::size_t kMeansFloats = kInitial * 3;
    fill_device_pattern(storage.block->device_ptr, kMeansFloats, 10.0f);

    // Also stamp scaling region so relocation is tested for a non-zero offset.
    void* scaling_ptr =
        static_cast<char*>(storage.block->device_ptr) + storage.region_offsets[SplatExportableStorage::Scaling];
    constexpr std::size_t kScalingFloats = kInitial * 3;
    fill_device_pattern(scaling_ptr, kScalingFloats, 100.0f);

    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);
    EXPECT_EQ(storage.capacity(), kGrown);
    EXPECT_EQ(storage.block->device_ptr, stable_ptr) << "device_ptr must stay stable across grow";
    EXPECT_GT(storage.generation(), gen_before);
    EXPECT_GE(storage.block->size, bytes_before);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, storage.block->size);
    EXPECT_GE(snap.process.exportable_splat_bytes, SplatExportableStorage::layoutBytes(kGrown, kShDegree));

    expect_device_pattern(storage.block->device_ptr, kMeansFloats, 10.0f);
    void* scaling_after =
        static_cast<char*>(storage.block->device_ptr) + storage.region_offsets[SplatExportableStorage::Scaling];
    expect_device_pattern(scaling_after, kScalingFloats, 100.0f);

    // Idempotent: grow to same capacity is a no-op.
    auto grew_again = storage.grow(kGrown);
    ASSERT_TRUE(grew_again.has_value());
    EXPECT_FALSE(*grew_again);
}

TEST(SplatExportableStorageTest, MidRelocateFailureRestoresPreviousLayout) {
    require_cuda();

    constexpr std::size_t kInitial = 128;
    constexpr std::size_t kGrown = 512;
    constexpr int kShDegree = 3;
    constexpr unsigned char kPattern = 0x5a;

    auto storage_result =
        SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    ASSERT_TRUE(storage_result.has_value()) << storage_result.error();
    auto storage = std::move(*storage_result);

    const std::size_t old_total = SplatExportableStorage::layoutBytes(kInitial, kShDegree);
    const auto old_offsets = storage.region_offsets;
    const auto old_bytes = storage.region_bytes;
    const auto old_generation = storage.generation();
    ASSERT_EQ(cudaMemset(storage.block->device_ptr, kPattern, old_total), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    set_splat_exportable_relocate_failure_for_testing(SplatExportableStorage::Rotation);
    const auto grew = storage.grow(kGrown);
    set_splat_exportable_relocate_failure_for_testing(std::nullopt);

    ASSERT_FALSE(grew.has_value());
    EXPECT_NE(grew.error().find("injected region"), std::string::npos) << grew.error();
    EXPECT_NE(grew.error().find("previous layout restored"), std::string::npos) << grew.error();
    EXPECT_TRUE(storage.valid());
    EXPECT_FALSE(storage.poisoned());
    EXPECT_EQ(storage.capacity(), kInitial);
    EXPECT_EQ(storage.generation(), old_generation);
    EXPECT_EQ(storage.region_offsets, old_offsets);
    EXPECT_EQ(storage.region_bytes, old_bytes);

    std::vector<unsigned char> restored(old_total);
    ASSERT_EQ(cudaMemcpy(restored.data(), storage.block->device_ptr, old_total,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_TRUE(std::all_of(restored.begin(), restored.end(), [](const unsigned char value) {
        return value == kPattern;
    }));
}

TEST(SplatExportableStorageTest, TensorViewsValidAfterGrowViaRebind) {
    require_cuda();

    constexpr std::size_t kInitial = 128;
    constexpr std::size_t kGrown = 256;
    constexpr int kShDegree = 0; // no shN rest; keeps the fixture minimal

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    // Build a tiny SplatData backed by exportable storage.
    const size_t n = 64;
    Tensor means = allocator(TensorShape({n, 3}), kInitial, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({n, 3}), kInitial, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({n, 4}), kInitial, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({n, 1}), kInitial, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({n, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
    Tensor shN; // degree 0: empty rest

    {
        std::vector<float> host(n * 3);
        for (size_t i = 0; i < n * 3; ++i)
            host[i] = static_cast<float>(i + 1);
        ASSERT_EQ(cudaMemcpy(means.ptr<float>(), host.data(), host.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    EXPECT_EQ(means.capacity(), kInitial);
    EXPECT_EQ(means.external_storage_kind(), "splat.exportable");

    // Grow storage (region offsets for non-means change).
    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);

    // Rebuild tensor views via rebind API (c).
    SplatData model(/*max_sh_degree=*/kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    /*scene_scale=*/1.0f,
                    SplatData::ShNLayout::Swizzled);

    auto rebound = storage.rebindSplatData(model);
    if (!rebound) {
        FAIL() << rebound.error();
    }

    EXPECT_EQ(model.means_raw().capacity(), kGrown);
    EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
    EXPECT_EQ(model.scaling_raw().capacity(), kGrown);

    // Means data preserved.
    {
        std::vector<float> host(n * 3);
        ASSERT_EQ(cudaMemcpy(host.data(), model.means_raw().ptr<float>(), host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < n * 3; ++i) {
            EXPECT_FLOAT_EQ(host[i], static_cast<float>(i + 1));
        }
    }

    // Allocator after grow hands out views at new capacity.
    Tensor means2 = storage.make_allocator()(
        TensorShape({n, 3}), /*requested max_cap-like*/ kGrown * 10, DataType::Float32, "SplatData.means");
    EXPECT_EQ(means2.capacity(), kGrown) << "allocator must clamp to committed capacity";
}

TEST(SplatExportableStorageTest, GrowthCapacityHelper) {
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100), 150u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100, 120), 120u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(100, 50), 50u);
    EXPECT_EQ(SplatExportableStorage::growthCapacity(0, 5'000'000), 1u);
}

// ---------------------------------------------------------------------------
// VRAM audit — multi-grow must plateau (no VMM physical chunk leak)
// ---------------------------------------------------------------------------

namespace {

    struct CudaMemSnapshot {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
    };

    CudaMemSnapshot cuda_mem_snapshot() {
        CudaMemSnapshot s;
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaMemGetInfo(&s.free_bytes, &s.total_bytes), cudaSuccess);
        return s;
    }

    // Allow a small absolute slack for driver fragmentation / other threads.
    constexpr std::size_t kVramSlackBytes = 16ull << 20; // 16 MiB

} // namespace

TEST(SplatExportableStorageTest, ManyGrowCyclesCudaMemGetInfoPlateaus) {
    require_cuda();

    constexpr std::size_t kInitial = 256;
    constexpr std::size_t kReserve = 200'000; // enough virtual headroom for steps
    constexpr int kShDegree = 3;
    // Capacities that force physical growth at several granularities.
    const std::vector<std::size_t> steps = {512, 1024, 2048, 4096, 8192, 16384, 32768};

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.valid());
    const void* const stable_ptr = storage.block->device_ptr;

    std::size_t free_at_plateau = 0;
    std::size_t block_at_plateau = 0;
    for (std::size_t cap : steps) {
        auto grew = storage.grow(cap);
        if (!grew) {
            FAIL() << grew.error();
        }
        EXPECT_EQ(storage.block->device_ptr, stable_ptr);
        EXPECT_EQ(storage.capacity(), cap);
        const auto snap = cuda_mem_snapshot();
        free_at_plateau = snap.free_bytes;
        block_at_plateau = storage.block->size;
    }

    // Further grows to the same capacity are no-ops; free VRAM must not keep dropping.
    for (int i = 0; i < 12; ++i) {
        auto grew = storage.grow(steps.back());
        ASSERT_TRUE(grew.has_value()) << grew.error();
        EXPECT_FALSE(*grew) << "idempotent grow must not re-commit physical";
        EXPECT_EQ(storage.block->size, block_at_plateau);
        const auto snap = cuda_mem_snapshot();
        // free may jitter slightly; it must not systematically fall.
        EXPECT_GE(snap.free_bytes + kVramSlackBytes, free_at_plateau)
            << "cudaMemGetInfo free dropped after plateau grow #" << i
            << " free_plateau=" << free_at_plateau << " free_now=" << snap.free_bytes
            << " (possible VMM physical chunk leak)";
        free_at_plateau = std::max(free_at_plateau, snap.free_bytes);
    }

    // Destroy storage: free VRAM must return near pre-allocation baseline of a probe.
    // Measure residual after destroy vs a fresh identical allocation's delta.
    const auto before_destroy = cuda_mem_snapshot();
    const std::size_t committed = storage.block->size;
    storage = SplatExportableStorage{}; // dtor teardown
    const auto after_destroy = cuda_mem_snapshot();
    EXPECT_GE(after_destroy.free_bytes + kVramSlackBytes, before_destroy.free_bytes + committed)
        << "destroy did not return committed exportable bytes (before_free="
        << before_destroy.free_bytes << " after_free=" << after_destroy.free_bytes
        << " committed=" << committed << ")";
}

TEST(SplatExportableStorageTest, RepeatedCreateGrowDestroyDoesNotLeakVmm) {
    require_cuda();

    constexpr std::size_t kInitial = 512;
    constexpr std::size_t kGrown = 8192;
    constexpr std::size_t kReserve = 100'000;
    constexpr int kShDegree = 3;
    constexpr int kCycles = 24;

    // Warm CUDA allocator / context so first-touch noise is out of the way.
    {
        auto warm = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
        ASSERT_TRUE(warm.has_value()) << warm.error();
        ASSERT_TRUE(warm->grow(kGrown).has_value());
    }
    const auto baseline = cuda_mem_snapshot();

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
        if (!storage_result) {
            FAIL() << "cycle " << cycle << ": " << storage_result.error();
        }
        auto storage = std::move(*storage_result);
        for (std::size_t cap : {std::size_t{1024}, std::size_t{2048}, std::size_t{4096}, kGrown}) {
            auto grew = storage.grow(cap);
            if (!grew) {
                FAIL() << "cycle " << cycle << " grow " << cap << ": " << grew.error();
            }
        }
        // Explicit destroy each cycle.
        storage = SplatExportableStorage{};
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    const auto after = cuda_mem_snapshot();
    // Across many create/grow/destroy cycles free VRAM must return to baseline.
    EXPECT_GE(after.free_bytes + kVramSlackBytes, baseline.free_bytes)
        << "VMM physical leaked across " << kCycles << " create/grow/destroy cycles"
        << " baseline_free=" << baseline.free_bytes << " after_free=" << after.free_bytes
        << " delta_MiB="
        << (static_cast<long long>(baseline.free_bytes) - static_cast<long long>(after.free_bytes)) /
               (1024 * 1024);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, 0u);
}

// Simulates the importer-lifetime protocol without a full Vulkan context:
// an "importer" holds a shared_ptr to the ExportableBlock (like
// VulkanExternalTensorStorage::extra_owner_). grow() must leave the block
// object + stable VA intact; consumers that still hold the block after grow
// see the new size/handle. The real Vulkan fix is to drop VkDeviceMemory
// BEFORE grow (see TrainerManager::installExportableCapacityEnsure).
TEST(SplatExportableStorageTest, GrowKeepsStableVaWhileImportersHoldBlock) {
    require_cuda();

    auto storage_result = SplatExportableStorage::create(256, /*sh=*/0, 0, 4096);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);

    // Simulated Vulkan import lifetime anchor (same type as extra_owner_).
    std::shared_ptr<void> importer_hold = storage.block;
    const void* const va = storage.block->device_ptr;
    const auto old_handle = storage.block->handle.native;
    const std::size_t old_size = storage.block->size;

    auto grew = storage.grow(1024);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);
    EXPECT_EQ(storage.block->device_ptr, va);
    EXPECT_GE(storage.block->size, old_size);
    // Export handle is re-exported on physical growth (Vulkan must re-import).
#ifdef _WIN32
    EXPECT_NE(storage.block->handle.native, old_handle);
#else
    // POSIX fd may be recycled to the same integer after close+export; size
    // and generation are the reliable signals.
    EXPECT_GT(storage.generation(), 1u);
#endif
    EXPECT_EQ(importer_hold.get(), storage.block.get());
    EXPECT_EQ(static_cast<ExportableBlock*>(importer_hold.get())->device_ptr, va);

    // Drop importer, then storage — no crash / CUDA death.
    importer_hold.reset();
    storage = SplatExportableStorage{};
    void* probe = nullptr;
    ASSERT_EQ(cudaMalloc(&probe, 4096), cudaSuccess);
    ASSERT_EQ(cudaFree(probe), cudaSuccess);
}

// Post-grow rebind must not copy from stale pre-grow views. The sequence mirrors
// TrainerManager::growExportableForDensify:
//   rebind(make_allocator) -> grow -> rebind(make_allocator)
// grow() relocates every region; a copying rebind overwrites correct new-offset
// data with garbage from old offsets (means@0 survives; scaling/rot/opacity die).
TEST(SplatExportableStorageTest, RebindGrowRebindPreservesAllRegionPatterns) {
    require_cuda();

    constexpr std::size_t kInitial = 128;
    constexpr std::size_t kGrown = 256;
    constexpr std::size_t kLiveN = 64;
    constexpr std::size_t kReserve = 1024;
    constexpr int kShDegree = 3;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitial, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitial, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
    const auto rest = static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(kShDegree));
    // Exportable ShN is pad-dropped q16 (u16 cells) + float2 bounds / 256.
    const size_t shN_cells = sh_value_quant::sh_value_u16_count(kLiveN, rest);
    const size_t shN_cap = sh_value_quant::sh_value_u16_count(kInitial, rest);
    const size_t bounds_n = sh_value_quant::n_bounds_for_prims(kLiveN) * 2u;
    const size_t bounds_cap = sh_value_quant::n_bounds_for_prims(kInitial) * 2u;
    Tensor shN = allocator(TensorShape({shN_cells}), shN_cap, DataType::Float16, "SplatData.shN");
    Tensor shN_bounds =
        allocator(TensorShape({bounds_n}), bounds_cap, DataType::Float32, "SplatData.shN_value_bounds");
    ASSERT_EQ(shN.dtype(), DataType::Float16);

    // Stamp raw u16 bit patterns via Float16 storage (values 0..63 as half).
    auto f16_val = [](float base, size_t i) {
        return base + static_cast<float>(i % 64u);
    };
    auto stamp = [&](Tensor& t, float base) {
        const size_t n = t.numel();
        if (t.dtype() == DataType::Float16) {
            Tensor host_f = Tensor::empty({n}, Device::CPU, DataType::Float32);
            float* hp = host_f.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                hp[i] = f16_val(base, i);
            }
            Tensor half = host_f.to(DataType::Float16).cuda();
            ASSERT_EQ(cudaMemcpy(t.data_ptr(), half.data_ptr(), n * sizeof(std::uint16_t),
                                 cudaMemcpyDeviceToDevice),
                      cudaSuccess);
            return;
        }
        std::vector<float> host(n);
        for (size_t i = 0; i < n; ++i) {
            host[i] = base + static_cast<float>(i);
        }
        ASSERT_EQ(cudaMemcpy(t.ptr<float>(), host.data(), n * sizeof(float), cudaMemcpyHostToDevice),
                  cudaSuccess);
    };
    auto expect_pattern = [&](const Tensor& t, float base, const char* label) {
        const size_t n = t.numel();
        if (t.dtype() == DataType::Float16) {
            Tensor fp32 = t.to(DataType::Float32).cpu();
            const float* host = fp32.ptr<float>();
            for (size_t i = 0; i < n; ++i) {
                EXPECT_FLOAT_EQ(host[i], f16_val(base, i))
                    << label << " index " << i << " (post-grow rebind integrity, q16 bits)";
            }
            return;
        }
        std::vector<float> host(n);
        ASSERT_EQ(cudaMemcpy(host.data(), t.ptr<float>(), n * sizeof(float), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_FLOAT_EQ(host[i], base + static_cast<float>(i))
                << label << " index " << i << " (post-grow rebind integrity)";
        }
    };

    stamp(means, 10.0f);
    stamp(scaling, 100.0f);
    stamp(rotation, 200.0f);
    stamp(opacity, 300.0f);
    stamp(sh0, 400.0f);
    stamp(shN, 1.0f);
    stamp(shN_bounds, 0.5f);

    SplatData model(kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    /*scene_scale=*/1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.shN_value_bounds() = std::move(shN_bounds);
    model.set_tensor_allocator(storage.make_allocator());

    // pre-grow rebind (GUI drops Vulkan import → CUDA-only views).
    // Self-alias into the same block at current offsets — must preserve patterns.
    {
        auto ok = storage.rebindSplatData(model, storage.make_allocator());
        ASSERT_TRUE(ok.has_value()) << ok.error();
    }
    expect_pattern(model.means_raw(), 10.0f, "means after pre-grow rebind");
    expect_pattern(model.scaling_raw(), 100.0f, "scaling after pre-grow rebind");
    expect_pattern(model.rotation_raw(), 200.0f, "rotation after pre-grow rebind");
    expect_pattern(model.opacity_raw(), 300.0f, "opacity after pre-grow rebind");
    expect_pattern(model.sh0_raw(), 400.0f, "sh0 after pre-grow rebind");
    expect_pattern(model.shN_raw(), 1.0f, "shN after pre-grow rebind");
    expect_pattern(model.shN_value_bounds(), 0.5f, "shN_bounds after pre-grow rebind");

    const auto gen_before = storage.generation();
    const std::size_t scaling_off_before = storage.region_offsets[SplatExportableStorage::Scaling];

    // grow relocates non-means regions to new offsets.
    auto grew = storage.grow(kGrown);
    ASSERT_TRUE(grew.has_value()) << grew.error();
    ASSERT_TRUE(*grew);
    EXPECT_GT(storage.generation(), gen_before);
    EXPECT_NE(storage.region_offsets[SplatExportableStorage::Scaling], scaling_off_before)
        << "grow must relocate scaling (capacity-dependent pack)";

    // Patterns intact at NEW offsets inside the block (grow did its job).
    expect_device_pattern(
        static_cast<const char*>(storage.block->device_ptr) +
            storage.region_offsets[SplatExportableStorage::Means],
        kLiveN * 3, 10.0f);
    expect_device_pattern(
        static_cast<const char*>(storage.block->device_ptr) +
            storage.region_offsets[SplatExportableStorage::Scaling],
        kLiveN * 3, 100.0f);
    expect_device_pattern(
        static_cast<const char*>(storage.block->device_ptr) +
            storage.region_offsets[SplatExportableStorage::Rotation],
        kLiveN * 4, 200.0f);
    expect_device_pattern(
        static_cast<const char*>(storage.block->device_ptr) +
            storage.region_offsets[SplatExportableStorage::Opacity],
        kLiveN * 1, 300.0f);

    // Install views at the new offsets without copying stale pre-grow views.
    {
        auto ok = storage.rebindSplatData(model, storage.make_allocator());
        ASSERT_TRUE(ok.has_value()) << ok.error();
    }

    EXPECT_EQ(model.means_raw().capacity(), kGrown);
    EXPECT_EQ(model.scaling_raw().capacity(), kGrown);
    EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");

    // All attributes must survive at new offsets through the model tensors.
    expect_pattern(model.means_raw(), 10.0f, "means after post-grow rebind");
    expect_pattern(model.scaling_raw(), 100.0f, "scaling after post-grow rebind");
    expect_pattern(model.rotation_raw(), 200.0f, "rotation after post-grow rebind");
    expect_pattern(model.opacity_raw(), 300.0f, "opacity after post-grow rebind");
    expect_pattern(model.sh0_raw(), 400.0f, "sh0 after post-grow rebind");
    expect_pattern(model.shN_raw(), 1.0f, "shN after post-grow rebind");
    expect_pattern(model.shN_value_bounds(), 0.5f, "shN_bounds after post-grow rebind");
    EXPECT_TRUE(model.shN_value_quantized());
}

// Grown slack rows must be non-renderable (opacity → sigmoid(−∞)≈0,
// identity quaternion) so an accidental stale-row read remains dark.
TEST(SplatExportableStorageTest, GrowSlackRowsAreNonRenderable) {
    require_cuda();

    constexpr std::size_t kInitial = 64;
    constexpr std::size_t kGrown = 128;
    constexpr int kShDegree = 0;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    ASSERT_TRUE(storage.grow(kGrown).value_or(false));

    // Slack rows [kInitial, kGrown): opacity raw must be −inf (or ≤ −20),
    // rotation identity (1,0,0,0).
    std::vector<float> opacity(kGrown - kInitial);
    std::vector<float> rotation((kGrown - kInitial) * 4);
    void* op_ptr = static_cast<char*>(storage.block->device_ptr) +
                   storage.region_offsets[SplatExportableStorage::Opacity] +
                   kInitial * sizeof(float);
    void* rot_ptr = static_cast<char*>(storage.block->device_ptr) +
                    storage.region_offsets[SplatExportableStorage::Rotation] +
                    kInitial * 4 * sizeof(float);
    ASSERT_EQ(cudaMemcpy(opacity.data(), op_ptr, opacity.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(rotation.data(), rot_ptr, rotation.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (size_t i = 0; i < opacity.size(); ++i) {
        EXPECT_LE(opacity[i], -20.0f) << "slack opacity[" << i << "] must be non-renderable";
    }
    for (size_t i = 0; i < kGrown - kInitial; ++i) {
        EXPECT_FLOAT_EQ(rotation[i * 4 + 0], 1.0f) << "slack quat w row " << i;
        EXPECT_FLOAT_EQ(rotation[i * 4 + 1], 0.0f) << "slack quat x row " << i;
        EXPECT_FLOAT_EQ(rotation[i * 4 + 2], 0.0f) << "slack quat y row " << i;
        EXPECT_FLOAT_EQ(rotation[i * 4 + 3], 0.0f) << "slack quat z row " << i;
    }
}

// densify past the initial live-N commit must grow the exportable
// block via capacity_ensure (storage layer, no GUI). Mirrors
// TrainerManager::growExportableForDensify: work lives outside the std::function
// so rebind can replace SplatData mid-grow without destroying the active frame.
TEST(SplatExportableStorageTest, CapacityEnsureGrowsPastInitialCommit) {
    require_cuda();

    constexpr std::size_t kInitialCap = 128;
    constexpr std::size_t kLiveN = 64;
    constexpr std::size_t kNeed = 200; // past initial commit
    constexpr std::size_t kReserve = 4096;
    constexpr int kShDegree = 0;

    auto storage_result = SplatExportableStorage::create(kInitialCap, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));
    auto allocator = storage->make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitialCap, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitialCap, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitialCap, DataType::Float32, "SplatData.sh0");
    Tensor shN;

    SplatData model(/*max_sh_degree=*/kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    /*scene_scale=*/1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.set_tensor_allocator(storage->make_allocator());

    const void* const stable_va = storage->block->device_ptr;
    ASSERT_EQ(model.means_raw().capacity(), kInitialCap);
    ASSERT_LT(model.means_raw().capacity(), kNeed);

    struct GrowHook {
        std::shared_ptr<SplatExportableStorage> storage;
        SplatData* model = nullptr;
        void install() {
            model->set_capacity_ensure([this](std::size_t needed_rows) { return grow(needed_rows); });
        }
        bool grow(std::size_t needed_rows) {
            if (storage->capacity() >= needed_rows &&
                model->means_raw().capacity() >= needed_rows) {
                return true;
            }
            const std::size_t want =
                SplatExportableStorage::growthCapacity(needed_rows, storage->reservedCapacity());
            auto grew = storage->grow(want);
            if (!grew || storage->capacity() < needed_rows) {
                return false;
            }
            // rebind assigns into *model and drops the trampoline; reinstall after.
            if (auto ok = storage->rebindSplatData(*model, storage->make_allocator()); !ok) {
                return false;
            }
            install();
            return model->means_raw().capacity() >= needed_rows;
        }
    } hook{storage, &model};
    hook.install();

    ASSERT_TRUE(model.has_capacity_ensure());
    ASSERT_TRUE(model.ensure_param_capacity(kNeed))
        << "capacity_ensure must grow exportable block past initial commit";
    EXPECT_GE(storage->capacity(), kNeed);
    EXPECT_GE(model.means_raw().capacity(), kNeed);
    EXPECT_EQ(storage->block->device_ptr, stable_va);
    EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
    EXPECT_TRUE(model.has_capacity_ensure()) << "hook must be reinstalled after grow+rebind";
}

// Migration must retain the capacity-ensure callback even when committed
// exportable headroom is below max_cap.
TEST(SplatExportableStorageTest, MigratePreservesCapacityEnsureUnderMaxCap) {
    require_cuda();

    constexpr std::size_t kInitialCap = 128;
    constexpr std::size_t kLiveN = 64;
    constexpr std::size_t kMaxCap = 5'000'000; // GUI default
    constexpr std::size_t kNeed = 200;
    constexpr int kShDegree = 0;

    auto storage_result = SplatExportableStorage::create(kInitialCap, kShDegree, 0, kMaxCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));
    auto allocator = storage->make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitialCap, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitialCap, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitialCap, DataType::Float32, "SplatData.sh0");
    Tensor shN;

    SplatData model(kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.set_tensor_allocator(storage->make_allocator());

    int ensure_calls = 0;
    struct GrowHook {
        std::shared_ptr<SplatExportableStorage> storage;
        SplatData* model = nullptr;
        int* ensure_calls = nullptr;
        void install() {
            model->set_capacity_ensure([this](std::size_t needed_rows) { return grow(needed_rows); });
        }
        bool grow(std::size_t needed_rows) {
            ++(*ensure_calls);
            if (storage->capacity() >= needed_rows &&
                model->means_raw().capacity() >= needed_rows) {
                return true;
            }
            const std::size_t want =
                SplatExportableStorage::growthCapacity(needed_rows, storage->reservedCapacity());
            auto grew = storage->grow(want);
            if (!grew || storage->capacity() < needed_rows) {
                return false;
            }
            if (auto ok = storage->rebindSplatData(*model, storage->make_allocator()); !ok) {
                return false;
            }
            install();
            return model->means_raw().capacity() >= needed_rows;
        }
    } hook{storage, &model, &ensure_calls};
    hook.install();

    lfs::core::param::TrainingParameters params;
    params.optimization.sh_degree = kShDegree;
    params.optimization.max_cap = static_cast<int>(kMaxCap);

    // Repeated migrates must be no-ops (ready under live-N commit) and must not
    // drop the densify grow hook — this is the GUI strategy-step path.
    for (int i = 0; i < 3; ++i) {
        auto result = lfs::training::migrateTrainingModelToAllocator(
            params, model, storage->make_allocator());
        ASSERT_TRUE(result.has_value()) << result.error() << " iter=" << i;
        ASSERT_TRUE(model.has_capacity_ensure())
            << "migrate wiped capacity_ensure at iter=" << i;
        EXPECT_EQ(model.means_raw().capacity(), kInitialCap);
        EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
    }

    // Force remigrate still preserves the hook.
    {
        auto result = lfs::training::migrateTrainingModelToAllocator(
            params, model, storage->make_allocator(), /*force_reallocation=*/true);
        ASSERT_TRUE(result.has_value()) << result.error();
        ASSERT_TRUE(model.has_capacity_ensure())
            << "forced migration wiped capacity_ensure";
    }

    ASSERT_TRUE(model.ensure_param_capacity(kNeed))
        << "after migrate, densify capacity_ensure must still grow the block";
    EXPECT_GE(ensure_calls, 1);
    EXPECT_GE(model.means_raw().capacity(), kNeed);
    EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
}

// SH1/SH3 float-swizzled shN at live-N == capacity exceeds the pad-dropped
// q16 region (12 vs 9 / 48 vs 45 cells per primitive); migrate used to abort
// with "shape for 'SplatData.shN' needs ... bytes". It must fall back to the
// float workspace and land q16-encoded.
TEST(SplatExportableStorageTest, MigrateFloatSwizzledShNFallsBackToQ16AtFullCapacity) {
    require_cuda();

    constexpr std::size_t kCap = 1000;

    for (int sh_degree : {1, 2, 3}) {
        auto storage_result = SplatExportableStorage::create(kCap, sh_degree, 0, kCap);
        if (!storage_result) {
            FAIL() << storage_result.error();
        }
        auto storage = std::move(*storage_result);

        const auto rest = sh_rest_coefficients_for_degree(sh_degree);
        Tensor means = Tensor::zeros({kCap, 3}, Device::CUDA);
        Tensor sh0 = Tensor::zeros({kCap, 1, 3}, Device::CUDA);
        Tensor scaling = Tensor::zeros({kCap, 3}, Device::CUDA);
        Tensor rotation = Tensor::zeros({kCap, 4}, Device::CUDA);
        Tensor opacity = Tensor::zeros({kCap, 1}, Device::CUDA);
        Tensor shN = Tensor::zeros_direct(
            TensorShape({sh_swizzled_float_count(kCap, rest)}),
            sh_swizzled_float_count(kCap, rest),
            Device::CUDA);

        SplatData model(sh_degree,
                        std::move(means),
                        std::move(sh0),
                        std::move(shN),
                        std::move(scaling),
                        std::move(rotation),
                        std::move(opacity),
                        1.0f,
                        SplatData::ShNLayout::Swizzled);

        lfs::core::param::TrainingParameters params;
        params.optimization.sh_degree = sh_degree;
        params.optimization.max_cap = static_cast<int>(kCap);

        auto result = lfs::training::migrateTrainingModelToAllocator(
            params, model, storage.make_allocator());
        ASSERT_TRUE(result.has_value()) << result.error() << " sh_degree=" << sh_degree;
        EXPECT_TRUE(model.shN_value_quantized());
        EXPECT_EQ(model.means_raw().external_storage_kind(), "splat.exportable");
        EXPECT_EQ(static_cast<std::size_t>(model.shN_raw().capacity()),
                  sh_value_quant::sh_value_u16_count(kCap, rest));
    }
}

// The reported failure: dataset init with init_points > 0.75 x max_cap and
// SH degree 1 threw from the exportable allocator inside
// init_model_from_pointcloud. The float shN must come back as an
// out-of-block workspace instead.
TEST(SplatExportableStorageTest, InitModelFromPointcloudSucceedsAtFullExportableCapacitySh1) {
    require_cuda();

    constexpr std::size_t kCap = 1000;

    auto storage_result = SplatExportableStorage::create(kCap, /*sh_degree=*/1, 0, kCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);

    Tensor means = Tensor::rand({kCap, 3}, Device::CPU);
    Tensor colors = Tensor::zeros({kCap, 3}, Device::CPU, DataType::UInt8);
    PointCloud pcd(std::move(means), std::move(colors));

    lfs::core::param::TrainingParameters params;
    params.optimization.sh_degree = 1;
    params.optimization.max_cap = static_cast<int>(kCap);
    params.optimization.random = false;

    auto model = init_model_from_pointcloud(params,
                                            Tensor::zeros({3}, Device::CPU),
                                            pcd,
                                            static_cast<int>(kCap),
                                            storage.make_allocator());
    ASSERT_TRUE(model.has_value()) << model.error();
    EXPECT_EQ(model->size(), kCap);
    EXPECT_EQ(model->shN_raw().dtype(), DataType::Float32);
    EXPECT_EQ(static_cast<std::size_t>(model->shN_raw().numel()),
              sh_swizzled_float_count(kCap, sh_rest_coefficients_for_degree(1)));
    EXPECT_EQ(model->means_raw().external_storage_kind(), "splat.exportable");
}

// A failed capacity ensure must abort before mutation and leave all parameter
// row counts unchanged.
TEST(SplatExportableStorageTest, ForcedGrowFailureLeavesModelUntouched) {
    require_cuda();

    constexpr std::size_t kInitialCap = 64;
    constexpr std::size_t kLiveN = 32;
    constexpr std::size_t kNeed = 200; // past initial commit → ensure required
    constexpr int kShDegree = 0;

    auto storage_result = SplatExportableStorage::create(kInitialCap, kShDegree, 0, kInitialCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    // reservedCapacity == kInitialCap: growth past initial commit cannot expand
    // the virtual reservation, so ensure fails (pathological forced failure).
    auto storage = std::make_shared<SplatExportableStorage>(std::move(*storage_result));
    auto allocator = storage->make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitialCap, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitialCap, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitialCap, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitialCap, DataType::Float32, "SplatData.sh0");
    Tensor shN;

    SplatData model(kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.set_tensor_allocator(storage->make_allocator());

    int ensure_calls = 0;
    model.set_capacity_ensure([&](std::size_t needed_rows) {
        ++ensure_calls;
        // Force failure: never grow (simulates VRAM OOM / reservation ceiling).
        return model.means_raw().capacity() >= needed_rows;
    });

    lfs::training::AdamOptimizer opt(model, lfs::training::AdamConfig{});
    const std::size_t size_before = static_cast<std::size_t>(model.size());
    const std::size_t means_cap_before = model.means_raw().capacity();
    const std::size_t scaling_cap_before = model.scaling_raw().capacity();
    const std::size_t means_rows_before = model.means_raw().shape()[0];
    const std::size_t scaling_rows_before = model.scaling_raw().shape()[0];

    const size_t n_new = kNeed - kLiveN;
    ASSERT_FALSE(opt.preflight_grow_capacity(n_new))
        << "forced failure must reject preflight";
    EXPECT_GE(ensure_calls, 1);

    // No param mutation: row counts and capacities unchanged.
    EXPECT_EQ(static_cast<std::size_t>(model.size()), size_before);
    EXPECT_EQ(model.means_raw().shape()[0], means_rows_before);
    EXPECT_EQ(model.scaling_raw().shape()[0], scaling_rows_before);
    EXPECT_EQ(model.means_raw().capacity(), means_cap_before);
    EXPECT_EQ(model.scaling_raw().capacity(), scaling_cap_before);

    // add_new_params must also throw without mutating when ensure fails.
    auto new_means = Tensor::zeros({n_new, 3}, Device::CUDA);
    EXPECT_THROW(
        opt.add_new_params(lfs::training::ParamType::Means, new_means, true),
        std::runtime_error);
    EXPECT_EQ(model.means_raw().shape()[0], means_rows_before);
    EXPECT_EQ(model.scaling_raw().shape()[0], scaling_rows_before);
    EXPECT_EQ(static_cast<std::size_t>(model.size()), size_before);
}

// storage-layer contract: growExportableDeviceBlock must only run after
// external importers (Vulkan VkDeviceMemory) are detached. Protocol mirror of
// TrainerManager::growExportableForDensify / shared-scratch commit path —
// drop the simulated import hold BEFORE grow, then grow keeps VA stable.
TEST(ExportableStorageTest, GrowAfterImporterDetachKeepsStableVa) {
    require_cuda();

    auto block_result = allocateExportableDeviceBlock(1 << 20, 0, false, 8 << 20);
    if (!block_result) {
        FAIL() << block_result.error();
    }
    auto block = std::move(*block_result);
    ASSERT_NE(block, nullptr);
    const void* const va = block->device_ptr;
    const std::size_t old_size = block->size;

    // Simulated Vulkan import lifetime (extra_owner_ / imported_buffer).
    std::shared_ptr<void> importer_hold = block;

    // correct order: detach import BEFORE release_physical inside grow.
    importer_hold.reset();
    auto grew = growExportableDeviceBlock(block, old_size * 2);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);
    EXPECT_EQ(block->device_ptr, va) << "VA must stay stable across grow";
    EXPECT_GE(block->size, old_size * 2);

    block.reset();
    void* probe = nullptr;
    ASSERT_EQ(cudaMalloc(&probe, 4096), cudaSuccess);
    ASSERT_EQ(cudaFree(probe), cudaSuccess);
}

// shape that overruns the packed region must fail loud (not silent OOB view).
TEST(SplatExportableStorageTest, AllocatorRejectsShapeOverrunRegion) {
    require_cuda();

    constexpr std::size_t kCap = 64;
    constexpr int kShDegree = 1;
    auto storage_result = SplatExportableStorage::create(kCap, kShDegree, 0, kCap);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    // Request means with more logical rows than the region can hold.
    const std::size_t too_many = kCap * 4;
    EXPECT_THROW(
        {
            (void)allocator(TensorShape({too_many, 3}), too_many, DataType::Float32,
                            "SplatData.means");
        },
        std::runtime_error);

    // q16 codes: request more u16 cells than ShN region bytes.
    const auto rest = static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(kShDegree));
    const size_t max_cells = sh_value_quant::sh_value_u16_count(kCap, rest);
    EXPECT_THROW(
        {
            (void)allocator(TensorShape({max_cells * 8}), max_cells * 8, DataType::Float16,
                            "SplatData.shN");
        },
        std::runtime_error);
}

// partially-constructed storage (no control) must refuse make_allocator
// no by-value snapshot flavor that hands out offsets which go stale on grow.
TEST(SplatExportableStorageTest, MakeAllocatorRequiresControlBlock) {
    require_cuda();
    SplatExportableStorage empty{};
    EXPECT_FALSE(empty.valid());
    EXPECT_THROW((void)empty.make_allocator(), std::runtime_error);
}

// Generation-checked resolve: holding a Tensor across grow sees live pointer via
// resolve_exportable_device_ptr (baked storage_ptr is stale after grow).
TEST(SplatExportableStorageTest, ResolveUsesLivePointerAfterGrow) {
    require_cuda();

    constexpr std::size_t kInitial = 128;
    constexpr std::size_t kGrown = 512;
    constexpr std::size_t kLiveN = 32;
    constexpr int kShDegree = 1;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    const auto rest = static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(kShDegree));
    const size_t cells = sh_value_quant::sh_value_u16_count(kLiveN, rest);
    const size_t cap_cells = sh_value_quant::sh_value_u16_count(kInitial, rest);
    const size_t bounds_n = sh_value_quant::n_bounds_for_prims(kLiveN) * 2u;
    const size_t bounds_cap = sh_value_quant::n_bounds_for_prims(kInitial) * 2u;

    Tensor shN = allocator(TensorShape({cells}), cap_cells, DataType::Float16, "SplatData.shN");
    Tensor bounds =
        allocator(TensorShape({bounds_n}), bounds_cap, DataType::Float32, "SplatData.shN_value_bounds");
    ASSERT_TRUE(shN.has_exportable_provenance());
    ASSERT_TRUE(bounds.has_exportable_provenance());

    void* const shN_before = shN.storage_ptr();
    void* const bounds_before = bounds.storage_ptr();
    void* const shN_resolved_before = resolve_exportable_device_ptr(shN);
    EXPECT_EQ(shN_resolved_before, shN_before);

    const auto gen_before = storage.generation();
    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);
    EXPECT_GT(storage.generation(), gen_before);

    // Baked storage_ptr is stale (pre-grow region base). Live resolve must differ
    // for regions that relocate (ShN is not at offset 0).
    void* const shN_live = resolve_exportable_device_ptr(shN);
    void* const bounds_live = resolve_exportable_device_ptr(bounds);
    EXPECT_EQ(shN_live, storage.live_region_ptr(SplatExportableStorage::ShN));
    EXPECT_EQ(bounds_live, storage.live_region_ptr(SplatExportableStorage::ShNBounds));
    // Means is at offset 0 and stays put; ShN relocates when capacity grows.
    EXPECT_NE(shN_live, shN_before)
        << "ShN region must relocate on grow; resolve must not return baked ptr";
    EXPECT_NE(bounds_live, bounds_before);

    // Fresh allocator views match live control.
    auto alloc2 = storage.make_allocator();
    const size_t cells2 = sh_value_quant::sh_value_u16_count(kLiveN, rest);
    const size_t cap2 = sh_value_quant::sh_value_u16_count(kGrown, rest);
    Tensor shN2 = alloc2(TensorShape({cells2}), cap2, DataType::Float16, "SplatData.shN");
    EXPECT_EQ(shN2.storage_ptr(), shN_live);
    EXPECT_EQ(shN2.exportable_bound_generation(), storage.generation());
}

// Stale held view + resolve pair for q16 codes/bounds must agree on generation
// when rebound, and survive a grow under a held Tensor without illegal address.
TEST(SplatExportableStorageTest, Q16BindPtrsSurviveGrowUnderHeldView) {
    require_cuda();

    constexpr std::size_t kInitial = 256;
    constexpr std::size_t kGrown = 1024;
    constexpr std::size_t kLiveN = 64;
    constexpr int kShDegree = 2;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kGrown * 2);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitial, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitial, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
    const auto rest = static_cast<std::uint32_t>(sh_rest_coefficients_for_degree(kShDegree));
    const size_t cells = sh_value_quant::sh_value_u16_count(kLiveN, rest);
    const size_t cap_cells = sh_value_quant::sh_value_u16_count(kInitial, rest);
    const size_t bounds_n = sh_value_quant::n_bounds_for_prims(kLiveN) * 2u;
    const size_t bounds_cap = sh_value_quant::n_bounds_for_prims(kInitial) * 2u;
    Tensor shN = allocator(TensorShape({cells}), cap_cells, DataType::Float16, "SplatData.shN");
    Tensor shN_bounds =
        allocator(TensorShape({bounds_n}), bounds_cap, DataType::Float32, "SplatData.shN_value_bounds");

    SplatData model(kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    1.0f,
                    SplatData::ShNLayout::Swizzled);
    model.shN_value_bounds() = std::move(shN_bounds);
    model.set_tensor_allocator(allocator);
    model.set_active_sh_degree(kShDegree);
    ASSERT_TRUE(model.shN_value_quantized());

    // Hold raw pointers to model a consumer that does not rebind after growth.
    Tensor held_codes = model.shN_raw();
    Tensor held_bounds = model.shN_value_bounds();
    void* const baked_codes = held_codes.storage_ptr();

    auto grew = storage.grow(kGrown);
    if (!grew) {
        FAIL() << grew.error();
    }
    ASSERT_TRUE(*grew);

    // Without rebind, resolve still returns LIVE region bases (no illegal address).
    void* const live_codes = resolve_exportable_device_ptr(held_codes);
    void* const live_bounds = resolve_exportable_device_ptr(held_bounds);
    EXPECT_EQ(live_codes, storage.live_region_ptr(SplatExportableStorage::ShN));
    EXPECT_EQ(live_bounds, storage.live_region_ptr(SplatExportableStorage::ShNBounds));
    EXPECT_NE(live_codes, baked_codes);

    // After rebind, model views are generation-fresh and resolve_q16 agrees.
    ASSERT_TRUE(storage.rebindSplatData(model, storage.make_allocator()).has_value());
    model.set_tensor_allocator(storage.make_allocator());
    ASSERT_TRUE(model.shN_value_quantized());
    const auto q16 = resolve_q16_bind_ptrs(model);
    ASSERT_NE(q16.codes, nullptr);
    ASSERT_NE(q16.bounds, nullptr);
    EXPECT_EQ(static_cast<const void*>(q16.codes),
              storage.live_region_ptr(SplatExportableStorage::ShN));
    EXPECT_TRUE(q16.generation_checked);
    EXPECT_EQ(q16.generation, storage.generation());
}
