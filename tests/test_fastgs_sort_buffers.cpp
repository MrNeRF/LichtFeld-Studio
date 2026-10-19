/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * FastGS sort storage is exact per-frame arena storage.
 */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/formats/colmap.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/fastgs/rasterization/include/forward.h"
#include "training/rasterization/fastgs/rasterization/include/rasterization_api.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

TEST(FastGSSortKeyTest, PreservesFullPositiveFloatDepthPrecision) {
    using InstanceKey = std::uint64_t;
    ASSERT_EQ(sizeof(InstanceKey), sizeof(std::uint64_t));

    const float near_depth = 1.0f;
    const float far_depth = std::nextafter(near_depth, 2.0f);
    const std::uint32_t near_bits = std::bit_cast<std::uint32_t>(near_depth);
    const std::uint32_t far_bits = std::bit_cast<std::uint32_t>(far_depth);
    ASSERT_LT(near_bits, far_bits);

    const InstanceKey near_key = (InstanceKey{7} << 32) | near_bits;
    const InstanceKey far_key = (InstanceKey{7} << 32) | far_bits;
    const InstanceKey next_tile_key = (InstanceKey{8} << 32) | near_bits;
    EXPECT_LT(near_key, far_key);
    EXPECT_LT(far_key, next_tile_key);
}

namespace {

    Camera make_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, /*fx=*/100.f, /*fy=*/100.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "test", "",
                      std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
        // Place a few gaussians in front of the camera so n_instances > 0.
        if (n > 0) {
            auto cpu = means.to(Device::CPU);
            float* p = cpu.ptr<float>();
            for (int i = 0; i < n; ++i) {
                p[i * 3 + 0] = (i % 5) * 0.3f - 0.6f;
                p[i * 3 + 1] = (i / 5) * 0.3f - 0.6f;
                p[i * 3 + 2] = 0.0f;
            }
            means = cpu.to(Device::CUDA);
        }
        auto sh0 = Tensor::full({static_cast<size_t>(n), 1, 3}, 0.5f, Device::CUDA);
        auto shN = Tensor::zeros({static_cast<size_t>(n), 0, 3}, Device::CUDA);
        auto scaling = Tensor::full({static_cast<size_t>(n), 3}, -2.0f, Device::CUDA);
        std::vector<float> rot(static_cast<size_t>(n) * 4, 0.f);
        for (int i = 0; i < n; ++i) {
            rot[static_cast<size_t>(i) * 4] = 1.f; // identity quat
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

    void cleanup_arena() {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

} // namespace

class FastGSSortBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        bg_ = Tensor::zeros({3}, Device::CUDA);
        camera_ = std::make_unique<Camera>(make_camera(64, 64));
        splat_ = make_splat(32);
    }

    void TearDown() override {
        splat_.reset();
        camera_.reset();
        release_fastgs_sort_workspace_buffers();
        cleanup_arena();
    }

    Tensor bg_;
    std::unique_ptr<Camera> camera_;
    std::unique_ptr<SplatData> splat_;
};

TEST_F(FastGSSortBufferTest, SortWorkspaceIsExactAndArenaOwned) {
    using fast_lfs::rasterization::sort_workspace_allocated_bytes;
    using fast_lfs::rasterization::sort_workspace_required_bytes;

    auto warm = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(warm.has_value()) << lfs::format_for_developer(warm.error());
    ASSERT_GT(warm->second.forward_ctx.n_instances, 0)
        << "fixture must produce visible instances so the sort path runs";
    const auto sort_bytes = warm->second.forward_ctx.per_instance_sort_total_size;
    ASSERT_GT(sort_bytes, 0u);
    EXPECT_EQ(sort_workspace_required_bytes(), 0u);
    EXPECT_EQ(sort_workspace_allocated_bytes(), 0u);

    const auto frame_buffers = GlobalArenaManager::instance().get_arena().get_frame_buffers(
        warm->second.forward_ctx.frame_id);
    EXPECT_TRUE(std::any_of(frame_buffers.begin(), frame_buffers.end(),
                            [sort_bytes](const auto& buffer) {
                                return buffer.size >= sort_bytes;
                            }))
        << "FastGS sort storage must be recorded in the arena frame";
    warm->second.release_forward_context();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST_F(FastGSSortBufferTest, ExactPathMatchesRepeatedPathPixels) {
    Tensor image_sync;
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        image_sync = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    Tensor image_async;
    {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        image_async = r->first.image.to(Device::CPU).contiguous();
        r->second.release_forward_context();
    }

    ASSERT_EQ(image_sync.numel(), image_async.numel());
    const float* a = image_sync.ptr<float>();
    const float* b = image_async.ptr<float>();
    for (size_t i = 0; i < image_sync.numel(); ++i) {
        ASSERT_EQ(a[i], b[i]) << "pixel mismatch at i=" << i
                              << " sync=" << a[i] << " async=" << b[i];
    }
}

TEST(FastGSSortKeyTest, DuplicatedRealSplatsKeepOrderAcrossArenaReallocation) {
    const auto sparse_path = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "sparse" / "0";
    if (!std::filesystem::exists(sparse_path)) {
        GTEST_SKIP() << "Requires the public bicycle COLMAP dataset at " << sparse_path;
    }

    struct ArenaCleanup {
        ~ArenaCleanup() { cleanup_arena(); }
    } cleanup;
    auto points = lfs::io::read_colmap_point3D_binary_records(sparse_path / "points3D.bin");
    ASSERT_TRUE(points.has_value());
    auto cameras = lfs::io::read_colmap_cameras_only(sparse_path, 4.0f);
    ASSERT_TRUE(cameras.has_value());
    auto& views = std::get<0>(*cameras);
    ASSERT_FALSE(views.empty());
    auto& camera = *views.front();
    const auto transform = camera.world_view_transform().to(Device::CPU).contiguous();
    ASSERT_EQ(transform.ndim(), 3u);
    ASSERT_EQ(transform.shape()[0], 1u);
    ASSERT_EQ(transform.shape()[1], 4u);
    ASSERT_EQ(transform.shape()[2], 4u);
    const auto* view = transform.ptr<float>();
    const auto [fx, fy, cx, cy] = camera.get_intrinsics();
    const lfs::io::Point3DData* source = nullptr;
    float source_depth = 0.0f;
    float nearest_center = std::numeric_limits<float>::infinity();
    for (const auto& point : points->value) {
        float position[3];
        for (int row = 0; row < 3; ++row)
            position[row] = view[4 * row] * point.xyz[0] + view[4 * row + 1] * point.xyz[1] +
                            view[4 * row + 2] * point.xyz[2] + view[4 * row + 3];
        if (position[2] <= 0.2f)
            continue;
        const float dx = fx * position[0] / position[2] + cx - camera.image_width() * 0.5f;
        const float dy = fy * position[1] / position[2] + cy - camera.image_height() * 0.5f;
        const float distance = dx * dx + dy * dy;
        if (distance < nearest_center) {
            nearest_center = distance;
            source = &point;
            source_depth = position[2];
        }
    }
    ASSERT_NE(source, nullptr);
    ASSERT_LT(nearest_center, 16.0f * 16.0f);
    RecordProperty("source_point_id", std::to_string(source->point3D_id));
    const float log_scale = std::log(source_depth * 64.0f / std::min(fx, fy));
    auto bg = Tensor::zeros({3}, Device::CUDA);
    constexpr size_t count = 1025;
    auto means = Tensor::empty({count, 3}, Device::CPU);
    auto colors = Tensor::zeros({count, 1, 3}, Device::CPU);
    auto rotations = Tensor::zeros({count, 4}, Device::CPU);
    for (size_t i = 0; i < count; ++i) {
        for (int axis = 0; axis < 3; ++axis)
            means.ptr<float>()[3 * i + axis] = static_cast<float>(source->xyz[axis]);
        colors.ptr<float>()[3 * i] = i % 2 == 0 ? 1.0f : -1.0f;
        colors.ptr<float>()[3 * i + 2] = i % 2 == 0 ? -1.0f : 1.0f;
        rotations.ptr<float>()[4 * i] = 1.0f;
    }
    SplatData duplicated(0,
                         means.to(Device::CUDA),
                         colors.to(Device::CUDA),
                         Tensor::zeros({count, size_t{0}, size_t{3}}, Device::CUDA),
                         Tensor::full({count, 3}, log_scale, Device::CUDA),
                         rotations.to(Device::CUDA),
                         Tensor::full({count}, 2.0f, Device::CUDA),
                         1.0f);

    Tensor reference;
    for (int repeat = 0; repeat < 8; ++repeat) {
        SCOPED_TRACE(repeat);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        if (repeat % 2 == 0)
            cleanup_arena();
        auto result = fast_rasterize_forward(camera, duplicated, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(result.has_value()) << lfs::format_for_developer(result.error());
        const auto& ctx = result->second.forward_ctx;
        ASSERT_EQ(ctx.n_visible, count);
        ASSERT_GT(ctx.n_instances, count * 128);
        ASSERT_EQ(ctx.n_instances % count, 0u);
        RecordProperty("tiles_per_splat", std::to_string(ctx.n_instances / count));
        std::vector<uint32_t> indices(ctx.n_instances);
        ASSERT_EQ(cudaStreamSynchronize(ctx.stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(indices.data(), ctx.sorted_primitive_indices,
                             indices.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        size_t wrong_indices = 0;
        for (size_t i = 0; i < indices.size(); ++i)
            wrong_indices += indices[i] != i % count;
        EXPECT_EQ(wrong_indices, 0u) << "Equal-depth instances must retain primitive-index order in every tile";
        auto image = result->first.image.to(Device::CPU).contiguous();
        ASSERT_EQ(image.ndim(), 3u);
        ASSERT_EQ(image.shape()[0], 3u);
        if (repeat == 0) {
            reference = image.clone();
        } else {
            ASSERT_EQ(image.shape(), reference.shape());
            EXPECT_EQ(std::memcmp(image.ptr<float>(), reference.ptr<float>(),
                                  image.numel() * sizeof(float)),
                      0);
        }
    }
}

TEST_F(FastGSSortBufferTest, RepeatedFramesUseTheCurrentArenaFrame) {
    auto first = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(first.has_value());
    const auto first_frame = first->second.forward_ctx.frame_id;
    const auto first_sort = first->second.forward_ctx.per_instance_sort_total_size;
    first->second.release_forward_context();

    auto second = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(second->second.forward_ctx.frame_id, first_frame);
    EXPECT_EQ(second->second.forward_ctx.per_instance_sort_total_size, first_sort);
    second->second.release_forward_context();
}

// cudaPointerGetAttributes preflight is debug-only.
// Release (NDEBUG) builds must not call it per-step; after N frames the
// instrumented counter stays 0.
TEST_F(FastGSSortBufferTest, ReleasePreflightPointerAttrsAreSkipped) {
    using fast_lfs::rasterization::preflight_pointer_attr_call_count;
    using fast_lfs::rasterization::reset_preflight_pointer_attr_call_count;

    reset_preflight_pointer_attr_call_count();
    constexpr int kFrames = 8;
    for (int i = 0; i < kFrames; ++i) {
        auto r = fast_rasterize_forward(*camera_, *splat_, bg_, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        r->second.release_forward_context();
    }
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto calls = preflight_pointer_attr_call_count();
#ifdef NDEBUG
    EXPECT_EQ(calls, 0u)
        << "Release/NDEBUG builds must not call cudaPointerGetAttributes per "
           "forward (observed "
        << calls << " after " << kFrames << " frames)";
#else
    // Debug builds keep full preflight (~10 attrs × frames).
    EXPECT_GE(calls, static_cast<std::uint64_t>(kFrames) * 8u)
        << "Debug builds should still run pointer attribute preflight";
#endif
}

// ---------------------------------------------------------------------------
// VRAM audit — arena-owned sort storage + raster output buffers release on join.
// Spawn N worker threads that each run a few FastGS forwards, explicitly
// release the remaining renderer caches, then join.
// cudaMemGetInfo free must return near the pre-spawn baseline.
// ---------------------------------------------------------------------------
TEST(FastGSThreadLocalCacheTest, SpawnRenderJoinReturnsVram) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    // Warm primary thread caches so first-touch noise is outside the measurement.
    {
        auto cam = make_camera(64, 64);
        auto splat = make_splat(64);
        auto bg = Tensor::zeros({3}, Device::CUDA);
        auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        r->second.release_forward_context();
        release_fast_rasterizer_thread_local_caches();
        release_fastgs_sort_workspace_buffers();
        cleanup_arena();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    // Device-wide free memory moves with every other process on the GPU; prefer this process's own usage.
    const auto device_bytes_in_use = [] {
        if (const auto own = lfs::diagnostics::process_device_memory_bytes())
            return *own;
        std::size_t free = 0, total = 0;
        EXPECT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
        return total - free;
    };
    const std::size_t used_before = device_bytes_in_use();

    constexpr int kThreads = 4;
    constexpr int kForwardsPerThread = 3;
    std::atomic<int> failures{0};

    auto worker = [&]() {
        if (cudaSetDevice(0) != cudaSuccess) {
            failures.fetch_add(1);
            return;
        }
        try {
            auto cam = make_camera(96, 96);
            auto splat = make_splat(128);
            auto bg = Tensor::zeros({3}, Device::CUDA);
            for (int i = 0; i < kForwardsPerThread; ++i) {
                auto r = fast_rasterize_forward(cam, *splat, bg, 0, 0, 0, 0, false);
                if (!r.has_value()) {
                    failures.fetch_add(1);
                    return;
                }
                r->second.release_forward_context();
            }
            // Explicit TLS release (mirrors training-thread shutdown). Without
            // this, join relies solely on TLS destructors — which must also free.
            release_fast_rasterizer_thread_local_caches();
            release_fastgs_sort_workspace_buffers();
            if (cudaDeviceSynchronize() != cudaSuccess) {
                failures.fetch_add(1);
            }
        } catch (...) {
            failures.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back(worker);
    }
    for (auto& th : threads) {
        th.join();
    }
    ASSERT_EQ(failures.load(), 0) << "one or more worker threads failed";

    cleanup_arena();
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const std::size_t used_after = device_bytes_in_use();

    // Each worker builds arena sort storage plus image TLS. If either leaks,
    // usage grows by multiple MiB * kThreads.
    constexpr std::size_t kSlack = 32ull << 20; // 32 MiB driver/fragmentation slack
    EXPECT_LE(used_after, used_before + kSlack)
        << "FastGS sort/raster storage leaked across spawn-render-join "
        << "used_before=" << used_before << " used_after=" << used_after
        << " delta_MiB="
        << (static_cast<long long>(used_after) - static_cast<long long>(used_before)) /
               (1024 * 1024);
}
