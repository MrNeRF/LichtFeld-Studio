/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * RAM and VRAM leak regression guard under system-memory pressure.
 * Runs N fixed-size "training-like" cycles and asserts host RSS and CUDA free
 * memory deltas are ~0 between cycle 10 (steady) and cycle N.
 */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;

namespace {

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    // Resident set size in bytes from /proc/self/status (Linux).
    std::size_t host_rss_bytes() {
        std::ifstream in("/proc/self/status");
        std::string key;
        while (in >> key) {
            if (key == "VmRSS:") {
                std::size_t kib = 0;
                in >> kib;
                return kib * 1024;
            }
            // Skip rest of line.
            std::string rest;
            std::getline(in, rest);
        }
        return 0;
    }

    std::size_t cuda_free_bytes() {
        std::size_t free_b = 0, total_b = 0;
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        EXPECT_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
        return free_b;
    }

    Camera make_camera(int w, int h) {
        std::vector<float> R_data = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> T_data = {0, 0, 4};
        auto R = Tensor::from_blob(R_data.data(), {3, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        auto T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        return Camera(R, T, /*fx=*/100.f, /*fy=*/100.f, /*cx=*/w * 0.5f, /*cy=*/h * 0.5f,
                      Tensor(), Tensor(), CameraModelType::PINHOLE, "leak_guard", "",
                      std::filesystem::path{}, w, h, 0);
    }

    std::unique_ptr<SplatData> make_splat(int n) {
        auto means = Tensor::zeros({static_cast<size_t>(n), 3}, Device::CUDA);
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
            rot[static_cast<size_t>(i) * 4] = 1.f;
        }
        auto rotation = Tensor::from_blob(rot.data(), {static_cast<size_t>(n), 4}, Device::CPU, DataType::Float32)
                            .to(Device::CUDA);
        auto opacity = Tensor::full({static_cast<size_t>(n)}, 2.0f, Device::CUDA);
        return std::make_unique<SplatData>(0, means, sh0, shN, scaling, rotation, opacity, 1.0f);
    }

} // namespace

// Training-like cycle: exportable grow steps + FastGS forwards + TLS release.
// Steady-state RSS and VRAM between cycle 10 and N must not drift.
TEST(VramLeakRegressionTest, FixedSizeCyclesHostRssAndVramStable) {
    require_cuda();

    constexpr int kWarmCycles = 10;
    constexpr int kTotalCycles = 40;
    constexpr int kForwardsPerCycle = 2;
    constexpr int kSplatN = 96;
    constexpr int kWidth = 64;
    constexpr int kHeight = 64;
    // Host RSS may grow a little from allocator freelists / gtest; 64 MiB is
    // generous for a fixed-size loop that should be steady after warm-up.
    constexpr std::size_t kRssSlack = 64ull << 20;
    constexpr std::size_t kVramSlack = 32ull << 20;

    auto camera = make_camera(kWidth, kHeight);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    // Warm CUDA + host paths outside the measured window.
    for (int w = 0; w < 3; ++w) {
        auto splat = make_splat(kSplatN);
        auto r = fast_rasterize_forward(camera, *splat, bg, 0, 0, 0, 0, false);
        ASSERT_TRUE(r.has_value()) << lfs::format_for_developer(r.error());
        r->second.release_forward_context();
        {
            auto storage = SplatExportableStorage::create(256, /*sh=*/0, 0, 8192);
            ASSERT_TRUE(storage.has_value()) << storage.error();
            ASSERT_TRUE(storage->grow(512).has_value());
            ASSERT_TRUE(storage->grow(1024).has_value());
        }
        release_fast_rasterizer_thread_local_caches();
        release_fastgs_sort_workspace_buffers();
        GlobalArenaManager::instance().get_arena().full_reset();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }

    std::size_t rss_at_steady = 0;
    std::size_t free_at_steady = 0;

    for (int cycle = 1; cycle <= kTotalCycles; ++cycle) {
        auto splat = make_splat(kSplatN);
        for (int f = 0; f < kForwardsPerCycle; ++f) {
            auto r = fast_rasterize_forward(camera, *splat, bg, 0, 0, 0, 0, false);
            ASSERT_TRUE(r.has_value()) << "cycle " << cycle << ": "
                                       << lfs::format_for_developer(r.error());
            r->second.release_forward_context();
        }

        // Mimic densify growth + teardown of exportable block.
        {
            auto storage = SplatExportableStorage::create(256, /*sh=*/0, 0, 8192);
            ASSERT_TRUE(storage.has_value()) << storage.error();
            ASSERT_TRUE(storage->grow(512).has_value());
            ASSERT_TRUE(storage->grow(1024).has_value());
            ASSERT_TRUE(storage->grow(2048).has_value());
        }

        // End-of-step cleanup (training thread shutdown pattern).
        release_fast_rasterizer_thread_local_caches();
        release_fastgs_sort_workspace_buffers();
        GlobalArenaManager::instance().get_arena().full_reset();
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

        if (cycle == kWarmCycles) {
            rss_at_steady = host_rss_bytes();
            free_at_steady = cuda_free_bytes();
            ASSERT_GT(rss_at_steady, 0u);
            ASSERT_GT(free_at_steady, 0u);
        }
    }

    const std::size_t rss_final = host_rss_bytes();
    const std::size_t free_final = cuda_free_bytes();

    EXPECT_LE(rss_final, rss_at_steady + kRssSlack)
        << "host RSS grew between cycle " << kWarmCycles << " and " << kTotalCycles
        << " rss_steady=" << rss_at_steady << " rss_final=" << rss_final
        << " delta_MiB="
        << (static_cast<long long>(rss_final) - static_cast<long long>(rss_at_steady)) /
               (1024 * 1024);

    EXPECT_GE(free_final + kVramSlack, free_at_steady)
        << "CUDA free dropped between cycle " << kWarmCycles << " and " << kTotalCycles
        << " free_steady=" << free_at_steady << " free_final=" << free_final
        << " delta_MiB="
        << (static_cast<long long>(free_at_steady) - static_cast<long long>(free_final)) /
               (1024 * 1024);

    const auto snap = lfs::diagnostics::VramProfiler::instance().snapshot();
    EXPECT_EQ(snap.process.exportable_splat_bytes, 0u);
}
