/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda_error.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/lazy_config.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "io/formats/colmap.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

    class DeferredD3PinTest : public ::testing::Test {
    protected:
        void SetUp() override {
            using namespace lfs::core;

            reset_cuda_diagnostics_for_testing();
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);

            constexpr size_t count = 64;
            means_ = Tensor::randn({count, 3}, Device::CUDA, DataType::Float32);
            sh0_ = Tensor::randn({count, 1, 3}, Device::CUDA, DataType::Float32).mul(0.3f);
            shN_ = Tensor::zeros({count, 0, 3}, Device::CUDA, DataType::Float32);
            scaling_ = Tensor::randn({count, 3}, Device::CUDA, DataType::Float32).mul(0.2f).sub(3.0f);
            rotation_ = Tensor::randn({count, 4}, Device::CUDA, DataType::Float32);
            rotation_ = rotation_.div(rotation_.pow(2.0f).sum(-1, true).sqrt());
            opacity_ = Tensor::randn({count}, Device::CUDA, DataType::Float32);
            splat_ = std::make_unique<SplatData>(
                0, means_, sh0_, shN_, scaling_, rotation_, opacity_, 1.0f);

            auto rotation = Tensor::eye(3, Device::CUDA);
            std::vector<float> translation_values{0.0f, 0.0f, 4.0f};
            auto translation = Tensor::from_blob(
                                   translation_values.data(), {3}, Device::CPU, DataType::Float32)
                                   .cuda();
            camera_ = std::make_unique<Camera>(
                rotation, translation,
                100.0f, 100.0f, 32.0f, 32.0f,
                Tensor{}, Tensor{}, lfs::core::CameraModelType::PINHOLE,
                "deferred-d3", "", std::filesystem::path{}, 64, 64, 0);
            background_ = Tensor::zeros({3}, Device::CUDA, DataType::Float32);
        }

        void TearDown() override {
            lfs::core::reset_cuda_diagnostics_for_testing();
            lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
            lfs::core::internal::clear_lazy_ir_for_testing();
            lfs::core::internal::lazy_executor_clear_registry_for_testing();
            lfs::core::internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            lfs::core::internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }

        lfs::core::Tensor means_;
        lfs::core::Tensor sh0_;
        lfs::core::Tensor shN_;
        lfs::core::Tensor scaling_;
        lfs::core::Tensor rotation_;
        lfs::core::Tensor opacity_;
        lfs::core::Tensor background_;
        std::unique_ptr<lfs::core::SplatData> splat_;
        std::unique_ptr<lfs::core::Camera> camera_;
    };

    TEST_F(DeferredD3PinTest, StridedViewOfDeferredIsReal) {
        auto deferred = lfs::core::Tensor::from_vector(
                            std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
                            {4}, lfs::core::Device::CPU)
                            .add(10.0f);
        ASSERT_TRUE(deferred.is_deferred());

        auto transposed = deferred.t();
        ASSERT_TRUE(transposed.is_deferred() || transposed.data_ptr() != nullptr);

        EXPECT_EQ(transposed.to_vector(),
                  (std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f}));
        EXPECT_NE(transposed.data_ptr(), nullptr);
    }

    TEST_F(DeferredD3PinTest, PinOperandsMaterializesAll) {
        using namespace lfs::core;

        auto a = Tensor::from_vector(
                     std::vector<float>{1.0f, 2.0f, 3.0f}, {3}, Device::CPU)
                     .add(5.0f);
        auto b = Tensor::from_vector(
                     std::vector<float>{2.0f, 4.0f}, {2}, Device::CPU)
                     .mul(3.0f);
        ASSERT_TRUE(a.is_deferred());
        ASSERT_TRUE(b.is_deferred());

        pin_operands({&a, &b});

        ASSERT_FALSE(a.is_deferred());
        ASSERT_FALSE(b.is_deferred());
        EXPECT_EQ(a.to_vector(), (std::vector<float>{6.0f, 7.0f, 8.0f}));
        EXPECT_EQ(b.to_vector(), (std::vector<float>{6.0f, 12.0f}));
        const void* const a_ptr = a.data_ptr();
        const void* const b_ptr = b.data_ptr();

        auto unrelated = Tensor::ones({8}, Device::CPU, DataType::Float32).add(9.0f);
        ASSERT_TRUE(unrelated.is_deferred());
        EXPECT_EQ(unrelated.to_vector(), (std::vector<float>(8, 10.0f)));
        EXPECT_EQ(a.data_ptr(), a_ptr);
        EXPECT_EQ(b.data_ptr(), b_ptr);
    }

    TEST_F(DeferredD3PinTest, MultiOperandLaunchWithDeferredInputs) {
        using namespace lfs::core;

        for (const Device device : {Device::CPU, Device::CUDA}) {
            SCOPED_TRACE(device == Device::CPU ? "CPU" : "CUDA");
            auto base = Tensor::from_vector(
                std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
            if (device == Device::CUDA) {
                base = base.cuda();
            }

            auto input = base.add(10.0f);
            auto mask = base.gt(2.0f);
            ASSERT_TRUE(input.is_deferred());
            ASSERT_TRUE(mask.is_deferred());

            const auto selected = input.masked_select(mask);
            EXPECT_EQ(selected.cpu().to_vector(),
                      (std::vector<float>{13.0f, 14.0f}));
        }
    }

    TEST_F(DeferredD3PinTest, RegistryPruneStillBoundsMap) {
        using namespace lfs::core;

        internal::clear_lazy_ir_for_testing();
        internal::lazy_executor_clear_registry_for_testing();
        internal::lazy_executor_reset_diagnostics_for_testing();

        constexpr int iterations = 1024;
        for (int i = 0; i < iterations; ++i) {
            auto deferred = Tensor::ones({32}, Device::CPU, DataType::Float32)
                                .add(1.0f)
                                .mul(2.0f);
            ASSERT_TRUE(deferred.has_lazy_expr());
            if ((i % 64) == 0) {
                (void)internal::lazy_executor_registered_node_count_for_testing();
            }
        }

        EXPECT_EQ(internal::lazy_executor_registered_node_count_for_testing(), 0u);
        const auto diagnostics = internal::lazy_executor_diagnostics_snapshot_for_testing();
        EXPECT_GT(diagnostics.max_registry_entries, 0u);
        EXPECT_LE(diagnostics.max_registry_entries, 16u);
    }

    TEST_F(DeferredD3PinTest, CorridorColorsChainRealData) {
        using namespace lfs::core;
        namespace fs = std::filesystem;

        const fs::path sparse_dir = fs::path(PROJECT_ROOT_PATH) / "data/bicycle/sparse/0";
        const bool has_binary = fs::exists(sparse_dir / "points3D.bin");
        const bool has_text = fs::exists(sparse_dir / "points3D.txt");
        if (!has_binary && !has_text) {
            GTEST_SKIP() << "data/bicycle sparse points3D not available";
        }

        auto loaded = has_binary
                          ? lfs::io::read_colmap_point_cloud(sparse_dir)
                          : lfs::io::read_colmap_point_cloud_text(sparse_dir);
        ASSERT_TRUE(loaded.has_value()) << loaded.error().format();
        auto point_cloud = std::move(loaded->value);
        ASSERT_TRUE(point_cloud.means.is_valid());
        ASSERT_TRUE(point_cloud.colors.is_valid());

        auto positions = point_cloud.means;
        auto colors_u8 = point_cloud.colors.cpu();
        ASSERT_EQ(colors_u8.dtype(), DataType::UInt8);
        const auto color_bytes = colors_u8.to_vector_uint8();
        std::vector<float> expected_colors(color_bytes.size());
        std::transform(color_bytes.begin(), color_bytes.end(), expected_colors.begin(),
                       [](const uint8_t value) {
                           return static_cast<float>(value) / 255.0f;
                       });

        auto colors = colors_u8.to(DataType::Float32).div(255.0f).cuda();
        auto [sorted_positions, sorted_indices] = positions.flatten().sort();
        (void)sorted_indices;
        ASSERT_GT(sorted_positions.numel(), 0u);
        const size_t median_index = sorted_positions.numel() / 2;
        const float median = sorted_positions.slice(0, median_index, median_index + 1).item();

        const auto positions_cpu = positions.cpu();
        const auto colors_cpu = colors.cpu();

        auto expected_positions = positions_cpu.to_vector();
        std::sort(expected_positions.begin(), expected_positions.end());
        EXPECT_FLOAT_EQ(median, expected_positions[median_index]);

        const auto actual_colors = colors_cpu.to_vector();
        ASSERT_EQ(actual_colors.size(), expected_colors.size());
        for (size_t i = 0; i < actual_colors.size(); ++i) {
            if (std::abs(actual_colors[i] - expected_colors[i]) > 1e-6f) {
                ADD_FAILURE() << "color mismatch at index " << i
                              << ": actual=" << actual_colors[i]
                              << ", expected=" << expected_colors[i];
                break;
            }
        }
    }

    TEST_F(DeferredD3PinTest, FreePathErrorDoesNotLatch) {
        using namespace lfs::core;

        ensure_cuda_success(
            cudaErrorInitializationError,
            "cudaFree(test-injected)",
            "injected teardown failure",
            LFS_SOURCE_SITE_CURRENT(),
            CudaFailureDisposition::LogOnlyNoLatch);
        EXPECT_FALSE(cuda_is_unavailable());

        {
            auto allocation = Tensor::zeros({16}, Device::CUDA, DataType::Float32);
            EXPECT_TRUE(allocation.is_valid());
            EXPECT_NE(allocation.data_ptr(), nullptr);
        }

        reset_cuda_diagnostics_for_testing();
        ensure_cuda_success(
            cudaErrorInitializationError,
            "cudaMalloc(test-injected)",
            "injected allocation failure",
            LFS_SOURCE_SITE_CURRENT(),
            CudaFailureDisposition::LogOnly);
        EXPECT_TRUE(cuda_is_unavailable());
    }

    TEST_F(DeferredD3PinTest, ThreadCacheReleaseLeavesNothingToFree) {
        bool fast_released = false;
        bool gsplat_released = false;
        bool intersect_released = false;
        bool nan_check_released = false;
        std::exception_ptr worker_error;

        std::thread worker([&] {
            try {
                {
                    auto output = lfs::training::gsplat_rasterize(
                        *camera_, *splat_, background_);
                    if (!output.image.is_valid()) {
                        throw std::runtime_error("gsplat forward produced no image");
                    }
                }
                {
                    auto output = lfs::training::fast_rasterize(
                        *camera_, *splat_, background_);
                    if (!output.image.is_valid()) {
                        throw std::runtime_error("FastGS forward produced no image");
                    }
                }

                (void)lfs::core::tensor_ops::has_nan_gpu(
                    means_.ptr<float>(), means_.numel(), means_.stream());
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    throw std::runtime_error("worker render synchronization failed");
                }

                fast_released = lfs::training::release_fast_rasterizer_thread_local_caches();
                gsplat_released = lfs::training::release_gsplat_rasterizer_thread_local_caches();
                intersect_released = gsplat_lfs::release_intersect_thread_local_cache();
                nan_check_released = lfs::core::tensor_ops::release_nan_check_thread_buffers();
            } catch (...) {
                worker_error = std::current_exception();
            }
        });
        worker.join();

        if (worker_error) {
            std::rethrow_exception(worker_error);
        }
        EXPECT_TRUE(fast_released);
        EXPECT_TRUE(gsplat_released);
        EXPECT_TRUE(intersect_released);
        EXPECT_TRUE(nan_check_released);
    }

} // namespace
