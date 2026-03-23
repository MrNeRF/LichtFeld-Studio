/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "rendering/rasterizer/rasterization/include/rasterization_api_tensor.h"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "selection/selection_service.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {

    template <typename Fn>
    double benchmark_us(Fn&& fn, const int warmup_iters = 10, const int bench_iters = 50) {
        for (int i = 0; i < warmup_iters; ++i) {
            fn();
            if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                ADD_FAILURE() << "cudaDeviceSynchronize failed during warmup: " << cudaGetErrorString(err);
                return 0.0;
            }
        }

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
            ADD_FAILURE() << "Failed to create CUDA benchmark events";
            if (start)
                cudaEventDestroy(start);
            if (stop)
                cudaEventDestroy(stop);
            return 0.0;
        }

        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; ++i) {
            fn();
        }
        cudaEventRecord(stop);
        if (const auto err = cudaEventSynchronize(stop); err != cudaSuccess) {
            ADD_FAILURE() << "cudaEventSynchronize failed during benchmark: " << cudaGetErrorString(err);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 0.0;
        }

        float elapsed_ms = 0.0f;
        if (const auto err = cudaEventElapsedTime(&elapsed_ms, start, stop); err != cudaSuccess) {
            ADD_FAILURE() << "cudaEventElapsedTime failed during benchmark: " << cudaGetErrorString(err);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 0.0;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<double>(elapsed_ms * 1000.0f) / static_cast<double>(bench_iters);
    }

    template <typename Fn>
    double benchmark_wall_us(Fn&& fn, const int warmup_iters = 3, const int bench_iters = 10) {
        for (int i = 0; i < warmup_iters; ++i) {
            fn();
            if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                ADD_FAILURE() << "cudaDeviceSynchronize failed during warmup: " << cudaGetErrorString(err);
                return 0.0;
            }
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < bench_iters; ++i) {
            fn();
            if (const auto err = cudaDeviceSynchronize(); err != cudaSuccess) {
                ADD_FAILURE() << "cudaDeviceSynchronize failed during benchmark: " << cudaGetErrorString(err);
                return 0.0;
            }
        }
        const auto stop = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(stop - start);
        return elapsed.count() / static_cast<double>(bench_iters);
    }

    void print_result(const std::string_view label, const size_t point_count, const double time_us) {
        const double points_per_second = static_cast<double>(point_count) / (time_us / 1e6);
        std::cout << std::left << std::setw(28) << label
                  << " | " << std::right << std::setw(8) << std::fixed << std::setprecision(2)
                  << time_us / 1000.0 << " ms"
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2)
                  << (points_per_second / 1e6) << " Mpts/s\n";
    }

    std::unique_ptr<lfs::core::SplatData> make_benchmark_splat(const size_t count) {
        auto means = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({count, size_t{1}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto shN = Tensor::zeros({count, size_t{3}, size_t{3}}, Device::CUDA, DataType::Float32);
        auto scaling = Tensor::zeros({count, size_t{3}}, Device::CUDA, DataType::Float32);

        std::vector<float> rotation_data(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            rotation_data[i * 4] = 1.0f;
        }
        auto rotation = Tensor::from_vector(rotation_data, {count, size_t{4}}, Device::CUDA).to(DataType::Float32);
        auto opacity = Tensor::zeros({count, size_t{1}}, Device::CUDA, DataType::Float32);

        return std::make_unique<lfs::core::SplatData>(
            1,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f);
    }

} // namespace

class SelectionModesBenchmark : public ::testing::Test {
protected:
    static constexpr size_t POINT_COUNT = 1'000'000;

    void SetUp() override {
        ASSERT_TRUE(Tensor::zeros({1}, Device::CUDA).is_valid());

        screen_positions_ = Tensor::uniform({POINT_COUNT, size_t{2}}, 0.0f, 2048.0f, Device::CUDA, DataType::Float32);
        brush_selection_ = Tensor::zeros({POINT_COUNT}, Device::CUDA, DataType::UInt8);
        bool_selection_ = Tensor::zeros({POINT_COUNT}, Device::CUDA, DataType::Bool);
        polygon_ = Tensor::from_vector(
                       {
                           256.0f,
                           256.0f,
                           1792.0f,
                           384.0f,
                           1600.0f,
                           1600.0f,
                           384.0f,
                           1728.0f,
                       },
                       {4, 2},
                       Device::CUDA)
                       .to(DataType::Float32);
    }

    Tensor screen_positions_;
    Tensor brush_selection_;
    Tensor bool_selection_;
    Tensor polygon_;
};

TEST_F(SelectionModesBenchmark, BrushSelect1M) {
    std::cout << "\n=== Selection Kernel Benchmarks (1M points) ===\n";

    const double time_us = benchmark_us([&]() {
        brush_selection_.zero_();
        lfs::rendering::brush_select_tensor(screen_positions_, 1024.0f, 1024.0f, 96.0f, brush_selection_);
    });

    print_result("Brush select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}

TEST_F(SelectionModesBenchmark, RectSelect1M) {
    const double time_us = benchmark_us([&]() {
        bool_selection_.zero_();
        lfs::rendering::rect_select_tensor(screen_positions_, 512.0f, 512.0f, 1536.0f, 1536.0f, bool_selection_);
    });

    print_result("Rect select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}

TEST_F(SelectionModesBenchmark, PolygonSelect1M) {
    const double time_us = benchmark_us([&]() {
        bool_selection_.zero_();
        lfs::rendering::polygon_select_tensor(screen_positions_, polygon_, bool_selection_);
    });

    print_result("Polygon select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}

class SelectionServiceBenchmark : public ::testing::Test {
protected:
    static constexpr size_t POINT_COUNT = 1'000'000;

    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        scene_manager_ = std::make_unique<lfs::vis::SceneManager>();
        rendering_manager_ = std::make_unique<lfs::vis::RenderingManager>();
        lfs::vis::services().set(scene_manager_.get());
        lfs::vis::services().set(rendering_manager_.get());

        scene_manager_->getScene().addNode("benchmark", make_benchmark_splat(POINT_COUNT));

        service_ = std::make_unique<lfs::vis::SelectionService>(scene_manager_.get(), rendering_manager_.get());
        screen_positions_ = std::make_shared<Tensor>(
            Tensor::uniform({POINT_COUNT, size_t{2}}, 0.0f, 2048.0f, Device::CUDA, DataType::Float32));
        service_->setTestingScreenPositions(screen_positions_);

        polygon_ = Tensor::from_vector(
                       {
                           256.0f,
                           256.0f,
                           1792.0f,
                           384.0f,
                           1600.0f,
                           1600.0f,
                           384.0f,
                           1728.0f,
                       },
                       {4, 2},
                       Device::CUDA)
                       .to(DataType::Float32);
    }

    void TearDown() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        service_.reset();
        screen_positions_.reset();
        rendering_manager_.reset();
        scene_manager_.reset();
        lfs::vis::op::undoHistory().clear();
    }

    std::unique_ptr<lfs::vis::SceneManager> scene_manager_;
    std::unique_ptr<lfs::vis::RenderingManager> rendering_manager_;
    std::unique_ptr<lfs::vis::SelectionService> service_;
    std::shared_ptr<Tensor> screen_positions_;
    Tensor polygon_;
};

TEST_F(SelectionServiceBenchmark, BrushSelect1MEndToEnd) {
    std::cout << "\n=== Selection Service Benchmarks (1M points end-to-end) ===\n";

    const double time_us = benchmark_wall_us([&]() {
        const auto result = service_->selectBrush(1024.0f, 1024.0f, 96.0f, lfs::vis::SelectionMode::Replace);
        ASSERT_TRUE(result.success);
        lfs::vis::op::undoHistory().clear();
    });

    print_result("Service brush select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}

TEST_F(SelectionServiceBenchmark, RectSelect1MEndToEnd) {
    const double time_us = benchmark_wall_us([&]() {
        const auto result = service_->selectRect(512.0f, 512.0f, 1536.0f, 1536.0f, lfs::vis::SelectionMode::Replace);
        ASSERT_TRUE(result.success);
        lfs::vis::op::undoHistory().clear();
    });

    print_result("Service rect select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}

TEST_F(SelectionServiceBenchmark, PolygonSelect1MEndToEnd) {
    const double time_us = benchmark_wall_us([&]() {
        const auto result = service_->selectPolygon(polygon_, lfs::vis::SelectionMode::Replace);
        ASSERT_TRUE(result.success);
        lfs::vis::op::undoHistory().clear();
    });

    print_result("Service polygon select", POINT_COUNT, time_us);
    EXPECT_GT(time_us, 0.0);
}
