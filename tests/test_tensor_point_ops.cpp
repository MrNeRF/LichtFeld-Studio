/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "cuda_backend_test.hpp"

#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_filters.hpp"
#include "core/tensor_histogram.hpp"
#include "core/tensor_labels.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_spatial.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;
    enum class Storage { CPU,
                         CUDA,
                         Vulkan };

    class TensorPointOps : public testing::TestWithParam<Storage> {
    protected:
        void SetUp() override {
            backend = GetParam() == Storage::Vulkan ? GpuBackend::Vulkan : GpuBackend::CUDA;
            device = GetParam() == Storage::CPU ? Device::CPU : Device::GPU;
            if (device == Device::GPU && !gpu_backend_available(backend))
                GTEST_SKIP() << "Backend unavailable";
            scope = std::make_unique<GpuBackendScope>(backend);
            projection.width = 400;
            projection.height = 200;
            projection.focal_x = 20;
            projection.focal_y = 30;
            projection.center_x = 100;
            projection.center_y = 80;
        }
        void TearDown() override {
            if (GetParam() == Storage::Vulkan) {
                EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
                for (const auto& message : internal::vulkan_validation_messages_for_testing())
                    ADD_FAILURE() << message;
            }
        }
        Tensor bytes(std::initializer_list<int> values) const { return bytes(std::vector<int>(values)); }
        template <class T>
        Tensor bytes(const std::vector<T>& values) const {
            Tensor cpu = Tensor::empty({values.size()}, Device::CPU, DataType::UInt8);
            if (!values.empty())
                std::copy(values.begin(), values.end(), cpu.ptr<uint8_t>());
            return device == Device::CPU ? cpu : cpu.gpu();
        }
        Tensor ints(const std::vector<int>& values) const { return Tensor::from_vector(values, {values.size()}, device); }
        Tensor floats(const std::vector<float>& values, TensorShape shape) const { return Tensor::from_vector(values, shape, device); }
        Tensor points3D(const std::vector<float>& values) const { return floats(values, {values.size() / 3, 3}); }
        Tensor screen_points(const std::vector<float>& values) const { return floats(values, {values.size() / 2, 2}); }
        Tensor tensor(const std::vector<float>& values) const { return floats(values, {values.size()}); }
        Tensor mask(size_t count, float value = 0) const { return Tensor::full({count}, value, device, DataType::UInt8); }
        Tensor output() const { return Tensor::full({257}, -7.f, device, DataType::Int32); }
        Tensor identity() const { return floats({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}); }
        GpuBackend backend;
        Device device;
        PointProjection projection;
        std::unique_ptr<GpuBackendScope> scope;
    };

    template <class Work>
    void withCudaStreams(Work work) {
        cudaStream_t producer = nullptr, consumer = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
        work(producer, consumer);
        CudaMemoryPool::instance().release_stream(producer);
        CudaMemoryPool::instance().release_stream(consumer);
        EXPECT_EQ(cudaStreamDestroy(producer), cudaSuccess);
        EXPECT_EQ(cudaStreamDestroy(consumer), cudaSuccess);
    }

    TEST_P(TensorPointOps, ProjectionStridesOffsetsAndStorageBackendArePreserved) {
        const auto source = Tensor::from_vector(
            {99.f, 99.f, 99.f, 99.f, 99.f,
             99.f, 1.f, 2.f, -4.f, 99.f,
             99.f, -1.f, 0.f, -2.f, 99.f},
            {3, 5}, device);
        const auto positions = source.slice(0, 1, 3).slice(1, 1, 4);
        ASSERT_FALSE(positions.is_contiguous());
        GpuBackendScope other(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const auto output = lfs::core::project_points(positions, projection);
        EXPECT_EQ(output.device(), device);
        EXPECT_EQ(gpu_backend_of(output), gpu_backend_of(positions));
        EXPECT_EQ(output.shape(), (TensorShape{2, 2}));
        EXPECT_EQ(output.cpu().to_vector(), (std::vector<float>{105, 65, 90, 80}));
        EXPECT_EQ(positions.cpu().to_vector(), (std::vector<float>{1, 2, -4, -1, 0, -2}));
    }

    TEST_P(TensorPointOps, ProjectionModelIndicesClampButVisibilityRejectsInvalidIndices) {
        const auto xyz = points3D({0, 0, -2, 0, 0, -2, 0, 0, -2, 0, 0, -2});
        const auto matrices = Tensor::from_vector(
            {1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f,
             1.f, 0.f, 0.f, 2.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f},
            {2, 4, 4}, device);
        const auto indices = Tensor::from_vector(std::vector<int>{99, -7, 0, 1, 9}, {5}, device).slice(0, 1, 5);
        const auto visibility = Tensor::from_vector(std::vector<bool>{false, true, false}, {3}, device).slice(0, 1, 3);
        EXPECT_EQ(lfs::core::project_points(xyz, projection, &matrices, &indices).cpu().to_vector(),
                  (std::vector<float>{100, 80, 100, 80, 120, 80, 120, 80}));
        const float bad = projection.invalid_value;
        EXPECT_EQ(lfs::core::project_points(xyz, projection, &matrices, &indices, &visibility).cpu().to_vector(),
                  (std::vector<float>{bad, bad, 100, 80, bad, bad, bad, bad}));
        EXPECT_EQ(lfs::core::project_points(xyz, projection, &matrices, nullptr, &visibility).cpu().to_vector(),
                  (std::vector<float>{100, 80, 100, 80, 100, 80, 100, 80}));
    }

    TEST_P(TensorPointOps, ProjectionCameraModelsAndNearDistance) {
        const auto xyz = points3D({1, 2, -4, 0, 0, 2, 0, 0, -0.5e-6f});
        projection.invalid_value = -9999;
        const float bad = projection.invalid_value;
        EXPECT_EQ(lfs::core::project_points(xyz, projection).cpu().to_vector(),
                  (std::vector<float>{105, 65, bad, bad, bad, bad}));
        projection.model = PointProjectionModel::Orthographic;
        projection.ortho_scale = 3;
        EXPECT_EQ(lfs::core::project_points(xyz, projection).cpu().to_vector(),
                  (std::vector<float>{103, 74, bad, bad, bad, bad}));
        projection.model = PointProjectionModel::Equirectangular;
        const auto directions = points3D({0, 0, -2, 2, 0, 0, 0, 0, 2, 0, 0, 0});
        const auto projected = lfs::core::project_points(directions, projection).cpu().to_vector();
        const std::vector<float> expected{200, 100, 300, 100, 400, 100, bad, bad};
        ASSERT_EQ(projected.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i)
            EXPECT_NEAR(projected[i], expected[i], 0.0001f);
    }

    TEST_P(TensorPointOps, ProjectionNonfiniteViewsAndInvalidOrthoScaleHaveDefinedFill) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        const auto xyz = points3D({nan, 0, -2, 0, 0, -2});
        projection.invalid_value = -12345;
        EXPECT_EQ(lfs::core::project_points(xyz, projection).cpu().to_vector(),
                  (std::vector<float>{-12345, -12345, 100, 80}));
        projection.model = PointProjectionModel::Orthographic;
        for (float scale : {0.f, -1.f, nan, std::numeric_limits<float>::infinity()}) {
            projection.ortho_scale = scale;
            EXPECT_EQ(lfs::core::project_points(xyz, projection).cpu().to_vector(),
                      (std::vector<float>{-12345, -12345, -12345, -12345}));
        }
    }

    TEST_P(TensorPointOps, ProjectionRejectsMalformedInputsAndKeepsEmptyShape) {
        const auto empty = lfs::core::project_points(Tensor::empty({0, 3}, device), projection);
        EXPECT_EQ(empty.shape(), (TensorShape{0, 2}));
        EXPECT_EQ(empty.device(), device);
        const auto xyz = points3D({0, 0, -2});
        auto bad_matrix = Tensor::zeros({15}, device);
        auto bad_indices = Tensor::zeros({2}, device, DataType::Int32);
        auto index = Tensor::zeros({1}, device, DataType::Int32);
        auto bad_visibility = Tensor::ones({1}, device);
        EXPECT_THROW(lfs::core::project_points(xyz.reshape({3}), projection), std::exception);
        EXPECT_THROW(lfs::core::project_points(xyz.to(DataType::Float16), projection), std::exception);
        EXPECT_THROW(lfs::core::project_points(xyz, projection, &bad_matrix), std::exception);
        EXPECT_THROW(lfs::core::project_points(xyz, projection, nullptr, &bad_indices), std::exception);
        EXPECT_THROW(lfs::core::project_points(xyz, projection, nullptr, &index, &bad_visibility), std::exception);
        projection.width = 0;
        EXPECT_THROW(lfs::core::project_points(xyz, projection), std::exception);
    }

    TEST_P(TensorPointOps, ProjectionQueuedOperationsOwnTemporaryTables) {
        std::vector<Tensor> results;
        for (int i = 1; i <= 24; ++i) {
            projection.model = i % 3 == 0   ? PointProjectionModel::Equirectangular
                               : i % 3 == 1 ? PointProjectionModel::Pinhole
                                            : PointProjectionModel::Orthographic;
            auto xyz = points3D({0, 0, -2});
            auto matrix = Tensor::from_vector(
                {1.f, 0.f, 0.f, float(i), 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f},
                {4, 4}, device);
            auto index = Tensor::zeros({1}, device, DataType::Int32);
            auto visibility = Tensor::from_vector(std::vector<bool>{true}, {1}, device);
            results.push_back(lfs::core::project_points(xyz, projection, &matrix, &index, &visibility));
        }
        for (size_t i = 0; i < results.size(); ++i) {
            const int node = static_cast<int>(i + 1);
            const auto xy = results[i].cpu().to_vector();
            ASSERT_EQ(xy.size(), 2);
            if (node % 3 == 0) {
                constexpr float pi = 3.14159265358979323846f;
                EXPECT_NEAR(xy[0], (std::atan2(static_cast<float>(node), 2.f) / (2 * pi) + .5f) * 400, .001f);
                EXPECT_FLOAT_EQ(xy[1], 100);
            } else {
                EXPECT_FLOAT_EQ(xy[0], 100 + (node % 3 == 1 ? 10 : 1) * node);
                EXPECT_FLOAT_EQ(xy[1], 80);
            }
        }
    }

    TEST_P(TensorPointOps, ProjectionLargeQueuedOutputsRemainIndependent) {
        constexpr size_t count = 2'100'001;
        const auto xyz = Tensor::full({count, 3}, 1.f, device);
        projection.translation = {0, 0, 3};
        std::vector<Tensor> outputs;
        for (int i = 0; i < 4; ++i) {
            projection.center_x = 100.f * i;
            outputs.push_back(lfs::core::project_points(xyz, projection));
            (void)lfs::core::project_points(xyz, projection);
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            EXPECT_EQ(outputs[i].shape(), (TensorShape{count, 2}));
            for (const size_t row : {size_t{0}, count / 2, count - 1})
                EXPECT_EQ(outputs[i].slice(0, row, row + 1).cpu().to_vector(),
                          (std::vector<float>{100.f * i + 10.f, 65.f}));
        }
    }

    class TensorPointOpsStreams : public lfs::test::CudaBackendTest {};

    TEST_F(TensorPointOpsStreams, ProjectionWaitsForEveryInputAndRetainsStorage) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor result;
            {
                Tensor xyz, matrix, index, visibility;
                {
                    CUDAStreamGuard guard(producer);
                    xyz = Tensor::from_vector({0.f, 0.f, -2.f}, {1, 3}, Device::GPU);
                    matrix = Tensor::from_vector(
                        {1.f, 0.f, 0.f, 2.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}, {4, 4}, Device::GPU);
                    index = Tensor::zeros({1}, Device::GPU, DataType::Int32);
                    visibility = Tensor::from_vector(std::vector<bool>{true}, {1}, Device::GPU);
                }
                CUDAStreamGuard guard(consumer);
                result = lfs::core::project_points(xyz, PointProjection{}, &matrix, &index, &visibility);
                EXPECT_EQ(result.stream(), consumer);
            }
            EXPECT_EQ(result.cpu().to_vector(), (std::vector<float>{1, 0}));
        });
    }

    TEST_P(TensorPointOps, PointRegionPreservesOffsetNeighborsAndStorageBackend) {
        const auto xy = screen_points({0, 0, 1, 0, 2, 0, 0, 1, 0, 2});
        for (int offset = 0; offset < 4; ++offset) {
            auto storage = mask(12, 9);
            auto selected = storage.slice(0, offset, offset + 5);
            GpuBackendScope other(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
            lfs::core::mark_points_2d(selected, xy, {.radius = 1});
            std::vector<float> expected(12, 9);
            for (int index : {0, 1, 3})
                expected[offset + index] = 1;
            EXPECT_EQ(storage.cpu().to_vector(), expected);
            EXPECT_EQ(gpu_backend_of(selected), gpu_backend_of(xy));
        }
    }

    TEST_P(TensorPointOps, PointRegionSupportsStridedInputAndOutputWithoutChangingSiblings) {
        const auto storage = Tensor::from_vector({9.f, 0.f, 0.f, 9.f, 9.f, 1.f, 1.f, 9.f, 9.f, 2.f, 2.f, 9.f}, {3, 4}, device);
        const auto xy = storage.slice(1, 1, 3);
        auto output = Tensor::full({3, 2}, 7, device, DataType::UInt8);
        auto selected = output.slice(1, 1, 2).squeeze(1);
        ASSERT_FALSE(xy.is_contiguous());
        ASSERT_FALSE(selected.is_contiguous());
        lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Rectangle, .x0 = 1, .y0 = 1, .x1 = 2, .y1 = 2});
        EXPECT_EQ(output.cpu().to_vector(), (std::vector<float>{7, 7, 7, 1, 7, 1}));
    }

    TEST_P(TensorPointOps, PointRegionPolygonEvenOddAndDiskUnionAccumulate) {
        const auto xy = screen_points({.5f, .5f, 2, .5f, .5f, 2, 2, 2, 5, 5});
        const auto concave = screen_points({0, 0, 3, 0, 3, 1, 1, 1, 1, 3, 0, 3});
        auto selected = mask(5);
        lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Polygon}, &concave);
        EXPECT_EQ(selected.cpu().to_vector(), (std::vector<float>{1, 1, 1, 0, 0}));
        const auto centers = screen_points({2, 2, 5, 5});
        lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Disks, .radius = 0}, &centers);
        EXPECT_EQ(selected.cpu().to_vector(), (std::vector<float>{1, 1, 1, 1, 1}));
        selected.zero_();
        const auto crossed = screen_points({0, 0, 3, 3, 0, 3, 3, 0});
        lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Polygon}, &crossed);
        EXPECT_EQ(selected.cpu().to_vector(), (std::vector<float>{1, 1, 0, 0, 0}));
    }

    TEST_P(TensorPointOps, PointRegionInvalidCoordinatesAndEmptyGeometryPreserveExistingValues) {
        const auto nan = std::numeric_limits<float>::quiet_NaN();
        const auto inf = std::numeric_limits<float>::infinity();
        const auto xy = screen_points({nan, 0, 0, nan, inf, 0, -inf, 0, -1e8f, 0, 0, 0});
        auto selected = mask(6, 7);
        lfs::core::mark_points_2d(selected, xy, {.radius = 1, .minimum_coordinate = -1000});
        EXPECT_EQ(selected.cpu().to_vector(), (std::vector<float>{7, 7, 7, 7, 7, 1}));
        for (const auto kind : {PointRegion2DKind::Disks, PointRegion2DKind::Polygon}) {
            const auto empty = screen_points({});
            lfs::core::mark_points_2d(selected, xy, {.kind = kind}, &empty);
        }
        EXPECT_EQ(selected.cpu().to_vector(), (std::vector<float>{7, 7, 7, 7, 7, 1}));
    }

    TEST_P(TensorPointOps, PointRegionRejectsMalformedInputs) {
        const auto xy = screen_points({0, 0});
        auto selected = mask(1, 9);
        auto wrong_mask = Tensor::zeros({1}, device);
        EXPECT_THROW(lfs::core::mark_points_2d(wrong_mask, xy, {}), std::exception);
        EXPECT_THROW(lfs::core::mark_points_2d(selected, xy.reshape({2}), {}), std::exception);
        EXPECT_THROW(lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Polygon}), std::exception);
        auto broadcast = selected.broadcast_to({2});
        EXPECT_THROW(lfs::core::mark_points_2d(broadcast, screen_points({0, 0, 0, 0}), {}), std::exception);
        auto empty_mask = mask(0);
        lfs::core::mark_points_2d(empty_mask, screen_points({}), {});
        EXPECT_EQ(empty_mask.numel(), 0);
    }

    TEST_P(TensorPointOps, PointRegionQueuedCallsRetainPointsGeometryAndUnalignedMasks) {
        std::vector<Tensor> results;
        for (int i = 0; i < 24; ++i) {
            auto allocation = Tensor::full({9}, 0, device, DataType::Bool);
            auto selected = allocation.slice(0, 1, 6);
            const auto xy = screen_points({0, 0, 1, 1, 2, 2, 3, 3, 4, 4});
            const auto centers = screen_points({float(i % 5), float(i % 5)});
            lfs::core::mark_points_2d(selected, xy, {.kind = PointRegion2DKind::Disks, .radius = 0}, &centers);
            results.push_back(allocation);
        }
        for (size_t i = 0; i < results.size(); ++i) {
            std::vector<bool> expected(9, false);
            expected[1 + i % 5] = true;
            EXPECT_EQ(results[i].cpu().to_vector_bool(), expected);
        }
    }

    TEST_F(TensorPointOpsStreams, PointRegionOrdersSeparateProducersAndRetainsInputs) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor mask;
            {
                const CUDAStreamGuard guard(consumer);
                mask = Tensor::zeros({2}, Device::GPU, DataType::Bool);
            }
            {
                Tensor xy, polygon;
                {
                    const CUDAStreamGuard guard(producer);
                    xy = Tensor::from_vector({0.f, 0.f, 4.f, 4.f}, {2, 2}, Device::GPU);
                    polygon = Tensor::from_vector({-1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f}, {4, 2}, Device::GPU);
                }
                const CUDAStreamGuard guard(consumer);
                lfs::core::mark_points_2d(mask, xy, {.kind = PointRegion2DKind::Polygon}, &polygon);
                EXPECT_EQ(mask.stream(), consumer);
            }
            EXPECT_EQ(mask.cpu().to_vector_bool(), (std::vector<bool>{true, false}));
        });
    }

    TEST_P(TensorPointOps, LabelsDenseModesPreserveLocksCategoriesOffsets) {
        const auto selected = bytes({1, 1, 0, 1, 1, 0, 1, 1, 1});
        const auto existing = bytes({2, 7, 1, 0, 255, 7, 0, 31, 1});
        const auto categories = ints({0, 0, 0, 1, 0, -1, 2, 9, 0});
        const auto allowed = bytes({1, 0, 1});
        std::vector<bool> flags(256, false);
        for (int group : {1, 2, 31, 255})
            flags[group] = true;
        const auto locked = Tensor::from_vector(flags, {256}, device);
        const std::vector<std::vector<uint8_t>> expected{
            {2, 1, 1, 0, 255, 7, 1, 31, 1},
            {2, 7, 1, 0, 255, 7, 0, 31, 0},
            {2, 1, 0, 0, 255, 7, 1, 31, 1}};
        for (int offset = 0; offset < 4; ++offset) {
            for (int mode = 0; mode < 3; ++mode) {
                auto storage = Tensor::full({16}, 99, device, DataType::UInt8);
                auto output = storage.slice(0, offset, offset + 9);
                const GpuBackendScope opposite(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
                update_labels(output, selected, {.label = 1, .mode = static_cast<LabelUpdateMode>(mode), .existing = &existing, .locked = &locked, .categories = &categories, .allowed = &allowed});
                auto reference = std::vector<uint8_t>(16, 99);
                std::copy(expected[mode].begin(), expected[mode].end(), reference.begin() + offset);
                EXPECT_EQ(storage.to_vector_uint8(), reference);
                EXPECT_EQ(gpu_backend_of(output), gpu_backend_of(selected));
            }
        }
    }

    TEST_P(TensorPointOps, LabelsIndexedReplacementKeepsOtherBytesAndSelectedDuplicates) {
        const auto selected = bytes({0, 1, 1, 1, 1, 1, 1});
        const auto indices = ints({4, 4, 0, 9, -1, 99, 1});
        const auto existing = bytes({0, 255, 1, 1, 1, 1, 1, 1, 1, 1});
        std::vector<bool> flags(256, false);
        flags[255] = true;
        const auto locks = Tensor::from_vector(flags, {256}, device);
        const auto categories = ints({0, 0, 1, 0, 0, 0, -1});
        const auto allowed = bytes({1, 0});
        for (int offset = 0; offset < 4; ++offset) {
            for (bool restrict : {false, true}) {
                auto parent = Tensor::full({16}, 77, device, DataType::UInt8);
                auto output = parent.slice(0, offset, offset + 10);
                update_labels(output, selected, {.label = 1, .mode = LabelUpdateMode::Replace, .existing = &existing, .locked = &locks, .indices = &indices, .categories = restrict ? &categories : nullptr, .allowed = restrict ? &allowed : nullptr});
                auto expected = std::vector<uint8_t>(16, 77);
                const std::vector<uint8_t> values{static_cast<uint8_t>(restrict ? 0 : 1), 255, 1, 1, 1, 1, 1, 1, 1, 1};
                std::copy(values.begin(), values.end(), expected.begin() + offset);
                EXPECT_EQ(parent.to_vector_uint8(), expected);
            }
        }
    }

    TEST_P(TensorPointOps, LabelsStridedAliasedInputsRetainOriginalValuesAndNeighbors) {
        auto parent = bytes({1, 99, 2, 99, 3, 99}).reshape({3, 2});
        auto output = parent.slice(1, 0, 1).squeeze(1);
        const auto selection_parent = bytes({1, 0, 0, 0, 2, 0}).reshape({3, 2});
        const auto selected = selection_parent.slice(1, 0, 1).squeeze(1);
        ASSERT_FALSE(output.is_contiguous());
        ASSERT_FALSE(selected.is_contiguous());
        const Tensor previous = output == 2;
        update_labels(output, selected, {.label = 2, .mode = LabelUpdateMode::Replace, .existing = &output});
        EXPECT_EQ(parent.to_vector_uint8(), (std::vector<uint8_t>{2, 99, 0, 99, 2, 99}));
        EXPECT_EQ(previous.to_vector_bool(), (std::vector<bool>{false, true, false}));
        auto shared = bytes({1, 0, 3});
        update_labels(shared, shared, {.label = 4, .existing = &shared});
        EXPECT_EQ(shared.to_vector_uint8(), (std::vector<uint8_t>{4, 0, 4}));
    }

    TEST_P(TensorPointOps, LabelsAllByteLabelsAndMissingOrEmptyLookups) {
        std::vector<int> values(256);
        std::iota(values.begin(), values.end(), 0);
        const auto existing = bytes(values);
        const auto selected = Tensor::full({256}, 2, device, DataType::UInt8);
        const auto locked = Tensor::full_bool({256}, true, device);
        auto output = Tensor::empty({256}, device, DataType::UInt8);
        update_labels(output, selected, {.label = 255, .existing = &existing, .locked = &locked});
        auto expected = existing.to_vector_uint8();
        expected[0] = 255;
        EXPECT_EQ(output.to_vector_uint8(), expected);
        update_labels(output, selected, {.label = 255});
        EXPECT_EQ(output.to_vector_uint8(), std::vector<uint8_t>(256, 255));
        const auto categories = Tensor::zeros({256}, device, DataType::Int32);
        const auto empty = Tensor::empty({0}, device, DataType::Bool);
        update_labels(output, selected, {.label = 1, .existing = &existing, .categories = &categories, .allowed = &empty});
        EXPECT_EQ(output.to_vector_uint8(), existing.to_vector_uint8());
    }

    TEST_P(TensorPointOps, LabelsQueuedTemporaryInputsKeepTheirStorage) {
        std::vector<Tensor> outputs;
        std::vector<uint8_t> expected;
        for (int i = 0; i < 24; ++i) {
            auto out = Tensor::empty({4099}, device, DataType::UInt8);
            const auto selected = Tensor::full({4099}, 2, device, DataType::UInt8);
            const auto existing = Tensor::full({4099}, 2, device, DataType::UInt8);
            std::vector<bool> flags(256, false);
            flags[2] = i % 2 == 0;
            const auto locked = Tensor::from_vector(flags, {256}, device);
            const auto categories = Tensor::full({4099}, i % 3, device, DataType::Int32);
            const auto allowed = bytes({1, 0});
            update_labels(out, selected, {.existing = &existing, .locked = &locked, .categories = &categories, .allowed = &allowed});
            outputs.push_back(std::move(out));
            expected.push_back(i % 2 != 0 && i % 3 == 0 ? 1 : 2);
        }
        for (size_t i = 0; i < outputs.size(); ++i)
            EXPECT_EQ(outputs[i].to_vector_uint8(), std::vector<uint8_t>(4099, expected[i]));
    }

    TEST_P(TensorPointOps, LabelsEmptyInputsAndInvalidContracts) {
        const auto empty = Tensor::empty({0}, device, DataType::Bool);
        const auto indices = Tensor::empty({0}, device, DataType::Int32);
        auto output = bytes({2, 3, 4});
        const auto old = bytes({5, 6, 7});
        update_labels(output, empty, {.existing = &old, .indices = &indices});
        EXPECT_EQ(output.to_vector_uint8(), (std::vector<uint8_t>{5, 6, 7}));
        update_labels(output, empty, {.indices = &indices});
        EXPECT_EQ(output.to_vector_uint8(), (std::vector<uint8_t>{0, 0, 0}));
        auto no_output = Tensor::empty({0}, device, DataType::UInt8);
        update_labels(no_output, empty, {});
        const auto selected = bytes({1, 0, 1});
        const auto short_locks = bytes({0});
        EXPECT_THROW(update_labels(output, selected, {.locked = &short_locks}), std::exception);
        EXPECT_THROW(update_labels(output, selected, {.categories = &indices}), std::exception);
        auto broadcast = bytes({0}).broadcast_to({3});
        EXPECT_THROW(update_labels(broadcast, selected, {}), std::exception);
    }

    TEST_F(TensorPointOpsStreams, LabelsHonorsConsumerStreamAndProducerDependencies) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor selected, existing;
            {
                const CUDAStreamGuard guard(producer);
                selected = Tensor::full_bool({8193}, true, Device::GPU);
                existing = Tensor::full({8193}, 3, Device::GPU, DataType::UInt8);
                (void)selected.data_ptr();
                (void)existing.data_ptr();
            }
            const CUDAStreamGuard output_scope(consumer);
            auto storage = Tensor::full({8197}, 99, Device::GPU, DataType::UInt8);
            auto output = storage.slice(0, 1, 8194);
            {
                const CUDAStreamGuard guard(consumer);
                update_labels(output, selected, {.label = 7, .existing = &existing});
            }
            EXPECT_EQ(output.stream(), consumer);
            EXPECT_EQ(output.to_vector_uint8(), std::vector<uint8_t>(8193, 7));
            auto expected_storage = std::vector<uint8_t>(8197, 7);
            expected_storage[0] = expected_storage[8194] = expected_storage[8195] = expected_storage[8196] = 99;
            EXPECT_EQ(storage.to_vector_uint8(), expected_storage);
        });
    }

    TEST_P(TensorPointOps, FiltersCategoriesKeepLabelsOffsetsAndEmptyTables) {
        const auto ids = ints({0, 1, -1, 3, 0, 9});
        const auto allowed = bytes({255, 0, 1, 2});
        const auto empty = bytes({});
        for (int offset = 0; offset < 4; ++offset) {
            auto storage = Tensor::full({12}, 255, device, DataType::UInt8);
            auto mask = storage.slice(0, offset, offset + 6);
            auto before = mask == 255;
            const GpuBackendScope opposite(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
            filter_points(mask, nullptr, {.indices = &ids, .allowed = &allowed});
            std::vector<uint8_t> expected(12, 255);
            expected[offset + 1] = expected[offset + 2] = expected[offset + 5] = 0;
            EXPECT_EQ(storage.to_vector_uint8(), expected);
            EXPECT_EQ(before.to_vector_bool(), std::vector<bool>(6, true));
            filter_points(mask, nullptr, {.indices = &ids, .allowed = &empty});
            EXPECT_EQ(mask.to_vector_uint8(), std::vector<uint8_t>(6, 0));
        }
    }

    TEST_P(TensorPointOps, FiltersStridedMasksPointsAndAliasedAllowedFlags) {
        auto alias = bytes({1, 1, 1});
        const auto ids = ints({3, 0, 0});
        filter_points(alias, nullptr, {.indices = &ids, .allowed = &alias});
        EXPECT_EQ(alias.to_vector_uint8(), (std::vector<uint8_t>{0, 1, 1}));
        auto storage = bytes({255, 77, 128, 77, 2, 77, 1, 77}).reshape({4, 2});
        auto mask = storage.slice(1, 0, 1).squeeze(1);
        const auto points = floats({-2, 0, 0, 99, 0, 0, 0, 99, 1, 0, 0, 99, 2, 0, 0, 99}, {4, 4}).slice(1, 0, 3);
        const auto box = identity(), lo = floats({-1, -1, -1}, {3}), hi = floats({1, 1, 1}, {3});
        filter_points(mask, &points, {.box_transform = &box, .box_min = &lo, .box_max = &hi});
        EXPECT_EQ(storage.flatten().to_vector_uint8(), (std::vector<uint8_t>{0, 77, 128, 77, 2, 77, 0, 77}));
    }

    TEST_P(TensorPointOps, FiltersShapesKeepInclusiveBoundariesAndInversion) {
        const auto points = floats({-2, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0}, {6, 3});
        const auto matrix = identity(), lo = floats({-1, -1, -1}, {3}), hi = floats({1, 1, 1}, {3}), r = floats({1, 1, 1}, {3});
        for (bool invert : {false, true})
            for (bool ellipse : {false, true}) {
                auto mask = bytes({1, 2, 3, 4, 5, 6});
                PointFilter filter{.box_inverse = invert, .ellipsoid_inverse = invert};
                if (ellipse) {
                    filter.ellipsoid_transform = &matrix;
                    filter.ellipsoid_radii = &r;
                } else {
                    filter.box_transform = &matrix;
                    filter.box_min = &lo;
                    filter.box_max = &hi;
                }
                filter_points(mask, &points, filter);
                EXPECT_EQ(mask.to_vector_uint8(), invert ? (std::vector<uint8_t>{1, 0, 0, 0, 5, 6}) : (std::vector<uint8_t>{0, 2, 3, 4, 0, 0}));
            }
    }

    TEST_P(TensorPointOps, FiltersWindowsUseRealIntrinsicsAndInclusiveDepth) {
        const auto points = floats({0, 0, -1, 0, 0, -2, 0, 0, -3, 0, 0, 1, 1, 0, -1, 0, 0, 0}, {6, 3});
        PointFilterWindow w{.projection = {.focal_x = 10, .focal_y = 10, .center_x = 60, .center_y = 50, .ortho_scale = 10, .width = 100, .height = 100}, .near_depth = 1, .far_depth = 2, .scale_x = .4f, .scale_y = .5f};
        for (auto model : {PointProjectionModel::Pinhole, PointProjectionModel::Orthographic, PointProjectionModel::Equirectangular}) {
            w.projection.model = model;
            auto mask = bytes({1, 2, 3, 4, 5, 6});
            filter_points(mask, &points, {.window = &w});
            EXPECT_EQ(mask.to_vector_uint8(), (std::vector<uint8_t>{1, 2, 0, 0, 5, 0}));
        }
        w.projection.model = PointProjectionModel::Pinhole;
        w.projection.center_x = 81;
        auto mask = bytes({1, 2, 3, 4, 5, 6});
        filter_points(mask, &points, {.window = &w});
        EXPECT_EQ(mask.to_vector_uint8(), std::vector<uint8_t>(6, 0));
    }

    TEST_P(TensorPointOps, FiltersQueuedTemporaryInputsRetainStorage) {
        std::vector<Tensor> outputs;
        for (int call = 0; call < 24; ++call) {
            auto mask = Tensor::full({8193}, float(call + 1), device, DataType::UInt8);
            const auto points = Tensor::zeros({8193, 3}, device);
            auto box = identity();
            auto lo = floats({float(call % 2), -1, -1}, {3});
            auto hi = floats({2, 1, 1}, {3});
            filter_points(mask, &points, {.box_transform = &box, .box_min = &lo, .box_max = &hi});
            outputs.push_back(mask);
        }
        for (int call = 0; call < 24; ++call)
            EXPECT_EQ(outputs[call].to_vector_uint8(), std::vector<uint8_t>(8193, call % 2 ? 0 : call + 1));
    }

    TEST_P(TensorPointOps, FiltersInvalidContractsAndEmptyInput) {
        auto empty = bytes({});
        filter_points(empty, nullptr, {});
        auto mask = bytes({1, 2});
        auto ids = ints({0});
        auto table = bytes({1});
        EXPECT_THROW(filter_points(mask, nullptr, {.allowed = &table}), std::exception);
        EXPECT_THROW(filter_points(mask, nullptr, {.indices = &ids}), std::exception);
        auto matrix = identity();
        EXPECT_THROW(filter_points(mask, nullptr, {.box_transform = &matrix}), std::exception);
        auto broadcast = bytes({1}).broadcast_to({2});
        EXPECT_THROW(filter_points(broadcast, nullptr, {}), std::exception);
    }

    TEST_F(TensorPointOpsStreams, FiltersHonorsConsumerStreamAndProducerDependencies) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor ids, allowed;
            {
                const CUDAStreamGuard guard(producer);
                ids = Tensor::zeros({8193}, Device::GPU, DataType::Int32);
                allowed = Tensor::zeros({1}, Device::GPU, DataType::UInt8);
                (void)ids.data_ptr();
                (void)allowed.data_ptr();
            }
            const CUDAStreamGuard output_scope(consumer);
            auto storage = Tensor::full({8197}, 99, Device::GPU, DataType::UInt8);
            auto mask = storage.slice(0, 1, 8194);
            {
                const CUDAStreamGuard guard(consumer);
                filter_points(mask, nullptr, {.indices = &ids, .allowed = &allowed});
            }
            EXPECT_EQ(mask.stream(), consumer);
            EXPECT_EQ(mask.to_vector_uint8(), std::vector<uint8_t>(8193, 0));
            std::vector<uint8_t> expected(8197, 0);
            expected[0] = expected[8194] = expected[8195] = expected[8196] = 99;
            EXPECT_EQ(storage.to_vector_uint8(), expected);
        });
    }

    std::vector<int> reference(const std::vector<uint8_t>& values, const size_t bins = 257) {
        std::vector<int> result(bins);
        for (auto value : values)
            if (value != 0)
                ++result[value];
        return result;
    }

    TEST_P(TensorPointOps, HistogramCountsAllBinsBeyondOneGridAndUsesStorageBackend) {
        std::vector<uint8_t> values(256 * 4096 + 31);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<uint8_t>(i % 256);
        const auto input = bytes(values);
        auto counts = output();
        GpuBackendScope different_default(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        {
            histogram_u8(input, counts);
            EXPECT_EQ(counts.cpu().to_vector_int(), reference(values));
            EXPECT_EQ(gpu_backend_of(counts), gpu_backend_of(input));
        }
        EXPECT_EQ(input.cpu().to_vector_uint8(), values);
    }

    TEST_P(TensorPointOps, HistogramClearsReusedCountsForEmptyAndUniformInputs) {
        auto counts = output();
        for (const size_t size : {65539u, 0u, 1u, 255u, 257u}) {
            for (const uint8_t value : {0, 1, 255}) {
                const std::vector<uint8_t> values(size, value);
                {
                    histogram_u8(bytes(values), counts);
                    EXPECT_EQ(counts.cpu().to_vector_int(), reference(values));
                }
            }
        }
    }

    TEST_P(TensorPointOps, HistogramAcceptsOffsetAndStridedInputAndOutputViews) {
        const auto input = bytes({99, 0, 1, 255, 7, 255, 99});
        auto backing = Tensor::full({259}, -7.f, device, DataType::Int32);
        auto counts = backing.slice(0, 1, 258);
        histogram_u8(input.slice(0, 1, 6), counts);
        EXPECT_EQ(counts.cpu().to_vector_int(), reference({0, 1, 255, 7, 255}));
        const auto whole = backing.cpu().to_vector_int();
        EXPECT_EQ(whole.front(), -7);
        EXPECT_EQ(whole.back(), -7);

        const auto strided = bytes({99, 0, 99, 255, 99, 3, 99, 255}).reshape({4, 2}).slice(1, 1, 2);
        ASSERT_FALSE(strided.is_contiguous());
        histogram_u8(strided, counts);
        EXPECT_EQ(counts.cpu().to_vector_int(), reference({0, 255, 3, 255}));
        histogram_u8(strided.gt(0.f), counts);
        EXPECT_EQ(counts.cpu().to_vector_int(), reference({0, 1, 1, 1}));
    }

    TEST_P(TensorPointOps, HistogramRetainsTemporaryInputsAndIndependentQueuedOutputs) {
        std::vector<Tensor> outputs;
        for (int i = 0; i < 16; ++i) {
            auto input = Tensor::empty({65539, 1}, device, DataType::Bool);
            input.fill_(float(i % 2), input.stream());
            auto counts = Tensor::empty({257}, device, DataType::Int32);
            histogram_u8(input, counts);
            outputs.push_back(std::move(counts));
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::vector<int> expected(257);
            if (i % 2)
                expected[1] = 65539;
            EXPECT_EQ(outputs[i].cpu().to_vector_int(), expected);
        }
    }

    TEST_P(TensorPointOps, HistogramRejectsInvalidInputAndOutputContracts) {
        const auto input = bytes({0, 1, 255});
        auto counts = output();
        EXPECT_THROW(histogram_u8(Tensor{}, counts), std::exception);
        EXPECT_THROW(histogram_u8(input.to(DataType::Int32), counts), std::exception);
        auto small = counts.slice(0, 0, 255);
        EXPECT_THROW(histogram_u8(input, small), std::exception);
        auto float_counts = counts.to(DataType::Float32);
        EXPECT_THROW(histogram_u8(input, float_counts), std::exception);
        auto strided_counts = Tensor::empty({257, 2}, device, DataType::Int32).slice(1, 0, 1).squeeze(1);
        EXPECT_THROW(histogram_u8(input, strided_counts), std::exception);
        auto matrix_counts = counts.reshape({257, 1});
        EXPECT_THROW(histogram_u8(input, matrix_counts), std::exception);
        EXPECT_EQ(counts.cpu().to_vector_int(), std::vector<int>(257, -7));
    }

    TEST(TensorPointOpsHistogramBackends, RejectsMixedBackendsAndDevices) {
        if (!gpu_backend_available(GpuBackend::CUDA) || !gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP();
        Tensor cuda_counts;
        {
            GpuBackendScope scope(GpuBackend::CUDA);
            cuda_counts = Tensor::empty({256}, Device::GPU, DataType::Int32);
        }
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const auto input = Tensor::zeros({3}, Device::GPU, DataType::UInt8);
            EXPECT_THROW(histogram_u8(input, cuda_counts), std::exception);
            EXPECT_THROW(histogram_u8(input.cpu(), cuda_counts), std::exception);
        }
        EXPECT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        EXPECT_TRUE(internal::vulkan_validation_messages_for_testing().empty());
    }

    TEST_F(TensorPointOpsStreams, HistogramWaitsForProducersAndRetainsTemporaryStorage) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor counts;
            {
                CUDAStreamGuard guard(consumer);
                counts = Tensor::empty({257}, Device::GPU, DataType::Int32);
            }
            for (const bool selected : {true, false, true}) {
                Tensor source;
                {
                    CUDAStreamGuard guard(producer);
                    source = Tensor::empty({1048607}, Device::GPU, DataType::Bool);
                    source.fill_(selected ? 1.f : 0.f, producer);
                }
                {
                    CUDAStreamGuard guard(consumer);
                    histogram_u8(source, counts);
                }
                source = {};
                std::vector<int> expected(257);
                if (selected)
                    expected[1] = 1048607;
                EXPECT_EQ(counts.cpu().to_vector_int(), expected);
            }
        });
    }

    std::vector<bool> brute_radius(const std::vector<float>& xyz, const std::vector<uint8_t>& mask,
                                   const float radius) {
        std::vector<bool> expected(mask.size(), false);
        for (size_t i = 0; i < mask.size(); ++i) {
            for (size_t j = 0; j < mask.size(); ++j) {
                if (!mask[j]) {
                    continue;
                }
                // Double precision provides an independent distance reference.
                const double x = double(xyz[i * 3]) - xyz[j * 3];
                const double y = double(xyz[i * 3 + 1]) - xyz[j * 3 + 1];
                const double z = double(xyz[i * 3 + 2]) - xyz[j * 3 + 2];
                if (x * x + y * y + z * z <= double(radius) * radius) {
                    expected[i] = true;
                    break;
                }
            }
        }
        return expected;
    }

    TEST_P(TensorPointOps, SpatialMatchesBruteForceAcrossCellsAndLargeExtents) {
        constexpr size_t count = 1027; // Includes a partial output word/workgroup.
        std::mt19937 random(921);
        std::uniform_int_distribution<int> coordinate(-128, 128);
        std::vector<float> xyz(count * 3);
        std::vector<uint8_t> mask(count);
        for (size_t i = 0; i < count; ++i) {
            // Exact quarter coordinates keep the boundary comparison unambiguous.
            for (size_t axis = 0; axis < 3; ++axis) {
                xyz[i * 3 + axis] = float(coordinate(random)) * 0.25f;
            }
            mask[i] = i % 13 == 0 ? 9 : 0;
        }
        xyz[0] = -1e10f;
        xyz[3] = -1e10f;
        xyz[1] = xyz[2] = xyz[4] = xyz[5] = 0.0f;
        xyz[6] = 1e10f;
        xyz[9] = 1e10f;
        xyz[7] = xyz[8] = xyz[10] = xyz[11] = 0.0f;
        mask[2] = 5;
        const auto points = points3D(xyz);
        const auto references = bytes(mask);
        const auto expected = brute_radius(xyz, mask, 1.0f);

        GpuBackendScope different_default(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const auto output = radius_neighbors(points, references, 1.0f);
        EXPECT_EQ(output.dtype(), DataType::Bool);
        EXPECT_EQ(output.device(), device);
        EXPECT_EQ(gpu_backend_of(output), gpu_backend_of(points));
        EXPECT_EQ(output.cpu().to_vector_bool(), expected);
        EXPECT_EQ(references.cpu().to_vector_uint8(), mask);
        EXPECT_EQ(points.cpu().to_vector(), xyz);
    }

    TEST_P(TensorPointOps, SpatialAcceptsOffsetStridedInputsAndBooleanReferences) {
        const auto padded_points = Tensor::from_vector(
            {99.f, 99.f, 99.f, 99.f, 99.f, -1.f, 0.f, 0.f,
             99.f, 0.f, 0.f, 0.f, 99.f, 1.f, 0.f, 0.f, 99.f, 4.f, 0.f, 0.f},
            {5, 4}, device);
        const auto points = padded_points.slice(0, 1, 5).slice(1, 1, 4);
        const auto padded_mask = bytes({99, 0, 99, 5, 99, 0, 99, 0}).reshape({4, 2});
        const auto references = padded_mask.slice(1, 1, 2).squeeze(1);
        ASSERT_FALSE(points.is_contiguous());
        ASSERT_FALSE(references.is_contiguous());
        const std::vector<bool> expected{true, true, true, false};
        EXPECT_EQ(radius_neighbors(points, references, 1.0f).cpu().to_vector_bool(), expected);
        EXPECT_EQ(radius_neighbors(points, references.gt(0.0f), 1.0f).cpu().to_vector_bool(), expected);
    }

    TEST_P(TensorPointOps, SpatialHandlesEmptyAndUniformMasksWithoutChangingStorageBackend) {
        const auto empty_points = points3D({});
        const auto empty_mask = bytes({});
        const auto points = points3D({0.f, 0.f, 0.f, 0.25f, 0.f, 0.f, 2.f, 0.f, 0.f});
        const auto none = bytes({0, 0, 0});
        const auto all = bytes({3, 7, 9});
        GpuBackendScope different_default(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const auto empty = radius_neighbors(empty_points, empty_mask, 1.0f);
        EXPECT_EQ(empty.numel(), 0u);
        EXPECT_EQ(empty.device(), device);
        EXPECT_EQ(gpu_backend_of(empty), gpu_backend_of(empty_points));
        EXPECT_EQ(radius_neighbors(points, none, 1.0f).to_vector_bool(), (std::vector<bool>{false, false, false}));
        EXPECT_EQ(radius_neighbors(points, all, 1.0f).to_vector_bool(), (std::vector<bool>{true, true, true}));
    }

    TEST_P(TensorPointOps, SpatialHandlesSmallAndLargeFiniteRadii) {
        for (const float radius : {1.0e-30f, 1.0e-20f, 1.0e30f}) {
            SCOPED_TRACE(radius);
            const auto points = points3D({0.f, 0.f, 0.f, radius * 0.5f, 0.f, 0.f,
                                          radius, 0.f, 0.f, radius * 2.f, 0.f, 0.f});
            EXPECT_EQ(radius_neighbors(points, bytes({1, 0, 0, 0}), radius).cpu().to_vector_bool(),
                      (std::vector<bool>{true, true, true, false}));
        }
    }

    TEST_P(TensorPointOps, SpatialRejectsInvalidRadiusAndMismatchedInputs) {
        const auto points = points3D({0.f, 0.f, 0.f});
        const auto mask = bytes({1});
        EXPECT_THROW(radius_neighbors(points, mask, 0.f), std::exception);
        EXPECT_THROW(radius_neighbors(points, mask, -1.f), std::exception);
        EXPECT_THROW(radius_neighbors(points, mask, std::numeric_limits<float>::quiet_NaN()), std::exception);
        EXPECT_THROW(radius_neighbors(points, mask, std::numeric_limits<float>::infinity()), std::exception);
        EXPECT_THROW(radius_neighbors(points, mask, std::numeric_limits<float>::denorm_min()), std::exception);
        EXPECT_THROW(radius_neighbors(points, bytes({1, 0}), 1.f), std::exception);
        EXPECT_THROW(radius_neighbors(points, mask.to(DataType::Int32), 1.f), std::exception);
        EXPECT_THROW(radius_neighbors(points.reshape({3}), mask, 1.f), std::exception);
    }

    TEST_F(TensorPointOpsStreams, SpatialWaitsForInputsAndRetainsTemporaryStorage) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            Tensor result;
            {
                Tensor points, mask;
                {
                    CUDAStreamGuard guard(producer);
                    points = Tensor::from_vector({0.f, 0.f, 0.f, 0.5f, 0.f, 0.f, 3.f, 0.f, 0.f}, {3, 3}, Device::GPU);
                    mask = Tensor::from_vector(std::vector<bool>{true, false, false}, {3}, Device::GPU);
                }
                CUDAStreamGuard guard(consumer);
                result = radius_neighbors(points, mask, 1.f);
            }
            // Inputs and the operation's hash storage have gone out of scope.
            EXPECT_EQ(result.cpu().to_vector_bool(), (std::vector<bool>{true, true, false}));
        });
    }

    TEST_P(TensorPointOps, ReadbackRetainsQueuedInputAndUsesItsBackend) {
        TensorReadback readback;
        {
            auto source = tensor({1.f, 2.f, 3.f, 4.f});
            GpuBackendScope opposite(backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
            readback.enqueue(source);
            source.fill_(99.f);
        }
        ASSERT_TRUE(readback.pending());
        std::array<float, 4> host{};
        readback.wait(std::as_writable_bytes(std::span(host)));
        EXPECT_EQ(host, (std::array<float, 4>{1.f, 2.f, 3.f, 4.f}));
        EXPECT_FALSE(readback.pending());
    }

    TEST_P(TensorPointOps, ReadbackMaterializesLazyExpressionsAndStridedOffsetViews) {
        auto values = tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}).reshape({4, 3});
        TensorReadback readback;
        readback.enqueue(values.slice(0, 1, 4).slice(1, 1, 3));
        std::array<float, 6> host{};
        readback.wait(std::as_writable_bytes(std::span(host)));
        EXPECT_EQ(host, (std::array<float, 6>{4, 5, 7, 8, 10, 11}));
        readback.enqueue(tensor({1, 2, 3, 4, 5, 6}).mul(2.f).add(0.5f));
        readback.wait(std::as_writable_bytes(std::span(host)));
        EXPECT_EQ(host, (std::array<float, 6>{2.5f, 4.5f, 6.5f, 8.5f, 10.5f, 12.5f}));
    }

    TEST_P(TensorPointOps, ReadbackReusesSlotForDifferentSizesIncludingEmptyAndByteTails) {
        TensorReadback readback;
        for (size_t size : {0, 3, 4099, 7, 0, 1, 1028}) {
            SCOPED_TRACE(size);
            auto cpu = Tensor::empty({size}, Device::CPU, DataType::UInt8);
            std::vector<uint8_t> expected(size);
            for (size_t i = 0; i < size; ++i)
                expected[i] = static_cast<uint8_t>(i % 251);
            if (size)
                std::copy(expected.begin(), expected.end(), cpu.ptr<uint8_t>());
            readback.enqueue(device == Device::CPU ? cpu : cpu.gpu());
            std::vector<uint8_t> host(size);
            readback.wait(std::as_writable_bytes(std::span(host)));
            EXPECT_EQ(host, expected);
        }
        auto bytes = Tensor::from_vector({11.f, 23.f, 47.f, 89.f, 101.f}, {5}, device).to(DataType::UInt8);
        readback.enqueue(bytes.slice(0, 1, 4));
        std::array<uint8_t, 3> host{};
        readback.wait(std::as_writable_bytes(std::span(host)));
        EXPECT_EQ(host, (std::array<uint8_t, 3>{23, 47, 89}));
    }

    TEST_P(TensorPointOps, ReadbackRejectsMisuseWithoutLosingPendingResult) {
        TensorReadback readback;
        std::array<float, 3> host{};
        EXPECT_THROW((void)readback.poll(std::as_writable_bytes(std::span(host))), std::logic_error);
        EXPECT_THROW(readback.enqueue(Tensor{}), std::invalid_argument);
        readback.enqueue(tensor({1, 2, 3}));
        EXPECT_THROW(readback.enqueue(tensor({8, 9})), std::logic_error);
        EXPECT_THROW((void)readback.poll({}), std::invalid_argument);
        EXPECT_THROW(readback.wait({}), std::invalid_argument);
        EXPECT_TRUE(readback.pending());
        readback.wait(std::as_writable_bytes(std::span(host)));
        EXPECT_EQ(host, (std::array<float, 3>{1, 2, 3}));
    }

    TEST_P(TensorPointOps, ReadbackMoveAndDestructionDrainPendingCopies) {
        std::array<float, 4> host{};
        for (int i = 0; i < 16; ++i) {
            TensorReadback first, second;
            first.enqueue(tensor({1, 2, 3, 4}));
            second.enqueue(tensor({9, 8, 7, 6}));
            second = std::move(first); // Discard and drain the previous download.
            EXPECT_FALSE(first.pending());
            first.enqueue(tensor({5, 6, 7, 8})); // A moved-from slot is reusable.
            TensorReadback third(std::move(second));
            third.wait(std::as_writable_bytes(std::span(host)));
            EXPECT_EQ(host, (std::array<float, 4>{1, 2, 3, 4}));
            // first is intentionally destroyed without polling.
        }
    }

    TEST_P(TensorPointOps, ReadbackPollCompletesIndependentQueuedSlots) {
        std::array<TensorReadback, 12> slots;
        for (size_t i = 0; i < slots.size(); ++i)
            slots[i].enqueue(tensor({float(i), float(i + 1), float(i + 2)}));
        for (size_t i = 0; i < slots.size(); ++i) {
            std::array<float, 3> host{};
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!slots[i].poll(std::as_writable_bytes(std::span(host)))) {
                ASSERT_LT(std::chrono::steady_clock::now(), deadline);
                std::this_thread::yield();
            }
            EXPECT_EQ(host, (std::array<float, 3>{float(i), float(i + 1), float(i + 2)}));
        }
    }

    TEST(TensorPointOpsReadbackBackends, ReusesSlotAcrossStorageBackends) {
        TensorReadback slot;
        for (const auto backend : {GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::CUDA}) {
            if (!gpu_backend_available(backend))
                GTEST_SKIP();
            GpuBackendScope scope(backend);
            for (auto device : {Device::CPU, Device::GPU}) {
                slot.enqueue(Tensor::from_vector({1.f, 2.f, 3.f}, {3}, device));
                std::array<float, 3> host{};
                slot.wait(std::as_writable_bytes(std::span(host)));
                EXPECT_EQ(host, (std::array<float, 3>{1, 2, 3}));
            }
        }
    }

    TEST_F(TensorPointOpsStreams, ReadbackCopyOrdersProducerDestinationAndReadback) {
        withCudaStreams([](cudaStream_t producer, cudaStream_t consumer) {
            TensorReadback readback;
            std::vector<uint8_t> host(65539);
            for (int iteration = 0; iteration < 8; ++iteration) {
                Tensor source, destination;
                {
                    CUDAStreamGuard guard(producer);
                    source = Tensor::full({host.size()}, float(iteration + 1), Device::GPU, DataType::UInt8);
                }
                {
                    CUDAStreamGuard guard(consumer);
                    destination = Tensor::zeros({host.size()}, Device::GPU, DataType::UInt8);
                }
                {
                    CUDAStreamGuard guard(producer);
                    destination.copy_from(source);
                    readback.enqueue(destination);
                }
                source = {};
                destination = {};
                readback.wait(std::as_writable_bytes(std::span(host)));
                EXPECT_TRUE(std::all_of(host.begin(), host.end(), [&](auto value) { return value == iteration + 1; }));
            }
        });
    }

    TEST_P(TensorPointOps, RadiusNeighborScratchSurvivesQueuedCalls) {
        constexpr size_t count = 1027;
        std::mt19937 rng(20260922);
        std::uniform_real_distribution<float> random(-1.f, 1.f);
        std::vector<float> xyz(count * 3);
        for (auto& value : xyz)
            value = random(rng);
        std::vector<bool> selected(count);
        for (size_t i = 0; i < count; ++i) {
            selected[i] = i % 17 == 0;
        }
        const auto cpu_points = Tensor::from_vector(xyz, {count, 3}, Device::CPU);
        const auto cpu_mask = Tensor::from_vector(selected, {count}, Device::CPU);
        const auto expected = radius_neighbors(cpu_points, cpu_mask, 0.2f).to_vector_bool();
        {
            const auto points = cpu_points.to(device);
            const auto mask = cpu_mask.to(device);
            std::vector<Tensor> pending;
            for (int i = 0; i < 24; ++i) {
                // Queued results stay valid after their inputs are released.
                pending.push_back(radius_neighbors(points.clone(), mask.clone(), 0.2f));
            }
            for (const auto& result : pending) {
                EXPECT_EQ(gpu_backend_of(result), gpu_backend_of(points));
                EXPECT_EQ(result.cpu().to_vector_bool(), expected);
            }
        }
    }

    TEST_P(TensorPointOps, CameraModelsAndTransformsAgreeWithCpu) {
        constexpr size_t count = 257;
        std::mt19937 rng(20260922);
        std::uniform_real_distribution<float> random(-5.f, 5.f);
        std::vector<float> coordinates(count * 3);
        for (auto& value : coordinates)
            value = random(rng);
        const auto cpu_points = Tensor::from_vector(coordinates, {count, 3}, Device::CPU);
        const auto positions = cpu_points.to(device);
        const auto cpu_matrix = Tensor::from_vector(std::vector<float>{
                                                        .95f, -.05f, .11f, .7f, .05f, 1.1f, -.03f, -.3f, -.11f, .03f, 1.03f, .2f, 0, 0, 0, 1},
                                                    {4, 4}, Device::CPU);
        const auto matrix = cpu_matrix.to(device);
        projection.rotation = {.9393727f, 0, -.3428978f, 0, 1, 0, .3428978f, 0, .9393727f};
        projection.translation = {1, -2, .5f};
        projection.ortho_scale = 13.f;
        for (auto model : {PointProjectionModel::Pinhole, PointProjectionModel::Orthographic, PointProjectionModel::Equirectangular}) {
            projection.model = model;
            for (bool transformed : {false, true}) {
                SCOPED_TRACE(testing::Message() << "model=" << int(model) << " transformed=" << transformed);
                const auto expected = project_points(cpu_points, projection, transformed ? &cpu_matrix : nullptr).to_vector();
                const auto actual = project_points(positions, projection, transformed ? &matrix : nullptr).to_vector();
                ASSERT_EQ(actual.size(), expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    if (expected[i] == projection.invalid_value)
                        EXPECT_EQ(actual[i], expected[i]);
                    else
                        EXPECT_NEAR(actual[i], expected[i], std::max(.002f, std::abs(expected[i]) * 2.e-6f)) << i;
                }
                PointFilterWindow window{.projection = projection, .near_depth = .25f, .far_depth = 6.f, .scale_x = .65f, .scale_y = .75f, .offset_x = .2f, .offset_y = -.15f};
                auto cpu_mask = Tensor::full_bool({count}, true, Device::CPU);
                auto gpu_mask = cpu_mask.to(device);
                filter_points(cpu_mask, &cpu_points, {.transforms = transformed ? &cpu_matrix : nullptr, .window = &window});
                filter_points(gpu_mask, &positions, {.transforms = transformed ? &matrix : nullptr, .window = &window});
                EXPECT_EQ(gpu_mask.to_vector_bool(), cpu_mask.to_vector_bool());
            }
        }
    }

    INSTANTIATE_TEST_SUITE_P(StorageBackends, TensorPointOps,
                             testing::Values(Storage::CPU, Storage::CUDA, Storage::Vulkan),
                             [](const testing::TestParamInfo<Storage>& info) {
                                 return info.param == Storage::CPU ? "Cpu" : info.param == Storage::CUDA ? "Cuda"
                                                                                                         : "Vulkan";
                             });
} // namespace
