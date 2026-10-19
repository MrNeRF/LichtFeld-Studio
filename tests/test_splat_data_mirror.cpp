/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/splat_data_mirror.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

using namespace lfs::core;

namespace {
    SplatData mirror_model(const std::vector<float>& xyz) {
        const size_t n = xyz.size() / 3;
        std::vector<float> rotations(n * 4, 0.0f);
        for (size_t i = 0; i < n; ++i)
            rotations[i * 4] = 1.0f;
        return SplatData(0,
                         Tensor::from_vector(xyz, {n, size_t{3}}, Device::CUDA),
                         Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CUDA),
                         Tensor::zeros({n, size_t{0}, size_t{3}}, Device::CUDA),
                         Tensor::zeros({n, size_t{3}}, Device::CUDA),
                         Tensor::from_vector(rotations, {n, size_t{4}}, Device::CUDA),
                         Tensor::zeros({n, size_t{1}}, Device::CUDA), 1.0f);
    }
} // namespace

TEST(MirrorPrecisionTest, RepeatedPairsStayWithinTwoCoordinateScaleUlps) {
    std::mt19937 random(314159);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    for (const size_t count : {size_t{1009}, size_t{100003}}) {
        std::vector<float> original(count * 3);
        for (size_t i = 0; i < original.size(); ++i)
            original[i] = value(random) + 0.731f * static_cast<float>(i % 3 + 1);
        const float extent = std::abs(*std::max_element(original.begin(), original.end(),
                                                        [](float a, float b) { return std::abs(a) < std::abs(b); }));
        const float ulp = std::nextafter(extent, std::numeric_limits<float>::infinity()) - extent;
        for (const bool partial : {false, true}) {
            std::vector<bool> selected(count);
            for (size_t i = 0; i < count; ++i)
                selected[i] = !partial || i % 3 != 0;
            const auto mask = Tensor::from_vector(selected, {count}, Device::CUDA);
            for (int axis = 0; axis < 3; ++axis) {
                SCOPED_TRACE(::testing::Message() << count << " partial=" << partial << " axis=" << axis);
                auto model = mirror_model(original);
                double maximum = 0.0;
                for (int pair = 0; pair < 128; ++pair) {
                    for (int step = 0; step < 2; ++step)
                        mirror_gaussians(model, mask, static_cast<MirrorAxis>(axis), compute_selection_center(model, mask));
                    const auto cpu = model.means().cpu().contiguous();
                    const auto* actual = cpu.ptr<float>();
                    for (size_t i = 0; i < original.size(); ++i) {
                        maximum = std::max(maximum, std::abs(static_cast<double>(actual[i]) - original[i]));
                        if (!selected[i / 3] || i % 3 != static_cast<size_t>(axis))
                            ASSERT_EQ(std::memcmp(&actual[i], &original[i], sizeof(float)), 0);
                    }
                    if ((pair & (pair + 1)) == 0)
                        std::cout << "mirror precision n=" << count << " partial=" << partial << " axis=" << axis
                                  << " pairs=" << pair + 1 << " error=" << maximum << " ulps=" << maximum / ulp << '\n';
                }
                EXPECT_LE(maximum, 2.0 * ulp);
            }
        }
    }
}

TEST(MirrorPrecisionTest, FixedPivotRoundingCannotRecoverLostBits) {
    const std::vector<float> original{-0x1p-25f, 0.0f, 0.0f};
    auto model = mirror_model(original);
    const auto mask = Tensor::from_vector(std::vector<bool>{true}, {size_t{1}}, Device::CUDA);
    mirror_gaussians(model, mask, MirrorAxis::X, {1, 0, 0});
    EXPECT_EQ(model.means().cpu().ptr<float>()[0], 2.0f);
    mirror_gaussians(model, mask, MirrorAxis::X, {1, 0, 0});
    EXPECT_EQ(model.means().cpu().ptr<float>()[0], 0.0f);
    EXPECT_NE(model.means().cpu().ptr<float>()[0], original[0]);
}

TEST(MirrorPrecisionTest, SelectedCentroidMatchesDoubleReference) {
    constexpr size_t count = 100003;
    std::mt19937 random(12345);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    std::vector<float> xyz(count * 3);
    std::vector<bool> selected(count);
    double sum[3]{};
    size_t selected_count = 0;
    for (size_t i = 0; i < count; ++i) {
        selected[i] = i % 3 != 0;
        selected_count += selected[i];
        for (int axis = 0; axis < 3; ++axis) {
            xyz[i * 3 + axis] = value(random) + 0.731f * (axis + 1);
            if (selected[i])
                sum[axis] += xyz[i * 3 + axis];
        }
    }
    auto model = mirror_model(xyz);
    for (const auto device : {Device::CPU, Device::CUDA}) {
        model.means() = model.means().to(device);
        const auto mask = Tensor::from_vector(selected, {count}, device);
        const auto center = compute_selection_center(model, mask);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_EQ(center[axis], static_cast<float>(sum[axis] / selected_count));
    }
}

TEST(MirrorPrecisionTest, StableCentroidDoesNotRegressReferenceCost) {
    constexpr size_t count = 100003;
    std::vector<float> xyz(count * 3);
    for (size_t i = 0; i < xyz.size(); ++i)
        xyz[i] = static_cast<float>(i % 137) * 0.013f + 0.731f;
    const auto model = mirror_model(xyz);
    const auto mask = Tensor::ones({count}, Device::CUDA, DataType::UInt8);
    const auto reference = [&] {
        const auto selected = mask.ne(0);
        const int n = selected.sum_scalar();
        const auto masked = model.means() * selected.to(DataType::Float32).unsqueeze(1);
        const auto sum = masked.sum({0}, false).cpu().contiguous();
        const auto* p = sum.ptr<float>();
        return glm::vec3(p[0], p[1], p[2]) * (1.0f / static_cast<float>(n));
    };
    std::vector<double> old_times, new_times;
    for (int sample = 0; sample < 40; ++sample) {
        const auto measure = [&](bool old) {
            const auto start = std::chrono::steady_clock::now();
            const auto result = old ? reference() : compute_selection_center(model, mask);
            const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            EXPECT_TRUE(std::isfinite(result.x));
            if (sample >= 8)
                (old ? old_times : new_times).push_back(elapsed);
        };
        measure(sample % 2 == 0);
        measure(sample % 2 != 0);
    }
    std::sort(old_times.begin(), old_times.end());
    std::sort(new_times.begin(), new_times.end());
    const double old_median = old_times[old_times.size() / 2];
    const double new_median = new_times[new_times.size() / 2];
    std::cout << "centroid median old_us=" << old_median << " new_us=" << new_median << '\n';
    EXPECT_LT(new_median, old_median * 1.5);
}

TEST(MirrorPrecisionTest, EmptyAndExactlyRepresentableCentersKeepTheirBehavior) {
    const std::vector<float> xyz{-1, 2, 4, 1, 4, 8, 3, 6, 12};
    for (const auto device : {Device::CPU, Device::CUDA}) {
        auto model = mirror_model(xyz);
        model.means() = model.means().to(device);
        const auto mask = Tensor::from_vector(std::vector<bool>{true, true, true}, {size_t{3}}, device);
        EXPECT_EQ(compute_selection_center(model, mask), glm::vec3(1, 4, 8));
        const auto empty = Tensor::zeros({size_t{3}}, device, DataType::UInt8);
        EXPECT_EQ(compute_selection_center(model, empty), glm::vec3(0));
    }
    for (int axis = 0; axis < 3; ++axis) {
        auto model = mirror_model(xyz);
        const auto mask = Tensor::from_vector(std::vector<bool>{true, true, true}, {size_t{3}}, Device::CUDA);
        const auto original_rotation = model.rotation_raw().cpu();
        for (int pair = 0; pair < 16; ++pair) {
            for (int step = 0; step < 2; ++step)
                mirror_gaussians(model, mask, static_cast<MirrorAxis>(axis), compute_selection_center(model, mask));
            const auto actual = model.means().cpu();
            EXPECT_EQ(std::memcmp(actual.ptr<float>(), xyz.data(), xyz.size() * sizeof(float)), 0);
            const auto rotation = model.rotation_raw().cpu();
            EXPECT_EQ(std::memcmp(rotation.data_ptr(), original_rotation.data_ptr(), rotation.bytes()), 0);
        }
    }
}

TEST(MirrorPrecisionTest, CentroidOrdersIndependentInputStreams) {
    cudaStream_t positions_stream{}, mask_stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&positions_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&mask_stream, cudaStreamNonBlocking), cudaSuccess);
    {
        const CUDAStreamGuard positions_guard(positions_stream);
        auto model = mirror_model({1, 2, 3, 3, 4, 5});
        auto mask = [&] {
            const CUDAStreamGuard mask_guard(mask_stream);
            return Tensor::ones({size_t{2}}, Device::CUDA, DataType::UInt8);
        }();
        EXPECT_EQ(compute_selection_center(model, mask), glm::vec3(2, 3, 4));
    }
    CudaMemoryPool::instance().release_stream(mask_stream);
    CudaMemoryPool::instance().release_stream(positions_stream);
    EXPECT_EQ(cudaStreamDestroy(mask_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(positions_stream), cudaSuccess);
}

TEST(MirrorPrecisionTest, CentroidRejectsMismatchedSelectionSize) {
    const auto model = mirror_model({1, 2, 3, 3, 4, 5});
    const auto mask = Tensor::ones({size_t{3}}, Device::CUDA, DataType::UInt8);
    EXPECT_THROW((void)compute_selection_center(model, mask), std::exception);
}
