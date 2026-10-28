/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace cuda_stub_test::precise {
    const void* division_stub();
    cudaError_t divide(const float*, const float*, float*, float*, size_t);
} // namespace cuda_stub_test::precise
namespace cuda_stub_test::fast {
    const void* division_stub();
    cudaError_t divide(const float*, const float*, float*, float*, size_t);
} // namespace cuda_stub_test::fast

TEST(CudaTemplateStub, DistinctTranslationUnitsHaveDistinctStubs) {
    EXPECT_NE(cuda_stub_test::precise::division_stub(), cuda_stub_test::fast::division_stub());
}

TEST(CudaTemplateStub, DivisionUsesTheCallingTranslationUnitsMathMode) {
    using namespace lfs::core;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device required";
    std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "data/bicycle/sparse/0/points3D.bin",
                       std::ios::binary);
    if (!file)
        GTEST_SKIP() << "data/bicycle sparse points3D not available";

    constexpr size_t sample_count = 1024;
    uint64_t point_count = 0;
    file.read(reinterpret_cast<char*>(&point_count), sizeof(point_count));
    ASSERT_GE(point_count, sample_count);
    std::vector<float> a, b;
    for (size_t i = 0; i < sample_count; ++i) {
        file.ignore(sizeof(uint64_t));
        double xyz[3];
        file.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
        file.ignore(3 + sizeof(double));
        uint64_t track_length = 0;
        file.read(reinterpret_cast<char*>(&track_length), sizeof(track_length));
        ASSERT_TRUE(file.good());
        file.ignore(static_cast<std::streamsize>(track_length * 2 * sizeof(uint32_t)));
        ASSERT_TRUE(file.good());
        for (size_t axis = 0; axis < 3; ++axis) {
            const float numerator = static_cast<float>(xyz[axis]);
            const float denominator = static_cast<float>(xyz[(axis + 1) % 3]);
            ASSERT_TRUE(std::isnormal(numerator));
            ASSERT_TRUE(std::isnormal(denominator));
            a.push_back(numerator);
            b.push_back(denominator);
        }
    }

    const auto numerator = Tensor::from_vector(a, {a.size()}, Device::CUDA);
    const auto denominator = Tensor::from_vector(b, {b.size()}, Device::CUDA);
    auto precise = Tensor::empty({a.size()}, Device::CUDA);
    auto fast = Tensor::empty({a.size()}, Device::CUDA);
    auto precise_reference = Tensor::empty({a.size()}, Device::CUDA);
    auto fast_reference = Tensor::empty({a.size()}, Device::CUDA);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cuda_stub_test::precise::divide(numerator.ptr<float>(), denominator.ptr<float>(),
                                              precise.ptr<float>(), precise_reference.ptr<float>(), a.size()),
              cudaSuccess);
    ASSERT_EQ(cuda_stub_test::fast::divide(numerator.ptr<float>(), denominator.ptr<float>(),
                                           fast.ptr<float>(), fast_reference.ptr<float>(), a.size()),
              cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto expected_precise = precise_reference.to_vector();
    const auto expected_fast = fast_reference.to_vector();
    ASSERT_NE(expected_precise, expected_fast) << "Fixture must distinguish precise and fast division";
    EXPECT_EQ(precise.to_vector(), expected_precise);
    EXPECT_EQ(fast.to_vector(), expected_fast);
}
