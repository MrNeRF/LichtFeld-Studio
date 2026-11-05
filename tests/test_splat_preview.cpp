/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {
    using namespace lfs::core;
    enum class Storage { Float32,
                         Float16,
                         Quantized16 };

    void check_preview(Device device, int degree, Storage format) {
        constexpr size_t n = 300;
        const auto rest = sh_rest_coefficients_for_degree(degree);
        const bool quantized = format == Storage::Quantized16;
        const size_t cells = quantized ? sh_swizzled_padded_n(n) * rest * 3u : sh_swizzled_float_count(n, rest);
        auto storage = Tensor::empty_pageable_host({cells}, format == Storage::Float32 ? DataType::Float32 : DataType::Float16);
        auto bounds = Tensor::from_vector(std::vector<float>{-3, -1, 1, 3}, {size_t{4}}, Device::CPU);
        for (size_t i = 0; i < cells; ++i) {
            if (quantized)
                static_cast<uint16_t*>(storage.data_ptr())[i] = static_cast<uint16_t>((i * 17u) % 65536u);
            else if (format == Storage::Float16)
                storage.ptr<__half>()[i] = __float2half(float(i % 101) / 101.f - .5f);
            else
                storage.ptr<float>()[i] = float(i % 101) / 101.f - .5f;
        }
        SplatData model(degree, Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{1}, size_t{3}}, device),
                        storage.to(device), Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{4}}, device),
                        Tensor::zeros({n, size_t{1}}, device), 1.f, SplatData::ShNLayout::Swizzled);
        if (quantized)
            model.set_active_sh_degree(degree, bounds.to(device));
        const std::vector<int> requested{299, 0, 31, 32, 255, 256, 257, 31};
        const auto rows = Tensor::from_vector(requested, {requested.size()}, Device::CPU);
        const auto before = model.shN_raw().to_pageable_host().clone();
        for (int selected_degree = 0; selected_degree <= degree; ++selected_degree) {
            SCOPED_TRACE(selected_degree);
            auto output = model.shN_canonical_cpu_rows(rows, selected_degree);
            const auto selected_rest = sh_rest_coefficients_for_degree(selected_degree);
            ASSERT_EQ(output.device(), Device::CPU);
            ASSERT_EQ(output.dtype(), DataType::Float32);
            ASSERT_EQ(output.shape(), TensorShape({requested.size(), selected_rest, size_t{3}}));
            for (size_t i = 0; i < requested.size(); ++i) {
                const auto row = static_cast<size_t>(requested[i]);
                for (size_t c = 0; c < selected_rest * 3u; ++c) {
                    float expected;
                    if (quantized) {
                        const size_t offset = (row / 32u) * rest * 3u * 32u + c * 32u + row % 32u;
                        const auto q = static_cast<const uint16_t*>(storage.data_ptr())[offset];
                        const auto* b = bounds.ptr<float>();
                        const volatile float range = b[row / 256u * 2u + 1u] - b[row / 256u * 2u];
                        const volatile float normalized = static_cast<float>(q) * (1.f / 65535.f);
                        expected = std::fma(static_cast<float>(range), static_cast<float>(normalized), b[row / 256u * 2u]);
                    } else {
                        const size_t slots = (rest * 3u + 3u) / 4u;
                        const size_t offset = ((row / 32u) * slots * 32u + (c / 4u) * 32u + row % 32u) * 4u + c % 4u;
                        expected = format == Storage::Float16 ? __half2float(storage.ptr<__half>()[offset]) : storage.ptr<float>()[offset];
                    }
                    EXPECT_EQ(output.ptr<float>()[i * selected_rest * 3u + c], expected);
                }
            }
            const auto int64_output = model.shN_canonical_cpu_rows(rows.to(DataType::Int64), selected_degree);
            ASSERT_EQ(int64_output.bytes(), output.bytes());
            if (output.bytes()) {
                EXPECT_EQ(std::memcmp(output.data_ptr(), int64_output.data_ptr(), output.bytes()), 0);
                output.ptr<float>()[0] = 123.f; // Returned storage must not alias the model.
            }
        }
        const auto after = model.shN_raw().to_pageable_host();
        EXPECT_EQ(std::memcmp(before.data_ptr(), after.data_ptr(), before.bytes()), 0);
        EXPECT_EQ(model.shN_canonical_cpu_rows(Tensor::empty_pageable_host({size_t{0}}, DataType::Int64), degree).numel(), 0);
        EXPECT_THROW(model.shN_canonical_cpu_rows(rows, -1), std::invalid_argument);
        EXPECT_THROW(model.shN_canonical_cpu_rows(rows, degree + 1), std::invalid_argument);
        EXPECT_THROW(model.shN_canonical_cpu_rows(rows.to(DataType::Float32), degree), std::invalid_argument);
        EXPECT_THROW(model.shN_canonical_cpu_rows(rows.reshape({size_t{2}, size_t{4}}), degree), std::invalid_argument);
        EXPECT_THROW(model.shN_canonical_cpu_rows(Tensor{}, degree), std::invalid_argument);
        for (int invalid : {-1, 300}) {
            const auto invalid_rows = Tensor::from_vector(std::vector<int>{invalid}, {size_t{1}}, Device::CPU);
            EXPECT_THROW(model.shN_canonical_cpu_rows(invalid_rows, degree), std::out_of_range);
        }
        if (device == Device::CUDA)
            EXPECT_THROW(model.shN_canonical_cpu_rows(rows.to(Device::CUDA), degree), std::invalid_argument);
    }

    void check_subnormal_bounds(Device device) {
        constexpr size_t n = 300;
        constexpr int degree = 3;
        const auto rest = sh_rest_coefficients_for_degree(degree);
        const auto tiny = std::numeric_limits<float>::denorm_min();
        const auto normal = std::numeric_limits<float>::min();
        const std::array<std::pair<float, float>, 6> ranges{{{tiny, tiny * 256}, {0, normal}, {tiny, normal}, {normal, std::nextafter(normal, 1.f)}, {tiny, tiny}, {-normal, normal}}};
        const std::array<uint16_t, 6> codes{0, 1, 32767, 32768, 65534, 65535};
        const size_t cells = sh_swizzled_padded_n(n) * rest * 3u;
        auto storage = Tensor::empty_pageable_host({cells}, DataType::Float16);
        for (size_t i = 0; i < cells; ++i)
            static_cast<uint16_t*>(storage.data_ptr())[i] = codes[i % codes.size()];
        const std::vector<int> requested{299, 0, 31, 32, 255, 256, 257, 31};
        const auto rows = Tensor::from_vector(requested, {requested.size()}, Device::CPU);
        for (const auto& [lo, hi] : ranges) {
            SCOPED_TRACE(std::bit_cast<uint32_t>(lo));
            SCOPED_TRACE(std::bit_cast<uint32_t>(hi));
            auto bounds = Tensor::from_vector(std::vector<float>{lo, hi, -hi, -lo}, {size_t{4}}, Device::CPU);
            SplatData model(degree, Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{1}, size_t{3}}, device),
                            storage.to(device), Tensor::zeros({n, size_t{3}}, device), Tensor::zeros({n, size_t{4}}, device),
                            Tensor::zeros({n, size_t{1}}, device), 1.f, SplatData::ShNLayout::Swizzled);
            model.set_active_sh_degree(degree, bounds.to(device));
            const auto canonical = model.shN_canonical_cpu();
            // Code zero must preserve even the smallest nonzero lower bound.
            ASSERT_EQ(std::bit_cast<uint32_t>(canonical.ptr<float>()[0]), std::bit_cast<uint32_t>(lo));
            for (int selected_degree = 1; selected_degree <= degree; ++selected_degree) {
                const auto selected_cells = sh_rest_coefficients_for_degree(selected_degree) * 3u;
                const auto output = model.shN_canonical_cpu_rows(rows, selected_degree);
                for (size_t i = 0; i < requested.size(); ++i) {
                    for (size_t cell = 0; cell < selected_cells; ++cell) {
                        EXPECT_EQ(std::bit_cast<uint32_t>(output.ptr<float>()[i * selected_cells + cell]),
                                  std::bit_cast<uint32_t>(canonical.ptr<float>()[requested[i] * rest * 3u + cell]));
                    }
                }
            }
        }
    }

    TEST(SplatPreview, QuantizedSubnormalBoundsMatchCanonicalCpu) {
        check_subnormal_bounds(Device::CPU);
    }

    TEST(SplatPreviewGpuEdgeCases, QuantizedSubnormalBoundsMatchCanonical) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "CUDA GPU required";
        check_subnormal_bounds(Device::CUDA);
    }

    using Profile = std::tuple<int, Storage>;
    class SplatPreviewCpu : public testing::TestWithParam<Profile> {};
    class SplatPreviewGpu : public testing::TestWithParam<Profile> {};

    TEST(SplatPreview, DegreeZeroModelHasNoRestStorage) {
        constexpr size_t n = 2;
        SplatData model(0, Tensor::zeros({n, size_t{3}}, Device::CPU), Tensor::zeros({n, size_t{1}, size_t{3}}, Device::CPU),
                        Tensor::empty_pageable_host({size_t{0}}, DataType::Float32), Tensor::zeros({n, size_t{3}}, Device::CPU),
                        Tensor::zeros({n, size_t{4}}, Device::CPU), Tensor::zeros({n, size_t{1}}, Device::CPU),
                        1.f, SplatData::ShNLayout::Swizzled);
        const auto rows = Tensor::from_vector(std::vector<int>{1, 0, 1}, {size_t{3}}, Device::CPU);
        const auto result = model.shN_canonical_cpu_rows(rows, 0);
        EXPECT_EQ(result.shape(), TensorShape({size_t{3}, size_t{0}, size_t{3}}));
        EXPECT_EQ(result.device(), Device::CPU);
        EXPECT_EQ(result.dtype(), DataType::Float32);
    }

    TEST_P(SplatPreviewCpu, SelectedRowsBandsAndOwnedStorage) {
        const auto [degree, format] = GetParam();
        check_preview(Device::CPU, degree, format);
    }

    TEST_P(SplatPreviewGpu, SelectedRowsBandsAndOwnedStorage) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            GTEST_SKIP() << "CUDA GPU required";
        const auto [degree, format] = GetParam();
        check_preview(Device::CUDA, degree, format);
    }

    INSTANTIATE_TEST_SUITE_P(StorageLayouts, SplatPreviewCpu,
                             testing::Combine(testing::Values(1, 2, 3), testing::Values(Storage::Float32, Storage::Float16, Storage::Quantized16)));
    INSTANTIATE_TEST_SUITE_P(StorageLayouts, SplatPreviewGpu,
                             testing::Combine(testing::Values(1, 2, 3), testing::Values(Storage::Float32, Storage::Float16, Storage::Quantized16)));
} // namespace
