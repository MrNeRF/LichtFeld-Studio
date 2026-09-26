/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

namespace {
    using namespace lfs::core;

    static_assert(kShReorderSize == 32u);
    static_assert(sh_float4_slots_for_rest(15) == 12u);
    static_assert(sh_swizzled_byte_count(33, 15) == 2u * 12u * 32u * 4u * sizeof(float));

    void expect_equal(const Tensor& expected, const Tensor& actual) {
        const Tensor a = expected.cpu().contiguous();
        const Tensor b = actual.cpu().contiguous();
        ASSERT_EQ(a.shape(), b.shape());
        ASSERT_EQ(a.dtype(), b.dtype());
        EXPECT_EQ(std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes()), 0);
    }

    void round_trip(GpuBackend backend) {
        if (!gpu_backend_available(backend))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(backend);
        for (const int degree : {1, 2, 3}) {
            const size_t rest = sh_rest_coefficients_for_degree(degree);
            for (const size_t n : {size_t{31}, size_t{32}, size_t{33}, size_t{65537}}) {
                SCOPED_TRACE(::testing::Message() << "degree=" << degree << " n=" << n);
                std::vector<float> values(n * rest * 3);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = static_cast<float>(static_cast<int>(i % 1024) - 512) / 511.0f;
                const Tensor canonical = Tensor::from_vector(values, {n, rest, size_t{3}}, Device::CPU);
                std::vector<float> packed_values(sh_swizzled_float_count(n, rest), 0.0f);
                for (size_t p = 0; p < n; ++p)
                    for (size_t c = 0; c < rest * 3; ++c)
                        packed_values[sh_swizzled_index(p, c / 4, rest) * 4 + c % 4] = values[p * rest * 3 + c];
                const Tensor packed = Tensor::from_vector(packed_values, {packed_values.size()}, Device::CPU);
                SplatData model(degree,
                                Tensor::zeros({n, size_t{3}}, Device::GPU),
                                Tensor::zeros({n, size_t{1}, size_t{3}}, Device::GPU),
                                canonical.gpu(),
                                Tensor::zeros({n, size_t{3}}, Device::GPU),
                                Tensor::zeros({n, size_t{4}}, Device::GPU),
                                Tensor::zeros({n, size_t{1}}, Device::GPU), 1.0f);
                expect_equal(packed, model.shN_raw());
                expect_equal(canonical, model.shN_canonical());
                for (const auto dtype : {DataType::Float32, DataType::Float16}) {
                    SCOPED_TRACE(static_cast<int>(dtype));
                    model.shN_raw() = packed.gpu().to(dtype);
                    model.shN_raw().reserve(model.shN_raw().numel());
                    // Exercise CPU input, then backend-resident input. Padding
                    // and inactive rows must stay zero even on buffer reuse.
                    model.shN_raw().copy_from(Tensor::ones(model.shN_raw().shape(), Device::GPU).to(dtype));
                    model.shN_set_from_canonical(canonical);
                    expect_equal(packed.to(dtype), model.shN_raw());
                    expect_equal(canonical.to(dtype).to(DataType::Float32), model.shN_canonical());
                    model.shN_raw().copy_from(Tensor::ones(model.shN_raw().shape(), Device::GPU).to(dtype));
                    model.shN_set_from_canonical(canonical.gpu());
                    expect_equal(packed.to(dtype), model.shN_raw());
                }
                if (rest > 3) {
                    const Tensor prefix = canonical.slice(1, 0, 3).contiguous();
                    Tensor expected = Tensor::zeros_like(canonical);
                    expected.slice(1, 0, 3).copy_from(prefix);
                    model.shN_raw() = packed.gpu();
                    model.shN_raw().reserve(model.shN_raw().numel());
                    model.shN_set_from_canonical(prefix.gpu());
                    expect_equal(expected, model.shN_canonical());
                }
            }
        }
    }

    void extraction_round_trip(GpuBackend backend) {
        if (!gpu_backend_available(backend))
            GTEST_SKIP() << "Backend unavailable";
        const GpuBackendScope scope(backend);
        struct ResetQuantization {
            ~ResetQuantization() { sh_value_quant::set_enabled_for_testing(std::nullopt); }
        } reset;
        for (const int degree : {1, 2, 3}) {
            const size_t n = degree == 3 ? 65571 : 513;
            const size_t rest = sh_rest_coefficients_for_degree(degree);
            std::vector<float> values(n * rest * 3);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = static_cast<float>(static_cast<int>((i * 17 + i / 256) % 1024) - 512) / 511.0f;
            const Tensor canonical = Tensor::from_vector(values, {n, rest, size_t{3}}, Device::CPU);
            for (const int encoding : {0, 1, 2}) {
                SCOPED_TRACE(::testing::Message() << "degree=" << degree << " encoding=" << encoding);
                SplatData model(degree, Tensor::zeros({n, size_t{3}}, Device::GPU),
                                Tensor::zeros({n, size_t{1}, size_t{3}}, Device::GPU), canonical.gpu(),
                                Tensor::zeros({n, size_t{3}}, Device::GPU), Tensor::zeros({n, size_t{4}}, Device::GPU),
                                Tensor::zeros({n, size_t{1}}, Device::GPU), 1.0f);
                if (encoding == 1) {
                    model.shN_raw() = model.shN_raw().to(DataType::Float16);
                    model.shN_raw().reserve(model.shN_raw().numel());
                } else if (encoding == 2) {
                    sh_value_quant::set_enabled_for_testing(true);
                    ASSERT_TRUE(model.apply_shN_value_quant());
                }
                model.set_active_sh_degree(1);
                sh_value_quant::set_enabled_for_testing(false);
                const Tensor reference = model.shN_canonical_cpu();
                for (const bool sparse : {true, false}) {
                    SCOPED_TRACE(sparse);
                    std::vector<int> mask(n, sparse ? 0 : 1);
                    for (const size_t row : {size_t{0}, size_t{30}, size_t{31}, size_t{32}, size_t{33},
                                             size_t{254}, size_t{255}, size_t{256}, size_t{257}, n - 2, n - 1})
                        mask[row] = sparse ? 1 : 0;
                    const Tensor keep = Tensor::from_vector(mask, {n}, Device::CPU).to(DataType::Bool);
                    const Tensor selected = keep.nonzero().squeeze(1);
                    const Tensor expected = reference.index_select(0, selected);
                    SplatData extracted = extract_by_mask(model, keep.gpu());
                    EXPECT_EQ(extracted.size(), selected.numel());
                    EXPECT_EQ(extracted.get_active_sh_degree(), 1);
                    EXPECT_EQ(extracted.get_max_sh_degree(), degree);
                    EXPECT_EQ(gpu_backend_of(extracted.shN_raw()), backend);
                    expect_equal(expected, extracted.shN_canonical_cpu());
                }
                expect_equal(reference, model.shN_canonical_cpu());
            }
        }
    }
} // namespace

TEST(SplatShTensorLayout, VulkanRoundTripAcrossBlocksAndBands) { round_trip(GpuBackend::Vulkan); }
TEST(SplatShTensorLayout, CudaRoundTripAcrossBlocksAndBands) { round_trip(GpuBackend::CUDA); }
TEST(SplatShTensorLayout, VulkanExtractionPreservesEveryColorStorageFormat) { extraction_round_trip(GpuBackend::Vulkan); }
TEST(SplatShTensorLayout, CudaExtractionPreservesEveryColorStorageFormat) { extraction_round_trip(GpuBackend::CUDA); }

TEST(SplatShTensorLayout, VulkanAliasedInputSurvivesRepacking) {
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan unavailable";
    const GpuBackendScope scope(GpuBackend::Vulkan);
    const size_t n = 64;
    const Tensor canonical = Tensor::arange(0, n * 8 * 3, 1).gpu().reshape(TensorShape({n, size_t{8}, size_t{3}}));
    SplatData model(2, Tensor::zeros({n, size_t{3}}, Device::GPU),
                    Tensor::zeros({n, size_t{1}, size_t{3}}, Device::GPU), canonical,
                    Tensor::zeros({n, size_t{3}}, Device::GPU), Tensor::zeros({n, size_t{4}}, Device::GPU),
                    Tensor::zeros({n, size_t{1}}, Device::GPU), 1.0f);
    const Tensor alias = model.shN_raw().reshape(canonical.shape());
    const Tensor expected = alias.cpu();
    model.shN_set_from_canonical(alias);
    expect_equal(expected, model.shN_canonical());
}

TEST(SplatShTensorLayout, VulkanReplacementKeepsModelBackend) {
    if (!gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan unavailable";
    const GpuBackendScope vulkan(GpuBackend::Vulkan);
    const size_t n = 33;
    const Tensor canonical = Tensor::full({n, size_t{3}, size_t{3}}, 0.25f, Device::GPU);
    SplatData model(1, Tensor::zeros({n, size_t{3}}, Device::GPU),
                    Tensor::zeros({n, size_t{1}, size_t{3}}, Device::GPU), canonical,
                    Tensor::zeros({n, size_t{3}}, Device::GPU), Tensor::zeros({n, size_t{4}}, Device::GPU),
                    Tensor::zeros({n, size_t{1}}, Device::GPU), 1.0f);
    model.shN_raw() = model.shN_raw().clone();
    const GpuBackendScope other_default(GpuBackend::CUDA);
    model.shN_set_from_canonical(canonical);
    EXPECT_EQ(gpu_backend_of(model.shN_raw()), GpuBackend::Vulkan);
    expect_equal(canonical, model.shN_canonical());
}
