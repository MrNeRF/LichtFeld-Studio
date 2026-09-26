/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_sh.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <random>

namespace {
    using namespace lfs::core;
    enum class Backend { CPU,
                         CUDA,
                         Vulkan };
    class ShCodecOps : public testing::TestWithParam<Backend> {
    protected:
        std::optional<GpuBackendScope> scope;
        Device device = Device::CPU;
        void SetUp() override {
            if (GetParam() == Backend::CPU)
                return;
            const auto backend = GetParam() == Backend::CUDA ? GpuBackend::CUDA : GpuBackend::Vulkan;
            if (!gpu_backend_available(backend))
                GTEST_SKIP();
            scope.emplace(backend);
            device = Device::GPU;
        }
        Tensor upload(const Tensor& t) { return t.to(device); }
        static Tensor canonical(size_t n, uint32_t rest) {
            std::mt19937 rng(13);
            std::uniform_real_distribution<float> distribution(-1.75f, 1.25f);
            std::vector<float> values(n * rest * 3);
            for (auto& v : values)
                v = distribution(rng);
            return Tensor::from_vector(values, {n, size_t(rest), 3}, Device::CPU);
        }
        static void equal(const Tensor& actual, const Tensor& expected) {
            auto a = actual.cpu().contiguous(), b = expected.cpu().contiguous();
            ASSERT_EQ(a.bytes(), b.bytes());
            ASSERT_EQ(a.dtype(), b.dtype());
            if (a.bytes() && std::memcmp(a.data_ptr(), b.data_ptr(), a.bytes())) {
                const auto* x = static_cast<const unsigned char*>(a.data_ptr());
                const auto* y = static_cast<const unsigned char*>(b.data_ptr());
                size_t i = 0;
                while (i < a.bytes() && x[i] == y[i])
                    ++i;
                ADD_FAILURE() << "first differing byte " << i << ": " << int(x[i]) << " vs " << int(y[i]);
            }
        }
        static void equal_quantized(const Tensor& actual, const Tensor& expected) {
            auto a = actual.cpu().contiguous(), b = expected.cpu().contiguous();
            ASSERT_EQ(a.bytes(), b.bytes());
            const auto* x = static_cast<const uint16_t*>(a.data_ptr());
            const auto* y = static_cast<const uint16_t*>(b.data_ptr());
            for (size_t i = 0; i < a.numel(); ++i)
                ASSERT_LE(std::abs(int(x[i]) - int(y[i])), 1) << "code " << i;
        }
        static Tensor storage(size_t n, uint32_t rest, ShFormat format, Device device) {
            size_t count = format == ShFormat::Q16 ? sh_value_quant::sh_value_u16_count(n, rest) : sh_swizzled_float_count(n, rest);
            return Tensor::zeros({count}, device, format == ShFormat::Float32 ? DataType::Float32 : DataType::Float16);
        }
    };
    TEST_P(ShCodecOps, LayoutAndQ16StayWithinCpuRoundingAtBoundaries) {
        for (uint32_t rest : {3u, 8u, 15u})
            for (size_t n : {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, 65535u, 65536u, 65537u, 524289u}) {
                SCOPED_TRACE(testing::Message() << "rest=" << rest << " rows=" << n);
                auto host = canonical(n, rest), input = upload(host);
                for (auto format : {ShFormat::Float32, ShFormat::Float16, ShFormat::Q16}) {
                    SCOPED_TRACE(int(format));
                    auto expected = storage(n, rest, format, Device::CPU), actual = storage(n, rest, format, device);
                    auto eb = Tensor::zeros({sh_value_quant::n_bounds_for_prims(n) * 2}, Device::CPU);
                    auto ab = upload(eb);
                    ShCodec pack{.source_format = ShFormat::Canonical, .destination_format = format, .source_rows = n, .destination_rows = n, .count = n, .source_rest = rest, .destination_rest = rest};
                    sh_codec(host, expected, pack, nullptr, nullptr, format == ShFormat::Q16 ? &eb : nullptr);
                    sh_codec(input, actual, pack, nullptr, nullptr, format == ShFormat::Q16 ? &ab : nullptr);
                    if (format == ShFormat::Q16)
                        equal_quantized(actual, expected);
                    else
                        equal(actual, expected);
                    if (format == ShFormat::Q16)
                        equal(ab, eb);
                    auto decoded = Tensor::empty({n, size_t(rest), 3}, device), ref = Tensor::empty({n, size_t(rest), 3}, Device::CPU);
                    ShCodec unpack{.source_format = format, .destination_format = ShFormat::Canonical, .source_rows = n, .destination_rows = n, .count = n, .source_rest = rest, .destination_rest = rest};
                    auto encoded = format == ShFormat::Q16 ? actual.cpu() : expected;
                    sh_codec(encoded, ref, unpack, nullptr, format == ShFormat::Q16 ? &eb : nullptr);
                    sh_codec(actual, decoded, unpack, nullptr, format == ShFormat::Q16 ? &ab : nullptr);
                    equal(decoded, ref);
                    if (format == ShFormat::Float32)
                        equal(decoded, host);
                }
            }
    }
    TEST_P(ShCodecOps, GatherAndCopyAcrossLayoutsAndBlocks) {
        constexpr size_t n = 513, m = 257;
        for (uint32_t rest : {3u, 8u, 15u}) {
            SCOPED_TRACE(rest);
            auto host = canonical(n, rest);
            std::vector<int> rows(m);
            for (size_t i = 0; i < m; ++i)
                rows[i] = (i * 127) % n;
            auto hi = Tensor::from_vector(rows, {m}, Device::CPU).to(DataType::Int64), indices = upload(hi);
            for (auto format : {ShFormat::Float32, ShFormat::Float16, ShFormat::Q16}) {
                SCOPED_TRACE(int(format));
                auto source = storage(n, rest, format, Device::CPU), bounds = Tensor::zeros({6}, Device::CPU);
                sh_codec(host, source, {.source_format = ShFormat::Canonical, .destination_format = format, .source_rows = n, .destination_rows = n, .count = n, .source_rest = rest, .destination_rest = rest},
                         nullptr, nullptr, format == ShFormat::Q16 ? &bounds : nullptr);
                auto gpu = upload(source), gb = upload(bounds);
                for (auto output : {ShFormat::Canonical, ShFormat::Float32, ShFormat::Q16}) {
                    SCOPED_TRACE(int(output));
                    auto ref = output == ShFormat::Canonical ? Tensor::zeros({m, size_t(rest), 3}, Device::CPU) : storage(m, rest, output, Device::CPU);
                    auto out = upload(ref), rb = Tensor::zeros({4}, Device::CPU), ob = upload(rb);
                    const ShCodec p{.source_format = format, .destination_format = output, .source_rows = n, .destination_rows = m, .count = m, .source_rest = rest, .destination_rest = rest};
                    sh_codec(source, ref, p, &hi, format == ShFormat::Q16 ? &bounds : nullptr, output == ShFormat::Q16 ? &rb : nullptr);
                    sh_codec(gpu, out, p, &indices, format == ShFormat::Q16 ? &gb : nullptr, output == ShFormat::Q16 ? &ob : nullptr);
                    if (output == ShFormat::Q16)
                        equal_quantized(out, ref);
                    else
                        equal(out, ref);
                    if (output == ShFormat::Q16)
                        equal(ob, rb);
                }
                auto ref = storage(600, 15, ShFormat::Float32, Device::CPU), out = upload(ref);
                ref.fill_(17);
                out.fill_(17);
                const ShCodec copy{.source_format = format, .source_rows = n, .destination_rows = 600, .count = m, .source_rest = rest, .destination_rest = 15, .source_offset = 17, .destination_offset = 31};
                sh_codec(source, ref, copy, nullptr, format == ShFormat::Q16 ? &bounds : nullptr);
                sh_codec(gpu, out, copy, nullptr, format == ShFormat::Q16 ? &gb : nullptr);
                equal(out, ref);
            }
        }
    }
    TEST_P(ShCodecOps, ScatterAndAliasedRowCopyPreserveUntouchedRows) {
        constexpr size_t n = 257, m = 33;
        constexpr uint32_t rest = 15;
        auto rows = canonical(m, rest), src = upload(rows);
        std::vector<int> ids(m);
        for (size_t i = 0; i < m; ++i)
            ids[i] = (i * 7) % n;
        auto hi = Tensor::from_vector(ids, {m}, Device::CPU), gi = upload(hi);
        auto ref = storage(n, rest, ShFormat::Float32, Device::CPU), out = upload(ref);
        const ShCodec scatter{.source_format = ShFormat::Canonical, .source_rows = m, .destination_rows = n, .count = m, .source_rest = rest, .destination_rest = rest, .scatter = true};
        sh_codec(rows, ref, scatter, &hi);
        sh_codec(src, out, scatter, &gi);
        equal(out, ref);
        const ShCodec copy{.source_rows = n, .destination_rows = n, .count = 200, .source_rest = rest, .destination_rest = rest, .source_offset = 1, .destination_offset = 31};
        sh_codec(ref, ref, copy);
        sh_codec(out, out, copy);
        equal(out, ref);
    }
    TEST_P(ShCodecOps, OffsetStorageAndHalfSourceQuantization) {
        constexpr size_t n = 257;
        constexpr uint32_t rest = 8;
        auto host = canonical(n, rest);
        auto source = storage(n, rest, ShFormat::Float16, Device::CPU);
        sh_codec(host, source, {.source_format = ShFormat::Canonical, .destination_format = ShFormat::Float16, .source_rows = n, .destination_rows = n, .count = n, .source_rest = rest, .destination_rest = rest});
        auto allocation = Tensor::zeros({source.numel() + 16}, device, DataType::Float16);
        auto view = allocation.slice(0, 8, 8 + source.numel());
        view.copy_from(upload(source));
        auto ref = storage(n, rest, ShFormat::Q16, Device::CPU), out = storage(n, rest, ShFormat::Q16, device);
        auto rb = Tensor::zeros({4}, Device::CPU), ob = upload(rb);
        const ShCodec p{.source_format = ShFormat::Float16, .destination_format = ShFormat::Q16, .source_rows = n, .destination_rows = n, .count = n, .source_rest = rest, .destination_rest = rest};
        sh_codec(source, ref, p, nullptr, nullptr, &rb);
        sh_codec(view, out, p, nullptr, nullptr, &ob);
        equal(out, ref);
        equal(ob, rb);
    }
    INSTANTIATE_TEST_SUITE_P(CpuCudaVulkan, ShCodecOps, testing::Values(Backend::CPU, Backend::CUDA, Backend::Vulkan),
                             [](const testing::TestParamInfo<Backend>& p) { return p.param == Backend::CPU ? "CPU" : p.param == Backend::CUDA ? "CUDA"
                                                                                                                                              : "Vulkan"; });
} // namespace
