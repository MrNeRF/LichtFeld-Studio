/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    void require_cuda() {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
    }

    std::vector<float> random_swizzled(const size_t n,
                                       const std::uint32_t rest,
                                       const std::uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.75f, 1.25f);
        const size_t count = sh_swizzled_float_count(n, rest);
        std::vector<float> values(count);
        for (float& v : values) {
            v = dist(rng);
        }
        return values;
    }

    std::vector<float> filled_swizzled(const size_t n,
                                       const std::uint32_t rest,
                                       const float value) {
        return std::vector<float>(sh_swizzled_float_count(n, rest), value);
    }

    void expect_u16_equal(const Tensor& actual_codes, const Tensor& expected_codes) {
        const Tensor actual = actual_codes.contiguous().cpu();
        const Tensor expected = expected_codes.contiguous().cpu();
        ASSERT_EQ(actual.bytes(), expected.bytes());
        const auto* a = static_cast<const std::uint16_t*>(actual.data_ptr());
        const auto* e = static_cast<const std::uint16_t*>(expected.data_ptr());
        const size_t words = actual.bytes() / sizeof(std::uint16_t);
        size_t mismatches = 0;
        size_t first = words;
        for (size_t i = 0; i < words; ++i) {
            if (a[i] != e[i]) {
                if (mismatches < 8) {
                    EXPECT_EQ(a[i], e[i]) << "u16 word " << i;
                }
                if (first == words) {
                    first = i;
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "first mismatch at word " << first;
    }

    void expect_bounds_equal(const Tensor& actual_bounds, const Tensor& expected_bounds) {
        const Tensor actual = actual_bounds.contiguous().cpu();
        const Tensor expected = expected_bounds.contiguous().cpu();
        ASSERT_EQ(actual.numel(), expected.numel());
        const auto* a = actual.ptr<float>();
        const auto* e = expected.ptr<float>();
        for (size_t i = 0; i < actual.numel(); ++i) {
            EXPECT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(e[i]))
                << "bounds float " << i;
        }
    }

    struct EncodedPair {
        Tensor codes;
        Tensor bounds;
    };

    EncodedPair kernel_pair(const std::vector<float>& host,
                            const size_t n,
                            const std::uint32_t rest) {
        GpuBackendScope scope(GpuBackend::CUDA);
        Tensor src = Tensor::from_vector(host, TensorShape{host.size()}, Device::CPU).to(Device::GPU);
        const size_t n_cells = sh_value_quant::sh_value_u16_count(n, rest);
        const size_t n_bounds = sh_value_quant::n_bounds_for_prims(n) * 2;
        Tensor codes = Tensor::empty(TensorShape{n_cells}, Device::GPU, DataType::Float16);
        Tensor bounds = Tensor::empty(TensorShape{n_bounds}, Device::GPU, DataType::Float32);
        codes.zero_();
        bounds.zero_();
        sh_value_quant::encode_shN_float4_to_u16(
            src.ptr<float>(),
            static_cast<std::uint16_t*>(codes.data_ptr()),
            bounds.ptr<float>(),
            n,
            rest,
            src.stream());
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        return {codes.contiguous().cpu(), bounds.contiguous().cpu()};
    }

    EncodedPair program_pair(const std::vector<float>& host,
                             const size_t n,
                             const std::uint32_t rest,
                             const GpuBackend backend) {
        GpuBackendScope scope(backend);
        Tensor src = Tensor::from_vector(host, TensorShape{host.size()}, Device::CPU).to(Device::GPU);
        Tensor codes;
        Tensor bounds;
        sh_value_quant::encode_shN_float4_to_u16_tensor(
            src,
            n,
            sh_float4_slots_for_rest(rest),
            sh_value_quant::n_value_cells_per_prim(rest),
            codes,
            bounds);
        return {codes.contiguous().cpu(), bounds.contiguous().cpu()};
    }

    void compare_program_to_kernel(const std::vector<float>& host,
                                   const size_t n,
                                   const std::uint32_t rest,
                                   const GpuBackend backend) {
        const EncodedPair kernel = kernel_pair(host, n, rest);
        const EncodedPair program = program_pair(host, n, rest, backend);
        expect_u16_equal(program.codes, kernel.codes);
        expect_bounds_equal(program.bounds, kernel.bounds);
    }
} // namespace

TEST(ShQuantTensorProgram, MatchesKernelOnCudaForRandomSizes) {
    require_cuda();
    const std::vector<size_t> ns{1, 255, 256, 257, 1000, 4097};
    const std::vector<std::uint32_t> rests{3, 8, 15, 24, 45};
    std::uint32_t seed = 7;
    for (const size_t n : ns) {
        for (const std::uint32_t rest : rests) {
            SCOPED_TRACE("n=" + std::to_string(n) + " rest=" + std::to_string(rest));
            compare_program_to_kernel(random_swizzled(n, rest, seed++), n, rest, GpuBackend::CUDA);
        }
    }
}

TEST(ShQuantTensorProgram, MatchesKernelOnVulkanForRandomSizes) {
    require_cuda();
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan backend unavailable";
    }
    const std::vector<size_t> ns{1, 255, 256, 257, 1000, 4097};
    const std::vector<std::uint32_t> rests{3, 8, 15, 24, 45};
    std::uint32_t seed = 11;
    for (const size_t n : ns) {
        for (const std::uint32_t rest : rests) {
            SCOPED_TRACE("n=" + std::to_string(n) + " rest=" + std::to_string(rest));
            compare_program_to_kernel(random_swizzled(n, rest, seed++), n, rest, GpuBackend::Vulkan);
        }
    }
}

TEST(ShQuantTensorProgram, DegenerateBlocksMatchKernel) {
    require_cuda();
    const std::vector<GpuBackend> backends{GpuBackend::CUDA};
    std::vector<GpuBackend> all = backends;
    if (gpu_backend_available(GpuBackend::Vulkan)) {
        all.push_back(GpuBackend::Vulkan);
    }
    for (const GpuBackend backend : all) {
        SCOPED_TRACE(backend == GpuBackend::CUDA ? "cuda" : "vulkan");
        compare_program_to_kernel(filled_swizzled(256, 15, 0.0f), 256, 15, backend);
        compare_program_to_kernel(filled_swizzled(257, 8, 0.25f), 257, 8, backend);
        compare_program_to_kernel(filled_swizzled(1, 3, -0.5f), 1, 3, backend);
    }
}
