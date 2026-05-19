/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "io/formats/ply.hpp"
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

namespace {

    // Look for a real trained PLY we can use; skip cleanly if none is available.
    [[nodiscard]] std::filesystem::path find_real_ply() {
        for (const char* candidate : {
                 "/tmp/swizzle_fold/splat_7000.ply",
                 "output_swizzle_test/splat_7000.ply",
                 "output_swizzle_test/splat_5.ply",
             }) {
            if (std::filesystem::exists(candidate))
                return candidate;
        }
        return {};
    }

    // Round-trip the shN buffer: canonical -> swizzled -> canonical and verify equality.
    TEST(ShSwizzleLayout, CanonicalRoundTrip_RealData) {
        using namespace lfs::core;

        const auto ply_path = find_real_ply();
        if (ply_path.empty()) {
            GTEST_SKIP() << "No trained PLY available; run the smoke test first";
        }

        auto loaded = lfs::io::load_ply(ply_path);
        ASSERT_TRUE(loaded.has_value()) << "Failed to load PLY at " << ply_path;

        SplatData splat = std::move(*loaded);
        // PLY load -> SplatData ctor reorders into swizzled storage. Now deswizzle and
        // round-trip through swizzle/deswizzle directly to validate the kernels.
        if (splat.get_active_sh_degree() == 0) {
            GTEST_SKIP() << "PLY has SH degree 0, no shN to round-trip";
        }

        const Tensor canonical = splat.shN_canonical();
        ASSERT_TRUE(canonical.is_valid());
        ASSERT_EQ(canonical.ndim(), 3);

        const size_t N = canonical.shape()[0];
        const std::uint32_t K = static_cast<std::uint32_t>(canonical.shape()[1]);
        ASSERT_GT(N, 0u);
        ASSERT_GT(K, 0u);
        ASSERT_EQ(canonical.shape()[2], 3u);

        // Build a swizzled buffer from the canonical view.
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);

        // Deswizzle back into a fresh canonical buffer.
        Tensor recovered = Tensor::empty({N, K, 3}, Device::CUDA);
        undo_reorder_sh_from_swizzled(swizzled.ptr<float>(), recovered.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        // Bitwise compare.
        auto cpu_canonical = canonical.contiguous().to(Device::CPU);
        auto cpu_recovered = recovered.contiguous().to(Device::CPU);
        const float* a = cpu_canonical.ptr<float>();
        const float* b = cpu_recovered.ptr<float>();
        const size_t total = N * K * 3;

        size_t mismatches = 0;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < total; ++i) {
            const float d = std::fabs(a[i] - b[i]);
            if (d > 0.0f) {
                ++mismatches;
                if (d > max_abs_diff)
                    max_abs_diff = d;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "max_abs_diff=" << max_abs_diff;
    }

    // Verify swizzled index math against the on-device kernel layout by checking a few
    // explicit positions are written where shAt() says they should be.
    TEST(ShSwizzleLayout, IndexFormulaMatchesKernel) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 96; // 3 full blocks of 32
        constexpr std::uint32_t K = 15;

        // Build a canonical buffer where each (p, k, c) holds the value p*1000 + k*10 + c.
        std::vector<float> host_canonical(N * K * 3);
        for (std::uint32_t p = 0; p < N; ++p) {
            for (std::uint32_t k = 0; k < K; ++k) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    host_canonical[p * K * 3 + k * 3 + c] = static_cast<float>(p * 1000 + k * 10 + c);
                }
            }
        }

        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        auto host = swizzled.to(Device::CPU);
        const float* sw = host.ptr<float>();

        // For each (p, k, c) the value should live at shAt(p, k) * 3 + c.
        for (std::uint32_t p : {0u, 1u, 31u, 32u, 64u, 95u}) {
            for (std::uint32_t k : {0u, 7u, 14u}) {
                for (std::uint32_t c : {0u, 1u, 2u}) {
                    const std::uint32_t idx = sh_swizzled_index(p, k) * 3 + c;
                    const float expected = static_cast<float>(p * 1000 + k * 10 + c);
                    EXPECT_EQ(sw[idx], expected)
                        << "p=" << p << " k=" << k << " c=" << c << " idx=" << idx;
                }
            }
        }
    }

    // Lane padding (primitives in the trailing block beyond N) should be zero after reorder.
    TEST(ShSwizzleLayout, PaddingLanesAreZero) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 70; // not a multiple of 32 — last block has 6 padding lanes
        constexpr std::uint32_t K = 15;

        std::vector<float> host_canonical(N * K * 3, 1.0f); // non-zero source so we know padding is from us
        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        auto host = swizzled.to(Device::CPU);
        const float* sw = host.ptr<float>();

        // Padding lanes are at primitive indices [N, ceil(N/32)*32). For each, all K
        // coeffs * 3 channels must be 0.
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(N));
        for (std::uint32_t p = N; p < padded_n; ++p) {
            for (std::uint32_t k = 0; k < K; ++k) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    const std::uint32_t idx = sh_swizzled_index(p, k) * 3 + c;
                    EXPECT_EQ(sw[idx], 0.0f) << "padding lane p=" << p << " k=" << k << " c=" << c;
                }
            }
        }
    }

} // namespace
