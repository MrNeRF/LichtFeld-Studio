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
    // Layout bijects the 45 active floats per primitive onto 12 float4 slots (48 floats
    // with 3 of tail padding); the active range round-trips bitwise.
    TEST(ShSwizzleLayout, CanonicalRoundTrip_RealData) {
        using namespace lfs::core;

        const auto ply_path = find_real_ply();
        if (ply_path.empty()) {
            GTEST_SKIP() << "No trained PLY available; run the smoke test first";
        }

        auto loaded = lfs::io::load_ply(ply_path);
        ASSERT_TRUE(loaded.has_value()) << "Failed to load PLY at " << ply_path;

        SplatData splat = std::move(loaded->value);
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
        const size_t swizzled_floats = sh_swizzled_float_count(N, K);
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

    TEST(ShSwizzleLayout, CompactStorageMatchesActiveDegree) {
        using namespace lfs::core;
        constexpr std::size_t N = 70;
        constexpr std::size_t BLOCKS = 3;

        EXPECT_EQ(sh_float4_slots_for_rest(0), 0u);
        EXPECT_EQ(sh_float4_slots_for_rest(3), 3u);
        EXPECT_EQ(sh_float4_slots_for_rest(8), 6u);
        EXPECT_EQ(sh_float4_slots_for_rest(15), 12u);

        EXPECT_EQ(sh_swizzled_float_count(N, 0), 0u);
        EXPECT_EQ(sh_swizzled_float_count(N, 3), BLOCKS * 3u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N, 8), BLOCKS * 6u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N, 15), BLOCKS * 12u * kShReorderSize * 4u);
        EXPECT_EQ(sh_swizzled_float_count(N), sh_swizzled_float_count(N, 15));
    }

    // Verify the float4 shuffle: the swizzled buffer's float4 at sh_swizzled_index(p, k)
    // holds 4 consecutive floats of the linear canonical row for primitive p (with the last
    // slot's tail floats zero-padded). Equivalent to vksplat's tight pack of 45 floats into
    // 12 float4 = 48 floats with 3 of tail padding.
    TEST(ShSwizzleLayout, IndexFormulaMatchesKernel) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 96; // 3 full blocks of 32
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t FLOATS_PER_PRIM = K * 3u; // 45
        constexpr std::uint32_t SLOTS_PER_PRIM = 12u;     // SH_REST_FLOAT4_PER_PRIMITIVE

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

        // For each primitive's float4 slot k, the 4 components map to the canonical row at
        // offsets [k*4 .. k*4+3]; out-of-range offsets are zero.
        for (std::uint32_t p : {0u, 1u, 31u, 32u, 64u, 95u}) {
            for (std::uint32_t k = 0; k < SLOTS_PER_PRIM; ++k) {
                const std::uint32_t base_float = sh_swizzled_index(p, k) * 4u;
                for (std::uint32_t i = 0; i < 4; ++i) {
                    const std::uint32_t canonical_off = k * 4u + i;
                    const float expected = canonical_off < FLOATS_PER_PRIM
                                               ? host_canonical[p * FLOATS_PER_PRIM + canonical_off]
                                               : 0.0f;
                    EXPECT_EQ(sw[base_float + i], expected)
                        << "p=" << p << " k=" << k << " i=" << i;
                }
            }
        }
    }

    TEST(ShSwizzleLayout, GatherSelectedRowsToLinearMatchesCanonical) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 70;
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t FLOATS_PER_PRIM = K * 3u;

        std::vector<float> host_canonical(N * K * 3);
        for (std::uint32_t p = 0; p < N; ++p) {
            for (std::uint32_t k = 0; k < K; ++k) {
                for (std::uint32_t c = 0; c < 3; ++c) {
                    host_canonical[p * K * 3 + k * 3 + c] =
                        static_cast<float>(p * 1000 + k * 10 + c);
                }
            }
        }

        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        Tensor swizzled = Tensor::zeros({sh_swizzled_float_count(N)}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);

        const std::vector<int> selected = {0, 5, 31, 32, 69};
        Tensor indices = Tensor::from_vector(selected, {selected.size()}, Device::CUDA).to(DataType::Int64);
        Tensor gathered = Tensor::empty({selected.size(), K, 3}, Device::CUDA);
        shN_swizzled_gather_to_linear_i64(
            swizzled.ptr<float>(), indices.ptr<std::int64_t>(),
            gathered.ptr<float>(), selected.size(), K);
        cudaDeviceSynchronize();

        auto gathered_cpu = gathered.to(Device::CPU);
        const float* g = gathered_cpu.ptr<float>();
        for (size_t row = 0; row < selected.size(); ++row) {
            const auto p = static_cast<std::uint32_t>(selected[row]);
            for (std::uint32_t off = 0; off < FLOATS_PER_PRIM; ++off) {
                EXPECT_EQ(g[row * FLOATS_PER_PRIM + off],
                          host_canonical[p * FLOATS_PER_PRIM + off])
                    << "row=" << row << " primitive=" << p << " off=" << off;
            }
        }
    }

    // Both lane padding (primitives in the trailing block beyond N) and the tail padding
    // (slot 11's .y/.z/.w per primitive) must be zero after reorder.
    TEST(ShSwizzleLayout, PaddingLanesAreZero) {
        using namespace lfs::core;
        constexpr std::uint32_t N = 70; // last block has 6 padding lanes
        constexpr std::uint32_t K = 15;
        constexpr std::uint32_t SLOTS_PER_PRIM = 12u;

        std::vector<float> host_canonical(N * K * 3, 1.0f); // non-zero source so padding is visibly distinct
        Tensor canonical = Tensor::from_vector(host_canonical, {N, K, 3}, Device::CUDA);
        const size_t swizzled_floats = sh_swizzled_float_count(N);
        Tensor swizzled = Tensor::zeros({swizzled_floats}, Device::CUDA);
        reorder_sh_to_swizzled(canonical.ptr<float>(), swizzled.ptr<float>(), N, K);
        cudaDeviceSynchronize();

        auto host = swizzled.to(Device::CPU);
        const float* sw = host.ptr<float>();

        // (a) Lane padding: primitives in [N, ceil(N/32)*32) must be all-zero across all 12
        // float4 slots.
        const std::uint32_t padded_n = static_cast<std::uint32_t>(sh_swizzled_padded_n(N));
        for (std::uint32_t p = N; p < padded_n; ++p) {
            for (std::uint32_t k = 0; k < SLOTS_PER_PRIM; ++k) {
                const std::uint32_t base = sh_swizzled_index(p, k) * 4u;
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    EXPECT_EQ(sw[base + i], 0.0f) << "lane padding p=" << p << " k=" << k << " i=" << i;
                }
            }
        }

        // (b) Tail padding: slot 11's .y/.z/.w (offsets 1..3) of every valid primitive must be 0.
        for (std::uint32_t p = 0; p < N; ++p) {
            const std::uint32_t base = sh_swizzled_index(p, 11u) * 4u;
            EXPECT_EQ(sw[base + 1u], 0.0f) << "tail .y p=" << p;
            EXPECT_EQ(sw[base + 2u], 0.0f) << "tail .z p=" << p;
            EXPECT_EQ(sw[base + 3u], 0.0f) << "tail .w p=" << p;
        }
    }

} // namespace
