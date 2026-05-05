/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Golden tests for the VkSplat host input layout.
//
// These tests pin down the exact contract that the VkSplat forward pass
// expects from the visualizer-side packer (currently the only producer).
// Any future optimization that bypasses the GPU->CPU->GPU staging path
// MUST keep the same byte layout in the four upload buffers, otherwise
// the precompiled SPIR-V projection and rasterization shaders read
// garbage. The tests below construct deterministic SplatData inputs and
// assert byte-level invariants on the packer's output.

#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/rasterizer/vksplat_fwd/src/buffer.h"
#include "rendering/rasterizer/vksplat_fwd/src/config.h"
#include "visualizer/rendering/vksplat_input_packer.hpp"

#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::vis::vksplat::buildPaddedShReference;
using lfs::vis::vksplat::DevicePackedInputs;
using lfs::vis::vksplat::packDeviceInputs;
using lfs::vis::vksplat::packHostInputs;

namespace {

    [[nodiscard]] float sigmoidf(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    struct SyntheticInputs {
        std::vector<float> means;    // [N,3]
        std::vector<float> sh0;      // [N,1,3]
        std::vector<float> shN;      // [N,K,3] (K = shn_coeffs)
        std::vector<float> scaling;  // [N,3]   raw (pre-exp)
        std::vector<float> rotation; // [N,4]   raw (pre-normalize)
        std::vector<float> opacity;  // [N,1]   raw (pre-sigmoid)
        std::size_t n = 0;
        int max_sh_degree = 3;
        int shn_coeffs = 0;
    };

    [[nodiscard]] SyntheticInputs makeInputs(std::size_t n, int max_sh_degree, std::uint32_t seed) {
        SyntheticInputs in;
        in.n = n;
        in.max_sh_degree = max_sh_degree;
        in.shn_coeffs = max_sh_degree > 0 ? (max_sh_degree + 1) * (max_sh_degree + 1) - 1 : 0;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> mean_dist(-2.0f, 2.0f);
        std::uniform_real_distribution<float> color_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> scale_dist(-3.0f, 0.5f);
        std::uniform_real_distribution<float> quat_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> opacity_dist(-2.0f, 2.0f);

        in.means.resize(n * 3);
        for (auto& v : in.means)
            v = mean_dist(rng);
        in.sh0.resize(n * 3);
        for (auto& v : in.sh0)
            v = color_dist(rng);
        in.shN.resize(n * static_cast<std::size_t>(in.shn_coeffs) * 3);
        for (auto& v : in.shN)
            v = color_dist(rng);
        in.scaling.resize(n * 3);
        for (auto& v : in.scaling)
            v = scale_dist(rng);
        in.rotation.resize(n * 4);
        for (auto& v : in.rotation)
            v = quat_dist(rng);
        in.opacity.resize(n);
        for (auto& v : in.opacity)
            v = opacity_dist(rng);
        return in;
    }

    [[nodiscard]] std::unique_ptr<SplatData> buildSplatData(const SyntheticInputs& in) {
        const std::size_t n = in.n;
        const int shn_coeffs = in.shn_coeffs;

        Tensor means = Tensor::from_vector(in.means, {n, std::size_t{3}}, Device::CUDA).to(DataType::Float32);
        Tensor sh0 = Tensor::from_vector(in.sh0, {n, std::size_t{1}, std::size_t{3}}, Device::CUDA).to(DataType::Float32);
        Tensor shN = shn_coeffs > 0
                         ? Tensor::from_vector(in.shN,
                                               {n, static_cast<std::size_t>(shn_coeffs), std::size_t{3}},
                                               Device::CUDA)
                               .to(DataType::Float32)
                         : Tensor{};
        Tensor scaling = Tensor::from_vector(in.scaling, {n, std::size_t{3}}, Device::CUDA).to(DataType::Float32);
        Tensor rotation = Tensor::from_vector(in.rotation, {n, std::size_t{4}}, Device::CUDA).to(DataType::Float32);
        Tensor opacity = Tensor::from_vector(in.opacity, {n, std::size_t{1}}, Device::CUDA).to(DataType::Float32);

        auto splat = std::make_unique<SplatData>(in.max_sh_degree,
                                                 std::move(means),
                                                 std::move(sh0),
                                                 std::move(shN),
                                                 std::move(scaling),
                                                 std::move(rotation),
                                                 std::move(opacity),
                                                 1.0f);
        splat->set_active_sh_degree(in.max_sh_degree);
        splat->set_max_sh_degree(in.max_sh_degree);
        return splat;
    }

    void verifyMeans(const SyntheticInputs& in, const Buffer<float>& xyz_ws) {
        ASSERT_EQ(xyz_ws.size(), in.n * 3);
        for (std::size_t i = 0; i < in.n; ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_FLOAT_EQ(xyz_ws[i * 3 + c], in.means[i * 3 + c])
                    << " at i=" << i << " c=" << c;
            }
        }
    }

    void verifyRotations(const SyntheticInputs& in, const Buffer<float>& rotations) {
        ASSERT_EQ(rotations.size(), in.n * 4);
        for (std::size_t i = 0; i < in.n; ++i) {
            const float* raw = &in.rotation[i * 4];
            float sq = 0.0f;
            for (int c = 0; c < 4; ++c)
                sq += raw[c] * raw[c];
            const float norm = std::max(std::sqrt(sq), 1e-12f);
            for (int c = 0; c < 4; ++c) {
                EXPECT_NEAR(rotations[i * 4 + c], raw[c] / norm, 1e-5f)
                    << " at i=" << i << " c=" << c;
            }
            float out_sq = 0.0f;
            for (int c = 0; c < 4; ++c) {
                const float v = rotations[i * 4 + c];
                out_sq += v * v;
            }
            EXPECT_NEAR(std::sqrt(out_sq), 1.0f, 1e-4f) << " quaternion not unit length, i=" << i;
        }
    }

    void verifyScalesOpacs(const SyntheticInputs& in, const Buffer<float>& scales_opacs) {
        ASSERT_EQ(scales_opacs.size(), in.n * 4);
        for (std::size_t i = 0; i < in.n; ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                const float expected = std::exp(in.scaling[i * 3 + c]);
                EXPECT_NEAR(scales_opacs[i * 4 + c], expected, 1e-5f * std::max(1.0f, std::abs(expected)))
                    << " at i=" << i << " c=" << c;
            }
            const float expected_opac = sigmoidf(in.opacity[i]);
            EXPECT_NEAR(scales_opacs[i * 4 + 3], expected_opac, 1e-5f)
                << " opacity at i=" << i;
        }
    }

    void verifyShCoeffs(const SyntheticInputs& in, const Buffer<float>& sh_coeffs) {
        // The reorderSH pass pads sh_coeffs up to the next 4 * SH_DIM * SH_REORDER_SIZE
        // multiple where SH_DIM=12 (=16 SH * 3 channels / 4). So the size grows in
        // groups of SH_REORDER_SIZE gaussians.
        constexpr std::size_t SH_DIM = 12;
        constexpr std::size_t REORDER = SH_REORDER_SIZE;
        const std::size_t expected_groups = (in.n + REORDER - 1) / REORDER;
        const std::size_t expected_size = expected_groups * 4 * SH_DIM * REORDER;
        ASSERT_EQ(sh_coeffs.size(), expected_size);
        ASSERT_EQ(sh_coeffs.size() % (16 * 3), 0u);

        // Round-trip through undoReorderSH to recover the un-reordered padded layout
        // and verify against the explicit reference.
        Buffer<float> reordered;
        reordered.assign(sh_coeffs.begin(), sh_coeffs.end());
        VulkanGSPipelineBuffers::undoReorderSH(reordered, in.n);
        ASSERT_EQ(reordered.size(), in.n * 16 * 3);

        const auto reference = buildPaddedShReference(*buildSplatData(in));
        ASSERT_TRUE(reference.has_value()) << reference.error();
        ASSERT_EQ(reference->size(), in.n * 16 * 3);

        for (std::size_t i = 0; i < reordered.size(); ++i) {
            EXPECT_FLOAT_EQ(reordered[i], (*reference)[i]) << " at flat index " << i;
        }
    }

    void verifyShReferenceContents(const SyntheticInputs& in, const std::vector<float>& reference) {
        ASSERT_EQ(reference.size(), in.n * 16 * 3);
        const std::size_t source_rest = static_cast<std::size_t>(in.shn_coeffs);
        const std::size_t rest = std::min<std::size_t>(15, source_rest);
        for (std::size_t i = 0; i < in.n; ++i) {
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_FLOAT_EQ(reference[(i * 16) * 3 + c], in.sh0[i * 3 + c])
                    << " DC mismatch at i=" << i << " c=" << c;
            }
            for (std::size_t k = 0; k < rest; ++k) {
                for (std::size_t c = 0; c < 3; ++c) {
                    EXPECT_FLOAT_EQ(reference[((i * 16) + (k + 1)) * 3 + c],
                                    in.shN[(i * source_rest + k) * 3 + c])
                        << " rest mismatch at i=" << i << " k=" << k << " c=" << c;
                }
            }
            // Slots after rest+1 must be zero.
            for (std::size_t slot = rest + 1; slot < 16; ++slot) {
                for (std::size_t c = 0; c < 3; ++c) {
                    EXPECT_FLOAT_EQ(reference[((i * 16) + slot) * 3 + c], 0.0f)
                        << " padding nonzero at i=" << i << " slot=" << slot << " c=" << c;
                }
            }
        }
    }

} // namespace

class VksplatInputPackerTest : public ::testing::TestWithParam<std::tuple<std::size_t, int>> {};

TEST_P(VksplatInputPackerTest, PackedLayoutMatchesContract) {
    const auto [n, max_sh_degree] = GetParam();
    const SyntheticInputs in = makeInputs(n, max_sh_degree, /*seed=*/0xA17u + static_cast<std::uint32_t>(n));
    auto splat = buildSplatData(in);

    Buffer<float> xyz_ws;
    Buffer<float> rotations;
    Buffer<float> scales_opacs;
    Buffer<float> sh_coeffs;
    auto result = packHostInputs(*splat, xyz_ws, rotations, scales_opacs, sh_coeffs);
    ASSERT_TRUE(result.has_value()) << result.error();

    verifyMeans(in, xyz_ws);
    verifyRotations(in, rotations);
    verifyScalesOpacs(in, scales_opacs);
    verifyShCoeffs(in, sh_coeffs);
}

INSTANTIATE_TEST_SUITE_P(
    VkSplatLayouts,
    VksplatInputPackerTest,
    ::testing::Values(
        std::make_tuple(static_cast<std::size_t>(1), 3),
        std::make_tuple(static_cast<std::size_t>(31), 3),
        std::make_tuple(static_cast<std::size_t>(SH_REORDER_SIZE), 3),
        std::make_tuple(static_cast<std::size_t>(SH_REORDER_SIZE + 1), 3),
        std::make_tuple(static_cast<std::size_t>(257), 2),
        std::make_tuple(static_cast<std::size_t>(513), 1),
        std::make_tuple(static_cast<std::size_t>(1024), 0)));

TEST(VksplatInputPackerTest, PaddedShReferenceMatchesSourceLayout) {
    const SyntheticInputs in = makeInputs(/*n=*/97, /*max_sh_degree=*/3, /*seed=*/0xBEEFu);
    auto splat = buildSplatData(in);

    auto reference = buildPaddedShReference(*splat);
    ASSERT_TRUE(reference.has_value()) << reference.error();
    verifyShReferenceContents(in, *reference);
}

TEST(VksplatInputPackerTest, PaddedShReferenceTruncatesShnTo15) {
    // Construct an oversized shN tensor with 18 coeffs and verify the packer
    // truncates to 15 (16 total slots minus the DC slot) rather than overrunning.
    constexpr std::size_t n = 8;
    constexpr int oversized = 18;

    SyntheticInputs in;
    in.n = n;
    in.max_sh_degree = 3;
    in.shn_coeffs = oversized;
    in.means.resize(n * 3, 0.5f);
    in.sh0.resize(n * 3);
    for (std::size_t i = 0; i < n * 3; ++i)
        in.sh0[i] = static_cast<float>(i) * 0.1f;
    in.shN.resize(n * oversized * 3);
    for (std::size_t i = 0; i < in.shN.size(); ++i)
        in.shN[i] = static_cast<float>(i + 1);
    in.scaling.resize(n * 3, -1.0f);
    in.rotation.assign(n * 4, 0.0f);
    for (std::size_t i = 0; i < n; ++i)
        in.rotation[i * 4] = 1.0f;
    in.opacity.resize(n, 0.0f);

    auto splat = buildSplatData(in);
    auto reference = buildPaddedShReference(*splat);
    ASSERT_TRUE(reference.has_value()) << reference.error();
    ASSERT_EQ(reference->size(), n * 16 * 3);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < 15; ++k) {
            for (std::size_t c = 0; c < 3; ++c) {
                EXPECT_FLOAT_EQ((*reference)[((i * 16) + (k + 1)) * 3 + c],
                                in.shN[(i * oversized + k) * 3 + c])
                    << " truncation mismatch at i=" << i << " k=" << k << " c=" << c;
            }
        }
    }
}

class VksplatDevicePackerTest : public ::testing::TestWithParam<std::tuple<std::size_t, int>> {};

TEST_P(VksplatDevicePackerTest, DeviceOutputMatchesHostReferenceByteForByte) {
    const auto [n, max_sh_degree] = GetParam();
    const SyntheticInputs in = makeInputs(n, max_sh_degree, /*seed=*/0xD0Cu + static_cast<std::uint32_t>(n));
    auto splat = buildSplatData(in);

    Buffer<float> host_xyz, host_rot, host_scales_opacs, host_sh;
    auto host = packHostInputs(*splat, host_xyz, host_rot, host_scales_opacs, host_sh);
    ASSERT_TRUE(host.has_value()) << host.error();

    auto device = packDeviceInputs(*splat);
    ASSERT_TRUE(device.has_value()) << device.error();
    ASSERT_EQ(device->num_splats, n);

    const auto compare = [](const Tensor& gpu_tensor,
                            const float* host_ptr,
                            std::size_t expected_count,
                            const char* label) {
        ASSERT_EQ(static_cast<std::size_t>(gpu_tensor.numel()), expected_count) << label;
        Tensor cpu = gpu_tensor.cpu().contiguous();
        const float* gpu_ptr = cpu.ptr<float>();
        ASSERT_NE(gpu_ptr, nullptr) << label;
        for (std::size_t i = 0; i < expected_count; ++i) {
            // Tensor-driven activations have a slightly different math path than
            // the std::exp/sigmoidf used in the CPU packer; permit a tight
            // floating-point tolerance per element.
            EXPECT_NEAR(gpu_ptr[i], host_ptr[i], 1e-5f * std::max(1.0f, std::abs(host_ptr[i])))
                << label << " mismatch at flat index " << i;
        }
    };

    compare(device->xyz_ws, host_xyz.data(), host_xyz.size(), "xyz_ws");
    compare(device->rotations, host_rot.data(), host_rot.size(), "rotations");
    compare(device->scales_opacs, host_scales_opacs.data(), host_scales_opacs.size(), "scales_opacs");
    compare(device->sh_coeffs, host_sh.data(), host_sh.size(), "sh_coeffs");
}

INSTANTIATE_TEST_SUITE_P(
    VkSplatDeviceLayouts,
    VksplatDevicePackerTest,
    ::testing::Values(
        std::make_tuple(static_cast<std::size_t>(1), 3),
        std::make_tuple(static_cast<std::size_t>(31), 3),
        std::make_tuple(static_cast<std::size_t>(SH_REORDER_SIZE), 3),
        std::make_tuple(static_cast<std::size_t>(SH_REORDER_SIZE + 1), 3),
        std::make_tuple(static_cast<std::size_t>(257), 2),
        std::make_tuple(static_cast<std::size_t>(513), 1),
        std::make_tuple(static_cast<std::size_t>(1024), 0)));

TEST(VksplatInputPackerTest, ScalesOpacsByteLayout) {
    // Locks down the (s0, s1, s2, opacity) interleave that the projection
    // shader assumes: every fourth float must be sigmoid(opacity_raw).
    constexpr std::size_t n = 12;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/0, /*seed=*/0xCAFEu);
    auto splat = buildSplatData(in);

    Buffer<float> xyz_ws, rotations, scales_opacs, sh_coeffs;
    ASSERT_TRUE(packHostInputs(*splat, xyz_ws, rotations, scales_opacs, sh_coeffs));

    ASSERT_EQ(scales_opacs.size(), n * 4);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GT(scales_opacs[i * 4 + 0], 0.0f);
        EXPECT_GT(scales_opacs[i * 4 + 1], 0.0f);
        EXPECT_GT(scales_opacs[i * 4 + 2], 0.0f);
        EXPECT_GE(scales_opacs[i * 4 + 3], 0.0f);
        EXPECT_LE(scales_opacs[i * 4 + 3], 1.0f);
        EXPECT_NEAR(scales_opacs[i * 4 + 3], sigmoidf(in.opacity[i]), 1e-5f);
    }
}
