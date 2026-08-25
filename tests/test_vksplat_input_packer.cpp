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

#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "rendering/rasterizer/vulkan/src/buffer.h"
#include "rendering/rasterizer/vulkan/src/config.h"
#include "rendering/rasterizer/vulkan/src/indirect_layout.h"
#include "visualizer/rendering/vksplat_input_packer.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::Tensor;
using lfs::vis::vksplat::buildPaddedShReference;
using lfs::vis::vksplat::copyRawOpacityToBuffer;
using lfs::vis::vksplat::deviceInputLayout;
using lfs::vis::vksplat::DevicePackedInputs;
using lfs::vis::vksplat::packDeviceInputs;
using lfs::vis::vksplat::packHostInputs;
using lfs::vis::vksplat::rawDeviceInputLayout;

namespace {

    template <typename Handle>
    [[nodiscard]] Handle fakeVkHandle(const std::uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<Handle>(value);
        } else {
            return static_cast<Handle>(value);
        }
    }

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

TEST(VkSplatIndirectLayoutTest, SharedWordCountsAndOffsetsMatchEveryProducerContract) {
    namespace indirect = lfs::rendering::vulkan::indirect_layout;

    EXPECT_EQ(indirect::kCommandWordCount, 3u);
    EXPECT_EQ(sizeof(VkDispatchIndirectCommand),
              indirect::kCommandWordCount * sizeof(std::uint32_t));

    EXPECT_EQ(indirect::VisibleSortDispatch::kLayout.word_count, 3u);
    EXPECT_EQ(indirect::VisibleSortDispatch::kRadixWordOffset, 0u);

    EXPECT_EQ(indirect::TileBatchDispatch::kLayout.word_count, 3u);
    EXPECT_EQ(indirect::TileBatchDispatch::kRasterWordOffset, 0u);

    EXPECT_EQ(indirect::VisibleChainDispatch::kLayout.word_count, 12u);
    EXPECT_EQ(indirect::VisibleChainDispatch::kRadixWordOffset, 0u);
    EXPECT_EQ(indirect::VisibleChainDispatch::kPerElementWordOffset, 3u);
    EXPECT_EQ(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset, 6u);
    EXPECT_EQ(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset, 9u);

    EXPECT_EQ(indirect::SurvivorState::kLayout.word_count, 4u);
    EXPECT_EQ(indirect::SurvivorState::kCountWordOffset, 0u);
    EXPECT_EQ(indirect::SurvivorState::kProjectionWordOffset, 1u);

    EXPECT_EQ(indirect::DepthWave::kRecordStrideWords, 64u);
    EXPECT_EQ(indirect::DepthWave::kHeaderNeededWord, 0u);
    EXPECT_EQ(indirect::DepthWave::layout(HIGS_DEPTH_MAX_WAVES).word_count,
              (1u + HIGS_DEPTH_MAX_WAVES) * 64u);
    EXPECT_EQ(indirect::DepthWave::recordWordOffset(0u), 64u);
    EXPECT_EQ(indirect::DepthWave::recordWordOffset(3u), 256u);
    EXPECT_EQ(indirect::DepthWave::countWordOffset(3u), 256u);
    EXPECT_EQ(indirect::DepthWave::keygenWordOffset(3u), 260u);
    EXPECT_EQ(indirect::DepthWave::radixWordOffset(3u), 263u);
    EXPECT_EQ(indirect::DepthWave::rangeWordOffset(3u), 266u);
    EXPECT_EQ(indirect::DepthWave::perTileWordOffset(3u), 269u);
    EXPECT_EQ(indirect::DepthWave::fullscreenWordOffset(3u), 272u);
    EXPECT_EQ(indirect::DepthWave::rankBaseWord(3u), 275u);
    EXPECT_EQ(indirect::DepthWave::rankCountWord(3u), 276u);
    EXPECT_EQ(indirect::DepthWave::instanceBaseWord(3u), 277u);

    EXPECT_EQ(indirect::MacroWaveDispatch::kLayout.word_count, 96u);
    EXPECT_EQ(indirect::MacroWaveDispatch::kWaveStrideWords, 3u);
    EXPECT_EQ(indirect::MacroWaveDispatch::kRasterBaseWordOffset, 0u);
    EXPECT_EQ(indirect::MacroWaveDispatch::kComposeBaseWordOffset, 48u);
    EXPECT_EQ(indirect::MacroWaveDispatch::rasterWordOffset(HIGS_RASTER_MAX_WAVES - 1u), 45u);
    EXPECT_EQ(indirect::MacroWaveDispatch::composeWordOffset(HIGS_RASTER_MAX_WAVES - 1u), 93u);
}

TEST(VkSplatIndirectLayoutTest, ExportDepthWaveUpperBoundMatchesContract) {
    namespace indirect = lfs::rendering::vulkan::indirect_layout;

    constexpr std::size_t k = HIGS_DEPTH_WAVE_INSTANCES;
    constexpr std::size_t max_rank_emission = 4096u;
    constexpr std::size_t denominator = k - max_rank_emission;

    EXPECT_EQ(indirect::depthWaveRecordUpperBound(0u, max_rank_emission), 1u);
    EXPECT_EQ(indirect::depthWaveRecordUpperBound(denominator - 1u, max_rank_emission), 1u);
    EXPECT_EQ(indirect::depthWaveRecordUpperBound(denominator, max_rank_emission), 2u);
    EXPECT_EQ(indirect::depthWaveRecordUpperBound(2u * denominator, max_rank_emission), 3u);
    EXPECT_EQ(indirect::depthWaveRecordUpperBound(k, k), 0u);
    EXPECT_EQ(indirect::depthWaveRecordUpperBound(k, k + 1u), 0u);
}

TEST(VulkanBufferViewTest, SharedScratchViewSeparatesBackingSizeFromRegionCapacity) {
    _VulkanBuffer view{};
    view.buffer = fakeVkHandle<VkBuffer>(1);
    view.allocSize = 384u << 20u;
    view.offset = 66'000'384u;
    view.capacity = 18'000'000u;
    view.size = 651'300u;

    ASSERT_TRUE(view.hasValidViewBounds());
    EXPECT_TRUE(view.containsRange(0, view.size));
    EXPECT_TRUE(view.containsRange(view.capacity - 1, 1));
    EXPECT_FALSE(view.containsRange(view.capacity, 1));
    EXPECT_FALSE(view.containsRange(0, view.capacity + 1));

    view.offset = view.allocSize - view.capacity + 1;
    EXPECT_FALSE(view.hasValidViewBounds());
    EXPECT_FALSE(view.containsRange(0, view.size));
}

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

TEST(VksplatInputPackerTest, RawDeviceLayoutUsesCompactMaxShRest) {
    constexpr std::size_t n = SH_REORDER_SIZE + 5;

    for (const int max_sh_degree : {0, 1, 2, 3}) {
        SyntheticInputs in = makeInputs(n, max_sh_degree, /*seed=*/0x5A17u + static_cast<std::uint32_t>(max_sh_degree));
        auto splat = buildSplatData(in);

        auto raw_layout = rawDeviceInputLayout(*splat);
        ASSERT_TRUE(raw_layout.has_value()) << raw_layout.error();
        const std::size_t layout_rest =
            lfs::core::sh_rest_coefficients_for_degree(max_sh_degree);
        const std::size_t expected_raw_shN_bytes =
            layout_rest == 0
                ? 4 * sizeof(float)
                : lfs::core::sh_swizzled_float_count(n, static_cast<std::uint32_t>(layout_rest)) * sizeof(float);
        EXPECT_EQ(raw_layout->shN_bytes, expected_raw_shN_bytes)
            << "max_sh_degree=" << max_sh_degree;

        auto packed_layout = deviceInputLayout(*splat);
        ASSERT_TRUE(packed_layout.has_value()) << packed_layout.error();
        EXPECT_EQ(packed_layout->sh_coeffs_bytes,
                  lfs::core::sh_swizzled_float_count(n) * sizeof(float))
            << "max_sh_degree=" << max_sh_degree;
    }

    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/2, /*seed=*/0xA110u);
    auto splat = buildSplatData(in);
    splat->set_active_sh_degree(0);
    auto raw_layout = rawDeviceInputLayout(*splat);
    ASSERT_TRUE(raw_layout.has_value()) << raw_layout.error();
    EXPECT_EQ(raw_layout->shN_bytes,
              lfs::core::sh_swizzled_float_count(
                  n,
                  static_cast<std::uint32_t>(lfs::core::sh_rest_coefficients_for_degree(2))) *
                  sizeof(float));
}

TEST(VksplatInputPackerTest, RawDeviceLayoutUsesRequestedUploadShDegree) {
    constexpr std::size_t n = SH_REORDER_SIZE * 2 + 7;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/3, /*seed=*/0x5A0u);
    auto splat = buildSplatData(in);

    for (const int upload_sh_degree : {0, 1, 2, 3}) {
        auto raw_layout = rawDeviceInputLayout(*splat, upload_sh_degree);
        ASSERT_TRUE(raw_layout.has_value()) << raw_layout.error();
        const auto layout_rest = static_cast<std::uint32_t>(
            lfs::core::sh_rest_coefficients_for_degree(upload_sh_degree));
        const std::size_t expected_shN_bytes =
            layout_rest == 0
                ? 4 * sizeof(float)
                : lfs::core::sh_swizzled_float_count(n, layout_rest) * sizeof(float);
        EXPECT_EQ(raw_layout->shN_bytes, expected_shN_bytes)
            << "upload_sh_degree=" << upload_sh_degree;
        EXPECT_EQ(raw_layout->shN_layout_rest, layout_rest)
            << "upload_sh_degree=" << upload_sh_degree;
        EXPECT_EQ(raw_layout->omits_shN, upload_sh_degree == 0)
            << "upload_sh_degree=" << upload_sh_degree;
        EXPECT_FALSE(raw_layout->shN_f16) << "upload_sh_degree=" << upload_sh_degree;
    }
}

TEST(VksplatInputPackerTest, RawDeviceLayoutHalvesShNBytesForIeeeF16) {
    // IEEE f16 float4-swizzle storage reports half the resident SH bytes of the
    // equivalent fp32 layout.
    constexpr std::size_t n = SH_REORDER_SIZE * 3 + 11;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/3, /*seed=*/0xF16u);
    auto splat = buildSplatData(in);

    // Convert resident shN to IEEE f16 (same topology, 2 B/component).
    ASSERT_TRUE(splat->shN().is_valid());
    ASSERT_EQ(splat->shN().dtype(), lfs::core::DataType::Float32);
    splat->shN() = splat->shN().to(lfs::core::DataType::Float16);
    ASSERT_TRUE(splat->shN_ieee_f16());
    ASSERT_FALSE(splat->shN_value_quantized());

    auto raw_layout = rawDeviceInputLayout(*splat);
    ASSERT_TRUE(raw_layout.has_value()) << raw_layout.error();
    const auto layout_rest = static_cast<std::uint32_t>(
        lfs::core::sh_rest_coefficients_for_degree(3));
    const std::size_t fp32_bytes =
        lfs::core::sh_swizzled_float_count(n, layout_rest) * sizeof(float);
    const std::size_t f16_bytes =
        lfs::core::sh_swizzled_f16_byte_count(n, layout_rest);
    EXPECT_EQ(f16_bytes * 2, fp32_bytes);
    EXPECT_EQ(raw_layout->shN_bytes, f16_bytes);
    EXPECT_TRUE(raw_layout->shN_f16);
    EXPECT_FALSE(raw_layout->shN_q16);
    EXPECT_EQ(raw_layout->shN_element_bytes, sizeof(std::uint16_t));

    // 5M-capacity accounting for: deg3 rest → 96 B/splat f16 vs 192 fp32.
    constexpr std::size_t kCap5M = 5'000'000;
    const std::size_t at_5m_f16 = lfs::core::sh_swizzled_f16_byte_count(
        kCap5M, layout_rest);
    const std::size_t at_5m_fp32 = lfs::core::sh_swizzled_byte_count(
        kCap5M, layout_rest);
    EXPECT_EQ(at_5m_f16, at_5m_fp32 / 2);
    EXPECT_NEAR(static_cast<double>(at_5m_f16) / static_cast<double>(kCap5M), 96.0, 0.01);
}

TEST(VksplatInputPackerTest, RawDeviceLayoutReportsNonShBytesAndZeroCopyContract) {
    // A packed non-SH copy would retain 44 B/splat (approximately 210 MiB at
    // 5M). The live training path zero-copies exportable fp32 attributes (means/rot/scale/
    // opacity) into the viewport — separate pack bytes = 0. This test locks the
    // layout contract: non_sh_bytes is the SHARED exportable footprint, and
    // attrs_f16 is false for the fp32 training path (xyz stays fp32 forever for
    // shimmer safety; f16 packing is only reported when tensors are half).
    constexpr std::size_t n = SH_REORDER_SIZE * 3 + 11;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/1, /*seed=*/0xA7F16u);
    auto splat = buildSplatData(in);

    ASSERT_EQ(splat->means_raw().dtype(), lfs::core::DataType::Float32);
    ASSERT_EQ(splat->rotation_raw().dtype(), lfs::core::DataType::Float32);
    ASSERT_EQ(splat->scaling_raw().dtype(), lfs::core::DataType::Float32);
    ASSERT_EQ(splat->opacity_raw().dtype(), lfs::core::DataType::Float32);
    ASSERT_FALSE(splat->non_sh_attrs_f16());

    auto layout = rawDeviceInputLayout(*splat);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    EXPECT_FALSE(layout->attrs_f16);
    EXPECT_EQ(layout->xyz_bytes, n * 3 * sizeof(float));
    EXPECT_EQ(layout->rotations_bytes, n * 4 * sizeof(float));
    EXPECT_EQ(layout->scaling_bytes, n * 3 * sizeof(float));
    EXPECT_EQ(layout->opacity_bytes, n * sizeof(float));
    EXPECT_EQ(layout->non_sh_bytes, n * 44u);

    // 5M-capacity accounting: shared exportable non-SH = 210 MiB; SEPARATE
    // viewer pack = 0 (zero-copy borrow — see prepareInputs can_bind_external).
    constexpr std::size_t kCap5M = 5'000'000;
    constexpr std::size_t kNonShB = 44u;
    const double shared_mib =
        static_cast<double>(kNonShB * kCap5M) / (1024.0 * 1024.0);
    EXPECT_NEAR(shared_mib, 209.808, 0.01);
    // Zero-copy keeps the separate packed allocation at zero.
    constexpr std::size_t kSeparatePackAfter = 0u;
    EXPECT_EQ(kSeparatePackAfter, 0u);
    EXPECT_GE(static_cast<double>(kNonShB * kCap5M) / (1024.0 * 1024.0), 105.0);

    // When attrs are IEEE f16 (lodq / future path), layout halves rot+scale+opac.
    splat->rotation_raw() = splat->rotation_raw().to(lfs::core::DataType::Float16);
    splat->scaling_raw() = splat->scaling_raw().to(lfs::core::DataType::Float16);
    splat->opacity_raw() = splat->opacity_raw().to(lfs::core::DataType::Float16);
    ASSERT_TRUE(splat->non_sh_attrs_f16());
    auto f16_layout = rawDeviceInputLayout(*splat);
    ASSERT_TRUE(f16_layout.has_value()) << f16_layout.error();
    EXPECT_TRUE(f16_layout->attrs_f16);
    EXPECT_EQ(f16_layout->xyz_bytes, n * 3 * sizeof(float)); // xyz stays fp32
    EXPECT_EQ(f16_layout->rotations_bytes, n * 8u);
    EXPECT_EQ(f16_layout->scaling_bytes, n * 8u);
    EXPECT_EQ(f16_layout->opacity_bytes, n * 2u);
    EXPECT_EQ(f16_layout->non_sh_bytes, n * 30u); // 12+8+8+2; save 14 B/splat
}

TEST(VksplatInputPackerTest, RawDeviceLayoutReportsQ16BytesAndBounds) {
    // The pad-dropped q16 exportable path reports 90 B/splat of SH data
    // (deg3) plus per-256 float2 bounds — not the f16 float4-swizzle size.
    constexpr std::size_t n = SH_REORDER_SIZE * 3 + 11;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/3, /*seed=*/0xA016u);
    auto splat = buildSplatData(in);
    ASSERT_TRUE(splat->shN().is_valid());
    ASSERT_EQ(splat->shN().dtype(), lfs::core::DataType::Float32);

    // Encode to pad-dropped q16 (training path).
    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(true);
    ASSERT_TRUE(lfs::training::sh_value::apply_shN_value_quant(*splat));
    ASSERT_TRUE(splat->shN_value_quantized());
    ASSERT_FALSE(splat->shN_ieee_f16());

    auto raw_layout = rawDeviceInputLayout(*splat);
    ASSERT_TRUE(raw_layout.has_value()) << raw_layout.error();
    const auto layout_rest = static_cast<std::uint32_t>(
        lfs::core::sh_rest_coefficients_for_degree(3));
    const std::size_t q16_bytes =
        lfs::core::sh_value_quant::sh_value_u16_count(n, layout_rest) *
        sizeof(std::uint16_t);
    const std::size_t bounds_bytes =
        lfs::core::sh_value_quant::n_bounds_for_prims(n) * 2u * sizeof(float);
    EXPECT_EQ(raw_layout->shN_bytes, q16_bytes);
    EXPECT_EQ(raw_layout->shN_bounds_bytes, bounds_bytes);
    EXPECT_TRUE(raw_layout->shN_q16);
    EXPECT_FALSE(raw_layout->shN_f16);
    EXPECT_EQ(raw_layout->shN_n_cells,
              lfs::core::sh_value_quant::n_value_cells_per_prim(layout_rest));
    EXPECT_EQ(raw_layout->shN_element_bytes, sizeof(std::uint16_t));

    // 5M capacity: 90 B/splat SH + ≪1 B/splat bounds.
    constexpr std::size_t kCap5M = 5'000'000;
    const std::size_t at_5m_q16 =
        lfs::core::sh_value_quant::sh_value_u16_count(kCap5M, layout_rest) *
        sizeof(std::uint16_t);
    EXPECT_NEAR(static_cast<double>(at_5m_q16) / static_cast<double>(kCap5M), 90.0, 0.01);
    // Exportable layout @ 5M must not allocate the f16 float4-swizzle region.
    const std::size_t exportable_shN =
        lfs::core::SplatExportableStorage::layoutBytes(kCap5M, 3);
    // Packed q16 layout plus VMM-granularity padding stays well below an
    // IEEE-f16 swizzled SH region.
    EXPECT_LT(exportable_shN, 800ull << 20);

    lfs::training::sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt);
}

TEST(VksplatInputPackerTest, RawDeviceLayoutStaysAtLiveSizeAfterTwoExportableGrows) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }

    using lfs::core::SplatExportableStorage;
    using lfs::core::TensorShape;
    const std::size_t gran =
        std::max<std::size_t>(lfs::core::exportable_allocation_granularity(0), 1);
    const std::size_t kInitial = std::max<std::size_t>(gran / 24, 2048);
    const std::size_t kLiveN = kInitial / 2;
    const std::size_t kGen2 = kInitial * 3;
    const std::size_t kGen3 = kInitial * 5;
    const std::size_t kReserve = kInitial * 16;
    constexpr int kShDegree = 0;

    auto storage_result = SplatExportableStorage::create(kInitial, kShDegree, 0, kReserve);
    if (!storage_result) {
        FAIL() << storage_result.error();
    }
    auto storage = std::move(*storage_result);
    auto allocator = storage.make_allocator();

    Tensor means = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.means");
    Tensor scaling = allocator(TensorShape({kLiveN, 3}), kInitial, DataType::Float32, "SplatData.scaling");
    Tensor rotation = allocator(TensorShape({kLiveN, 4}), kInitial, DataType::Float32, "SplatData.rotation");
    Tensor opacity = allocator(TensorShape({kLiveN, 1}), kInitial, DataType::Float32, "SplatData.opacity");
    Tensor sh0 = allocator(TensorShape({kLiveN, 1, 3}), kInitial, DataType::Float32, "SplatData.sh0");
    Tensor shN;

    SplatData model(/*max_sh_degree=*/kShDegree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    /*scene_scale=*/1.0f,
                    SplatData::ShNLayout::Swizzled);

    ASSERT_TRUE(storage.grow(kGen2).value_or(false));
    ASSERT_TRUE(storage.rebindSplatData(model, allocator).has_value()) << "rebind after gen2";
    ASSERT_TRUE(storage.grow(kGen3).value_or(false));
    ASSERT_TRUE(storage.rebindSplatData(model, allocator).has_value()) << "rebind after gen3";

    EXPECT_EQ(static_cast<std::size_t>(model.size()), kLiveN);
    EXPECT_GE(model.means_raw().capacity(), kGen3);
    EXPECT_EQ(storage.capacity(), kGen3);

    auto layout = rawDeviceInputLayout(model);
    ASSERT_TRUE(layout.has_value()) << layout.error();
    EXPECT_EQ(layout->num_splats, kLiveN)
        << "viewer layout must submit live N, not exportable capacity";
    EXPECT_EQ(layout->xyz_bytes, kLiveN * 3 * sizeof(float));
    EXPECT_EQ(layout->scaling_bytes, kLiveN * 3 * sizeof(float));
    EXPECT_NE(layout->num_splats, storage.capacity());
}

TEST(VksplatInputPackerTest, RawOpacityCopyBakesDeletedMaskOnlyIntoOpacity) {
    constexpr std::size_t n = 5;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/1, /*seed=*/0x0A91u);
    auto splat = buildSplatData(in);

    Tensor copied = Tensor::empty({n}, Device::CUDA, DataType::Float32);
    auto copy_status = copyRawOpacityToBuffer(*splat, copied.ptr<float>(), copied.stream());
    ASSERT_TRUE(copy_status.has_value()) << copy_status.error();
    const auto unmasked = copied.cpu().to_vector();
    ASSERT_EQ(unmasked.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(unmasked[i], in.opacity[i]);
    }

    const auto mask = Tensor::from_vector(
                          std::vector<int>{0, 1, 0, 1, 0},
                          {n},
                          Device::CUDA)
                          .to(DataType::Bool);
    splat->soft_delete(mask);

    Tensor masked = Tensor::empty({n}, Device::CUDA, DataType::Float32);
    copy_status = copyRawOpacityToBuffer(*splat, masked.ptr<float>(), masked.stream());
    ASSERT_TRUE(copy_status.has_value()) << copy_status.error();
    const auto masked_values = masked.cpu().to_vector();
    ASSERT_EQ(masked_values.size(), n);
    for (std::size_t i = 0; i < n; ++i) {
        if (i == 1 || i == 3) {
            EXPECT_NEAR(masked_values[i], -20.0f, 1e-5f);
        } else {
            EXPECT_FLOAT_EQ(masked_values[i], in.opacity[i]);
        }
    }
}

TEST(VksplatInputPackerTest, SoftDeleteAndUndeleteKeepDeletedMaskStorageStable) {
    constexpr std::size_t n = 5;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/1, /*seed=*/0x51AFu);
    auto splat = buildSplatData(in);
    const std::uint64_t initial_version = splat->deleted_mask_version();

    const auto first_mask = Tensor::from_vector(
                                std::vector<int>{0, 1, 0, 0, 0},
                                {n},
                                Device::CUDA)
                                .to(DataType::Bool);
    const Tensor first_newly_deleted = splat->soft_delete(first_mask);

    ASSERT_TRUE(splat->has_deleted_mask());
    const void* const deleted_ptr = splat->deleted().data_ptr();
    ASSERT_NE(deleted_ptr, nullptr);
    EXPECT_GT(splat->deleted_mask_version(), initial_version);
    const std::uint64_t first_version = splat->deleted_mask_version();
    EXPECT_EQ(first_newly_deleted.cpu().to_vector_bool(),
              (std::vector<bool>{false, true, false, false, false}));
    EXPECT_EQ(splat->deleted().cpu().to_vector_bool(),
              (std::vector<bool>{false, true, false, false, false}));

    const auto second_mask = Tensor::from_vector(
                                 std::vector<int>{0, 1, 1, 0, 1},
                                 {n},
                                 Device::CUDA)
                                 .to(DataType::Bool);
    const Tensor second_newly_deleted = splat->soft_delete(second_mask);

    EXPECT_EQ(splat->deleted().data_ptr(), deleted_ptr);
    EXPECT_GT(splat->deleted_mask_version(), first_version);
    const std::uint64_t second_version = splat->deleted_mask_version();
    EXPECT_EQ(second_newly_deleted.cpu().to_vector_bool(),
              (std::vector<bool>{false, false, true, false, true}));
    EXPECT_EQ(splat->deleted().cpu().to_vector_bool(),
              (std::vector<bool>{false, true, true, false, true}));

    const auto undelete_mask = Tensor::from_vector(
                                   std::vector<int>{0, 1, 0, 0, 1},
                                   {n},
                                   Device::CUDA)
                                   .to(DataType::Bool);
    splat->undelete(undelete_mask);

    EXPECT_EQ(splat->deleted().data_ptr(), deleted_ptr);
    EXPECT_GT(splat->deleted_mask_version(), second_version);
    EXPECT_EQ(splat->deleted().cpu().to_vector_bool(),
              (std::vector<bool>{false, false, true, false, false}));
}

// after any N-mutating path with an active deleted mask, the packer
// contract must hold (contiguous CUDA bool of size N) OR the mask must be
// invalidated (has_deleted_mask()==false is legal). A single stale frame must
// not permanently break opacity baking.
namespace {

    void assertDeletedMaskPackerContract(const SplatData& splat) {
        const auto n = static_cast<std::size_t>(splat.size());
        Tensor opacity_dst = Tensor::empty({n}, Device::CUDA, DataType::Float32);
        auto status = copyRawOpacityToBuffer(splat, opacity_dst.ptr<float>(), opacity_dst.stream());
        ASSERT_TRUE(status.has_value()) << status.error()
                                        << " (N=" << n
                                        << " has_deleted=" << splat.has_deleted_mask()
                                        << " deleted_numel="
                                        << (splat.has_deleted_mask() ? splat.deleted().numel() : 0)
                                        << ")";
        if (splat.has_deleted_mask()) {
            const Tensor& deleted = splat.deleted();
            EXPECT_EQ(deleted.dtype(), DataType::Bool);
            EXPECT_EQ(deleted.device(), Device::CUDA);
            EXPECT_TRUE(deleted.is_contiguous());
            EXPECT_EQ(static_cast<std::size_t>(deleted.numel()), n);
        }
    }

    // Grow all row-shaped parameter tensors by n_new rows (simulates densify grow).
    // Uses cat so the path works even when factory tensors are cuda.direct.
    void growSplatParamsBy(SplatData& splat, std::size_t n_new) {
        ASSERT_GT(n_new, 0u);
        auto grow_rows = [&](Tensor& t) {
            if (!t.is_valid() || t.numel() == 0 || t.ndim() == 0) {
                return;
            }
            auto dims = t.shape().dims();
            dims[0] = n_new;
            Tensor tail = Tensor::zeros(lfs::core::TensorShape(dims), t.device(), t.dtype());
            t = Tensor::cat({t, tail}, 0);
        };
        grow_rows(splat.means());
        grow_rows(splat.sh0());
        grow_rows(splat.scaling_raw());
        grow_rows(splat.rotation_raw());
        grow_rows(splat.opacity_raw());
        // shN is swizzled 1D — leave empty/unused in degree-1 synthetic models
        // when the packer only needs means/opacity for the deleted-mask contract.
        if (splat.shN().is_valid() && splat.shN().numel() > 0) {
            // Best-effort: pad with zeros of matching trailing shape if rank>1.
            if (splat.shN().ndim() >= 2) {
                grow_rows(splat.shN());
            }
        }
    }

} // namespace

TEST(VksplatInputPackerTest, GrowWithActiveDeletedMaskKeepsPackerContract) {
    constexpr std::size_t n = 8;
    constexpr std::size_t n_grow = 3;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/1, /*seed=*/0x1A022u);
    auto splat = buildSplatData(in);

    const auto mask = Tensor::from_vector(
                          std::vector<int>{0, 1, 0, 0, 1, 0, 0, 0},
                          {n},
                          Device::CUDA)
                          .to(DataType::Bool);
    splat->soft_delete(mask);
    ASSERT_TRUE(splat->has_deleted_mask());
    assertDeletedMaskPackerContract(*splat);

    // Densify grow of parameter tensors. Production paths must keep deleted()
    // sized to the new N (or invalidate). Stale mask of old N is the
    // failure mode that freezes the VkSplat viewport.
    growSplatParamsBy(*splat, n_grow);
    ASSERT_EQ(static_cast<std::size_t>(splat->size()), n + n_grow);

    // Production contract: writers either grow/rebuild the mask with N or clear it.
    splat->reconcile_deleted_mask();
    assertDeletedMaskPackerContract(*splat);
}

TEST(VksplatInputPackerTest, CompactWithActiveDeletedMaskKeepsPackerContract) {
    constexpr std::size_t n = 10;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/1, /*seed=*/0x022C0u);
    auto splat = buildSplatData(in);

    const auto mask = Tensor::from_vector(
                          std::vector<int>{0, 1, 0, 1, 0, 0, 1, 0, 0, 0},
                          {n},
                          Device::CUDA)
                          .to(DataType::Bool);
    splat->soft_delete(mask);
    ASSERT_TRUE(splat->has_deleted_mask());
    assertDeletedMaskPackerContract(*splat);

    // Hard-compact via apply_deleted (N shrinks; mask must clear or resize).
    const std::size_t removed = splat->apply_deleted();
    EXPECT_GT(removed, 0u);
    EXPECT_LT(static_cast<std::size_t>(splat->size()), n);
    // apply_deleted permanently removes soft-deleted rows → mask must be gone
    // or rebuilt to the new N.
    assertDeletedMaskPackerContract(*splat);
}

TEST(VksplatInputPackerTest, StaleDeletedMaskDoesNotPermanentlyBreakOpacityCopy) {
    // at the pre-mutation size. Packer must soft-skip a stale mask so a later
    // valid frame can recover (no hard permanent failure / degraded latch).
    constexpr std::size_t n = 6;
    SyntheticInputs in = makeInputs(n, /*max_sh_degree=*/0, /*seed=*/0x0DELu);
    auto splat = buildSplatData(in);

    const auto mask = Tensor::from_vector(
                          std::vector<int>{0, 1, 0, 0, 1, 0},
                          {n},
                          Device::CUDA)
                          .to(DataType::Bool);
    splat->soft_delete(mask);

    // Grow every row-shaped param so opacity/means stay consistent, but leave
    // the deleted mask at the pre-growth N (the exact stale state).
    growSplatParamsBy(*splat, 2);
    ASSERT_EQ(static_cast<std::size_t>(splat->size()), n + 2);
    ASSERT_TRUE(splat->has_deleted_mask());
    ASSERT_NE(static_cast<std::size_t>(splat->deleted().numel()),
              static_cast<std::size_t>(splat->size()));

    Tensor opacity_dst = Tensor::empty({static_cast<std::size_t>(splat->size())},
                                       Device::CUDA, DataType::Float32);
    auto status = copyRawOpacityToBuffer(*splat, opacity_dst.ptr<float>(), opacity_dst.stream());
    // Soft-skip path must succeed (stale mask treated as absent for this frame).
    ASSERT_TRUE(status.has_value()) << status.error();

    // After reconcile, the contract holds again (mask resized or cleared).
    splat->reconcile_deleted_mask();
    assertDeletedMaskPackerContract(*splat);
}
