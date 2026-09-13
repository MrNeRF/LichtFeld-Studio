/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/alloc_counter.hpp"
#include "core/tensor.hpp"
#include "mask_loss_reference.hpp"
#include "training/losses/mask_loss.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::losses;

namespace {

    Tensor u8_mask_from(const std::vector<uint8_t>& data, size_t H, size_t W) {
        auto t = Tensor::empty({H, W}, Device::CPU, DataType::UInt8);
        std::copy(data.begin(), data.end(), t.ptr<uint8_t>());
        return t.to(Device::GPU);
    }

    Tensor f32_from(const std::vector<float>& data, size_t H, size_t W) {
        return Tensor::from_vector(data, {H, W}, Device::GPU);
    }

    /// Use the same CUDA Tensor normalization as the pipelined loader.
    Tensor normalized_mask_from(const std::vector<uint8_t>& data, size_t H, size_t W) {
        return u8_mask_from(data, H, W).to(DataType::Float32) / 255.0f;
    }

    /// Reference: SegmentAndIgnore photometric remap (old trainer chain) + ROI compose.
    Tensor ref_photometric_sai(const Tensor& mask_u8, const Tensor& roi) {
        auto m = mask_u8;
        m = m.masked_fill(m <= 250, 0);
        m = m.masked_fill(m > 250, 255);
        return test_reference::compose_pixel_loss_weights_reference(m, roi);
    }

    /// Reference: Segment opacity penalty (old trainer chain).
    MaskOpacityPenalty ref_opacity_segment(
        const Tensor& alpha,
        const Tensor& mask,
        const Tensor& roi,
        float power,
        float scale) {
        const auto mask_f =
            (mask.dtype() == DataType::UInt8 || mask.dtype() == DataType::Bool)
                ? mask.gt(0).to(DataType::Float32)
                : mask;
        const auto bg = Tensor::full(mask_f.shape(), 1.0f, mask_f.device()) - mask_f;
        const auto pen_w = bg.pow(power);
        return test_reference::compute_mask_opacity_penalty_reference(alpha, pen_w, roi, scale);
    }

    /// Reference: SegmentAndIgnore opacity penalty (old trainer chain).
    MaskOpacityPenalty ref_opacity_sai(
        const Tensor& alpha,
        const Tensor& mask,
        const Tensor& roi,
        float power,
        float scale) {
        auto m = mask;
        m = m.masked_fill(m < 128, 255);
        m = m.masked_fill(m >= 128 && m <= 250, 0);
        m = m.masked_fill(m > 250, 255);
        const auto mask_f =
            (m.dtype() == DataType::UInt8 || m.dtype() == DataType::Bool)
                ? m.gt(0).to(DataType::Float32)
                : m;
        const auto bg = Tensor::full(mask_f.shape(), 1.0f, mask_f.device()) - mask_f;
        const auto pen_w = bg.pow(power);
        return test_reference::compute_mask_opacity_penalty_reference(alpha, pen_w, roi, scale);
    }

    /// Reference: AlphaConsistent (old trainer chain).
    MaskOpacityPenalty ref_alpha_consistent(
        const Tensor& alpha,
        const Tensor& mask,
        const Tensor& roi,
        float weight) {
        const auto mask_f =
            (mask.dtype() == DataType::UInt8 || mask.dtype() == DataType::Bool)
                ? mask.gt(0).to(DataType::Float32)
                : mask;
        auto err = (alpha - mask_f).abs();
        auto grad = (alpha - mask_f).sign();
        if (roi.is_valid()) {
            err = err * roi;
            grad = grad * roi;
        }
        return MaskOpacityPenalty{
            .loss = err.mean() * weight,
            .grad_alpha = grad * (weight / static_cast<float>(alpha.numel()))};
    }

    void expect_near_vec(const Tensor& a, const Tensor& b, float tol, const char* tag) {
        ASSERT_EQ(a.numel(), b.numel()) << tag;
        const auto ac = a.to(Device::CPU);
        const auto bc = b.to(Device::CPU);
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        float max_abs = 0.f;
        for (size_t i = 0; i < a.numel(); ++i) {
            max_abs = std::max(max_abs, std::fabs(pa[i] - pb[i]));
        }
        EXPECT_LE(max_abs, tol) << tag << " max_abs=" << max_abs;
    }

} // namespace

TEST(MaskPreprocessFusionTest, SegmentAndIgnorePhotometricMatchesReference) {
    // Bands: 0 ignore-low, 100 ignore, 180 segment-BG, 255 keep
    const auto mask = u8_mask_from({0, 100, 180, 255, 200, 40, 255, 128}, 2, 4);
    const auto roi = f32_from({1.f, 0.5f, 0.25f, 0.f, 1.f, 1.f, 0.1f, 0.75f}, 2, 4);

    const auto ref = ref_photometric_sai(mask, roi);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_photometric_mask_weight(ws, mask, roi, /*segment_and_ignore=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    expect_near_vec(fused, ref, 1e-6f, "photo_sai");
}

TEST(MaskPreprocessFusionTest, SegmentAndIgnorePhotometricMatchesReferenceForNormalizedMask) {
    // Same bands as above, but handed over the way the pipelined loader does:
    // Float32 in [0,1]. Must classify identically to the raw eight-bit mask.
    const std::vector<uint8_t> bands{0, 100, 180, 255, 200, 40, 255, 128};
    const auto roi = f32_from({1.f, 0.5f, 0.25f, 0.f, 1.f, 1.f, 0.1f, 0.75f}, 2, 4);

    const auto ref = ref_photometric_sai(u8_mask_from(bands, 2, 4), roi);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_photometric_mask_weight(
        ws, normalized_mask_from(bands, 2, 4), roi, /*segment_and_ignore=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    expect_near_vec(fused, ref, 1e-6f, "photo_sai_normalized");
}

TEST(MaskPreprocessFusionTest, SegmentAndIgnoreBandLevelsMatchForBothMaskTypes) {
    // Cover all eight-bit levels, including 127/128 and 250/251 band edges.
    constexpr size_t n = 256;
    std::vector<uint8_t> bands(n);
    std::vector<float> photo_weights(n);
    std::vector<float> background_weights(n);
    for (size_t i = 0; i < n; ++i) {
        bands[i] = static_cast<uint8_t>(i);
        photo_weights[i] = i > 250 ? 1.0f : 0.0f;
        background_weights[i] = i >= 128 && i <= 250 ? 1.0f : 0.0f;
    }
    const auto expected_photo = f32_from(photo_weights, 1, n);
    const auto expected_background = f32_from(background_weights, 1, n);
    const auto alpha = Tensor::full({size_t{1}, n}, 1.0f, Device::CUDA);
    const Tensor no_roi{};

    MaskPreprocessWorkspace ws;
    for (const auto& mask : {u8_mask_from(bands, 1, n), normalized_mask_from(bands, 1, n)}) {
        const auto photo = fuse_photometric_mask_weight(ws, mask, no_roi, /*segment_and_ignore=*/true);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        expect_near_vec(photo, expected_photo, 1e-6f, "photo_levels");

        // power = 1 and scale = n make grad_alpha the per-pixel background weight.
        const auto penalty = fuse_mask_opacity_penalty(
            ws, alpha, mask, no_roi, /*power=*/1.0f, /*scale=*/static_cast<float>(n),
            /*segment_and_ignore=*/true);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        expect_near_vec(penalty.grad_alpha, expected_background, 1e-6f, "opacity_levels");
    }
}

TEST(MaskPreprocessFusionTest, SegmentOpacityPenaltyMatchesReference) {
    const auto alpha = f32_from({0.2f, 0.4f, 0.6f, 0.8f}, 2, 2);
    // UInt8 binary-ish mask (object=1, bg=0)
    const auto mask = u8_mask_from({1, 0, 1, 0}, 2, 2);
    const auto roi = f32_from({1.f, 0.5f, 0.25f, 0.f}, 2, 2);
    constexpr float power = 2.0f;
    constexpr float scale = 3.0f;

    const auto ref = ref_opacity_segment(alpha, mask, roi, power, scale);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_mask_opacity_penalty(
        ws, alpha, mask, roi, power, scale, /*segment_and_ignore=*/false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(fused.loss.item<float>(), ref.loss.item<float>(), 1e-5f);
    expect_near_vec(fused.grad_alpha, ref.grad_alpha, 1e-5f, "opacity_seg");
}

TEST(MaskPreprocessFusionTest, SegmentAndIgnoreOpacityPenaltyMatchesReference) {
    const auto alpha = f32_from({0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f}, 2, 3);
    // 40 ignore, 180 segment-BG, 255 keep
    const auto mask = u8_mask_from({40, 180, 255, 128, 200, 10}, 2, 3);
    const auto roi = f32_from({1.f, 1.f, 0.5f, 0.f, 0.25f, 1.f}, 2, 3);
    constexpr float power = 2.0f;
    constexpr float scale = 1.5f;

    const auto ref = ref_opacity_sai(alpha, mask, roi, power, scale);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_mask_opacity_penalty(
        ws, alpha, mask, roi, power, scale, /*segment_and_ignore=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(fused.loss.item<float>(), ref.loss.item<float>(), 1e-5f);
    expect_near_vec(fused.grad_alpha, ref.grad_alpha, 1e-5f, "opacity_sai");
}

TEST(MaskPreprocessFusionTest, SegmentAndIgnoreOpacityPenaltyMatchesReferenceForNormalizedMask) {
    const auto alpha = f32_from({0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f}, 2, 3);
    const std::vector<uint8_t> bands{40, 180, 255, 128, 200, 10};
    const auto roi = f32_from({1.f, 1.f, 0.5f, 0.f, 0.25f, 1.f}, 2, 3);
    constexpr float power = 2.0f;
    constexpr float scale = 1.5f;

    const auto ref = ref_opacity_sai(alpha, u8_mask_from(bands, 2, 3), roi, power, scale);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_mask_opacity_penalty(
        ws, alpha, normalized_mask_from(bands, 2, 3), roi, power, scale,
        /*segment_and_ignore=*/true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(fused.loss.item<float>(), ref.loss.item<float>(), 1e-5f);
    expect_near_vec(fused.grad_alpha, ref.grad_alpha, 1e-5f, "opacity_sai_normalized");
}

TEST(MaskPreprocessFusionTest, SoftFloatOpacityPenaltyMatchesReference) {
    // Soft float masks: bg = 1-m, pow(bg, p) is continuous.
    const auto alpha = f32_from({0.5f, 0.5f, 0.5f, 0.5f}, 2, 2);
    const auto mask = f32_from({0.0f, 0.25f, 0.75f, 1.0f}, 2, 2);
    const Tensor roi{}; // none
    constexpr float power = 3.0f;
    constexpr float scale = 2.0f;

    const auto ref = ref_opacity_segment(alpha, mask, roi, power, scale);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_mask_opacity_penalty(
        ws, alpha, mask, roi, power, scale, /*segment_and_ignore=*/false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(fused.loss.item<float>(), ref.loss.item<float>(), 1e-5f);
    expect_near_vec(fused.grad_alpha, ref.grad_alpha, 1e-5f, "opacity_soft");
}

TEST(MaskPreprocessFusionTest, AlphaConsistentMatchesReference) {
    const auto alpha = f32_from({0.1f, 0.9f, 0.5f, 0.0f}, 2, 2);
    const auto mask = f32_from({0.0f, 1.0f, 0.25f, 0.5f}, 2, 2);
    const auto roi = f32_from({1.f, 0.5f, 0.f, 1.f}, 2, 2);
    constexpr float weight = 10.0f;

    const auto ref = ref_alpha_consistent(alpha, mask, roi, weight);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_alpha_consistent(ws, alpha, mask, roi, weight);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    EXPECT_NEAR(fused.loss.item<float>(), ref.loss.item<float>(), 1e-5f);
    expect_near_vec(fused.grad_alpha, ref.grad_alpha, 1e-5f, "alpha_cons");
}

TEST(MaskPreprocessFusionTest, SteadyStateRoiSegmentPathIsAllocationFree) {
    constexpr size_t H = 128;
    constexpr size_t W = 192;
    std::vector<uint8_t> mask_h(H * W);
    std::vector<float> alpha_h(H * W);
    std::vector<float> roi_h(H * W);
    for (size_t i = 0; i < H * W; ++i) {
        mask_h[i] = static_cast<uint8_t>((i % 3 == 0) ? 255 : ((i % 3 == 1) ? 180 : 40));
        alpha_h[i] = 0.1f + 0.001f * static_cast<float>(i % 97);
        roi_h[i] = (i % 5 == 0) ? 0.0f : 1.0f;
    }
    const auto mask = u8_mask_from(mask_h, H, W);
    const auto alpha = f32_from(alpha_h, H, W);
    const auto roi = f32_from(roi_h, H, W);

    MaskPreprocessWorkspace ws;
    // Warm: grow workspace + first launches.
    (void)fuse_photometric_mask_weight(ws, mask, roi, true);
    (void)fuse_mask_opacity_penalty(ws, alpha, mask, roi, 2.0f, 1.0f, true);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto snap = alloc_counter::snapshot();
    (void)fuse_photometric_mask_weight(ws, mask, roi, true);
    (void)fuse_mask_opacity_penalty(ws, alpha, mask, roi, 2.0f, 1.0f, true);
    (void)fuse_alpha_consistent(ws, alpha, mask, roi, 10.0f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto delta = alloc_counter::delta_since(snap);
    EXPECT_EQ(delta, 0u)
        << "steady ROI/segment mask preprocess must be allocation-free after warm";
}

TEST(MaskPreprocessFusionTest, BinaryGt0PhotoWithRoiMatchesCompose) {
    const auto mask = u8_mask_from({0, 1, 1, 0}, 2, 2);
    const auto roi = f32_from({1.f, 0.5f, 0.25f, 0.0f}, 2, 2);
    const auto ref = test_reference::compose_pixel_loss_weights_reference(mask, roi);

    MaskPreprocessWorkspace ws;
    const auto fused = fuse_photometric_mask_weight(ws, mask, roi, /*segment_and_ignore=*/false);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    expect_near_vec(fused, ref, 1e-6f, "photo_bin_roi");
}
