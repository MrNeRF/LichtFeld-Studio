/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /// SegmentAndIgnore band bounds for Float32 masks in [0,1], as returned by
    /// the pipelined loader. UInt8 masks (including Camera::load_and_get_mask)
    /// still carry 0..255 samples and are normalized by the fused kernels.
    ///
    ///   value > 250       → keep    (photometric weight 1, no opacity penalty)
    ///   128 ≤ value ≤ 250 → segment (opacity penalty, no photometric weight)
    ///   value < 128       → ignore  (no loss at all)
    ///
    /// Midpoints between adjacent eight-bit levels preserve their classification
    /// despite floating-point error in the normalization by 255.
    inline constexpr float kMaskKeepMin = 250.5f / 255.0f;
    inline constexpr float kMaskSegmentMin = 127.5f / 255.0f;

    /// Photometric mask weight modes for the fused preprocess kernel.
    /// BinaryGt0: weight = (mask > 0)  — Segment / Ignore after binarize
    /// SegmentAndIgnore: weight = (mask > kMaskKeepMin) — keep only the "keep" band
    enum class MaskPhotoMode : int {
        BinaryGt0 = 0,
        SegmentAndIgnore = 1,
    };

    /// Opacity-penalty band modes.
    /// BinaryGt0: bg = 1 - mask_as_float (UInt8/Bool → 0/1; Float32 pass-through)
    /// SegmentAndIgnore: bg = 1 iff kMaskSegmentMin ≤ mask ≤ kMaskKeepMin
    /// (the Ignore band is FG for the penalty)
    enum class MaskOpacityMode : int {
        BinaryGt0 = 0,
        SegmentAndIgnore = 1,
    };

    /// Fuse SegmentAndIgnore / Segment / Ignore photometric remapping + optional ROI
    /// into a single float32 [H,W] weight map (allocation-free; writes into `out`).
    ///
    /// UInt8 samples are scaled by 1/255 before the band comparison, so both
    /// entry points share one value domain:
    /// BinaryGt0 — UInt8: nonzero → 1; Float32: pass-through.
    /// SegmentAndIgnore — value > kMaskKeepMin → 1.
    void launch_fuse_photometric_mask_weight_u8(
        const uint8_t* mask,
        const float* roi_weight, // nullable
        float* out,
        int H,
        int W,
        MaskPhotoMode mode,
        cudaStream_t stream = nullptr);

    void launch_fuse_photometric_mask_weight_f32(
        const float* mask,
        const float* roi_weight, // nullable
        float* out,
        int H,
        int W,
        MaskPhotoMode mode,
        cudaStream_t stream = nullptr);

    /// Fuse band remap + (1-mask)^power + opacity penalty loss/grad into one pass.
    /// Writes grad_alpha[i] = effective_weight[i] * (scale / n)
    /// and loss_out[0] = mean(alpha * effective_weight) * scale
    /// where effective_weight = penalty_weight * (roi or 1).
    /// Uses two-stage block reduce via `reduce_temp` (≥ min(num_blocks, 1024) floats).
    void launch_fuse_mask_opacity_penalty_u8(
        const float* alpha,
        const uint8_t* mask,
        const float* roi_weight, // nullable
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        int H,
        int W,
        float power,
        float scale,
        MaskOpacityMode mode,
        cudaStream_t stream = nullptr);

    void launch_fuse_mask_opacity_penalty_f32(
        const float* alpha,
        const float* mask,
        const float* roi_weight, // nullable
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        int H,
        int W,
        float power,
        float scale,
        MaskOpacityMode mode,
        cudaStream_t stream = nullptr);

    /// Fuse AlphaConsistent path: abs/sign of (alpha - mask_as_float), optional ROI,
    /// mean * weight → loss, grad_alpha = sign * (weight / n) * roi.
    void launch_fuse_alpha_consistent_u8(
        const float* alpha,
        const uint8_t* mask,
        const float* roi_weight, // nullable
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        int H,
        int W,
        float weight,
        cudaStream_t stream = nullptr);

    void launch_fuse_alpha_consistent_f32(
        const float* alpha,
        const float* mask,
        const float* roi_weight, // nullable
        float* grad_alpha,
        float* reduce_temp,
        float* loss_out,
        int H,
        int W,
        float weight,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
