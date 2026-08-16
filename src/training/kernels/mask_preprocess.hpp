/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    /// Photometric mask weight modes for the fused preprocess kernel.
    /// BinaryGt0: weight = (mask > 0)  — Segment / Ignore after binarize
    /// SegmentAndIgnore: weight = (mask > 250) — keep only the "keep" band
    enum class MaskPhotoMode : int {
        BinaryGt0 = 0,
        SegmentAndIgnore = 1,
    };

    /// Opacity-penalty band modes.
    /// BinaryGt0: bg = 1 - mask_as_float (UInt8/Bool → 0/1; Float32 pass-through)
    /// SegmentAndIgnore: bg = 1 iff 128 ≤ mask ≤ 250 (Ignore band is FG for penalty)
    enum class MaskOpacityMode : int {
        BinaryGt0 = 0,
        SegmentAndIgnore = 1,
    };

    /// Fuse SegmentAndIgnore / Segment / Ignore photometric remapping + optional ROI
    /// into a single float32 [H,W] weight map (allocation-free; writes into `out`).
    ///
    /// UInt8: nonzero → 1 (BinaryGt0) or value>250 → 1 (SegmentAndIgnore)
    /// Float32: same comparisons on the raw float value.
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
