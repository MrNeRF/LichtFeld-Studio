/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <vector>

namespace lfs::training::losses {

    /// Grow-only workspace for ROI/segment mask preprocess (allocation-free steady state).
    struct MaskPreprocessWorkspace {
        lfs::core::Tensor photometric_weight; // [H,W] Float32
        lfs::core::Tensor grad_alpha;         // [H,W] Float32
        lfs::core::Tensor loss_scalar;        // [1] Float32
        lfs::core::Tensor reduce_temp;        // [1024] Float32
        size_t allocated_h = 0;
        size_t allocated_w = 0;

        void ensure_size(size_t H, size_t W);
    };

    struct MaskOpacityPenalty {
        lfs::core::Tensor loss;
        lfs::core::Tensor grad_alpha;
    };

    // -------------------------------------------------------------------------
    // Fused single-kernel preprocess path.
    // -------------------------------------------------------------------------

    /// Fuse SegmentAndIgnore photometric band remap (or BinaryGt0) + optional ROI
    /// into `ws.photometric_weight`. Returns a view of that buffer.
    /// When `user_mask` is invalid, returns `roi_weight` (or empty) without writes.
    [[nodiscard]] lfs::core::Tensor fuse_photometric_mask_weight(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& user_mask,
        const lfs::core::Tensor& roi_weight,
        bool segment_and_ignore);

    /// Fuse Segment / SegmentAndIgnore opacity-penalty band + pow + mean/grad.
    [[nodiscard]] MaskOpacityPenalty fuse_mask_opacity_penalty(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& mask_raw,
        const lfs::core::Tensor& roi_weight,
        float power,
        float scale,
        bool segment_and_ignore);

    /// Fuse AlphaConsistent abs/sign + mean + grad (weight typically 10.0f).
    [[nodiscard]] MaskOpacityPenalty fuse_alpha_consistent(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& roi_weight,
        float weight);

} // namespace lfs::training::losses
