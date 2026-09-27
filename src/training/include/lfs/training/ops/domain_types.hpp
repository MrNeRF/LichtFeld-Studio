/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Relocated unchanged from the fast rasterizer config. Values are the ones
// the fused backward kernels already switch on.
enum class DensificationType : int {
    None = 0,
    MCMC = 1,
    MRNF = 2
};

namespace lfs::gpu_ops {

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

} // namespace lfs::gpu_ops
