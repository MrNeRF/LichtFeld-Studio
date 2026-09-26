/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <span>

namespace lfs::gpu_ops {

    enum class JointLayout { Rows,
                             SwizzledSH };

    struct AdamHyper {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-15f;
    };

    // Invalid bindings are absent. raw_scales and far_mask feed the per-splat
    // mean step, screen_share the scale hinge.
    struct AdamMasks {
        In frozen, crop_damping, raw_scales, far_mask, screen_share;
    };

    struct AdamModifiers {
        float frozen_lr_scale = 0.f;
        float cropbox_lr_scale = 1.f;
        float median_extent = 0.f;
        float r_min = 1.f;
        float r_max = 300.f;
        float screen_share_limit = 0.f;
        float screen_share_penalty = 0.f;
    };

    // One contiguous [primitives, attributes] parameter with its packed joint
    // moments and per-block bounds.
    struct JointStep {
        Out parameter, packed, bounds;
        In gradient;
        int primitives = 0;
        int attributes = 0;
        int bits = 0;
        float lr = 0.f;
        float bc1_rcp = 1.f;
        float bc2_sqrt_rcp = 1.f;
        bool apply_mean_step = false;
        bool apply_screen_share = false;
    };

    // value_bits is 0 for float32 values and 16 for half or block-quantized
    // values; block-quantized values bind value_bounds.
    struct ShStepParams {
        int primitives = 0;
        int layout_slots = 0;
        int active_bases = 0;
        int value_bits = 0;
        int value_cells = 0;
        float step_size = 0.f;
        float bc2_sqrt_rcp = 1.f;
    };

    // attributes_or_slots counts cells per row for Rows and float4 slots per
    // primitive for SwizzledSH.
    struct JointCodecParams {
        JointLayout layout = JointLayout::Rows;
        int primitives = 0;
        int attributes_or_slots = 0;
        int bits = 0;
    };

    struct AdamOps {
        void (*step)(
            const JointStep&, const AdamMasks&,
            const AdamHyper&, const AdamModifiers&);

        // One update over every present step. A step whose parameter binding is
        // invalid is absent.
        void (*step_batch)(
            std::span<const JointStep>, const AdamMasks&,
            const AdamHyper&, const AdamModifiers&);

        void (*step_sh)(
            Out parameter, Out packed, Out bounds, Out value_bounds, In gradient,
            const AdamMasks&, const AdamHyper&, const AdamModifiers&,
            const ShStepParams&);

        // Encodes zero moments at int64 indices. Block bounds may widen.
        void (*encode_zero)(
            Out packed, Out bounds, In indices, const JointCodecParams&);

        // Re-encodes rows appended by a gather of int64 source indices under
        // their destination block bounds.
        void (*transcode_gathered)(
            Out packed, In bounds, In indices,
            int old_primitives, int attributes, int bits);
    };

} // namespace lfs::gpu_ops
