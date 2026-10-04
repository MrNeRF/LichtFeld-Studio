/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor_image.hpp"
#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/types.hpp"
#include <array>

namespace lfs::training::kernels {
    /// SegmentAndIgnore band bounds for Float32 masks in [0,1].
    /// Midpoints between adjacent eight-bit levels preserve their classification
    /// despite floating-point error in the normalization by 255.
    inline constexpr float kMaskKeepMin = 250.5f / 255.0f;
    inline constexpr float kMaskSegmentMin = 127.5f / 255.0f;
} // namespace lfs::training::kernels

namespace lfs::gpu_ops {
    struct MeshMaskCamera {
        std::array<float, 12> world_to_camera{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f};
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        int width = 0;
        int height = 0;
    };

    struct MaskOps {
        void (*photometric_weight)(In mask, In roi, Out weight, MaskPhotoMode);
        void (*opacity_penalty)(In alpha, In mask, In roi, Out grad_alpha,
                                Out reduction_temp, Out loss, MaskOpacityMode, float power, float scale);
        void (*alpha_consistency)(In alpha, In mask, In roi, Out grad_alpha,
                                  Out reduction_temp, Out loss, float weight);
        core::Tensor (*mesh_coverage)(In vertices, In indices, const MeshMaskCamera&,
                                      In samples, const core::UndistortParams*, float z_near) = nullptr;
    };
} // namespace lfs::gpu_ops
