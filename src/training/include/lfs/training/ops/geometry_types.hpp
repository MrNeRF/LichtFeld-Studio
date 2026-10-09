/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <vector>

namespace lfs::training::kernels {

    struct DepthAnchorCandidate {
        bool valid = false;
        float scale = 0.0f;
        float shift = 0.0f;
        float corr = 0.0f;
        int samples = 0;
    };

    // Per-camera alignment of the depth prior against sparse anchor points
    // (COLMAP / init point cloud), fitted once at startup. Keeps the target
    // depth absolute and multi-view consistent instead of chasing the render.
    struct DepthAnchor {
        bool valid = false;
        int model = 0; // 0 = disparity-space fit, 1 = depth-space fit
        float scale = 0.0f;
        float shift = 0.0f;
        float floor = 0.0f;
        float corr = 0.0f;
        int samples = 0;
        DepthAnchorCandidate disparity;
        DepthAnchorCandidate depth;
    };

    enum class DepthPriorType : int {
        Auto = 0,
        Disparity = 1,
        Depth = 2,
    };

    namespace depth_loss_slots {
        constexpr int kValid = 0;
        constexpr int kModel = 1; // 0 = disparity-space anchor, 1 = depth-space anchor
        constexpr int kScale = 2;
        constexpr int kShift = 3;
        constexpr int kFloor = 4;
        constexpr int kInvNorm = 5;
        constexpr int kSumAlpha = 6;
        constexpr int kCount = 7;
        constexpr int kMeanExpectedDepth = 8;
        constexpr int kSigmaP = 9;
        constexpr int kSlotCount = 10;
    } // namespace depth_loss_slots

    namespace normal_loss_slots {
        constexpr int kValid = 0;
        constexpr int kSumAlpha = 1;
        constexpr int kCount = 2;
        constexpr int kMeanCos = 3;
        constexpr int kInvNorm = 4;
        constexpr int kSlotCount = 6;
    } // namespace normal_loss_slots

    namespace normal_consistency_slots {
        constexpr int kValid = 0;
        constexpr int kSumAlpha = 1;
        constexpr int kCount = 2;
        constexpr int kMeanCos = 3;
        constexpr int kInvNorm = 4;
        constexpr int kSlotCount = 6;
    } // namespace normal_consistency_slots

    // One grid for the depth, normal, and consistency reductions.
    constexpr int kGeometryLossThreads = 256;
    constexpr size_t kGeometryLossMaxBlocks = 1024;
    constexpr int kDepthPrimaryStatCount = 3;
    constexpr int kNormalStatCount = 3;

    constexpr float kDepthLossFloorFraction = 0.05f;
    constexpr float kDepthLossMinAlpha = 1.0e-3f;
    constexpr float kDepthLossResidualScale = 2.0f;
    constexpr float kDepthLossTargetVarRidge = 1.5e-5f;
    constexpr float kDepthLossFlatPriorVar = 4.0f * kDepthLossTargetVarRidge;
    constexpr double kDepthLossMinVariance = 1.0e-20;
    constexpr int kMinAnchorSamples = 256;

    constexpr float kNormalLossMinAlpha = 1.0e-3f;
    constexpr float kNormalLossMinPriorNorm = 0.5f;
    constexpr float kNormalLossMinRenderNorm = 0.1f;
    constexpr float kNormalLossMinValidCount = 64.0f;
    constexpr float kNormalLossMinValidWeight = 16.0f;

    constexpr float kNormalConsistencyMinAlpha = 0.5f;
    constexpr float kNormalConsistencyMaxRelDepthJump = 0.05f;
    constexpr float kNormalConsistencyMinValidCount = 64.0f;
    constexpr float kNormalConsistencyMinValidWeight = 16.0f;

    [[nodiscard]] inline size_t geometry_loss_block_count(const size_t num_pixels) {
        const size_t blocks =
            (num_pixels + static_cast<size_t>(kGeometryLossThreads) - 1) /
            static_cast<size_t>(kGeometryLossThreads);
        return blocks < kGeometryLossMaxBlocks ? blocks : kGeometryLossMaxBlocks;
    }

    [[nodiscard]] inline size_t depth_loss_partial_count(const size_t num_pixels) {
        return static_cast<size_t>(depth_loss_slots::kSlotCount) +
               2 * static_cast<size_t>(kDepthPrimaryStatCount) * geometry_loss_block_count(num_pixels);
    }

    [[nodiscard]] inline size_t normal_loss_partial_count(const size_t num_pixels) {
        return static_cast<size_t>(normal_loss_slots::kSlotCount) +
               2 * static_cast<size_t>(kNormalStatCount) * geometry_loss_block_count(num_pixels);
    }

    [[nodiscard]] inline size_t normal_consistency_partial_count(const size_t num_pixels) {
        return static_cast<size_t>(normal_consistency_slots::kSlotCount) +
               2 * static_cast<size_t>(kNormalStatCount) * geometry_loss_block_count(num_pixels);
    }

} // namespace lfs::training::kernels

namespace lfs::gpu_ops {
    struct alignas(8) AnchorSample {
        float x, y;
    };
} // namespace lfs::gpu_ops

namespace lfs::training::kernels {
    struct DepthCameraProjection {
        int model = 0; // CameraModelType: pinhole=0, fisheye=2, equi=3, thin-prism=4
        float radial[6] = {};
        float tangential[2] = {};
        float thin_prism[4] = {};
    };

    // Robust affine fits over collected samples. Host work, safe on a worker pool.
    [[nodiscard]] DepthAnchor fit_depth_anchor_from_samples(
        const std::vector<lfs::gpu_ops::AnchorSample>& pairs);
} // namespace lfs::training::kernels
