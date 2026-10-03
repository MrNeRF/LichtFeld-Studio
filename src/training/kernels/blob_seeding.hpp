/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>

namespace lfs::training::kernels::blob_seeding {

    enum PeakField : int { PeakX = 0,
                           PeakY,
                           PeakView,
                           PeakPolarity,
                           PeakR,
                           PeakG,
                           PeakB,
                           PeakFieldCount };

    enum SweepField : int { SweepDepth = 0,
                            SweepSupport,
                            SweepConfidence,
                            SweepFieldCount };

    enum class Confidence : int { None = 0,
                                  Low = 1,
                                  Ambiguous = 2,
                                  Unique = 3 };

    struct ViewPeaks {
        lfs::core::Tensor peaks;  // [count, PeakFieldCount] float32
        lfs::core::Tensor bitmap; // ceil(H*W/16) uint32 words, 2 bits per pixel, dilated by one pixel
        std::array<float, 2> density{};
    };

    struct SweepView {
        float R[9]; // world to camera, row major
        float t[3];
        float fx, fy, cx, cy;
        int width, height;
        const uint32_t* bitmap;
        float density[2];
        float z_min, z_max;
    };

    // Detects compact bright and dark blobs (local contrast maxima over a 5x5 window, contrast =
    // 0-255 luminance against its 7x7 grey opening or closing) in a CUDA float32 [C, H, W] image
    // with C >= 3 and values in [0, 1].
    LFS_CUDA_API ViewPeaks detect_peaks(const lfs::core::Tensor& image, int view);

    // For every peak, sweeps depth along its pixel ray and counts the neighbour views whose
    // peak bitmap of the same polarity is hit, minus the hits expected by chance. views and
    // neighbors ([V, K] int32, -1 padded) are device arrays. Returns [P, SweepFieldCount].
    LFS_CUDA_API lfs::core::Tensor sweep_peaks(const lfs::core::Tensor& peaks,
                                               const SweepView* views,
                                               const lfs::core::Tensor& neighbors);

} // namespace lfs::training::kernels::blob_seeding
