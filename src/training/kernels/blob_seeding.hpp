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

    // SweepAccepted is 1 when enough covisible views support SweepDepth, else 0.
    enum SweepField : int { SweepDepth = 0,
                            SweepAccepted,
                            SweepFieldCount };

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
    // Scratch buffers reused across detect_peaks calls; growing them per call fragments the memory pool
    // because every call also allocates the peaks and bitmap it returns.
    struct DetectionWorkspace {
        lfs::core::Tensor rgb, rows, a, b, c, mask, peaks, counters, bitmap;
    };

    // The returned peaks and bitmap are views into the workspace, valid until its next use.
    LFS_CUDA_API ViewPeaks detect_peaks(const lfs::core::Tensor& image, int view, DetectionWorkspace& workspace);

    // Averages factor x factor pixel blocks of a CUDA [C, H, W] uint8 or float32 image (C >= 3) into a
    // float32 [3, H / factor, W / factor] image in [0, 1], stored in workspace.rgb.
    LFS_CUDA_API lfs::core::Tensor downsample_rgb(const lfs::core::Tensor& image, int factor,
                                                  DetectionWorkspace& workspace);

    // For every peak, sweeps depth along its pixel ray and counts the neighbour views whose
    // peak bitmap of the same polarity is hit, minus the hits expected by chance. views and
    // neighbors ([V, K] int32, -1 padded) are device arrays. Returns [P, SweepFieldCount].
    LFS_CUDA_API lfs::core::Tensor sweep_peaks(const lfs::core::Tensor& peaks,
                                               const SweepView* views,
                                               const lfs::core::Tensor& neighbors);

} // namespace lfs::training::kernels::blob_seeding
