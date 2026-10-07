/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "lfs/training/ops/types.hpp"

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
        uint64_t bitmap_offset; // Offset in uint32 words into the packed bitmap tensor.
        float density[2];
        float z_min, z_max;
    };

    // Detects compact bright and dark blobs (local contrast maxima over a 5x5 window, contrast =
    // 0-255 luminance against its 7x7 grey opening or closing) in a GPU float32 [C, H, W] image
    // with C >= 3 and values in [0, 1].
    // Scratch buffers reused across detect_peaks calls; growing them per call fragments the memory pool
    // because every call also allocates the peaks and bitmap it returns.
    struct DetectionWorkspace {
        lfs::core::Tensor rgb, rows, a, b, c, mask, peaks, counters, bitmap;
    };

    // The returned peaks and bitmap are views into the workspace, valid until its next use.
    ViewPeaks detect_peaks(const lfs::core::Tensor& image, int view, DetectionWorkspace& workspace);

    // Averages factor x factor pixel blocks of a GPU [C, H, W] uint8 or float32 image (C >= 3) into a
    // float32 [3, H / factor, W / factor] image in [0, 1], stored in workspace.rgb.
    lfs::core::Tensor downsample_rgb(const lfs::core::Tensor& image, int factor,
                                     DetectionWorkspace& workspace);

    // For every peak, sweeps depth along its pixel ray and counts the neighbour views whose
    // peak bitmap of the same polarity is hit, minus the hits expected by chance. views and
    // neighbors ([V, K] int32, -1 padded) are device arrays. Returns [P, SweepFieldCount].
    lfs::core::Tensor sweep_peaks(const lfs::core::Tensor& peaks,
                                  const lfs::core::Tensor& views,
                                  const lfs::core::Tensor& neighbors,
                                  const lfs::core::Tensor& bitmaps);

} // namespace lfs::training::kernels::blob_seeding

namespace lfs::gpu_ops {
    struct BlobOps {
        training::kernels::blob_seeding::ViewPeaks (*detect)(In, int, training::kernels::blob_seeding::DetectionWorkspace&);
        Tensor (*downsample)(In, int, training::kernels::blob_seeding::DetectionWorkspace&);
        Tensor (*sweep)(In, In, In, In);
    };
} // namespace lfs::gpu_ops
