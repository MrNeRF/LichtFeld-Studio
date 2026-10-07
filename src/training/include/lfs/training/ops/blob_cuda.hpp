/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/blob.hpp"
namespace lfs::training {
    const gpu_ops::BlobOps& cuda_blob_ops();
}
namespace lfs::training::kernels::blob_seeding {
    ViewPeaks cuda_detect_peaks(const core::Tensor&, int, DetectionWorkspace&);
    core::Tensor cuda_downsample_rgb(const core::Tensor&, int, DetectionWorkspace&);
    core::Tensor cuda_sweep_peaks(const core::Tensor&, const core::Tensor&, const core::Tensor&, const core::Tensor&);
} // namespace lfs::training::kernels::blob_seeding
