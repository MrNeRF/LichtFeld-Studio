/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/blob_cuda.hpp"
namespace lfs::training {
    const gpu_ops::BlobOps& cuda_blob_ops() {
        namespace bs = kernels::blob_seeding;
        static const gpu_ops::BlobOps result{bs::cuda_detect_peaks, bs::cuda_downsample_rgb, bs::cuda_sweep_peaks};
        return result;
    }
} // namespace lfs::training
