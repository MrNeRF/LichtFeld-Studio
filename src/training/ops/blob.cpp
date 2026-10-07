/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/blob.hpp"
#include "lfs/training/ops/registry.hpp"
#include <stdexcept>
namespace lfs::training::kernels::blob_seeding {
    namespace {
        const gpu_ops::BlobOps& operations(const core::Tensor& tensor) {
            const auto backend = core::gpu_backend_of(tensor).value_or(core::default_gpu_backend());
            const auto* result = training_ops(backend).blob;
            if (!result)
                throw std::runtime_error(unavailable_training_family(backend, Family::Blob)
                                             .value_or("Blob training ops are unavailable"));
            return *result;
        }
    } // namespace
    ViewPeaks detect_peaks(const core::Tensor& image, int view, DetectionWorkspace& workspace) {
        return operations(image).detect(image, view, workspace);
    }
    core::Tensor downsample_rgb(const core::Tensor& image, int factor, DetectionWorkspace& workspace) {
        return operations(image).downsample(image, factor, workspace);
    }
    core::Tensor sweep_peaks(const core::Tensor& peaks, const core::Tensor& views, const core::Tensor& neighbors, const core::Tensor& bitmaps) {
        return operations(peaks).sweep(peaks, views, neighbors, bitmaps);
    }
} // namespace lfs::training::kernels::blob_seeding
