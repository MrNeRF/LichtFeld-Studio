// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/loss.hpp"
#include "lfs/training/ops/registry.hpp"

namespace lfs::test {
    struct PhotoTestResult {
        core::Tensor loss, grad_corrected, grad_raw;
    };

    class TrainingPhotoTestWorkspace {
    public:
        PhotoTestResult evaluate(const core::Tensor& render, const core::Tensor& raw,
                                 const core::Tensor& target, const core::Tensor& mask,
                                 float mix, float sigma = 0, float denominator = 0) {
            using gpu_ops::PhotoPath;
            const auto path = mask.is_valid()  ? (raw.is_valid() ? PhotoPath::MaskedDecoupled : PhotoPath::MaskedFused)
                              : raw.is_valid() ? PhotoPath::Decoupled
                              : mix == 0.f     ? PhotoPath::L1
                                               : PhotoPath::Fused;
            PhotoTestResult result;
            ops_.evaluate(saved_, render, raw, target, mask,
                          {.path = path, .ssim_weight = mix, .denominator = denominator, .blur_sigma = sigma, .calibration_fit = true},
                          result.loss, result.grad_corrected, result.grad_raw);
            return result;
        }

    private:
        const gpu_ops::PhotometricOps& ops_ = *training::training_ops(core::default_gpu_backend()).photometric;
        gpu_ops::PhotoSaved saved_{.backend = ops_.create()};
    };
} // namespace lfs::test
