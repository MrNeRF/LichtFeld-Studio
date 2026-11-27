/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"

#include <array>
#include <memory>

namespace lfs::training {

    // MRNF's blur-tolerant SSIM through the portable program in ops/blur_ssim.slang, on the backend that holds the
    // images (Metal and Vulkan dispatch it; CUDA keeps ssim.cu). The images are contiguous [N,C,H,W] Float32, the photo
    // Float32 or UInt8, the mask one [H,W] plane of Float32 or bytes, invalid when unmasked; raw is the corrected image
    // again when not decoupled.
    struct BlurSsimImages {
        core::Tensor corrected, raw, photo, mask;
    };

    // What the passes write, each contiguous on the images' backend: both maps [N,1,H,W] Float32; sharp and blurred 8
    // bytes and vu 4 bytes per image element; partials 1024 floats (2048 masked); loss and, masked, mask_sum one float;
    // grad (and grad_raw when decoupled, else invalid) like the images. grad and grad_raw may share blurred's storage:
    // the last pass writes them once the branch pass has read it.
    struct BlurSsimWorkspace {
        core::Tensor ssim_map, cs_map, sharp, blurred, vu, partials, loss, mask_sum, grad, grad_raw;
    };

    // Centre-first normalised taps of a blur sigma in (0, 1.5]: radius ceil(3 sigma) <= 5.
    struct BlurSsimTaps {
        std::array<float, 6> weight{};
        int radius = 0;
    };
    [[nodiscard]] BlurSsimTaps blur_ssim_taps(float sigma);

    // The programs a photometric state keeps loaded, one per blur radius, so a training step loads none. A state lives
    // within one backend session, as its workspace does.
    struct BlurSsimPrograms {
        std::array<std::unique_ptr<core::GpuKernelModule>, 5> loaded;
    };

    // The forward with the loss summed per group, the reduction, the blurred branch's backward and the gradients.
    // The loss goes to workspace.loss: over the valid crop unmasked; masked over the denominator, the caller's when
    // above zero, else the mask sum times N * C, which the reduction also leaves in workspace.mask_sum on the GPU.
    void blur_tolerant_ssim(BlurSsimPrograms& programs, const BlurSsimImages& images,
                            const BlurSsimWorkspace& workspace, float sigma, float ssim_weight, float denominator);

} // namespace lfs::training
