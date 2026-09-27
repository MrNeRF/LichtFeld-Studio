/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/masks_cuda.hpp"

#include "core/assert.hpp"
#include "kernels/mask_preprocess.hpp"

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;

        const float* roi_ptr(In roi) {
            return roi.is_valid() && roi.numel() > 0 ? roi.ptr<float>() : nullptr;
        }

        void photometric_weight(In mask, In roi, Out weight, MaskPhotoMode mode) {
            const int h = mask.shape()[0], w = mask.shape()[1];
            if (mask.dtype() == core::DataType::Float32) {
                kernels::launch_fuse_photometric_mask_weight_f32(mask.ptr<float>(), roi_ptr(roi), weight.ptr<float>(), h, w, mode);
            } else {
                kernels::launch_fuse_photometric_mask_weight_u8(mask.ptr<uint8_t>(), roi_ptr(roi), weight.ptr<float>(), h, w, mode);
            }
        }

        void opacity_penalty(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp,
                             Out loss, MaskOpacityMode mode, float power, float scale) {
            const int h = alpha.shape()[0], w = alpha.shape()[1];
            if (mask.dtype() == core::DataType::Float32) {
                kernels::launch_fuse_mask_opacity_penalty_f32(alpha.ptr<float>(), mask.ptr<float>(), roi_ptr(roi),
                                                              grad_alpha.ptr<float>(), reduction_temp.ptr<float>(), loss.ptr<float>(), h, w, power, scale, mode);
            } else {
                LFS_ASSERT_MSG(mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool, "mask dtype");
                kernels::launch_fuse_mask_opacity_penalty_u8(alpha.ptr<float>(), mask.ptr<uint8_t>(), roi_ptr(roi),
                                                             grad_alpha.ptr<float>(), reduction_temp.ptr<float>(), loss.ptr<float>(), h, w, power, scale, mode);
            }
        }

        void alpha_consistency(In alpha, In mask, In roi, Out grad_alpha, Out reduction_temp,
                               Out loss, float weight) {
            const int h = alpha.shape()[0], w = alpha.shape()[1];
            if (mask.dtype() == core::DataType::Float32) {
                kernels::launch_fuse_alpha_consistent_f32(alpha.ptr<float>(), mask.ptr<float>(), roi_ptr(roi),
                                                          grad_alpha.ptr<float>(), reduction_temp.ptr<float>(), loss.ptr<float>(), h, w, weight);
            } else {
                LFS_ASSERT_MSG(mask.dtype() == core::DataType::UInt8 || mask.dtype() == core::DataType::Bool, "mask dtype");
                kernels::launch_fuse_alpha_consistent_u8(alpha.ptr<float>(), mask.ptr<uint8_t>(), roi_ptr(roi),
                                                         grad_alpha.ptr<float>(), reduction_temp.ptr<float>(), loss.ptr<float>(), h, w, weight);
            }
        }

        const MaskOps kCudaMaskOps{
            .photometric_weight = photometric_weight,
            .opacity_penalty = opacity_penalty,
            .alpha_consistency = alpha_consistency,
        };
    } // namespace

    const lfs::gpu_ops::MaskOps& cuda_masks_ops() { return kCudaMaskOps; }
} // namespace lfs::training
