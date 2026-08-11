/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mask_loss.hpp"

#include "core/assert.hpp"
#include "training/kernels/mask_preprocess.hpp"

#include <cmath>

namespace lfs::training::losses {

    void MaskPreprocessWorkspace::ensure_size(const size_t H, const size_t W) {
        if (H == 0 || W == 0) {
            return;
        }
        if (allocated_h == H && allocated_w == W &&
            photometric_weight.is_valid() && grad_alpha.is_valid() &&
            loss_scalar.is_valid() && reduce_temp.is_valid()) {
            return;
        }
        photometric_weight = lfs::core::Tensor::empty(
            {H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        grad_alpha = lfs::core::Tensor::empty(
            {H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        loss_scalar = lfs::core::Tensor::empty(
            {1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        reduce_temp = lfs::core::Tensor::empty(
            {1024}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        allocated_h = H;
        allocated_w = W;
    }

    namespace {

        [[nodiscard]] std::pair<size_t, size_t> hw2(const lfs::core::Tensor& t) {
            LFS_ASSERT_MSG(t.is_valid() && t.ndim() == 2, "mask preprocess expects 2D tensor");
            return {t.shape()[0], t.shape()[1]};
        }

        [[nodiscard]] const float* roi_ptr_or_null(const lfs::core::Tensor& roi) {
            if (!roi.is_valid() || roi.numel() == 0) {
                return nullptr;
            }
            LFS_ASSERT_MSG(roi.dtype() == lfs::core::DataType::Float32, "ROI must be Float32");
            return roi.ptr<float>();
        }

    } // namespace

    lfs::core::Tensor fuse_photometric_mask_weight(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& user_mask,
        const lfs::core::Tensor& roi_weight,
        const bool segment_and_ignore) {
        if (!user_mask.is_valid() || user_mask.numel() == 0) {
            return roi_weight;
        }

        LFS_ASSERT_MSG(user_mask.ndim() == 2, "photometric mask must be 2D");
        LFS_ASSERT_MSG(
            user_mask.device() == lfs::core::Device::CUDA,
            "photometric mask must be CUDA");
        LFS_ASSERT_MSG(
            user_mask.dtype() == lfs::core::DataType::Float32 ||
                user_mask.dtype() == lfs::core::DataType::UInt8 ||
                user_mask.dtype() == lfs::core::DataType::Bool,
            "photometric mask dtype must be Float32, UInt8, or Bool");

        const auto [H, W] = hw2(user_mask);
        if (roi_weight.is_valid()) {
            LFS_ASSERT_MSG(
                roi_weight.ndim() == 2 &&
                    roi_weight.shape()[0] == H &&
                    roi_weight.shape()[1] == W,
                "ROI weight shape must match mask");
            LFS_ASSERT_MSG(
                roi_weight.dtype() == lfs::core::DataType::Float32,
                "ROI weight must be Float32");
        }

        // No user remap + no ROI: return original (UInt8 ok for masked SSIM).
        if (!segment_and_ignore && !roi_weight.is_valid()) {
            return user_mask;
        }

        ws.ensure_size(H, W);
        const auto mode = segment_and_ignore
                              ? lfs::training::kernels::MaskPhotoMode::SegmentAndIgnore
                              : lfs::training::kernels::MaskPhotoMode::BinaryGt0;
        const float* roi = roi_ptr_or_null(roi_weight);

        if (user_mask.dtype() == lfs::core::DataType::Float32) {
            lfs::training::kernels::launch_fuse_photometric_mask_weight_f32(
                user_mask.ptr<float>(),
                roi,
                ws.photometric_weight.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                mode);
        } else {
            // Bool is stored as uint8 in this codebase for dense masks.
            lfs::training::kernels::launch_fuse_photometric_mask_weight_u8(
                user_mask.ptr<uint8_t>(),
                roi,
                ws.photometric_weight.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                mode);
        }
        return ws.photometric_weight;
    }

    MaskOpacityPenalty fuse_mask_opacity_penalty(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& mask_raw,
        const lfs::core::Tensor& roi_weight,
        const float power,
        const float scale,
        const bool segment_and_ignore) {
        LFS_ASSERT_MSG(alpha.is_valid() && mask_raw.is_valid(), "opacity penalty inputs");
        LFS_ASSERT_MSG(alpha.dtype() == lfs::core::DataType::Float32, "alpha must be Float32");
        LFS_ASSERT_MSG(alpha.ndim() == 2 && mask_raw.ndim() == 2, "opacity penalty expects 2D");
        LFS_ASSERT_MSG(alpha.shape() == mask_raw.shape(), "alpha/mask shape mismatch");
        LFS_ASSERT_MSG(std::isfinite(scale) && scale >= 0.0f, "scale must be finite >= 0");
        LFS_ASSERT_MSG(std::isfinite(power), "power must be finite");

        const auto [H, W] = hw2(alpha);
        if (roi_weight.is_valid()) {
            LFS_ASSERT_MSG(
                roi_weight.dtype() == lfs::core::DataType::Float32 &&
                    roi_weight.shape() == alpha.shape(),
                "ROI must match alpha");
        }

        ws.ensure_size(H, W);
        const auto mode = segment_and_ignore
                              ? lfs::training::kernels::MaskOpacityMode::SegmentAndIgnore
                              : lfs::training::kernels::MaskOpacityMode::BinaryGt0;
        const float* roi = roi_ptr_or_null(roi_weight);

        if (mask_raw.dtype() == lfs::core::DataType::Float32) {
            lfs::training::kernels::launch_fuse_mask_opacity_penalty_f32(
                alpha.ptr<float>(),
                mask_raw.ptr<float>(),
                roi,
                ws.grad_alpha.ptr<float>(),
                ws.reduce_temp.ptr<float>(),
                ws.loss_scalar.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                power,
                scale,
                mode);
        } else {
            LFS_ASSERT_MSG(
                mask_raw.dtype() == lfs::core::DataType::UInt8 ||
                    mask_raw.dtype() == lfs::core::DataType::Bool,
                "mask dtype");
            lfs::training::kernels::launch_fuse_mask_opacity_penalty_u8(
                alpha.ptr<float>(),
                mask_raw.ptr<uint8_t>(),
                roi,
                ws.grad_alpha.ptr<float>(),
                ws.reduce_temp.ptr<float>(),
                ws.loss_scalar.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                power,
                scale,
                mode);
        }

        return MaskOpacityPenalty{
            .loss = ws.loss_scalar,
            .grad_alpha = ws.grad_alpha};
    }

    MaskOpacityPenalty fuse_alpha_consistent(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& alpha,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& roi_weight,
        const float weight) {
        LFS_ASSERT_MSG(alpha.is_valid() && mask.is_valid(), "alpha consistent inputs");
        LFS_ASSERT_MSG(alpha.dtype() == lfs::core::DataType::Float32, "alpha must be Float32");
        LFS_ASSERT_MSG(alpha.ndim() == 2 && mask.ndim() == 2, "alpha consistent expects 2D");
        LFS_ASSERT_MSG(alpha.shape() == mask.shape(), "alpha/mask shape mismatch");
        LFS_ASSERT_MSG(std::isfinite(weight), "weight must be finite");

        const auto [H, W] = hw2(alpha);
        if (roi_weight.is_valid()) {
            LFS_ASSERT_MSG(
                roi_weight.dtype() == lfs::core::DataType::Float32 &&
                    roi_weight.shape() == alpha.shape(),
                "ROI must match alpha");
        }

        ws.ensure_size(H, W);
        const float* roi = roi_ptr_or_null(roi_weight);

        if (mask.dtype() == lfs::core::DataType::Float32) {
            lfs::training::kernels::launch_fuse_alpha_consistent_f32(
                alpha.ptr<float>(),
                mask.ptr<float>(),
                roi,
                ws.grad_alpha.ptr<float>(),
                ws.reduce_temp.ptr<float>(),
                ws.loss_scalar.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                weight);
        } else {
            LFS_ASSERT_MSG(
                mask.dtype() == lfs::core::DataType::UInt8 ||
                    mask.dtype() == lfs::core::DataType::Bool,
                "mask dtype");
            lfs::training::kernels::launch_fuse_alpha_consistent_u8(
                alpha.ptr<float>(),
                mask.ptr<uint8_t>(),
                roi,
                ws.grad_alpha.ptr<float>(),
                ws.reduce_temp.ptr<float>(),
                ws.loss_scalar.ptr<float>(),
                static_cast<int>(H),
                static_cast<int>(W),
                weight);
        }

        return MaskOpacityPenalty{
            .loss = ws.loss_scalar,
            .grad_alpha = ws.grad_alpha};
    }

} // namespace lfs::training::losses
