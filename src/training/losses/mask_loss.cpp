/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mask_loss.hpp"

#include "core/assert.hpp"
#include "lfs/training/ops/registry.hpp"

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
            {H, W}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
        grad_alpha = lfs::core::Tensor::empty(
            {H, W}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
        loss_scalar = lfs::core::Tensor::empty(
            {1}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
        reduce_temp = lfs::core::Tensor::empty(
            {1024}, lfs::core::Device::GPU, lfs::core::DataType::Float32);
        allocated_h = H;
        allocated_w = W;
    }

    namespace {

        [[nodiscard]] std::pair<size_t, size_t> hw2(const lfs::core::Tensor& t) {
            LFS_ASSERT_MSG(t.is_valid() && t.ndim() == 2, "mask preprocess expects 2D tensor");
            return {t.shape()[0], t.shape()[1]};
        }

    } // namespace

    lfs::core::Tensor fuse_photometric_mask_weight(
        MaskPreprocessWorkspace& ws,
        const lfs::core::Tensor& user_mask,
        const lfs::core::Tensor& roi_weight,
        const bool segment_and_ignore,
        const bool require_float) {
        if (!user_mask.is_valid() || user_mask.numel() == 0) {
            return roi_weight;
        }

        LFS_ASSERT_MSG(user_mask.ndim() == 2, "photometric mask must be 2D");
        LFS_ASSERT_MSG(
            user_mask.device() == lfs::core::Device::GPU,
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
        if (!segment_and_ignore && !roi_weight.is_valid() &&
            (!require_float || user_mask.dtype() == lfs::core::DataType::Float32)) {
            return user_mask;
        }

        ws.ensure_size(H, W);
        const auto mode = segment_and_ignore
                              ? lfs::gpu_ops::MaskPhotoMode::SegmentAndIgnore
                              : lfs::gpu_ops::MaskPhotoMode::BinaryGt0;
        training_ops(core::default_gpu_backend()).masks->photometric_weight(user_mask, roi_weight, ws.photometric_weight, mode);
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
                              ? lfs::gpu_ops::MaskOpacityMode::SegmentAndIgnore
                              : lfs::gpu_ops::MaskOpacityMode::BinaryGt0;
        training_ops(core::default_gpu_backend()).masks->opacity_penalty(alpha, mask_raw, roi_weight, ws.grad_alpha, ws.reduce_temp, ws.loss_scalar, mode, power, scale);

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
        training_ops(core::default_gpu_backend()).masks->alpha_consistency(alpha, mask, roi_weight, ws.grad_alpha, ws.reduce_temp, ws.loss_scalar, weight);

        return MaskOpacityPenalty{
            .loss = ws.loss_scalar,
            .grad_alpha = ws.grad_alpha};
    }

} // namespace lfs::training::losses
