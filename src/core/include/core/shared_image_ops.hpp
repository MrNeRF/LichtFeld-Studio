/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"
#include "core/tensor_image.hpp"
#include <cstddef>
#include <cstdint>

namespace lfs::gpu_ops {

    // Decode-side transform parameters for normal priors (v = n*0.5 + 0.5 file encoding).
    struct NormalPriorTransform {
        bool srgb = false;
        bool flip_yz = false;
        bool world_to_camera = false;
        float w2c[9] = {};
    };

    enum class ImageConversion {
        U8HWCToF32CHW,
        U16HWCToF32CHW,
        F32HWCToU16HWC,
        U16HWCToF32HWC,
        NormalCHWToJ2KHWC,
        J2KHWCToNormalCHW,
        NormalPriorU8,
        NormalPriorU16,
        U8HWCToU8CHW,
        U16HWCToU8CHW,
        F32CHWToU8CHW,
        U8HWToF32HW
    };
    enum class MaskTransform { Invert,
                               Threshold };
    enum class Resample { LanczosRGB,
                          LanczosGray,
                          LanczosFloatCHW,
                          DepthPrior,
                          NormalPrior };

    // IO and camera preprocessing share this table with the trainer. Dimensions
    // describe pixels, including uint16 codec storage held in byte tensors.
    // Entries use the current execution scope; callers own borrowed buffers.
    struct SharedImageOps {
        void (*sentinel_fill)(core::Tensor& bytes, uint32_t seed);
        void (*sentinel_check)(const core::Tensor& bytes, core::Tensor& unchanged, uint32_t seed);
        void (*convert)(const core::Tensor& source, core::Tensor& destination, ImageConversion,
                        size_t height, size_t width, size_t channels, const NormalPriorTransform&);
        void (*rgba_split)(const core::Tensor& rgba, core::Tensor& rgb, core::Tensor& alpha);
        void (*mask)(core::Tensor& mask, MaskTransform, float threshold);
        // Returning the result keeps allocation inside the existing launchers.
        core::Tensor (*resize)(const core::Tensor& source, int height, int width, Resample, int kernel_size);
        core::Tensor (*undistort)(const core::Tensor& source, const core::UndistortParams&, bool mask);
    };

} // namespace lfs::gpu_ops

namespace lfs::core {
    LFS_CORE_API const gpu_ops::SharedImageOps* shared_image_ops(GpuBackend backend);
} // namespace lfs::core
