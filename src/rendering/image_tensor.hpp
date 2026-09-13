/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "image_layout.hpp"

namespace lfs::rendering {

    namespace detail {

        // HWC UInt8 with 1/3/4 channels -> packed RGBA8. Gray fills RGB and alpha 255;
        // RGB gets opaque alpha; RGBA alpha is preserved. Stays on the source backend.
        [[nodiscard]] inline Tensor expandHwcToRgba8(const Tensor& hwc_u8) {
            const int channels = static_cast<int>(hwc_u8.size(2));
            if (channels == 4) {
                return hwc_u8;
            }
            const Tensor rgb = (channels == 1) ? Tensor::cat({hwc_u8, hwc_u8, hwc_u8}, 2)
                                               : hwc_u8;
            const Tensor alpha = Tensor::full_like(hwc_u8.slice(2, 0, 1), 255.0f);
            return Tensor::cat({rgb, alpha}, 2);
        }

        // Nearest uint8 matching cuda_vulkan_interop::toByte and export_post_process::floatToU8:
        // unsigned char(fminf(fmaxf(v, 0), 1) * 255 + 0.5). Tensor::clamp preserves NaN on CPU
        // and CUDA; to(UInt8) then uses torch_uint8_cast, which maps non-finite values to 0, so
        // NaN still becomes 0 (same as fmaxf(NaN, 0)). Inf clamps to the [0, 1] endpoints before
        // rounding. CPU previously truncated (0.5 -> 127); that mismatch is treated as a
        // harmonized fix — both backends now use nearest.
        [[nodiscard]] inline Tensor floatImageToUInt8Nearest(const Tensor& image) {
            Tensor formatted = image;
            if (formatted.dtype() != lfs::core::DataType::Float32) {
                formatted = formatted.to(lfs::core::DataType::Float32);
            }
            return (formatted.clamp(0.0f, 1.0f) * 255.0f + 0.5f)
                .to(lfs::core::DataType::UInt8);
        }

        [[nodiscard]] inline Tensor prepareImageRgba8OnSourceBackend(const Tensor& image,
                                                                     const ImageLayout layout,
                                                                     const bool flip_y) {
            const Tensor hwc = (layout == ImageLayout::HWC) ? image
                                                            : image.permute({1, 2, 0});
            const Tensor bytes = hwc.dtype() == lfs::core::DataType::UInt8
                                     ? hwc
                                     : floatImageToUInt8Nearest(hwc);
            const Tensor oriented = flip_y ? flipImageVertical(bytes, ImageLayout::HWC)
                                           : bytes;
            return expandHwcToRgba8(oriented).contiguous();
        }

    } // namespace detail

    // Backend-preserving packed HWC UInt8 RGBA. CPU sources stay on the host (no GPU
    // upload). GPU sources convert with tensor ops on their own backend, including when
    // the active factory backend is the opposite one. Float conversion is nearest
    // (clamp[0,1]*255+0.5), matching CUDA surface / image-export bytes. Empty tensor on
    // invalid input.
    [[nodiscard]] inline Tensor prepareImageRgba8(const Tensor& image, const bool flip_y = false) {
        if (!image.is_valid() || image.ndim() != 3) {
            return {};
        }

        const ImageLayout layout = detectImageLayout(image);
        if (layout == ImageLayout::Unknown) {
            return {};
        }

        const int channels = imageChannels(image, layout);
        if (channels != 1 && channels != 3 && channels != 4) {
            return {};
        }
        if (imageWidth(image, layout) <= 0 || imageHeight(image, layout) <= 0) {
            return {};
        }

        const auto convert = [&]() {
            return detail::prepareImageRgba8OnSourceBackend(image, layout, flip_y);
        };
        if (const auto backend = lfs::core::gpu_backend_of(image)) {
            lfs::core::GpuBackendScope scope(*backend);
            return convert();
        }
        return convert();
    }

} // namespace lfs::rendering
