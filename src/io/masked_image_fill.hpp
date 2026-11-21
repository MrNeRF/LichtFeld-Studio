/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/tensor.hpp"
#include <cstdint>
#include <filesystem>

namespace lfs::io {
    // Keep policy for a source- or output-grid mask, applied while preparing a derived cache entry.
    struct MaskedImageFill {
        lfs::core::Tensor mask;
        bool alpha = false;
        bool invert = false;
        float threshold = 0.f;
        bool keep_high_band = false;
        int resize_factor = 1;
        int max_width = 0;
    };
    const char* masked_image_fill_error();

    // Host form of fill_masked_image for CPU-decoded pixels: the same keep rule, nearest kept pixel,
    // fallbacks and mask grids. Samples are interleaved (HWC), or planar (CHW) when `planar` is set;
    // 16-bit samples are unsigned. Mutates only ignored RGB; alpha is never written.
    LFS_IO_API void fill_masked_image_host(std::uint8_t* pixels, int width, int height, int channels, bool planar,
                                           const MaskedImageFill& policy);
    LFS_IO_API void fill_masked_image_host(std::uint16_t* pixels, int width, int height, int channels, bool planar,
                                           const MaskedImageFill& policy);
    LFS_IO_API void fill_masked_image_host(float* pixels, int width, int height, int channels, bool planar,
                                           const MaskedImageFill& policy);
    // A contiguous CPU image in UInt8, Float32, or Float16 holding 16-bit samples as on CUDA.
    LFS_IO_API void fill_masked_image_host(lfs::core::Tensor& image, bool planar, const MaskedImageFill& policy);
    // False when the fill leaves every image of this size unchanged: a mask on neither grid.
    [[nodiscard]] LFS_IO_API bool masked_image_fill_applies(const MaskedImageFill& policy, int width, int height);
    // Planar (CHW) RGB(A) on a GPU, filled through a host copy and uploaded to its backend again. An
    // image the policy cannot apply to is left as it is, without a copy.
    void fill_masked_planar_image(lfs::core::Tensor& image, const MaskedImageFill& policy);
    // The source-grid fill mask of a mask file, decoded as load_mask_image_cpu_decoded decodes it:
    // UInt8 [H, W] on the host, or Float32 [1, H, W] when a TIFF keeps its precision.
    [[nodiscard]] lfs::core::Tensor load_fill_mask(const std::filesystem::path& path);
} // namespace lfs::io
