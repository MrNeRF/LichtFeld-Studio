/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// GPU post-process stages for streamed high-resolution viewport exports: band
// unpack/pack between u8 HWC and float CHW, and environment-background
// compositing that matches the CPU software composite pixel-for-pixel.

#pragma once

#include "core/tensor.hpp"
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::rendering {

    template <typename T>
    using ExportResult = std::expected<T, std::string>;

    struct EnvironmentMap {
        lfs::core::Tensor pixels; // [height, width, 3] float32 on the requested GPU backend
        int width = 0;
        int height = 0;
    };

    // One entry per backend, keyed by resolved path. Uploads complete before a
    // map is published; active shared users keep their storage alive.
    ExportResult<std::shared_ptr<const EnvironmentMap>> getOrLoadEnvironmentMap(
        const std::filesystem::path& path, lfs::core::GpuBackend backend);
    // Releases both decoded host pixels and GPU residency. Active shared
    // users keep their map alive until their work completes.
    void releaseEnvironmentMapCaches();

    // u8 HWC [bh, W, 3|4] GPU -> float CHW rgb [3, bh, W] (*1/255). With a 4-channel
    // source and non-null alpha_out, also writes straight alpha [bh, W].
    ExportResult<void> unpackU8HwcBandToChwFloat(const lfs::core::Tensor& band_u8_hwc,
                                                 lfs::core::Tensor& rgb_chw_out,
                                                 lfs::core::Tensor* alpha_out);

    // float CHW rgb [3, bh, W] (+ optional alpha [bh, W]) -> u8 HWC [bh, W, 3|4],
    // clamped to [0,1] and rounded.
    ExportResult<void> packChwFloatBandToU8Hwc(const lfs::core::Tensor& rgb_chw,
                                               const lfs::core::Tensor* alpha,
                                               lfs::core::Tensor& band_u8_hwc_out);

    struct EnvironmentCompositeBandParams {
        glm::mat3 camera_rotation{1.0f};
        glm::ivec2 full_size{0, 0};
        int y_offset = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        bool equirectangular_view = false;
        float exposure = 0.0f; // EV; exp2 applied internally
        float rotation_degrees = 0.0f;
    };

    // out = mix(shaded_environment, rgb, alpha) with straight alpha, written as
    // u8 HWC RGB [bh, W, 3]. All inputs and the output use the same GPU backend.
    ExportResult<void> compositeEnvironmentBand(const EnvironmentMap& env,
                                                const EnvironmentCompositeBandParams& params,
                                                const lfs::core::Tensor& rgb_chw,
                                                const lfs::core::Tensor& alpha,
                                                lfs::core::Tensor& band_u8_hwc_out);

} // namespace lfs::rendering
