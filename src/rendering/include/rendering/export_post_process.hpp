/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// GPU post-process stages for streamed high-resolution viewport exports: band
// unpack/pack between u8 HWC and float CHW, and environment-background
// compositing that matches the CPU software composite pixel-for-pixel.

#pragma once

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::rendering {

    template <typename T>
    using ExportResult = std::expected<T, std::string>;

    struct CudaEnvironmentMap {
        lfs::core::Tensor pixels; // [height, width, 3] float32 CUDA
        int width = 0;
        int height = 0;
    };

    // Single-entry cache keyed by resolved path (mirrors the CPU environment cache).
    ExportResult<std::shared_ptr<const CudaEnvironmentMap>> getOrLoadCudaEnvironmentMap(
        const std::filesystem::path& path);
    void releaseCudaEnvironmentMapCache();
    // Releases both decoded host pixels and CUDA residency. Active shared
    // users keep their map alive until their work completes.
    void releaseEnvironmentMapCaches();

    // u8 HWC [bh, W, 3|4] CUDA -> float CHW rgb [3, bh, W] (*1/255). With a 4-channel
    // source and non-null alpha_out, also writes straight alpha [bh, W].
    ExportResult<void> unpackU8HwcBandToChwFloat(const lfs::core::Tensor& band_u8_hwc,
                                                 lfs::core::Tensor& rgb_chw_out,
                                                 lfs::core::Tensor* alpha_out,
                                                 cudaStream_t stream = nullptr);

    // float CHW rgb [3, bh, W] (+ optional alpha [bh, W]) -> u8 HWC [bh, W, 3|4],
    // clamped to [0,1] and rounded.
    ExportResult<void> packChwFloatBandToU8Hwc(const lfs::core::Tensor& rgb_chw,
                                               const lfs::core::Tensor* alpha,
                                               lfs::core::Tensor& band_u8_hwc_out,
                                               cudaStream_t stream = nullptr);

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
    // u8 HWC RGB [bh, W, 3]. rgb_chw [3, bh, W] and alpha [bh, W] are float CUDA.
    ExportResult<void> compositeEnvironmentBand(const CudaEnvironmentMap& env,
                                                const EnvironmentCompositeBandParams& params,
                                                const lfs::core::Tensor& rgb_chw,
                                                const lfs::core::Tensor& alpha,
                                                lfs::core::Tensor& band_u8_hwc_out,
                                                cudaStream_t stream = nullptr);

} // namespace lfs::rendering
