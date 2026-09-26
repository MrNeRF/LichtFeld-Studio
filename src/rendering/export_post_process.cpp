/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/export_post_process.hpp"
#include "core/path_utils.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_environment.hpp"
#include "environment_image.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <limits>
#include <mutex>

namespace lfs::rendering {

    namespace {

        struct EnvironmentMapCache {
            std::filesystem::path path;
            std::shared_ptr<const EnvironmentMap> map;
        };

        std::mutex environment_cache_mutex;
        std::array<EnvironmentMapCache, lfs::core::kGpuBackendCount> environment_cache;

        [[nodiscard]] ExportResult<void> validateBandU8Hwc(const lfs::core::Tensor& band, const char* const what) {
            if (!band.is_valid() || band.device() != lfs::core::Device::GPU ||
                band.dtype() != lfs::core::DataType::UInt8 || band.ndim() != 3 || !band.is_contiguous() ||
                (band.size(2) != 3 && band.size(2) != 4) || band.size(0) <= 0 || band.size(1) <= 0) {
                return std::unexpected(std::format("{} must be a contiguous GPU u8 HWC RGB/RGBA tensor", what));
            }
            return {};
        }

        [[nodiscard]] ExportResult<void> validateRgbChw(const lfs::core::Tensor& rgb, const char* const what) {
            if (!rgb.is_valid() || rgb.device() != lfs::core::Device::GPU ||
                rgb.dtype() != lfs::core::DataType::Float32 || rgb.ndim() != 3 || !rgb.is_contiguous() ||
                rgb.size(0) != 3 || rgb.size(1) <= 0 || rgb.size(2) <= 0) {
                return std::unexpected(std::format("{} must be a contiguous GPU float [3,H,W] tensor", what));
            }
            return {};
        }

    } // namespace

    ExportResult<std::shared_ptr<const EnvironmentMap>> getOrLoadEnvironmentMap(
        const std::filesystem::path& path, const lfs::core::GpuBackend backend) {
        auto image = loadEnvironmentImageShared(path);
        if (!image) {
            return std::unexpected(image.error());
        }
        std::lock_guard lock(environment_cache_mutex);
        auto& cache = environment_cache[static_cast<size_t>(backend)];
        if (cache.map && cache.path == (*image)->path) {
            return cache.map;
        }
        try {
            const lfs::core::GpuBackendScope scope(backend);
            auto map = std::make_shared<EnvironmentMap>();
            map->width = (*image)->width;
            map->height = (*image)->height;
            map->pixels = lfs::core::Tensor::from_blob(
                              const_cast<float*>((*image)->pixels.data()),
                              {static_cast<size_t>(map->height), static_cast<size_t>(map->width), 3},
                              lfs::core::Device::CPU, lfs::core::DataType::Float32)
                              .gpu();
            if (!map->pixels.is_valid()) {
                return std::unexpected("failed to allocate GPU environment map");
            }
            // The cache becomes visible only after its upload completes.
            lfs::core::TensorCompletion completion;
            completion.include(map->pixels);
            completion.wait();
            map->pixels.set_stream(nullptr);
            cache.path = (*image)->path;
            cache.map = std::move(map);
            return cache.map;
        } catch (const std::exception& error) {
            return std::unexpected(std::format("failed to upload environment map {}: {}",
                                               lfs::core::path_to_utf8((*image)->path), error.what()));
        }
    }

    void releaseEnvironmentMapCaches() {
        {
            std::lock_guard lock(environment_cache_mutex);
            for (auto& cache : environment_cache) {
                cache.map.reset();
                cache.path.clear();
            }
        }
        releaseEnvironmentImageCache();
    }

    ExportResult<void> unpackU8HwcBandToChwFloat(const lfs::core::Tensor& band_u8_hwc,
                                                 lfs::core::Tensor& rgb_chw_out,
                                                 lfs::core::Tensor* const alpha_out) {
        if (auto valid = validateBandU8Hwc(band_u8_hwc, "unpack source"); !valid) {
            return valid;
        }

        const auto height = band_u8_hwc.size(0);
        const auto width = band_u8_hwc.size(1);
        const int channels = static_cast<int>(band_u8_hwc.size(2));
        const auto backend = *lfs::core::gpu_backend_of(band_u8_hwc);
        const lfs::core::GpuBackendScope scope(backend);
        const lfs::core::Tensor chw =
            band_u8_hwc.to(lfs::core::DataType::Float32).mul(1.0f / 255.0f).permute({2, 0, 1});
        rgb_chw_out = chw.slice(0, 0, 3).contiguous();
        if (alpha_out != nullptr) {
            *alpha_out = lfs::core::Tensor{};
            if (channels == 4) {
                *alpha_out = chw.slice(0, 3, 4).contiguous().reshape(
                    {static_cast<int>(height), static_cast<int>(width)});
            }
        }
        return {};
    }

    ExportResult<void> packChwFloatBandToU8Hwc(const lfs::core::Tensor& rgb_chw,
                                               const lfs::core::Tensor* const alpha,
                                               lfs::core::Tensor& band_u8_hwc_out) {
        if (auto valid = validateRgbChw(rgb_chw, "pack source"); !valid) {
            return valid;
        }

        const auto backend = *lfs::core::gpu_backend_of(rgb_chw);
        const lfs::core::GpuBackendScope scope(backend);
        const auto height = rgb_chw.size(1);
        const auto width = rgb_chw.size(2);
        const size_t num_pixels = height * width;
        const bool has_alpha = alpha != nullptr && alpha->is_valid();
        if (has_alpha &&
            (alpha->device() != lfs::core::Device::GPU || alpha->dtype() != lfs::core::DataType::Float32 ||
             !alpha->is_contiguous() || alpha->numel() != num_pixels || lfs::core::gpu_backend_of(*alpha) != backend)) {
            return std::unexpected("pack alpha must be a contiguous GPU float tensor matching the band");
        }
        lfs::core::Tensor planes = rgb_chw;
        if (has_alpha) {
            planes = lfs::core::Tensor::empty({4, height, width}, rgb_chw.device(), rgb_chw.dtype());
            planes.slice(0, 0, 3).copy_from(rgb_chw);
            planes.slice(0, 3, 4).copy_from(alpha->reshape({1, static_cast<int>(height), static_cast<int>(width)}));
        }
        const lfs::core::Tensor bytes = planes
                                            .clamp(0.0f, 1.0f)
                                            .mul(255.0f)
                                            .add(0.5f)
                                            .to(lfs::core::DataType::UInt8);
        band_u8_hwc_out = bytes.permute({1, 2, 0}).contiguous();
        return {};
    }

    ExportResult<void> compositeEnvironmentBand(const EnvironmentMap& env,
                                                const EnvironmentCompositeBandParams& params,
                                                const lfs::core::Tensor& rgb_chw,
                                                const lfs::core::Tensor& alpha,
                                                lfs::core::Tensor& band_u8_hwc_out) {
        if (auto valid = validateRgbChw(rgb_chw, "composite source"); !valid) {
            return valid;
        }
        if (env.width <= 0 || env.height <= 0 || !env.pixels.is_valid() ||
            env.pixels.device() != lfs::core::Device::GPU ||
            env.pixels.dtype() != lfs::core::DataType::Float32 || !env.pixels.is_contiguous() ||
            env.pixels.shape() != lfs::core::TensorShape({static_cast<size_t>(env.height), static_cast<size_t>(env.width), 3}) ||
            lfs::core::gpu_backend_of(env.pixels) != lfs::core::gpu_backend_of(rgb_chw) ||
            static_cast<size_t>(env.width) * env.height > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return std::unexpected("composite environment map is not resident on the image backend");
        }

        const auto backend = *lfs::core::gpu_backend_of(rgb_chw);
        const lfs::core::GpuBackendScope scope(backend);
        const auto height = rgb_chw.size(1);
        const auto width = rgb_chw.size(2);
        const size_t num_pixels = height * width;
        if (!alpha.is_valid() || alpha.device() != lfs::core::Device::GPU ||
            alpha.dtype() != lfs::core::DataType::Float32 || !alpha.is_contiguous() ||
            alpha.numel() != num_pixels || lfs::core::gpu_backend_of(alpha) != backend) {
            return std::unexpected("composite alpha must be a contiguous GPU float tensor matching the band");
        }
        if (params.full_size.x != static_cast<int>(width) || params.y_offset < 0 ||
            params.full_size.y <= 0 || height > static_cast<size_t>(params.full_size.y) ||
            params.y_offset > params.full_size.y - static_cast<int>(height) ||
            num_pixels > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return std::unexpected("composite band region does not match the full image size");
        }

        lfs::core::EnvironmentCompositeParams device_params;
        static_assert(sizeof(device_params.rotation) == sizeof(params.camera_rotation));
        std::memcpy(device_params.rotation, &params.camera_rotation[0][0], sizeof(device_params.rotation));
        device_params.full_width = params.full_size.x;
        device_params.full_height = params.full_size.y;
        device_params.band_width = static_cast<int>(width);
        device_params.band_height = static_cast<int>(height);
        device_params.y_offset = params.y_offset;
        device_params.focal_x = params.focal_x;
        device_params.focal_y = params.focal_y;
        device_params.center_x = params.center_x;
        device_params.center_y = params.center_y;
        device_params.equirect_view = params.equirectangular_view;
        device_params.exposure_factor = std::exp2(params.exposure);
        device_params.env_rotation_radians = glm::radians(params.rotation_degrees);
        device_params.env_width = env.width;
        device_params.env_height = env.height;

        band_u8_hwc_out = lfs::core::environment_composite(rgb_chw, alpha, env.pixels, device_params);
        return {};
    }

} // namespace lfs::rendering
