/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/export_post_process.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "environment_image.hpp"
#include "export_post_process_kernels.cuh"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <mutex>

namespace lfs::rendering {

    namespace {

        [[nodiscard]] cudaStream_t resolveExportStream(const cudaStream_t stream) {
            if (stream != nullptr) {
                return stream;
            }
            return lfs::core::getCurrentCUDAStream();
        }

        [[nodiscard]] ExportResult<void> launchStatus(const char* const kernel_name, const cudaError_t status) {
            if (status != cudaSuccess) {
                return std::unexpected(std::format("{}: {}", kernel_name, cudaGetErrorString(status)));
            }
            return {};
        }

        struct CudaEnvironmentMapCache {
            std::mutex mutex;
            std::filesystem::path path;
            std::shared_ptr<const CudaEnvironmentMap> map;
        };

        [[nodiscard]] CudaEnvironmentMapCache& cudaEnvironmentMapCache() {
            static CudaEnvironmentMapCache cache;
            return cache;
        }

        [[nodiscard]] ExportResult<void> validateBandU8Hwc(const lfs::core::Tensor& band, const char* const what) {
            if (!band.is_valid() || band.device() != lfs::core::Device::CUDA ||
                band.dtype() != lfs::core::DataType::UInt8 || band.ndim() != 3 || !band.is_contiguous() ||
                (band.size(2) != 3 && band.size(2) != 4) || band.size(0) <= 0 || band.size(1) <= 0) {
                return std::unexpected(std::format("{} must be a contiguous CUDA u8 HWC RGB/RGBA tensor", what));
            }
            return {};
        }

        [[nodiscard]] ExportResult<void> validateRgbChw(const lfs::core::Tensor& rgb, const char* const what) {
            if (!rgb.is_valid() || rgb.device() != lfs::core::Device::CUDA ||
                rgb.dtype() != lfs::core::DataType::Float32 || rgb.ndim() != 3 || !rgb.is_contiguous() ||
                rgb.size(0) != 3 || rgb.size(1) <= 0 || rgb.size(2) <= 0) {
                return std::unexpected(std::format("{} must be a contiguous CUDA float [3,H,W] tensor", what));
            }
            return {};
        }

    } // namespace

    ExportResult<std::shared_ptr<const CudaEnvironmentMap>> getOrLoadCudaEnvironmentMap(
        const std::filesystem::path& path) {
        auto image = loadEnvironmentImageShared(path);
        if (!image) {
            return std::unexpected(image.error());
        }

        auto& cache = cudaEnvironmentMapCache();
        std::lock_guard lock(cache.mutex);
        if (cache.map && cache.path == (*image)->path) {
            return cache.map;
        }

        auto map = std::make_shared<CudaEnvironmentMap>();
        map->width = (*image)->width;
        map->height = (*image)->height;
        cudaStream_t upload_stream = nullptr;
        cudaError_t status = cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            return std::unexpected(std::format("failed to create environment upload stream: {} ({})",
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }
        {
            lfs::core::CUDAStreamGuard stream_guard(upload_stream);
            map->pixels = lfs::core::Tensor::empty(
                {static_cast<size_t>((*image)->height),
                 static_cast<size_t>((*image)->width),
                 size_t{3}},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }
        if (!map->pixels.is_valid()) {
            lfs::core::CudaMemoryPool::instance().release_stream(upload_stream);
            (void)cudaStreamDestroy(upload_stream);
            return std::unexpected(std::format("failed to allocate CUDA environment map {}",
                                               (*image)->path.string()));
        }

        if (status == cudaSuccess) {
            status = cudaMemcpyAsync(map->pixels.data_ptr(),
                                     (*image)->pixels.data(),
                                     map->pixels.bytes(),
                                     cudaMemcpyHostToDevice,
                                     upload_stream);
        }
        if (status == cudaSuccess) {
            status = cudaStreamSynchronize(upload_stream);
        }
        // The cached tensor was allocated on this temporary lane. Sever every
        // allocator reference before destroying the stream so a later cache
        // release never dereferences a dead cudaStream_t.
        lfs::core::CudaMemoryPool::instance().release_stream(upload_stream);
        const cudaError_t destroy_status = cudaStreamDestroy(upload_stream);
        if (status != cudaSuccess) {
            return std::unexpected(std::format("failed to upload environment map {} to CUDA: {} ({})",
                                               (*image)->path.string(),
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }
        if (destroy_status != cudaSuccess) {
            return std::unexpected(std::format("failed to destroy environment upload stream: {} ({})",
                                               cudaGetErrorName(destroy_status),
                                               cudaGetErrorString(destroy_status)));
        }

        cache.path = (*image)->path;
        cache.map = map;
        return cache.map;
    }

    void releaseCudaEnvironmentMapCache() {
        auto& cache = cudaEnvironmentMapCache();
        std::lock_guard lock(cache.mutex);
        cache.map.reset();
        cache.path.clear();
    }

    void releaseEnvironmentMapCaches() {
        releaseCudaEnvironmentMapCache();
        releaseEnvironmentImageCache();
    }

    ExportResult<void> unpackU8HwcBandToChwFloat(const lfs::core::Tensor& band_u8_hwc,
                                                 lfs::core::Tensor& rgb_chw_out,
                                                 lfs::core::Tensor* const alpha_out,
                                                 const cudaStream_t stream) {
        if (auto valid = validateBandU8Hwc(band_u8_hwc, "unpack source"); !valid) {
            return valid;
        }

        const auto height = band_u8_hwc.size(0);
        const auto width = band_u8_hwc.size(1);
        const int channels = static_cast<int>(band_u8_hwc.size(2));
        const int num_pixels = static_cast<int>(height * width);

        rgb_chw_out = lfs::core::Tensor::empty(
            {size_t{3}, static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        if (!rgb_chw_out.is_valid()) {
            return std::unexpected("failed to allocate unpacked RGB band");
        }

        float* alpha_ptr = nullptr;
        if (alpha_out != nullptr) {
            *alpha_out = lfs::core::Tensor{};
            if (channels == 4) {
                *alpha_out = lfs::core::Tensor::empty(
                    {static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                if (!alpha_out->is_valid()) {
                    return std::unexpected("failed to allocate unpacked alpha band");
                }
                alpha_ptr = alpha_out->ptr<float>();
            }
        }

        return launchStatus("unpack_u8_hwc_kernel",
                            exportpp::launchUnpackU8Hwc(band_u8_hwc.ptr<unsigned char>(), num_pixels, channels,
                                                        rgb_chw_out.ptr<float>(), alpha_ptr,
                                                        resolveExportStream(stream)));
    }

    ExportResult<void> packChwFloatBandToU8Hwc(const lfs::core::Tensor& rgb_chw,
                                               const lfs::core::Tensor* const alpha,
                                               lfs::core::Tensor& band_u8_hwc_out,
                                               const cudaStream_t stream) {
        if (auto valid = validateRgbChw(rgb_chw, "pack source"); !valid) {
            return valid;
        }

        const auto height = rgb_chw.size(1);
        const auto width = rgb_chw.size(2);
        const int num_pixels = static_cast<int>(height * width);
        const bool has_alpha = alpha != nullptr && alpha->is_valid();
        if (has_alpha &&
            (alpha->device() != lfs::core::Device::CUDA || alpha->dtype() != lfs::core::DataType::Float32 ||
             !alpha->is_contiguous() || alpha->numel() != static_cast<size_t>(num_pixels))) {
            return std::unexpected("pack alpha must be a contiguous CUDA float tensor matching the band");
        }
        const int channels = has_alpha ? 4 : 3;

        band_u8_hwc_out = lfs::core::Tensor::empty(
            {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)},
            lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        if (!band_u8_hwc_out.is_valid()) {
            return std::unexpected("failed to allocate packed u8 band");
        }

        return launchStatus("pack_chw_u8_hwc_kernel",
                            exportpp::launchPackChwU8Hwc(rgb_chw.ptr<float>(),
                                                         has_alpha ? alpha->ptr<float>() : nullptr,
                                                         band_u8_hwc_out.ptr<unsigned char>(), num_pixels, channels,
                                                         resolveExportStream(stream)));
    }

    ExportResult<void> compositeEnvironmentBand(const CudaEnvironmentMap& env,
                                                const EnvironmentCompositeBandParams& params,
                                                const lfs::core::Tensor& rgb_chw,
                                                const lfs::core::Tensor& alpha,
                                                lfs::core::Tensor& band_u8_hwc_out,
                                                const cudaStream_t stream) {
        if (auto valid = validateRgbChw(rgb_chw, "composite source"); !valid) {
            return valid;
        }
        if (env.width <= 0 || env.height <= 0 || !env.pixels.is_valid() ||
            env.pixels.device() != lfs::core::Device::CUDA ||
            env.pixels.dtype() != lfs::core::DataType::Float32) {
            return std::unexpected("composite environment map is not resident on CUDA");
        }

        const auto height = rgb_chw.size(1);
        const auto width = rgb_chw.size(2);
        const int num_pixels = static_cast<int>(height * width);
        if (!alpha.is_valid() || alpha.device() != lfs::core::Device::CUDA ||
            alpha.dtype() != lfs::core::DataType::Float32 || !alpha.is_contiguous() ||
            alpha.numel() != static_cast<size_t>(num_pixels)) {
            return std::unexpected("composite alpha must be a contiguous CUDA float tensor matching the band");
        }
        if (params.full_size.x != static_cast<int>(width) || params.y_offset < 0 ||
            params.y_offset + static_cast<int>(height) > params.full_size.y) {
            return std::unexpected("composite band region does not match the full image size");
        }

        band_u8_hwc_out = lfs::core::Tensor::empty(
            {static_cast<size_t>(height), static_cast<size_t>(width), size_t{3}},
            lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        if (!band_u8_hwc_out.is_valid()) {
            return std::unexpected("failed to allocate composited u8 band");
        }

        exportpp::CompositeParams device_params;
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

        const cudaStream_t execution_stream = resolveExportStream(stream);
        // The environment upload is complete before it enters the cache, so
        // only lifetime tracking is needed here. The band inputs may still be
        // produced elsewhere and need explicit ordering onto this stream.
        env.pixels.record_stream(execution_stream);
        rgb_chw.sync_to_stream(execution_stream);
        alpha.sync_to_stream(execution_stream);
        band_u8_hwc_out.set_stream(execution_stream);
        return launchStatus(
            "composite_environment_band_kernel",
            exportpp::launchCompositeEnvironmentBand(device_params,
                                                     env.pixels.ptr<float>(),
                                                     rgb_chw.ptr<float>(),
                                                     alpha.ptr<float>(),
                                                     band_u8_hwc_out.ptr<unsigned char>(),
                                                     execution_stream));
    }

} // namespace lfs::rendering
