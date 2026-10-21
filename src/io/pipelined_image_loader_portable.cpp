/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_codecs.hpp"
#include "core/image_exr.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/shared_image_ops.hpp"
#include "io/pipelined_image_loader.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace lfs::io {

    namespace {

        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;
        using lfs::core::TensorShape;

        constexpr float UINT8_SCALE = 1.0f / 255.0f;
        constexpr float UINT16_SCALE = 1.0f / 65535.0f;

        using ReleaseFn = void (*)(void*);

        // stb's filename API copies the path into a fixed buffer and opens it plainly, so long paths go
        // through lfs::core::open_file and stb's FILE readers instead.
        class StbFile {
        public:
            explicit StbFile(const std::filesystem::path& path) : file_(lfs::core::open_file(path, "rb")) {}
            ~StbFile() {
                if (file_)
                    std::fclose(file_);
            }
            StbFile(const StbFile&) = delete;
            StbFile& operator=(const StbFile&) = delete;
            std::FILE* get() const { return file_; }

        private:
            std::FILE* file_;
        };

        bool stb_is_16_bit(const std::filesystem::path& path) {
            const StbFile file(path);
            return file.get() && stbi_is_16_bit_from_file(file.get());
        }

        stbi_uc* stb_load(const std::filesystem::path& path, int* w, int* h, int* channels, const int desired) {
            const StbFile file(path);
            return file.get() ? stbi_load_from_file(file.get(), w, h, channels, desired) : nullptr;
        }

        stbi_us* stb_load_16(const std::filesystem::path& path, int* w, int* h, int* channels, const int desired) {
            const StbFile file(path);
            return file.get() ? stbi_load_from_file_16(file.get(), w, h, channels, desired) : nullptr;
        }

        TensorShape image_shape(const int height, const int width, const int channels) {
            if (channels == 1)
                return TensorShape({static_cast<size_t>(height), static_cast<size_t>(width)});
            return TensorShape({static_cast<size_t>(height), static_cast<size_t>(width),
                                static_cast<size_t>(channels)});
        }

        // Takes ownership of a decoder allocation of 8-bit samples.
        Tensor host_uint8(void* const data, TensorShape shape, const ReleaseFn release) {
            return Tensor::from_external_owner(data, std::move(shape), Device::CPU, DataType::UInt8,
                                               std::shared_ptr<void>(data, release));
        }

        // Takes ownership of decoded interleaved 8-bit samples and returns them
        // planar, so the device copy needs no transpose.
        Tensor host_uint8_planar(void* const data, const int height, const int width, const int channels,
                                 const ReleaseFn release) {
            const std::unique_ptr<void, ReleaseFn> owner(data, release);
            const size_t pixels = static_cast<size_t>(height) * static_cast<size_t>(width);
            const auto c = static_cast<size_t>(channels);
            auto planar = Tensor::empty_pageable_host(
                TensorShape({c, static_cast<size_t>(height), static_cast<size_t>(width)}), DataType::UInt8);
            const auto* const src = static_cast<const uint8_t*>(data);
            auto* const dst = planar.ptr<uint8_t>();
            for (size_t ch = 0; ch < c; ++ch) {
                uint8_t* const plane = dst + ch * pixels;
                for (size_t i = 0; i < pixels; ++i)
                    plane[i] = src[i * c + ch];
            }
            return planar;
        }

        // 16-bit samples have no tensor dtype; Int32 holds them exactly.
        Tensor host_uint16(void* const data, TensorShape shape, const ReleaseFn release) {
            const std::unique_ptr<void, ReleaseFn> owner(data, release);
            auto widened = Tensor::empty_pageable_host(std::move(shape), DataType::Int32);
            std::copy_n(static_cast<const uint16_t*>(data), widened.numel(), widened.ptr<int32_t>());
            return widened;
        }

        // Upload slots are reused only after the previous copy completed.
        Tensor to_device(lfs::core::TensorUpload& upload, const Tensor& host) {
            upload.wait();
            auto device = Tensor::empty(host.shape(), Device::GPU, host.dtype());
            upload.enqueue(device, host);
            // The upload retains its source; GPU consumers follow the
            // destination timeline without blocking this decode worker.
            return device;
        }

        Tensor float_to_uint8(const Tensor& image) {
            return image.clamp(0.0f, 1.0f).mul(255.0f).add(0.5f).to(DataType::UInt8).contiguous();
        }

        lfs::core::UndistortParams undistort_for(const lfs::core::UndistortParams& params,
                                                 const Tensor& image,
                                                 const int resize_factor, const int max_width) {
            const size_t rank = image.ndim();
            return lfs::core::prepare_undistort_params(
                params,
                static_cast<int>(image.shape()[rank - 1]),
                static_cast<int>(image.shape()[rank - 2]),
                resize_factor, max_width);
        }

        // Inversion precedes binarization; threshold compares with >=.
        Tensor finish_mask(Tensor mask, const MaskParams& params) {
            const auto backend = lfs::core::gpu_backend_of(mask).value_or(lfs::core::default_gpu_backend());
            const auto* const ops = lfs::core::shared_image_ops(backend);
            if (params.invert && ops && ops->mask)
                ops->mask(mask, lfs::gpu_ops::MaskTransform::Invert, 0.0f);
            if (params.threshold > 0.0f && ops && ops->mask)
                ops->mask(mask, lfs::gpu_ops::MaskTransform::Threshold, params.threshold);
            else
                mask = mask.clamp(0.0f, 1.0f);
            return mask.contiguous();
        }

        std::pair<Tensor, Tensor> decode_rgba(const std::filesystem::path& path,
                                              const LoadParams& params,
                                              const bool decode_16bit,
                                              const bool alpha_is_transparency) {
            const auto rgba = load_rgba_image_cpu_decoded(path, params.undistort ? 1 : params.resize_factor,
                                                          params.undistort ? 0 : params.max_width, nullptr,
                                                          decode_16bit, !alpha_is_transparency);
            Tensor rgb = rgba.slice(0, 0, 3).contiguous();
            if (params.output_uint8)
                rgb = float_to_uint8(rgb);
            Tensor alpha = rgba.slice(0, 3, 4).squeeze(0).contiguous();
            if (params.undistort) {
                if (params.output_uint8)
                    rgb = rgb.to(DataType::Float32).div(255.0f);
                const auto scaled = undistort_for(*params.undistort, alpha, params.resize_factor, params.max_width);
                rgb = lfs::core::undistort_image(rgb.contiguous(), scaled, nullptr);
                alpha = lfs::core::undistort_mask(alpha.contiguous(), scaled, nullptr);
                if (params.output_uint8)
                    rgb = float_to_uint8(rgb);
            }
            alpha = alpha.clamp(0.0f, 1.0f).mul(65535.0f).add(0.5f).to(DataType::Int32).to(DataType::Float32).div(65535.0f);
            return {rgb.contiguous(), alpha};
        }

        // Normal maps store v = n * 0.5 + 0.5, optionally through the sRGB display transform.
        Tensor decode_normal_prior(const Tensor& samples,
                                   const float scale,
                                   const bool srgb,
                                   const bool flip_yz,
                                   const std::array<float, 9>* world_to_camera) {
            Tensor encoded = samples.permute({2, 0, 1}).to(DataType::Float32).mul(scale);
            if (srgb) {
                encoded = Tensor::where(encoded.le(0.04045f),
                                        encoded.div(12.92f),
                                        encoded.add(0.055f).div(1.055f).pow(2.4f));
            }
            encoded = encoded.mul(2.0f).sub(1.0f);
            Tensor x = encoded.slice(0, 0, 1);
            Tensor y = encoded.slice(0, 1, 2);
            Tensor z = encoded.slice(0, 2, 3);
            if (flip_yz) {
                y = y.neg();
                z = z.neg();
            }
            if (world_to_camera) {
                const auto& m = *world_to_camera;
                const auto row = [&](const size_t r) {
                    return x.mul(m[r * 3]).add(y.mul(m[r * 3 + 1])).add(z.mul(m[r * 3 + 2]));
                };
                return Tensor::cat({row(0), row(1), row(2)}, 0).contiguous();
            }
            return Tensor::cat({x, y, z}, 0).contiguous();
        }

    } // namespace

    bool PipelinedImageLoader::uses_portable_prepared_cache(const LoadParams& params) const {
        if (!params.undistort || params.skip_blob_cache || params.masked_fill)
            return false;
        if (!(decodes_16bit(params) || params.decode_float))
            return true;
        if (params.output_uint8)
            return false;
        if (params.undistort->dst_width <= 0 || params.undistort->dst_height <= 0)
            return false;
        const auto grid = lfs::core::compute_undistort_grid(*params.undistort, params.resize_factor, params.max_width);
        const size_t sample_bytes = params.decode_float ? sizeof(float) : sizeof(uint16_t);
        // Bound the extra interleaved RGB staging image as well as encoded extent.
        // Large outputs keep the original path rather than repeatedly building
        // an optional cache that may exhaust its byte limit.
        return grid.width > 0 && grid.height > 0 &&
               uint64_t(grid.width) * uint64_t(grid.height) <= PORTABLE_CACHE_MAX_BYTES / (3 * sample_bytes);
    }

    void PipelinedImageLoader::warn_portable_cache_failure(const std::string_view message) {
        if (!portable_cache_warning_logged_.exchange(true))
            LOG_WARN("[PipelinedImageLoader] Optional prepared image cache unavailable: {}; continuing with source image preparation", message);
    }

    void PipelinedImageLoader::cache_portable_prepared_image(
        const std::string& key, const Tensor& image, const size_t max_output_bytes, const bool lossless) {
        // Cache failure must never replace or invalidate the already prepared image.
        try {
            if (!running_.load(std::memory_order_acquire))
                return;
            if (lossless && image.dtype() == DataType::Int32) {
                if (image.ndim() != 3 || image.shape()[0] != 3)
                    throw std::runtime_error("prepared PNG cache expects UInt16 RGB");
                const auto pixels = image.to_pageable_host();
                const size_t width = pixels.shape()[2], height = pixels.shape()[1], plane = width * height;
                std::vector<uint16_t> interleaved(pixels.numel());
                const auto* src = pixels.ptr<int32_t>();
                for (size_t i = 0; i < plane; ++i)
                    for (size_t ch = 0; ch < 3; ++ch) {
                        const auto value = src[ch * plane + i];
                        if (value < 0 || value > 65535)
                            throw std::runtime_error("prepared PNG cache sample is outside the UInt16 grid");
                        interleaved[i * 3 + ch] = static_cast<uint16_t>(value);
                    }
                lfs::core::image_codecs::PngEncodeOptions options;
                options.compression_level = 3;
                options.max_output_bytes = max_output_bytes;
                std::vector<uint8_t> bytes;
                const auto status = lfs::core::image_codecs::encode_png_memory(
                    {reinterpret_cast<const uint8_t*>(interleaved.data()), interleaved.size() * sizeof(uint16_t)},
                    static_cast<int>(width), static_cast<int>(height), 3, 16, width * 3 * sizeof(uint16_t), bytes, options);
                if (!status) {
                    LOG_DEBUG("[PipelinedImageLoader] Optional prepared image cache PNG encoding failed for {}: {}", key, status.error().detail());
                    warn_portable_cache_failure(status.error().detail());
                    return;
                }
                if (!running_.load(std::memory_order_acquire))
                    return;
                bytes.shrink_to_fit();
                put_in_jpeg_cache(key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
                return;
            }
            if (lossless) {
                if (image.dtype() != DataType::Float32 || image.ndim() != 3 || image.shape()[0] != 3)
                    throw std::runtime_error("prepared EXR cache expects Float32 RGB");
                // Explicit decode_float requests retain their native values.
                // Training currently uses the UInt16 grid branch above.
                const auto pixels = image.to_pageable_host();
                const size_t width = pixels.shape()[2], height = pixels.shape()[1];
                const size_t plane = width * height;
                std::vector<float> interleaved(pixels.numel());
                const auto* src = pixels.ptr<float>();
                for (size_t i = 0; i < plane; ++i)
                    for (size_t ch = 0; ch < 3; ++ch)
                        interleaved[i * 3 + ch] = src[ch * plane + i];
                lfs::core::image_codecs::ExrEncodeOptions options;
                options.precision = lfs::core::image_codecs::ExrSampleType::Float;
                options.cancelled = [this] { return !running_.load(std::memory_order_acquire); };
                std::vector<uint8_t> bytes;
                const auto status = lfs::core::image_codecs::encode_exr_memory(
                    {static_cast<int>(width), static_cast<int>(height), 3, width * 3 * sizeof(float), {reinterpret_cast<const uint8_t*>(interleaved.data()), interleaved.size() * sizeof(float)}},
                    bytes, options, max_output_bytes);
                if (!status) {
                    if (status.error().code() == lfs::ErrorCode::Cancelled)
                        return;
                    LOG_DEBUG("[PipelinedImageLoader] Optional prepared image cache EXR encoding failed for {}: {}", key, status.error().detail());
                    warn_portable_cache_failure(status.error().detail());
                    return;
                }
                bytes.shrink_to_fit();
                put_in_jpeg_cache(key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
                return;
            }
            // Match the existing derived RGB cache: truncate normalized floats
            // to bytes and preserve full chroma at the configured JPEG quality.
            auto pixels = (image.dtype() == DataType::UInt8
                               ? image
                               : image.mul(255.0f).clamp(0.0f, 255.0f).to(DataType::UInt8))
                              .to_pageable_host();
            const size_t width = pixels.shape()[2], height = pixels.shape()[1];
            const size_t plane = width * height;
            std::vector<uint8_t> interleaved(pixels.numel());
            const auto* src = pixels.ptr<uint8_t>();
            for (size_t i = 0; i < plane; ++i)
                for (size_t ch = 0; ch < 3; ++ch)
                    interleaved[i * 3 + ch] = src[ch * plane + i];
            lfs::core::image_codecs::JpegEncodeOptions options;
            options.quality = config_.cache_jpeg_quality;
            options.full_chroma = true;
            options.max_output_bytes = max_output_bytes;
            std::vector<uint8_t> bytes;
            const auto status = lfs::core::image_codecs::encode_jpeg_memory(
                interleaved, static_cast<int>(width), static_cast<int>(height), 3,
                width * 3, bytes, options);
            if (!status) {
                LOG_DEBUG("[PipelinedImageLoader] Optional prepared image cache JPEG encoding failed for {}: {}", key, status.error().detail());
                warn_portable_cache_failure(status.error().detail());
                return;
            }
            bytes.shrink_to_fit();
            put_in_jpeg_cache(key, std::make_shared<std::vector<uint8_t>>(std::move(bytes)));
        } catch (const std::exception& e) {
            LOG_DEBUG("[PipelinedImageLoader] Optional prepared image cache store failed for {}: {}", key, e.what());
            warn_portable_cache_failure(e.what());
        }
    }

    lfs::core::Tensor PipelinedImageLoader::decode_portable_rgb(
        const std::filesystem::path& path,
        const LoadParams& params,
        lfs::core::TensorUpload& upload,
        const std::vector<uint8_t>* const encoded) {
        {
            std::lock_guard stats_lock(stats_mutex_);
            ++stats_.cpu_decode_calls;
        }
        // Preserve the existing lossy JPEG policy for 8-bit RGB. High-precision
        // Quantized results use PNG16, explicit floats EXR; evaluation and mask filling bypass.
        std::string prepared_key;
        const bool lossless_cache = decodes_16bit(params) || params.decode_float;
        if (uses_portable_prepared_cache(params)) {
            try {
                prepared_key = make_cache_key(path, params);
                const auto cached = prepared_key.empty() ? nullptr : load_cached_jpeg_blob(prepared_key);
                if (cached) {
                    const auto grid = lfs::core::compute_undistort_grid(*params.undistort, params.resize_factor, params.max_width);
                    Tensor host;
                    if (lossless_cache && !params.decode_float) {
                        constexpr std::array<uint8_t, 8> png_signature{137, 80, 78, 71, 13, 10, 26, 10};
                        if (cached->size() < png_signature.size() || !std::equal(png_signature.begin(), png_signature.end(), cached->begin()))
                            throw std::runtime_error("invalid prepared PNG bytes");
                        struct BoundedPixels {
                            size_t expected;
                            std::vector<uint16_t> values;
                        } pixels{size_t(grid.width) * grid.height * 3 * sizeof(uint16_t), {}};
                        lfs::core::image_codecs::DecodeTarget target;
                        target.channels = 3;
                        target.sample_type = lfs::core::image_codecs::SampleType::UInt16;
                        target.user = &pixels;
                        target.allocate = [](size_t bytes, void* user) -> void* {
                            auto& buffer = *static_cast<BoundedPixels*>(user);
                            if (bytes != buffer.expected)
                                return nullptr;
                            buffer.values.resize(bytes / sizeof(uint16_t));
                            return buffer.values.data();
                        };
                        lfs::core::image_codecs::Probe decoded;
                        std::string error;
                        if (!lfs::core::image_codecs::decode_memory_to_buffer(cached->data(), cached->size(), target, decoded, error))
                            throw std::runtime_error("invalid prepared PNG bytes: " + error);
                        if (decoded.width != grid.width || decoded.height != grid.height)
                            throw std::runtime_error("prepared PNG dimensions do not match the requested warp");
                        host = Tensor::empty_pageable_host({3, size_t(grid.height), size_t(grid.width)}, DataType::Int32);
                        const size_t plane = size_t(grid.width) * grid.height;
                        auto* dst = host.ptr<int32_t>();
                        for (size_t i = 0; i < plane; ++i)
                            for (size_t ch = 0; ch < 3; ++ch)
                                dst[ch * plane + i] = pixels.values[i * 3 + ch];
                    } else if (lossless_cache) {
                        lfs::core::image_codecs::Image decoded;
                        std::string error;
                        const size_t decoded_limit = size_t(grid.width) * grid.height * 4 * sizeof(float);
                        if (!lfs::core::image_codecs::decode_exr_memory(*cached, decoded, error, 1, decoded_limit))
                            throw std::runtime_error("invalid prepared EXR bytes: " + error);
                        if (decoded.width != grid.width || decoded.height != grid.height || decoded.channels != 4 ||
                            decoded.sample_type != lfs::core::image_codecs::SampleType::Float32)
                            throw std::runtime_error("prepared EXR layout does not match the requested warp");
                        host = Tensor::empty_pageable_host({3, size_t(grid.height), size_t(grid.width)}, DataType::Float32);
                        auto* dst = host.ptr<float>();
                        const size_t plane = size_t(grid.width) * grid.height;
                        for (size_t i = 0; i < plane; ++i)
                            for (size_t ch = 0; ch < 3; ++ch) {
                                std::memcpy(dst + ch * plane + i, decoded.data.data() + (i * 4 + ch) * sizeof(float), sizeof(float));
                                // Our encoder only accepts finite samples. A
                                // nonfinite cached value cannot be its output.
                                if (!std::isfinite(dst[ch * plane + i]))
                                    throw std::runtime_error("prepared EXR contains a nonfinite sample");
                            }
                    } else {
                        auto [data, width, height, channels] = lfs::core::load_image_from_memory(cached->data(), cached->size(), 1, 0);
                        if (!data)
                            throw std::runtime_error("invalid prepared JPEG bytes");
                        host = host_uint8_planar(data, height, width, channels, lfs::core::free_image);
                        if (channels != 3 || width != grid.width || height != grid.height)
                            throw std::runtime_error("prepared JPEG dimensions do not match the requested warp");
                    }
                    auto image = to_device(upload, host);
                    if (lossless_cache && !params.decode_float)
                        image = image.to(DataType::Float32).div(65535.0f);
                    else if (!lossless_cache && !params.output_uint8)
                        image = image.to(DataType::Float32).mul(UINT8_SCALE);
                    {
                        std::lock_guard stats_lock(stats_mutex_);
                        ++stats_.hot_path_hits;
                    }
                    // The upload retains only this small prepared CPU image;
                    // consumers follow its fence without blocking this worker.
                    return image;
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("[PipelinedImageLoader] Optional prepared image cache read failed for {}: {}", lfs::core::path_to_utf8(path), e.what());
                if (!prepared_key.empty())
                    invalidate_cache_entry(prepared_key);
                warn_portable_cache_failure(e.what());
            }
        }
        const bool float_decode = decodes_float(path, params);
        if (!params.undistort && !float_decode && !encoded)
            return load_rgb_decoded_ahead(path, params, decodes_16bit(params));
        Tensor host;
        if (float_decode || (params.undistort && decodes_16bit(params))) {
            auto ahead = take_decoded_ahead(path, HostDecodeKind::Float32);
            auto [data, width, height, channels] = ahead
                                                       ? std::tuple{static_cast<float*>(ahead->data.release()), ahead->width, ahead->height, ahead->channels}
                                                       : lfs::core::load_image_float(path);
            if (!data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            const std::unique_ptr<float, decltype(&lfs::core::free_image_float)> owner(data, lfs::core::free_image_float);
            host = Tensor::from_blob(data, {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)}, lfs::core::Device::CPU, DataType::Float32).clone();
            if (channels <= 2) {
                auto gray = host.slice(2, 0, 1);
                host = Tensor::cat({gray, gray, gray}, 2);
            } else if (channels == 4) {
                host = host.slice(2, 0, 3);
            }
            host = host.permute({2, 0, 1}).contiguous();
        } else if (decodes_16bit(params)) {
            auto [data, width, height, channels] = lfs::core::load_image_u16(path, params.resize_factor, params.max_width);
            if (!data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            host = host_uint16(data, image_shape(height, width, channels), lfs::core::free_image)
                       .permute({2, 0, 1})
                       .contiguous();
        } else {
            // Warps consume full-resolution UInt8 directly; do not resize or
            // expand these sources to float before the area warp.
            const int resize_factor = params.undistort ? 1 : params.resize_factor;
            const int max_width = params.undistort ? 0 : params.max_width;
            auto ahead = params.undistort ? take_decoded_ahead(path, HostDecodeKind::UInt8) : std::nullopt;
            auto [data, width, height, channels] =
                ahead ? std::tuple{static_cast<unsigned char*>(ahead->data.release()), ahead->width, ahead->height, ahead->channels}
                : encoded != nullptr
                    ? lfs::core::load_image_from_memory(encoded->data(), encoded->size(), resize_factor, max_width)
                    : lfs::core::load_image(path, resize_factor, max_width);
            if (!data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            // Planar bytes avoid a second device image and a per-step GPU transpose.
            host = host_uint8_planar(data, height, width, channels, lfs::core::free_image);
        }
        std::unique_lock<std::mutex> large_prepare_lock;
        constexpr size_t LARGE_PORTABLE_SOURCE_BYTES = size_t{128} * 1024 * 1024;
        if (params.undistort && host.bytes() > LARGE_PORTABLE_SOURCE_BYTES) {
            large_prepare_lock = std::unique_lock(portable_large_prepare_mutex_);
            if (!portable_large_prepare_logged_) {
                const char* backend_name = config_.backend == lfs::core::GpuBackend::Vulkan ? "Vulkan" : "Metal";
#ifdef LFS_TENSOR_HIP
                if (config_.backend == lfs::core::GpuBackend::HIP)
                    backend_name = "HIP";
#endif
                LOG_INFO("{} image preparation: limiting large full-resolution GPU sources to one at a time (source {:.1f} MiB); CPU decoding and compressed prepared-image caching remain in RAM",
                         backend_name,
                         host.bytes() / (1024.0 * 1024.0));
                portable_large_prepare_logged_ = true;
            }
        }
        Tensor image = to_device(upload, host);
        if (!params.undistort && float_decode) {
            const auto [target_width, target_height] = lfs::core::resized_image_dimensions(static_cast<int>(image.shape()[2]), static_cast<int>(image.shape()[1]), params.resize_factor, params.max_width);
            if (target_width != static_cast<int>(image.shape()[2]) || target_height != static_cast<int>(image.shape()[1]))
                image = lfs::core::shared_image_ops(config_.backend)->resize(image, target_height, target_width, lfs::gpu_ops::Resample::LanczosFloatCHW, 2);
            if (params.output_uint8)
                image = float_to_uint8(image);
        } else if (!params.undistort && decodes_16bit(params)) {
            image = image.to(DataType::Float32).mul(UINT16_SCALE);
            if (params.output_uint8)
                image = float_to_uint8(image);
        } else if (!params.undistort && !params.output_uint8) {
            image = image.to(DataType::Float32).mul(UINT8_SCALE);
        }

        Tensor cache_grid;
        if (params.undistort) {
            const bool restore_uint8 = params.output_uint8;
            image = lfs::core::undistort_image(
                image.contiguous(), undistort_for(*params.undistort, image, params.resize_factor, params.max_width), nullptr);
            if (restore_uint8)
                image = float_to_uint8(image);
            else if (!params.decode_float) {
                const auto quantized = image.clamp(0.0f, 1.0f).mul(65535.0f).add(0.5f).to(DataType::Int32);
                if (!prepared_key.empty() && lossless_cache)
                    cache_grid = quantized;
                image = quantized.to(DataType::Float32).div(65535.0f);
            }
        }
        image = image.contiguous();
        if (!prepared_key.empty()) {
            // PNG can expand incompressible UInt16 rows. Keep the staging bound,
            // but allow 25% encoded headroom for filter bytes, deflate and chunks.
            const size_t encoded_limit = cache_grid.is_valid() ? PORTABLE_PNG_CACHE_MAX_BYTES
                                                               : PORTABLE_CACHE_MAX_BYTES;
            cache_portable_prepared_image(prepared_key, cache_grid.is_valid() ? cache_grid : image, encoded_limit, lossless_cache);
        }
        if (params.undistort) {
            if (large_prepare_lock.owns_lock()) {
                // Keep the permit until the large GPU source can be retired.
                lfs::core::TensorCompletion ready;
                ready.include(image);
                ready.wait();
            }
            // Settle the original upload on a miss, so an idle worker does not
            // retain a full-resolution device source. Hits return above.
            upload.wait();
        }
        return image;
    }

    void PipelinedImageLoader::portable_process_thread_func() {
        const lfs::core::GpuBackendScope backend(config_.backend);
        // A worker upload must not order unrelated prefetched images against
        // the legacy training queue. Consumers follow each image storage fence.
        std::unique_ptr<lfs::core::TensorWorkQueue> queue;
        std::unique_ptr<lfs::core::TensorWorkQueue::Scope> queue_scope;
        if (config_.backend == lfs::core::GpuBackend::Vulkan) {
            queue = std::make_unique<lfs::core::TensorWorkQueue>(config_.backend);
            queue_scope = std::make_unique<lfs::core::TensorWorkQueue::Scope>(*queue);
        }
        lfs::core::TensorUpload upload;
        while (running_) {
            PrefetchedImage item;
            try {
                item = cold_queue_.pop();
            } catch (const std::runtime_error&) {
                break;
            }
            LoadParams params = item.params;
            params.undistort = item.undistort;

            const auto fail = [&](const std::string& failure) {
                if (item.alpha_as_mask) {
                    try {
                        auto image = decode_portable_rgb(item.path, params, upload);
                        {
                            std::lock_guard<std::mutex> lock(pending_pairs_mutex_);
                            if (auto it = pending_pairs_.find(item.sequence_id);
                                it != pending_pairs_.end() &&
                                it->second.loader_generation == item.loader_generation) {
                                it->second.mask_failed = true;
                                it->second.mask_expected = false;
                            }
                        }
                        try_complete_pair(item.sequence_id, item.loader_generation, std::move(image),
                                          std::nullopt);
                    } catch (const std::exception& e) {
                        LOG_ERROR("[PipelinedImageLoader] RGB fallback also failed {}: {}",
                                  lfs::core::path_to_utf8(item.path), e.what());
                        publish_image_failure(item.sequence_id, item.loader_generation, item.path, e.what());
                    } catch (...) {
                        LOG_ERROR("[PipelinedImageLoader] RGB fallback also failed {}",
                                  lfs::core::path_to_utf8(item.path));
                        publish_image_failure(item.sequence_id, item.loader_generation, item.path,
                                              "non-standard image loader exception");
                    }
                } else if (item.is_mask || item.is_depth || item.is_normal) {
                    std::unique_lock<std::mutex> lock(pending_pairs_mutex_);
                    const auto kind = item.is_mask    ? SidecarKind::Mask
                                      : item.is_depth ? SidecarKind::Depth
                                                      : SidecarKind::Normal;
                    fail_sidecar_locked(item.sequence_id, item.loader_generation, kind, item.path, failure, lock);
                } else {
                    publish_image_failure(item.sequence_id, item.loader_generation, item.path, failure);
                }
            };

            try {
                if (item.alpha_as_mask) {
                    auto [rgb, alpha] = decode_rgba(item.path, params, config_.use_16bit_color,
                                                    item.alpha_is_transparency);
                    try_complete_pair(item.sequence_id, item.loader_generation, std::move(rgb),
                                      finish_mask(std::move(alpha), item.alpha_mask_params));
                } else if (item.is_mask) {
                    int width = 0, height = 0, channels = 0;
                    stbi_uc* const gray = stb_load(item.path, &width, &height, &channels, 1);
                    if (!gray)
                        throw std::runtime_error("Failed to decode mask");
                    const auto [target_w, target_h] = sidecar_target_size(item, width, height);
                    Tensor mask;
                    if (target_w == width && target_h == height) {
                        auto bytes = to_device(upload, host_uint8(gray, image_shape(height, width, 1), stbi_image_free));
                        mask = Tensor::empty(bytes.shape(), Device::GPU, DataType::Float32);
                        lfs::core::shared_image_ops(config_.backend)->convert(bytes, mask, lfs::gpu_ops::ImageConversion::MaskU8HWToF32HW, static_cast<size_t>(height), static_cast<size_t>(width), 1, {});
                    } else {
                        auto bytes = to_device(upload, host_uint8(gray, image_shape(height, width, 1), stbi_image_free));
                        Tensor source = Tensor::empty(bytes.shape(), Device::GPU, DataType::Float32);
                        lfs::core::shared_image_ops(config_.backend)->convert(bytes, source, lfs::gpu_ops::ImageConversion::MaskU8HWToF32HW, static_cast<size_t>(height), static_cast<size_t>(width), 1, {});
                        mask = lfs::core::shared_image_ops(config_.backend)->resize(source, target_h, target_w, lfs::gpu_ops::Resample::LanczosGray, 2);
                    }
                    if (item.undistort) {
                        const auto scaled = undistort_for(*item.undistort, mask, params.resize_factor, params.max_width);
                        const auto* const ops = lfs::core::shared_image_ops(config_.backend);
                        mask = ops && ops->undistort ? ops->undistort(mask.contiguous(), scaled, true)
                                                     : lfs::core::undistort_mask(mask.contiguous(), scaled, nullptr);
                    }
                    if (const auto* const ops = lfs::core::shared_image_ops(config_.backend); ops && ops->mask)
                        ops->mask(mask, lfs::gpu_ops::MaskTransform::QuantizeByteGrid, 0.0f);
                    try_complete_pair(item.sequence_id, item.loader_generation, std::nullopt,
                                      finish_mask(std::move(mask), item.mask_params));
                } else if (item.is_depth || item.is_normal) {
                    const int channels = item.is_depth ? 1 : 3;
                    const bool sixteen_bit = stb_is_16_bit(item.path);
                    int width = 0, height = 0, source_channels = 0;
                    Tensor samples;
                    if (sixteen_bit) {
                        stbi_us* const data = stb_load_16(item.path, &width, &height, &source_channels, channels);
                        if (!data)
                            throw std::runtime_error(item.is_depth ? "Failed to decode 16-bit depth"
                                                                   : "Failed to decode 16-bit normal map");
                        samples = host_uint16(data, image_shape(height, width, channels), stbi_image_free);
                    } else {
                        stbi_uc* const data = stb_load(item.path, &width, &height, &source_channels, channels);
                        if (data) {
                            samples = host_uint8(data, image_shape(height, width, channels), stbi_image_free);
                        } else {
                            auto [pixels, w, h, count] = lfs::core::load_image_float(item.path);
                            if (!pixels || count != channels) {
                                if (pixels)
                                    lfs::core::free_image_float(pixels);
                                throw std::runtime_error(item.is_depth ? "Failed to decode depth" : "Failed to decode normal map");
                            }
                            width = w;
                            height = h;
                            samples = Tensor::from_blob(pixels, image_shape(height, width, channels),
                                                        Device::CPU, DataType::Float32)
                                          .clone();
                            lfs::core::free_image_float(pixels);
                        }
                    }
                    samples = to_device(upload, samples);

                    const auto [target_w, target_h] = sidecar_target_size(item, width, height);
                    Tensor prior;
                    if (item.is_depth) {
                        prior = samples.dtype() == DataType::Float32 ? samples
                                : sixteen_bit                        ? samples.to(DataType::Float32).mul(UINT16_SCALE)
                                                                     : samples.to(DataType::Float32).div(255.0f);
                    } else {
                        prior = decode_normal_prior(samples, samples.dtype() == DataType::Float32 ? 1.0f : sixteen_bit ? UINT16_SCALE
                                                                                                                       : UINT8_SCALE,
                                                    item.normal_srgb, item.normal_flip_yz,
                                                    item.normal_transform_world_to_camera ? &item.normal_world_to_camera
                                                                                          : nullptr);
                    }
                    const auto resize_prior = [&item](const Tensor& input, const int height, const int width) {
                        return item.is_depth ? lfs::core::resize_depth_prior(input.contiguous(), height, width, nullptr)
                                             : lfs::core::resize_normal_prior(input.contiguous(), height, width, nullptr);
                    };
                    prior = resize_prior(prior, target_h, target_w);
                    if (item.undistort) {
                        const auto scaled = undistort_for(*item.undistort, prior, params.resize_factor, params.max_width);
                        if (item.is_depth) {
                            prior = lfs::core::undistort_depth_area(prior, scaled, nullptr);
                        } else {
                            prior = lfs::core::undistort_normal_area(prior, scaled, nullptr);
                        }
                    }
                    const size_t rank = prior.ndim();
                    if (item.aux_target_width > 0 && item.aux_target_height > 0 &&
                        (static_cast<int>(prior.shape()[rank - 1]) != item.aux_target_width ||
                         static_cast<int>(prior.shape()[rank - 2]) != item.aux_target_height)) {
                        prior = resize_prior(prior, item.aux_target_height, item.aux_target_width);
                    }
                    prior = prior.contiguous();
                    if (item.is_depth) {
                        try_complete_pair(item.sequence_id, item.loader_generation, std::nullopt,
                                          std::nullopt, std::move(prior));
                    } else {
                        try_complete_pair(item.sequence_id, item.loader_generation, std::nullopt,
                                          std::nullopt, std::nullopt, std::move(prior));
                    }
                } else {
                    auto image = decode_portable_rgb(item.path, params, upload, item.jpeg_data.get());
                    try_complete_pair(item.sequence_id, item.loader_generation, std::move(image),
                                      std::nullopt);
                }
            } catch (const std::exception& e) {
                LOG_WARN("[PipelinedImageLoader] Host processing failed {}: {}",
                         lfs::core::path_to_utf8(item.path), e.what());
                fail(e.what());
            } catch (...) {
                LOG_WARN("[PipelinedImageLoader] Host processing failed {}",
                         lfs::core::path_to_utf8(item.path));
                fail("non-standard image loader exception");
            }
        }
    }

} // namespace lfs::io
