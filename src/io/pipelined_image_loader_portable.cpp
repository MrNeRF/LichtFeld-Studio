/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/pipelined_image_loader.hpp"

#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
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
            return device;
        }

        Tensor float_to_uint8(const Tensor& image) {
            return image.clamp(0.0f, 1.0f).mul(255.0f).add(0.5f).to(DataType::UInt8).contiguous();
        }

        lfs::core::UndistortParams undistort_for(const lfs::core::UndistortParams& params,
                                                 const Tensor& image,
                                                 const int max_width) {
            const size_t rank = image.ndim();
            return lfs::core::scale_undistort_params(
                params,
                static_cast<int>(image.shape()[rank - 1]),
                static_cast<int>(image.shape()[rank - 2]),
                max_width);
        }

        // Inversion precedes binarization; threshold compares with >=.
        Tensor finish_mask(Tensor mask, const MaskParams& params) {
            if (params.invert)
                mask = mask.neg().add(1.0f);
            mask = params.threshold > 0.0f ? mask.ge(params.threshold).to(DataType::Float32)
                                           : mask.clamp(0.0f, 1.0f);
            return mask.contiguous();
        }

        std::pair<Tensor, Tensor> decode_rgba(const std::filesystem::path& path,
                                              const LoadParams& params,
                                              lfs::core::TensorUpload& upload) {
            auto [data, width, height, channels] =
                lfs::core::load_image_with_alpha(path, params.resize_factor, params.max_width);
            if (!data || channels != 4)
                throw std::runtime_error("Failed to load RGBA image");
            const auto rgba = to_device(upload, host_uint8(data, image_shape(height, width, 4),
                                                           lfs::core::free_image));

            Tensor rgb = rgba.slice(2, 0, 3).permute({2, 0, 1});
            if (!params.output_uint8)
                rgb = rgb.to(DataType::Float32).mul(UINT8_SCALE);
            Tensor alpha = rgba.slice(2, 3, 4).squeeze(2).to(DataType::Float32).mul(UINT8_SCALE);
            if (params.undistort) {
                if (params.output_uint8)
                    rgb = rgb.to(DataType::Float32).div(255.0f);
                const auto scaled = undistort_for(*params.undistort, alpha, params.max_width);
                rgb = lfs::core::undistort_image(rgb.contiguous(), scaled, nullptr);
                alpha = lfs::core::undistort_mask(alpha.contiguous(), scaled, nullptr);
                if (params.output_uint8)
                    rgb = float_to_uint8(rgb);
            }
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

    lfs::core::Tensor PipelinedImageLoader::decode_portable_rgb(
        const std::filesystem::path& path,
        const LoadParams& params,
        lfs::core::TensorUpload& upload) const {
        {
            std::lock_guard<std::mutex> stats_lock(stats_mutex_);
            ++stats_.cpu_decode_calls;
        }

        Tensor image;
        if (config_.use_16bit_color) {
            auto [data, width, height, channels] =
                lfs::core::load_image_u16(path, params.resize_factor, params.max_width);
            if (!data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            image = to_device(upload, host_uint16(data, image_shape(height, width, channels),
                                                  lfs::core::free_image))
                        .permute({2, 0, 1})
                        .to(DataType::Float32)
                        .mul(UINT16_SCALE);
            if (params.output_uint8)
                image = float_to_uint8(image);
        } else {
            auto [data, width, height, channels] =
                lfs::core::load_image(path, params.resize_factor, params.max_width);
            if (!data)
                throw std::runtime_error("Failed to decode image: " + lfs::core::path_to_utf8(path));
            image = to_device(upload, host_uint8(data, image_shape(height, width, channels),
                                                 lfs::core::free_image))
                        .permute({2, 0, 1});
            if (!params.output_uint8)
                image = image.to(DataType::Float32).mul(UINT8_SCALE);
        }

        if (params.undistort) {
            const bool restore_uint8 = image.dtype() == DataType::UInt8;
            if (restore_uint8)
                image = image.to(DataType::Float32).div(255.0f);
            image = lfs::core::undistort_image(
                image.contiguous(), undistort_for(*params.undistort, image, params.max_width), nullptr);
            if (restore_uint8)
                image = float_to_uint8(image);
        }
        return image.contiguous();
    }

    void PipelinedImageLoader::portable_process_thread_func() {
        const lfs::core::GpuBackendScope backend(config_.backend);
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
                                          std::nullopt, nullptr);
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
                    auto [rgb, alpha] = decode_rgba(item.path, params, upload);
                    try_complete_pair(item.sequence_id, item.loader_generation, std::move(rgb),
                                      finish_mask(std::move(alpha), item.alpha_mask_params), nullptr);
                } else if (item.is_mask) {
                    int width = 0, height = 0, channels = 0;
                    stbi_uc* const gray = stbi_load(lfs::core::path_to_utf8(item.path).c_str(),
                                                    &width, &height, &channels, 1);
                    if (!gray)
                        throw std::runtime_error("Failed to decode mask");
                    const auto [target_w, target_h] = sidecar_target_size(item, width, height);
                    Tensor mask;
                    if (target_w == width && target_h == height) {
                        mask = to_device(upload, host_uint8(gray, image_shape(height, width, 1), stbi_image_free))
                                   .to(DataType::Float32)
                                   .mul(UINT8_SCALE);
                    } else {
                        const std::unique_ptr<void, ReleaseFn> owner(gray, stbi_image_free);
                        auto source = Tensor::empty_pageable_host(image_shape(height, width, 1), DataType::Float32);
                        std::transform(gray, gray + source.numel(), source.ptr<float>(),
                                       [](const stbi_uc value) { return static_cast<float>(value) * UINT8_SCALE; });
                        auto resized = Tensor::empty_pageable_host(image_shape(target_h, target_w, 1), DataType::Float32);
                        lfs::core::resample_bilinear_f32(source.ptr<float>(), width, height, 1,
                                                         resized.ptr<float>(), target_w, target_h);
                        mask = to_device(upload, resized);
                    }
                    if (item.undistort) {
                        mask = lfs::core::undistort_mask(
                            mask.contiguous(), undistort_for(*item.undistort, mask, params.max_width), nullptr);
                    }
                    try_complete_pair(item.sequence_id, item.loader_generation, std::nullopt,
                                      finish_mask(std::move(mask), item.mask_params), nullptr);
                } else if (item.is_depth || item.is_normal) {
                    const std::string path_utf8 = lfs::core::path_to_utf8(item.path);
                    const int channels = item.is_depth ? 1 : 3;
                    const bool sixteen_bit = stbi_is_16_bit(path_utf8.c_str()) != 0;
                    int width = 0, height = 0, source_channels = 0;
                    Tensor samples;
                    if (sixteen_bit) {
                        stbi_us* const data = stbi_load_16(path_utf8.c_str(), &width, &height, &source_channels, channels);
                        if (!data)
                            throw std::runtime_error(item.is_depth ? "Failed to decode 16-bit depth"
                                                                   : "Failed to decode 16-bit normal map");
                        samples = host_uint16(data, image_shape(height, width, channels), stbi_image_free);
                    } else {
                        stbi_uc* const data = stbi_load(path_utf8.c_str(), &width, &height, &source_channels, channels);
                        if (!data)
                            throw std::runtime_error(item.is_depth ? "Failed to decode depth" : "Failed to decode normal map");
                        samples = host_uint8(data, image_shape(height, width, channels), stbi_image_free);
                    }
                    samples = to_device(upload, samples);

                    const auto [target_w, target_h] = sidecar_target_size(item, width, height);
                    Tensor prior;
                    if (item.is_depth) {
                        prior = sixteen_bit ? samples.to(DataType::Float32).mul(UINT16_SCALE)
                                            : samples.to(DataType::Float32).div(255.0f);
                    } else {
                        prior = decode_normal_prior(samples, sixteen_bit ? UINT16_SCALE : UINT8_SCALE,
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
                        const auto scaled = undistort_for(*item.undistort, prior, params.max_width);
                        if (item.is_depth) {
                            prior = lfs::core::undistort_mask(prior, scaled, nullptr);
                        } else {
                            prior = lfs::core::undistort_image(prior, scaled, nullptr);
                            prior = resize_prior(prior, static_cast<int>(prior.shape()[1]), static_cast<int>(prior.shape()[2]));
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
                                          std::nullopt, nullptr, std::move(prior));
                    } else {
                        try_complete_pair(item.sequence_id, item.loader_generation, std::nullopt,
                                          std::nullopt, nullptr, std::nullopt, std::move(prior));
                    }
                } else {
                    auto image = decode_portable_rgb(item.path, params, upload);
                    try_complete_pair(item.sequence_id, item.loader_generation, std::move(image),
                                      std::nullopt, nullptr);
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
