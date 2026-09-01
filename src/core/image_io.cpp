/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"

#include "image_codecs.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace image_codecs = lfs::core::image_codecs;

namespace {

    constexpr int DEFAULT_JPEG_QUALITY = 95;

    template <typename T>
    T* downscale_resample_nch(const T* src,
                              int w, int h, int nw, int nh,
                              int channels) {
        const size_t outbytes = static_cast<size_t>(nw) * nh * channels * sizeof(T);
        auto* out = static_cast<T*>(std::malloc(outbytes));
        if (!out)
            throw std::bad_alloc();
        for (int y = 0; y < nh; ++y) {
            const float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(h) / nh - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, h - 1);
            const int y1 = std::min(y0 + 1, h - 1);
            const float fy = sy - std::floor(sy);
            for (int x = 0; x < nw; ++x) {
                const float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(w) / nw - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, w - 1);
                const int x1 = std::min(x0 + 1, w - 1);
                const float fx = sx - std::floor(sx);
                for (int c = 0; c < channels; ++c) {
                    const float top = static_cast<float>(src[(static_cast<size_t>(y0) * w + x0) * channels + c]) * (1.0f - fx) +
                                      static_cast<float>(src[(static_cast<size_t>(y0) * w + x1) * channels + c]) * fx;
                    const float bottom = static_cast<float>(src[(static_cast<size_t>(y1) * w + x0) * channels + c]) * (1.0f - fx) +
                                         static_cast<float>(src[(static_cast<size_t>(y1) * w + x1) * channels + c]) * fx;
                    const float value = top * (1.0f - fy) + bottom * fy;
                    if constexpr (std::is_floating_point_v<T>)
                        out[(static_cast<size_t>(y) * nw + x) * channels + c] = static_cast<T>(value);
                    else
                        out[(static_cast<size_t>(y) * nw + x) * channels + c] = static_cast<T>(std::clamp(std::lround(value), 0l, static_cast<long>(std::numeric_limits<T>::max())));
                }
            }
        }
        return out;
    }

    template <typename T>
    T* downscale_resample_direct(const T* src_rgb,
                                 int w, int h, int nw, int nh,
                                 int) {
        return downscale_resample_nch<T>(src_rgb, w, h, nw, nh, 3);
    }

    template <typename T>
    void expand_pixel_to_rgb(T* destination, const int channels, const T red, const T green, const T blue) {
        destination[0] = red;
        destination[1] = channels > 1 ? green : red;
        destination[2] = channels > 2 ? blue
                         : channels == 2
                             ? static_cast<T>((static_cast<unsigned long long>(red) + green) / 2)
                             : red;
    }

    lfs::core::Tensor normalize_image_for_save(lfs::core::Tensor image) {
        if (image.ndim() == 4)
            image = image.squeeze(0); // [B,C,H,W] -> [C,H,W]
        if (image.ndim() == 3 && image.shape()[0] <= 4)
            image = image.permute({1, 2, 0}); // [C,H,W] -> [H,W,C]
        image = image.to(lfs::core::Device::CPU).to(lfs::core::DataType::Float32).contiguous();
        return image;
    }

    lfs::core::Tensor prepare_image_for_write(lfs::core::Tensor image) {
        auto normalized = normalize_image_for_save(std::move(image));
        return (normalized.clamp(0, 1) * 255.0f)
            .to(lfs::core::DataType::UInt8)
            .to(lfs::core::Device::CPU)
            .contiguous();
    }

    lfs::core::Tensor prepare_image_grid_for_write(const std::vector<lfs::core::Tensor>& images,
                                                   bool horizontal,
                                                   int separator_width) {
        if (images.empty())
            throw std::runtime_error("No images provided");
        if (images.size() == 1)
            return prepare_image_for_write(images[0]);

        std::vector<lfs::core::Tensor> xs;
        xs.reserve(images.size());
        for (const auto& image : images)
            xs.push_back(prepare_image_for_write(image));

        lfs::core::Tensor sep;
        if (separator_width > 0) {
            const auto& ref = xs[0];
            const auto sep_shape = horizontal
                                       ? lfs::core::TensorShape({ref.shape()[0], static_cast<size_t>(separator_width), ref.shape()[2]})
                                       : lfs::core::TensorShape({static_cast<size_t>(separator_width), ref.shape()[1], ref.shape()[2]});
            sep = lfs::core::Tensor::full(sep_shape, 255.0f, ref.device(), ref.dtype());
        }

        lfs::core::Tensor combo = xs[0];
        for (size_t i = 1; i < xs.size(); ++i) {
            combo = (separator_width > 0)
                        ? lfs::core::Tensor::cat({combo, sep, xs[i]}, horizontal ? 1 : 0)
                        : lfs::core::Tensor::cat({combo, xs[i]}, horizontal ? 1 : 0);
        }
        return combo.contiguous();
    }

    void write_prepared_image(const std::filesystem::path& path,
                              const lfs::core::Tensor& image,
                              const int jpeg_quality,
                              const std::optional<std::string>& metadata_comment = {}) {
        if (image.ndim() != 3)
            throw std::runtime_error("save_image: expected a 3D HxWxC tensor");
        if (image.device() != lfs::core::Device::CPU)
            throw std::runtime_error("save_image: expected CPU tensor");
        if (image.dtype() != lfs::core::DataType::UInt8)
            throw std::runtime_error("save_image: expected uint8 tensor");

        const auto prepared = image.contiguous();
        const int height = static_cast<int>(prepared.shape()[0]);
        const int width = static_cast<int>(prepared.shape()[1]);
        int channels = static_cast<int>(prepared.shape()[2]);
        if (channels < 1 || channels > 4)
            throw std::runtime_error("save_image: channels must be in [1..4]");

        const std::string path_utf8 = lfs::core::path_to_utf8(path);
        LOG_INFO("Saving image: {} shape: [{}, {}, {}]", path_utf8, height, width, channels);
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        std::string error;
        bool success = false;
        if (ext == ".jpg" || ext == ".jpeg") {
            if (channels == 4) {
                std::vector<std::uint8_t> rgb(static_cast<size_t>(width) * height * 3);
                for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {
                    rgb[i * 3 + 0] = prepared.ptr<uint8_t>()[i * 4 + 0];
                    rgb[i * 3 + 1] = prepared.ptr<uint8_t>()[i * 4 + 1];
                    rgb[i * 3 + 2] = prepared.ptr<uint8_t>()[i * 4 + 2];
                }
                success = image_codecs::write_jpeg(path, rgb.data(), width, height, 3, jpeg_quality, metadata_comment, error);
            } else if (channels == 1 || channels == 3) {
                success = image_codecs::write_jpeg(path, prepared.ptr<uint8_t>(), width, height, channels, jpeg_quality, metadata_comment, error);
            } else {
                throw std::runtime_error("save_image: unsupported JPEG channel count");
            }
        } else if (ext == ".png") {
            success = image_codecs::write_png(path, prepared.ptr<uint8_t>(), width, height, channels, 8, 6, metadata_comment, error);
        } else if (ext == ".tif" || ext == ".tiff") {
            success = image_codecs::write_tiff(path, prepared.ptr<uint8_t>(), width, height, channels, error);
        } else {
            throw std::runtime_error("Unsupported image extension: " + ext);
        }
        if (!success)
            throw std::runtime_error("Failed to save " + path_utf8 + (error.empty() ? "" : ": " + error));
    }

    template <typename T>
    std::tuple<T*, int, int, int>
    load_image_t(std::filesystem::path p, int res_div, int max_width) {
        LOG_TIMER("load_image total");
        image_codecs::Image decoded;
        std::string error;
        const std::string path_utf8 = lfs::core::path_to_utf8(p);
        if (!image_codecs::decode(p, decoded, error))
            throw std::runtime_error("Load failed: " + path_utf8 + (error.empty() ? "" : " : " + error));
        if (decoded.width <= 0 || decoded.height <= 0 || decoded.channels <= 0)
            throw std::runtime_error("Invalid image dimensions: " + path_utf8);

        const size_t pixels = static_cast<size_t>(decoded.width) * decoded.height;
        auto* base = static_cast<T*>(std::malloc(pixels * 3 * sizeof(T)));
        if (!base)
            throw std::bad_alloc();
        if constexpr (std::is_same_v<T, uint8_t>) {
            if (decoded.sample_type == image_codecs::SampleType::UInt8) {
                const auto* source = decoded.data.data();
                if (decoded.channels == 3) {
                    std::copy_n(source, pixels * 3, base);
                } else {
                    for (size_t pixel = 0; pixel < pixels; ++pixel) {
                        const size_t index = pixel * decoded.channels;
                        expand_pixel_to_rgb(base + pixel * 3, decoded.channels, source[index],
                                            decoded.channels > 1 ? source[index + 1] : source[index],
                                            decoded.channels > 2 ? source[index + 2] : source[index]);
                    }
                }
            } else if (decoded.sample_type == image_codecs::SampleType::UInt16) {
                const auto* source = reinterpret_cast<const uint16_t*>(decoded.data.data());
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    const size_t index = pixel * decoded.channels;
                    const auto convert = [&](const size_t channel) {
                        return static_cast<uint8_t>((static_cast<uint32_t>(source[index + channel]) * 255u + 32767u) / 65535u);
                    };
                    const T first = convert(0);
                    expand_pixel_to_rgb(base + pixel * 3, decoded.channels, first,
                                        decoded.channels > 1 ? convert(1) : first,
                                        decoded.channels > 2 ? convert(2) : first);
                }
            } else {
                const auto* source = reinterpret_cast<const float*>(decoded.data.data());
                const auto convert = [&](const size_t index) {
                    return static_cast<uint8_t>(std::lround(std::clamp(static_cast<double>(source[index]), 0.0, 1.0) * 255.0));
                };
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    const size_t index = pixel * decoded.channels;
                    const T first = convert(index);
                    expand_pixel_to_rgb(base + pixel * 3, decoded.channels, first,
                                        decoded.channels > 1 ? convert(index + 1) : first,
                                        decoded.channels > 2 ? convert(index + 2) : first);
                }
            }
        } else {
            if (decoded.sample_type == image_codecs::SampleType::UInt16) {
                const auto* source = reinterpret_cast<const uint16_t*>(decoded.data.data());
                if (decoded.channels == 3) {
                    std::copy_n(source, pixels * 3, base);
                } else {
                    for (size_t pixel = 0; pixel < pixels; ++pixel) {
                        const size_t index = pixel * decoded.channels;
                        expand_pixel_to_rgb(base + pixel * 3, decoded.channels, source[index],
                                            decoded.channels > 1 ? source[index + 1] : source[index],
                                            decoded.channels > 2 ? source[index + 2] : source[index]);
                    }
                }
            } else if (decoded.sample_type == image_codecs::SampleType::UInt8) {
                const auto* source = decoded.data.data();
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    const size_t index = pixel * decoded.channels;
                    const auto convert = [&](const size_t channel) {
                        return static_cast<uint16_t>(static_cast<uint16_t>(source[index + channel]) * 257u);
                    };
                    const T first = convert(0);
                    expand_pixel_to_rgb(base + pixel * 3, decoded.channels, first,
                                        decoded.channels > 1 ? convert(1) : first,
                                        decoded.channels > 2 ? convert(2) : first);
                }
            } else {
                const auto* source = reinterpret_cast<const float*>(decoded.data.data());
                const auto convert = [&](const size_t index) {
                    return static_cast<uint16_t>(std::lround(std::clamp(static_cast<double>(source[index]), 0.0, 1.0) * 65535.0));
                };
                for (size_t pixel = 0; pixel < pixels; ++pixel) {
                    const size_t index = pixel * decoded.channels;
                    const T first = convert(index);
                    expand_pixel_to_rgb(base + pixel * 3, decoded.channels, first,
                                        decoded.channels > 1 ? convert(index + 1) : first,
                                        decoded.channels > 2 ? convert(index + 2) : first);
                }
            }
        }

        int target_width = decoded.width;
        int target_height = decoded.height;
        if (res_div == 2 || res_div == 4 || res_div == 8) {
            target_width = std::max(1, target_width / res_div);
            target_height = std::max(1, target_height / res_div);
        } else if (res_div > 1) {
            LOG_ERROR("load_image: unsupported resize factor {}", res_div);
        }
        if (max_width > 0 && (target_width > max_width || target_height > max_width)) {
            if (target_width > target_height) {
                target_height = std::max(1, max_width * target_height / target_width);
                target_width = max_width;
            } else {
                target_width = std::max(1, max_width * target_width / target_height);
                target_height = max_width;
            }
        }
        if (target_width == decoded.width && target_height == decoded.height)
            return {base, decoded.width, decoded.height, 3};
        T* resized = nullptr;
        try {
            resized = downscale_resample_direct<T>(base, decoded.width, decoded.height,
                                                   target_width, target_height, 0);
        } catch (...) {
            std::free(base);
            throw;
        }
        std::free(base);
        return {resized, target_width, target_height, 3};
    }

} // namespace

namespace lfs::core {

    std::tuple<int, int, int> get_image_info(std::filesystem::path p) {
        image_codecs::Probe probe;
        std::string error;
        if (!image_codecs::probe(p, probe, error)) {
            image_codecs::Image decoded;
            if (!image_codecs::decode(p, decoded, error))
                throw std::runtime_error("Image probe failed: " + lfs::core::path_to_utf8(p) + (error.empty() ? "" : " : " + error));
            return {decoded.width, decoded.height, decoded.channels};
        }
        return {probe.width, probe.height, probe.channels};
    }

    std::tuple<unsigned char*, int, int, int>
    load_image_with_alpha(std::filesystem::path p, int res_div, int max_width) {
        image_codecs::Image decoded;
        std::string error;
        if (!image_codecs::decode(p, decoded, error))
            throw std::runtime_error("Load failed: " + lfs::core::path_to_utf8(p) + (error.empty() ? "" : " : " + error));
        if (decoded.channels != 4) {
            LOG_ERROR("load_image_with_alpha: expected 4 channels, got {}", decoded.channels);
            return std::make_tuple(nullptr, 0, 0, 0);
        }
        const size_t pixel_count = static_cast<size_t>(decoded.width) * decoded.height;
        auto* out = static_cast<unsigned char*>(std::malloc(pixel_count * 4));
        if (!out) {
            throw std::bad_alloc();
        }
        for (size_t i = 0; i < pixel_count * 4; ++i) {
            if (decoded.sample_type == image_codecs::SampleType::UInt8)
                out[i] = decoded.data[i];
            else if (decoded.sample_type == image_codecs::SampleType::UInt16)
                out[i] = static_cast<unsigned char>(std::lround(reinterpret_cast<const uint16_t*>(decoded.data.data())[i] / 257.0));
            else
                out[i] = static_cast<unsigned char>(std::lround(std::clamp(reinterpret_cast<const float*>(decoded.data.data())[i], 0.0f, 1.0f) * 255.0f));
        }

        int nw = decoded.width, nh = decoded.height;
        if (res_div == 2 || res_div == 4 || res_div == 8) {
            nw = std::max(1, decoded.width / res_div);
            nh = std::max(1, decoded.height / res_div);
        }
        if (max_width > 0 && (nw > max_width || nh > max_width)) {
            if (nw > nh) {
                nh = std::max(1, max_width * nh / nw);
                nw = max_width;
            } else {
                nw = std::max(1, max_width * nw / nh);
                nh = max_width;
            }
        }

        if (nw != decoded.width || nh != decoded.height) {
            unsigned char* resized = nullptr;
            try {
                resized = downscale_resample_nch<unsigned char>(out, decoded.width, decoded.height, nw, nh, 4);
            } catch (...) {
                std::free(out);
                throw;
            }
            std::free(out);
            return {resized, nw, nh, 4};
        }

        return {out, decoded.width, decoded.height, 4};
    }

    std::tuple<unsigned char*, int, int, int>
    load_image_from_memory(const uint8_t* const data, const size_t size) {
        image_codecs::Image decoded;
        std::string error;
        if (!image_codecs::decode_memory(data, size, decoded, error))
            throw std::runtime_error("Load from memory failed: " + error);
        const size_t pixel_count = static_cast<size_t>(decoded.width) * decoded.height;
        auto* out = static_cast<unsigned char*>(std::malloc(pixel_count * 3));
        if (!out) {
            throw std::bad_alloc();
        }
        const auto sample_to_u8 = [&](const size_t index) {
            if (decoded.sample_type == image_codecs::SampleType::UInt16) {
                const auto value = reinterpret_cast<const uint16_t*>(decoded.data.data())[index];
                return static_cast<unsigned char>(std::lround(value / 257.0));
            }
            if (decoded.sample_type == image_codecs::SampleType::Float32) {
                const auto value = reinterpret_cast<const float*>(decoded.data.data())[index];
                return static_cast<unsigned char>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
            }
            return decoded.data[index];
        };
        for (size_t i = 0; i < pixel_count; ++i) {
            const size_t src = i * decoded.channels;
            const auto first = sample_to_u8(src);
            expand_pixel_to_rgb(out + i * 3, decoded.channels, first,
                                decoded.channels > 1 ? sample_to_u8(src + 1) : first,
                                decoded.channels > 2 ? sample_to_u8(src + 2) : first);
        }
        return {out, decoded.width, decoded.height, 3};
    }

    std::tuple<unsigned char*, int, int, int>
    load_image(std::filesystem::path p, int res_div, int max_width) {
        return ::load_image_t<unsigned char>(p, res_div, max_width);
    }

    std::tuple<uint16_t*, int, int, int>
    load_image_u16(std::filesystem::path p, int res_div, int max_width) {
        return ::load_image_t<uint16_t>(p, res_div, max_width);
    }

    void save_image(const std::filesystem::path& path, lfs::core::Tensor image,
                    const std::optional<std::string>& metadata_comment) {
        write_prepared_image(path,
                             prepare_image_for_write(std::move(image)),
                             DEFAULT_JPEG_QUALITY,
                             metadata_comment);
    }

    void save_image_u8(const std::filesystem::path& path,
                       lfs::core::Tensor image,
                       const int jpeg_quality,
                       const std::optional<std::string>& metadata_comment) {
        if (image.ndim() == 4)
            image = image.squeeze(0);
        if (image.ndim() == 3 && image.shape()[0] <= 4 && image.shape()[2] > 4)
            image = image.permute({1, 2, 0});
        // to() clones even when already on the target device/dtype; guard to avoid
        // duplicating gigapixel exports.
        if (image.device() != lfs::core::Device::CPU)
            image = image.to(lfs::core::Device::CPU);
        if (image.dtype() != lfs::core::DataType::UInt8)
            image = image.to(lfs::core::DataType::UInt8);
        image = image.contiguous();
        write_prepared_image(path, image, jpeg_quality, metadata_comment);
    }

    void save_image(const std::filesystem::path& path,
                    const std::vector<lfs::core::Tensor>& images,
                    bool horizontal,
                    int separator_width,
                    const std::optional<std::string>& metadata_comment) {
        if (images.empty())
            throw std::runtime_error("No images provided");
        write_prepared_image(path,
                             prepare_image_grid_for_write(images, horizontal, separator_width),
                             DEFAULT_JPEG_QUALITY,
                             metadata_comment);
    }

    void free_image(void* img) { std::free(img); }

    std::tuple<float*, int, int> load_image_gray_high_bitdepth(std::filesystem::path p) {
        image_codecs::Probe probe;
        std::string error;
        if (!image_codecs::probe(p, probe, error) || probe.sample_type == image_codecs::SampleType::UInt8)
            return {nullptr, 0, 0};
        auto [decoded, width, height, channels] = load_image_float(p);
        if (!decoded || channels < 1) {
            return {nullptr, 0, 0};
        }
        auto* out = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(width) * height));
        if (!out) {
            free_image_float(decoded);
            throw std::bad_alloc();
        }
        for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i)
            out[i] = decoded[i * channels];
        free_image_float(decoded);
        return {out, width, height};
    }

    std::tuple<float*, int, int> load_image_rgb_high_bitdepth(std::filesystem::path p) {
        image_codecs::Probe probe;
        std::string error;
        if (!image_codecs::probe(p, probe, error) || probe.sample_type == image_codecs::SampleType::UInt8 || probe.channels < 3)
            return {nullptr, 0, 0};
        auto [decoded, width, height, channels] = load_image_float(p);
        if (!decoded || channels < 3) {
            return {nullptr, 0, 0};
        }
        auto* out = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(width) * height * 3));
        if (!out) {
            free_image_float(decoded);
            throw std::bad_alloc();
        }
        for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {
            out[i * 3 + 0] = decoded[i * channels + 0];
            out[i * 3 + 1] = decoded[i * channels + 1];
            out[i * 3 + 2] = decoded[i * channels + 2];
        }
        free_image_float(decoded);
        return {out, width, height};
    }

    void free_image_float(float* img) { std::free(img); }

    std::tuple<float*, int, int, int> load_image_float(const std::filesystem::path& p) {
        image_codecs::Image decoded;
        std::string error;
        if (!image_codecs::decode(p, decoded, error)) {
            LOG_ERROR("load_image_float: failed to read {}{}", lfs::core::path_to_utf8(p),
                      error.empty() ? "" : ": " + error);
            return {nullptr, 0, 0, 0};
        }
        const size_t values = static_cast<size_t>(decoded.width) * decoded.height * decoded.channels;
        auto* output = static_cast<float*>(std::malloc(values * sizeof(float)));
        if (!output)
            throw std::bad_alloc();
        if (decoded.sample_type == image_codecs::SampleType::Float32) {
            std::memcpy(output, decoded.data.data(), values * sizeof(float));
        } else if (decoded.sample_type == image_codecs::SampleType::UInt16) {
            const auto* input = reinterpret_cast<const uint16_t*>(decoded.data.data());
            for (size_t i = 0; i < values; ++i)
                output[i] = static_cast<float>(input[i]) / 65535.0f;
        } else {
            for (size_t i = 0; i < values; ++i)
                output[i] = static_cast<float>(decoded.data[i]) / 255.0f;
        }
        return {output, decoded.width, decoded.height, decoded.channels};
    }

    void resample_bilinear_f32(const float* src, const int sw, const int sh, const int channels,
                               float* dst, const int dw, const int dh) {
        if (!src || !dst || sw <= 0 || sh <= 0 || channels <= 0 || dw <= 0 || dh <= 0)
            return;
        for (int y = 0; y < dh; ++y) {
            const float sy = (static_cast<float>(y) + 0.5f) * static_cast<float>(sh) / dh - 0.5f;
            const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, sh - 1);
            const int y1 = std::min(y0 + 1, sh - 1);
            const float fy = sy - std::floor(sy);
            for (int x = 0; x < dw; ++x) {
                const float sx = (static_cast<float>(x) + 0.5f) * static_cast<float>(sw) / dw - 0.5f;
                const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, sw - 1);
                const int x1 = std::min(x0 + 1, sw - 1);
                const float fx = sx - std::floor(sx);
                for (int c = 0; c < channels; ++c) {
                    const float top = src[(static_cast<size_t>(y0) * sw + x0) * channels + c] * (1.0f - fx) +
                                      src[(static_cast<size_t>(y0) * sw + x1) * channels + c] * fx;
                    const float bottom = src[(static_cast<size_t>(y1) * sw + x0) * channels + c] * (1.0f - fx) +
                                         src[(static_cast<size_t>(y1) * sw + x1) * channels + c] * fx;
                    dst[(static_cast<size_t>(y) * dw + x) * channels + c] = top * (1.0f - fy) + bottom * fy;
                }
            }
        }
    }

    bool save_png(const std::filesystem::path& p, const void* data, const int w, const int h,
                  const int channels, const int bit_depth, const int compression_level) {
        std::string error;
        return image_codecs::write_png(p, data, w, h, channels, bit_depth, compression_level, {}, error);
    }

    float image_quantization_step(const std::filesystem::path& p) {
        image_codecs::Probe probe;
        std::string error;
        if (!image_codecs::probe(p, probe, error))
            return 0.0f;
        if (probe.sample_type == image_codecs::SampleType::UInt16)
            return 1.0f / 65535.0f;
        if (probe.sample_type == image_codecs::SampleType::UInt8)
            return 1.0f / 255.0f;
        return 0.0f;
    }

    namespace {
        void flip_normal_prior_yz(float* ys, float* zs,
                                  const size_t pixel_count, const size_t stride) {
            for (size_t i = 0; i < pixel_count; ++i) {
                ys[i * stride] = -ys[i * stride];
                zs[i * stride] = -zs[i * stride];
            }
        }

        void transform_normal_world_to_camera(float& x, float& y, float& z, const std::array<float, 9>& w2c) {
            const float wx = x;
            const float wy = y;
            const float wz = z;
            x = w2c[0] * wx + w2c[1] * wy + w2c[2] * wz;
            y = w2c[3] * wx + w2c[4] * wy + w2c[5] * wz;
            z = w2c[6] * wx + w2c[7] * wy + w2c[8] * wz;
        }
    } // namespace

    void flip_normal_prior_yz_hwc(float* data, const size_t pixel_count) {
        flip_normal_prior_yz(data + 1, data + 2, pixel_count, 3);
    }

    void flip_normal_prior_yz_chw(float* data, const size_t pixel_count) {
        flip_normal_prior_yz(data + pixel_count, data + 2 * pixel_count, pixel_count, 1);
    }

    void transform_normal_prior_world_to_camera_hwc(
        float* data, const size_t pixel_count, const std::array<float, 9>& w2c) {
        for (size_t i = 0; i < pixel_count; ++i) {
            transform_normal_world_to_camera(data[i * 3], data[i * 3 + 1], data[i * 3 + 2], w2c);
        }
    }

    void transform_normal_prior_world_to_camera_chw(
        float* data, const size_t pixel_count, const std::array<float, 9>& w2c) {
        float* const xs = data;
        float* const ys = data + pixel_count;
        float* const zs = data + 2 * pixel_count;
        for (size_t i = 0; i < pixel_count; ++i) {
            transform_normal_world_to_camera(xs[i], ys[i], zs[i], w2c);
        }
    }

    float srgb_encoding_to_linear(const float v) {
        if (v <= 0.04045f) {
            return v / 12.92f;
        }
        return std::pow((v + 0.055f) / 1.055f, 2.4f);
    }

    void srgb_normal_prior_to_linear_chw(float* data, const size_t value_count) {
        for (size_t i = 0; i < value_count; ++i) {
            const float encoded = data[i] * 0.5f + 0.5f;
            data[i] = srgb_encoding_to_linear(encoded) * 2.0f - 1.0f;
        }
    }

    std::vector<NormalPriorSample> sample_normal_prior_pixels(
        const std::filesystem::path& p, const size_t max_samples) {
        auto [pixels, width, height, channels] = load_image_float(p);
        if (!pixels || channels < 3 || width <= 0 || height <= 0 || max_samples == 0) {
            free_image_float(pixels);
            return {};
        }
        const size_t pixel_count = static_cast<size_t>(width) * height;
        const size_t stride = std::max<size_t>(1, pixel_count / max_samples);
        const float inv_width = 1.0f / static_cast<float>(width);
        const float inv_height = 1.0f / static_cast<float>(height);
        std::vector<NormalPriorSample> samples;
        samples.reserve(pixel_count / stride + 1);
        for (size_t i = 0; i < pixel_count; i += stride) {
            const size_t x = i % static_cast<size_t>(width);
            const size_t y = i / static_cast<size_t>(width);
            samples.push_back(NormalPriorSample{
                (static_cast<float>(x) + 0.5f) * inv_width,
                (static_cast<float>(y) + 0.5f) * inv_height,
                pixels[i * channels],
                pixels[i * channels + 1],
                pixels[i * channels + 2]});
        }
        free_image_float(pixels);
        return samples;
    }

    bool save_img_data(const std::filesystem::path& p, const std::tuple<unsigned char*, int, int, int>& image_data) {
        auto [data, width, height, channels] = image_data;

        if (!data || width <= 0 || height <= 0 || channels <= 0) {
            return false;
        }

        // Get file extension to determine format
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        // Check if format is supported
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".tif" && ext != ".tiff") {
            return false;
        }

        std::string error;
        bool success = false;
        if (ext == ".jpg" || ext == ".jpeg") {
            if (channels == 4) {
                std::vector<unsigned char> rgb(static_cast<size_t>(width) * height * 3);
                for (size_t i = 0, n = static_cast<size_t>(width) * height; i < n; ++i) {
                    rgb[i * 3 + 0] = data[i * 4 + 0];
                    rgb[i * 3 + 1] = data[i * 4 + 1];
                    rgb[i * 3 + 2] = data[i * 4 + 2];
                }
                success = image_codecs::write_jpeg(p, rgb.data(), width, height, 3, 95, {}, error);
            } else if (channels == 1 || channels == 3) {
                success = image_codecs::write_jpeg(p, data, width, height, channels, 95, {}, error);
            }
        } else if (ext == ".png") {
            success = image_codecs::write_png(p, data, width, height, channels, 8, 6, {}, error);
        } else if (ext == ".tif" || ext == ".tiff") {
            success = image_codecs::write_tiff(p, data, width, height, channels, error);
        }
        return success;
    }

} // namespace lfs::core

namespace lfs::core::image_io {

    namespace {
        std::atomic<BatchImageSaver*> g_batch_image_saver{nullptr};
    }

    BatchImageSaver& BatchImageSaver::instance() {
        static BatchImageSaver instance;
        return instance;
    }

    BatchImageSaver* BatchImageSaver::try_instance() {
        return g_batch_image_saver.load(std::memory_order_acquire);
    }

    void BatchImageSaver::wait_all_if_initialized() {
        if (auto* saver = try_instance()) {
            saver->wait_all();
        }
    }

    size_t BatchImageSaver::pending_count_if_initialized() {
        if (auto* saver = try_instance()) {
            return saver->pending_count();
        }
        return 0;
    }

    BatchImageSaver::BatchImageSaver(size_t num_workers)
        : num_workers_(std::max(size_t(1), std::min(num_workers, std::min(size_t(8), size_t(std::thread::hardware_concurrency()))))),
          max_pending_tasks_(num_workers_) {

        g_batch_image_saver.store(this, std::memory_order_release);
        LOG_INFO("[BatchImageSaver] Starting with {} worker threads (max pending tasks: {})", num_workers_, max_pending_tasks_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&BatchImageSaver::worker_thread, this);
        }
    }

    BatchImageSaver::~BatchImageSaver() {
        shutdown();
        g_batch_image_saver.store(nullptr, std::memory_order_release);
    }

    void BatchImageSaver::shutdown() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_)
                return;
            stop_ = true;
            LOG_INFO("[BatchImageSaver] Shutting down...");
        }
        cv_.notify_all();
        cv_space_.notify_all();

        for (auto& w : workers_)
            if (w.joinable())
                w.join();

        while (!task_queue_.empty()) {
            process_task(task_queue_.front());
            task_queue_.pop();
        }
        LOG_INFO("[BatchImageSaver] Shutdown complete");
    }

    void BatchImageSaver::queue_save(const std::filesystem::path& path, lfs::core::Tensor image,
                                     const std::optional<std::string>& metadata_comment) {
        if (!enabled_) {
            lfs::core::save_image(path, image, metadata_comment);
            return;
        }
        SaveTask t;
        t.path = path;
        t.images.push_back(prepare_image_for_write(std::move(image)));
        t.metadata_comment = metadata_comment;
        enqueue_task(std::move(t));
    }

    void BatchImageSaver::queue_save_multiple(const std::filesystem::path& path,
                                              const std::vector<lfs::core::Tensor>& images,
                                              bool horizontal,
                                              int separator_width,
                                              const std::optional<std::string>& metadata_comment) {
        if (!enabled_) {
            lfs::core::save_image(path, images, horizontal, separator_width, metadata_comment);
            return;
        }
        SaveTask t;
        t.path = path;
        t.images.push_back(prepare_image_grid_for_write(images, horizontal, separator_width));
        t.metadata_comment = metadata_comment;
        enqueue_task(std::move(t));
    }

    void BatchImageSaver::wait_all() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        cv_finished_.wait(lock, [this] { return task_queue_.empty() && active_tasks_ == 0; });
    }

    size_t BatchImageSaver::pending_count() const {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return task_queue_.size() + active_tasks_;
    }

    void BatchImageSaver::worker_thread() {
        while (true) {
            SaveTask t;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });
                if (stop_ && task_queue_.empty())
                    break;
                if (task_queue_.empty())
                    continue;
                t = std::move(task_queue_.front());
                task_queue_.pop();
            }
            process_task(t);
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                active_tasks_--;
            }
            cv_space_.notify_all();
            cv_finished_.notify_all();
        }
    }

    void BatchImageSaver::process_task(const SaveTask& t) {
        try {
            assert(!t.images.empty());
            write_prepared_image(t.path, t.images[0], DEFAULT_JPEG_QUALITY, t.metadata_comment);
        } catch (const std::exception& e) {
            LOG_ERROR("[BatchImageSaver] Error saving {}: {}", lfs::core::path_to_utf8(t.path), e.what());
        }
    }

    void BatchImageSaver::enqueue_task(SaveTask task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_space_.wait(lock, [this] {
                return stop_ || (task_queue_.size() + active_tasks_) < max_pending_tasks_;
            });
            if (stop_) {
                assert(!task.images.empty());
                write_prepared_image(task.path, task.images[0], DEFAULT_JPEG_QUALITY, task.metadata_comment);
                return;
            }
            task_queue_.push(std::move(task));
            active_tasks_++;
        }
        cv_.notify_one();
    }
} // namespace lfs::core::image_io
