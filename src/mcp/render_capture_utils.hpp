/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/base64.hpp"
#include "core/tensor.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace lfs::mcp {

    namespace detail {

        inline void stbi_write_png_callback(void* context, void* data, int size) {
            auto* buf = static_cast<std::vector<uint8_t>*>(context);
            auto* bytes = static_cast<const uint8_t*>(data);
            buf->insert(buf->end(), bytes, bytes + size);
        }

        inline std::vector<uint8_t> resize_image_nearest(const uint8_t* src,
                                                         int src_width,
                                                         int src_height,
                                                         int channels,
                                                         int dst_width,
                                                         int dst_height) {
            std::vector<uint8_t> dst(static_cast<size_t>(dst_width) * dst_height * channels);
            for (int y = 0; y < dst_height; ++y) {
                const int src_y = std::min(src_height - 1, (y * src_height) / dst_height);
                for (int x = 0; x < dst_width; ++x) {
                    const int src_x = std::min(src_width - 1, (x * src_width) / dst_width);
                    const auto* const src_pixel = src + (static_cast<size_t>(src_y) * src_width + src_x) * channels;
                    auto* const dst_pixel = dst.data() + (static_cast<size_t>(y) * dst_width + x) * channels;
                    std::copy_n(src_pixel, channels, dst_pixel);
                }
            }
            return dst;
        }

        inline std::expected<std::pair<int, int>, std::string> resolve_capture_size(int src_width,
                                                                                     int src_height,
                                                                                     int width,
                                                                                     int height) {
            if (width <= 0 && height <= 0) {
                return std::pair{src_width, src_height};
            }
            if (width <= 0) {
                width = std::max(1, static_cast<int>(std::lround(
                                        static_cast<double>(src_width) * static_cast<double>(height) /
                                        static_cast<double>(src_height))));
            }
            if (height <= 0) {
                height = std::max(1, static_cast<int>(std::lround(
                                         static_cast<double>(src_height) * static_cast<double>(width) /
                                         static_cast<double>(src_width))));
            }
            if (width <= 0 || height <= 0) {
                return std::unexpected("Capture size must be positive");
            }
            return std::pair{width, height};
        }

    } // namespace detail

    inline std::expected<std::string, std::string> encode_pixels_to_base64(const uint8_t* src_pixels,
                                                                            int src_width,
                                                                            int src_height,
                                                                            int channels,
                                                                            int width = 0,
                                                                            int height = 0) {
        if (!src_pixels)
            return std::unexpected("Pixel buffer is null");
        if (src_width <= 0 || src_height <= 0)
            return std::unexpected("Pixel buffer dimensions must be positive");
        if (channels < 1 || channels > 4)
            return std::unexpected("Pixel buffer channel count must be between 1 and 4");

        const auto size = detail::resolve_capture_size(src_width, src_height, width, height);
        if (!size)
            return std::unexpected(size.error());

        const auto [out_width, out_height] = *size;

        const uint8_t* pixels = src_pixels;
        std::vector<uint8_t> resized;
        if (out_width != src_width || out_height != src_height) {
            resized = detail::resize_image_nearest(
                pixels, src_width, src_height, channels, out_width, out_height);
            pixels = resized.data();
        }

        std::vector<uint8_t> png_buf;
        png_buf.reserve(static_cast<size_t>(out_width) * out_height * channels);
        const int ok = stbi_write_png_to_func(
            detail::stbi_write_png_callback,
            &png_buf,
            out_width,
            out_height,
            channels,
            pixels,
            out_width * channels);
        if (!ok)
            return std::unexpected("PNG encoding failed");

        return core::base64_encode(png_buf);
    }

    inline std::expected<std::string, std::string> encode_render_tensor_to_base64(core::Tensor image,
                                                                                   int width = 0,
                                                                                   int height = 0) {
        image = image.clone().to(core::Device::CPU).to(core::DataType::Float32);
        if (image.ndim() == 4)
            image = image.squeeze(0);
        if (image.ndim() == 3 && image.shape()[0] <= 4)
            image = image.permute({1, 2, 0});
        image = (image.clamp(0, 1) * 255.0f).to(core::DataType::UInt8).contiguous();

        const int src_height = static_cast<int>(image.shape()[0]);
        const int src_width = static_cast<int>(image.shape()[1]);
        const int channels = static_cast<int>(image.shape()[2]);
        assert(channels >= 1 && channels <= 4);

        return encode_pixels_to_base64(image.ptr<uint8_t>(), src_width, src_height, channels, width, height);
    }

} // namespace lfs::mcp
