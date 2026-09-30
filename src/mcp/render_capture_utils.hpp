/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/base64.hpp"
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "mcp/shared_scene_tools.hpp"
#include "rendering/image_layout.hpp"

#include <stb_image_write.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
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
                const auto src_y = std::min<std::int64_t>(
                    src_height - 1, static_cast<std::int64_t>(y) * src_height / dst_height);
                for (int x = 0; x < dst_width; ++x) {
                    const auto src_x = std::min<std::int64_t>(
                        src_width - 1, static_cast<std::int64_t>(x) * src_width / dst_width);
                    const auto* const src_pixel = src + (static_cast<size_t>(src_y) * src_width + src_x) * channels;
                    auto* const dst_pixel = dst.data() + (static_cast<size_t>(y) * dst_width + x) * channels;
                    std::copy_n(src_pixel, channels, dst_pixel);
                }
            }
            return dst;
        }

        // A width/height of 0 means "not given": both omitted keeps the source size, one
        // omitted follows the source aspect ratio. Negative sizes and sizes past
        // MAX_CAPTURE_DIMENSION are rejected.
        inline std::expected<std::pair<int, int>, std::string> resolve_capture_size(int src_width,
                                                                                    int src_height,
                                                                                    int width,
                                                                                    int height) {
            if (width < 0 || height < 0) {
                return std::unexpected(std::format("Capture size must be positive (got {}x{})", width, height));
            }
            if (width == 0 && height == 0) {
                return std::pair{src_width, src_height};
            }
            const auto follow_aspect = [](const int src_along, const int src_across, const int across) {
                return std::max(1.0, std::round(static_cast<double>(src_along) * across / src_across));
            };
            const double out_width = width > 0 ? width : follow_aspect(src_width, src_height, height);
            const double out_height = height > 0 ? height : follow_aspect(src_height, src_width, width);
            if (out_width > MAX_CAPTURE_DIMENSION || out_height > MAX_CAPTURE_DIMENSION) {
                return std::unexpected(std::format(
                    "Capture size {:.0f}x{:.0f} exceeds the {} pixel limit per side (requested {}x{} from a {}x{} source)",
                    out_width, out_height, MAX_CAPTURE_DIMENSION, width, height, src_width, src_height));
            }
            return std::pair{static_cast<int>(out_width), static_cast<int>(out_height)};
        }

    } // namespace detail

    // MCP-domain error for a failed capture, detected at the caller's site.
    [[nodiscard]] inline lfs::Error capture_error(const lfs::ErrorCode code, std::string message,
                                                  const core::SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
        return lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::MCP,
            .user_message = std::move(message),
            .detection = site,
        });
    }

    // Encodes a capture as base64 PNG. A requested size the source cannot satisfy (negative,
    // or past MAX_CAPTURE_DIMENSION once the omitted side follows the source aspect) is the
    // caller's error and fails as InvalidArgument; a bad source buffer as Internal.
    inline lfs::Result<std::string> encode_pixels_to_base64(const uint8_t* src_pixels,
                                                            int src_width,
                                                            int src_height,
                                                            int channels,
                                                            int width = 0,
                                                            int height = 0) {
        if (!src_pixels)
            return capture_error(lfs::ErrorCode::Internal, "Pixel buffer is null");
        if (src_width <= 0 || src_height <= 0 || channels < 1 || channels > 4)
            return capture_error(lfs::ErrorCode::Internal,
                                 std::format("Pixel buffer {}x{} with {} channels cannot be encoded", src_width,
                                             src_height, channels));

        const auto size = detail::resolve_capture_size(src_width, src_height, width, height);
        if (!size)
            return capture_error(lfs::ErrorCode::InvalidArgument, size.error());

        const auto [out_width, out_height] = *size;
        // stb_image_write sizes its filtered scanlines, (width * channels + 1) * height, in int.
        if ((static_cast<std::int64_t>(out_width) * channels + 1) * out_height > std::numeric_limits<int>::max())
            return capture_error(lfs::ErrorCode::InvalidArgument,
                                 std::format("Capture size {}x{} with {} channels is too large to encode as PNG",
                                             out_width, out_height, channels));

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
            return capture_error(lfs::ErrorCode::Internal,
                                 std::format("PNG encoding of a {}x{} capture failed", out_width, out_height));

        return core::base64_encode(png_buf);
    }

    inline lfs::Result<std::string> encode_render_tensor_to_base64(core::Tensor image,
                                                                   int width = 0,
                                                                   int height = 0) {
        image = image.clone().to(core::Device::CPU).to(core::DataType::Float32);
        if (image.ndim() == 4)
            image = image.squeeze(0);
        if (image.ndim() != 3)
            return capture_error(lfs::ErrorCode::Internal,
                                 std::format("Render tensor must be 3D (got {} dimensions)", image.ndim()));
        const auto layout = rendering::detectImageLayout(image);
        if (layout == rendering::ImageLayout::Unknown)
            return capture_error(lfs::ErrorCode::Internal, "Render tensor has an unsupported image layout");
        if (layout == rendering::ImageLayout::CHW)
            image = image.permute({1, 2, 0});
        image = (image.clamp(0, 1) * 255.0f).to(core::DataType::UInt8).contiguous();

        const int src_height = static_cast<int>(image.shape()[0]);
        const int src_width = static_cast<int>(image.shape()[1]);
        const int channels = static_cast<int>(image.shape()[2]);
        assert(channels >= 1 && channels <= 4);

        return encode_pixels_to_base64(image.ptr<uint8_t>(), src_width, src_height, channels, width, height);
    }

} // namespace lfs::mcp
