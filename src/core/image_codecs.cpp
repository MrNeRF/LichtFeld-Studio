/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_codecs.hpp"
#include "core/source_metadata.hpp"
#include "image_avif.hpp"
#include "image_exr.hpp"
#include "image_heif.hpp"
#include "image_jxl.hpp"

#include "core/color_transfer.h"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"

#include <jerror.h>
#include <jpeglib.h>
#include <png.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <tiffio.h>
#include <webp/decode.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <set>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfs::core::image_codecs {

    namespace {

        constexpr std::array<std::uint8_t, 8> kPngSignature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};

        // Shared bounded ownership for encoded destinations. C callbacks can
        // record allocation failure without unwinding through a codec ABI.
        struct EncodedMemory {
            enum class Failure { None,
                                 Limit,
                                 Allocation };
            Failure failure = Failure::None;
            std::vector<std::uint8_t> bytes;
            std::size_t limit = 0;

            bool write_at(std::size_t position, const std::uint8_t* data, std::size_t count) noexcept {
                if (position > limit || count > limit - position) {
                    failure = Failure::Limit;
                    return false;
                }
                if (!count)
                    return true;
                try {
                    const auto required = position + count;
                    if (required > bytes.capacity()) {
                        // Keep amortized geometric growth without doubling a
                        // large output's allocation up to the full byte limit.
                        const auto extra = bytes.capacity() / 2;
                        const auto grown = bytes.capacity() + std::min(extra, limit - bytes.capacity());
                        bytes.reserve(std::max(required, grown));
                    }
                    if (position == bytes.size())
                        bytes.insert(bytes.end(), data, data + count);
                    else {
                        if (required > bytes.size())
                            bytes.resize(required, 0);
                        std::memcpy(bytes.data() + position, data, count);
                    }
                    return true;
                } catch (const std::bad_alloc&) {
                    // LFS-CENSUS-OK(empty-catch): retain the failure at the C
                    // boundary; the caller returns ResourceExhausted.
                    failure = Failure::Allocation;
                    return false;
                }
            }
            bool append(const std::uint8_t* data, std::size_t count) noexcept {
                return write_at(bytes.size(), data, count);
            }
        };

        std::string lower_extension(const std::filesystem::path& path) {
            auto extension = lfs::core::path_to_utf8(path.extension());
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension;
        }

        bool read_file(const std::filesystem::path& path, std::vector<std::uint8_t>& data, std::string& error) {
            std::ifstream file(native_fs::native_path(path), std::ios::binary | std::ios::ate);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            const auto end = file.tellg();
            if (end < 0) {
                error = "Could not determine file size for " + path_to_utf8(path);
                return false;
            }
            const auto size = static_cast<std::size_t>(end);
            file.seekg(0);
            data.resize(size);
            if (size != 0 && !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
                error = "Could not read file " + path_to_utf8(path);
                return false;
            }
            return true;
        }

        bool read_prefix(const std::filesystem::path& path, std::vector<std::uint8_t>& data, std::string& error,
                         const std::size_t limit = 32) {
            std::ifstream file(native_fs::native_path(path), std::ios::binary);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            data.resize(limit);
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            data.resize(static_cast<std::size_t>(file.gcount()));
            if (file.bad()) {
                error = "Could not read file " + path_to_utf8(path);
                return false;
            }
            return true;
        }

        bool is_png(const std::vector<std::uint8_t>& data) {
            return data.size() >= kPngSignature.size() &&
                   std::equal(kPngSignature.begin(), kPngSignature.end(), data.begin());
        }

        bool png_gray_has_transparency(const std::vector<std::uint8_t>& data) {
            if (!is_png(data) || data.size() < 33 || data[25] != PNG_COLOR_TYPE_GRAY)
                return false;
            std::size_t offset = 33;
            while (offset + 12 <= data.size()) {
                const auto length = (static_cast<std::size_t>(data[offset]) << 24) |
                                    (static_cast<std::size_t>(data[offset + 1]) << 16) |
                                    (static_cast<std::size_t>(data[offset + 2]) << 8) |
                                    static_cast<std::size_t>(data[offset + 3]);
                if (length > data.size() - offset - 12)
                    return false;
                if (std::memcmp(data.data() + offset + 4, "tRNS", 4) == 0)
                    return true;
                if (std::memcmp(data.data() + offset + 4, "IDAT", 4) == 0)
                    return false;
                offset += 12 + length;
            }
            return false;
        }

        bool is_jpeg(const std::vector<std::uint8_t>& data) {
            return data.size() >= 2 && data[0] == 0xff && data[1] == 0xd8;
        }

        bool is_webp(const std::vector<std::uint8_t>& data) {
            return data.size() >= 12 && std::memcmp(data.data(), "RIFF", 4) == 0 &&
                   std::memcmp(data.data() + 8, "WEBP", 4) == 0;
        }

        bool is_bmp(const std::vector<std::uint8_t>& data) {
            return data.size() >= 2 && data[0] == 'B' && data[1] == 'M';
        }

        bool is_tiff(std::span<const std::uint8_t> data) {
            return data.size() >= 4 && ((data[0] == 'I' && data[1] == 'I' && (data[2] == 42 || data[2] == 43) && data[3] == 0) ||
                                        (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && (data[3] == 42 || data[3] == 43)));
        }

        bool is_hdr(const std::vector<std::uint8_t>& data) {
            return data.size() >= 10 && (std::memcmp(data.data(), "#?RADIANCE", 10) == 0 ||
                                         std::memcmp(data.data(), "#?RGBE", 6) == 0);
        }

        bool is_exr(std::span<const std::uint8_t> data) {
            return data.size() >= 4 && data[0] == 0x76 && data[1] == 0x2f && data[2] == 0x31 && data[3] == 0x01;
        }

        void set_error(std::string& error, const char* prefix, const char* detail) {
            error = prefix;
            if (detail && *detail) {
                error += ": ";
                error += detail;
            }
        }

        std::FILE* open_binary_file(const std::filesystem::path& path) {
            return open_file(path, "rb");
        }

        std::FILE* open_output_file(const std::filesystem::path& path) {
            return open_file(path, "wb");
        }

        void* allocate_target(DecodeTarget& target, const std::size_t bytes, std::string& error) {
            if (target.data)
                return target.data;
            target.data = target.allocate ? target.allocate(bytes, target.user) : std::malloc(bytes);
            if (!target.data)
                error = "Could not allocate image buffer";
            return target.data;
        }

        bool decode_exr_memory_to_buffer(std::span<const std::uint8_t> input, DecodeTarget& target,
                                         Probe& result, std::string& error) {
            error.clear();
            const auto channels = target.channels == 0 ? 4 : target.channels;
            if (target.sample_type != SampleType::Float32 || (channels != 3 && channels != 4)) {
                error = "EXR: target requires Float32 RGB/RGBA samples";
                return false;
            }
            Image image;
            if (!decode_exr_memory(input, image, error))
                return false;
            const auto pixels = size_t(image.width) * image.height;
            void* output = allocate_target(target, pixels * channels * sizeof(float), error);
            if (!output)
                return false;
            if (channels == 4)
                std::memcpy(output, image.data.data(), image.data.size());
            else
                for (size_t i = 0; i < pixels; ++i)
                    std::memcpy(static_cast<uint8_t*>(output) + i * 3 * sizeof(float),
                                image.data.data() + i * 4 * sizeof(float), 3 * sizeof(float));
            result = {image.width, image.height, channels, SampleType::Float32};
            return true;
        }

        struct PngReadOwner {
            png_structp png = nullptr;
            png_infop info = nullptr;
            FILE* file = nullptr;
            ~PngReadOwner() {
                png_destroy_read_struct(&png, &info, nullptr);
                if (file)
                    std::fclose(file);
            }
        };

        struct PngReadContext {
            const std::uint8_t* data = nullptr;
            std::size_t size = 0;
            std::size_t offset = 0;
        };

        void png_read_callback(png_structp png, png_bytep output, const png_size_t size) {
            auto* context = static_cast<PngReadContext*>(png_get_io_ptr(png));
            if (size > context->size - context->offset)
                png_error(png, "read failed");
            std::memcpy(output, context->data + context->offset, size);
            context->offset += size;
        }

        bool read_png_pixels(png_structp png, png_infop info, png_bytepp rows) {
            // Catch libpng errors in a frame with no C++ owners. Jumping to the
            // header reader's guard would bypass the caller's row-vector destructor.
            if (setjmp(png_jmpbuf(png)))
                return false;
            png_read_image(png, rows);
            png_read_end(png, info);
            return true;
        }

        bool configure_png_target(png_structp png, png_infop info, DecodeTarget& target,
                                  Probe& result, std::string& error) {
            png_uint_32 width = 0;
            png_uint_32 height = 0;
            int bit_depth = 0;
            int color_type = 0;
            png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);
            const bool gray_alpha = color_type == PNG_COLOR_TYPE_GRAY_ALPHA ||
                                    (png_get_valid(png, info, PNG_INFO_tRNS) && color_type == PNG_COLOR_TYPE_GRAY);
            if (width == 0 || height == 0 || target.channels != 3 ||
                (target.sample_type != SampleType::UInt8 && target.sample_type != SampleType::UInt16) ||
                (bit_depth == 16 && target.sample_type == SampleType::UInt8) || gray_alpha) {
                error = "PNG layout requires conversion";
                return false;
            }
            if (color_type == PNG_COLOR_TYPE_PALETTE)
                png_set_palette_to_rgb(png);
            if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
                png_set_expand_gray_1_2_4_to_8(png);
            const bool compact_gray = color_type == PNG_COLOR_TYPE_GRAY && target.sample_type == SampleType::UInt16;
            if ((color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) && !compact_gray)
                png_set_gray_to_rgb(png);
            if (color_type == PNG_COLOR_TYPE_GRAY_ALPHA || color_type == PNG_COLOR_TYPE_RGBA)
                png_set_strip_alpha(png);
            if (bit_depth == 8 && target.sample_type == SampleType::UInt16)
                png_set_expand_16(png);
            png_read_update_info(png, info);
            const int output_channels = png_get_channels(png, info);
            const int output_bit_depth = png_get_bit_depth(png, info);
            const int expected_bit_depth = target.sample_type == SampleType::UInt16 ? 16 : 8;
            if (output_channels != (compact_gray ? 1 : target.channels) || output_bit_depth != expected_bit_depth) {
                error = "PNG target layout mismatch";
                return false;
            }
            if constexpr (std::endian::native == std::endian::little) {
                if (output_bit_depth == 16)
                    png_set_swap(png);
            }
            const auto bytes_per_sample = static_cast<std::size_t>(output_bit_depth / 8);
            const auto decoded_row_bytes = static_cast<std::size_t>(width) * (compact_gray ? 1 : target.channels) * bytes_per_sample;
            const auto output_row_bytes = static_cast<std::size_t>(width) * target.channels * bytes_per_sample;
            if (!allocate_target(target, output_row_bytes * height, error))
                return false;
            result = {static_cast<int>(width), static_cast<int>(height), target.channels, target.sample_type};
            std::vector<png_bytep> rows(height);
            for (png_uint_32 y = 0; y < height; ++y)
                rows[y] = static_cast<png_bytep>(target.data) + static_cast<std::size_t>(y) * decoded_row_bytes;
            if (!read_png_pixels(png, info, rows.data())) {
                error = "PNG decode failed";
                return false;
            }
            if (compact_gray) {
                auto* output = static_cast<std::uint8_t*>(target.data);
                const auto pixels = static_cast<std::size_t>(width) * height;
                for (std::size_t pixel = pixels; pixel-- > 0;) {
                    std::array<std::uint8_t, 2> value;
                    std::memcpy(value.data(), output + pixel * 2, 2);
                    for (std::size_t channel = 0; channel < 3; ++channel)
                        std::memcpy(output + (pixel * 3 + channel) * 2, value.data(), 2);
                }
            }
            return true;
        }

        bool decode_png_memory_to_buffer(const std::uint8_t* data, std::size_t size,
                                         DecodeTarget& target, Probe& result, std::string& error);

        bool decode_png_file_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                                       Probe& result, std::string& error) {
            if (target.sample_type == SampleType::UInt16 && target.channels == 3) {
                std::vector<std::uint8_t> file_data;
                if (!read_file(path, file_data, error))
                    return false;
                const bool is_16_bit_png = is_png(file_data) && file_data.size() >= 25 && file_data[24] == 16;
                int width = 0;
                int height = 0;
                int source_channels = 0;
                // STB owns its allocation. Keep that fast path only when the
                // caller did not supply a buffer or an allocator to honor.
                if (is_16_bit_png && !target.data && !target.allocate) {
                    auto* decoded = stbi_load_16_from_memory(file_data.data(), static_cast<int>(file_data.size()),
                                                             &width, &height, &source_channels, target.channels);
                    if (decoded && source_channels != 2 && !png_gray_has_transparency(file_data)) {
                        target.data = decoded;
                        result = {width, height, target.channels, target.sample_type};
                        return true;
                    }
                    stbi_image_free(decoded);
                }
                return decode_png_memory_to_buffer(file_data.data(), file_data.size(), target, result, error);
            }
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            PngReadOwner owner;
            owner.file = file;
            png_structp png = owner.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = owner.info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                error = "PNG decode failed";
                return false;
            }
            png_init_io(png, file);
            png_read_info(png, info);
            const bool success = configure_png_target(png, info, target, result, error);
            return success;
        }

        bool decode_png_memory_to_buffer(const std::uint8_t* data, const std::size_t size,
                                         DecodeTarget& target, Probe& result, std::string& error) {
            PngReadOwner owner;
            png_structp png = owner.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = owner.info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                error = "PNG decode failed";
                return false;
            }
            PngReadContext input{data, size, 0};
            png_set_read_fn(png, &input, png_read_callback);
            png_read_info(png, info);
            const bool success = configure_png_target(png, info, target, result, error);
            return success;
        }

        bool decode_png(const std::vector<std::uint8_t>& file_data, Image& result, std::string& error) {
            if (!is_png(file_data)) {
                error = "Invalid PNG signature";
                return false;
            }
            PngReadOwner owner;
            png_structp png = owner.png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = owner.info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                error = "PNG decode failed";
                return false;
            }

            PngReadContext input{file_data.data(), file_data.size(), 0};
            png_set_read_fn(png, &input, png_read_callback);
            png_read_info(png, info);
            png_uint_32 width = 0;
            png_uint_32 height = 0;
            int bit_depth = 0;
            int color_type = 0;
            png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);
            if (color_type == PNG_COLOR_TYPE_PALETTE)
                png_set_palette_to_rgb(png);
            if (png_get_valid(png, info, PNG_INFO_tRNS))
                png_set_tRNS_to_alpha(png);
            if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
                png_set_expand_gray_1_2_4_to_8(png);
            png_read_update_info(png, info);
            const int channels = png_get_channels(png, info);
            bit_depth = png_get_bit_depth(png, info);
            if (width == 0 || height == 0 || channels < 1 || channels > 4 || (bit_depth != 8 && bit_depth != 16)) {
                error = "Unsupported PNG layout";
                return false;
            }
            if constexpr (std::endian::native == std::endian::little) {
                if (bit_depth == 16)
                    png_set_swap(png);
            }
            const std::size_t bytes_per_sample = static_cast<std::size_t>(bit_depth / 8);
            const std::size_t row_bytes = static_cast<std::size_t>(width) * channels * bytes_per_sample;
            result.width = static_cast<int>(width);
            result.height = static_cast<int>(height);
            result.channels = channels;
            result.sample_type = bit_depth == 16 ? SampleType::UInt16 : SampleType::UInt8;
            result.data.resize(row_bytes * height);
            std::vector<png_bytep> rows(height);
            for (png_uint_32 y = 0; y < height; ++y)
                rows[y] = result.data.data() + static_cast<std::size_t>(y) * row_bytes;
            const bool success = read_png_pixels(png, info, rows.data());
            if (!success)
                error = "PNG decode failed";
            return success;
        }

        struct JpegError {
            jpeg_error_mgr base;
            jmp_buf jump;
            char message[JMSG_LENGTH_MAX]{};
        };

        void jpeg_error_exit(j_common_ptr common) {
            auto* error = reinterpret_cast<JpegError*>(common->err);
            (*common->err->format_message)(common, error->message);
            longjmp(error->jump, 1);
        }

        bool decode_jpeg_bytes(const std::uint8_t* bytes, std::size_t size, Image& result, std::string& error) {
            jpeg_decompress_struct cinfo{};
            JpegError jerror;
            cinfo.err = jpeg_std_error(&jerror.base);
            jerror.base.error_exit = jpeg_error_exit;
            if (setjmp(jerror.jump)) {
                jpeg_destroy_decompress(&cinfo);
                set_error(error, "JPEG decode failed", jerror.message);
                return false;
            }
            jpeg_create_decompress(&cinfo);
            jpeg_mem_src(&cinfo, const_cast<unsigned char*>(bytes), size);
            jpeg_read_header(&cinfo, TRUE);
            jpeg_start_decompress(&cinfo);
            result.width = static_cast<int>(cinfo.output_width);
            result.height = static_cast<int>(cinfo.output_height);
            result.channels = cinfo.output_components;
            result.sample_type = SampleType::UInt8;
            result.data.resize(static_cast<std::size_t>(result.width) * result.height * result.channels);
            while (cinfo.output_scanline < cinfo.output_height) {
                auto* row = result.data.data() + static_cast<std::size_t>(cinfo.output_scanline) * result.width * result.channels;
                JSAMPROW rows[] = {row};
                jpeg_read_scanlines(&cinfo, rows, 1);
            }
            jpeg_finish_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            return true;
        }

        bool decode_jpeg_source(jpeg_decompress_struct& cinfo, DecodeTarget& target,
                                Probe& result, std::string& error) {
            jpeg_read_header(&cinfo, TRUE);
            if (target.sample_type != SampleType::UInt8 || (target.channels != 1 && target.channels != 3)) {
                error = "Unsupported JPEG target layout";
                return false;
            }
            cinfo.out_color_space = target.channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
            if (target.max_width > 0) {
                cinfo.scale_num = 1;
                cinfo.scale_denom = jpeg_scale_denominator(std::max(cinfo.image_width, cinfo.image_height),
                                                           static_cast<unsigned>(target.max_width));
            }
            jpeg_start_decompress(&cinfo);
            if (cinfo.output_components != target.channels) {
                error = "JPEG target channel mismatch";
                return false;
            }
            const auto width = static_cast<int>(cinfo.output_width);
            const auto height = static_cast<int>(cinfo.output_height);
            const auto row_bytes = static_cast<std::size_t>(width) * target.channels;
            if (!allocate_target(target, row_bytes * height, error))
                return false;
            result = {width, height, target.channels, SampleType::UInt8};
            while (cinfo.output_scanline < cinfo.output_height) {
                auto* row = static_cast<JSAMPLE*>(target.data) + static_cast<std::size_t>(cinfo.output_scanline) * row_bytes;
                JSAMPROW rows[] = {row};
                jpeg_read_scanlines(&cinfo, rows, 1);
            }
            jpeg_finish_decompress(&cinfo);
            return true;
        }

        bool decode_jpeg_file_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                                        Probe& result, std::string& error) {
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            jpeg_decompress_struct cinfo{};
            JpegError jerror;
            cinfo.err = jpeg_std_error(&jerror.base);
            jerror.base.error_exit = jpeg_error_exit;
            if (setjmp(jerror.jump)) {
                jpeg_destroy_decompress(&cinfo);
                std::fclose(file);
                set_error(error, "JPEG decode failed", jerror.message);
                return false;
            }
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            const bool success = decode_jpeg_source(cinfo, target, result, error);
            jpeg_destroy_decompress(&cinfo);
            std::fclose(file);
            return success;
        }

        bool decode_jpeg_memory_to_buffer(const std::uint8_t* data, const std::size_t size,
                                          DecodeTarget& target, Probe& result, std::string& error) {
            jpeg_decompress_struct cinfo{};
            JpegError jerror;
            cinfo.err = jpeg_std_error(&jerror.base);
            jerror.base.error_exit = jpeg_error_exit;
            if (setjmp(jerror.jump)) {
                jpeg_destroy_decompress(&cinfo);
                set_error(error, "JPEG decode failed", jerror.message);
                return false;
            }
            jpeg_create_decompress(&cinfo);
            jpeg_mem_src(&cinfo, const_cast<unsigned char*>(data), size);
            const bool success = decode_jpeg_source(cinfo, target, result, error);
            jpeg_destroy_decompress(&cinfo);
            return success;
        }

        bool probe_jpeg_file(const std::filesystem::path& path, Probe& result, std::string& error) {
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            jpeg_decompress_struct cinfo{};
            JpegError jerror;
            cinfo.err = jpeg_std_error(&jerror.base);
            jerror.base.error_exit = jpeg_error_exit;
            if (setjmp(jerror.jump)) {
                jpeg_destroy_decompress(&cinfo);
                std::fclose(file);
                set_error(error, "JPEG header probe failed", jerror.message);
                return false;
            }
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            jpeg_read_header(&cinfo, TRUE);
            result = {static_cast<int>(cinfo.image_width), static_cast<int>(cinfo.image_height),
                      cinfo.num_components == 1 ? 1 : 3, SampleType::UInt8};
            jpeg_destroy_decompress(&cinfo);
            std::fclose(file);
            return true;
        }

        bool probe_jpeg_bytes(const std::uint8_t* bytes, std::size_t size, Probe& result, std::string& error) {
            jpeg_decompress_struct cinfo{};
            JpegError jerror;
            cinfo.err = jpeg_std_error(&jerror.base);
            jerror.base.error_exit = jpeg_error_exit;
            if (setjmp(jerror.jump)) {
                jpeg_destroy_decompress(&cinfo);
                set_error(error, "JPEG header probe failed", jerror.message);
                return false;
            }
            jpeg_create_decompress(&cinfo);
            jpeg_mem_src(&cinfo, const_cast<unsigned char*>(bytes), size);
            jpeg_read_header(&cinfo, TRUE);
            result = {static_cast<int>(cinfo.image_width), static_cast<int>(cinfo.image_height),
                      cinfo.num_components == 1 ? 1 : 3, SampleType::UInt8};
            jpeg_destroy_decompress(&cinfo);
            return true;
        }

        void tiff_quiet_handler(const char*, const char*, va_list) {}

        void install_tiff_quiet_handlers() {
            static const bool installed = [] {
                TIFFSetErrorHandler(tiff_quiet_handler);
                TIFFSetWarningHandler(tiff_quiet_handler);
                return true;
            }();
            (void)installed;
        }

        int tiff_error_handler_ext_r(TIFF*, void* user_data, const char* module,
                                     const char* format, va_list args) noexcept {
            auto& error = *static_cast<std::string*>(user_data);
            try {
                std::array<char, 512> message{};
                std::vsnprintf(message.data(), message.size(), format, args);
                if (!error.empty())
                    error += "; ";
                if (module && *module) {
                    // libtiff names the file it was given; show the caller's spelling, not the extended one.
                    std::string_view name = module;
                    if (name.starts_with(R"(\\?\UNC\)"))
                        error += R"(\\)" + std::string(name.substr(8)) + ": ";
                    else
                        error += std::string(name.starts_with(R"(\\?\)") ? name.substr(4) : name) + ": ";
                }
                error += message.data();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): preserve the C boundary. The
                // failed codec call returns its fallback diagnostic outside C.
                error.clear();
            }
            return 1;
        }

        int tiff_warning_handler_ext_r(TIFF*, void*, const char*, const char*, va_list) {
            return 1;
        }

        TIFF* open_tiff(const std::filesystem::path& path, const char* mode, std::string& error) {
            install_tiff_quiet_handlers();
            error.clear();
            std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
            if (!options) {
                error = "TIFF open failed: could not allocate codec options";
                return nullptr;
            }
            TIFFOpenOptionsSetErrorHandlerExtR(options.get(), tiff_error_handler_ext_r, &error);
            TIFFOpenOptionsSetWarningHandlerExtR(options.get(), tiff_warning_handler_ext_r, nullptr);
#ifdef _WIN32
            TIFF* tiff = TIFFOpenWExt(native_fs::native_path(path).c_str(), mode, options.get());
#else
            const auto path_utf8 = path_to_utf8(path);
            TIFF* tiff = TIFFOpenExt(path_utf8.c_str(), mode, options.get());
#endif
            if (!tiff) {
                if (error.empty())
                    error = "TIFF open failed: invalid file structure or unsupported codec";
                else
                    error = "TIFF open failed: " + error;
            }
            return tiff;
        }

        // Borrowed input or a bounded seekable destination. All callbacks are
        // nonthrowing; diagnostics and failure flags are consumed outside C.
        struct TiffMemory {
            std::span<const std::uint8_t> input;
            EncodedMemory* output = nullptr;
            std::size_t position = 0;
            bool failed = false;
            bool allocation_failed = false;
            std::array<char, 512> message{};

            std::span<const std::uint8_t> bytes() const noexcept {
                return output ? std::span<const std::uint8_t>(output->bytes) : input;
            }
            void describe(const char* text) noexcept {
                std::strncpy(message.data(), text, message.size() - 1);
                message.back() = '\0';
            }
            void fail(const char* text) noexcept {
                failed = true;
                describe(text);
            }
            static tmsize_t read(thandle_t handle, void* data, tmsize_t size) noexcept {
                auto& self = *static_cast<TiffMemory*>(handle);
                if (size < 0) {
                    self.fail("Invalid TIFF read size");
                    return -1;
                }
                const auto bytes = self.bytes();
                const auto count = self.position >= bytes.size() ? 0 : std::min<std::size_t>(size, bytes.size() - self.position);
                if (count)
                    std::memcpy(data, bytes.data() + self.position, count);
                self.position += count;
                return static_cast<tmsize_t>(count);
            }
            static tmsize_t write(thandle_t handle, void* data, tmsize_t size) noexcept {
                auto& self = *static_cast<TiffMemory*>(handle);
                // libtiff can reach a directory callback with a null payload
                // after an internal allocation fails. Never copy that payload.
                if (!data && size > 0)
                    self.allocation_failed = true;
                if (self.failed || !self.output || size < 0 || (!data && size != 0)) {
                    self.fail("Invalid TIFF write");
                    return -1;
                }
                if (!self.output->write_at(self.position, static_cast<const std::uint8_t*>(data), size)) {
                    self.fail("TIFF encoded destination exhausted");
                    return -1;
                }
                self.position += static_cast<std::size_t>(size);
                return size;
            }
            static toff_t seek(thandle_t handle, toff_t offset, int origin) noexcept {
                auto& self = *static_cast<TiffMemory*>(handle);
                const uint64_t limit = self.output ? self.output->limit : self.input.size();
                uint64_t target = offset;
                if (origin == SEEK_CUR || origin == SEEK_END) {
                    const uint64_t base = origin == SEEK_CUR ? self.position : self.bytes().size();
                    // Unsigned toff_t carries signed relative offsets.
                    const auto delta = static_cast<int64_t>(offset);
                    const auto magnitude = delta < 0 ? uint64_t(-(delta + 1)) + 1 : uint64_t(delta);
                    if ((delta < 0 && magnitude > base) || (delta >= 0 && magnitude > limit - base)) {
                        if (self.output)
                            self.output->failure = EncodedMemory::Failure::Limit;
                        self.fail("TIFF relative seek outside buffer bounds");
                        return static_cast<toff_t>(-1);
                    }
                    target = delta < 0 ? base - magnitude : base + magnitude;
                } else if (origin != SEEK_SET) {
                    self.fail("Invalid TIFF seek origin");
                    return static_cast<toff_t>(-1);
                }
                if (target > limit) {
                    if (self.output)
                        self.output->failure = EncodedMemory::Failure::Limit;
                    self.fail("TIFF seek outside buffer bounds");
                    return static_cast<toff_t>(-1);
                }
                self.position = static_cast<std::size_t>(target);
                return target;
            }
            static int diagnostic(TIFF*, void* context, const char*, const char* format, va_list args) noexcept {
                auto& self = *static_cast<TiffMemory*>(context);
                // libtiff can report a recoverable tag error and still decode.
                // Only client I/O callbacks mark the input as failed.
                std::array<char, 512> text{};
                std::vsnprintf(text.data(), text.size(), format ? format : "TIFF codec error", args);
                // Inspect the provider's format, not interpolated tag names or
                // file data. libtiff has no structured allocation error code.
                const std::string_view message = format ? format : "";
                if (message.starts_with("Failed to allocate ") || message.starts_with("Cannot allocate ") ||
                    message.starts_with("No space ") || message.starts_with("Out of memory") ||
                    message == "Not enough memory" || message.starts_with("%s: Out of memory") ||
                    message.starts_with("%s: Failed to allocate ") || message.starts_with("Memory allocation of ") ||
                    message.starts_with("Cumulated memory allocation of ") ||
                    message == "malloc(sizeof(TIFFOffsetAndDirNumber)) failed" ||
                    message == "Insertion in tif_map_dir_offset_to_number failed" ||
                    message == "Insertion in tif_map_dir_number_to_offset failed")
                    self.allocation_failed = true;
                if (!self.failed)
                    self.describe(text.data());
                return 1;
            }
            TIFF* open(const char* mode) {
                std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
                if (!options) {
                    allocation_failed = true;
                    fail("Cannot allocate TIFF codec options");
                    return nullptr;
                }
                TIFFOpenOptionsSetErrorHandlerExtR(options.get(), diagnostic, this);
                TIFFOpenOptionsSetWarningHandlerExtR(options.get(), tiff_warning_handler_ext_r, nullptr);
                return TIFFClientOpenExt("TIFF buffer", mode, this, read, write, seek, [](thandle_t) { return 0; }, [](thandle_t handle) -> toff_t { return static_cast<TiffMemory*>(handle)->bytes().size(); }, [](thandle_t, void**, toff_t*) { return 0; }, [](thandle_t, void*, toff_t) {}, options.get());
            }
        };

        bool supported_tiff_samples(const std::uint16_t bits, const std::uint16_t sample_format) {
            return (sample_format == SAMPLEFORMAT_UINT && (bits == 8 || bits == 16 || bits == 32)) ||
                   (sample_format == SAMPLEFORMAT_IEEEFP && bits == 32);
        }

        void invert_tiff_samples(std::uint8_t* samples, const std::size_t count,
                                 const std::uint16_t bits_per_sample) {
            const auto byte_count = count * (bits_per_sample / 8);
            for (std::size_t i = 0; i < byte_count; ++i)
                samples[i] = static_cast<std::uint8_t>(~samples[i]);
        }

        bool tiff_photometric_is_raw(const std::uint16_t photometric) {
            return photometric == PHOTOMETRIC_MINISBLACK || photometric == PHOTOMETRIC_MINISWHITE ||
                   photometric == PHOTOMETRIC_RGB;
        }

        // Stored sample order for gray and RGB. The orientation tag is metadata for
        // viewers; camera poses are computed on the bytes in the file, so it is not applied.
        bool read_tiff_tiles(TIFF* tiff, std::uint8_t* dest, const std::uint32_t width, const std::uint32_t height,
                             const std::uint16_t channels, const std::size_t sample_size, const std::uint16_t planar,
                             std::string& error) {
            std::uint32_t tile_width = 0;
            std::uint32_t tile_height = 0;
            TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &tile_width);
            TIFFGetField(tiff, TIFFTAG_TILELENGTH, &tile_height);
            const auto tile_bytes = TIFFTileSize(tiff);
            const auto tile_row = TIFFTileRowSize(tiff);
            if (tile_width == 0 || tile_height == 0 || tile_bytes <= 0 || tile_row <= 0) {
                error = "TIFF tile size is zero";
                return false;
            }
            const auto tile_row_bytes = static_cast<std::size_t>(tile_row);
            const auto pixel_bytes = static_cast<std::size_t>(channels) * sample_size;
            const bool separate = planar == PLANARCONFIG_SEPARATE;
            if (tile_row_bytes < static_cast<std::size_t>(tile_width) * (separate ? sample_size : pixel_bytes) ||
                tile_height > static_cast<std::size_t>(tile_bytes) / tile_row_bytes) {
                error = "TIFF tile is smaller than its declared samples";
                return false;
            }
            if (!separate && reinterpret_cast<std::uintptr_t>(dest) % sample_size == 0 &&
                tile_width == width && tile_height == height &&
                static_cast<std::size_t>(tile_bytes) == static_cast<std::size_t>(width) * height * pixel_bytes) {
                if (TIFFReadTile(tiff, dest, 0, 0, 0, 0) < 0) {
                    error = error.empty() ? "TIFF tile read failed" : "TIFF tile read failed: " + error;
                    return false;
                }
                return true;
            }
            std::vector<std::uint8_t> tile(static_cast<std::size_t>(tile_bytes));
            const std::uint16_t planes = separate ? channels : static_cast<std::uint16_t>(1);
            for (std::uint16_t plane = 0; plane < planes; ++plane) {
                for (std::uint32_t y = 0; y < height;) {
                    const auto rows = std::min(tile_height, height - y);
                    for (std::uint32_t x = 0; x < width;) {
                        const auto cols = std::min(tile_width, width - x);
                        if (TIFFReadTile(tiff, tile.data(), x, y, 0, plane) < 0) {
                            error = error.empty() ? "TIFF tile read failed" : "TIFF tile read failed: " + error;
                            return false;
                        }
                        for (std::uint32_t row = 0; row < rows; ++row) {
                            const auto* src = tile.data() + static_cast<std::size_t>(row) * tile_row_bytes;
                            if (!separate) {
                                std::memcpy(dest + (static_cast<std::size_t>(y + row) * width + x) * pixel_bytes, src,
                                            static_cast<std::size_t>(cols) * pixel_bytes);
                            } else {
                                auto* dst = dest + ((static_cast<std::size_t>(y + row) * width + x) * channels + plane) * sample_size;
                                for (std::uint32_t col = 0; col < cols; ++col)
                                    std::memcpy(dst + static_cast<std::size_t>(col) * channels * sample_size,
                                                src + static_cast<std::size_t>(col) * sample_size, sample_size);
                            }
                        }
                        if (cols < tile_width)
                            break;
                        x += tile_width;
                    }
                    if (rows < tile_height)
                        break;
                    y += tile_height;
                }
            }
            return true;
        }

        // Palette, YCbCr and CMYK need the RGBA reader for their colour conversion.
        // Pass the file orientation so that conversion does not flip the stored layout.
        bool read_tiff_rgba_stored(TIFF* tiff, Image& result, const std::uint32_t width, const std::uint32_t height,
                                   std::string& error) {
            std::uint16_t orientation = ORIENTATION_TOPLEFT;
            TIFFGetFieldDefaulted(tiff, TIFFTAG_ORIENTATION, &orientation);
            if (static_cast<std::size_t>(width) > std::vector<std::uint32_t>{}.max_size() / height ||
                static_cast<std::size_t>(width) > result.data.max_size() / height / 4) {
                error = "TIFF RGBA image size exceeds supported storage";
                return false;
            }
            std::vector<std::uint32_t> rgba(static_cast<std::size_t>(width) * height);
            if (!TIFFReadRGBAImageOriented(tiff, width, height, rgba.data(), orientation, 0)) {
                error = error.empty() ? "TIFF RGBA fallback failed" : "TIFF RGBA fallback failed: " + error;
                return false;
            }
            result.width = static_cast<int>(width);
            result.height = static_cast<int>(height);
            result.channels = 4;
            result.sample_type = SampleType::UInt8;
            result.data.resize(rgba.size() * 4);
            for (std::size_t i = 0; i < rgba.size(); ++i) {
                result.data[i * 4 + 0] = static_cast<std::uint8_t>(rgba[i] & 0xff);
                result.data[i * 4 + 1] = static_cast<std::uint8_t>((rgba[i] >> 8) & 0xff);
                result.data[i * 4 + 2] = static_cast<std::uint8_t>((rgba[i] >> 16) & 0xff);
                result.data[i * 4 + 3] = static_cast<std::uint8_t>((rgba[i] >> 24) & 0xff);
            }
            return true;
        }

        bool decode_tiff_samples(TIFF* tiff, Image& result, std::string& error) {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            std::uint16_t photometric = PHOTOMETRIC_MINISBLACK;
            std::uint16_t planar = PLANARCONFIG_CONTIG;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PHOTOMETRIC, &photometric);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar);
            if (width == 0 || height == 0 || width > uint32_t(std::numeric_limits<int>::max()) ||
                height > uint32_t(std::numeric_limits<int>::max()) || channels < 1 || channels > 4) {
                error = "Unsupported TIFF layout";
                return false;
            }
            if (TIFFIsTiled(tiff) && !tiff_photometric_is_raw(photometric)) {
                const bool read = read_tiff_rgba_stored(tiff, result, width, height, error);

                return read;
            }
            if (!supported_tiff_samples(bits, sample_format)) {
                error = "Unsupported TIFF sample format";
                return false;
            }
            const std::size_t sample_size = bits / 8;
            if (static_cast<std::size_t>(width) > result.data.max_size() / height / channels / sample_size) {
                error = "TIFF decoded image size exceeds supported storage";
                return false;
            }
            result.width = static_cast<int>(width);
            result.height = static_cast<int>(height);
            result.channels = channels;
            result.sample_type = sample_format == SAMPLEFORMAT_IEEEFP || bits == 32 ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
                                                                                                                       : SampleType::UInt8;
            result.data.resize(static_cast<std::size_t>(width) * height * channels * sample_size);
            const auto row_bytes = static_cast<std::size_t>(width) * channels * sample_size;
            if (TIFFIsTiled(tiff)) {
                if (!read_tiff_tiles(tiff, result.data.data(), width, height, channels, sample_size, planar, error)) {
                    return false;
                }
            } else {
                const tmsize_t scanline_size = TIFFScanlineSize(tiff);
                const auto required_row = planar == PLANARCONFIG_CONTIG ? row_bytes : row_bytes / channels;
                if (scanline_size <= 0 || static_cast<std::size_t>(scanline_size) < required_row) {
                    error = "TIFF scanline is smaller than its declared samples";
                    return false;
                }
                std::vector<std::uint8_t> row(static_cast<std::size_t>(std::max<tmsize_t>(scanline_size, 1)));
                if (planar == PLANARCONFIG_CONTIG) {
                    for (std::uint32_t y = 0; y < height; ++y) {
                        if (TIFFReadScanline(tiff, row.data(), y, 0) < 0) {
                            error = error.empty() ? "TIFF scanline read failed" : "TIFF scanline read failed: " + error;
                            return false;
                        }
                        std::memcpy(result.data.data() + static_cast<std::size_t>(y) * row_bytes, row.data(), row_bytes);
                    }
                } else {
                    // Compressed scanlines must be consumed sequentially within
                    // each plane. Alternating planes by row requires unsupported
                    // random seeks into partially decoded LZW/ZIP strips.
                    for (std::uint16_t channel = 0; channel < channels; ++channel)
                        for (std::uint32_t y = 0; y < height; ++y) {
                            if (TIFFReadScanline(tiff, row.data(), y, channel) < 0) {
                                error = error.empty() ? "TIFF planar scanline read failed" : "TIFF planar scanline read failed: " + error;
                                return false;
                            }
                            for (std::uint32_t x = 0; x < width; ++x)
                                std::memcpy(result.data.data() + (static_cast<std::size_t>(y) * width * channels +
                                                                  static_cast<std::size_t>(x) * channels + channel) *
                                                                     sample_size,
                                            row.data() + static_cast<std::size_t>(x) * sample_size, sample_size);
                        }
                }
            }
            if (photometric == PHOTOMETRIC_MINISWHITE && channels == 1 && sample_format == SAMPLEFORMAT_UINT)
                invert_tiff_samples(result.data.data(), static_cast<std::size_t>(width) * height, bits);
            // UINT32 TIFF is retained as normalized float rather than being
            // mislabeled UInt8 or rejected. Preserve endpoints using 2^32-1;
            // float32 precision is the supported in-memory representation.
            if (sample_format == SAMPLEFORMAT_UINT && bits == 32) {
                for (size_t offset = 0; offset < result.data.size(); offset += 4) {
                    uint32_t sample;
                    std::memcpy(&sample, result.data.data() + offset, 4);
                    const float normalized = static_cast<float>(double(sample) / 4294967295.0);
                    std::memcpy(result.data.data() + offset, &normalized, 4);
                }
            }

            return true;
        }

        bool decode_tiff(const std::filesystem::path& path, Image& result, std::string& error) {
            std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(open_tiff(path, "r", error), TIFFClose);
            return tiff && decode_tiff_samples(tiff.get(), result, error);
        }

        bool decode_tiff_samples_to_buffer(TIFF* tiff, DecodeTarget& target,
                                           Probe& result, std::string& error) {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            std::uint16_t photometric = PHOTOMETRIC_MINISBLACK;
            std::uint16_t planar = PLANARCONFIG_CONTIG;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PHOTOMETRIC, &photometric);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar);
            if (!width || !height || width > uint32_t(std::numeric_limits<int>::max()) ||
                height > uint32_t(std::numeric_limits<int>::max()) || channels < 1 || channels > 4 ||
                !supported_tiff_samples(bits, sample_format) ||
                static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / height / channels / (bits / 8)) {
                error = "TIFF layout requires conversion";
                return false;
            }
            const auto source_type = sample_format == SAMPLEFORMAT_IEEEFP ? SampleType::Float32
                                     : bits == 16                         ? SampleType::UInt16
                                                                          : SampleType::UInt8;
            const auto sample_size = static_cast<std::size_t>(bits / 8);
            const auto row_bytes = static_cast<std::size_t>(width) * channels * sample_size;
            const bool raw_samples = tiff_photometric_is_raw(photometric);
            const bool direct = supported_tiff_samples(bits, sample_format) &&
                                width > 0 && height > 0 && channels == target.channels && planar == PLANARCONFIG_CONTIG &&
                                source_type == target.sample_type &&
                                ((bits == 8 && target.sample_type == SampleType::UInt8) ||
                                 (bits == 16 && target.sample_type == SampleType::UInt16) ||
                                 (bits == 32 && target.sample_type == SampleType::Float32)) &&
                                (TIFFIsTiled(tiff) ? raw_samples : TIFFScanlineSize(tiff) == static_cast<tmsize_t>(row_bytes));
            if (!direct) {
                error = "TIFF layout requires conversion";
                return false;
            }
            if (!allocate_target(target, row_bytes * height, error)) {
                return false;
            }
            result = {static_cast<int>(width), static_cast<int>(height), target.channels, target.sample_type};
            if (TIFFIsTiled(tiff)) {
                if (!read_tiff_tiles(tiff, static_cast<std::uint8_t*>(target.data), width, height, channels, sample_size,
                                     planar, error)) {
                    return false;
                }
            } else {
                // libtiff's native sample transforms may require word alignment.
                // Keep the zero-copy path for normal aligned caller buffers.
                const bool unaligned = reinterpret_cast<std::uintptr_t>(target.data) % sample_size != 0;
                std::vector<std::uint32_t> scratch(unaligned ? (row_bytes + 3) / 4 : 0);
                for (std::uint32_t y = 0; y < height; ++y) {
                    auto* row = static_cast<std::uint8_t*>(target.data) + static_cast<std::size_t>(y) * row_bytes;
                    if (TIFFReadScanline(tiff, unaligned ? static_cast<void*>(scratch.data()) : row, y, 0) < 0) {
                        error = error.empty() ? "TIFF scanline read failed" : "TIFF scanline read failed: " + error;
                        return false;
                    }
                    if (unaligned)
                        std::memcpy(row, scratch.data(), row_bytes);
                }
            }
            if (photometric == PHOTOMETRIC_MINISWHITE && channels == 1 && sample_format == SAMPLEFORMAT_UINT)
                invert_tiff_samples(static_cast<std::uint8_t*>(target.data),
                                    static_cast<std::size_t>(width) * height, bits);
            // Keep dev's normalized-float representation for direct UINT32 buffers.
            if (sample_format == SAMPLEFORMAT_UINT && bits == 32) {
                auto* const samples = static_cast<std::uint8_t*>(target.data);
                for (std::size_t offset = 0; offset < row_bytes * height; offset += sizeof(std::uint32_t)) {
                    std::uint32_t sample;
                    std::memcpy(&sample, samples + offset, sizeof(sample));
                    const float normalized = static_cast<float>(static_cast<double>(sample) / 4294967295.0);
                    std::memcpy(samples + offset, &normalized, sizeof(normalized));
                }
            }

            return true;
        }

        bool decode_tiff_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                                   Probe& result, std::string& error) {
            std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(open_tiff(path, "r", error), TIFFClose);
            return tiff && decode_tiff_samples_to_buffer(tiff.get(), target, result, error);
        }

        bool decode_tiff_memory(std::span<const std::uint8_t> input, Image& result, std::string& error) {
            error.clear();
            TiffMemory source{input};
            std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(source.open("rm"), TIFFClose);
            Image staged;
            const bool decoded = tiff && decode_tiff_samples(tiff.get(), staged, error);
            tiff.reset();
            if (!decoded || source.failed) {
                if (error.empty())
                    error = source.message[0] ? source.message.data() : "TIFF memory decode failed";
                return false;
            }
            result = std::move(staged);
            return true;
        }

        bool decode_tiff_memory_to_buffer(std::span<const std::uint8_t> input, DecodeTarget& target,
                                          Probe& result, std::string& error) {
            error.clear();
            TiffMemory source{input};
            std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(source.open("rm"), TIFFClose);
            const bool decoded = tiff && decode_tiff_samples_to_buffer(tiff.get(), target, result, error);
            tiff.reset();
            if ((!decoded || source.failed) && error.empty())
                error = source.message[0] ? source.message.data() : "TIFF memory decode failed";
            return decoded && !source.failed;
        }

        bool decode_webp(const std::vector<std::uint8_t>& file_data, Image& result, std::string& error) {
            int width = 0;
            int height = 0;
            if (!WebPGetInfo(file_data.data(), file_data.size(), &width, &height)) {
                error = "Invalid WebP image";
                return false;
            }
            WebPBitstreamFeatures features{};
            if (WebPGetFeatures(file_data.data(), file_data.size(), &features) != VP8_STATUS_OK) {
                error = "Invalid WebP features";
                return false;
            }
            const int channels = features.has_alpha ? 4 : 3;
            result.width = width;
            result.height = height;
            result.channels = channels;
            result.sample_type = SampleType::UInt8;
            result.data.resize(static_cast<std::size_t>(width) * height * channels);
            const auto decoded = channels == 4
                                     ? WebPDecodeRGBAInto(file_data.data(), file_data.size(), result.data.data(), result.data.size(), width * channels)
                                     : WebPDecodeRGBInto(file_data.data(), file_data.size(), result.data.data(), result.data.size(), width * channels);
            if (!decoded) {
                error = "WebP decode failed";
                return false;
            }
            return true;
        }

        bool hdr_payload_is_complete(stbi__context& context) {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!stbi__hdr_info(&context, &width, &height, &channels) || width <= 0 || height <= 0 ||
                !stbi__mad4sizes_valid(width, height, 3, sizeof(float), 0))
                return false;

            const auto read_byte = [&context](stbi_uc& value) {
                if (context.img_buffer >= context.img_buffer_end) {
                    if (!context.read_from_callbacks || stbi__at_eof(&context))
                        return false;
                    stbi__refill_buffer(&context);
                    if (context.img_buffer >= context.img_buffer_end)
                        return false;
                }
                value = stbi__get8(&context);
                return true;
            };
            const auto has_bytes_remaining = [&context](const std::uint64_t count) {
                const auto buffered = static_cast<std::uint64_t>(context.img_buffer_end - context.img_buffer);
                if (!context.read_from_callbacks)
                    return buffered >= count;

                auto* file = static_cast<std::FILE*>(context.io_user_data);
                const auto current = std::ftell(file);
                if (current < 0 || std::fseek(file, 0, SEEK_END) != 0)
                    return false;
                const auto end = std::ftell(file);
                const bool restored = std::fseek(file, current, SEEK_SET) == 0;
                return restored && end >= current &&
                       buffered + static_cast<std::uint64_t>(end - current) >= count;
            };

            const auto pixel_count = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
            const auto flat_payload_size = pixel_count * 4;
            if (width < 8 || width >= 32768)
                return has_bytes_remaining(flat_payload_size);

            stbi_uc marker[4]{};
            for (auto& value : marker)
                if (!read_byte(value))
                    return false;
            if (marker[0] != 2 || marker[1] != 2 || (marker[2] & 0x80) != 0)
                return has_bytes_remaining(flat_payload_size - sizeof(marker));
            if ((static_cast<int>(marker[2]) << 8 | marker[3]) != width)
                return false;

            for (int row = 0; row < height; ++row) {
                if (row != 0) {
                    for (auto& value : marker)
                        if (!read_byte(value))
                            return false;
                    if (marker[0] != 2 || marker[1] != 2 || (marker[2] & 0x80) != 0 ||
                        (static_cast<int>(marker[2]) << 8 | marker[3]) != width)
                        return false;
                }
                for (int channel = 0; channel < 4; ++channel) {
                    int decoded = 0;
                    while (decoded < width) {
                        stbi_uc count = 0;
                        if (!read_byte(count) || count == 0)
                            return false;
                        if (count > 128) {
                            stbi_uc value = 0;
                            const int run = count - 128;
                            if (run > width - decoded || !read_byte(value))
                                return false;
                            decoded += run;
                        } else {
                            if (count > width - decoded)
                                return false;
                            for (int i = 0; i < count; ++i) {
                                stbi_uc value = 0;
                                if (!read_byte(value))
                                    return false;
                            }
                            decoded += count;
                        }
                    }
                }
            }
            return true;
        }

        bool hdr_payload_is_complete(const stbi_uc* data, const std::size_t size) {
            if (!data || size == 0 || size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                return false;
            stbi__context context{};
            stbi__start_mem(&context, data, static_cast<int>(size));
            return hdr_payload_is_complete(context);
        }

        bool hdr_payload_is_complete(std::FILE* file) {
            stbi__context context{};
            stbi__start_file(&context, file);
            const bool complete = hdr_payload_is_complete(context);
            std::clearerr(file);
            return std::fseek(file, 0, SEEK_SET) == 0 && complete;
        }

        bool decode_stb(const std::vector<std::uint8_t>& file_data, bool hdr, Image& result, std::string& error) {
            if (hdr && !hdr_payload_is_complete(file_data.data(), file_data.size())) {
                error = "HDR pixel data is truncated or malformed";
                return false;
            }
            int width = 0;
            int height = 0;
            int channels = 0;
            if (hdr) {
                float* decoded = stbi_loadf_from_memory(file_data.data(), static_cast<int>(file_data.size()),
                                                        &width, &height, &channels, 0);
                if (!decoded) {
                    error = "HDR decode failed";
                    return false;
                }
                result.width = width;
                result.height = height;
                result.channels = channels;
                result.sample_type = SampleType::Float32;
                const auto bytes = static_cast<std::size_t>(width) * height * channels * sizeof(float);
                result.data.resize(bytes);
                std::memcpy(result.data.data(), decoded, bytes);
                stbi_image_free(decoded);
                return true;
            }
            stbi_uc* decoded = stbi_load_from_memory(file_data.data(), static_cast<int>(file_data.size()),
                                                     &width, &height, &channels, 0);
            if (!decoded) {
                error = "STB decode failed";
                return false;
            }
            result.width = width;
            result.height = height;
            result.channels = channels;
            result.sample_type = SampleType::UInt8;
            const auto bytes = static_cast<std::size_t>(width) * height * channels;
            result.data.assign(decoded, decoded + bytes);
            stbi_image_free(decoded);
            return true;
        }

        bool decode_webp_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                                   Probe& result, std::string& error) {
            std::vector<std::uint8_t> file_data;
            if (!read_file(path, file_data, error))
                return false;
            int width = 0;
            int height = 0;
            if (!WebPGetInfo(file_data.data(), file_data.size(), &width, &height)) {
                error = "Invalid WebP image";
                return false;
            }
            if (target.sample_type != SampleType::UInt8 || target.channels != 3) {
                error = "Unsupported WebP target layout";
                return false;
            }
            const auto bytes = static_cast<std::size_t>(width) * height * target.channels;
            if (!allocate_target(target, bytes, error))
                return false;
            if (!WebPDecodeRGBInto(file_data.data(), file_data.size(), static_cast<std::uint8_t*>(target.data), bytes,
                                   width * target.channels)) {
                error = "WebP decode failed";
                return false;
            }
            result = {width, height, target.channels, target.sample_type};
            return true;
        }

        bool decode_stb_to_buffer(const std::filesystem::path& path, const bool hdr,
                                  DecodeTarget& target, Probe& result, std::string& error) {
#ifndef _WIN32
            if (hdr && target.sample_type == SampleType::Float32) {
                const int descriptor = open(path.c_str(), O_RDONLY);
                struct stat file_status{};
                if (descriptor >= 0 && fstat(descriptor, &file_status) == 0 && file_status.st_size > 0 &&
                    file_status.st_size <= std::numeric_limits<int>::max()) {
                    const auto size = static_cast<std::size_t>(file_status.st_size);
                    void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
                    if (mapped != MAP_FAILED) {
                        int width = 0;
                        int height = 0;
                        int source_channels = 0;
                        if (!hdr_payload_is_complete(static_cast<const stbi_uc*>(mapped), size)) {
                            munmap(mapped, size);
                            close(descriptor);
                            error = "HDR pixel data is truncated or malformed";
                            return false;
                        }
                        auto* decoded = stbi_loadf_from_memory(static_cast<const stbi_uc*>(mapped),
                                                               static_cast<int>(size), &width, &height,
                                                               &source_channels, target.channels);
                        munmap(mapped, size);
                        close(descriptor);
                        if (decoded) {
                            target.data = decoded;
                            result = {width, height, target.channels, target.sample_type};
                            return true;
                        }
                    } else {
                        close(descriptor);
                    }
                } else if (descriptor >= 0) {
                    close(descriptor);
                }
            }
#endif
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            int width = 0;
            int height = 0;
            int source_channels = 0;
            if (hdr) {
                if (target.sample_type != SampleType::Float32) {
                    std::fclose(file);
                    error = "Unsupported HDR target layout";
                    return false;
                }
                if (!hdr_payload_is_complete(file)) {
                    std::fclose(file);
                    error = "HDR pixel data is truncated or malformed";
                    return false;
                }
                auto* decoded = stbi_loadf_from_file(file, &width, &height, &source_channels, target.channels);
                std::fclose(file);
                if (!decoded) {
                    error = "HDR decode failed";
                    return false;
                }
                target.data = decoded;
            } else {
                if (target.sample_type != SampleType::UInt8) {
                    std::fclose(file);
                    error = "Unsupported STB target layout";
                    return false;
                }
                auto* decoded = stbi_load_from_file(file, &width, &height, &source_channels, target.channels);
                std::fclose(file);
                if (!decoded) {
                    error = "STB decode failed";
                    return false;
                }
                target.data = decoded;
            }
            result = {width, height, target.channels, target.sample_type};
            return true;
        }

        bool probe_png_file(const std::filesystem::path& path, Probe& result, std::string& error) {
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                png_destroy_read_struct(&png, &info, nullptr);
                std::fclose(file);
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                png_destroy_read_struct(&png, &info, nullptr);
                std::fclose(file);
                error = "PNG header probe failed";
                return false;
            }
            png_init_io(png, file);
            png_read_info(png, info);
            png_uint_32 width = 0;
            png_uint_32 height = 0;
            int bit_depth = 0;
            int color_type = 0;
            png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, nullptr, nullptr, nullptr);
            const int channels = color_type == PNG_COLOR_TYPE_GRAY ? 1 : color_type == PNG_COLOR_TYPE_GRAY_ALPHA                                ? 2
                                                                     : color_type == PNG_COLOR_TYPE_PALETTE || color_type == PNG_COLOR_TYPE_RGB ? 3
                                                                                                                                                : 4;
            result = {static_cast<int>(width), static_cast<int>(height), channels,
                      bit_depth == 16 ? SampleType::UInt16 : SampleType::UInt8};
            png_destroy_read_struct(&png, &info, nullptr);
            std::fclose(file);
            return true;
        }

        bool probe_stb_file(const std::filesystem::path& path, const bool hdr,
                            Probe& result, std::string& error) {
            auto* file = open_binary_file(path);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            int width = 0;
            int height = 0;
            int channels = 0;
            const bool success = stbi_info_from_file(file, &width, &height, &channels) != 0;
            std::fclose(file);
            if (!success) {
                error = hdr ? "Invalid HDR header" : "Invalid STB image header";
                return false;
            }
            result = {width, height, channels, hdr ? SampleType::Float32 : SampleType::UInt8};
            return true;
        }

        bool probe_webp_file(const std::filesystem::path& path, Probe& result, std::string& error) {
            std::vector<std::uint8_t> header;
            if (!read_prefix(path, header, error))
                return false;
            if (!is_webp(header)) {
                error = "Invalid WebP header";
                return false;
            }
            int width = 0;
            int height = 0;
            if (!WebPGetInfo(header.data(), header.size(), &width, &height)) {
                error = "Invalid WebP header";
                return false;
            }
            WebPBitstreamFeatures features{};
            if (WebPGetFeatures(header.data(), header.size(), &features) != VP8_STATUS_OK) {
                error = "Invalid WebP features";
                return false;
            }
            result = {width, height, features.has_alpha ? 4 : 3, SampleType::UInt8};
            return true;
        }

    } // namespace

    bool radiance_scanlines_complete(const std::filesystem::path& path) {
        auto* file = open_binary_file(path);
        if (!file)
            return false;
        const bool complete = hdr_payload_is_complete(file);
        std::fclose(file);
        return complete;
    }

    bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                          Probe& result, std::string& error) {
        return decode_to_buffer(path, target, result, error, false);
    }

    bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                          Probe& result, std::string& error, bool display) {
        const auto extension = lower_extension(path);
        if (extension == ".jxl")
            return decode_jxl_to_buffer(path, target, result, error, display);
        if (extension == ".avif" || extension == ".heic" || extension == ".heif")
            return decode_heif_to_buffer(path, target, result, error, display);
        if (extension == ".jpg" || extension == ".jpeg")
            return decode_jpeg_file_to_buffer(path, target, result, error);
        if (extension == ".png")
            return decode_png_file_to_buffer(path, target, result, error);
        if (extension == ".webp")
            return decode_webp_to_buffer(path, target, result, error);
        if (extension == ".tif" || extension == ".tiff")
            return decode_tiff_to_buffer(path, target, result, error);
        if (extension == ".hdr")
            return decode_stb_to_buffer(path, true, target, result, error);
        if (extension == ".bmp" || extension == ".tga")
            return decode_stb_to_buffer(path, false, target, result, error);
        error = "Unsupported image extension: " + path_to_utf8(path);
        return false;
    }

    bool decode_memory_to_buffer(const std::uint8_t* data, const std::size_t size,
                                 DecodeTarget& target, Probe& result, std::string& error) {
        return decode_memory_to_buffer(data, size, target, result, error, false);
    }

    bool decode_memory_to_buffer(const std::uint8_t* data, const std::size_t size,
                                 DecodeTarget& target, Probe& result, std::string& error, bool display) {
        if (!data || size == 0) {
            error = "Empty image buffer";
            return false;
        }
        if (is_exr({data, size}))
            return decode_exr_memory_to_buffer({data, size}, target, result, error);
        if (is_jxl({data, size}))
            return decode_jxl_to_buffer(std::span<const uint8_t>{data, size}, target, result, error, display);
        if (is_tiff({data, size}))
            return decode_tiff_memory_to_buffer({data, size}, target, result, error);
        if (is_heif({data, size}))
            return decode_heif_to_buffer(std::span<const uint8_t>{data, size}, target, result, error, display);
        if (size >= 2 && data[0] == 0xff && data[1] == 0xd8)
            return decode_jpeg_memory_to_buffer(data, size, target, result, error);
        if (size >= kPngSignature.size() && std::equal(kPngSignature.begin(), kPngSignature.end(), data))
            return decode_png_memory_to_buffer(data, size, target, result, error);
        error = "Unsupported image buffer format";
        return false;
    }

    bool probe_bounded_header(const std::filesystem::path& path, std::string& error) {
        constexpr std::size_t kReadLimit = 4096;
        std::vector<std::uint8_t> header;
        if (!read_prefix(path, header, error, kReadLimit))
            return false;

        if (is_heif(header) || is_jxl(header))
            return true; // Advisory bounded recognition; full decoding validates the payload.
        if (is_png(header)) {
            if (header.size() >= 26 && std::memcmp(header.data() + 12, "IHDR", 4) == 0 &&
                std::any_of(header.begin() + 16, header.begin() + 20, [](auto byte) { return byte != 0; }) &&
                std::any_of(header.begin() + 20, header.begin() + 24, [](auto byte) { return byte != 0; }))
                return true;
            error = "Invalid PNG header";
            return false;
        }
        if (is_webp(header)) {
            int width = 0;
            int height = 0;
            if (WebPGetInfo(header.data(), header.size(), &width, &height))
                return true;
            error = "Invalid WebP header";
            return false;
        }
        if (is_tiff(header) || is_exr(header)) {
            if (header.size() >= 16)
                return true;
            error = "Incomplete image header";
            return false;
        }
        if (is_jpeg(header) || is_bmp(header) || is_hdr(header) || lower_extension(path) == ".tga") {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (stbi_info_from_memory(header.data(), static_cast<int>(header.size()), &width, &height, &channels))
                return true;
            // Large metadata can put the dimensions beyond the prefix. Keep this advisory
            // check inconclusive so a valid image remains accepted for the later loader.
            if (header.size() == kReadLimit)
                return true;
        }
        error = "Unrecognized or incomplete image header";
        return false;
    }

    bool probe(const std::filesystem::path& path, Probe& result, std::string& error) {
        const auto extension = lower_extension(path);
        if (extension == ".jxl")
            return probe_jxl(path, result, error);
        if (extension == ".avif" || extension == ".heic" || extension == ".heif")
            return probe_heif(path, result, error);
        if (extension == ".exr")
            return probe_exr(path, result, error);
        if (extension == ".jpg" || extension == ".jpeg")
            return probe_jpeg_file(path, result, error);
        if (extension == ".png")
            return probe_png_file(path, result, error);
        if (extension == ".webp")
            return probe_webp_file(path, result, error);
        if (extension == ".bmp" || extension == ".tga")
            return probe_stb_file(path, false, result, error);
        if (extension == ".hdr")
            return probe_stb_file(path, true, result, error);
        if (extension == ".tif" || extension == ".tiff") {
            TIFF* tiff = open_tiff(path, "r", error);
            if (!tiff)
                return false;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFClose(tiff);
            if (!supported_tiff_samples(bits, sample_format)) {
                error = "Unsupported TIFF sample format";
                return false;
            }
            result = {static_cast<int>(width), static_cast<int>(height), static_cast<int>(channels),
                      sample_format == SAMPLEFORMAT_IEEEFP || bits == 32 ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
                                                                                                            : SampleType::UInt8};
            return true;
        }

        std::vector<std::uint8_t> prefix;
        if (!read_prefix(path, prefix, error))
            return false;
        if (is_exr(prefix))
            return probe_exr(path, result, error);
        if (is_tiff(prefix) || extension == ".tif" || extension == ".tiff") {
            TIFF* tiff = open_tiff(path, "r", error);
            if (!tiff)
                return false;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFClose(tiff);
            if (!supported_tiff_samples(bits, sample_format)) {
                error = "Unsupported TIFF sample format";
                return false;
            }
            result = {static_cast<int>(width), static_cast<int>(height), static_cast<int>(channels),
                      sample_format == SAMPLEFORMAT_IEEEFP || bits == 32 ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
                                                                                                            : SampleType::UInt8};
            return true;
        }
        std::vector<std::uint8_t> data;
        if (!read_file(path, data, error))
            return false;
        if (is_png(data)) {
            if (data.size() < 26 || std::memcmp(data.data() + 12, "IHDR", 4) != 0) {
                error = "Invalid PNG header";
                return false;
            }
            const auto read_be32 = [&data](std::size_t offset) {
                return (static_cast<std::uint32_t>(data[offset]) << 24) |
                       (static_cast<std::uint32_t>(data[offset + 1]) << 16) |
                       (static_cast<std::uint32_t>(data[offset + 2]) << 8) |
                       static_cast<std::uint32_t>(data[offset + 3]);
            };
            result.width = static_cast<int>(read_be32(16));
            result.height = static_cast<int>(read_be32(20));
            const int bit_depth = data[24];
            const int color_type = data[25];
            result.channels = color_type == 0 ? 1 : color_type == 2 ? 3
                                                : color_type == 3   ? 3
                                                : color_type == 4   ? 2
                                                                    : 4;
            result.sample_type = bit_depth == 16 ? SampleType::UInt16 : SampleType::UInt8;
            return true;
        }
        if (is_jpeg(data)) {
            return probe_jpeg_bytes(data.data(), data.size(), result, error);
        }
        if (is_webp(data)) {
            int width = 0;
            int height = 0;
            if (!WebPGetInfo(data.data(), data.size(), &width, &height)) {
                error = "Invalid WebP header";
                return false;
            }
            WebPBitstreamFeatures features{};
            if (WebPGetFeatures(data.data(), data.size(), &features) != VP8_STATUS_OK) {
                error = "Invalid WebP features";
                return false;
            }
            result = Probe{width, height, features.has_alpha ? 4 : 3, SampleType::UInt8};
            return true;
        }
        if (is_bmp(data)) {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!stbi_info_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, &channels)) {
                error = "Invalid BMP header";
                return false;
            }
            result = {width, height, channels, SampleType::UInt8};
            return true;
        }
        if (extension == ".tga") {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!stbi_info_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, &channels)) {
                error = "Invalid TGA header";
                return false;
            }
            result = {width, height, channels, SampleType::UInt8};
            return true;
        }
        if (is_hdr(data) || extension == ".hdr") {
            int width = 0;
            int height = 0;
            int channels = 0;
            if (!stbi_info_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, &channels)) {
                error = "Invalid HDR header";
                return false;
            }
            result = {width, height, channels, SampleType::Float32};
            return true;
        }
        error = "Unsupported image format: " + path_to_utf8(path);
        return false;
    }

    bool decode(const std::filesystem::path& path, Image& result, std::string& error) {
        return decode(path, result, error, false);
    }

    bool decode(const std::filesystem::path& path, Image& result, std::string& error, bool display) {
        const auto extension = lower_extension(path);
        if (extension == ".jxl")
            return decode_jxl(path, result, error, display);
        if (extension == ".avif" || extension == ".heic" || extension == ".heif")
            return decode_heif(path, result, error, display);
        if (extension == ".tif" || extension == ".tiff")
            return decode_tiff(path, result, error);
        if (extension == ".exr")
            return decode_exr(path, result, error);
        std::vector<std::uint8_t> data;
        if (!read_file(path, data, error))
            return false;
        if (is_tiff(data))
            return decode_tiff(path, result, error);
        if (is_exr(data))
            return decode_exr(path, result, error);
        if (is_jpeg(data))
            return decode_jpeg_bytes(data.data(), data.size(), result, error);
        if (is_png(data))
            return decode_png(data, result, error);
        if (is_webp(data))
            return decode_webp(data, result, error);
        if (is_hdr(data) || extension == ".hdr")
            return decode_stb(data, true, result, error);
        if (is_bmp(data))
            return decode_stb(data, false, result, error);
        if (extension == ".tga")
            return decode_stb(data, false, result, error);
        error = "Unsupported image format: " + path_to_utf8(path);
        return false;
    }

    bool decode_memory(const std::uint8_t* data, const size_t size, Image& result, std::string& error) {
        return decode_memory(data, size, result, error, false);
    }

    bool decode_memory(const std::uint8_t* data, const size_t size, Image& result, std::string& error, bool display) {
        if (!data || size == 0) {
            error = "Empty image buffer";
            return false;
        }
        if (is_exr({data, size}))
            return decode_exr_memory({data, size}, result, error);
        if (is_tiff({data, size}))
            return decode_tiff_memory({data, size}, result, error);
        if (is_heif({data, size}))
            return decode_heif(std::span<const uint8_t>{data, size}, result, error, display);
        if (is_jxl({data, size}))
            return decode_jxl(std::span<const uint8_t>{data, size}, result, error, display);
        if (size >= 2 && data[0] == 0xff && data[1] == 0xd8)
            return decode_jpeg_bytes(data, size, result, error);
        if (size >= kPngSignature.size() && std::equal(kPngSignature.begin(), kPngSignature.end(), data)) {
            std::vector<std::uint8_t> image_data(data, data + size);
            return decode_png(image_data, result, error);
        }
        std::vector<std::uint8_t> image_data(data, data + size);
        return decode_stb(image_data, false, result, error);
    }

    namespace {
        struct JpegEncodeState {
            jpeg_compress_struct codec{};
            JpegError error{};
            std::FILE* file = nullptr;
            bool created = false;
            bool owns_file = true;
            EncodedMemory memory;
            jpeg_destination_mgr destination{};
            std::array<JOCTET, 4096> chunk{};

            static void init_output(j_compress_ptr codec) noexcept {
                auto& self = *static_cast<JpegEncodeState*>(codec->client_data);
                self.destination.next_output_byte = self.chunk.data();
                self.destination.free_in_buffer = self.chunk.size();
            }
            static boolean flush_output(j_compress_ptr codec) noexcept {
                auto& self = *static_cast<JpegEncodeState*>(codec->client_data);
                // append returns before jumping: no exception or C++
                // destructor-bearing frame is crossed by libjpeg's error jump.
                if (!self.memory.append(self.chunk.data(), self.chunk.size()))
                    std::longjmp(self.error.jump, 1);
                init_output(codec);
                return TRUE;
            }
            static void finish_output(j_compress_ptr codec) noexcept {
                auto& self = *static_cast<JpegEncodeState*>(codec->client_data);
                if (!self.memory.append(self.chunk.data(), self.chunk.size() - self.destination.free_in_buffer))
                    std::longjmp(self.error.jump, 1);
            }
            ~JpegEncodeState() {
                if (created)
                    jpeg_destroy_compress(&codec);
                if (file && owns_file)
                    std::fclose(file);
            }
        };

        // The jump stays inside a scalar-only frame; ownership lives in the caller.
        bool encode_jpeg(JpegEncodeState* state, const std::uint8_t* data,
                         int width, int height, int channels, int quality,
                         const char* comment, std::size_t comment_size, bool full_chroma, std::size_t row_stride) {
            state->codec.err = jpeg_std_error(&state->error.base);
            state->error.base.error_exit = jpeg_error_exit;
            if (setjmp(state->error.jump)) {
                return false;
            }
            state->created = true;
            jpeg_create_compress(&state->codec);
            if (state->file)
                jpeg_stdio_dest(&state->codec, state->file);
            else {
                state->codec.client_data = state;
                state->destination.init_destination = JpegEncodeState::init_output;
                state->destination.empty_output_buffer = JpegEncodeState::flush_output;
                state->destination.term_destination = JpegEncodeState::finish_output;
                state->codec.dest = &state->destination;
            }
            state->codec.image_width = width;
            state->codec.image_height = height;
            state->codec.input_components = channels;
            state->codec.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
            jpeg_set_defaults(&state->codec);
            if (full_chroma && channels == 3) {
                state->codec.comp_info[0].h_samp_factor = 1;
                state->codec.comp_info[0].v_samp_factor = 1;
            }
            jpeg_set_quality(&state->codec, std::clamp(quality, 1, 100), TRUE);
            jpeg_start_compress(&state->codec, TRUE);
            if (comment_size != 0) {
                jpeg_write_marker(&state->codec, JPEG_COM, reinterpret_cast<const JOCTET*>(comment),
                                  static_cast<unsigned int>(std::min<std::size_t>(comment_size, 65533)));
            }
            while (state->codec.next_scanline < state->codec.image_height) {
                JSAMPROW row = const_cast<JSAMPROW>(data + static_cast<std::size_t>(state->codec.next_scanline) * row_stride);
                jpeg_write_scanlines(&state->codec, &row, 1);
            }
            jpeg_finish_compress(&state->codec);
            return true;
        }
    } // namespace

    Status encode_jpeg_memory(std::span<const std::uint8_t> data, int width, int height, int channels,
                              std::size_t row_stride, std::vector<std::uint8_t>& output,
                              const JpegEncodeOptions& options) {
        const auto failure = [](ErrorCode code, std::string message) -> Status {
            return Status::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        if (width <= 0 || height <= 0 || width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION ||
            (channels != 1 && channels != 3))
            return failure(ErrorCode::InvalidArgument, "Unsupported JPEG layout");
        const auto row_bytes = static_cast<std::size_t>(width) * channels;
        if (row_stride < row_bytes || static_cast<std::size_t>(height - 1) > (std::numeric_limits<std::size_t>::max() - row_bytes) / row_stride ||
            data.size() < row_stride * (height - 1) + row_bytes)
            return failure(ErrorCode::InvalidArgument, "Unsupported JPEG buffer size or stride");
        if (!options.max_output_bytes || options.max_output_bytes > output.max_size())
            return failure(ErrorCode::InvalidArgument, "Invalid JPEG encoded-byte limit");
        try {
            auto state = std::make_unique<JpegEncodeState>();
            state->memory.limit = options.max_output_bytes;
            if (!encode_jpeg(state.get(), data.data(), width, height, channels, options.quality,
                             options.comment ? options.comment->data() : nullptr,
                             options.comment ? options.comment->size() : 0, options.full_chroma, row_stride)) {
                if (state->memory.failure != EncodedMemory::Failure::None)
                    return failure(ErrorCode::ResourceExhausted, state->memory.failure == EncodedMemory::Failure::Limit
                                                                     ? "JPEG encoded-byte limit exhausted"
                                                                     : "Cannot allocate JPEG output buffer");
                return failure(state->error.base.msg_code == JERR_OUT_OF_MEMORY ? ErrorCode::ResourceExhausted : ErrorCode::Unavailable,
                               std::string("JPEG encode failed: ") + state->error.message);
            }
            output = std::move(state->memory.bytes);
            return {};
        } catch (const std::bad_alloc&) {
            return failure(ErrorCode::ResourceExhausted, "Cannot allocate JPEG encoder state");
        }
    }

    Result<void> encode_exr_memory(const ExrEncodeView& view, std::vector<std::uint8_t>& output,
                                   const ExrEncodeOptions& options, std::size_t max_output_bytes) {
        const auto failure = [](ErrorCode code, std::string detail) -> Result<void> {
            return Result<void>::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        if (!max_output_bytes || max_output_bytes > output.max_size() ||
            max_output_bytes > uint64_t(std::numeric_limits<int64_t>::max()))
            return failure(ErrorCode::InvalidArgument, std::format("Invalid EXR encoded-byte limit (limit={})", max_output_bytes));
        EncodedMemory memory;
        memory.limit = max_output_bytes;
        const ExrWriteStream stream{"EXR buffer", &memory,
                                    [](void* user, const void* data, uint64_t size, uint64_t offset) noexcept -> int64_t {
                                        auto& destination = *static_cast<EncodedMemory*>(user);
                                        if (destination.failure != EncodedMemory::Failure::None ||
                                            size > uint64_t(std::numeric_limits<int64_t>::max()) ||
                                            offset > destination.limit || size > destination.limit - offset) {
                                            if (destination.failure == EncodedMemory::Failure::None)
                                                destination.failure = EncodedMemory::Failure::Limit;
                                            return -1;
                                        }
                                        if (!data && size)
                                            return -1;
                                        return destination.write_at(static_cast<size_t>(offset), static_cast<const uint8_t*>(data), static_cast<size_t>(size))
                                                   ? static_cast<int64_t>(size)
                                                   : -1;
                                    }};
        const auto status = encode_exr(view, stream, options);
        if (!status || memory.failure != EncodedMemory::Failure::None) {
            if (memory.failure != EncodedMemory::Failure::None)
                return failure(ErrorCode::ResourceExhausted, memory.failure == EncodedMemory::Failure::Limit
                                                                 ? "EXR encoded-byte limit exhausted"
                                                                 : "Cannot allocate EXR output buffer");
            return status;
        }
        output = std::move(memory.bytes);
        return {};
    }

    bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                    const int width, const int height, const int channels, const int quality,
                    const std::optional<std::string>& comment, std::string& error) {
        return write_jpeg(path, data, width, height, channels, quality, comment, error, false);
    }

    namespace {
        constexpr char kJpegXmp[] = "http://ns.adobe.com/xap/1.0/";
        constexpr char kJpegExtendedXmp[] = "http://ns.adobe.com/xmp/extension/";
        constexpr char kPngXmp[] = "XML:com.adobe.xmp";

        struct XmpCollector {
            SourceXmp& result;
            const XmpLimits& limits;
            size_t retained = 0;
            std::optional<MetadataStatus> scan_status;
            std::string scan_diagnostic;
            bool supported_carrier = false;
            bool canRetain(size_t size) {
                if (result.blocks.size() >= limits.blocks || size > limits.block_bytes || size > limits.total_bytes - retained) {
                    result.status = MetadataStatus::Limited;
                    result.diagnostic = "XMP collection budget exhausted";
                    return false;
                }
                return true;
            }
            void collectTiff(const SourceExif& exif) {
                result.status = MetadataStatus::Absent;
                result.scope = "tiff-libtiff-observed-tag700";
                if (exif.tiff) {
                    const auto& directories = exif.tiff->directories;
                    for (size_t directory = 0; directory < directories.size(); ++directory) {
                        const auto& tags = directories[directory].tags;
                        for (size_t tag = 0; tag < tags.size(); ++tag) {
                            if (tags[tag].id != TIFFTAG_XMLPACKET)
                                continue;
                            XmpBlock block;
                            block.envelope = XmpEnvelope::TiffTag;
                            block.tiff_tag = TiffXmpReference{directory, tag};
                            const auto bytes = source_xmp_bytes(exif, block);
                            if (!bytes) {
                                if (result.status != MetadataStatus::Limited) {
                                    result.status = MetadataStatus::Unsupported;
                                    result.diagnostic = "TIFF XMP tag has no byte-preserving representation";
                                }
                                continue;
                            }
                            if (!canRetain(bytes->size()))
                                continue;
                            retained += bytes->size();
                            if (bytes->empty()) {
                                block.status = MetadataStatus::Malformed;
                                block.diagnostic = "TIFF XMP tag has zero byte count";
                                if (result.status != MetadataStatus::Limited)
                                    result.status = MetadataStatus::Malformed;
                                if (result.diagnostic.empty())
                                    result.diagnostic = block.diagnostic;
                            }
                            result.blocks.push_back(std::move(block));
                            if (result.status == MetadataStatus::Absent)
                                result.status = MetadataStatus::Collected;
                        }
                    }
                }
                if (exif.status == MetadataStatus::Unsupported) {
                    if (result.status != MetadataStatus::Limited)
                        result.status = MetadataStatus::Unsupported;
                    if (!result.diagnostic.empty() && !exif.diagnostic.empty())
                        result.diagnostic += "; ";
                    result.diagnostic += exif.diagnostic;
                }
            }
            void append(std::vector<uint8_t> bytes, XmpEnvelope envelope, std::optional<bool> crc = std::nullopt) {
                retained += bytes.size();
                XmpBlock block;
                block.bytes = std::move(bytes);
                block.envelope = envelope;
                block.crc_valid = crc;
                auto malformed = [&](const char* message) {
                    block.status = MetadataStatus::Malformed;
                    block.diagnostic = message;
                };
                if (envelope == XmpEnvelope::JpegApp1) {
                    block.payload_offset = sizeof(kJpegXmp);
                } else if (envelope == XmpEnvelope::JpegExtendedApp1) {
                    constexpr size_t header = sizeof(kJpegExtendedXmp) + 40;
                    block.payload_offset = std::min(header, block.bytes.size());
                    const char* diagnostic = nullptr;
                    if (!source_extended_xmp_segment(block, &diagnostic))
                        malformed(diagnostic);
                } else if (envelope == XmpEnvelope::PngInternationalText) {
                    if (crc && !*crc)
                        malformed("XMP iTXt CRC mismatch");
                    size_t offset = sizeof(kPngXmp);
                    if (block.bytes.size() < offset + 2) {
                        malformed("Truncated XMP iTXt header");
                    } else {
                        const auto compressed = block.bytes[offset++];
                        block.compressed = compressed == 1;
                        const auto method = block.bytes[offset++];
                        if (compressed > 1 || (compressed == 1 && method != 0)) {
                            malformed("Invalid XMP iTXt compression fields");
                        } else {
                            for (int field = 0; field < 2; ++field) {
                                while (offset < block.bytes.size() && block.bytes[offset])
                                    ++offset;
                                if (offset == block.bytes.size()) {
                                    malformed("Unterminated XMP iTXt language or translated keyword");
                                    break;
                                }
                                ++offset;
                            }
                            block.payload_offset = offset;
                            if (block.status == MetadataStatus::Collected && compressed) {
                                const size_t cap = std::min(limits.inflated_bytes, limits.total_bytes - retained);
                                z_stream stream{};
                                const int initialized = inflateInit(&stream);
                                if (initialized == Z_MEM_ERROR)
                                    throw std::bad_alloc();
                                if (initialized != Z_OK) {
                                    malformed("Cannot initialize XMP iTXt decompression");
                                } else {
                                    struct InflaterGuard {
                                        z_stream& stream;
                                        ~InflaterGuard() { inflateEnd(&stream); }
                                    } guard{stream};
                                    stream.next_in = block.bytes.data() + offset;
                                    stream.avail_in = static_cast<uInt>(block.bytes.size() - offset);
                                    std::array<uint8_t, 8192> scratch{};
                                    std::vector<uint8_t> output;
                                    for (;;) {
                                        // The extra byte detects excess output, including at a zero limit.
                                        const auto available = std::min(scratch.size(), cap - output.size() + 1);
                                        stream.next_out = scratch.data();
                                        stream.avail_out = static_cast<uInt>(available);
                                        const auto input_before = stream.avail_in;
                                        const int status = inflate(&stream, Z_NO_FLUSH);
                                        const auto produced = available - stream.avail_out;
                                        if (status == Z_MEM_ERROR)
                                            throw std::bad_alloc();
                                        if (produced > cap - output.size()) {
                                            block.status = MetadataStatus::Limited;
                                            block.diagnostic = "XMP inflated text budget exhausted";
                                            break;
                                        }
                                        if (status != Z_OK && status != Z_STREAM_END) {
                                            malformed("Invalid or trailing XMP iTXt compressed data");
                                            break;
                                        }
                                        output.insert(output.end(), scratch.begin(), scratch.begin() + produced);
                                        if (status == Z_STREAM_END) {
                                            if (stream.avail_in) {
                                                malformed("Invalid or trailing XMP iTXt compressed data");
                                            } else {
                                                retained += output.size();
                                                block.inflated.emplace(output.begin(), output.end());
                                            }
                                            break;
                                        }
                                        if (!produced && stream.avail_in == input_before) {
                                            malformed("Invalid or trailing XMP iTXt compressed data");
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (result.status == MetadataStatus::Absent)
                    result.status = MetadataStatus::Collected;
                if (block.status == MetadataStatus::Limited ||
                    (block.status == MetadataStatus::Malformed && result.status != MetadataStatus::Limited))
                    result.status = block.status;
                if (result.diagnostic.empty() && !block.diagnostic.empty())
                    result.diagnostic = block.diagnostic;
                result.blocks.push_back(std::move(block));
            }
        };

        // libpng's eXIf accessor retains one value, while this archive must keep
        // duplicate original chunks and inspect both sides of IDAT without
        // inflating pixels. Only chunk framing and CRC are handled here; TIFF
        // interpretation remains in the existing media provider.
        SourceExif read_png_exif(std::ifstream& input, const MetadataLimits& limits, XmpCollector* xmp) {
            SourceExif result;
            result.scope = "png-chunks-through-iend";
            result.status = MetadataStatus::Absent;
            input.clear();
            input.seekg(0, std::ios::end);
            const auto end = input.tellg();
            if (end < 0) {
                result.status = MetadataStatus::IoError;
                result.diagnostic = "Cannot determine PNG source extent";
                return result;
            }
            const auto source_size = static_cast<uint64_t>(static_cast<std::streamoff>(end));
            input.seekg(kPngSignature.size());
            if (!input) {
                result.status = MetadataStatus::IoError;
                result.diagnostic = "Cannot seek PNG metadata source";
                return result;
            }
            size_t scanned = kPngSignature.size(), total = 0, chunks = 0;
            uint64_t position = kPngSignature.size();
            bool idat = false, exif = false, exif_stopped = false;
            auto fail = [&](MetadataStatus status, const char* text) {
                // After the first omitted EXIF block, only the combined scan advances.
                auto& state = exif_stopped ? *xmp->scan_status : result.status;
                auto& diagnostic = exif_stopped ? xmp->scan_diagnostic : result.diagnostic;
                if (state != MetadataStatus::Malformed || status == MetadataStatus::IoError)
                    state = status;
                if (diagnostic.find(text) == std::string::npos) {
                    if (!diagnostic.empty())
                        diagnostic += "; ";
                    diagnostic += text;
                }
            };
            auto read = [&](uint8_t* data, size_t size) {
                if (position > source_size || size > source_size - position) {
                    fail(MetadataStatus::Malformed, "Truncated PNG chunk");
                    return false;
                }
                if (size > limits.png_scan_bytes - scanned) {
                    fail(MetadataStatus::Limited, "PNG scan byte budget exhausted");
                    return false;
                }
                input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
                scanned += static_cast<size_t>(input.gcount());
                position += static_cast<size_t>(input.gcount());
                if (!input) {
                    fail(input.bad() ? MetadataStatus::IoError : MetadataStatus::Malformed,
                         input.bad() ? "Cannot read PNG metadata source" : "Truncated PNG chunk");
                    return false;
                }
                return true;
            };
            auto u32 = [](const uint8_t* p) {
                return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]);
            };
            std::array<uint8_t, 8192> buffer{};
            for (;;) {
                std::array<uint8_t, 8> header{};
                if (!read(header.data(), header.size()))
                    return result;
                const size_t length = u32(header.data());
                const auto* type = header.data() + 4;
                const bool first = position == kPngSignature.size() + header.size();
                const bool is_idat = !std::memcmp(type, "IDAT", 4);
                // Pixel segmentation is an encoder detail, not metadata work.
                // IDAT headers and CRC bytes still consume the scan budget.
                if (!is_idat) {
                    if (chunks == limits.png_chunks) {
                        fail(MetadataStatus::Limited, "PNG non-IDAT chunk count budget exhausted");
                        return result;
                    }
                    ++chunks;
                }
                if (length > 0x7fffffff ||
                    std::any_of(type, type + 4, [](uint8_t c) { return !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')); }) ||
                    (type[2] & 32) ||
                    (first && (std::memcmp(type, "IHDR", 4) || length != 13)) ||
                    (!first && !std::memcmp(type, "IHDR", 4))) {
                    fail(MetadataStatus::Malformed, "Invalid PNG chunk framing");
                    return result;
                }
                const bool is_exif = !std::memcmp(type, "eXIf", 4);
                const bool is_end = !std::memcmp(type, "IEND", 4);
                if (is_end && (length || !idat)) {
                    fail(MetadataStatus::Malformed, "Invalid PNG end chunk");
                    return result;
                }
                // A declared length beyond the opened source is truncation,
                // not evidence that the inspection budget was consumed.
                if (position > source_size || source_size - position < 4 || length > source_size - position - 4) {
                    fail(MetadataStatus::Malformed, "Truncated PNG chunk");
                    return result;
                }
                if (is_idat) {
                    // Pixel payloads are outside this metadata-only operation.
                    // Discard small spans through the stream buffer rather than
                    // invalidating it with a seek for every tiny chunk. Larger
                    // spans still seek; neither path decodes or checks pixel CRC.
                    if (length <= buffer.size()) {
                        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(length));
                        if (!input || static_cast<size_t>(input.gcount()) != length) {
                            fail(input.bad() ? MetadataStatus::IoError : MetadataStatus::Malformed,
                                 input.bad() ? "Cannot read PNG pixel payload" : "Truncated PNG pixel payload");
                            return result;
                        }
                    } else {
                        input.seekg(static_cast<std::streamoff>(length), std::ios::cur);
                        if (!input) {
                            fail(MetadataStatus::IoError, "Cannot skip PNG pixel payload");
                            return result;
                        }
                    }
                    position += length;
                    std::array<uint8_t, 4> checksum{};
                    if (!read(checksum.data(), checksum.size()))
                        return result;
                    idat = true;
                    continue;
                }
                if (is_exif && !exif_stopped && (result.blocks.size() >= limits.blocks || length > limits.block_bytes || length > limits.total_bytes - total)) {
                    fail(MetadataStatus::Limited, "EXIF collection budget exhausted");
                    if (!xmp)
                        return result;
                    xmp->scan_status = result.status;
                    xmp->scan_diagnostic = result.diagnostic;
                    exif_stopped = true;
                }
                // Charge data and CRC before allocating a retained block.
                if (limits.png_scan_bytes - scanned < 4 || length > limits.png_scan_bytes - scanned - 4) {
                    fail(MetadataStatus::Limited, "PNG scan byte budget exhausted");
                    return result;
                }
                const bool retain_exif = is_exif && !exif_stopped;
                ExifBlock block;
                block.envelope = ExifEnvelope::PngExif;
                if (retain_exif)
                    block.bytes.resize(length);
                std::vector<uint8_t> xmp_bytes;
                const bool possible_xmp = xmp && !std::memcmp(type, "iTXt", 4) && length >= sizeof(kPngXmp);
                uLong crc = crc32_z(0, type, 4);
                for (size_t offset = 0; offset < length;) {
                    const size_t count = std::min(length - offset, buffer.size());
                    auto* destination = retain_exif ? block.bytes.data() + offset : buffer.data();
                    if (!read(destination, count))
                        return result;
                    crc = crc32_z(crc, destination, count);
                    if (possible_xmp && offset == 0 && !std::memcmp(destination, kPngXmp, sizeof(kPngXmp)) && xmp->canRetain(length))
                        xmp_bytes.resize(length);
                    if (!xmp_bytes.empty())
                        std::memcpy(xmp_bytes.data() + offset, destination, count);
                    offset += count;
                }
                std::array<uint8_t, 4> checksum{};
                if (!read(checksum.data(), checksum.size()))
                    return result;
                if (!xmp_bytes.empty())
                    xmp->append(std::move(xmp_bytes), XmpEnvelope::PngInternationalText, crc == u32(checksum.data()));
                if (retain_exif) {
                    result.blocks.push_back(std::move(block));
                    total += length;
                    if (result.status == MetadataStatus::Absent)
                        result.status = MetadataStatus::Collected;
                    if (exif || idat)
                        fail(MetadataStatus::Malformed, "Duplicate or post-IDAT PNG eXIf chunk");
                    exif = true;
                }
                if (crc != u32(checksum.data())) {
                    fail(MetadataStatus::Malformed, "PNG chunk CRC mismatch");
                    // Ancillary CRC damage does not erase later intact EXIF.
                    // Trust no resynchronization: the next framing is checked
                    // normally, and damaged critical chunks remain terminal.
                    if (!(type[0] & 32))
                        return result;
                }
                if (is_end)
                    return result;
            }
        }

        // Metadata-only client I/O preserves the exact spans the existing TIFF
        // reader requests. It never rewrites offsets or synthesizes EXIF bytes.
        struct TiffMetadataReader {
            std::ifstream& input;
            const MetadataLimits& limits;
            SourceExif& result;
            uint64_t size = 0, position = 0;
            size_t reads = 0, read_bytes = 0, retained = 0, values = 0, entries = 0;
            bool stopped = false;
            const char* callback_failure = nullptr;
            void finish() {
                if (callback_failure)
                    result.diagnostic = callback_failure;
            }
            void fail(MetadataStatus status, const char* message) {
                if (stopped)
                    return;
                if ((result.status == MetadataStatus::Malformed && (status == MetadataStatus::Malformed || status == MetadataStatus::Unsupported)) ||
                    (result.status == MetadataStatus::Unsupported && status == MetadataStatus::Unsupported))
                    return;
                result.status = status;
                result.diagnostic = message;
                stopped = status == MetadataStatus::Limited || status == MetadataStatus::IoError;
            }
            static tmsize_t read(thandle_t handle, void* data, tmsize_t length) noexcept {
                auto& self = *static_cast<TiffMetadataReader*>(handle);
                try {
                    if (self.stopped || length < 0)
                        return -1;
                    const auto count = static_cast<size_t>(length);
                    if (self.position > self.size || count > self.size - self.position) {
                        self.fail(MetadataStatus::Malformed, "TIFF read outside source extent");
                        return -1;
                    }
                    bool already_retained = false;
                    for (const auto& span : self.result.tiff->source_ranges)
                        if (self.position >= span.offset && self.position - span.offset <= span.bytes.size() &&
                            count <= span.bytes.size() - (self.position - span.offset)) {
                            already_retained = true;
                            break;
                        }
                    if (self.reads >= self.limits.tiff_reads || count > self.limits.tiff_read_bytes - self.read_bytes ||
                        (!already_retained && (count > self.limits.block_bytes || count > self.limits.total_bytes - self.retained))) {
                        self.fail(MetadataStatus::Limited, "TIFF source read or retained-byte budget exhausted");
                        return -1;
                    }
                    ++self.reads;
                    self.read_bytes += count;
                    self.input.read(static_cast<char*>(data), static_cast<std::streamsize>(count));
                    if (!self.input || static_cast<size_t>(self.input.gcount()) != count) {
                        self.fail(self.input.bad() ? MetadataStatus::IoError : MetadataStatus::Malformed,
                                  "Cannot read complete TIFF source span");
                        return -1;
                    }
                    if (count && already_retained) {
                        for (const auto& span : self.result.tiff->source_ranges)
                            if (self.position >= span.offset && self.position - span.offset <= span.bytes.size() &&
                                count <= span.bytes.size() - (self.position - span.offset)) {
                                if (std::memcmp(data, span.bytes.data() + (self.position - span.offset), count)) {
                                    self.fail(MetadataStatus::Malformed, "TIFF source changed between metadata reads");
                                    self.stopped = true;
                                    return -1;
                                }
                                break;
                            }
                    }
                    if (count && !already_retained) {
                        const auto* bytes = static_cast<const uint8_t*>(data);
                        self.result.tiff->source_ranges.push_back({self.position, {bytes, bytes + count}});
                        self.retained += count;
                    }
                    self.position += count;
                    return length;
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): contain at the C ABI; an error
                    // status and literal diagnostic are returned outside it.
                    self.callback_failure = "Exception while retaining TIFF source bytes";
                    self.stopped = true;
                    self.result.status = MetadataStatus::Limited;
                    return -1;
                }
            }
            static toff_t seek(thandle_t handle, toff_t offset, int origin) noexcept {
                auto& self = *static_cast<TiffMetadataReader*>(handle);
                try {
                    uint64_t target = offset;
                    if (origin != SEEK_SET && origin != SEEK_CUR && origin != SEEK_END)
                        return static_cast<toff_t>(-1);
                    if (origin != SEEK_SET) {
                        // libtiff uses unsigned toff_t to carry negative relative offsets.
                        const auto delta = static_cast<int64_t>(offset);
                        const auto base = origin == SEEK_CUR ? self.position : self.size;
                        if ((delta < 0 && uint64_t(-(delta + 1)) + 1 > base) ||
                            (delta >= 0 && uint64_t(delta) > self.size - base))
                            return static_cast<toff_t>(-1);
                        target = delta < 0 ? base - (uint64_t(-(delta + 1)) + 1) : base + uint64_t(delta);
                    }
                    if (self.stopped || target > self.size || target > uint64_t(std::numeric_limits<std::streamoff>::max()))
                        return static_cast<toff_t>(-1);
                    self.input.clear();
                    self.input.seekg(static_cast<std::streamoff>(target));
                    if (!self.input) {
                        self.fail(MetadataStatus::IoError, "Cannot seek TIFF metadata source");
                        return static_cast<toff_t>(-1);
                    }
                    self.position = target;
                    return target;
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): return I/O failure and defer
                    // diagnostic storage until outside the nonthrowing C ABI.
                    self.callback_failure = "Exception while seeking TIFF metadata source";
                    self.stopped = true;
                    self.result.status = MetadataStatus::IoError;
                    return static_cast<toff_t>(-1);
                }
            }
            static int diagnostic(TIFF*, void* context, const char*, const char* format, va_list args) noexcept {
                auto& self = *static_cast<TiffMetadataReader*>(context);
                try {
                    std::array<char, 512> message{};
                    std::vsnprintf(message.data(), message.size(), format, args);
                    // Unknown tag registration warnings are handled separately.
                    // Other warnings can mean dropped/normalized source values.
                    const bool allocation_limit =
                        (std::strncmp(format, "Memory allocation of %", 22) == 0 && std::strstr(format, " byte limit defined in open options")) ||
                        (std::strncmp(format, "Cumulated memory allocation of %", 32) == 0 && std::strstr(format, " cumulated byte limit defined in open options"));
                    self.fail(allocation_limit ? MetadataStatus::Limited : std::strcmp(format, "%s compression support is not configured") == 0 ? MetadataStatus::Unsupported
                                                                                                                                                : MetadataStatus::Malformed,
                              message.data());
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): allocation failure is observable
                    // through Limited and a literal diagnostic stored outside C.
                    self.callback_failure = "Cannot retain TIFF library diagnostic";
                    self.stopped = true;
                    self.result.status = MetadataStatus::Limited;
                }
                return 1;
            }
            static int warning(TIFF* tiff, void* context, const char* module, const char* format, va_list args) noexcept {
                // These fields are auto-registered and collected below. Unlike
                // normalization/damage warnings this does not indicate loss.
                if (std::strncmp(format, "Unknown field with tag ", 23) == 0)
                    return 1;
                return diagnostic(tiff, context, module, format, args);
            }
            bool charge(size_t count, size_t width) {
                if (stopped)
                    return false;
                if (entries >= limits.tiff_entries || !width || count > (limits.tiff_value_bytes - values) / width) {
                    fail(MetadataStatus::Limited, "TIFF decoded-value budget exhausted");
                    return false;
                }
                values += count * width;
                ++entries;
                return true;
            }
        };

        template <typename T>
        void copy_tiff_values(TiffSourceTag& tag, const void* data, size_t count) {
            // memcpy avoids alignment/aliasing assumptions about library storage.
            for (size_t i = 0; i < count; ++i) {
                T value{};
                std::memcpy(&value, static_cast<const uint8_t*>(data) + i * sizeof(T), sizeof(T));
                if constexpr (std::is_floating_point_v<T>)
                    tag.floating_values.push_back(value);
                else if constexpr (std::is_signed_v<T>)
                    tag.signed_values.push_back(value);
                else
                    tag.unsigned_values.push_back(value);
            }
        }

        bool collect_tiff_custom(TIFF* tiff, uint32_t id, TiffMetadataReader& reader, TiffSourceDirectory& directory) {
            const auto* field = TIFFFieldWithTag(tiff, id);
            if (!field)
                return true;
            const auto type = TIFFFieldDataType(field);
            int width = TIFFFieldSetGetSize(field);
            const int declared = TIFFFieldReadCount(field);
            const int count_width = TIFFFieldSetGetCountSize(field);
            TiffSourceTag tag;
            tag.id = id;
            tag.type = static_cast<uint16_t>(type);
            uint32_t count = declared > 0 ? static_cast<uint32_t>(declared) : 0;
            void* data = nullptr;
            std::array<uint8_t, 8> scalar{};
            std::array<uint16_t, 2> dot_range{};
            auto scalar_get = [&]<typename T>() {
                T value{};
                const bool ok = TIFFGetField(tiff, id, &value) != 0;
                std::memcpy(scalar.data(), &value, sizeof(T));
                data = scalar.data();
                return ok;
            };
            bool got = false;
            if (id == TIFFTAG_DOTRANGE && std::strcmp(TIFFFieldName(field), "DotRange") == 0) {
                // This public getter is explicitly a two-scalar exception to
                // libtiff's ordinary custom-field varargs convention.
                got = TIFFGetField(tiff, id, &dot_range[0], &dot_range[1]) != 0;
                data = dot_range.data();
                count = 2;
                width = sizeof(dot_range[0]);
            } else if (TIFFFieldPassCount(field)) {
                if (count_width == 4)
                    got = TIFFGetField(tiff, id, &count, &data) != 0;
                else if (count_width == 2) {
                    uint16_t short_count = 0;
                    got = TIFFGetField(tiff, id, &short_count, &data) != 0;
                    count = short_count;
                }
            } else if (declared <= 0 && type != TIFF_ASCII) {
                reader.fail(MetadataStatus::Unsupported, "TIFF tag has no public decoded count");
                return true;
            } else if (type == TIFF_ASCII || declared != 1) {
                got = TIFFGetField(tiff, id, &data) != 0;
                if (got && data && type == TIFF_ASCII) {
                    // This is libtiff's NUL-terminated decoded string, not its
                    // original declared byte count. Even fixed-count ASCII fields can
                    // have shorter decoded strings after libtiff coercion.
                    // Source spans retain the original bytes.
                    const auto* text = static_cast<const char*>(data);
                    const size_t cap = reader.limits.tiff_value_bytes - reader.values;
                    size_t length = 0;
                    while (length < cap && text[length])
                        ++length;
                    if (length == cap) {
                        reader.fail(MetadataStatus::Limited, "TIFF decoded string budget exhausted");
                        return false;
                    }
                    count = static_cast<uint32_t>(length + 1);
                }
            } else {
                switch (type) {
                case TIFF_BYTE:
                case TIFF_UNDEFINED: got = scalar_get.template operator()<uint8_t>(); break;
                case TIFF_SBYTE: got = scalar_get.template operator()<int8_t>(); break;
                case TIFF_SHORT: got = scalar_get.template operator()<uint16_t>(); break;
                case TIFF_SSHORT: got = scalar_get.template operator()<int16_t>(); break;
                case TIFF_LONG:
                case TIFF_IFD: got = scalar_get.template operator()<uint32_t>(); break;
                case TIFF_SLONG: got = scalar_get.template operator()<int32_t>(); break;
                case TIFF_LONG8:
                case TIFF_IFD8: got = scalar_get.template operator()<uint64_t>(); break;
                case TIFF_SLONG8: got = scalar_get.template operator()<int64_t>(); break;
                case TIFF_RATIONAL:
                case TIFF_SRATIONAL:
                    got = width == 8 ? scalar_get.template operator()<double>() : scalar_get.template operator()<float>();
                    break;
                case TIFF_FLOAT: got = scalar_get.template operator()<float>(); break;
                case TIFF_DOUBLE: got = scalar_get.template operator()<double>(); break;
                default: break;
                }
            }
            if (!got || (!data && count)) {
                reader.fail(MetadataStatus::Unsupported, "TIFF tag getter representation is not supported");
                return true;
            }
            const int expected_width = type == TIFF_RATIONAL || type == TIFF_SRATIONAL || type == TIFF_IFD ? width : type == TIFF_ASCII || type == TIFF_BYTE || type == TIFF_UNDEFINED || type == TIFF_SBYTE ? 1
                                                                                                                 : type == TIFF_SHORT || type == TIFF_SSHORT                                                 ? 2
                                                                                                                 : type == TIFF_LONG || type == TIFF_SLONG || type == TIFF_FLOAT                             ? 4
                                                                                                                 : type == TIFF_LONG8 || type == TIFF_SLONG8 || type == TIFF_IFD8 || type == TIFF_DOUBLE     ? 8
                                                                                                                                                                                                             : 0;
            if (!expected_width || width != expected_width || ((type == TIFF_RATIONAL || type == TIFF_SRATIONAL) && width != 4 && width != 8)) {
                reader.fail(MetadataStatus::Unsupported, "Unsupported TIFF library storage width");
                return true;
            }
            if (!reader.charge(count, type == TIFF_ASCII || type == TIFF_BYTE || type == TIFF_UNDEFINED ? 1 : 8))
                return false;
            tag.count = count;
            tag.decoded_storage_bytes = static_cast<size_t>(width);
            switch (type) {
            case TIFF_ASCII:
            case TIFF_BYTE:
            case TIFF_UNDEFINED:
                if (count)
                    tag.bytes.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + count);
                break;
            case TIFF_SBYTE: copy_tiff_values<int8_t>(tag, data, count); break;
            case TIFF_SHORT: copy_tiff_values<uint16_t>(tag, data, count); break;
            case TIFF_SSHORT: copy_tiff_values<int16_t>(tag, data, count); break;
            case TIFF_LONG: copy_tiff_values<uint32_t>(tag, data, count); break;
            case TIFF_IFD:
                // libtiff widens counted IFD arrays to its 64-bit offset type,
                // while the scalar TIFF_IFD getter uses a uint32_t parameter.
                if (width == 8 && data != scalar.data())
                    copy_tiff_values<uint64_t>(tag, data, count);
                else
                    copy_tiff_values<uint32_t>(tag, data, count);
                break;
            case TIFF_SLONG: copy_tiff_values<int32_t>(tag, data, count); break;
            case TIFF_LONG8:
            case TIFF_IFD8: copy_tiff_values<uint64_t>(tag, data, count); break;
            case TIFF_SLONG8: copy_tiff_values<int64_t>(tag, data, count); break;
            case TIFF_RATIONAL:
            case TIFF_SRATIONAL:
                if (width == 8)
                    copy_tiff_values<double>(tag, data, count);
                else if (width == 4)
                    copy_tiff_values<float>(tag, data, count);
                break;
            case TIFF_FLOAT: copy_tiff_values<float>(tag, data, count); break;
            case TIFF_DOUBLE: copy_tiff_values<double>(tag, data, count); break;
            default: reader.fail(MetadataStatus::Unsupported, "Unsupported TIFF decoded value type"); break;
            }
            directory.tags.push_back(std::move(tag));
            return true;
        }

        SourceExif read_tiff_source_metadata(std::ifstream& input, const MetadataLimits& limits) {
            SourceExif result;
            result.scope = "tiff-libtiff-observed-metadata";
            result.status = MetadataStatus::Collected;
            result.tiff.emplace();
            TiffMetadataReader reader{input, limits, result};
            input.clear();
            input.seekg(0, std::ios::end);
            const auto end = input.tellg();
            if (end < 0) {
                result.status = MetadataStatus::IoError;
                result.diagnostic = "Cannot determine TIFF source extent";
                return result;
            }
            reader.size = static_cast<uint64_t>(end);
            input.seekg(0);
            std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
            if (!options) {
                result.status = MetadataStatus::Limited;
                result.diagnostic = "Cannot allocate TIFF open options";
                return result;
            }
            TIFFOpenOptionsSetMaxSingleMemAlloc(options.get(), 8 * 1024 * 1024);
            TIFFOpenOptionsSetMaxCumulatedMemAlloc(options.get(), 32 * 1024 * 1024);
            TIFFOpenOptionsSetErrorHandlerExtR(options.get(), TiffMetadataReader::diagnostic, &reader);
            TIFFOpenOptionsSetWarningHandlerExtR(options.get(), TiffMetadataReader::warning, &reader);
            std::unique_ptr<TIFF, decltype(&TIFFClose)> tiff(TIFFClientOpenExt("metadata source", "rhmOc", &reader, TiffMetadataReader::read, [](thandle_t, void*, tmsize_t) -> tmsize_t { return -1; }, TiffMetadataReader::seek, [](thandle_t) { return 0; }, [](thandle_t handle) -> toff_t { return static_cast<TiffMetadataReader*>(handle)->size; }, [](thandle_t, void**, toff_t*) { return 0; }, [](thandle_t, void*, toff_t) {}, options.get()), TIFFClose);
            if (!tiff) {
                if (result.status == MetadataStatus::Collected)
                    reader.fail(MetadataStatus::Malformed, "Cannot initialize TIFF metadata reader");
                reader.finish();
                return result;
            }
            result.tiff->big_tiff = TIFFIsBigTIFF(tiff.get());
            result.tiff->big_endian = TIFFIsBigEndian(tiff.get());
            struct Pending {
                uint64_t offset;
                std::optional<uint64_t> parent;
                std::string kind;
            };
            std::vector<Pending> pending;
            std::set<uint64_t> visited;
            bool root = true;
            while (!reader.stopped) {
                Pending node{0, {}, "image"};
                int ok = 0;
                if (root) {
                    // Header-only open leaves the first directory for this call.
                    ok = TIFFReadDirectory(tiff.get());
                    root = false;
                } else if (!pending.empty()) {
                    node = std::move(pending.back());
                    pending.pop_back();
                    if (!node.offset || visited.contains(node.offset)) {
                        bool cycle = false;
                        auto ancestor = node.parent;
                        while (ancestor) {
                            if (*ancestor == node.offset) {
                                cycle = true;
                                break;
                            }
                            const auto found = std::find_if(result.tiff->directories.begin(), result.tiff->directories.end(),
                                                            [&](const auto& value) { return value.offset == *ancestor; });
                            ancestor = found == result.tiff->directories.end() ? std::nullopt : found->parent;
                        }
                        reader.fail(cycle ? MetadataStatus::Malformed : MetadataStatus::Unsupported,
                                    cycle ? "Cyclic TIFF directory reference" : "Repeated TIFF directory offset is not revisited");
                        continue;
                    }
                    ok = node.kind == "exif" || node.kind == "interoperability" ? TIFFReadEXIFDirectory(tiff.get(), node.offset) : node.kind == "gps" ? TIFFReadGPSDirectory(tiff.get(), node.offset)
                                                                                                                                                      : TIFFSetSubDirectory(tiff.get(), node.offset);
                } else
                    break;
                if (!ok) {
                    if (result.status == MetadataStatus::Collected)
                        reader.fail(MetadataStatus::Malformed, "Cannot read TIFF metadata directory");
                    break;
                }
                node.offset = TIFFCurrentDirOffset(tiff.get());
                if (!visited.insert(node.offset).second) {
                    reader.fail(MetadataStatus::Unsupported, "Repeated TIFF directory offset is not revisited");
                    break;
                }
                TiffSourceDirectory directory{node.offset, node.parent, node.kind, {}};
                bool directory_limit = false;
                auto child = [&](uint64_t offset, const char* kind) {
                    if (offset)
                        pending.push_back({offset, node.offset, kind});
                };
                // Image chains are queued before children; all offsets are owned
                // before switching libtiff's single current-directory context.
                if (node.kind == "image" || node.kind == "subimage") {
                    // libtiff exposes the next main IFD by reading it. Defer that
                    // until after harvesting this directory (see below).
                    uint16_t count = 0;
                    uint64_t* offsets = nullptr;
                    if (TIFFGetField(tiff.get(), TIFFTAG_SUBIFD, &count, &offsets)) {
                        if (count > limits.tiff_directories - visited.size()) {
                            // Harvest this already-decoded directory before stopping traversal.
                            directory_limit = true;
                        } else
                            for (uint16_t i = 0; i < count; ++i)
                                child(offsets[i], "subimage");
                    }
                    for (const auto id : {TIFFTAG_EXIFIFD, TIFFTAG_GPSIFD}) {
                        uint64_t offset = 0;
                        if (TIFFGetField(tiff.get(), id, &offset))
                            child(offset, id == TIFFTAG_EXIFIFD ? "exif" : "gps");
                    }
                    // Baseline scalar tags are not in libtiff's custom tag list.
                    for (const auto id : {TIFFTAG_IMAGEWIDTH, TIFFTAG_IMAGELENGTH, TIFFTAG_ROWSPERSTRIP, TIFFTAG_SUBFILETYPE}) {
                        uint32_t value = 0;
                        if (TIFFGetField(tiff.get(), id, &value) && reader.charge(1, 8))
                            directory.tags.push_back({static_cast<uint32_t>(id), TIFF_LONG, 1, {value}, {}, {}, {}, sizeof(value)});
                    }
                    for (const auto id : {TIFFTAG_BITSPERSAMPLE, TIFFTAG_SAMPLESPERPIXEL, TIFFTAG_SAMPLEFORMAT,
                                          TIFFTAG_ORIENTATION, TIFFTAG_PHOTOMETRIC, TIFFTAG_COMPRESSION,
                                          TIFFTAG_PLANARCONFIG, TIFFTAG_RESOLUTIONUNIT, TIFFTAG_FILLORDER}) {
                        uint16_t value = 0;
                        if (TIFFGetField(tiff.get(), id, &value) && reader.charge(1, 8))
                            directory.tags.push_back({static_cast<uint32_t>(id), TIFF_SHORT, 1, {value}, {}, {}, {}, sizeof(value)});
                    }
                    uint16_t extra_count = 0;
                    uint16_t* extra_samples = nullptr;
                    if (TIFFGetField(tiff.get(), TIFFTAG_EXTRASAMPLES, &extra_count, &extra_samples) && reader.charge(extra_count, 8)) {
                        TiffSourceTag tag;
                        tag.id = TIFFTAG_EXTRASAMPLES;
                        tag.type = TIFF_SHORT;
                        tag.count = extra_count;
                        tag.decoded_storage_bytes = sizeof(uint16_t);
                        copy_tiff_values<uint16_t>(tag, extra_samples, extra_count);
                        directory.tags.push_back(std::move(tag));
                    }
                    for (const auto id : {TIFFTAG_XRESOLUTION, TIFFTAG_YRESOLUTION}) {
                        float value = 0;
                        if (TIFFGetField(tiff.get(), id, &value) && reader.charge(1, 8))
                            directory.tags.push_back({static_cast<uint32_t>(id), TIFF_RATIONAL, 1, {}, {}, {value}, {}, sizeof(value)});
                    }
                }
                const int tags = TIFFGetTagListCount(tiff.get());
                for (int i = 0; i < tags && !reader.stopped; ++i)
                    collect_tiff_custom(tiff.get(), TIFFGetTagListEntry(tiff.get(), i), reader, directory);
                if (node.kind == "exif") {
                    for (const auto& tag : directory.tags)
                        if (tag.id == 0xa005 && tag.unsigned_values.size() == 1)
                            child(tag.unsigned_values[0], "interoperability");
                }
                result.tiff->directories.push_back(std::move(directory));
                if (directory_limit || visited.size() + pending.size() > limits.tiff_directories) {
                    reader.fail(MetadataStatus::Limited, "TIFF directory budget exhausted");
                    break;
                }
                // Capture the next main directory offset without a second parser.
                if ((node.kind == "image" || node.kind == "subimage") && !TIFFLastDirectory(tiff.get())) {
                    if (visited.size() + pending.size() >= limits.tiff_directories) {
                        reader.fail(MetadataStatus::Limited, "TIFF directory budget exhausted");
                        break;
                    }
                    if (!TIFFReadDirectory(tiff.get())) {
                        if (result.status == MetadataStatus::Collected)
                            reader.fail(MetadataStatus::Malformed, "Cannot read next TIFF directory");
                        break;
                    }
                    pending.push_back({TIFFCurrentDirOffset(tiff.get()), node.parent, node.kind});
                }
            }
            tiff.reset();
            reader.finish();
            return result;
        }

    } // namespace

    // Header-only collection: payload extents and names are bounded before
    // allocation. Chunk tables and pixel data are never read or decoded.
    static void collect_exr_xmp(std::ifstream& input, const std::array<uint8_t, 12>& signature, size_t signature_size, XmpCollector& collector) {
        collector.supported_carrier = true;
        collector.result.status = MetadataStatus::Absent;
        collector.result.scope = "exr-single-header-string-xmp";
        const auto fail = [&](MetadataStatus status, const char* diagnostic) {
            collector.scan_status = status;
            collector.scan_diagnostic = diagnostic;
        };
        if (signature_size < 8) {
            fail(MetadataStatus::Malformed, "Truncated EXR version");
            return;
        }
        const auto u32 = [](const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; };
        const auto version = u32(signature.data() + 4);
        if ((version & 255) != 2 || (version & ~uint32_t(0x6ff))) {
            collector.result.status = MetadataStatus::Unsupported;
            collector.result.diagnostic = "EXR XMP collection requires a single-part non-deep version 2 header";
            return;
        }
        input.clear();
        input.seekg(0, std::ios::end);
        const auto end = input.tellg();
        if (end < 8) {
            fail(MetadataStatus::IoError, "Cannot determine EXR source extent");
            return;
        }
        const auto file_bytes = static_cast<uint64_t>(end);
        input.seekg(8, std::ios::beg);
        constexpr uint64_t header_limit = 64 * 1024 * 1024;
        constexpr size_t attribute_limit = 4096;
        const size_t name_limit = version & 0x400 ? 255 : 31;
        uint64_t position = 8;
        const auto extent = [&](size_t count) {
            if (count > header_limit - position) {
                fail(MetadataStatus::Limited, "EXR header scan budget exhausted");
                return false;
            }
            if (count > file_bytes - position) {
                fail(MetadataStatus::Malformed, "Truncated EXR header value");
                return false;
            }
            return true;
        };
        const auto read = [&](void* bytes, size_t count) {
            if (!extent(count))
                return false;
            if (!count)
                return true;
            input.read(static_cast<char*>(bytes), static_cast<std::streamsize>(count));
            if (!input) {
                fail(MetadataStatus::IoError, "Cannot read EXR header value");
                return false;
            }
            position += count;
            return true;
        };
        const auto name = [&](std::string& text) {
            text.clear();
            for (size_t n = 0; n <= name_limit; ++n) {
                char c = 0;
                if (!read(&c, 1))
                    return false;
                if (!c)
                    return true;
                text += c;
            }
            fail(MetadataStatus::Malformed, "EXR attribute name/type exceeds its declared limit");
            return false;
        };
        std::set<std::string> names;
        for (size_t index = 0;; ++index) {
            std::string key, type;
            if (!name(key) || key.empty())
                return;
            if (index >= attribute_limit) {
                fail(MetadataStatus::Limited, "EXR attribute count budget exhausted");
                return;
            }
            if (!name(type))
                return;
            std::array<uint8_t, 4> raw{};
            if (!read(raw.data(), raw.size()))
                return;
            const auto size = u32(raw.data());
            if (type.empty() || size > uint32_t(std::numeric_limits<int32_t>::max())) {
                fail(MetadataStatus::Malformed, "Invalid EXR attribute type or size");
                return;
            }
            if (!names.insert(key).second)
                fail(MetadataStatus::Malformed, "Duplicate EXR header attribute");
            if (!extent(size))
                return;
            if (key == "xmp" && type == "string" && collector.canRetain(size)) {
                std::vector<uint8_t> bytes(size);
                if (!read(bytes.data(), bytes.size()))
                    return;
                collector.append(std::move(bytes), XmpEnvelope::ExrString);
            } else {
                if (key == "xmp" && type != "string") {
                    if (collector.result.status != MetadataStatus::Limited) {
                        collector.result.status = MetadataStatus::Unsupported;
                        collector.result.diagnostic = "EXR xmp attribute is not a string";
                    }
                }
                input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
                if (!input) {
                    fail(MetadataStatus::IoError, "Cannot skip EXR header attribute");
                    return;
                }
                position += size;
            }
        }
    }

    static SourceExif collect_source_exif(const std::filesystem::path& path, const MetadataLimits& limits, XmpCollector* xmp) {
        SourceExif result;
        if (!limits.block_bytes || !limits.total_bytes) {
            result.status = MetadataStatus::Limited;
            result.diagnostic = "Invalid EXIF collection limits";
            return result;
        }
        std::error_code source_error;
        if (native_fs::is_directory(path, source_error)) {
            result.status = MetadataStatus::IoError;
            result.diagnostic = "Metadata source is a directory";
            return result;
        }
        std::ifstream input(native_fs::native_path(path), std::ios::binary);
        if (!input) {
            result.status = MetadataStatus::IoError;
            result.diagnostic = "Cannot open metadata source";
            return result;
        }
        std::array<uint8_t, 12> signature{};
        input.read(reinterpret_cast<char*>(signature.data()), signature.size());
        const auto signature_size = static_cast<size_t>(input.gcount());
        if (input.bad()) {
            result.status = MetadataStatus::IoError;
            result.diagnostic = "Cannot read metadata source";
            return result;
        }
        if (xmp && signature_size >= 4 && signature[0] == 0x76 && signature[1] == 0x2f && signature[2] == 0x31 && signature[3] == 0x01) {
            result.status = MetadataStatus::Unsupported;
            result.scope = "exr-header-xmp-only";
            result.diagnostic = "Standalone EXIF collection is not defined for this EXR observation";
            collect_exr_xmp(input, signature, signature_size, *xmp);
            return result;
        }
        const bool tiff_signature = signature_size >= 4 &&
                                    ((signature[0] == 'I' && signature[1] == 'I' && (signature[2] == 42 || signature[2] == 43) && signature[3] == 0) ||
                                     (signature[0] == 'M' && signature[1] == 'M' && signature[2] == 0 && (signature[3] == 42 || signature[3] == 43)));
        if (tiff_signature) {
            if (xmp) {
                xmp->supported_carrier = true;
                xmp->result.status = MetadataStatus::Absent;
                xmp->result.scope = "tiff-libtiff-observed-tag700";
            }
            if (!limits.tiff_reads || limits.tiff_reads > 65536 || !limits.tiff_read_bytes || limits.tiff_read_bytes > 1024 * 1024 * 1024 ||
                !limits.tiff_directories || limits.tiff_directories > 256 || !limits.tiff_entries || limits.tiff_entries > 65536 || !limits.tiff_value_bytes || limits.tiff_value_bytes > 64 * 1024 * 1024) {
                result.status = MetadataStatus::Limited;
                result.diagnostic = "Invalid TIFF collection limits";
                return result;
            }
            auto observed = read_tiff_source_metadata(input, limits);
            if (xmp)
                xmp->collectTiff(observed);
            return observed;
        }
        if (!limits.blocks || limits.blocks > 4096) {
            result.status = MetadataStatus::Limited;
            result.diagnostic = "Invalid EXIF block count limit";
            return result;
        }
        if (is_heif(std::span(signature).first(signature_size))) {
            if (!xmp)
                return read_heif_exif(path, limits);
            xmp->supported_carrier = true;
            auto observed = read_heif_metadata(path, limits, xmp->limits);
            xmp->result = std::move(observed.xmp);
            return std::move(observed.exif);
        }
        if (signature_size >= kPngSignature.size() && std::equal(kPngSignature.begin(), kPngSignature.end(), signature.begin())) {
            if (limits.png_scan_bytes < kPngSignature.size() || limits.png_scan_bytes > 1024 * 1024 * 1024 ||
                !limits.png_chunks || limits.png_chunks > 65536) {
                result.status = MetadataStatus::Limited;
                result.diagnostic = "Invalid PNG collection limits";
                return result;
            }
            if (xmp) {
                xmp->supported_carrier = true;
                xmp->result.status = MetadataStatus::Absent;
                xmp->result.scope = "png-itxt-through-iend";
            }
            return read_png_exif(input, limits, xmp);
        }
        if (is_jxl(std::span(signature).first(signature_size))) {
            auto observed = read_jxl_metadata(path, limits, xmp ? &xmp->limits : nullptr);
            if (xmp) {
                xmp->supported_carrier = true;
                // The shared JXL scan already reports enclosing failures to both
                // carriers. EXIF-specific quota exhaustion must not limit XML.
                xmp->scan_status = observed.xmp.status;
                xmp->scan_diagnostic = observed.xmp.diagnostic;
                xmp->result = std::move(observed.xmp);
            }
            return std::move(observed.exif);
        }
        if (signature_size < 2 || signature[0] != 0xff || signature[1] != 0xd8) {
            result.status = MetadataStatus::Unsupported;
            result.diagnostic = "EXIF collection supports JPEG, PNG, TIFF, HEIF and JPEG XL sources";
            return result;
        }
        result.scope = "jpeg-before-sos";
        if (xmp) {
            xmp->supported_carrier = true;
            xmp->result.status = MetadataStatus::Absent;
            xmp->result.scope = "jpeg-app1-before-sos";
        }
        // Validate the JPEG-only window after detecting the source format;
        // unrelated collection budgets must not reject PNG or HEIF metadata.
        if (!limits.jpeg_header_bytes || limits.jpeg_header_bytes > 4 * 1024 * 1024) {
            result.status = MetadataStatus::Limited;
            result.diagnostic = "Invalid JPEG collection limits";
            return result;
        }
        input.clear();
        input.seekg(0);
        std::vector<uint8_t> window(limits.jpeg_header_bytes);
        input.read(reinterpret_cast<char*>(window.data()), static_cast<std::streamsize>(window.size()));
        const auto bytes_read = static_cast<size_t>(input.gcount());
        if (input.bad()) {
            result.status = MetadataStatus::IoError;
            result.diagnostic = "Cannot read metadata source";
            return result;
        }
        const bool capped = bytes_read == window.size() && input.peek() != std::char_traits<char>::eof();
        window.resize(bytes_read);
        // Heap state remains valid after libjpeg's error longjmp. No C++ object
        // requiring destruction is created across a call that can longjmp.
        struct State {
            jpeg_decompress_struct codec{};
            JpegError error{};
            int header = JPEG_SUSPENDED;
            bool exhausted = false;
            bool failed = false;
            ~State() { jpeg_destroy_decompress(&codec); }
        };
        auto state = std::make_unique<State>();
        state->codec.err = jpeg_std_error(&state->error.base);
        state->error.base.error_exit = jpeg_error_exit;
        // Diagnostics are returned, never printed by a headless inspection.
        state->error.base.output_message = [](j_common_ptr common) {
            auto* error = reinterpret_cast<JpegError*>(common->err);
            (*common->err->format_message)(common, error->message);
        };
        if (setjmp(state->error.jump)) {
            state->failed = true;
        } else {
            jpeg_create_decompress(&state->codec);
            state->codec.client_data = state.get();
            jpeg_mem_src(&state->codec, window.data(), static_cast<unsigned long>(window.size()));
            state->codec.src->fill_input_buffer = [](j_decompress_ptr codec) -> boolean {
                auto* state = static_cast<State*>(codec->client_data);
                state->exhausted = true;
                // jpeg_mem_src normally fabricates an EOI at exhaustion, even
                // inside an unfinished APP1. Stop before invented bytes could
                // complete a saved marker. This also exits skip_input_data,
                // which cannot suspend. Completed marker_list entries survive.
                longjmp(state->error.jump, 1);
            };
            jpeg_save_markers(&state->codec, JPEG_APP0 + 1, 0xffff);
            state->header = jpeg_read_header(&state->codec, TRUE);
        }
        result.status = MetadataStatus::Absent;
        if (state->exhausted && capped) {
            result.status = MetadataStatus::Limited;
            result.diagnostic = "JPEG header window exhausted";
        } else if (state->failed || state->header != JPEG_HEADER_OK || state->error.base.num_warnings) {
            result.status = MetadataStatus::Malformed;
            result.diagnostic = state->exhausted ? "Truncated JPEG header" : state->error.message;
        }
        size_t total = 0;
        bool exif_stopped = false;
        for (auto* marker = state->codec.marker_list; marker; marker = marker->next) {
            if (xmp) {
                const bool standard = marker->data_length >= sizeof(kJpegXmp) && !std::memcmp(marker->data, kJpegXmp, sizeof(kJpegXmp));
                const bool extended = marker->data_length >= sizeof(kJpegExtendedXmp) && !std::memcmp(marker->data, kJpegExtendedXmp, sizeof(kJpegExtendedXmp));
                if ((standard || extended) && xmp->canRetain(marker->data_length))
                    xmp->append(std::vector<uint8_t>(marker->data, marker->data + marker->data_length),
                                standard ? XmpEnvelope::JpegApp1 : XmpEnvelope::JpegExtendedApp1);
            }
            if (exif_stopped || marker->data_length < 6 || std::memcmp(marker->data, "Exif\0\0", 6))
                continue;
            const size_t size = marker->data_length;
            if (result.blocks.size() >= limits.blocks || size > limits.block_bytes || size > limits.total_bytes - total) {
                if (result.status != MetadataStatus::Malformed)
                    result.status = MetadataStatus::Limited;
                if (!result.diagnostic.empty())
                    result.diagnostic += "; ";
                if (result.diagnostic.find("EXIF collection budget exhausted") == std::string::npos)
                    result.diagnostic += "EXIF collection budget exhausted";
                exif_stopped = true;
                if (xmp)
                    continue;
                break;
            }
            ExifBlock block;
            block.bytes.assign(marker->data, marker->data + size);
            result.blocks.push_back(std::move(block));
            total += size;
            if (result.status == MetadataStatus::Absent)
                result.status = MetadataStatus::Collected;
        }
        return result;
    }

    std::optional<JpegExtendedXmpSegment> source_extended_xmp_segment(const XmpBlock& block, const char** diagnostic) noexcept {
        const auto invalid = [&](const char* message) -> std::optional<JpegExtendedXmpSegment> {
            if (diagnostic)
                *diagnostic = message;
            return std::nullopt;
        };
        if (diagnostic)
            *diagnostic = nullptr;
        if (block.envelope != XmpEnvelope::JpegExtendedApp1)
            return invalid("Extended XMP segment has a different carrier");
        if (block.inflated || block.compressed || block.crc_valid || block.tiff_tag || block.heif_item || block.jxl_box)
            return invalid("Extended XMP segment has incompatible carrier provenance");
        constexpr size_t header = sizeof(kJpegExtendedXmp) + 40;
        if (block.bytes.size() < header)
            return invalid("Truncated extended XMP header");
        if (block.bytes.size() > 65533)
            return invalid("Extended XMP APP1 payload exceeds the JPEG marker limit");
        if (std::memcmp(block.bytes.data(), kJpegExtendedXmp, sizeof(kJpegExtendedXmp)))
            return invalid("Invalid extended XMP signature");
        const auto* guid = block.bytes.data() + sizeof(kJpegExtendedXmp);
        const auto u32 = [](const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]); };
        const auto extent = u32(guid + 32), offset = u32(guid + 36);
        if (!extent || offset > extent || block.bytes.size() - header > extent - offset)
            return invalid("Invalid extended XMP segment range");
        if (!std::all_of(guid, guid + 32, [](uint8_t c) { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F'); }))
            return invalid("Invalid extended XMP GUID");
        return JpegExtendedXmpSegment{std::string_view(reinterpret_cast<const char*>(guid), 32), extent, offset,
                                      std::span<const uint8_t>(block.bytes).subspan(header)};
    }

    std::optional<std::span<const uint8_t>> source_xmp_bytes(const SourceExif& exif, const XmpBlock& block) noexcept {
        if (block.envelope != XmpEnvelope::TiffTag) {
            if (block.tiff_tag || (block.envelope == XmpEnvelope::HeifMime) != block.heif_item.has_value() ||
                (block.envelope == XmpEnvelope::JxlXml) != block.jxl_box.has_value())
                return std::nullopt;
            if (block.heif_item && (!block.heif_item->image_item || !block.heif_item->metadata_item ||
                                    block.inflated || block.compressed || block.payload_offset || block.crc_valid))
                return std::nullopt;
            if ((block.jxl_box || block.envelope == XmpEnvelope::ExrString) &&
                (block.inflated || block.compressed || block.payload_offset || block.crc_valid))
                return std::nullopt;
            return std::span<const uint8_t>(block.bytes);
        }
        if (!block.tiff_tag || block.heif_item || block.jxl_box || !exif.tiff || !block.bytes.empty() || block.inflated || block.compressed || block.payload_offset || block.crc_valid)
            return std::nullopt;
        const auto reference = *block.tiff_tag;
        const auto& directories = exif.tiff->directories;
        if (reference.directory >= directories.size() || reference.tag >= directories[reference.directory].tags.size())
            return std::nullopt;
        const auto& tag = directories[reference.directory].tags[reference.tag];
        if (tag.id != TIFFTAG_XMLPACKET || (tag.type != TIFF_BYTE && tag.type != TIFF_UNDEFINED) || tag.count != tag.bytes.size() ||
            !tag.unsigned_values.empty() || !tag.signed_values.empty() || !tag.floating_values.empty())
            return std::nullopt;
        return std::span<const uint8_t>(tag.bytes);
    }

    SourceExif read_source_exif(const std::filesystem::path& path, const MetadataLimits& limits) {
        return collect_source_exif(path, limits, nullptr);
    }

    SourceImageMetadata read_source_metadata(const std::filesystem::path& path, const MetadataLimits& limits, const XmpLimits& xmp_limits) {
        SourceImageMetadata result;
        if (!xmp_limits.blocks || xmp_limits.blocks > 4096 || !xmp_limits.block_bytes || xmp_limits.block_bytes > 16 * 1024 * 1024 ||
            !xmp_limits.total_bytes || xmp_limits.total_bytes > 64 * 1024 * 1024 || xmp_limits.inflated_bytes > 16 * 1024 * 1024) {
            result.xmp.status = MetadataStatus::Limited;
            result.xmp.diagnostic = "Invalid XMP collection limits";
            result.exif = read_source_exif(path, limits);
            return result;
        }
        XmpCollector collector{result.xmp, xmp_limits};
        result.exif = collect_source_exif(path, limits, &collector);
        const auto scan_status = collector.scan_status.value_or(result.exif.status);
        const auto& scan_diagnostic = collector.scan_status ? collector.scan_diagnostic : result.exif.diagnostic;
        if (result.xmp.status == MetadataStatus::NotExamined) {
            result.xmp.status = (scan_status == MetadataStatus::IoError || scan_status == MetadataStatus::Limited) ? scan_status : MetadataStatus::Unsupported;
            result.xmp.diagnostic = (scan_status == MetadataStatus::IoError || scan_status == MetadataStatus::Limited) ? scan_diagnostic : "XMP collection supports JPEG APP1, PNG iTXt, observed TIFF tag 700, HEIF primary-image MIME items, JPEG XL XML boxes and single-header EXR xmp strings";
        } else if ((collector.supported_carrier || result.xmp.status != MetadataStatus::Unsupported) &&
                   (scan_status == MetadataStatus::Limited || scan_status == MetadataStatus::Malformed || scan_status == MetadataStatus::IoError)) {
            // Conservatively retain the enclosing scan's incomplete/invalid status.
            // Individual blocks report carrier framing, never XML validity.
            if (result.xmp.status != MetadataStatus::Limited || scan_status == MetadataStatus::IoError)
                result.xmp.status = scan_status;
            if (!scan_diagnostic.empty() && result.xmp.diagnostic.find(scan_diagnostic) == std::string::npos) {
                if (!result.xmp.diagnostic.empty())
                    result.xmp.diagnostic += "; ";
                result.xmp.diagnostic += scan_diagnostic;
            }
        }
        return result;
    }

    static bool write_jpeg_stream(const std::filesystem::path& path, const std::uint8_t* data,
                                  const int width, const int height, const int channels, const int quality,
                                  const std::optional<std::string>& comment, std::string& error, const bool full_chroma, FILE* borrowed) {
        if (!data || width <= 0 || height <= 0 || width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION ||
            (channels != 1 && channels != 3)) {
            error = "Unsupported JPEG layout";
            return false;
        }
        auto state = std::make_unique<JpegEncodeState>();
        state->owns_file = !borrowed;
        state->file = borrowed ? borrowed : open_output_file(path);
        if (!state->file) {
            error = "Could not open JPEG output " + path_to_utf8(path);
            return false;
        }
        if (!encode_jpeg(state.get(), data, width, height, channels, quality,
                         comment ? comment->data() : nullptr, comment ? comment->size() : 0, full_chroma,
                         static_cast<std::size_t>(width) * channels)) {
            const auto message = "JPEG encode failed for " + path_to_utf8(path);
            set_error(error, message.c_str(), state->error.message);
            return false;
        }
        const bool success = borrowed ? !std::ferror(borrowed) : std::fclose(state->file) == 0;
        state->file = nullptr;
        if (!success)
            error = "Could not write JPEG output " + path_to_utf8(path);
        return success;
    }

    bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                    const int width, const int height, const int channels, const int quality,
                    const std::optional<std::string>& comment, std::string& error, const bool full_chroma) {
        return write_jpeg_stream(path, data, width, height, channels, quality, comment, error, full_chroma, nullptr);
    }

    namespace {
        // Large 8-bit images: filter and deflate horizontal stripes on all cores. Each stripe ends
        // on a byte-aligned sync flush, so the stripes concatenate into one zlib stream (as pigz does).
        constexpr std::size_t kParallelPngMinPixels = std::size_t{1} << 20;

        void append_u32(std::vector<std::uint8_t>& out, const std::uint32_t value) {
            out.push_back(static_cast<std::uint8_t>(value >> 24));
            out.push_back(static_cast<std::uint8_t>(value >> 16));
            out.push_back(static_cast<std::uint8_t>(value >> 8));
            out.push_back(static_cast<std::uint8_t>(value));
        }

        // Keyword, separator, compression flag/method and empty language/translation.
        constexpr char png_xmp_prefix[] = "XML:com.adobe.xmp\0\0\0\0";
        bool valid_png_xmp(std::span<const std::string_view> parts) {
            std::size_t remaining = PNG_UINT_31_MAX - sizeof(png_xmp_prefix);
            for (const auto part : parts) {
                if (part.size() > remaining || part.find('\0') != std::string_view::npos)
                    return false;
                remaining -= part.size();
            }
            return true;
        }
        // Only used after validation, or with the empty default packet.
        std::size_t png_xmp_bytes(std::span<const std::string_view> parts) {
            std::size_t size = 0;
            for (const auto part : parts)
                size += part.size();
            return size;
        }

        struct PngOutput {
            std::FILE* file = nullptr; // Borrowed; ownership stays outside C callbacks.
            EncodedMemory* memory = nullptr;
            bool write(const std::uint8_t* data, std::size_t size) noexcept {
                return memory ? memory->append(data, size)
                              : !size || std::fwrite(data, 1, size, file) == size;
            }
        };

        struct PngFileCloser {
            void operator()(FILE* file) const noexcept { std::fclose(file); }
        };

        bool write_png_chunk(PngOutput& output, const char* type, const std::uint8_t* data, const std::size_t size,
                             std::span<const std::uint8_t> tail = {}, std::span<const std::string_view> fragments = {}) {
            if (size > PNG_UINT_31_MAX || tail.size() > PNG_UINT_31_MAX - size)
                return false;
            auto length = static_cast<std::uint32_t>(size + tail.size());
            for (const auto fragment : fragments) {
                if (fragment.size() > PNG_UINT_31_MAX - length)
                    return false;
                length += static_cast<std::uint32_t>(fragment.size());
            }
            const std::array<std::uint8_t, 8> header{
                std::uint8_t(length >> 24), std::uint8_t(length >> 16), std::uint8_t(length >> 8), std::uint8_t(length),
                std::uint8_t(type[0]), std::uint8_t(type[1]), std::uint8_t(type[2]), std::uint8_t(type[3])};
            uLong crc = crc32(0L, reinterpret_cast<const Bytef*>(type), 4);
            if (size)
                crc = crc32_z(crc, data, size);
            if (!tail.empty())
                crc = crc32_z(crc, tail.data(), tail.size());
            for (const auto fragment : fragments)
                if (!fragment.empty())
                    crc = crc32_z(crc, reinterpret_cast<const Bytef*>(fragment.data()), fragment.size());
            const std::array<std::uint8_t, 4> trailer{
                std::uint8_t(crc >> 24), std::uint8_t(crc >> 16), std::uint8_t(crc >> 8), std::uint8_t(crc)};
            if (!output.write(header.data(), header.size()) || !output.write(data, size) || !output.write(tail.data(), tail.size()))
                return false;
            for (const auto fragment : fragments)
                if (!fragment.empty() && !output.write(reinterpret_cast<const std::uint8_t*>(fragment.data()), fragment.size()))
                    return false;
            return output.write(trailer.data(), trailer.size());
        }

        std::uint8_t paeth_predictor(const int a, const int b, const int c) {
            const int p = a + b - c;
            const int pa = std::abs(p - a);
            const int pb = std::abs(p - b);
            const int pc = std::abs(p - c);
            if (pa <= pb && pa <= pc)
                return static_cast<std::uint8_t>(a);
            return static_cast<std::uint8_t>(pb <= pc ? b : c);
        }

        // libpng's default choice: the filter whose residuals have the smallest absolute sum.
        void filter_png_row(const std::uint8_t* row, const std::uint8_t* prev, const std::size_t row_bytes,
                            const std::size_t bpp, std::uint8_t* out,
                            std::array<std::vector<std::uint8_t>, 5>& candidates) {
            std::array<std::uint64_t, 5> sums{};
            for (std::size_t i = 0; i < row_bytes; ++i) {
                const int x = row[i];
                const int a = i >= bpp ? row[i - bpp] : 0;
                const int b = prev ? prev[i] : 0;
                const int c = prev && i >= bpp ? prev[i - bpp] : 0;
                const std::array<std::uint8_t, 5> residual = {
                    static_cast<std::uint8_t>(x),
                    static_cast<std::uint8_t>(x - a),
                    static_cast<std::uint8_t>(x - b),
                    static_cast<std::uint8_t>(x - ((a + b) >> 1)),
                    static_cast<std::uint8_t>(x - paeth_predictor(a, b, c)),
                };
                for (std::size_t f = 0; f < residual.size(); ++f) {
                    candidates[f][i] = residual[f];
                    sums[f] += static_cast<std::uint64_t>(std::abs(static_cast<int>(static_cast<std::int8_t>(residual[f]))));
                }
            }
            const auto best = static_cast<std::size_t>(std::min_element(sums.begin(), sums.end()) - sums.begin());
            out[0] = static_cast<std::uint8_t>(best);
            std::memcpy(out + 1, candidates[best].data(), row_bytes);
        }

        struct PngStripe {
            std::vector<std::uint8_t> deflated;
            uLong adler = 1;
            std::size_t filtered_bytes = 0;
            bool ok = false;
            std::exception_ptr exception;
        };

        void encode_png_stripe(const std::uint8_t* pixels, const int width, const int channels, const int y_begin,
                               const int y_end, const bool last, const int level, const std::size_t row_stride, PngStripe& stripe) {
            const auto row_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
            const auto rows = static_cast<std::size_t>(y_end - y_begin);
            if (row_bytes == std::numeric_limits<std::size_t>::max() ||
                rows > std::numeric_limits<uInt>::max() / (row_bytes + 1))
                return;
            std::vector<std::uint8_t> filtered(rows * (row_bytes + 1));
            std::array<std::vector<std::uint8_t>, 5> candidates;
            for (auto& candidate : candidates)
                candidate.resize(row_bytes);
            for (int y = y_begin; y < y_end; ++y) {
                const auto* row = pixels + static_cast<std::size_t>(y) * row_stride;
                const auto* prev = y > 0 ? row - row_stride : nullptr;
                filter_png_row(row, prev, row_bytes, static_cast<std::size_t>(channels),
                               filtered.data() + static_cast<std::size_t>(y - y_begin) * (row_bytes + 1), candidates);
            }

            z_stream stream{};
            const int initialized = deflateInit2(&stream, level, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY);
            if (initialized == Z_MEM_ERROR)
                throw std::bad_alloc();
            if (initialized != Z_OK)
                return;
            struct DeflateGuard {
                z_stream* stream;
                ~DeflateGuard() { deflateEnd(stream); }
            } guard{&stream};
            const auto bound = deflateBound(&stream, static_cast<uLong>(filtered.size()));
            if (bound > std::numeric_limits<uInt>::max() - 64)
                return;
            stripe.deflated.resize(bound + 64);
            stream.next_in = filtered.data();
            stream.avail_in = static_cast<uInt>(filtered.size());
            stream.next_out = stripe.deflated.data();
            stream.avail_out = static_cast<uInt>(stripe.deflated.size());
            const int status = deflate(&stream, last ? Z_FINISH : Z_SYNC_FLUSH);
            if (status == Z_MEM_ERROR)
                throw std::bad_alloc();
            const bool done = last ? status == Z_STREAM_END : (status == Z_OK && stream.avail_in == 0);
            stripe.deflated.resize(stream.total_out);
            stripe.adler = adler32_z(1L, filtered.data(), filtered.size());
            stripe.filtered_bytes = filtered.size();
            stripe.ok = done;
        }

        bool write_png_parallel(const std::filesystem::path& path, const std::uint8_t* pixels, const int width,
                                const int height, const int channels, const int compression_level,
                                const std::optional<std::string>& comment, std::string& error, FILE* borrowed,
                                std::size_t row_stride, EncodedMemory* memory = nullptr, std::span<const std::string_view> xmp = {}) {
            const auto row_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(channels);
            const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
            // About 8 MiB of pixels per stripe keeps every core busy without hurting the ratio.
            const int stripe_rows = std::max(16, static_cast<int>((std::size_t{8} << 20) / std::max<std::size_t>(row_bytes, 1)));
            const int stripe_count = 1 + (height - 1) / stripe_rows;
            const int level = std::clamp(compression_level, 0, 9);

            std::vector<PngStripe> stripes(static_cast<std::size_t>(stripe_count));
            std::atomic<int> next{0};
            const auto worker = [&] {
                for (int s = next.fetch_add(1); s < stripe_count; s = next.fetch_add(1)) {
                    const int y_begin = s * stripe_rows;
                    const int y_end = y_begin + std::min(height - y_begin, stripe_rows);
                    auto& stripe = stripes[static_cast<std::size_t>(s)];
                    try {
                        encode_png_stripe(pixels, width, channels, y_begin, y_end, s == stripe_count - 1, level,
                                          row_stride, stripe);
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): forward the worker exception
                        // after every worker has joined, never terminate a thread.
                        stripe.exception = std::current_exception();
                    }
                }
            };
            std::vector<std::jthread> pool;
            const unsigned pool_size = std::min<unsigned>(threads, static_cast<unsigned>(stripe_count));
            pool.reserve(pool_size);
            for (unsigned t = 1; t < pool_size; ++t)
                pool.emplace_back(worker);
            worker();
            for (auto& thread : pool)
                thread.join();
            for (const auto& stripe : stripes)
                if (stripe.exception)
                    std::rethrow_exception(stripe.exception);
            if (std::any_of(stripes.begin(), stripes.end(), [](const PngStripe& stripe) { return !stripe.ok; })) {
                error = "PNG deflate failed";
                return false;
            }

            uLong adler = stripes.front().adler;
            for (std::size_t s = 1; s < stripes.size(); ++s)
                adler = adler32_combine(adler, stripes[s].adler, static_cast<z_off_t>(stripes[s].filtered_bytes));

            std::FILE* file = memory ? nullptr : borrowed ? borrowed
                                                          : open_output_file(path);
            std::unique_ptr<FILE, PngFileCloser> owned(borrowed || memory ? nullptr : file);
            if (!memory && !file) {
                error = "Could not open PNG output " + path_to_utf8(path);
                return false;
            }
            static constexpr std::array<std::uint8_t, 8> signature = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
            std::vector<std::uint8_t> ihdr;
            append_u32(ihdr, static_cast<std::uint32_t>(width));
            append_u32(ihdr, static_cast<std::uint32_t>(height));
            const std::uint8_t color_type = channels == 1 ? 0 : channels == 2 ? 4
                                                            : channels == 3   ? 2
                                                                              : 6;
            ihdr.insert(ihdr.end(), {8, color_type, 0, 0, 0});
            PngOutput output{file, memory};
            bool ok = output.write(signature.data(), signature.size()) &&
                      write_png_chunk(output, "IHDR", ihdr.data(), ihdr.size());
            if (ok && comment && !comment->empty()) {
                constexpr std::array<std::uint8_t, 8> key{'C', 'o', 'm', 'm', 'e', 'n', 't', 0};
                ok = write_png_chunk(output, "tEXt", key.data(), key.size(),
                                     {reinterpret_cast<const std::uint8_t*>(comment->data()), comment->size()});
            }
            if (ok && png_xmp_bytes(xmp))
                ok = write_png_chunk(output, "iTXt", reinterpret_cast<const std::uint8_t*>(png_xmp_prefix), sizeof(png_xmp_prefix), {}, xmp);
            const std::uint8_t level_bits = level < 2 ? 0 : level < 6 ? 1
                                                        : level == 6  ? 2
                                                                      : 3;
            std::uint8_t zlib_header[2] = {0x78, static_cast<std::uint8_t>(level_bits << 6)};
            zlib_header[1] = static_cast<std::uint8_t>(zlib_header[1] + (31 - (zlib_header[0] * 256 + zlib_header[1]) % 31));
            ok = ok && write_png_chunk(output, "IDAT", zlib_header, sizeof(zlib_header));
            for (const auto& stripe : stripes) {
                if (!ok)
                    break;
                ok = write_png_chunk(output, "IDAT", stripe.deflated.data(), stripe.deflated.size());
            }
            std::vector<std::uint8_t> checksum;
            append_u32(checksum, static_cast<std::uint32_t>(adler));
            ok = ok && write_png_chunk(output, "IDAT", checksum.data(), checksum.size()) &&
                 write_png_chunk(output, "IEND", nullptr, 0);
            const bool closed = memory || (borrowed ? !std::ferror(borrowed) : std::fclose(owned.release()) == 0);
            if (!ok || !closed)
                error = "Could not write PNG output " + path_to_utf8(path);
            return ok && closed;
        }
    } // namespace

    namespace {
        struct PngEncodeState {
            png_structp png = nullptr;
            png_infop info = nullptr;
            PngOutput output;
            std::vector<png_bytep> rows;
            bool allocation_failed = false;
            std::array<char, 256> message{};

            static png_voidp allocate(png_structp png, png_alloc_size_t size) noexcept {
                auto& self = *static_cast<PngEncodeState*>(png_get_mem_ptr(png));
                void* result = std::malloc(size);
                if (!result)
                    self.allocation_failed = true;
                return result;
            }
            static void release(png_structp, png_voidp pointer) noexcept { std::free(pointer); }
            static void error(png_structp png, png_const_charp message) noexcept {
                auto& self = *static_cast<PngEncodeState*>(png_get_error_ptr(png));
                std::strncpy(self.message.data(), message, self.message.size() - 1);
                png_longjmp(png, 1);
            }
            static void write(png_structp png, png_bytep data, png_size_t size) noexcept {
                auto& self = *static_cast<PngEncodeState*>(png_get_io_ptr(png));
                // append returns before png_error: no C++ owner or exception
                // crosses the longjmp boundary.
                if (!self.output.write(data, size))
                    png_error(png, "PNG destination write failed");
            }
            static void flush(png_structp) noexcept {}
            bool initialize() {
                png = png_create_write_struct_2(PNG_LIBPNG_VER_STRING, this, error, nullptr,
                                                this, allocate, release);
                info = png ? png_create_info_struct(png) : nullptr;
                if (!png || !info) {
                    allocation_failed = true;
                    return false;
                }
                return true;
            }
            ~PngEncodeState() { png_destroy_write_struct(&png, &info); }
        };

        // Codec jumps stay in this scalar-only frame. All C++ owners and mutable
        // state live in the caller, including the row-pointer array.
        bool encode_png_serial(PngEncodeState* state, const std::uint8_t* data,
                               int width, int height, int channels, int bit_depth,
                               std::size_t row_stride, int compression_level,
                               const std::optional<std::string>& comment, std::span<const std::string_view> xmp = {}) {
            if (!state->png && !state->initialize())
                return false;
            if (setjmp(png_jmpbuf(state->png)))
                return false;
            png_set_write_fn(state->png, state, PngEncodeState::write, PngEncodeState::flush);
            const int color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 2 ? PNG_COLOR_TYPE_GRAY_ALPHA
                                                                     : channels == 3   ? PNG_COLOR_TYPE_RGB
                                                                                       : PNG_COLOR_TYPE_RGBA;
            png_set_IHDR(state->png, state->info, width, height, bit_depth, color_type, PNG_INTERLACE_NONE,
                         PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
            png_set_compression_level(state->png, std::clamp(compression_level, 0, 9));
            png_set_compression_strategy(state->png, Z_DEFAULT_STRATEGY);
            png_text text{};
            if (comment && !comment->empty()) {
                text.compression = PNG_TEXT_COMPRESSION_NONE;
                text.key = const_cast<png_charp>("Comment");
                text.text = const_cast<png_charp>(comment->c_str());
                png_set_text(state->png, state->info, &text, 1);
            }
            png_write_info(state->png, state->info);
            const auto xmp_size = png_xmp_bytes(xmp);
            if (xmp_size) {
                png_write_chunk_start(state->png, reinterpret_cast<png_const_bytep>("iTXt"), static_cast<png_uint_32>(sizeof(png_xmp_prefix) + xmp_size));
                png_write_chunk_data(state->png, reinterpret_cast<png_const_bytep>(png_xmp_prefix), sizeof(png_xmp_prefix));
                for (const auto part : xmp)
                    if (!part.empty())
                        png_write_chunk_data(state->png, reinterpret_cast<png_const_bytep>(part.data()), part.size());
                png_write_chunk_end(state->png);
            }
            if constexpr (std::endian::native == std::endian::little) {
                if (bit_depth == 16)
                    png_set_swap(state->png);
            }
            state->rows.resize(height);
            for (int y = 0; y < height; ++y)
                state->rows[y] = const_cast<png_bytep>(data + static_cast<std::size_t>(y) * row_stride);
            png_write_image(state->png, state->rows.data());
            png_write_end(state->png, state->info);
            return true;
        }
    } // namespace

    static bool write_png_stream(const std::filesystem::path& path, const void* data,
                                 const int width, const int height, const int channels, const int bit_depth,
                                 const int compression_level, const std::optional<std::string>& comment,
                                 std::string& error, FILE* borrowed, std::span<const std::string_view> xmp = {}) {
        if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4 || (bit_depth != 8 && bit_depth != 16) ||
            static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / channels / (bit_depth / 8)) {
            error = "Unsupported PNG layout";
            return false;
        }
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bit_depth / 8);
        if (bit_depth == 8 && static_cast<std::size_t>(width) * height >= kParallelPngMinPixels)
            return write_png_parallel(path, static_cast<const std::uint8_t*>(data), width, height, channels,
                                      compression_level, comment, error, borrowed, row_bytes, nullptr, xmp);
        // Preserve the file writer's preflight: failure to create the codec
        // must not open/truncate the destination.
        PngEncodeState state;
        if (!state.initialize()) {
            error = "Could not allocate PNG encoder";
            return false;
        }
        std::FILE* file = borrowed ? borrowed : open_output_file(path);
        std::unique_ptr<FILE, PngFileCloser> owned(borrowed ? nullptr : file);
        if (!file) {
            error = "Could not open PNG output " + path_to_utf8(path);
            return false;
        }
        state.output.file = file;
        const bool encoded = encode_png_serial(&state, static_cast<const std::uint8_t*>(data), width, height,
                                               channels, bit_depth, row_bytes, compression_level, comment, xmp);
        const bool closed = borrowed ? !std::ferror(borrowed) : std::fclose(owned.release()) == 0;
        if (!encoded)
            error = "PNG encode failed";
        else if (!closed)
            error = "Could not write PNG output " + path_to_utf8(path);
        return encoded && closed;
    }

    Status encode_png_memory(std::span<const std::uint8_t> data, int width, int height, int channels,
                             int bit_depth, std::size_t row_stride, std::vector<std::uint8_t>& output,
                             const PngEncodeOptions& options) {
        const auto failure = [](ErrorCode code, std::string message) -> Status {
            return Status::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        const auto invalid = [&](std::string_view reason) -> Status {
            return failure(ErrorCode::InvalidArgument,
                           std::format("{} (width={}, height={}, channels={}, depth={}, stride={}, span_bytes={}, limit={})",
                                       reason, width, height, channels, bit_depth, row_stride, data.size(), options.max_output_bytes));
        };
        if (width <= 0 || height <= 0 || channels < 1 || channels > 4 || (bit_depth != 8 && bit_depth != 16) ||
            static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / channels / (bit_depth / 8))
            return invalid("Unsupported PNG layout");
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bit_depth / 8);
        if (row_stride < row_bytes || static_cast<std::size_t>(height - 1) > (std::numeric_limits<std::size_t>::max() - row_bytes) / row_stride ||
            data.size() < row_stride * (height - 1) + row_bytes)
            return invalid("Unsupported PNG buffer size or stride");
        if (!options.max_output_bytes || options.max_output_bytes > output.max_size())
            return invalid("Invalid PNG encoded-byte limit");
        if (options.comment && (options.comment->find('\0') != std::string::npos || options.comment->size() > PNG_UINT_31_MAX - 8))
            return invalid("Invalid PNG Comment text");
        try {
            EncodedMemory memory;
            memory.limit = options.max_output_bytes;
            bool encoded;
            bool allocation_failed = false;
            std::string error;
            if (bit_depth == 8 && static_cast<std::size_t>(width) * height >= kParallelPngMinPixels) {
                encoded = write_png_parallel({}, data.data(), width, height, channels, options.compression_level,
                                             options.comment, error, nullptr, row_stride, &memory);
            } else {
                PngEncodeState state;
                state.output.memory = &memory;
                encoded = encode_png_serial(&state, data.data(), width, height, channels, bit_depth, row_stride,
                                            options.compression_level, options.comment);
                allocation_failed = state.allocation_failed;
                if (allocation_failed)
                    encoded = false;
                if (!encoded)
                    error = state.message.data();
            }
            if (!encoded) {
                if (allocation_failed || memory.failure != EncodedMemory::Failure::None)
                    return failure(ErrorCode::ResourceExhausted, memory.failure == EncodedMemory::Failure::Limit
                                                                     ? "PNG encoded-byte limit exhausted"
                                                                     : "Cannot allocate PNG encoder or output");
                return failure(ErrorCode::Unavailable, "PNG encode failed: " + error);
            }
            output = std::move(memory.bytes);
            return {};
        } catch (const std::bad_alloc&) {
            return failure(ErrorCode::ResourceExhausted, "Cannot allocate PNG encoder or output");
        } catch (const std::system_error& error) {
            return failure(ErrorCode::Unavailable, std::string("PNG worker creation failed: ") + error.what());
        }
    }

    bool write_png(const std::filesystem::path& path, const void* data,
                   const int width, const int height, const int channels, const int bit_depth,
                   const int compression_level, const std::optional<std::string>& comment, std::string& error) {
        return write_png_stream(path, data, width, height, channels, bit_depth, compression_level, comment, error, nullptr);
    }

    namespace {
        toff_t tiff_seek(thandle_t handle, toff_t offset, int origin) {
            auto* file = static_cast<FILE*>(handle);
#ifdef _WIN32
            if (::_fseeki64(file, static_cast<int64_t>(offset), origin) != 0)
                return toff_t(-1);
            const auto position = ::_ftelli64(file);
#else
            if (::fseeko(file, static_cast<off_t>(offset), origin) != 0)
                return toff_t(-1);
            const auto position = ::ftello(file);
#endif
            return position < 0 ? toff_t(-1) : toff_t(position);
        }
        toff_t tiff_size(thandle_t handle) {
            const auto current = tiff_seek(handle, 0, SEEK_CUR);
            const auto end = tiff_seek(handle, 0, SEEK_END);
            return current == toff_t(-1) || tiff_seek(handle, current, SEEK_SET) == toff_t(-1) ? 0 : end;
        }
        TIFF* tiff_stream(const std::filesystem::path& path, FILE* file) {
            install_tiff_quiet_handlers();
            return TIFFClientOpen(path_to_utf8(path).c_str(), "w", file, [](thandle_t h, void* p, tmsize_t n) -> tmsize_t { return n < 0 ? -1 : tmsize_t(std::fread(p, 1, size_t(n), static_cast<FILE*>(h))); }, [](thandle_t h, void* p, tmsize_t n) -> tmsize_t { return n < 0 ? -1 : tmsize_t(std::fwrite(p, 1, size_t(n), static_cast<FILE*>(h))); }, tiff_seek, [](thandle_t) { return 0; }, tiff_size, [](thandle_t, void**, toff_t*) { return 0; }, [](thandle_t, void*, toff_t) {});
        }
    } // namespace

    static void close_tiff_writer(TIFF* tiff) noexcept {
        // The encoder explicitly writes and flushes its directory. Cleanup must
        // not retry a partially failed directory or allocate during unwinding.
        TIFFSetMode(tiff, 0); // O_RDONLY, on all supported platforms.
        TIFFClose(tiff);
    }

    static bool encode_tiff_samples(TIFF* tiff, const std::uint8_t* data,
                                    int width, int height, int channels, int bits, std::size_t row_stride,
                                    const std::optional<std::string>& description, const TiffColorTags* color,
                                    const AtomicFileOptions* policy, bool* was_cancelled, std::string& error,
                                    bool* allocation_failed = nullptr, std::string_view xmp = {}) {
        const auto missing_tag_storage = [&]() {
            if (allocation_failed)
                *allocation_failed = true;
            error = "Cannot allocate TIFF tag storage";
            return false;
        };
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bits / 8);
        if (!TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width) ||
            !TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height) ||
            !TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, channels) ||
            !TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits)) {
            error = "TIFF layout tag write failed";
            return false;
        }
        if (bits == 16 && !TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT)) {
            error = "TIFF sample format write failed";
            return false;
        }
        if (description && !TIFFSetField(tiff, TIFFTAG_IMAGEDESCRIPTION, description->c_str())) {
            error = "TIFF description write failed";
            return false;
        }
        if (description) {
            char* stored = nullptr;
            if (!TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &stored) || !stored ||
                std::strcmp(stored, description->c_str()) != 0)
                return missing_tag_storage();
        }
        if (!xmp.empty()) {
            if (!TIFFSetField(tiff, TIFFTAG_XMLPACKET, static_cast<uint32_t>(xmp.size()), xmp.data())) {
                error = "TIFF XMP write failed";
                return false;
            }
            uint32_t count = 0;
            void* stored = nullptr;
            if (!TIFFGetField(tiff, TIFFTAG_XMLPACKET, &count, &stored) || !stored ||
                count != xmp.size() || std::memcmp(stored, xmp.data(), xmp.size()) != 0)
                return missing_tag_storage();
        }
        if (color && color->chromaticities) {
            const auto& xy = *color->chromaticities;
            const std::array<float, 6> primaries{xy[0], xy[1], xy[2], xy[3], xy[4], xy[5]};
            const std::array<float, 2> white{xy[6], xy[7]};
            if (!TIFFSetField(tiff, TIFFTAG_PRIMARYCHROMATICITIES, primaries.data()) ||
                !TIFFSetField(tiff, TIFFTAG_WHITEPOINT, white.data())) {
                error = "TIFF chromaticities write failed";
                return false;
            }
        }
        if (color && color->chromaticities) {
            float *primaries = nullptr, *white = nullptr;
            if (!TIFFGetField(tiff, TIFFTAG_PRIMARYCHROMATICITIES, &primaries) || !primaries ||
                !TIFFGetField(tiff, TIFFTAG_WHITEPOINT, &white) || !white ||
                std::memcmp(primaries, color->chromaticities->data(), 6 * sizeof(float)) != 0 ||
                std::memcmp(white, color->chromaticities->data() + 6, 2 * sizeof(float)) != 0)
                return missing_tag_storage();
        }
        if (color && !color->transfer_function.empty() &&
            !TIFFSetField(tiff, TIFFTAG_TRANSFERFUNCTION, color->transfer_function.data(),
                          color->transfer_function.data(), color->transfer_function.data())) {
            error = "TIFF transfer function write failed";
            return false;
        }
        if (color && !color->transfer_function.empty()) {
            uint16_t *red = nullptr, *green = nullptr, *blue = nullptr;
            const auto count = color->transfer_function.size_bytes();
            if (!TIFFGetField(tiff, TIFFTAG_TRANSFERFUNCTION, &red, &green, &blue) ||
                !red || !green || !blue ||
                std::memcmp(red, color->transfer_function.data(), count) != 0 ||
                std::memcmp(green, color->transfer_function.data(), count) != 0 ||
                std::memcmp(blue, color->transfer_function.data(), count) != 0)
                return missing_tag_storage();
        }
        // Keep the existing explicit strip geometry and LZW policy.
        std::uint16_t extra_sample = EXTRASAMPLE_UNASSALPHA;
        if (!TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT) ||
            !TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) ||
            !TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, channels < 3 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB) ||
            !TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, std::min<uint32_t>(height, TIFFDefaultStripSize(tiff, 0))) ||
            !TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW) ||
            ((channels == 2 || channels == 4) && !TIFFSetField(tiff, TIFFTAG_EXTRASAMPLES, 1, &extra_sample))) {
            error = "TIFF encoding tag write failed";
            return false;
        }
        if (channels == 2 || channels == 4) {
            uint16_t count = 0;
            uint16_t* extras = nullptr;
            if (!TIFFGetField(tiff, TIFFTAG_EXTRASAMPLES, &count, &extras) || count != 1 ||
                !extras || extras[0] != EXTRASAMPLE_UNASSALPHA)
                return missing_tag_storage();
        }
        // libtiff may transform its input buffer. Keep borrowed UINT16 pixels
        // read-only and aligned, including spans with an unaligned start.
        std::vector<std::uint16_t> row(bits == 16 ? row_bytes / 2 : 0);
        bool success = true;
        for (int y = 0; y < height; ++y) {
            if (policy && policy->cancelled && policy->cancelled()) {
                if (was_cancelled)
                    *was_cancelled = true;
                error = "TIFF write cancelled";
                return false;
            }
            auto* source = const_cast<std::uint8_t*>(data) + static_cast<std::size_t>(y) * row_stride;
            if (bits == 16)
                std::memcpy(row.data(), source, row_bytes);
            if (TIFFWriteScanline(tiff, bits == 16 ? static_cast<void*>(row.data()) : source, y, 0) < 0) {
                success = false;
                break;
            }
        }
        success = success && TIFFWriteDirectory(tiff) != 0 && TIFFFlush(tiff) != 0;
        if (!success)
            error = "TIFF scanline write failed";
        return success;
    }

    static bool write_tiff_stream(const std::filesystem::path& path, const std::uint8_t* data,
                                  const int width, const int height, const int channels, std::string& error, FILE* borrowed,
                                  const int bits = 8, std::size_t row_stride = 0,
                                  const std::optional<std::string>& description = std::nullopt,
                                  const AtomicFileOptions* policy = nullptr, bool* was_cancelled = nullptr,
                                  const TiffColorTags* color = nullptr, std::string_view xmp = {}) {
        if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4 || (bits != 8 && bits != 16) ||
            static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / channels / (bits / 8)) {
            error = "Unsupported TIFF layout";
            return false;
        }
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bits / 8);
        if (!row_stride)
            row_stride = row_bytes;
        if (row_stride < row_bytes || row_stride > std::numeric_limits<std::size_t>::max() / height) {
            error = "Unsupported TIFF stride";
            return false;
        }
        TIFF* tiff = borrowed ? tiff_stream(path, borrowed) : open_tiff(path, "w", error);
        if (!tiff) {
            if (borrowed)
                error = "Could not initialize TIFF codec for " + path_to_utf8(path);
            return false;
        }
        const std::unique_ptr<TIFF, decltype(&close_tiff_writer)> owner(tiff, close_tiff_writer);
        return encode_tiff_samples(tiff, data, width, height, channels, bits, row_stride,
                                   description, color, policy, was_cancelled, error, nullptr, xmp);
    }

    bool write_tiff(const std::filesystem::path& path, const std::uint8_t* data,
                    const int width, const int height, const int channels, std::string& error) {
        return write_tiff_stream(path, data, width, height, channels, error, nullptr);
    }

    std::span<const std::uint16_t> tiff_transfer_function(std::string_view transfer) {
        const auto build = [](std::string_view name) {
            std::array<std::uint16_t, 65536> table{};
            for (size_t i = 0; i < table.size(); ++i) {
                const float value = float(i) / 65535.f;
                const float linear = name == "srgb"    ? color::srgbToLinear(value)
                                     : name == "bt709" ? color::bt709ToLinear(value)
                                                       : value;
                table[i] = static_cast<uint16_t>(std::floor(std::clamp(double(linear), 0., 1.) * 65535. + .5));
            }
            return table;
        };
        if (transfer == "linear") {
            static const auto table = build("linear");
            return table;
        }
        if (transfer == "srgb") {
            static const auto table = build("srgb");
            return table;
        }
        if (transfer == "bt709") {
            static const auto table = build("bt709");
            return table;
        }
        return {};
    }

    bool read_tiff_metadata(const std::filesystem::path& path, TiffMetadata& result, std::string& error) {
        result = {};
        TIFF* tiff = open_tiff(path, "r", error);
        if (!tiff)
            return false;
        const std::unique_ptr<TIFF, decltype(&TIFFClose)> owner(tiff, TIFFClose);
        uint16_t bits = 0, channels = 0, format = 0, extra_count = 0;
        uint16_t* extras = nullptr;
        TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
        TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &format);
        result.bits = bits;
        result.channels = channels;
        result.unsigned_integer = format == SAMPLEFORMAT_UINT;
        result.straight_alpha = TIFFGetField(tiff, TIFFTAG_EXTRASAMPLES, &extra_count, &extras) &&
                                extra_count == 1 && extras && extras[0] == EXTRASAMPLE_UNASSALPHA;
        char* text = nullptr;
        if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &text) && text) {
            constexpr size_t limit = 1024 * 1024;
            size_t length = 0;
            while (length <= limit && text[length])
                ++length;
            if (length > limit) {
                // Oversized external descriptions are not our bounded export
                // contract. Preserve legacy pixel loading without parsing them.
                return true;
            }
            result.description = std::string(text, length);
        }
        // Color tables are needed only for our explicitly versioned exports.
        // Avoid allocating/parsing arbitrary external TIFF profiles.
        if (!result.description)
            return true;
        const auto metadata = nlohmann::json::parse(*result.description, nullptr, false);
        if (!metadata.is_object() || !metadata.contains("lfsEncoding") ||
            !metadata["lfsEncoding"].is_string() || metadata["lfsEncoding"].get<std::string>() != tiff_sdr_encoding)
            return true;
        float *primaries = nullptr, *white = nullptr;
        if (TIFFGetField(tiff, TIFFTAG_PRIMARYCHROMATICITIES, &primaries) &&
            TIFFGetField(tiff, TIFFTAG_WHITEPOINT, &white))
            result.chromaticities = std::array<float, 8>{primaries[0], primaries[1], primaries[2], primaries[3], primaries[4], primaries[5], white[0], white[1]};
        uint16_t *red = nullptr, *green = nullptr, *blue = nullptr;
        const auto transfer = metadata.find("lfsTransfer");
        const auto expected = transfer != metadata.end() && transfer->is_string()
                                  ? tiff_transfer_function(transfer->get<std::string>())
                                  : std::span<const uint16_t>{};
        if (bits == 16 && channels >= 3 && TIFFGetField(tiff, TIFFTAG_TRANSFERFUNCTION, &red, &green, &blue) && red && green && blue) {
            result.matching_rgb_transfer = !expected.empty();
            for (size_t i = 0; result.matching_rgb_transfer && i < expected.size(); ++i)
                result.matching_rgb_transfer = red[i] == green[i] && red[i] == blue[i] &&
                                               std::abs(int(red[i]) - int(expected[i])) <= 1;
        }
        return true;
    }

    namespace {
        enum class TiffColorValidation { Valid,
                                         Unsupported,
                                         Chromaticities };
        TiffColorValidation validate_tiff_color(const TiffColorTags& color, int channels, int bits) {
            if ((!color.transfer_function.empty() && color.transfer_function.size() != (std::size_t{1} << bits)) ||
                ((color.chromaticities || !color.transfer_function.empty()) && channels < 3))
                return TiffColorValidation::Unsupported;
            if (color.chromaticities) {
                const auto& xy = *color.chromaticities;
                for (size_t i = 0; i < xy.size(); i += 2)
                    if (!std::isfinite(xy[i]) || !std::isfinite(xy[i + 1]) ||
                        xy[i] < 0 || xy[i + 1] <= 0 || xy[i] + xy[i + 1] > 1.000001f)
                        return TiffColorValidation::Chromaticities;
            }
            return TiffColorValidation::Valid;
        }
    } // namespace

    Status encode_tiff_memory(std::span<const std::uint8_t> data, int width, int height, int channels,
                              int bit_depth, std::size_t row_stride, std::vector<std::uint8_t>& output,
                              const TiffEncodeOptions& options) {
        const auto failure = [](ErrorCode code, std::string detail) -> Status {
            return Status::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(detail), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        const auto invalid = [&](std::string_view reason) -> Status {
            return failure(ErrorCode::InvalidArgument,
                           std::format("{} (width={}, height={}, channels={}, depth={}, stride={}, span_bytes={}, limit={})",
                                       reason, width, height, channels, bit_depth, row_stride, data.size(), options.max_output_bytes));
        };
        if (width <= 0 || height <= 0 || channels < 1 || channels > 4 || (bit_depth != 8 && bit_depth != 16) ||
            static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / channels / (bit_depth / 8))
            return invalid("Unsupported TIFF layout");
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bit_depth / 8);
        if (row_stride < row_bytes || static_cast<std::size_t>(height - 1) > (std::numeric_limits<std::size_t>::max() - row_bytes) / row_stride ||
            data.size() < row_stride * (height - 1) + row_bytes)
            return invalid("Unsupported TIFF buffer size or stride");
        if (!options.max_output_bytes || options.max_output_bytes > std::numeric_limits<uint32_t>::max() || options.max_output_bytes > output.max_size())
            return invalid("Invalid classic TIFF encoded-byte limit");
        if (options.description && (options.description->find('\0') != std::string::npos || options.description->size() > 1024 * 1024))
            return invalid("Unsupported TIFF description");
        const auto color = validate_tiff_color(options.color, channels, bit_depth);
        if (color != TiffColorValidation::Valid)
            return invalid(color == TiffColorValidation::Unsupported ? "Unsupported TIFF color tags" : "Invalid TIFF chromaticities");
        try {
            EncodedMemory memory;
            memory.limit = options.max_output_bytes;
            TiffMemory destination{{}, &memory};
            std::string error;
            std::unique_ptr<TIFF, decltype(&close_tiff_writer)> tiff(destination.open("wm"), close_tiff_writer);
            const bool encoded = tiff && encode_tiff_samples(tiff.get(), data.data(), width, height, channels, bit_depth, row_stride,
                                                             options.description, &options.color, nullptr, nullptr, error, &destination.allocation_failed);
            // The shared writer explicitly finalizes the directory and flushes.
            // Release all codec state before publishing the owned destination.
            tiff.reset();
            // Preserve the writer's strict diagnostic policy; readers instead
            // follow libtiff's return value for recoverable metadata errors.
            if (!encoded || destination.failed || destination.message[0]) {
                if (memory.failure != EncodedMemory::Failure::None || destination.allocation_failed)
                    return failure(ErrorCode::ResourceExhausted, memory.failure == EncodedMemory::Failure::Limit
                                                                     ? "TIFF encoded-byte limit exhausted"
                                                                     : "Cannot allocate TIFF encoder or output");
                return failure(ErrorCode::Unavailable, std::format("TIFF encoding failed: {}{}", error, destination.message.data()));
            }
            output = std::move(memory.bytes);
            return {};
        } catch (const std::bad_alloc&) {
            return failure(ErrorCode::ResourceExhausted, "Cannot allocate TIFF encoder or output");
        }
    }

    Status write_tiff_u16(const std::filesystem::path& path, std::span<const std::uint8_t> data,
                          int width, int height, int channels, std::size_t row_stride,
                          const std::optional<std::string>& description, const AtomicFileOptions& options,
                          const TiffColorTags& color, std::string_view xmp) {
        const auto invalid = [](std::string message) -> Status {
            return Status::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()}));
        };
        if (width <= 0 || height <= 0 || channels < 1 || channels > 4 ||
            static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / channels / 2)
            return invalid("Unsupported TIFF16 layout");
        if (!valid_xmp_packet(xmp))
            return invalid("Invalid TIFF16 XMP packet");
        const auto row_bytes = static_cast<std::size_t>(width) * channels * 2;
        if (row_stride < row_bytes || row_stride > std::numeric_limits<std::size_t>::max() / height ||
            data.size() < row_stride * (height - 1) + row_bytes)
            return invalid("Unsupported TIFF16 buffer size or stride");
        if (description && (description->find('\0') != std::string::npos || description->size() > 1024 * 1024))
            return invalid("Unsupported TIFF16 description");
        const auto color_validation = validate_tiff_color(color, channels, 16);
        if (color_validation == TiffColorValidation::Unsupported)
            return invalid("Unsupported TIFF16 color tags");
        if (color_validation == TiffColorValidation::Chromaticities)
            return invalid("Invalid TIFF16 chromaticities");
        const auto extension = lower_extension(path);
        if (extension != ".tif" && extension != ".tiff")
            return invalid("TIFF16 output requires a TIFF extension");
        return writeFileAtomically(path, [&](FILE* stream) -> Status {
            std::string error;
            bool cancelled = false;
            if (!write_tiff_stream(path, data.data(), width, height, channels, error, stream,
                                   16, row_stride, description, &options, &cancelled, &color, xmp))
                return Status::failure(make_error({.code = cancelled ? ErrorCode::Cancelled : ErrorCode::Unavailable,
                                                    .domain = ErrorDomain::IO, .detail = std::move(error), .detection = LFS_SOURCE_SITE_CURRENT()}));
            return {}; }, options);
    }

    Status write_tiff_u16(const std::filesystem::path& path, std::span<const std::uint8_t> data,
                          int width, int height, int channels, std::size_t row_stride,
                          const std::optional<std::string>& description, const AtomicFileOptions& options,
                          const TiffColorTags& color) {
        return write_tiff_u16(path, data, width, height, channels, row_stride, description, options, color, {});
    }

    static bool encode_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                                const int width, const int height, const int channels, const int jpeg_quality,
                                const std::optional<std::string>& comment, std::string& error, const bool jpeg_full_chroma, FILE* borrowed, std::span<const std::string_view> png_xmp) {
        auto extension = lfs::core::path_to_utf8(path.extension());
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
        if (extension == ".png")
            return write_png_stream(path, data, width, height, channels, 8, 6, comment, error, borrowed, png_xmp);
        if (extension == ".tif" || extension == ".tiff")
            return write_tiff_stream(path, data, width, height, channels, error, borrowed);
        if (extension == ".jpg" || extension == ".jpeg") {
            if (channels != 4)
                return write_jpeg_stream(path, data, width, height, channels, jpeg_quality, comment, error, jpeg_full_chroma, borrowed);
            if (!data || width <= 0 || height <= 0 || width > JPEG_MAX_DIMENSION || height > JPEG_MAX_DIMENSION ||
                static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height) / 4) {
                error = "Unsupported JPEG layout";
                return false;
            }
            const auto pixels = static_cast<std::size_t>(width) * height;
            std::vector<std::uint8_t> rgb(pixels * 3);
            for (std::size_t pixel = 0; pixel < pixels; ++pixel)
                std::copy_n(data + pixel * 4, 3, rgb.data() + pixel * 3);
            return write_jpeg_stream(path, rgb.data(), width, height, 3, jpeg_quality, comment, error, jpeg_full_chroma, borrowed);
        }
        error = "Unsupported image extension: " + extension;
        return false;
    }
    Status write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                          const int width, const int height, const int channels, const int jpeg_quality,
                          const std::optional<std::string>& comment, const AtomicFileOptions& options,
                          const bool jpeg_full_chroma, std::span<const std::string_view> png_xmp) {
        if (!valid_png_xmp(png_xmp) || (png_xmp_bytes(png_xmp) && lower_extension(path) != ".png"))
            return Status::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Invalid PNG XMP packet or output format", .detection = LFS_SOURCE_SITE_CURRENT()}));
        if (lower_extension(path) == ".avif") {
            if (comment && !comment->empty())
                return Status::failure(make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = "AVIF metadata comments require an explicit metadata policy", .detection = LFS_SOURCE_SITE_CURRENT()}));
            if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4 || size_t(width) > SIZE_MAX / size_t(height) / size_t(channels))
                return Status::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Invalid packed AVIF byte image", .detection = LFS_SOURCE_SITE_CURRENT()}));
            const size_t stride = size_t(width) * channels;
            return write_avif(path, {data, stride * size_t(height)}, width, height, channels, stride, {}, options);
        }
        if (lower_extension(path) == ".jxl") {
            if (comment && !comment->empty())
                return Status::failure(make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = "JXL metadata comments require an explicit metadata policy", .detection = LFS_SOURCE_SITE_CURRENT()}));
            if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4 ||
                size_t(width) > SIZE_MAX / size_t(height) / size_t(channels))
                return Status::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "Invalid packed JXL byte image", .detection = LFS_SOURCE_SITE_CURRENT()}));
            const size_t stride = size_t(width) * channels;
            return write_jxl(path, {data, stride * size_t(height)}, width, height, channels, SampleType::UInt8, stride, {}, options);
        }
        return writeFileAtomically(path, [&](FILE* stream) -> Status {
            std::string error;
            if (!encode_image_u8(path, data, width, height, channels, jpeg_quality, comment, error, jpeg_full_chroma, stream, png_xmp))
                return Status::failure(make_error({.code = error.starts_with("Unsupported") ? ErrorCode::InvalidArgument : ErrorCode::Unavailable,
                                                  .domain = ErrorDomain::IO, .detail = std::move(error), .detection = LFS_SOURCE_SITE_CURRENT()}));
            return {}; }, options);
    }
    Status write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                          const int width, const int height, const int channels, const int jpeg_quality,
                          const std::optional<std::string>& comment, const AtomicFileOptions& options,
                          const bool jpeg_full_chroma) {
        return write_image_u8(path, data, width, height, channels, jpeg_quality, comment, options, jpeg_full_chroma, {});
    }
    bool write_image_u8(const std::filesystem::path& path, const std::uint8_t* data,
                        const int width, const int height, const int channels, const int jpeg_quality,
                        const std::optional<std::string>& comment, std::string& error, const bool jpeg_full_chroma) {
        const auto result = write_image_u8(path, data, width, height, channels, jpeg_quality, comment,
                                           {.overwrite = true, .durable = false, .create_directories = false}, jpeg_full_chroma);
        if (!result)
            error = std::string(result.error().detail());
        return bool(result);
    }
} // namespace lfs::core::image_codecs
