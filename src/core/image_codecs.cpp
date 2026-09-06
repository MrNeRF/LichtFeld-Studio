/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "image_codecs.hpp"

#include "core/path_utils.hpp"

#include <jpeglib.h>
#include <png.h>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <tiffio.h>
#include <tinyexr.h>
#include <webp/decode.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lfs::core::image_codecs {

    namespace {

        constexpr std::array<std::uint8_t, 8> kPngSignature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};

        std::string lower_extension(const std::filesystem::path& path) {
            auto extension = path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return extension;
        }

        bool read_file(const std::filesystem::path& path, std::vector<std::uint8_t>& data, std::string& error) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
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

        bool read_prefix(const std::filesystem::path& path, std::vector<std::uint8_t>& data, std::string& error) {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                error = "Could not open file " + path_to_utf8(path);
                return false;
            }
            data.resize(32);
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

        bool is_tiff(const std::vector<std::uint8_t>& data) {
            return data.size() >= 4 && ((data[0] == 'I' && data[1] == 'I' && data[2] == 42 && data[3] == 0) ||
                                        (data[0] == 'M' && data[1] == 'M' && data[2] == 0 && data[3] == 42));
        }

        bool is_hdr(const std::vector<std::uint8_t>& data) {
            return data.size() >= 10 && (std::memcmp(data.data(), "#?RADIANCE", 10) == 0 ||
                                         std::memcmp(data.data(), "#?RGBE", 6) == 0);
        }

        bool is_exr(const std::vector<std::uint8_t>& data) {
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
#ifdef _WIN32
            return _wfopen(path.c_str(), L"rb");
#else
            const auto path_utf8 = path_to_utf8(path);
            return std::fopen(path_utf8.c_str(), "rb");
#endif
        }

        std::FILE* open_output_file(const std::filesystem::path& path) {
#ifdef _WIN32
            return _wfopen(path.c_str(), L"wb");
#else
            const auto path_utf8 = path_to_utf8(path);
            return std::fopen(path_utf8.c_str(), "wb");
#endif
        }

        void* allocate_target(DecodeTarget& target, const std::size_t bytes, std::string& error) {
            if (target.data)
                return target.data;
            target.data = target.allocate ? target.allocate(bytes, target.user) : std::malloc(bytes);
            if (!target.data)
                error = "Could not allocate image buffer";
            return target.data;
        }

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
            if (output_channels != target.channels || output_bit_depth != expected_bit_depth) {
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
            png_read_image(png, rows.data());
            png_read_end(png, info);
            if (compact_gray) {
                auto* output = static_cast<std::uint16_t*>(target.data);
                const auto* gray = output;
                const auto pixels = static_cast<std::size_t>(width) * height;
                for (std::size_t pixel = pixels; pixel-- > 0;) {
                    const auto value = gray[pixel];
                    output[pixel * 3 + 0] = value;
                    output[pixel * 3 + 1] = value;
                    output[pixel * 3 + 2] = value;
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
                if (is_16_bit_png) {
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
                error = "PNG decode failed";
                return false;
            }
            png_init_io(png, file);
            png_read_info(png, info);
            const bool success = configure_png_target(png, info, target, result, error);
            png_destroy_read_struct(&png, &info, nullptr);
            std::fclose(file);
            return success;
        }

        bool decode_png_memory_to_buffer(const std::uint8_t* data, const std::size_t size,
                                         DecodeTarget& target, Probe& result, std::string& error) {
            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                png_destroy_read_struct(&png, &info, nullptr);
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                png_destroy_read_struct(&png, &info, nullptr);
                error = "PNG decode failed";
                return false;
            }
            PngReadContext input{data, size, 0};
            png_set_read_fn(png, &input, png_read_callback);
            png_read_info(png, info);
            const bool success = configure_png_target(png, info, target, result, error);
            png_destroy_read_struct(&png, &info, nullptr);
            return success;
        }

        bool decode_png(const std::vector<std::uint8_t>& file_data, Image& result, std::string& error) {
            if (!is_png(file_data)) {
                error = "Invalid PNG signature";
                return false;
            }
            png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            png_infop info = png ? png_create_info_struct(png) : nullptr;
            if (!png || !info) {
                png_destroy_read_struct(&png, &info, nullptr);
                error = "Could not allocate PNG decoder";
                return false;
            }
            if (setjmp(png_jmpbuf(png))) {
                png_destroy_read_struct(&png, &info, nullptr);
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
                png_destroy_read_struct(&png, &info, nullptr);
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
            png_read_image(png, rows.data());
            png_read_end(png, info);
            png_destroy_read_struct(&png, &info, nullptr);
            return true;
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
                const auto max_dimension = std::max(cinfo.image_width, cinfo.image_height);
                for (const unsigned int denominator : {1u, 2u, 4u, 8u}) {
                    if ((max_dimension + denominator - 1) / denominator <=
                        static_cast<unsigned int>(target.max_width)) {
                        cinfo.scale_num = 1;
                        cinfo.scale_denom = denominator;
                        break;
                    }
                }
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

        TIFF* open_tiff(const std::filesystem::path& path, const char* mode, std::string& error) {
            install_tiff_quiet_handlers();
#ifdef _WIN32
            TIFF* tiff = TIFFOpenW(path.c_str(), mode);
#else
            const auto path_utf8 = path_to_utf8(path);
            TIFF* tiff = TIFFOpen(path_utf8.c_str(), mode);
#endif
            if (!tiff)
                error = "Could not initialize TIFF codec for " + path_to_utf8(path);
            return tiff;
        }

        bool decode_tiff(const std::filesystem::path& path, Image& result, std::string& error) {
            TIFF* tiff = open_tiff(path, "r", error);
            if (!tiff)
                return false;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            std::uint16_t planar = PLANARCONFIG_CONTIG;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar);
            if (width == 0 || height == 0 || channels < 1 || channels > 4) {
                TIFFClose(tiff);
                error = "Unsupported TIFF layout";
                return false;
            }
            if (TIFFIsTiled(tiff) && bits == 8 && sample_format == SAMPLEFORMAT_UINT) {
                std::vector<std::uint32_t> rgba(static_cast<std::size_t>(width) * height);
                if (!TIFFReadRGBAImageOriented(tiff, width, height, rgba.data(), ORIENTATION_TOPLEFT, 0)) {
                    TIFFClose(tiff);
                    error = "TIFF RGBA fallback failed";
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
                TIFFClose(tiff);
                return true;
            }
            if ((bits != 8 && bits != 16 && bits != 32) ||
                (sample_format != SAMPLEFORMAT_UINT && sample_format != SAMPLEFORMAT_IEEEFP) ||
                (sample_format == SAMPLEFORMAT_IEEEFP && bits != 32)) {
                TIFFClose(tiff);
                error = "Unsupported TIFF sample format";
                return false;
            }
            const std::size_t sample_size = bits / 8;
            result.width = static_cast<int>(width);
            result.height = static_cast<int>(height);
            result.channels = channels;
            result.sample_type = sample_format == SAMPLEFORMAT_IEEEFP ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
                                                                                                         : SampleType::UInt8;
            result.data.resize(static_cast<std::size_t>(width) * height * channels * sample_size);
            const tmsize_t scanline_size = TIFFScanlineSize(tiff);
            std::vector<std::uint8_t> row(static_cast<std::size_t>(std::max<tmsize_t>(scanline_size, 1)));
            const auto row_bytes = static_cast<std::size_t>(width) * channels * sample_size;
            for (std::uint32_t y = 0; y < height; ++y) {
                if (planar == PLANARCONFIG_CONTIG) {
                    if (TIFFReadScanline(tiff, row.data(), y, 0) < 0) {
                        TIFFClose(tiff);
                        error = "TIFF scanline read failed";
                        return false;
                    }
                    std::memcpy(result.data.data() + static_cast<std::size_t>(y) * row_bytes, row.data(), row_bytes);
                } else {
                    for (std::uint16_t channel = 0; channel < channels; ++channel) {
                        if (TIFFReadScanline(tiff, row.data(), y, channel) < 0) {
                            TIFFClose(tiff);
                            error = "TIFF planar scanline read failed";
                            return false;
                        }
                        for (std::uint32_t x = 0; x < width; ++x) {
                            std::memcpy(result.data.data() + (static_cast<std::size_t>(y) * width * channels +
                                                              static_cast<std::size_t>(x) * channels + channel) *
                                                                 sample_size,
                                        row.data() + static_cast<std::size_t>(x) * sample_size, sample_size);
                        }
                    }
                }
            }
            TIFFClose(tiff);
            return true;
        }

        bool decode_tiff_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                                   Probe& result, std::string& error) {
            TIFF* tiff = open_tiff(path, "r", error);
            if (!tiff)
                return false;
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint16_t channels = 1;
            std::uint16_t bits = 8;
            std::uint16_t sample_format = SAMPLEFORMAT_UINT;
            std::uint16_t planar = PLANARCONFIG_CONTIG;
            TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
            TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_BITSPERSAMPLE, &bits);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &sample_format);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_PLANARCONFIG, &planar);
            const auto source_type = sample_format == SAMPLEFORMAT_IEEEFP ? SampleType::Float32
                                     : bits == 16                         ? SampleType::UInt16
                                                                          : SampleType::UInt8;
            const auto sample_size = static_cast<std::size_t>(bits / 8);
            const auto row_bytes = static_cast<std::size_t>(width) * channels * sample_size;
            const bool direct = width > 0 && height > 0 && channels == target.channels && planar == PLANARCONFIG_CONTIG &&
                                !TIFFIsTiled(tiff) && source_type == target.sample_type &&
                                ((bits == 8 && target.sample_type == SampleType::UInt8) ||
                                 (bits == 16 && target.sample_type == SampleType::UInt16) ||
                                 (bits == 32 && target.sample_type == SampleType::Float32)) &&
                                TIFFScanlineSize(tiff) == static_cast<tmsize_t>(row_bytes);
            if (!direct) {
                TIFFClose(tiff);
                error = "TIFF layout requires conversion";
                return false;
            }
            if (!allocate_target(target, row_bytes * height, error)) {
                TIFFClose(tiff);
                return false;
            }
            result = {static_cast<int>(width), static_cast<int>(height), target.channels, target.sample_type};
            for (std::uint32_t y = 0; y < height; ++y) {
                auto* row = static_cast<std::uint8_t*>(target.data) + static_cast<std::size_t>(y) * row_bytes;
                if (TIFFReadScanline(tiff, row, y, 0) < 0) {
                    TIFFClose(tiff);
                    error = "TIFF scanline read failed";
                    return false;
                }
            }
            TIFFClose(tiff);
            return true;
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

        bool decode_stb(const std::vector<std::uint8_t>& file_data, bool hdr, Image& result, std::string& error) {
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
                struct stat file_status {};
                if (descriptor >= 0 && fstat(descriptor, &file_status) == 0 && file_status.st_size > 0 &&
                    file_status.st_size <= std::numeric_limits<int>::max()) {
                    const auto size = static_cast<std::size_t>(file_status.st_size);
                    void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
                    if (mapped != MAP_FAILED) {
                        int width = 0;
                        int height = 0;
                        int source_channels = 0;
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

        bool decode_exr(const std::filesystem::path& path, Image& result, std::string& error) {
            float* decoded = nullptr;
            int width = 0;
            int height = 0;
            const auto path_utf8 = path_to_utf8(path);
            const char* exr_error = nullptr;
            const int status = LoadEXR(&decoded, &width, &height, path_utf8.c_str(), &exr_error);
            if (status != TINYEXR_SUCCESS || !decoded) {
                set_error(error, "EXR decode failed", exr_error);
                if (exr_error)
                    FreeEXRErrorMessage(exr_error);
                return false;
            }
            result.width = width;
            result.height = height;
            result.channels = 4;
            result.sample_type = SampleType::Float32;
            const auto bytes = static_cast<std::size_t>(width) * height * 4 * sizeof(float);
            result.data.resize(bytes);
            std::memcpy(result.data.data(), decoded, bytes);
            free(decoded);
            return true;
        }

    } // namespace

    bool decode_to_buffer(const std::filesystem::path& path, DecodeTarget& target,
                          Probe& result, std::string& error) {
        const auto extension = lower_extension(path);
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
        if (!data || size == 0) {
            error = "Empty image buffer";
            return false;
        }
        if (size >= 2 && data[0] == 0xff && data[1] == 0xd8)
            return decode_jpeg_memory_to_buffer(data, size, target, result, error);
        if (size >= kPngSignature.size() && std::equal(kPngSignature.begin(), kPngSignature.end(), data))
            return decode_png_memory_to_buffer(data, size, target, result, error);
        error = "Unsupported image buffer format";
        return false;
    }

    bool probe(const std::filesystem::path& path, Probe& result, std::string& error) {
        const auto extension = lower_extension(path);
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
            result = {static_cast<int>(width), static_cast<int>(height), static_cast<int>(channels),
                      sample_format == SAMPLEFORMAT_IEEEFP ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
                                                                                              : SampleType::UInt8};
            return true;
        }

        std::vector<std::uint8_t> prefix;
        if (!read_prefix(path, prefix, error))
            return false;
        if (is_exr(prefix)) {
            error = "EXR dimensions require codec decode";
            return false;
        }
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
            result = {static_cast<int>(width), static_cast<int>(height), static_cast<int>(channels),
                      sample_format == SAMPLEFORMAT_IEEEFP ? SampleType::Float32 : bits == 16 ? SampleType::UInt16
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
        const auto extension = lower_extension(path);
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
        if (!data || size == 0) {
            error = "Empty image buffer";
            return false;
        }
        if (size >= 2 && data[0] == 0xff && data[1] == 0xd8)
            return decode_jpeg_bytes(data, size, result, error);
        if (size >= kPngSignature.size() && std::equal(kPngSignature.begin(), kPngSignature.end(), data)) {
            std::vector<std::uint8_t> image_data(data, data + size);
            return decode_png(image_data, result, error);
        }
        std::vector<std::uint8_t> image_data(data, data + size);
        return decode_stb(image_data, false, result, error);
    }

    bool decode_jpeg_memory(const std::uint8_t* data, const size_t size, Image& result, std::string& error) {
        if (!data || size == 0) {
            error = "Empty JPEG buffer";
            return false;
        }
        return decode_jpeg_bytes(data, size, result, error);
    }

    bool write_jpeg(const std::filesystem::path& path, const std::uint8_t* data,
                    const int width, const int height, const int channels, const int quality,
                    const std::optional<std::string>& comment, std::string& error) {
        if (!data || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
            error = "Unsupported JPEG layout";
            return false;
        }
        jpeg_compress_struct cinfo{};
        JpegError jerror;
        cinfo.err = jpeg_std_error(&jerror.base);
        jerror.base.error_exit = jpeg_error_exit;
        if (setjmp(jerror.jump)) {
            jpeg_destroy_compress(&cinfo);
            set_error(error, "JPEG encode failed", jerror.message);
            return false;
        }
        jpeg_create_compress(&cinfo);
        unsigned char* output = nullptr;
        unsigned long output_size = 0;
        jpeg_mem_dest(&cinfo, &output, &output_size);
        cinfo.image_width = width;
        cinfo.image_height = height;
        cinfo.input_components = channels;
        cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
        jpeg_set_defaults(&cinfo);
        jpeg_set_quality(&cinfo, std::clamp(quality, 1, 100), TRUE);
        jpeg_start_compress(&cinfo, TRUE);
        if (comment && !comment->empty()) {
            const auto length = std::min<std::size_t>(comment->size(), 65533);
            jpeg_write_marker(&cinfo, JPEG_COM, reinterpret_cast<const JOCTET*>(comment->data()), length);
        }
        while (cinfo.next_scanline < cinfo.image_height) {
            JSAMPROW row = const_cast<JSAMPROW>(data + static_cast<std::size_t>(cinfo.next_scanline) * width * channels);
            jpeg_write_scanlines(&cinfo, &row, 1);
        }
        jpeg_finish_compress(&cinfo);
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            free(output);
            jpeg_destroy_compress(&cinfo);
            error = "Could not open JPEG output " + path_to_utf8(path);
            return false;
        }
        file.write(reinterpret_cast<const char*>(output), static_cast<std::streamsize>(output_size));
        const bool success = static_cast<bool>(file);
        free(output);
        jpeg_destroy_compress(&cinfo);
        if (!success)
            error = "Could not write JPEG output " + path_to_utf8(path);
        return success;
    }

    bool write_png(const std::filesystem::path& path, const void* data,
                   const int width, const int height, const int channels, const int bit_depth,
                   const int compression_level, const std::optional<std::string>& comment,
                   std::string& error) {
        if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4 || (bit_depth != 8 && bit_depth != 16)) {
            error = "Unsupported PNG layout";
            return false;
        }
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        png_infop info = png ? png_create_info_struct(png) : nullptr;
        if (!png || !info) {
            png_destroy_write_struct(&png, &info);
            error = "Could not allocate PNG encoder";
            return false;
        }
        std::FILE* file = nullptr;
        if (setjmp(png_jmpbuf(png))) {
            if (file)
                std::fclose(file);
            png_destroy_write_struct(&png, &info);
            error = "PNG encode failed";
            return false;
        }
        file = open_output_file(path);
        if (!file) {
            png_destroy_write_struct(&png, &info);
            error = "Could not open PNG output " + path_to_utf8(path);
            return false;
        }
        png_init_io(png, file);
        const int color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 2 ? PNG_COLOR_TYPE_GRAY_ALPHA
                                                                 : channels == 3   ? PNG_COLOR_TYPE_RGB
                                                                                   : PNG_COLOR_TYPE_RGBA;
        png_set_IHDR(png, info, width, height, bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_compression_level(png, std::clamp(compression_level, 0, 9));
        png_set_compression_strategy(png, Z_DEFAULT_STRATEGY);
        png_text text{};
        if (comment && !comment->empty()) {
            text.compression = PNG_TEXT_COMPRESSION_NONE;
            text.key = const_cast<png_charp>("Comment");
            text.text = const_cast<png_charp>(comment->c_str());
            png_set_text(png, info, &text, 1);
        }
        png_write_info(png, info);
        if constexpr (std::endian::native == std::endian::little) {
            if (bit_depth == 16)
                png_set_swap(png);
        }
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bit_depth / 8);
        std::vector<png_bytep> rows(height);
        auto* bytes = static_cast<png_bytep>(const_cast<void*>(data));
        for (int y = 0; y < height; ++y)
            rows[y] = bytes + static_cast<std::size_t>(y) * row_bytes;
        png_write_image(png, rows.data());
        png_write_end(png, info);
        png_destroy_write_struct(&png, &info);
        const bool success = std::fclose(file) == 0;
        if (!success)
            error = "Could not write PNG output " + path_to_utf8(path);
        return success;
    }

    bool write_tiff(const std::filesystem::path& path, const std::uint8_t* data,
                    const int width, const int height, const int channels, std::string& error) {
        if (!data || width <= 0 || height <= 0 || channels < 1 || channels > 4) {
            error = "Unsupported TIFF layout";
            return false;
        }
        TIFF* tiff = open_tiff(path, "w", error);
        if (!tiff)
            return false;
        TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, channels);
        TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB);
        TIFFSetField(tiff, TIFFTAG_COMPRESSION, COMPRESSION_LZW);
        std::uint16_t extra_sample = EXTRASAMPLE_UNASSALPHA;
        if (channels == 4)
            TIFFSetField(tiff, TIFFTAG_EXTRASAMPLES, 1, &extra_sample);
        const auto row_bytes = static_cast<std::size_t>(width) * channels;
        bool success = true;
        for (int y = 0; y < height; ++y) {
            if (TIFFWriteScanline(tiff, const_cast<std::uint8_t*>(data) + static_cast<std::size_t>(y) * row_bytes, y, 0) < 0) {
                success = false;
                break;
            }
        }
        TIFFClose(tiff);
        if (!success)
            error = "TIFF scanline write failed";
        return success;
    }

} // namespace lfs::core::image_codecs
