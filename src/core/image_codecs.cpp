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

#include <algorithm>
#include <array>
#include <cctype>
#include <csetjmp>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>

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
                error = "Could not open file " + path.string();
                return false;
            }
            const auto end = file.tellg();
            if (end < 0) {
                error = "Could not determine file size for " + path.string();
                return false;
            }
            const auto size = static_cast<std::size_t>(end);
            file.seekg(0);
            data.resize(size);
            if (size != 0 && !file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
                error = "Could not read file " + path.string();
                return false;
            }
            return true;
        }

        bool read_prefix(const std::filesystem::path& path, std::vector<std::uint8_t>& data, std::string& error) {
            std::ifstream file(path, std::ios::binary);
            if (!file) {
                error = "Could not open file " + path.string();
                return false;
            }
            data.resize(32);
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            data.resize(static_cast<std::size_t>(file.gcount()));
            if (file.bad()) {
                error = "Could not read file " + path.string();
                return false;
            }
            return true;
        }

        bool is_png(const std::vector<std::uint8_t>& data) {
            return data.size() >= kPngSignature.size() &&
                   std::equal(kPngSignature.begin(), kPngSignature.end(), data.begin());
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

        struct PngWriteContext {
            std::ofstream file;
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

        void png_write_callback(png_structp png, png_bytep data, png_size_t size) {
            auto* context = static_cast<PngWriteContext*>(png_get_io_ptr(png));
            context->file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
            if (!context->file) {
                png_error(png, "write failed");
            }
        }

        void png_flush_callback(png_structp) {}

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
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            if (bit_depth == 16)
                png_set_swap(png);
#endif
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

        struct TiffStream {
            std::fstream file;
        };

        void tiff_quiet_handler(const char*, const char*, va_list) {}

        void install_tiff_quiet_handlers() {
            static const bool installed = [] {
                TIFFSetErrorHandler(tiff_quiet_handler);
                TIFFSetWarningHandler(tiff_quiet_handler);
                return true;
            }();
            (void)installed;
        }

        tmsize_t tiff_read(thandle_t handle, void* buffer, tmsize_t size) {
            auto* stream = static_cast<TiffStream*>(handle);
            stream->file.read(static_cast<char*>(buffer), size);
            return static_cast<tmsize_t>(stream->file.gcount());
        }

        tmsize_t tiff_write(thandle_t handle, void* buffer, tmsize_t size) {
            auto* stream = static_cast<TiffStream*>(handle);
            stream->file.write(static_cast<const char*>(buffer), size);
            return stream->file ? size : 0;
        }

        toff_t tiff_seek(thandle_t handle, toff_t offset, int whence) {
            auto* stream = static_cast<TiffStream*>(handle);
            std::ios_base::seekdir direction = std::ios::beg;
            if (whence == SEEK_CUR)
                direction = std::ios::cur;
            else if (whence == SEEK_END)
                direction = std::ios::end;
            stream->file.clear();
            stream->file.seekg(static_cast<std::streamoff>(offset), direction);
            stream->file.seekp(static_cast<std::streamoff>(offset), direction);
            return stream->file ? static_cast<toff_t>(stream->file.tellg()) : static_cast<toff_t>(-1);
        }

        int tiff_close(thandle_t handle) {
            auto* stream = static_cast<TiffStream*>(handle);
            stream->file.close();
            delete stream;
            return 0;
        }

        toff_t tiff_size(thandle_t handle) {
            auto* stream = static_cast<TiffStream*>(handle);
            const auto position = stream->file.tellg();
            stream->file.seekg(0, std::ios::end);
            const auto size = stream->file.tellg();
            stream->file.seekg(position);
            return size < 0 ? 0 : static_cast<toff_t>(size);
        }

        int tiff_map(thandle_t, void**, toff_t*) { return 0; }
        void tiff_unmap(thandle_t, void*, toff_t) {}

        TIFF* open_tiff(const std::filesystem::path& path, const char* mode, std::string& error) {
            install_tiff_quiet_handlers();
            auto* stream = new TiffStream;
            const auto open_mode = mode[0] == 'r' ? (std::ios::in | std::ios::binary) : (std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
            stream->file.open(path, open_mode);
            if (!stream->file) {
                delete stream;
                error = "Could not open TIFF " + path.string();
                return nullptr;
            }
            TIFF* tiff = TIFFClientOpen(path.string().c_str(), mode, stream, tiff_read, tiff_write,
                                        tiff_seek, tiff_close, tiff_size, tiff_map, tiff_unmap);
            if (!tiff) {
                delete stream;
                error = "Could not initialize TIFF codec for " + path.string();
            }
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

    bool probe(const std::filesystem::path& path, Probe& result, std::string& error) {
        std::vector<std::uint8_t> prefix;
        if (!read_prefix(path, prefix, error))
            return false;
        const auto extension = lower_extension(path);
        if (is_exr(prefix) || extension == ".exr") {
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
        error = "Unsupported image format: " + path.string();
        return false;
    }

    bool decode(const std::filesystem::path& path, Image& result, std::string& error) {
        std::vector<std::uint8_t> prefix;
        if (!read_prefix(path, prefix, error))
            return false;
        const auto extension = lower_extension(path);
        if (is_tiff(prefix) || extension == ".tif" || extension == ".tiff")
            return decode_tiff(path, result, error);
        if (is_exr(prefix) || extension == ".exr")
            return decode_exr(path, result, error);
        std::vector<std::uint8_t> data;
        if (!read_file(path, data, error))
            return false;
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
        error = "Unsupported image format: " + path.string();
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
            error = "Could not open JPEG output " + path.string();
            return false;
        }
        file.write(reinterpret_cast<const char*>(output), static_cast<std::streamsize>(output_size));
        const bool success = static_cast<bool>(file);
        free(output);
        jpeg_destroy_compress(&cinfo);
        if (!success)
            error = "Could not write JPEG output " + path.string();
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
        if (setjmp(png_jmpbuf(png))) {
            png_destroy_write_struct(&png, &info);
            error = "PNG encode failed";
            return false;
        }
        PngWriteContext context;
        context.file.open(path, std::ios::binary);
        if (!context.file) {
            png_destroy_write_struct(&png, &info);
            error = "Could not open PNG output " + path.string();
            return false;
        }
        png_set_write_fn(png, &context, png_write_callback, png_flush_callback);
        const int color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : channels == 2 ? PNG_COLOR_TYPE_GRAY_ALPHA
                                                                 : channels == 3   ? PNG_COLOR_TYPE_RGB
                                                                                   : PNG_COLOR_TYPE_RGBA;
        png_set_IHDR(png, info, width, height, bit_depth, color_type, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_compression_level(png, std::clamp(compression_level, 0, 9));
        png_text text{};
        if (comment && !comment->empty()) {
            text.compression = PNG_TEXT_COMPRESSION_NONE;
            text.key = const_cast<png_charp>("Comment");
            text.text = const_cast<png_charp>(comment->c_str());
            png_set_text(png, info, &text, 1);
        }
        png_write_info(png, info);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        if (bit_depth == 16)
            png_set_swap(png);
#endif
        const auto row_bytes = static_cast<std::size_t>(width) * channels * (bit_depth / 8);
        std::vector<png_bytep> rows(height);
        auto* bytes = static_cast<png_bytep>(const_cast<void*>(data));
        for (int y = 0; y < height; ++y)
            rows[y] = bytes + static_cast<std::size_t>(y) * row_bytes;
        png_write_image(png, rows.data());
        png_write_end(png, info);
        context.file.close();
        png_destroy_write_struct(&png, &info);
        return true;
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
