/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_heif.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <lcms2.h>
#include <libheif/heif.h>
#include <limits>
#include <memory>

namespace lfs::core::image_codecs {
    namespace {
        bool checked(heif_error status, std::string& error) {
            if (status.code == heif_error_Ok)
                return true;
            error = "HEIF: " + std::string(status.message ? status.message : "codec failure");
            return false;
        }
        struct Source {
            // Keep the reader alive until the context and primary handle have been released.
            std::ifstream file;
            int64_t size = 0;
            std::unique_ptr<heif_context, decltype(&heif_context_free)> context{nullptr, heif_context_free};
            std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> handle{nullptr, heif_image_handle_release};
            bool open(const std::filesystem::path& path, std::string& error) {
                file.open(path, std::ios::binary | std::ios::ate);
                if (!file || file.tellg() < 0) {
                    error = "HEIF: cannot open source";
                    return false;
                }
                size = static_cast<int64_t>(file.tellg());
                file.seekg(0);
                context.reset(heif_context_alloc());
                if (!context) {
                    error = "HEIF: cannot allocate context";
                    return false;
                }
                static const heif_reader reader{
                    .reader_api_version = 1,
                    .get_position = [](void* user) -> int64_t { return static_cast<int64_t>(static_cast<Source*>(user)->file.tellg()); },
                    .read = [](void* data, size_t size, void* user) -> int {
                        if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
                            return 1;
                        auto& file = static_cast<Source*>(user)->file;
                        file.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
                        return file ? 0 : 1;
                    },
                    .seek = [](int64_t position, void* user) -> int {
                        auto& source = *static_cast<Source*>(user);
                        if (position < 0 || position > source.size)
                            return 1;
                        source.file.clear();
                        source.file.seekg(position);
                        return source.file ? 0 : 1;
                    },
                    .wait_for_file_size = [](int64_t target, void* user) -> heif_reader_grow_status {
                        return target >= 0 && target <= static_cast<Source*>(user)->size
                                   ? heif_reader_grow_status_size_reached
                                   : heif_reader_grow_status_size_beyond_eof;
                    }};
                if (!checked(heif_context_read_from_reader(context.get(), &reader, this, nullptr), error))
                    return false;
                return primary_image(error);
            }
            bool open(std::span<const uint8_t> bytes, std::string& error) {
                if (bytes.empty()) {
                    error = "HEIF: empty input";
                    return false;
                }
                context.reset(heif_context_alloc());
                if (!context) {
                    error = "HEIF: cannot allocate context";
                    return false;
                }
                if (!checked(heif_context_read_from_memory_without_copy(context.get(), bytes.data(), bytes.size(), nullptr), error))
                    return false;
                return primary_image(error);
            }
            bool primary_image(std::string& error) {
                heif_image_handle* primary = nullptr;
                if (!checked(heif_context_get_primary_image_handle(context.get(), &primary), error))
                    return false;
                handle.reset(primary);
                return true;
            }
        };
        bool layout(const heif_image_handle* handle, Probe& result, std::string& error) {
            const int width = heif_image_handle_get_width(handle), height = heif_image_handle_get_height(handle);
            const int bits = heif_image_handle_get_luma_bits_per_pixel(handle);
            const int channels = heif_image_handle_has_alpha_channel(handle) ? 4 : 3;
            if (width <= 0 || height <= 0 || bits < 1 || bits > 16 ||
                static_cast<uint64_t>(width) * height > (uint64_t{1} << 28)) {
                error = "HEIF: unsupported dimensions or sample precision";
                return false;
            }
            result = {width, height, channels, bits > 8 ? SampleType::UInt16 : SampleType::UInt8};
            return true;
        }
        using Profile = std::unique_ptr<void, decltype(&cmsCloseProfile)>;
        bool to_display(const heif_image_handle* handle, Image& image, std::string& error) {
            heif_color_profile_nclx* declared = nullptr;
            const auto declared_status = heif_image_handle_get_nclx_color_profile(handle, &declared);
            std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> declared_profile(declared, heif_nclx_color_profile_free);
            if (declared_status.code == heif_error_Ok &&
                (declared->transfer_characteristics == 16 || declared->transfer_characteristics == 18)) {
                error = "HEIF: HDR display import requires an explicit conversion to SDR; raw decoding remains available";
                return false;
            }
            Profile source{nullptr, cmsCloseProfile};
            const auto profile_size = heif_image_handle_get_raw_color_profile_size(handle);
            if (profile_size) {
                if (profile_size > 4 * 1024 * 1024) {
                    error = "HEIF: ICC profile exceeds import limit";
                    return false;
                }
                std::vector<uint8_t> bytes(profile_size);
                if (!checked(heif_image_handle_get_raw_color_profile(handle, bytes.data()), error))
                    return false;
                source.reset(cmsOpenProfileFromMem(bytes.data(), static_cast<cmsUInt32Number>(bytes.size())));
            } else {
                const auto status = declared_status;
                const auto& nclx = declared_profile;
                if (status.code == heif_error_Color_profile_does_not_exist)
                    return true; // Untagged RGB follows the existing sRGB assumption.
                if (!checked(status, error))
                    return false;
                if (nclx->transfer_characteristics == 2 && nclx->color_primaries == 2)
                    return true;
                const int transfer = nclx->transfer_characteristics;
                // PQ and HLG have different physical units; never label their samples SDR.
                if (transfer != 1 && transfer != 6 && transfer != 13 && transfer != 8) {
                    error = "HEIF: display import requires a supported SDR transfer; raw decoding remains available";
                    return false;
                }
                cmsCIExyY white{nclx->color_primary_white_x, nclx->color_primary_white_y, 1};
                cmsCIExyYTRIPLE primaries{{nclx->color_primary_red_x, nclx->color_primary_red_y, 1},
                                          {nclx->color_primary_green_x, nclx->color_primary_green_y, 1},
                                          {nclx->color_primary_blue_x, nclx->color_primary_blue_y, 1}};
                if (white.y <= 0 || primaries.Red.y <= 0 || primaries.Green.y <= 0 || primaries.Blue.y <= 0) {
                    error = "HEIF: unsupported color primaries";
                    return false;
                }
                // Analytic parametric curves avoid rebuilding a lookup table per image.
                const double srgb[]{2.4, 1. / 1.055, .055 / 1.055, 1. / 12.92, .04045};
                const double bt709[]{1. / .45, 1. / 1.099, .099 / 1.099, 1. / 4.5, .081};
                std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)> curve(
                    transfer == 8 ? cmsBuildGamma(nullptr, 1.) : cmsBuildParametricToneCurve(nullptr, 4, transfer == 13 ? srgb : bt709), cmsFreeToneCurve);
                if (!curve) {
                    error = "HEIF: cannot create transfer curve";
                    return false;
                }
                cmsToneCurve* curves[]{curve.get(), curve.get(), curve.get()};
                source.reset(cmsCreateRGBProfile(&white, &primaries, curves));
            }
            if (!source || cmsGetColorSpace(source.get()) != cmsSigRgbData) {
                error = "HEIF: invalid or unsupported RGB color profile";
                return false;
            }
            Profile destination(cmsCreate_sRGBProfile(), cmsCloseProfile);
            if (!destination) {
                error = "HEIF: cannot create sRGB profile";
                return false;
            }
            const bool wide = image.sample_type == SampleType::UInt16;
            const auto format = image.channels == 4 ? (wide ? TYPE_RGBA_16 : TYPE_RGBA_8) : (wide ? TYPE_RGB_16 : TYPE_RGB_8);
            std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(cmsCreateTransform(source.get(), format, destination.get(), format, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_COPY_ALPHA), cmsDeleteTransform);
            if (!transform) {
                error = "HEIF: cannot create display color transform";
                return false;
            }
            cmsDoTransform(transform.get(), image.data.data(), image.data.data(), static_cast<cmsUInt32Number>(image.width * image.height));
            return true;
        }
    } // namespace
    bool probe_heif(const std::filesystem::path& path, Probe& result, std::string& error) {
        result = {};
        Source source;
        return source.open(path, error) && layout(source.handle.get(), result, error);
    }
    namespace {
        bool decode_source(Source& source, Image& result, std::string& error, bool display) {
            Probe info;
            if (!layout(source.handle.get(), info, error))
                return false;
            const bool wide = info.sample_type == SampleType::UInt16;
            const bool alpha = info.channels == 4;
            const auto chroma = wide ? (std::endian::native == std::endian::little ? (alpha ? heif_chroma_interleaved_RRGGBBAA_LE : heif_chroma_interleaved_RRGGBB_LE)
                                                                                   : (alpha ? heif_chroma_interleaved_RRGGBBAA_BE : heif_chroma_interleaved_RRGGBB_BE))
                                     : (alpha ? heif_chroma_interleaved_RGBA : heif_chroma_interleaved_RGB);
            std::unique_ptr<heif_decoding_options, decltype(&heif_decoding_options_free)> options(heif_decoding_options_alloc(), heif_decoding_options_free);
            if (!options) {
                error = "HEIF: cannot allocate decoding options";
                return false;
            }
            options->convert_hdr_to_8bit = 0;
            options->strict_decoding = 1;
            heif_image* raw = nullptr;
            if (!checked(heif_decode_image(source.handle.get(), &raw, heif_colorspace_RGB, chroma, options.get()), error))
                return false;
            std::unique_ptr<heif_image, decltype(&heif_image_release)> decoded(raw, heif_image_release);
            const int width = heif_image_get_width(raw, heif_channel_interleaved), height = heif_image_get_height(raw, heif_channel_interleaved);
            const int bits = heif_image_get_bits_per_pixel_range(raw, heif_channel_interleaved);
            int stride = 0;
            const auto* plane = heif_image_get_plane_readonly(raw, heif_channel_interleaved, &stride);
            const size_t row_bytes = static_cast<size_t>(width) * info.channels * (wide ? 2 : 1);
            if (!plane || width != info.width || height != info.height || bits < 1 || bits > 16 || stride < 0 || static_cast<size_t>(stride) < row_bytes) {
                error = "HEIF: inconsistent decoded layout";
                return false;
            }
            Image image{width, height, info.channels, info.sample_type, std::vector<uint8_t>(row_bytes * height)};
            for (int y = 0; y < height; ++y)
                std::memcpy(image.data.data() + row_bytes * y, plane + static_cast<size_t>(stride) * y, row_bytes);
            // libheif stores high-bitdepth integer samples right-aligned. The core
            // UINT16 contract spans [0,65535], including alpha.
            if (wide && bits < 16) {
                const uint32_t maximum = (uint32_t{1} << bits) - 1;
                for (size_t i = 0; i < image.data.size(); i += 2) {
                    uint16_t value;
                    std::memcpy(&value, image.data.data() + i, 2);
                    value = static_cast<uint16_t>((uint32_t(value) * 65535 + maximum / 2) / maximum);
                    std::memcpy(image.data.data() + i, &value, 2);
                }
            }
            if (display && alpha && heif_image_handle_is_premultiplied_alpha(source.handle.get())) {
                const uint32_t maximum = wide ? 65535 : 255;
                const size_t step = wide ? 2 : 1;
                for (size_t pixel = 0; pixel < static_cast<size_t>(width) * height; ++pixel) {
                    const auto read = [&](size_t channel) -> uint32_t {
                        const auto* p = image.data.data() + (pixel * 4 + channel) * step;
                        if (!wide)
                            return *p;
                        uint16_t value;
                        std::memcpy(&value, p, 2);
                        return value;
                    };
                    const auto a = read(3);
                    for (size_t channel = 0; channel < 3; ++channel) {
                        const uint32_t v = a ? std::min(maximum, (read(channel) * maximum + a / 2) / a) : 0;
                        auto* p = image.data.data() + (pixel * 4 + channel) * step;
                        if (wide) {
                            const auto value = static_cast<uint16_t>(v);
                            std::memcpy(p, &value, 2);
                        } else
                            *p = static_cast<uint8_t>(v);
                    }
                }
            }
            if (display && !to_display(source.handle.get(), image, error))
                return false;
            result = std::move(image);
            return true;
        }
        bool copy_target(const Image& image, DecodeTarget& target, Probe& info, std::string& error) {
            if ((target.channels != 3 && target.channels != 4) ||
                (target.sample_type != SampleType::UInt8 && target.sample_type != SampleType::UInt16 && target.sample_type != SampleType::Float32)) {
                error = "HEIF: unsupported output target";
                return false;
            }
            const size_t pixels = static_cast<size_t>(image.width) * image.height;
            const size_t step = target.sample_type == SampleType::UInt8 ? 1 : target.sample_type == SampleType::UInt16 ? 2
                                                                                                                       : 4;
            const size_t bytes = pixels * target.channels * step;
            if (!target.data)
                target.data = target.allocate ? target.allocate(bytes, target.user) : std::malloc(bytes);
            if (!target.data) {
                error = "HEIF: cannot allocate output target";
                return false;
            }
            for (size_t pixel = 0; pixel < pixels; ++pixel)
                for (int channel = 0; channel < target.channels; ++channel) {
                    uint32_t value = 65535;
                    if (channel < image.channels) {
                        const size_t index = pixel * image.channels + channel;
                        if (image.sample_type == SampleType::UInt8)
                            value = image.data[index] * 257u;
                        else {
                            uint16_t sample;
                            std::memcpy(&sample, image.data.data() + index * 2, 2);
                            value = sample;
                        }
                    }
                    const size_t index = pixel * target.channels + channel;
                    if (target.sample_type == SampleType::UInt8)
                        static_cast<uint8_t*>(target.data)[index] = static_cast<uint8_t>((value + 128) / 257);
                    else if (target.sample_type == SampleType::UInt16)
                        static_cast<uint16_t*>(target.data)[index] = static_cast<uint16_t>(value);
                    else
                        static_cast<float*>(target.data)[index] = value / 65535.f;
                }
            info = {image.width, image.height, target.channels, target.sample_type};
            return true;
        }
        template <class Input>
        bool decode_target(Input input, DecodeTarget& target, Probe& info, std::string& error, bool display) {
            info = {};
            Image image;
            return decode_heif(input, image, error, display) && copy_target(image, target, info, error);
        }
    } // namespace
    bool is_heif(std::span<const uint8_t> bytes) {
        if (bytes.size() < 12)
            return false;
        const auto type = heif_check_filetype(bytes.data(), static_cast<int>(std::min(bytes.size(), size_t{4096})));
        return type == heif_filetype_yes_supported || type == heif_filetype_yes_unsupported;
    }
    bool decode_heif(const std::filesystem::path& path, Image& result, std::string& error, bool display) {
        result = {};
        Source source;
        return source.open(path, error) && decode_source(source, result, error, display);
    }
    bool decode_heif(std::span<const uint8_t> bytes, Image& result, std::string& error, bool display) {
        result = {};
        Source source;
        return source.open(bytes, error) && decode_source(source, result, error, display);
    }
    bool decode_heif_to_buffer(const std::filesystem::path& path, DecodeTarget& target, Probe& info, std::string& error, bool display) {
        return decode_target(path, target, info, error, display);
    }
    bool decode_heif_to_buffer(std::span<const uint8_t> bytes, DecodeTarget& target, Probe& info, std::string& error, bool display) {
        return decode_target(bytes, target, info, error, display);
    }
} // namespace lfs::core::image_codecs
