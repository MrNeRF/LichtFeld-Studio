/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "image_heif.hpp"
#include "image_heic_system.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <lcms2.h>
#include <libheif/heif.h>
#include <libheif/heif_items.h>
#include <libheif/heif_properties.h>
#include <limits>
#include <memory>
#include <unordered_set>

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
            const std::filesystem::path* input_path = nullptr;
            std::span<const uint8_t> input_bytes;
            bool hevc = false;
            bool hdr = false;
            int orientation = 1;
            std::unique_ptr<heif_context, decltype(&heif_context_free)> context{nullptr, heif_context_free};
            std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> handle{nullptr, heif_image_handle_release};
            bool open(const std::filesystem::path& path, std::string& error) {
                input_path = &path;
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
                input_bytes = bytes;
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
                heif_item_id id = 0;
                if (!checked(heif_context_get_primary_image_ID(context.get(), &id), error))
                    return false;
                std::vector<heif_item_id> pending{id};
                std::unordered_set<heif_item_id> visited;
                bool av1 = false;
                bool grid = false;
                int hevc_bits = 0;
                while (!pending.empty()) {
                    const auto item = pending.back();
                    pending.pop_back();
                    if (!visited.insert(item).second)
                        continue;
                    const auto type = heif_item_get_item_type(context.get(), item);
                    if (type == heif_fourcc('a', 'v', '0', '1')) {
                        av1 = true;
                        continue;
                    }
                    if (type == heif_fourcc('h', 'v', 'c', '1') || type == heif_fourcc('h', 'e', 'v', '1')) {
                        heif_image_handle* raw = nullptr;
                        if (!checked(heif_context_get_image_handle(context.get(), item, &raw), error))
                            return false;
                        std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> leaf(raw, heif_image_handle_release);
                        const int bits = heif_image_handle_get_luma_bits_per_pixel(leaf.get());
                        if (bits < 1 || bits > 16 || (hevc_bits && bits != hevc_bits)) {
                            error = "HEIF: inconsistent HEVC tile precision";
                            return false;
                        }
                        hevc_bits = bits;
                        hevc = true;
                        continue;
                    }
                    if (type != heif_fourcc('g', 'r', 'i', 'd') && type != heif_fourcc('i', 'd', 'e', 'n')) {
                        error = "HEIF: unsupported primary image codec";
                        return false;
                    }
                    grid = grid || type == heif_fourcc('g', 'r', 'i', 'd');
                    for (int index = 0;; ++index) {
                        uint32_t reference_type = 0;
                        heif_item_id* raw = nullptr;
                        const auto count = heif_context_get_item_references(context.get(), item, index, &reference_type, &raw);
                        const auto release = [this](heif_item_id* p) { heif_release_item_references(context.get(), &p); };
                        std::unique_ptr<heif_item_id, decltype(release)> references(raw, release);
                        if (!count)
                            break;
                        if (reference_type == heif_fourcc('d', 'i', 'm', 'g'))
                            pending.insert(pending.end(), raw, raw + count);
                    }
                }
                if (av1 == hevc) {
                    error = "HEIF: a consistent AV1 or HEVC primary image is required";
                    return false;
                }
                if (hevc && hevc_bits != heif_image_handle_get_luma_bits_per_pixel(handle.get())) {
                    error = "HEIF: inconsistent HEVC primary precision";
                    return false;
                }
                if (hevc) {
                    heif_color_profile_nclx* declared = nullptr;
                    const auto status = heif_image_handle_get_nclx_color_profile(handle.get(), &declared);
                    std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> owner(declared, heif_nclx_color_profile_free);
                    hdr = status.code == heif_error_Ok && declared &&
                          (declared->transfer_characteristics == 16 || declared->transfer_characteristics == 18);
                }
#ifdef _WIN32
                if (hevc) {
                    bool rotation = false, mirror = false;
                    std::array<heif_property_id, 32> properties{};
                    const int count = heif_item_get_transformation_properties(context.get(), id, nullptr, 0);
                    if (count < 0 || count > static_cast<int>(properties.size()) ||
                        heif_item_get_transformation_properties(context.get(), id, properties.data(), count) != count) {
                        error = "HEIF: unsupported HEVC transformation list";
                        return false;
                    }
                    for (int i = 0; i < count; ++i) {
                        const auto type = heif_item_get_property_type(context.get(), id, properties[i]);
                        rotation |= type == heif_item_property_type_transform_rotation;
                        mirror |= type == heif_item_property_type_transform_mirror;
                    }
                    if (rotation && mirror) {
                        error = "HEIF: Windows HEVC system provider unsupported for combined rotation and mirroring";
                        return false;
                    }
                }
                if (hevc && grid) {
                    heif_colorspace colorspace;
                    heif_chroma chroma;
                    if (!checked(heif_image_handle_get_preferred_decoding_colorspace(handle.get(), &colorspace, &chroma), error))
                        return false;
                    // WIC reports successful copying for the qualified planar-RGB
                    // grid but produces incorrect pixels. Never accept that silently.
                    if (colorspace == heif_colorspace_RGB) {
                        error = "HEIF: Windows HEVC system provider unsupported for RGB grid images";
                        return false;
                    }
                }
#else
                (void)grid;
#endif
#ifdef __APPLE__
                if (hevc) {
                    // Use container transforms in libheif's specified order.
                    // ImageIO's generic Orientation may also describe EXIF,
                    // which is not an extra transform in the raw codec contract.
                    constexpr std::array<std::array<int, 4>, 8> matrices{{{1, 0, 0, 1}, {-1, 0, 0, 1}, {-1, 0, 0, -1}, {1, 0, 0, -1}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, -1, -1, 0}, {0, 1, -1, 0}}};
                    auto matrix = matrices[0];
                    bool cropped = false, transformed = false;
                    std::array<heif_property_id, 32> properties{};
                    const int count = heif_item_get_transformation_properties(context.get(), id, nullptr, 0);
                    if (count < 0 || count > static_cast<int>(properties.size()) ||
                        heif_item_get_transformation_properties(context.get(), id, properties.data(), count) != count) {
                        error = "HEIF: unsupported HEVC transformation list";
                        return false;
                    }
                    for (int i = 0; i < count; ++i) {
                        const auto property = properties[i];
                        const auto type = heif_item_get_property_type(context.get(), id, property);
                        if (type == heif_item_property_type_transform_crop) {
                            // ImageIO already applies the aperture before exposing
                            // unrotated pixels. Validate final dimensions in the adapter.
                            if (cropped || transformed) {
                                error = "HEIF: Apple HEVC system provider unsupported for this clean-aperture ordering";
                                return false;
                            }
                            cropped = true;
                            continue;
                        }
                        transformed = true;
                        int operation = 1;
                        if (type == heif_item_property_type_transform_rotation) {
                            const auto angle = heif_item_get_property_transform_rotation_ccw(context.get(), id, property);
                            operation = angle == 0 ? 1 : angle == 90 ? 8
                                                     : angle == 180  ? 3
                                                     : angle == 270  ? 6
                                                                     : 0;
                        } else if (type == heif_item_property_type_transform_mirror) {
                            const auto direction = heif_item_get_property_transform_mirror(context.get(), id, property);
                            operation = direction == heif_transform_mirror_direction_horizontal ? 2 : direction == heif_transform_mirror_direction_vertical ? 4
                                                                                                                                                            : 0;
                        } else {
                            error = "HEIF: unsupported HEVC container transform";
                            return false;
                        }
                        if (!operation) {
                            error = "HEIF: invalid HEVC container transform";
                            return false;
                        }
                        const auto& op = matrices[operation - 1];
                        matrix = {op[0] * matrix[0] + op[1] * matrix[2], op[0] * matrix[1] + op[1] * matrix[3],
                                  op[2] * matrix[0] + op[3] * matrix[2], op[2] * matrix[1] + op[3] * matrix[3]};
                    }
                    const auto found = std::find(matrices.begin(), matrices.end(), matrix);
                    if (found == matrices.end()) {
                        error = "HEIF: invalid HEVC composed transform";
                        return false;
                    }
                    orientation = static_cast<int>(found - matrices.begin()) + 1;
                }
#endif
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
        using Transform = std::unique_ptr<void, decltype(&cmsDeleteTransform)>;
        struct DisplayKey {
            std::vector<uint8_t> icc;
            // White point followed by red, green and blue chromaticities.
            std::array<double, 8> primaries{};
            int transfer = 0;
            cmsUInt32Number format = 0;
            bool operator==(const DisplayKey&) const = default;
        };
        struct DisplayTransform {
            DisplayKey key;
            Transform transform{nullptr, cmsDeleteTransform};
        };
        bool to_display(const heif_image_handle* handle, Image& image, std::string& error, bool apply = true, bool raw_available = true) {
            const bool wide = image.sample_type == SampleType::UInt16;
            DisplayKey key;
            key.format = image.channels == 4 ? (wide ? TYPE_RGBA_16 : TYPE_RGBA_8) : (wide ? TYPE_RGB_16 : TYPE_RGB_8);
            const auto profile_size = heif_image_handle_get_raw_color_profile_size(handle);
            if (profile_size) {
                if (profile_size > 4 * 1024 * 1024) {
                    error = "HEIF: ICC profile exceeds import limit";
                    return false;
                }
                key.icc.resize(profile_size);
                if (!checked(heif_image_handle_get_raw_color_profile(handle, key.icc.data()), error))
                    return false;
            } else {
                heif_color_profile_nclx* declared = nullptr;
                const auto status = heif_image_handle_get_nclx_color_profile(handle, &declared);
                std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> nclx(declared, heif_nclx_color_profile_free);
                if (status.code == heif_error_Color_profile_does_not_exist)
                    return true; // Untagged RGB follows the existing sRGB assumption.
                if (!checked(status, error))
                    return false;
                if (nclx->transfer_characteristics == 2 && nclx->color_primaries == 2)
                    return true;
                const int transfer = nclx->transfer_characteristics;
                // PQ and HLG have different physical units; never label their samples SDR.
                if (transfer == 16 || transfer == 18) {
                    error = "HEIF: HDR display import requires an explicit conversion to SDR";
                    if (raw_available)
                        error += "; raw decoding remains available";
                    return false;
                }
                if (transfer != 1 && transfer != 6 && transfer != 13 && transfer != 8) {
                    error = "HEIF: display import requires a supported SDR transfer";
                    if (raw_available)
                        error += "; raw decoding remains available";
                    return false;
                }
                key.transfer = transfer;
                key.primaries = {nclx->color_primary_white_x, nclx->color_primary_white_y,
                                 nclx->color_primary_red_x, nclx->color_primary_red_y,
                                 nclx->color_primary_green_x, nclx->color_primary_green_y,
                                 nclx->color_primary_blue_x, nclx->color_primary_blue_y};
                if (key.primaries[1] <= 0 || key.primaries[3] <= 0 || key.primaries[5] <= 0 || key.primaries[7] <= 0) {
                    error = "HEIF: unsupported color primaries";
                    return false;
                }
            }
            // One entry per worker bounds retention and avoids sharing LCMS's mutable
            // transform state across threads. Layout/precision are part of the key.
            thread_local DisplayTransform cached;
            if (cached.transform && cached.key == key) {
                if (apply)
                    cmsDoTransform(cached.transform.get(), image.data.data(), image.data.data(), static_cast<cmsUInt32Number>(image.width * image.height));
                return true;
            }
            Profile source{nullptr, cmsCloseProfile};
            if (!key.icc.empty()) {
                source.reset(cmsOpenProfileFromMem(key.icc.data(), static_cast<cmsUInt32Number>(key.icc.size())));
            } else {
                const auto& p = key.primaries;
                cmsCIExyY white{p[0], p[1], 1};
                cmsCIExyYTRIPLE primaries{{p[2], p[3], 1}, {p[4], p[5], 1}, {p[6], p[7], 1}};
                // Analytic parametric curves avoid rebuilding a lookup table per image.
                const double srgb[]{2.4, 1. / 1.055, .055 / 1.055, 1. / 12.92, .04045};
                const double bt709[]{1. / .45, 1. / 1.099, .099 / 1.099, 1. / 4.5, .081};
                std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)> curve(
                    key.transfer == 8 ? cmsBuildGamma(nullptr, 1.) : cmsBuildParametricToneCurve(nullptr, 4, key.transfer == 13 ? srgb : bt709), cmsFreeToneCurve);
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
            Transform transform(cmsCreateTransform(source.get(), key.format, destination.get(), key.format, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_COPY_ALPHA), cmsDeleteTransform);
            if (!transform) {
                error = "HEIF: cannot create display color transform";
                return false;
            }
            cached.key = std::move(key);
            cached.transform = std::move(transform);
            if (apply)
                cmsDoTransform(cached.transform.get(), image.data.data(), image.data.data(), static_cast<cmsUInt32Number>(image.width * image.height));
            return true;
        }
    } // namespace
    bool probe_heif(const std::filesystem::path& path, Probe& result, std::string& error) {
        result = {};
        Source source;
        Probe info;
        if (!source.open(path, error) || !layout(source.handle.get(), info, error))
            return false;
        if (source.hevc) {
            if (!heic_system_available(error))
                return false;
            Image metadata{info.width, info.height, info.channels, info.sample_type, {}};
            if (source.hdr && !to_display(source.handle.get(), metadata, error, false, false))
                return false;
            if (!read_heic_system(source.input_path, source.input_bytes, info,
                                  heif_image_handle_get_luma_bits_per_pixel(source.handle.get()), source.orientation, source.hdr, nullptr, error))
                return false;
        }
        result = info;
        return true;
    }
    namespace {
        bool decode_source(Source& source, Image& result, std::string& error, bool display) {
            Probe info;
            if (!layout(source.handle.get(), info, error))
                return false;
            if (source.hevc) {
                if (!heic_system_available(error))
                    return false;
                Image image{info.width, info.height, info.channels, info.sample_type, {}};
                // Validate shared ICC/HDR policy before an OS decoder can select
                // an HDR layout or obscure an invalid ICC with its own failure.
                if ((display || source.hdr) && !to_display(source.handle.get(), image, error, false, false))
                    return false;
                if (!read_heic_system(source.input_path, source.input_bytes, info,
                                      heif_image_handle_get_luma_bits_per_pixel(source.handle.get()), source.orientation, source.hdr, &image, error) ||
                    (display && !to_display(source.handle.get(), image, error, true, false)))
                    return false;
                result = std::move(image);
                return true;
            }
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
            options->decoder_id = "dav1d";
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
