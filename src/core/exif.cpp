/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/exif.hpp"
#include "core/path_utils.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <span>
#include <vector>

namespace lfs::core {
    namespace {

        constexpr std::size_t kMaxScanBytes = 64 * 1024;
        constexpr std::uint16_t kTagExifIfd = 0x8769;
        constexpr std::uint16_t kTagExposureTime = 0x829A;
        constexpr std::uint16_t kTagFNumber = 0x829D;
        constexpr std::uint16_t kTagIso = 0x8827;
        constexpr std::uint16_t kTagStandardOutputSensitivity = 0x8831;
        constexpr std::uint16_t kTagRecommendedExposureIndex = 0x8832;
        constexpr std::uint16_t kTagIsoSpeed = 0x8833;
        constexpr std::uint16_t kTagShutterSpeedValue = 0x9201;
        constexpr std::uint16_t kTagApertureValue = 0x9202;

        constexpr std::uint16_t kTypeShort = 3;
        constexpr std::uint16_t kTypeLong = 4;
        constexpr std::uint16_t kTypeRational = 5;
        constexpr std::uint16_t kTypeSLong = 9;
        constexpr std::uint16_t kTypeSRational = 10;

        constexpr int kMaxIfdEntries = 256;

        [[nodiscard]] bool fits(const std::span<const std::uint8_t> data, const std::size_t offset,
                                const std::size_t nbytes) {
            return nbytes <= data.size() && offset <= data.size() - nbytes;
        }

        [[nodiscard]] std::optional<std::uint16_t> read_u16(const std::span<const std::uint8_t> data,
                                                            const std::size_t offset, const bool little) {
            if (!fits(data, offset, 2)) {
                return std::nullopt;
            }
            const std::uint16_t lo = data[offset];
            const std::uint16_t hi = data[offset + 1];
            return little ? static_cast<std::uint16_t>(lo | (hi << 8))
                          : static_cast<std::uint16_t>(hi | (lo << 8));
        }

        [[nodiscard]] std::optional<std::uint32_t> read_u32(const std::span<const std::uint8_t> data,
                                                            const std::size_t offset, const bool little) {
            if (!fits(data, offset, 4)) {
                return std::nullopt;
            }
            if (little) {
                return static_cast<std::uint32_t>(data[offset]) |
                       (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
                       (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
                       (static_cast<std::uint32_t>(data[offset + 3]) << 24);
            }
            return (static_cast<std::uint32_t>(data[offset]) << 24) |
                   (static_cast<std::uint32_t>(data[offset + 1]) << 16) |
                   (static_cast<std::uint32_t>(data[offset + 2]) << 8) |
                   static_cast<std::uint32_t>(data[offset + 3]);
        }

        [[nodiscard]] std::optional<std::int32_t> read_i32(const std::span<const std::uint8_t> data,
                                                           const std::size_t offset, const bool little) {
            const auto raw = read_u32(data, offset, little);
            if (!raw) {
                return std::nullopt;
            }
            return static_cast<std::int32_t>(*raw);
        }

        [[nodiscard]] std::size_t tiff_type_size(const std::uint16_t type) {
            switch (type) {
            case 1:
            case 2:
            case 7:
                return 1;
            case kTypeShort:
            case 8:
                return 2;
            case kTypeLong:
            case kTypeSLong:
            case 11:
                return 4;
            case kTypeRational:
            case kTypeSRational:
            case 12:
                return 8;
            default:
                return 0;
            }
        }

        struct IfdEntry {
            std::uint16_t tag = 0;
            std::uint16_t type = 0;
            std::uint32_t count = 0;
            std::uint32_t value_or_offset = 0;
        };

        [[nodiscard]] std::optional<IfdEntry> read_ifd_entry(const std::span<const std::uint8_t> tiff,
                                                             const std::size_t offset, const bool little) {
            const auto tag = read_u16(tiff, offset, little);
            const auto type = read_u16(tiff, offset + 2, little);
            const auto count = read_u32(tiff, offset + 4, little);
            const auto value = read_u32(tiff, offset + 8, little);
            if (!tag || !type || !count || !value) {
                return std::nullopt;
            }
            return IfdEntry{*tag, *type, *count, *value};
        }

        [[nodiscard]] std::optional<std::size_t> entry_value_offset(const IfdEntry& entry,
                                                                    const std::size_t entry_offset) {
            const std::size_t elem = tiff_type_size(entry.type);
            if (elem == 0 || entry.count == 0) {
                return std::nullopt;
            }
            if (entry.count > (std::numeric_limits<std::size_t>::max() / elem)) {
                return std::nullopt;
            }
            const std::size_t nbytes = elem * static_cast<std::size_t>(entry.count);
            if (nbytes <= 4) {
                return entry_offset + 8;
            }
            return static_cast<std::size_t>(entry.value_or_offset);
        }

        [[nodiscard]] bool usable_positive(const double value) {
            return std::isfinite(value) && value > 0.0;
        }

        [[nodiscard]] std::optional<double> read_unsigned_number(const std::span<const std::uint8_t> tiff,
                                                                 const IfdEntry& entry,
                                                                 const std::size_t entry_offset,
                                                                 const bool little) {
            const auto data_off = entry_value_offset(entry, entry_offset);
            if (!data_off) {
                return std::nullopt;
            }
            switch (entry.type) {
            case kTypeShort: {
                const auto v = read_u16(tiff, *data_off, little);
                return v ? std::optional<double>{static_cast<double>(*v)} : std::nullopt;
            }
            case kTypeLong: {
                const auto v = read_u32(tiff, *data_off, little);
                return v ? std::optional<double>{static_cast<double>(*v)} : std::nullopt;
            }
            case kTypeRational: {
                const auto num = read_u32(tiff, *data_off, little);
                const auto den = read_u32(tiff, *data_off + 4, little);
                if (!num || !den || *den == 0) {
                    return std::nullopt;
                }
                return static_cast<double>(*num) / static_cast<double>(*den);
            }
            default:
                return std::nullopt;
            }
        }

        [[nodiscard]] std::optional<double> read_signed_number(const std::span<const std::uint8_t> tiff,
                                                               const IfdEntry& entry,
                                                               const std::size_t entry_offset,
                                                               const bool little) {
            const auto data_off = entry_value_offset(entry, entry_offset);
            if (!data_off) {
                return std::nullopt;
            }
            switch (entry.type) {
            case kTypeSLong: {
                const auto v = read_i32(tiff, *data_off, little);
                return v ? std::optional<double>{static_cast<double>(*v)} : std::nullopt;
            }
            case kTypeSRational: {
                const auto num = read_i32(tiff, *data_off, little);
                const auto den = read_i32(tiff, *data_off + 4, little);
                if (!num || !den || *den == 0) {
                    return std::nullopt;
                }
                return static_cast<double>(*num) / static_cast<double>(*den);
            }
            case kTypeRational:
            case kTypeShort:
            case kTypeLong:
                return read_unsigned_number(tiff, entry, entry_offset, little);
            default:
                return std::nullopt;
            }
        }

        struct ExposureTags {
            std::optional<double> exposure_time;
            std::optional<double> f_number;
            std::optional<double> iso;
            std::optional<double> iso_sos;
            std::optional<double> iso_rei;
            std::optional<double> iso_speed;
            std::optional<double> shutter_speed_apex;
            std::optional<double> aperture_apex;
            std::optional<std::uint32_t> exif_ifd_offset;
        };

        void ingest_entry(ExposureTags& tags, const std::span<const std::uint8_t> tiff, const IfdEntry& entry,
                          const std::size_t entry_offset, const bool little) {
            switch (entry.tag) {
            case kTagExifIfd: {
                if (entry.type == kTypeLong || entry.type == kTypeShort) {
                    const auto off = read_unsigned_number(tiff, entry, entry_offset, little);
                    if (off && *off <= static_cast<double>(std::numeric_limits<std::uint32_t>::max()) &&
                        *off >= 0.0) {
                        tags.exif_ifd_offset = static_cast<std::uint32_t>(*off);
                    }
                }
                break;
            }
            case kTagExposureTime:
                tags.exposure_time = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagFNumber:
                tags.f_number = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagIso:
                tags.iso = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagStandardOutputSensitivity:
                tags.iso_sos = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagRecommendedExposureIndex:
                tags.iso_rei = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagIsoSpeed:
                tags.iso_speed = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            case kTagShutterSpeedValue:
                tags.shutter_speed_apex = read_signed_number(tiff, entry, entry_offset, little);
                break;
            case kTagApertureValue:
                tags.aperture_apex = read_unsigned_number(tiff, entry, entry_offset, little);
                break;
            default:
                break;
            }
        }

        bool parse_ifd(const std::span<const std::uint8_t> tiff, const std::uint32_t ifd_offset, const bool little,
                       ExposureTags& tags) {
            const auto count = read_u16(tiff, static_cast<std::size_t>(ifd_offset), little);
            if (!count || *count > kMaxIfdEntries) {
                return false;
            }
            if (*count == 0) {
                return true;
            }
            const std::size_t entries_off = static_cast<std::size_t>(ifd_offset) + 2;
            if (!fits(tiff, entries_off, static_cast<std::size_t>(*count) * 12)) {
                return false;
            }
            for (std::uint16_t i = 0; i < *count; ++i) {
                const std::size_t entry_off = entries_off + static_cast<std::size_t>(i) * 12;
                const auto entry = read_ifd_entry(tiff, entry_off, little);
                if (!entry) {
                    return false;
                }
                ingest_entry(tags, tiff, *entry, entry_off, little);
            }
            return true;
        }

        [[nodiscard]] std::optional<double> ev_from_tags(const ExposureTags& tags) {
            std::optional<double> t;
            if (usable_positive(tags.exposure_time.value_or(0.0))) {
                t = tags.exposure_time;
            } else if (tags.shutter_speed_apex && std::isfinite(*tags.shutter_speed_apex)) {
                const double time = std::exp2(-*tags.shutter_speed_apex);
                if (usable_positive(time)) {
                    t = time;
                }
            }

            std::optional<double> n;
            if (usable_positive(tags.f_number.value_or(0.0))) {
                n = tags.f_number;
            } else if (tags.aperture_apex && std::isfinite(*tags.aperture_apex)) {
                const double fno = std::exp2(*tags.aperture_apex / 2.0);
                if (usable_positive(fno)) {
                    n = fno;
                }
            }

            std::optional<double> iso;
            for (const auto& candidate : {tags.iso, tags.iso_sos, tags.iso_rei, tags.iso_speed}) {
                if (usable_positive(candidate.value_or(0.0))) {
                    iso = candidate;
                    break;
                }
            }

            if (!t && !n && !iso) {
                return std::nullopt;
            }
            const double time = t.value_or(1.0);
            const double fno = n.value_or(1.0);
            const double speed = iso.value_or(1.0);
            const double ev = std::log2(time / (fno * fno) * speed);
            if (!std::isfinite(ev)) {
                return std::nullopt;
            }
            return ev;
        }

        [[nodiscard]] std::optional<double> parse_tiff_ev(const std::span<const std::uint8_t> tiff) {
            if (tiff.size() < 8) {
                return std::nullopt;
            }
            const bool little = tiff[0] == 'I' && tiff[1] == 'I';
            const bool big = tiff[0] == 'M' && tiff[1] == 'M';
            if (!little && !big) {
                return std::nullopt;
            }
            const auto magic = read_u16(tiff, 2, little);
            if (!magic || *magic != 42) {
                return std::nullopt;
            }
            const auto ifd0 = read_u32(tiff, 4, little);
            if (!ifd0) {
                return std::nullopt;
            }

            ExposureTags tags;
            if (!parse_ifd(tiff, *ifd0, little, tags)) {
                return std::nullopt;
            }
            if (tags.exif_ifd_offset) {
                if (!parse_ifd(tiff, *tags.exif_ifd_offset, little, tags)) {
                    return std::nullopt;
                }
            }
            return ev_from_tags(tags);
        }

        [[nodiscard]] std::optional<double> parse_jpeg_ev(const std::span<const std::uint8_t> jpeg) {
            if (jpeg.size() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
                return std::nullopt;
            }

            std::size_t i = 2;
            while (i < jpeg.size()) {
                if (jpeg[i] != 0xFF) {
                    return std::nullopt;
                }
                while (i < jpeg.size() && jpeg[i] == 0xFF) {
                    ++i;
                }
                if (i >= jpeg.size()) {
                    return std::nullopt;
                }
                const std::uint8_t marker = jpeg[i++];
                if (marker == 0xD9 || marker == 0xDA) {
                    return std::nullopt;
                }
                if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7)) {
                    continue;
                }
                const auto length = read_u16(jpeg, i, false);
                if (!length || *length < 2) {
                    return std::nullopt;
                }
                const std::size_t payload_off = i + 2;
                const std::size_t payload_len = static_cast<std::size_t>(*length) - 2;
                if (!fits(jpeg, payload_off, payload_len)) {
                    return std::nullopt;
                }
                if (marker == 0xE1 && payload_len >= 8) {
                    const auto payload = jpeg.subspan(payload_off, payload_len);
                    if (payload[0] == 'E' && payload[1] == 'x' && payload[2] == 'i' && payload[3] == 'f' &&
                        payload[4] == 0 && payload[5] == 0) {
                        return parse_tiff_ev(payload.subspan(6));
                    }
                }
                i = payload_off + payload_len;
            }
            return std::nullopt;
        }

    } // namespace

    std::optional<double> exif_exposure_ev(const std::filesystem::path& image) {
        std::ifstream file;
        if (!open_file_for_read(image, std::ios::in | std::ios::binary, file)) {
            return std::nullopt;
        }
        std::vector<std::uint8_t> buffer(kMaxScanBytes);
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        const auto nread = static_cast<std::size_t>(file.gcount());
        if (nread < 2) {
            return std::nullopt;
        }
        buffer.resize(nread);
        return parse_jpeg_ev(buffer);
    }

    std::optional<double> exif_exposure_ev_for_training_image(const std::filesystem::path& image_path,
                                                              const std::filesystem::path& dataset_root) {
        if (auto ev = exif_exposure_ev(image_path)) {
            return ev;
        }
        if (image_path.empty() || image_path.filename().empty()) {
            return std::nullopt;
        }
        const auto fallback = dataset_root / "images" / image_path.filename();
        if (fallback == image_path) {
            return std::nullopt;
        }
        return exif_exposure_ev(fallback);
    }

} // namespace lfs::core
