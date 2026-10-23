// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/source_exif.hpp"
#include "media/media_ingest.hpp"
#include "media/source_exif_internal.hpp"
extern "C" {
#include <libavcodec/exif.h>
}
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>

namespace lfs::media {
    namespace {
        constexpr std::array<size_t, 14> widths{0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4};
        struct Tag {
            uint16_t id, type;
            uint32_t count, child = 0;
        };
        struct Directory {
            uint32_t offset = 0;
            std::vector<Tag> tags;
            std::vector<uint32_t> path;
        };
        // A bounds/provenance guard, not a second value decoder. It counts all
        // allocations FFmpeg could request, rejects revisited directories and
        // keeps MakerNotes opaque in a private parse copy. Originals never change.
        struct Guard {
            std::vector<uint8_t> data;
            bool little = true;
            ExifParseLimits limits;
            size_t entries = 0, values = 0;
            std::set<uint32_t> visited;
            std::vector<Directory> directories;
            struct Patch {
                size_t offset;
                uint32_t value;
                size_t bytes;
            };
            std::vector<Patch> patches;
            std::vector<std::pair<size_t, size_t>> tables, payloads;
            ExifParseStatus status = ExifParseStatus::Malformed;
            std::string diagnostic;
            bool fail(ExifParseStatus s, const char* message) {
                status = s;
                diagnostic = message;
                return false;
            }
            bool range(size_t offset, size_t size) const { return offset <= data.size() && size <= data.size() - offset; }
            uint32_t read(size_t offset, size_t bytes) const {
                uint32_t value = 0;
                for (size_t i = 0; i < bytes; ++i)
                    value |= uint32_t(data[offset + i]) << (8 * (little ? i : bytes - 1 - i));
                return value;
            }
            void write(size_t offset, uint32_t value, size_t bytes) {
                for (size_t i = 0; i < bytes; ++i)
                    data[offset + i] = static_cast<uint8_t>(value >> (8 * (little ? i : bytes - 1 - i)));
            }
            bool directory(uint32_t offset, std::vector<uint32_t> path, unsigned depth, uint32_t& next) {
                if (depth > 3 || directories.size() >= limits.directories)
                    return fail(ExifParseStatus::Limited, "EXIF directory budget exhausted");
                if (offset < 8)
                    return fail(ExifParseStatus::Malformed, "EXIF directory overlaps header");
                if (!visited.insert(offset).second)
                    return fail(ExifParseStatus::Unsupported, "Repeated EXIF directory offsets are not interpreted");
                if (!range(offset, 2))
                    return fail(ExifParseStatus::Malformed, "EXIF directory outside payload");
                const auto count = read(offset, 2);
                if (count > 4096 || count > limits.entries - entries)
                    return fail(ExifParseStatus::Limited, "EXIF entry budget exhausted");
                const size_t table_end = size_t(offset) + 2 + size_t(count) * 12;
                if (!range(offset + 2, size_t(count) * 12 + 4))
                    return fail(ExifParseStatus::Malformed, "Truncated EXIF directory");
                entries += count;
                next = read(table_end, 4);
                tables.emplace_back(offset, table_end + 4);
                patches.push_back({table_end, 0, 4}); // No synthetic-ID collisions.
                Directory node{offset, {}, path};
                node.tags.reserve(count);
                for (uint32_t i = 0; i < count; ++i) {
                    const size_t at = size_t(offset) + 2 + size_t(i) * 12;
                    Tag tag{static_cast<uint16_t>(read(at, 2)), static_cast<uint16_t>(read(at + 2, 2)), read(at + 4, 4)};
                    const auto payload = read(at + 8, 4);
                    if (!tag.type || tag.type >= widths.size())
                        return fail(ExifParseStatus::Unsupported, "Unsupported EXIF value type");
                    const bool pointer = tag.id != 0x927c &&
                                         (tag.type == AV_TIFF_IFD || tag.id == 0x8769 || tag.id == 0x8825 || tag.id == 0xa005);
                    if (pointer) {
                        if ((tag.type != AV_TIFF_LONG && tag.type != AV_TIFF_IFD) || tag.count != 1 || !payload)
                            return fail(ExifParseStatus::Unsupported, "Unsupported EXIF directory pointer representation");
                        tag.child = payload;
                    } else {
                        const uint64_t bytes = uint64_t(tag.count) * widths[tag.type];
                        const size_t value_at = bytes <= 4 ? at + 8 : payload;
                        if (bytes > data.size() || !range(value_at, static_cast<size_t>(bytes)))
                            return fail(ExifParseStatus::Malformed, "EXIF value outside payload");
                        if (bytes > 4)
                            payloads.emplace_back(value_at, value_at + static_cast<size_t>(bytes));
                        // Covers FFmpeg's widened integer/float values and the
                        // owned adapter values, including repeated value offsets.
                        const uint64_t storage = uint64_t(tag.count) * 32;
                        if (storage > limits.value_bytes - values)
                            return fail(ExifParseStatus::Limited, "EXIF decoded value budget exhausted");
                        values += static_cast<size_t>(storage);
                        if (!tag.count)
                            return fail(ExifParseStatus::Unsupported, "Zero-count EXIF value retained without interpretation");
                        if (tag.id == 0x927c) {
                            if (tag.type != AV_TIFF_UNDEFINED && tag.type != AV_TIFF_BYTE)
                                return fail(ExifParseStatus::Unsupported, "Unsupported MakerNote value type");
                            // Prevent FFmpeg's vendor-IFD guessing. Restore the
                            // original ID by entry ordinal when adapting values.
                            patches.push_back({at, 0xffff, 2});
                        }
                    }
                    node.tags.push_back(tag);
                }
                directories.push_back(node);
                for (uint32_t i = 0; i < count; ++i) {
                    if (!node.tags[i].child)
                        continue;
                    auto child_path = path;
                    child_path.push_back(i);
                    uint32_t child_next = 0;
                    if (!directory(node.tags[i].child, std::move(child_path), depth + 1, child_next))
                        return false;
                    if (child_next)
                        return fail(ExifParseStatus::Unsupported, "Chained nested EXIF directories retained without interpretation");
                }
                return true;
            }
        };
        bool adapt(const AVExifMetadata& parsed, const Guard& guard, const Directory& node,
                   std::vector<ExifEntry>& result) {
            if (parsed.count != node.tags.size())
                return false;
            for (uint32_t i = 0; i < parsed.count; ++i) {
                const auto& source = parsed.entries[i];
                const auto& tag = node.tags[i];
                if (source.id != (tag.id == 0x927c ? 0xffff : tag.id) ||
                    source.type != (tag.child ? AV_TIFF_IFD : tag.type) ||
                    source.count != (tag.child ? 1 : tag.count))
                    return false;
                ExifEntry entry;
                entry.directory_path = node.path;
                entry.directory_offset = node.offset;
                entry.entry_index = i;
                entry.id = tag.id;
                entry.type = tag.type;
                entry.count = tag.count;
                entry.directory = tag.child != 0;
                if (!entry.directory) {
                    switch (source.type) {
                    case AV_TIFF_SHORT:
                    case AV_TIFF_LONG: entry.unsigned_values.reserve(tag.count); break;
                    case AV_TIFF_SSHORT:
                    case AV_TIFF_SLONG:
                    case AV_TIFF_SBYTE: entry.signed_values.reserve(tag.count); break;
                    case AV_TIFF_FLOAT:
                    case AV_TIFF_DOUBLE: entry.floating_values.reserve(tag.count); break;
                    case AV_TIFF_RATIONAL:
                    case AV_TIFF_SRATIONAL: entry.rational_values.reserve(tag.count); break;
                    default: entry.bytes.reserve(tag.count); break;
                    }
                    for (uint32_t j = 0; j < tag.count; ++j) {
                        switch (source.type) {
                        case AV_TIFF_SHORT:
                        case AV_TIFF_LONG: entry.unsigned_values.push_back(source.value.uint[j]); break;
                        case AV_TIFF_SSHORT:
                        case AV_TIFF_SLONG: entry.signed_values.push_back(source.value.sint[j]); break;
                        case AV_TIFF_SBYTE: entry.signed_values.push_back(source.value.sbytes[j]); break;
                        case AV_TIFF_FLOAT:
                        case AV_TIFF_DOUBLE: entry.floating_values.push_back(source.value.dbl[j]); break;
                        case AV_TIFF_RATIONAL:
                            entry.rational_values.push_back({static_cast<uint32_t>(source.value.rat[j].num), static_cast<uint32_t>(source.value.rat[j].den)});
                            break;
                        case AV_TIFF_SRATIONAL: entry.rational_values.push_back({source.value.rat[j].num, source.value.rat[j].den}); break;
                        default: entry.bytes.push_back(source.value.ubytes[j]); break;
                        }
                    }
                }
                result.push_back(std::move(entry));
                if (tag.child) {
                    const auto child = std::find_if(guard.directories.begin(), guard.directories.end(), [&](const auto& d) { return d.offset == tag.child; });
                    if (child == guard.directories.end() || !adapt(source.value.ifd, guard, *child, result))
                        return false;
                }
            }
            return true;
        }
    } // namespace

    ExifInterpretation interpretExif(const core::image_codecs::ExifBlock& block, const ExifParseLimits& limits) {
        ExifInterpretation result;
        if (!limits.entries || limits.entries > 65536 || !limits.directories || limits.directories > 256 ||
            !limits.value_bytes || limits.value_bytes > 64 * 1024 * 1024 || block.bytes.size() > 4 * 1024 * 1024) {
            result.status = ExifParseStatus::Limited;
            result.diagnostic = "Invalid EXIF interpretation limits or oversized payload";
            return result;
        }
        size_t offset = 0;
        if (block.envelope == core::image_codecs::ExifEnvelope::JpegApp1) {
            if (block.bytes.size() < 6 || std::memcmp(block.bytes.data(), "Exif\0\0", 6)) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "Invalid JPEG EXIF envelope";
                return result;
            }
            offset = 6;
        } else if (block.envelope == core::image_codecs::ExifEnvelope::HeifOffset) {
            if (block.bytes.size() < 4) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "Truncated HEIF EXIF envelope";
                return result;
            }
            uint64_t skip = 4;
            for (int i = 0; i < 4; ++i)
                skip += uint64_t(block.bytes[i]) << (8 * (3 - i));
            if (skip > block.bytes.size()) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "HEIF EXIF offset outside payload";
                return result;
            }
            offset = static_cast<size_t>(skip);
        } else if (block.envelope != core::image_codecs::ExifEnvelope::PngExif) {
            result.status = ExifParseStatus::Unsupported;
            result.diagnostic = "Unsupported EXIF envelope";
            return result;
        }
        Guard guard;
        guard.limits = limits;
        guard.data.assign(block.bytes.begin() + offset, block.bytes.end());
        if (guard.data.size() < 8 || !((guard.data[0] == 'I' && guard.data[1] == 'I') || (guard.data[0] == 'M' && guard.data[1] == 'M'))) {
            result.status = ExifParseStatus::Malformed;
            result.diagnostic = "Invalid EXIF TIFF header";
            return result;
        }
        guard.little = guard.data[0] == 'I';
        if (guard.read(2, 2) != 42) {
            result.status = ExifParseStatus::Unsupported;
            result.diagnostic = "Only classic TIFF EXIF is interpreted";
            return result;
        }
        uint32_t next = guard.read(4, 4);
        uint32_t root = 0;
        if (!next) {
            result.status = ExifParseStatus::Malformed;
            result.diagnostic = "Missing EXIF root directory";
            return result;
        }
        while (next) {
            if (!guard.directory(next, {root++}, 0, next)) {
                result.status = guard.status;
                result.diagnostic = std::move(guard.diagnostic);
                return result;
            }
        }
        // Reject aliased tables/payloads before applying any private parse patch.
        // This also prevents a hostile value from being silently changed by an
        // adjacent IFD header rewrite. Shared value payloads remain supported.
        std::sort(guard.tables.begin(), guard.tables.end());
        size_t end = 8;
        for (const auto& table : guard.tables) {
            if (table.first < end) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "Overlapping EXIF directory tables";
                return result;
            }
            end = table.second;
            for (const auto& payload : guard.payloads) {
                if (payload.first < 8 || (payload.first < table.second && table.first < payload.second)) {
                    result.status = ExifParseStatus::Malformed;
                    result.diagnostic = "EXIF value overlaps a directory or header";
                    return result;
                }
            }
        }
        for (const auto& patch : guard.patches)
            guard.write(patch.offset, patch.value, patch.bytes);
        for (const auto& node : guard.directories) {
            if (node.path.size() != 1)
                continue;
            guard.write(4, node.offset, 4);
            struct Parsed {
                AVExifMetadata value{};
                ~Parsed() { av_exif_free(&value); }
            } parsed;
            const int status = av_exif_parse_buffer(nullptr, guard.data.data(), guard.data.size(), &parsed.value, AV_EXIF_TIFF_HEADER);
            if (status < 0 || parsed.value.count != node.tags.size()) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "FFmpeg rejected guarded EXIF structure";
                result.entries.clear();
                return result;
            }
            if (!adapt(parsed.value, guard, node, result.entries)) {
                result.status = ExifParseStatus::Malformed;
                result.diagnostic = "FFmpeg EXIF structure differs from guarded directory";
                result.entries.clear();
                return result;
            }
        }
        result.status = ExifParseStatus::Parsed;
        result.directory_count = guard.directories.size();
        result.value_storage_charge = guard.values;
        return result;
    }

    ImageExifDescription detail::describeSourceExif(core::image_codecs::SourceExif original) {
        ImageExifDescription result;
        result.original = std::move(original);
        ExifParseLimits remaining;
        for (const auto& block : result.original.blocks) {
            auto interpreted = interpretExif(block, remaining);
            remaining.entries -= interpreted.entries.size();
            remaining.directories -= interpreted.directory_count;
            remaining.value_bytes -= interpreted.value_storage_charge;
            result.interpretations.push_back(std::move(interpreted));
        }
        return result;
    }

    Result<ImageExifDescription> MediaIngest::sourceExif(const std::filesystem::path& path) {
        if (path.empty())
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = "EXIF source path must not be empty", .detection = LFS_SOURCE_SITE_CURRENT()});
        return detail::describeSourceExif(core::image_codecs::read_source_exif(path));
    }
} // namespace lfs::media
