// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"
#include "media/source_exif.hpp"
#include <cmath>
#include <nlohmann/json.hpp>
namespace lfs::media::json_detail {
    using nlohmann::json;
    inline json rational(const std::optional<Rational>& value) {
        return value ? json::array({value->numerator, value->denominator}) : json(nullptr);
    }
    inline json timestamp(const std::optional<Timestamp>& value) {
        return value ? json{{"ticks", value->ticks}, {"time_base", {value->time_base.numerator, value->time_base.denominator}}} : json(nullptr);
    }
    template <class T>
    inline json optional(const std::optional<T>& value) { return value ? json(*value) : json(nullptr); }
    inline std::string hexBytes(const std::vector<uint8_t>& bytes) {
        constexpr char digits[] = "0123456789abcdef";
        std::string result(bytes.size() * 2, '0');
        for (size_t i = 0; i < bytes.size(); ++i) {
            result[2 * i] = digits[bytes[i] >> 4];
            result[2 * i + 1] = digits[bytes[i] & 15];
        }
        return result;
    }
    inline const char* metadataStatus(core::image_codecs::MetadataStatus status) {
        using core::image_codecs::MetadataStatus;
        switch (status) {
        case MetadataStatus::Absent: return "absent";
        case MetadataStatus::Collected: return "collected";
        case MetadataStatus::Limited: return "limited";
        case MetadataStatus::Malformed: return "malformed";
        case MetadataStatus::Unsupported: return "unsupported";
        case MetadataStatus::IoError: return "io_error";
        default: return "not_examined";
        }
    }
    inline const char* exifParseStatus(ExifParseStatus status) {
        switch (status) {
        case ExifParseStatus::Parsed: return "parsed";
        case ExifParseStatus::Limited: return "limited";
        case ExifParseStatus::Malformed: return "malformed";
        default: return "unsupported";
        }
    }
    inline json sourceExif(const ImageExifDescription& description) {
        const auto& original = description.original;
        json result{{"status", metadataStatus(original.status)}, {"scope", original.scope}, {"diagnostic", original.diagnostic}, {"binary_encoding", "hex"}, {"blocks", json::array()}};
        for (size_t i = 0; i < original.blocks.size(); ++i) {
            const auto& block = original.blocks[i];
            json item{{"envelope", block.envelope == core::image_codecs::ExifEnvelope::JpegApp1 ? "jpeg_app1" : "heif_offset"},
                      {"image_item", optional(block.image_item)},
                      {"metadata_item", optional(block.metadata_item)},
                      {"bytes", hexBytes(block.bytes)},
                      {"interpretation", nullptr}};
            if (i < description.interpretations.size()) {
                const auto& parsed = description.interpretations[i];
                json interpretation{{"status", exifParseStatus(parsed.status)}, {"diagnostic", parsed.diagnostic}, {"entries", json::array()}};
                for (const auto& entry : parsed.entries) {
                    json rationals = json::array(), floating = json::array();
                    for (const auto& r : entry.rational_values)
                        rationals.push_back({{"numerator", r.numerator}, {"denominator", r.denominator}});
                    for (const auto value : entry.floating_values)
                        floating.push_back(std::isfinite(value) ? json(value) : json(std::isnan(value) ? "nan" : value < 0 ? "-inf"
                                                                                                                           : "+inf"));
                    interpretation["entries"].push_back({{"directory_path", entry.directory_path}, {"directory_offset", entry.directory_offset}, {"entry_index", entry.entry_index}, {"id", entry.id}, {"type", entry.type}, {"count", entry.count}, {"directory", entry.directory}, {"unsigned_values", entry.unsigned_values}, {"signed_values", entry.signed_values}, {"floating_values", floating}, {"rational_values", rationals}, {"bytes", hexBytes(entry.bytes)}});
                }
                item["interpretation"] = std::move(interpretation);
            }
            result["blocks"].push_back(std::move(item));
        }
        return result;
    }
    inline const char* streamKind(StreamKind kind) {
        switch (kind) {
        case StreamKind::Video: return "video";
        case StreamKind::Audio: return "audio";
        case StreamKind::Subtitle: return "subtitle";
        case StreamKind::Data: return "data";
        case StreamKind::Attachment: return "attachment";
        default: return "unknown";
        }
    }
    inline const char* orientationSource(OrientationSource source) {
        switch (source) {
        case OrientationSource::RotateTag: return "rotate_tag";
        case OrientationSource::DisplayMatrix: return "display_matrix";
        default: return "none";
        }
    }
    inline json description(const MediaDescription& media) {
        json result{{"container", media.container}, {"stream_info_probed", media.stream_info_probed}, {"start", timestamp(media.start)}, {"duration", timestamp(media.duration)}, {"selected_video_stream", optional(media.selected_video_stream)}, {"streams", json::array()}};
        for (const auto& s : media.streams) {
            result["streams"].push_back({{"index", s.index}, {"kind", streamKind(s.kind)}, {"codec", s.codec}, {"width", optional(s.width)}, {"height", optional(s.height)}, {"pixel_format", optional(s.pixel_format)}, {"time_base", rational(s.time_base)}, {"nominal_frame_rate", rational(s.nominal_frame_rate)}, {"average_frame_rate", rational(s.average_frame_rate)}, {"sample_aspect_ratio", rational(s.sample_aspect_ratio)}, {"start", timestamp(s.start)}, {"duration", timestamp(s.duration)}, {"declared_frame_count", optional(s.declared_frame_count)}, {"attached_picture", s.attached_picture}, {"default_disposition", s.default_disposition}, {"color", {{"primaries", optional(s.color.primaries)}, {"transfer", optional(s.color.transfer)}, {"matrix", optional(s.color.matrix)}, {"range", optional(s.color.range)}, {"component_depth", optional(s.color.component_depth)}}}, {"orientation", {{"source", orientationSource(s.orientation.source)}, {"rotate_tag", optional(s.orientation.rotate_tag)}, {"display_matrix", optional(s.orientation.display_matrix)}, {"clockwise_degrees", optional(s.orientation.clockwise_degrees)}, {"reflected", optional(s.orientation.reflected)}}}});
        }
        return result;
    }
} // namespace lfs::media::json_detail
