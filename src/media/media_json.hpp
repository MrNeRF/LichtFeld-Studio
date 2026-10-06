// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"
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
