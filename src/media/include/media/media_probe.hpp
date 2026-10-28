// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "core/export.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace lfs::media {
    struct Rational {
        int numerator = 0;
        int denominator = 1;
    };
    struct Timestamp {
        int64_t ticks = 0;
        Rational time_base;
    };
    enum class StreamKind { Video,
                            Audio,
                            Subtitle,
                            Data,
                            Attachment,
                            Unknown };
    enum class OrientationSource { None,
                                   RotateTag,
                                   DisplayMatrix };
    struct Orientation {
        static constexpr std::size_t max_rotate_tag_bytes = 4096;
        OrientationSource source = OrientationSource::None;
        std::optional<std::string> rotate_tag;
        std::optional<std::array<int32_t, 9>> display_matrix;
        // Clockwise degrees. Unknown/malformed values are not replaced with zero.
        std::optional<double> clockwise_degrees;
        std::optional<bool> reflected;
        // Oversized tag text is omitted; numeric interpretation still borrows
        // the demuxer value. Raw collection has its own explicit byte budget.
        bool rotate_tag_limited = false;
    };
    struct ColorDescription {
        std::optional<std::string> primaries;
        std::optional<std::string> transfer;
        std::optional<std::string> matrix;
        std::optional<std::string> range;
        std::optional<int> component_depth;
    };
    struct StreamDescription {
        int index = -1;
        StreamKind kind = StreamKind::Unknown;
        std::string codec;
        std::optional<int> width;
        std::optional<int> height;
        std::optional<std::string> pixel_format;
        std::optional<Rational> time_base;
        std::optional<Rational> nominal_frame_rate;
        std::optional<Rational> average_frame_rate;
        std::optional<Rational> sample_aspect_ratio;
        std::optional<Timestamp> start;
        std::optional<Timestamp> duration;
        std::optional<int64_t> declared_frame_count;
        bool attached_picture = false;
        bool default_disposition = false;
        ColorDescription color;
        Orientation orientation;
    };
    struct MetadataTextEntry {
        // Demuxer-exposed bytes, not guaranteed UTF-8 or an original file span.
        std::string key;
        std::string value;
    };
    struct MetadataSideData {
        // Numeric FFmpeg AVPacketSideDataType; not a portable vendor tag ID.
        int type = 0;
        std::vector<std::uint8_t> bytes;
    };
    struct StreamSourceMetadata {
        int stream_index = -1;
        std::vector<MetadataTextEntry> tags;
        std::vector<MetadataSideData> coded_side_data;
    };
    struct MetadataProviderInfo {
        std::string name;
        std::string version;
        std::uint32_t codec_api_version = 0;
    };
    struct ContainerSourceMetadata {
        // Identifies the version-specific namespace/layout of coded side data.
        std::optional<MetadataProviderInfo> provider;
        // Limited describes this retained demuxer observation, not file damage.
        bool limited = false;
        std::vector<MetadataTextEntry> tags;
        std::vector<StreamSourceMetadata> streams;
    };
    struct ContainerMetadataLimits {
        std::size_t streams = 1024;
        // Aggregate tags and coded side-data blocks across container/streams.
        std::size_t records = 4096;
        // Aggregate copied key/value bytes and coded side-data payload bytes.
        std::size_t bytes = 8 * 1024 * 1024;
    };
    struct MediaDescription {
        std::string container;
        // FFmpeg may derive properties during packet probing; this is not a CFR guarantee.
        bool stream_info_probed = false;
        std::optional<Timestamp> start;
        std::optional<Timestamp> duration;
        std::vector<StreamDescription> streams;
        // First usable header video, matching the existing player/extractor.
        std::optional<int> selected_video_stream;
        // Absent unless explicitly requested; no packet telemetry scan.
        std::optional<ContainerSourceMetadata> source_metadata;
    };
    enum class ProbeDepth { Headers,
                            StreamInfo };
    struct ProbeOptions {
        ProbeDepth depth = ProbeDepth::StreamInfo;
        // Cooperative FFmpeg I/O interruption, not a hard wall-clock guarantee.
        std::chrono::milliseconds timeout{10000};
        bool collect_source_metadata = false;
        ContainerMetadataLimits metadata_limits;
    };
    // Metadata only: no image output, hardware context or application runtime.
    // StreamInfo may read packets and use FFmpeg's software codec probing.
    class LFS_MEDIA_API MediaProbe {
    public:
        [[nodiscard]] static Result<MediaDescription> inspect(const std::filesystem::path& path,
                                                              const ProbeOptions& options = {});
    };
    // Legacy preview accepts quarter turns only; the description retains raw values.
    [[nodiscard]] LFS_MEDIA_API int legacyQuarterTurn(const Orientation& orientation);
} // namespace lfs::media
