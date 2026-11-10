// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/media_probe.hpp"
#include <cstddef>
struct AVFormatContext;
struct AVStream;
struct AVRational;
namespace lfs::media::detail {
    // Non-owning descriptions: no reads, seeks or changes to the supplied context.
    [[nodiscard]] LFS_MEDIA_API MediaDescription describeContext(const AVFormatContext* context);
    [[nodiscard]] MetadataProviderInfo metadataProviderInfo();
    [[nodiscard]] std::optional<Timestamp> probeTimestamp(std::int64_t, AVRational);
    [[nodiscard]] Error probeInvalidOptions(std::string);
    [[nodiscard]] Error probeFailure(int, const char*, bool);
    // Reads only the chosen data/subtitle stream using the existing context.
    [[nodiscard]] Result<TimedMetadataObservation> collectTimedMetadata(AVFormatContext*, int, const TimedMetadataLimits&);

    // Copies only demuxer-exposed static metadata, without reads or mutations.
    LFS_MEDIA_API void collectContextMetadata(const AVFormatContext*, MediaDescription&, const ContainerMetadataLimits& = {});
    [[nodiscard]] LFS_MEDIA_API StreamDescription describeStream(const AVStream* stream);
    [[nodiscard]] LFS_MEDIA_API int findUsableHeaderVideoStream(const AVFormatContext* context);
    struct VideoStreamProbe {
        int stream_index = -1;
        int ffmpeg_error = 0;
        bool metadata_complete = false;
    };
    // Probes at most once: a fallback full probe already supplies video metadata.
    [[nodiscard]] LFS_MEDIA_API VideoStreamProbe probeVideoStream(AVFormatContext* context);
    LFS_MEDIA_API void keepOnlyStream(AVFormatContext* context, int stream_index);
    [[nodiscard]] LFS_MEDIA_API Orientation describeDisplayMatrix(const unsigned char* data, std::size_t size);
} // namespace lfs::media::detail
