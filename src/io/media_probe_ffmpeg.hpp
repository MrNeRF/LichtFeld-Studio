// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "io/media/media_probe.hpp"
#include <cstddef>
struct AVFormatContext;
struct AVStream;
namespace lfs::media::detail {
    // Non-owning descriptions: no reads, seeks or changes to the supplied context.
    [[nodiscard]] MediaDescription describeContext(const AVFormatContext* context);
    [[nodiscard]] StreamDescription describeStream(const AVStream* stream);
    [[nodiscard]] int findUsableHeaderVideoStream(const AVFormatContext* context);
    struct VideoStreamProbe {
        int stream_index = -1;
        int ffmpeg_error = 0;
        bool metadata_complete = false;
    };
    // Probes at most once: a fallback full probe already supplies video metadata.
    [[nodiscard]] VideoStreamProbe probeVideoStream(AVFormatContext* context);
    void discardNonVideoStreams(AVFormatContext* context, int video_stream_index);
    [[nodiscard]] Orientation describeDisplayMatrix(const unsigned char* data, std::size_t size);
} // namespace lfs::media::detail
