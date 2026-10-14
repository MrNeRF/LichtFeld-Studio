/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/hdr_tonemap.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

struct AVFormatContext;

namespace lfs::io::video {

    struct VideoMetadata {
        int stream_index = -1;
        int rotation = 0;
        double fps = 30.0;
        double duration = 0.0;
        int64_t total_frames = 0;
        bool untimed_stream = false;
        bool is_hdr = false;
        bool has_dolby_vision = false;
        HdrFormat hdr_format = HdrFormat::SDR;
    };

    // Reads stream-level metadata from an already opened demuxer. It does not configure a
    // decoder, create a CUDA context, or decode a preview frame.
    [[nodiscard]] bool probeVideoMetadata(AVFormatContext* context, VideoMetadata& metadata,
                                          std::string& error);

    // Opens and closes a demuxer around probeVideoMetadata().
    [[nodiscard]] bool probeVideoFileMetadata(const std::filesystem::path& path,
                                              VideoMetadata& metadata, std::string& error);

} // namespace lfs::io::video
