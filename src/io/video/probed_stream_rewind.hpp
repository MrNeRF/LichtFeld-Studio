/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

extern "C" {
#include <libavformat/avformat.h>
}

namespace lfs::io::video {

    inline void limitUntimedVideoProbe(AVFormatContext* const context) {
        if (!context || !context->iformat ||
            !(context->iformat->flags & AVFMT_NOTIMESTAMPS))
            return;

        constexpr int64_t max_probe_size = 1024 * 1024;
        constexpr int64_t max_analyze_duration = AV_TIME_BASE;
        constexpr int max_probe_packets = 2048;
        if (context->probesize <= 0 || context->probesize > max_probe_size)
            context->probesize = max_probe_size;
        if (context->max_analyze_duration <= 0 ||
            context->max_analyze_duration > max_analyze_duration)
            context->max_analyze_duration = max_analyze_duration;
        if (context->max_probe_packets <= 0 ||
            context->max_probe_packets > max_probe_packets)
            context->max_probe_packets = max_probe_packets;
    }

    inline bool rewindProbedVideoStream(
        AVFormatContext* const context, const int stream_index) {
        if (!context || !context->iformat || stream_index < 0 ||
            stream_index >= static_cast<int>(context->nb_streams))
            return false;

        // A timestamp seek on a raw stream can consume it to EOF without rewinding.
        const bool untimed = (context->iformat->flags & AVFMT_NOTIMESTAMPS) != 0;
        // Rewind after probing so decoding starts at the first packet.
        return (untimed ? av_seek_frame(context, -1, 0, AVSEEK_FLAG_BYTE)
                        : av_seek_frame(context, stream_index, 0, AVSEEK_FLAG_BACKWARD)) >= 0;
    }

} // namespace lfs::io::video
