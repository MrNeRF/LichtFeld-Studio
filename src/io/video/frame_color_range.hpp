/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
}

#include <cstring>

namespace lfs::io::video {

    inline void restoreCuvidFullRange(AVFrame* frame, const AVCodecContext* decoder,
                                      const AVCodecParameters* stream) {
        // CUVID reports limited range for VP9/AV1 even when the probed stream
        // explicitly signals full range. Preserve all other decoder metadata.
        if (stream->color_range == AVCOL_RANGE_JPEG && frame->color_range == AVCOL_RANGE_MPEG &&
            (std::strcmp(decoder->codec->name, "vp9_cuvid") == 0 ||
             std::strcmp(decoder->codec->name, "av1_cuvid") == 0)) {
            frame->color_range = AVCOL_RANGE_JPEG;
        }
    }

} // namespace lfs::io::video
