/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <vector>

struct AVFrame;
struct AVStream;

namespace lfs::io {

    /// Vulkan/libplacebo SDR renderer for all HDR sources supported by FFmpeg.
    ///
    /// The input is deliberately the decoded AVFrame, before swscale changes
    /// its representation. libplacebo then owns color metadata interpretation,
    /// Dolby Vision reshaping, gamut conversion, tone mapping and 8-bit dithering.
    class HdrLibplaceboRenderer {
    public:
        HdrLibplaceboRenderer();
        ~HdrLibplaceboRenderer();

        HdrLibplaceboRenderer(const HdrLibplaceboRenderer&) = delete;
        HdrLibplaceboRenderer& operator=(const HdrLibplaceboRenderer&) = delete;

        [[nodiscard]] bool tonemapToSdr(const AVFrame* frame, const AVStream* stream,
                                        int output_width, int output_height,
                                        std::vector<unsigned char>& output_rgb,
                                        std::string& error);
        void reset();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
