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

    struct HdrTonemapTiming {
        double initialization_seconds = 0.0;
        double render_seconds = 0.0;
        double readback_seconds = 0.0;
        double rgba_to_rgb_seconds = 0.0;
    };

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
                                        std::string& error,
                                        HdrTonemapTiming* timing = nullptr);
        // Preview consumers can keep libplacebo's native RGBA8 readback and
        // upload it directly to a Vulkan texture, avoiding RGB/RGBA CPU copies.
        [[nodiscard]] bool tonemapToSdrRgba(const AVFrame* frame, const AVStream* stream,
                                            int output_width, int output_height,
                                            int rotation_degrees,
                                            std::vector<unsigned char>& output_rgba,
                                            std::string& error);
        void reset();

    private:
        class Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io
