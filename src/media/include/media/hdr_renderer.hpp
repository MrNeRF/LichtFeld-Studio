/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "media/hdr_tonemap.hpp"

#include <memory>
#include <string>
#include <string_view>
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

    class LFS_MEDIA_API HdrRenderer {
    public:
        virtual ~HdrRenderer() = default;
        virtual std::string_view backendName() const noexcept { return "host"; }
        virtual bool isAvailable(std::string&) = 0;
        virtual bool tonemapToSdr(const AVFrame*, const AVStream*, HdrFormat, int, int,
                                  std::vector<unsigned char>&, std::string&, HdrTonemapTiming*) = 0;
        virtual bool tonemapToSdrRgba(const AVFrame*, const AVStream*, HdrFormat, int, int, int,
                                      std::vector<unsigned char>&, std::string&) = 0;
        virtual void reset() = 0;
    };

    class LFS_MEDIA_API HdrLibplaceboRenderer {
    public:
        HdrLibplaceboRenderer();
        ~HdrLibplaceboRenderer();

        HdrLibplaceboRenderer(const HdrLibplaceboRenderer&) = delete;
        HdrLibplaceboRenderer& operator=(const HdrLibplaceboRenderer&) = delete;

        [[nodiscard]] bool isAvailable(std::string& error);
        [[nodiscard]] std::string_view backendName() const noexcept;
        [[nodiscard]] bool tonemapToSdr(const AVFrame* frame, const AVStream* stream,
                                        HdrFormat source_format,
                                        int output_width, int output_height,
                                        std::vector<unsigned char>& output_rgb,
                                        std::string& error,
                                        HdrTonemapTiming* timing = nullptr);
        [[nodiscard]] bool tonemapToSdrRgba(const AVFrame* frame, const AVStream* stream,
                                            HdrFormat source_format,
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
