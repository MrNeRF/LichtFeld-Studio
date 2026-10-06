/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "media/hdr_renderer.hpp"

#include <memory>
#include <string>
#include <vector>

struct AVFrame;
struct AVStream;

namespace lfs::io {

    class HdrStudioRenderer final : public HdrRenderer {
    public:
        HdrStudioRenderer();
        ~HdrStudioRenderer();

        HdrStudioRenderer(const HdrStudioRenderer&) = delete;
        HdrStudioRenderer& operator=(const HdrStudioRenderer&) = delete;

        [[nodiscard]] bool isAvailable(std::string& error);
        [[nodiscard]] std::string_view backendName() const noexcept override;
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
