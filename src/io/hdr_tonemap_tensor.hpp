/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "media/hdr_renderer.hpp"

#include <memory>

namespace lfs::io {

    // HdrLibplaceboRenderer's libplacebo output, rendered by Slang programs on
    // the tensor backend. Builds without Vulkan use it in place of libplacebo.
    class HdrTensorRenderer {
    public:
        HdrTensorRenderer();
        ~HdrTensorRenderer();

        HdrTensorRenderer(const HdrTensorRenderer&) = delete;
        HdrTensorRenderer& operator=(const HdrTensorRenderer&) = delete;

        [[nodiscard]] bool isAvailable(std::string& error);
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
