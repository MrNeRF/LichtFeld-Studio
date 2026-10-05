/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_libplacebo.hpp"

namespace lfs::io {
    class HdrLibplaceboRenderer::Impl {};

    HdrLibplaceboRenderer::HdrLibplaceboRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;

    bool HdrLibplaceboRenderer::isAvailable(std::string& error) {
        error = "HDR libplacebo is unavailable in the Metal-only build because its current path requires Vulkan";
        return false;
    }

    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame*, const AVStream*, HdrFormat,
                                             int, int, std::vector<unsigned char>&,
                                             std::string& error, HdrTonemapTiming*) {
        return isAvailable(error);
    }

    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame*, const AVStream*, HdrFormat,
                                                 int, int, int,
                                                 std::vector<unsigned char>&,
                                                 std::string& error) {
        return isAvailable(error);
    }

    void HdrLibplaceboRenderer::reset() {}
} // namespace lfs::io
