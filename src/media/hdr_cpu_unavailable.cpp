// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "io/hdr_libplacebo.hpp"
namespace lfs::io {
    class HdrLibplaceboRenderer::Impl {};
    HdrLibplaceboRenderer::HdrLibplaceboRenderer() = default;
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;
    bool HdrLibplaceboRenderer::isAvailable(std::string& error) {
        error = "HDR tone mapping is unavailable in the CPU Media Ingest profile";
        return false;
    }
    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame*, const AVStream*, HdrFormat, int, int,
                                             std::vector<unsigned char>&, std::string& error, HdrTonemapTiming*) {
        return isAvailable(error);
    }
    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame*, const AVStream*, HdrFormat, int, int, int,
                                                 std::vector<unsigned char>&, std::string& error) {
        return isAvailable(error);
    }
    void HdrLibplaceboRenderer::reset() {}
} // namespace lfs::io
