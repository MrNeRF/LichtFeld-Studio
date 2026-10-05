/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_libplacebo.hpp"
#include "hdr_tonemap_tensor.hpp"

namespace lfs::io {
    class HdrLibplaceboRenderer::Impl {
    public:
        HdrTensorRenderer renderer;
    };

    HdrLibplaceboRenderer::HdrLibplaceboRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;

    bool HdrLibplaceboRenderer::isAvailable(std::string& error) {
        return impl_->renderer.isAvailable(error);
    }

    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame* frame, const AVStream* stream,
                                             HdrFormat format, int width, int height,
                                             std::vector<unsigned char>& output,
                                             std::string& error, HdrTonemapTiming* timing) {
        return impl_->renderer.tonemapToSdr(frame, stream, format, width, height, output, error, timing);
    }

    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame* frame, const AVStream* stream,
                                                 HdrFormat format, int width, int height,
                                                 int rotation, std::vector<unsigned char>& output,
                                                 std::string& error) {
        return impl_->renderer.tonemapToSdrRgba(frame, stream, format, width, height, rotation, output, error);
    }

    void HdrLibplaceboRenderer::reset() { impl_->renderer.reset(); }
} // namespace lfs::io
