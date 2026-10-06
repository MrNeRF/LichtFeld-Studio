/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_studio_backend.hpp"
#include "hdr_tonemap_tensor.hpp"

namespace lfs::io {
    class HdrStudioRenderer::Impl {
    public:
        HdrTensorRenderer renderer;
    };

    HdrStudioRenderer::HdrStudioRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrStudioRenderer::~HdrStudioRenderer() = default;
    std::string_view HdrStudioRenderer::backendName() const noexcept { return "tensor"; }

    bool HdrStudioRenderer::isAvailable(std::string& error) {
        return impl_->renderer.isAvailable(error);
    }

    bool HdrStudioRenderer::tonemapToSdr(const AVFrame* frame, const AVStream* stream,
                                         HdrFormat format, int width, int height,
                                         std::vector<unsigned char>& output,
                                         std::string& error, HdrTonemapTiming* timing) {
        return impl_->renderer.tonemapToSdr(frame, stream, format, width, height, output, error, timing);
    }

    bool HdrStudioRenderer::tonemapToSdrRgba(const AVFrame* frame, const AVStream* stream,
                                             HdrFormat format, int width, int height,
                                             int rotation, std::vector<unsigned char>& output,
                                             std::string& error) {
        return impl_->renderer.tonemapToSdrRgba(frame, stream, format, width, height, rotation, output, error);
    }

    void HdrStudioRenderer::reset() { impl_->renderer.reset(); }
} // namespace lfs::io
