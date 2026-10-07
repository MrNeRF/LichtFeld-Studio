// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media_backends.hpp"
namespace lfs::io {
    class HdrLibplaceboRenderer::Impl {
    public:
        std::unique_ptr<HdrRenderer> renderer = media::detail::createHdrRenderer();
    };
    HdrLibplaceboRenderer::HdrLibplaceboRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;
    std::string_view HdrLibplaceboRenderer::backendName() const noexcept {
        return impl_->renderer ? impl_->renderer->backendName() : "none";
    }
    bool HdrLibplaceboRenderer::isAvailable(std::string& error) {
        if (impl_->renderer)
            return impl_->renderer->isAvailable(error);
        error = "No HDR renderer is registered in this host";
        return false;
    }
    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame* frame, const AVStream* stream,
                                             HdrFormat format, int width, int height, std::vector<unsigned char>& output,
                                             std::string& error, HdrTonemapTiming* timing) {
        if (!impl_->renderer)
            return isAvailable(error);
        return impl_->renderer->tonemapToSdr(frame, stream, format, width, height, output, error, timing);
    }
    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame* frame, const AVStream* stream,
                                                 HdrFormat format, int width, int height, int rotation, std::vector<unsigned char>& output,
                                                 std::string& error) {
        if (!impl_->renderer)
            return isAvailable(error);
        return impl_->renderer->tonemapToSdrRgba(frame, stream, format, width, height, rotation, output, error);
    }
    void HdrLibplaceboRenderer::reset() {
        if (impl_->renderer)
            impl_->renderer->reset();
    }
} // namespace lfs::io
