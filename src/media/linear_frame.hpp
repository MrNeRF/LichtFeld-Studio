// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/frame_sink.hpp"
extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/frame.h>
}
#include "video_color.hpp"

namespace lfs::media::detail {
    // Integer SDR through shared color primitives and the optional tensor host.
    class LinearFrameConverter {
    public:
        LinearFrameConverter(const AVCodecParameters&, FrameColor overrides);
        ~LinearFrameConverter();
        LinearFrameConverter(const LinearFrameConverter&) = delete;
        LinearFrameConverter& operator=(const LinearFrameConverter&) = delete;
        [[nodiscard]] Result<FrameColor> outputColor() const;
        [[nodiscard]] SinkResult convert(const AVFrame&, int width, int height, uint8_t* destination);
        [[nodiscard]] ColorTransfer inputTransfer() const { return transfer_; }

    private:
        AVCodecParameters source_{};
        FrameColor overrides_;
        ColorTransfer transfer_ = ColorTransfer::Unspecified;
        std::vector<uint8_t> planes_;
        std::unique_ptr<LinearVideoRenderer> renderer_;
        bool renderer_attempted_ = false;
    };
} // namespace lfs::media::detail
