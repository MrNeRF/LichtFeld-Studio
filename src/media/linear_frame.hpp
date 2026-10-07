// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/frame_sink.hpp"
extern "C" {
#include <libavcodec/codec_par.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}
#include <array>

namespace lfs::media::detail {
    // Integer SDR: RGB48 unpacking for RGB, direct float matrix for YUV444,
    // then inverse transfer and pixel-center resize in linear light. No RGB8.
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
        std::array<float, 65536> lut_{};
        std::vector<uint8_t> rgb48_;
        SwsContext* context_ = nullptr;
        int cached_width_ = 0, cached_height_ = 0, cached_format_ = -1;
    };
} // namespace lfs::media::detail
