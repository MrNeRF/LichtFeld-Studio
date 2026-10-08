// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/video_color_math.h"
#include "media/frame_sink.hpp"
#include <span>

namespace lfs::media::detail {
    using core::color::VideoColorComponent;
    using core::color::VideoColorParameters;
    using core::color::VideoColorSampler;
    // Generalized from the tensor HDR decoder; shared with the linear profile.
    inline bool yuvLumaCoefficients(ColorMatrix matrix, int width, int height, std::array<float, 3>& out) {
        switch (matrix) {
        case ColorMatrix::Bt709: out = {0.2126f, 0.7152f, 0.0722f}; return true;
        case ColorMatrix::Bt470Bg:
        case ColorMatrix::Smpte170M: out = {0.2990f, 0.5870f, 0.1140f}; return true;
        case ColorMatrix::Smpte240M: out = {0.2122f, 0.7013f, 0.0865f}; return true;
        case ColorMatrix::Bt2020Ncl: out = {0.2627f, 0.6780f, 0.0593f}; return true;
        case ColorMatrix::Unspecified: out = width >= 1280 || height > 576 ? std::array{0.2126f, 0.7152f, 0.0722f} : std::array{0.2990f, 0.5870f, 0.1140f}; return true;
        default: return false;
        }
    }
    inline std::array<std::array<float, 3>, 3> yuvDecodeMatrix(const std::array<float, 3>& k) {
        return {{{1, 0, 2 * (1 - k[0])},
                 {1, -2 * (1 - k[2]) * k[2] / k[1], -2 * (1 - k[0]) * k[0] / k[1]},
                 {1, 2 * (1 - k[2]), 0}}};
    }
    struct YuvDecodeLevels {
        std::array<double, 3> multiplier, black;
    };
    inline YuvDecodeLevels yuvDecodeLevels(bool full, double expand, bool symmetric_full_chroma = true) {
        const double ymin = full ? 0.0 : 16 / 256.0 * expand, ymax = full ? 1.0 : 235 / 256.0 * expand;
        const double cmid = 128 / 256.0 * expand, cmax = full ? 1.0 : 240 / 256.0 * expand;
        // Display tonemapping uses a symmetric +/-0.5 chroma span. Linear
        // export retains full-range codes, including the asymmetric endpoint.
        const double chroma = full && !symmetric_full_chroma ? 1.0 : 0.5 / (cmax - cmid);
        return {{1.0 / (ymax - ymin), chroma, chroma}, {ymin, cmid, cmid}};
    }
    inline std::array<float, 4> chromaMapping(ChromaLocation location, int chroma_w, int chroma_h) {
        const float shift_x = location == ChromaLocation::Center || location == ChromaLocation::Top || location == ChromaLocation::Bottom ? 0.f : -.5f;
        const float shift_y = location == ChromaLocation::TopLeft || location == ChromaLocation::Top ? -.5f : location == ChromaLocation::BottomLeft || location == ChromaLocation::Bottom ? .5f
                                                                                                                                                                                           : 0.f;
        const float rx = 1.f / (1 << chroma_w), ry = 1.f / (1 << chroma_h);
        return {rx, (0.5f - shift_x) * rx - 0.5f, ry, (0.5f - shift_y) * ry - 0.5f};
    }
    class LFS_MEDIA_API LinearVideoRenderer {
    public:
        virtual ~LinearVideoRenderer() = default;
        virtual SinkResult convert(std::span<const uint8_t>, const VideoColorParameters&, int width, int height, std::span<uint8_t>) = 0;
    };
} // namespace lfs::media::detail
