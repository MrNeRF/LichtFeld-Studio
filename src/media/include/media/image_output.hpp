// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/frame_sink.hpp"
#include <functional>
namespace lfs::media {
    enum class ExrPrecision { Half,
                              Float };
    enum class ExrCompression { None,
                                ZIP };
    struct ExrOutputOptions {
        ExrPrecision precision = ExrPrecision::Half;
        ExrCompression compression = ExrCompression::ZIP;
        bool overwrite = false;
        std::string provenance;
        std::function<bool()> cancelled;
    };
    struct Tiff16OutputOptions {
        ColorTransfer transfer = ColorTransfer::Srgb;
        bool overwrite = false;
        std::string provenance;
        std::function<bool()> cancelled;
    };
    struct JxlOutputOptions {
        bool lossless = true;
        float distance = 1.f;
        int effort = 4; // Float32 lossless: 4..9; integer or lossy: 1..9.
        bool overwrite = false;
        std::function<bool()> cancelled;
    };
    // Synchronous CPU image output independent of decoder or renderer. Float
    // RGB/RGBA must be linear with declared primaries. Straight alpha is converted
    // to EXR premultiplied convention. Failed/cancelled writes never commit target.
    class LFS_MEDIA_API ImageOutput {
    public:
        [[nodiscard]] static SinkResult writeExr(const std::filesystem::path&, const FrameView&, const ExrOutputOptions& = {});
        // Relative scene-linear RGB/RGBA to UINT16 SDR. Explicit transfer,
        // primaries preserved, straight alpha, finite samples clamped to [0,1].
        [[nodiscard]] static SinkResult writeTiff16(const std::filesystem::path&, const FrameView&, const Tiff16OutputOptions& = {});
        // SDR sRGB RGB8 or relative linear BT709/BT2020 Float32 RGB/RGBA.
        // Preserve precision; straight alpha in output, no implicit tone mapping.
        [[nodiscard]] static SinkResult writeJxl(const std::filesystem::path&, const FrameView&, const JxlOutputOptions& = {});
    };
} // namespace lfs::media
