// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/frame_sink.hpp"
#include <functional>
#include <string_view>
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
    enum class JxlPrecision { Float32,
                              UInt16 };
    struct JxlOutputOptions {
        bool lossless = true;
        float distance = 1.f;
        int effort = 4; // Float32 lossless: 4..9; integer or lossy: 1..9.
        bool overwrite = false;
        std::function<bool()> cancelled;
        JxlPrecision precision = JxlPrecision::Float32;
        ColorTransfer transfer = ColorTransfer::Linear; // UInt16: linear or sRGB; Float32: linear only.
    };
    struct AvifOutputOptions {
        bool lossless = true;
        int quality = 90;
        int speed = 8;
        int bit_depth = 8;
        bool overwrite = false;
        std::function<bool()> cancelled;
    };
    // Synchronous CPU image output independent of decoder or renderer. Float
    // RGB/RGBA must be linear with declared primaries. Straight alpha is converted
    // to EXR premultiplied convention. Failed/cancelled writes never commit target.
    class LFS_MEDIA_API ImageOutput {
    public:
        [[nodiscard]] static SinkResult writeExr(const std::filesystem::path&, const FrameView&, const ExrOutputOptions& = {});
        // Explicit UTF-8 XMP, at most 32 MiB, consumed synchronously.
        [[nodiscard]] static SinkResult writeExr(const std::filesystem::path&, const FrameView&, const ExrOutputOptions&, const std::string& xmp);
        // Relative scene-linear RGB/RGBA to UINT16 SDR. Explicit transfer,
        // primaries preserved, straight alpha, finite samples clamped to [0,1].
        [[nodiscard]] static SinkResult writeTiff16(const std::filesystem::path&, const FrameView&, const Tiff16OutputOptions& = {});
        // Explicit UTF-8 XMP, bounded to 32 MiB and consumed synchronously.
        [[nodiscard]] static SinkResult writeTiff16(const std::filesystem::path&, const FrameView&, const Tiff16OutputOptions&, std::string_view xmp);
        // SDR sRGB RGB8 or relative linear BT709/BT2020 Float32 RGB/RGBA.
        // Default preserves Float32. Explicit UInt16 clamps and rounds relative
        // linear float to linear/sRGB codes; lossless then applies to those codes.
        // Straight alpha in output, no implicit tone mapping.
        [[nodiscard]] static SinkResult writeJxl(const std::filesystem::path&, const FrameView&, const JxlOutputOptions& = {});
        // Explicit synchronous XMP packet; empty keeps the previous behavior.
        [[nodiscard]] static SinkResult writeJxl(const std::filesystem::path&, const FrameView&, const JxlOutputOptions&, std::string_view xmp);
        // Relative linear float is clamped, sRGB-transferred and rounded to the
        // selected 8/10/12-bit codes. Lossless applies to those integer codes.
        // RGB8 requires declared sRGB BT709 and bit_depth=8. Straight alpha.
        [[nodiscard]] static SinkResult writeAvif(const std::filesystem::path&, const FrameView&, const AvifOutputOptions& = {});
    };
} // namespace lfs::media
