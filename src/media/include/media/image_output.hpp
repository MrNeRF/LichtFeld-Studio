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
    // Synchronous CPU image output independent of decoder or renderer. Float
    // RGB/RGBA must be linear with declared primaries. Straight alpha is converted
    // to EXR premultiplied convention. Failed/cancelled writes never commit target.
    class LFS_MEDIA_API ImageOutput {
    public:
        [[nodiscard]] static SinkResult writeExr(const std::filesystem::path&, const FrameView&, const ExrOutputOptions& = {});
    };
} // namespace lfs::media
