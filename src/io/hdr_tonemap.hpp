/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::io {

    /// HDR format detected from color metadata
    enum class HdrFormat {
        SDR,
        HLG,   // ARIB STD-B67 (Hybrid Log-Gamma)
        HDR10, // PQ / SMPTE ST 2084
        DOLBY_VISION_HLG,
        DOLBY_VISION_HDR10,
        DOLBY_VISION_NATIVE,
        UNKNOWN,
    };

    [[nodiscard]] HdrFormat detectHdrFormat(int color_trc, int bit_depth,
                                            bool has_mastering_display_metadata = false,
                                            bool has_content_light_metadata = false);
    [[nodiscard]] HdrFormat detectDolbyVisionFormat(int color_trc, int dv_profile,
                                                    int compatibility_id);

    [[nodiscard]] bool isHdrFormat(HdrFormat format);
    [[nodiscard]] bool isHdrTonemapSupported(HdrFormat format);

    /// Human-readable label for HDR format
    [[nodiscard]] const char* hdrFormatLabel(HdrFormat fmt);
    [[nodiscard]] const char* hdrFormatType(HdrFormat fmt);

} // namespace lfs::io
