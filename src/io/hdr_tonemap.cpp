/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_tonemap.hpp"

namespace lfs::io {

    HdrFormat detectHdrFormat(const int color_trc, int) {
        switch (color_trc) {
        case 18: return HdrFormat::HLG;   // ARIB STD-B67
        case 16: return HdrFormat::HDR10; // SMPTE ST 2084 / PQ
        default: return HdrFormat::SDR;
        }
    }

    HdrFormat detectDolbyVisionFormat(const int color_trc, const int dv_profile,
                                      const int compatibility_id) {
        if (dv_profile <= 0)
            return detectHdrFormat(color_trc, 0);

        // Compatibility id identifies the base layer for decoders that do not
        // apply RPU. Apple Dolby Vision 8.4 is HLG (id 4); DV 8.1/10.1 is
        // HDR10/PQ (id 1). Native profiles remain explicitly marked as such.
        if (compatibility_id == 4 || color_trc == 18)
            return HdrFormat::DOLBY_VISION_HLG;
        if (compatibility_id == 1 || color_trc == 16)
            return HdrFormat::DOLBY_VISION_HDR10;
        return HdrFormat::DOLBY_VISION_NATIVE;
    }

    bool isHdrFormat(const HdrFormat format) {
        return format != HdrFormat::SDR && format != HdrFormat::UNKNOWN;
    }

    bool isHdrTonemapSupported(const HdrFormat format) {
        return format == HdrFormat::HLG || format == HdrFormat::HDR10 ||
               format == HdrFormat::DOLBY_VISION_HLG ||
               format == HdrFormat::DOLBY_VISION_HDR10 ||
               format == HdrFormat::DOLBY_VISION_NATIVE;
    }

    const char* hdrFormatLabel(const HdrFormat fmt) {
        switch (fmt) {
        case HdrFormat::HLG: return "HLG";
        case HdrFormat::HDR10: return "HDR10";
        case HdrFormat::DOLBY_VISION_HLG: return "Dolby Vision 8.x (HLG)";
        case HdrFormat::DOLBY_VISION_HDR10: return "Dolby Vision 8.x (HDR10)";
        case HdrFormat::DOLBY_VISION_NATIVE: return "Dolby Vision (RPU)";
        default: return "SDR";
        }
    }

    const char* hdrFormatType(const HdrFormat fmt) {
        return isHdrFormat(fmt) ? "HDR" : "SDR";
    }

} // namespace lfs::io
