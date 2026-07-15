/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::io {

/// HDR format detected from color metadata
enum class HdrFormat {
    SDR,
    HLG,        // ARIB STD-B67 (Hybrid Log-Gamma)
    HDR10,      // PQ / SMPTE ST 2084
    UNKNOWN,
};

/// Detect HDR format from color transfer characteristic
[[nodiscard]] HdrFormat detectHdrFormat(int color_trc, int pix_fmt);

/// Human-readable label for HDR format
[[nodiscard]] const char* hdrFormatLabel(HdrFormat fmt);
[[nodiscard]] const char* hdrFormatType(HdrFormat fmt);

/**
 * @brief Apply HDR->SDR tone mapping to RGB24 pixel data in-place.
 *
 * Handles HLG (ARIB STD-B67) and HDR10 / PQ (SMPTE ST 2084) input.
 * Pipeline:
 *   1. Normalize [0,255] -> [0,1]
 *   2. Inverse OETF/EOTF -> linear scene light
 *   3. BT.2020 -> BT.709 primaries matrix
 *   4. Hable tone mapping curve
 *   5. BT.709 OETF (gamma)
 *   6. Denormalize -> [0,255]
 *
 * @param data       RGB pixel data (3 bytes per pixel in R,G,B order)
 * @param width      Image width in pixels
 * @param height     Image height in pixels
 * @param stride     Row stride in bytes (width*3 typically, padded otherwise)
 * @param format     Source HDR format (HLG or HDR10)
 * @param exposure   Exposure multiplier applied before tone mapping (default 1.0f)
 */
void tonemapHdrToSdr(unsigned char* data, int width, int height, int stride,
                     HdrFormat format, float exposure = 1.0f);

} // namespace lfs::io
