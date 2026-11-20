/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstdint>

namespace lfs::io {
    enum class SharpnessAlgorithm {
        LAPLACIAN, // Ratio of pixels with a strong Laplacian response
        TENENGRAD, // Sobel energy — directional blur detection
        COMBINED   // Tenengrad + Laplacian — best overall
    };

    // Sharp-edge pixel ratio (0-100), evaluated on the green channel of packed RGB8.
    // rgb must contain w*h*3 bytes for positive dimensions; images smaller than 3x3 score zero.
    [[nodiscard]] double computeSharpnessScore(const uint8_t* rgb, int w, int h, SharpnessAlgorithm algorithm);
} // namespace lfs::io
