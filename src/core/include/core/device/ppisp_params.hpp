/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

namespace lfs::core {
    struct PpispParams {
        float exposure_factor = 1.0f;
        float vignetting[15]{};                                                       // Per channel: center x/y, radial r2/r4/r6 coefficients.
        float color_matrix[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};                             // Row-major homography in R/G/intensity space.
        float crf[15]{1, 1, 1, 0.5f, 0.5f, 1, 1, 1, 0.5f, 0.5f, 1, 1, 1, 0.5f, 0.5f}; // toe, shoulder, gamma, center, a.
        int y_offset = 0;
        int full_height = 0;
        int x_offset = 0;
        int full_width = 0;
    };
} // namespace lfs::core
