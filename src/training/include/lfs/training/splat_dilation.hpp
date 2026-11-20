/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::training {

    // Screen-space dilation (px^2) added to the diagonal of each projected 2D covariance. The mip filter
    // uses the smaller one and scales the opacity by sqrt(det(cov2d) / det(cov2d + dilation * I)).
    inline constexpr float kSplatDilation = 0.3f;
    inline constexpr float kMipFilterDilation = 0.1f;

} // namespace lfs::training
