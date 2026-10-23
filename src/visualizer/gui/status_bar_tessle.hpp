/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <cstdint>
#include <string>

namespace lfs::vis::gui::tessle {

    struct Frame {
        float left_dp = 0.0f;
        std::string sprite;
        std::string scenery_rml;
    };

    // Animation time is training elapsed time, so pausing freezes the entire scene.
    [[nodiscard]] LFS_VIS_API Frame buildFrame(float progress, float bar_dp, int64_t animation_ms);

} // namespace lfs::vis::gui::tessle
