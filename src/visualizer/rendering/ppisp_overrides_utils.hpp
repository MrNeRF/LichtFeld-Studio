/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_metrics.hpp"
#include "rendering_types.hpp"
#if LFS_BUILD_TRAINER
#include "training/trainer.hpp"
#endif

namespace lfs::vis {

    template <typename Destination>
    [[nodiscard]] inline Destination copyPpispOverrides(const PPISPOverrides& overrides) {
        Destination out{};
        out.exposure_offset = overrides.exposure_offset;
        out.vignette_enabled = overrides.vignette_enabled;
        out.vignette_strength = overrides.vignette_strength;
        out.wb_temperature = overrides.wb_temperature;
        out.wb_tint = overrides.wb_tint;
        out.color_red_x = overrides.color_red_x;
        out.color_red_y = overrides.color_red_y;
        out.color_green_x = overrides.color_green_x;
        out.color_green_y = overrides.color_green_y;
        out.color_blue_x = overrides.color_blue_x;
        out.color_blue_y = overrides.color_blue_y;
        out.gamma_multiplier = overrides.gamma_multiplier;
        out.gamma_red = overrides.gamma_red;
        out.gamma_green = overrides.gamma_green;
        out.gamma_blue = overrides.gamma_blue;
        out.crf_toe = overrides.crf_toe;
        out.crf_shoulder = overrides.crf_shoulder;
        return out;
    }

} // namespace lfs::vis
