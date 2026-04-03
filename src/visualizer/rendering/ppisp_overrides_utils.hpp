/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering_types.hpp"
#include "training/trainer.hpp"

namespace lfs::vis {

    [[nodiscard]] inline lfs::training::PPISPViewportOverrides toTrainerPPISPOverrides(
        const PPISPOverrides& overrides) {
        lfs::training::PPISPViewportOverrides trainer_overrides{};
        trainer_overrides.exposure_offset = overrides.exposure_offset;
        trainer_overrides.vignette_enabled = overrides.vignette_enabled;
        trainer_overrides.vignette_strength = overrides.vignette_strength;
        trainer_overrides.wb_temperature = overrides.wb_temperature;
        trainer_overrides.wb_tint = overrides.wb_tint;
        trainer_overrides.color_red_x = overrides.color_red_x;
        trainer_overrides.color_red_y = overrides.color_red_y;
        trainer_overrides.color_green_x = overrides.color_green_x;
        trainer_overrides.color_green_y = overrides.color_green_y;
        trainer_overrides.color_blue_x = overrides.color_blue_x;
        trainer_overrides.color_blue_y = overrides.color_blue_y;
        trainer_overrides.gamma_multiplier = overrides.gamma_multiplier;
        trainer_overrides.gamma_red = overrides.gamma_red;
        trainer_overrides.gamma_green = overrides.gamma_green;
        trainer_overrides.gamma_blue = overrides.gamma_blue;
        trainer_overrides.crf_toe = overrides.crf_toe;
        trainer_overrides.crf_shoulder = overrides.crf_shoulder;
        return trainer_overrides;
    }

    [[nodiscard]] inline bool ppispOverridesEqual(const PPISPOverrides& a,
                                                  const PPISPOverrides& b) {
        return a.exposure_offset == b.exposure_offset &&
               a.vignette_enabled == b.vignette_enabled &&
               a.vignette_strength == b.vignette_strength &&
               a.wb_temperature == b.wb_temperature &&
               a.wb_tint == b.wb_tint &&
               a.color_red_x == b.color_red_x &&
               a.color_red_y == b.color_red_y &&
               a.color_green_x == b.color_green_x &&
               a.color_green_y == b.color_green_y &&
               a.color_blue_x == b.color_blue_x &&
               a.color_blue_y == b.color_blue_y &&
               a.gamma_multiplier == b.gamma_multiplier &&
               a.gamma_red == b.gamma_red &&
               a.gamma_green == b.gamma_green &&
               a.gamma_blue == b.gamma_blue &&
               a.crf_toe == b.crf_toe &&
               a.crf_shoulder == b.crf_shoulder;
    }

} // namespace lfs::vis
