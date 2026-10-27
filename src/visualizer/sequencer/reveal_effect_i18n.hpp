/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "sequencer/timeline.hpp"
#include <array>

namespace lfs::vis {
    inline constexpr std::array reveal_effect_keys{
        "sequencer.reveal_effect_magic", "sequencer.reveal_effect_spread",
        "sequencer.reveal_effect_unroll", "sequencer.reveal_effect_twister",
        "sequencer.reveal_effect_rain", "sequencer.reveal_effect_unveil",
        "sequencer.reveal_effect_light_edge", "sequencer.reveal_effect_rainbow",
        "sequencer.reveal_effect_dark_to_light", "sequencer.reveal_effect_distortion",
        "sequencer.reveal_effect_camera_light"};

    inline const char* revealEffectKey(const sequencer::RevealEffect effect) {
        const int index = static_cast<int>(effect) - 1;
        return index >= 0 && index < static_cast<int>(reveal_effect_keys.size())
                   ? reveal_effect_keys[index]
                   : reveal_effect_keys.front();
    }
} // namespace lfs::vis
