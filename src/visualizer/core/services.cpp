/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/services.hpp"
#include "visualizer/app_store.hpp"
#include <algorithm>

namespace lfs::vis {

    void PointPickState::addPairColorSlot() {
        size_t slot = 0;
        while (std::find(pair_color_slots_.begin(), pair_color_slots_.end(), slot) != pair_color_slots_.end())
            ++slot;
        pair_color_slots_.push_back(slot);
    }

    void PointPickState::removePairColorSlot(const size_t pair) {
        if (pair < pair_color_slots_.size())
            pair_color_slots_.erase(pair_color_slots_.begin() + pair);
    }
    Services::Services()
        : align_point_pick_(publish_align_state_generation, false),
          scale_distance_point_pick_(publish_scale_distance_state_generation, true) {}

    Services& Services::instance() {
        static Services s;
        return s;
    }

} // namespace lfs::vis
