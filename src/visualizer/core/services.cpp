/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/services.hpp"
#include "visualizer/app_store.hpp"

namespace lfs::vis {

    Services::Services()
        : align_point_pick_(publish_align_state_generation, false),
          scale_distance_point_pick_(publish_scale_distance_state_generation, true) {}

    Services& Services::instance() {
        static Services s;
        return s;
    }

} // namespace lfs::vis
