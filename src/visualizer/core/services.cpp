/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/services.hpp"
#include "visualizer/app_store.hpp"

namespace lfs::vis {

    void Services::notifyAlignStateChanged() {
        publish_align_state_generation();
    }

    Services& Services::instance() {
        static Services s;
        return s;
    }

} // namespace lfs::vis
