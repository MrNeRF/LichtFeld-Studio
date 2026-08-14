/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/gui_focus_state.hpp"

namespace lfs::vis::gui {

    GuiFocusState& guiFocusState() {
        static GuiFocusState state;
        return state;
    }

} // namespace lfs::vis::gui
