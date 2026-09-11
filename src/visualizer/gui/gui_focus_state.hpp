/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

namespace lfs::vis::gui {

    struct GuiFocusState {
        bool want_capture_mouse = false;
        bool want_capture_keyboard = false;
        bool want_text_input = false;
        bool any_item_active = false;

        void reset() {
            want_capture_mouse = false;
            want_capture_keyboard = false;
            want_text_input = false;
            any_item_active = false;
        }
    };

    LFS_VIS_API GuiFocusState& guiFocusState();

} // namespace lfs::vis::gui
