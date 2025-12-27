/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis::gui {

    // Global DPI scale factor for UI elements
    // This is set during GuiManager::init() and should be used to scale
    // hardcoded UI dimensions (window sizes, button widths, etc.)
    inline float& dpiScale() {
        static float scale = 1.0f;
        return scale;
    }

    inline float getDpiScale() { return dpiScale(); }
    inline void setDpiScale(float scale) { dpiScale() = scale; }

} // namespace lfs::vis::gui
