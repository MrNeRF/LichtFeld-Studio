/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>

namespace lfs::vis::gui {

    enum class PanelSpace : uint8_t {
        SidePanel,
        Floating,
        ViewportOverlay,
        MainPanelTab,
        SceneHeader,
        StatusBar
    };

} // namespace lfs::vis::gui
