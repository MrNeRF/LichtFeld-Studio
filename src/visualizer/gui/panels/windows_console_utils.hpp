/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"

namespace lfs::vis::gui::panels {

    void DrawSystemConsoleButton(const UIContext& ctx);
    void ToggleSystemConsole(const UIContext& ctx);
    void SetSystemConsoleVisible(const UIContext& ctx, bool visible);

} // namespace lfs::vis::gui::panels
