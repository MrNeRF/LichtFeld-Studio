/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/windows_console_utils.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#ifdef WIN32
#include <windows.h>
#endif

namespace lfs::vis::gui::panels {

    using namespace lichtfeld::Strings;

    void DrawSystemConsoleButton(const UIContext&) {}

    void SetSystemConsoleVisible([[maybe_unused]] const UIContext& ctx,
                                 [[maybe_unused]] const bool visible) {
#ifdef WIN32
        if (!ctx.window_states)
            return;
        bool& current = (*ctx.window_states)["system_console"];
        if (current == visible)
            return;
        HWND hwnd = GetConsoleWindow();
        Sleep(1);
        HWND owner = GetWindow(hwnd, GW_OWNER);
        HWND target = (owner == NULL) ? hwnd : owner;
        ShowWindow(target, visible ? SW_SHOW : SW_HIDE);
        current = visible;
#endif
    }

    void ToggleSystemConsole([[maybe_unused]] const UIContext& ctx) {
#ifdef WIN32
        if (!ctx.window_states)
            return;
        const bool current = ctx.window_states->contains("system_console") &&
                             ctx.window_states->at("system_console");
        SetSystemConsoleVisible(ctx, !current);
#endif
    }

} // namespace lfs::vis::gui::panels
