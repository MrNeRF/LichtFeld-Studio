/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Input.h>
#include <imgui.h>

namespace lfs::vis::gui {

    inline Rml::Input::KeyIdentifier imguiKeyToRml(ImGuiKey key) {
        // clang-format off
        switch (key) {
        case ImGuiKey_Space:      return Rml::Input::KI_SPACE;
        case ImGuiKey_Backspace:  return Rml::Input::KI_BACK;
        case ImGuiKey_Tab:        return Rml::Input::KI_TAB;
        case ImGuiKey_Enter:      return Rml::Input::KI_RETURN;
        case ImGuiKey_Escape:     return Rml::Input::KI_ESCAPE;
        case ImGuiKey_Delete:     return Rml::Input::KI_DELETE;
        case ImGuiKey_Home:       return Rml::Input::KI_HOME;
        case ImGuiKey_End:        return Rml::Input::KI_END;
        case ImGuiKey_LeftArrow:  return Rml::Input::KI_LEFT;
        case ImGuiKey_RightArrow: return Rml::Input::KI_RIGHT;
        case ImGuiKey_UpArrow:    return Rml::Input::KI_UP;
        case ImGuiKey_DownArrow:  return Rml::Input::KI_DOWN;
        default: break;
        }
        // clang-format on

        if (key >= ImGuiKey_A && key <= ImGuiKey_Z)
            return static_cast<Rml::Input::KeyIdentifier>(
                Rml::Input::KI_A + (key - ImGuiKey_A));
        if (key >= ImGuiKey_0 && key <= ImGuiKey_9)
            return static_cast<Rml::Input::KeyIdentifier>(
                Rml::Input::KI_0 + (key - ImGuiKey_0));

        return Rml::Input::KI_UNKNOWN;
    }

    inline int buildRmlModifiers() {
        ImGuiIO& io = ImGui::GetIO();
        int mods = 0;
        if (io.KeyCtrl)
            mods |= Rml::Input::KM_CTRL;
        if (io.KeyShift)
            mods |= Rml::Input::KM_SHIFT;
        if (io.KeyAlt)
            mods |= Rml::Input::KM_ALT;
        return mods;
    }

} // namespace lfs::vis::gui
