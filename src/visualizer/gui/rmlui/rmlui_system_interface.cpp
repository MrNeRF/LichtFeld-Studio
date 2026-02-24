/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_system_interface.hpp"
#include "core/logger.hpp"

#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_timer.h>

namespace lfs::vis::gui {

    RmlSystemInterface::RmlSystemInterface(SDL_Window* window) : window_(window) {}

    double RmlSystemInterface::GetElapsedTime() {
        return static_cast<double>(SDL_GetTicks()) / 1000.0;
    }

    bool RmlSystemInterface::LogMessage(Rml::Log::Type type, const Rml::String& message) {
        switch (type) {
        case Rml::Log::LT_ERROR:
        case Rml::Log::LT_ASSERT:
            LOG_ERROR("[RmlUI] {}", message);
            break;
        case Rml::Log::LT_WARNING:
            LOG_WARN("[RmlUI] {}", message);
            break;
        case Rml::Log::LT_INFO:
            LOG_INFO("[RmlUI] {}", message);
            break;
        default:
            LOG_DEBUG("[RmlUI] {}", message);
            break;
        }
        return true;
    }

    void RmlSystemInterface::SetClipboardText(const Rml::String& text) {
        SDL_SetClipboardText(text.c_str());
    }

    void RmlSystemInterface::GetClipboardText(Rml::String& text) {
        char* clipboard = SDL_GetClipboardText();
        if (clipboard) {
            text = clipboard;
            SDL_free(clipboard);
        }
    }

} // namespace lfs::vis::gui
