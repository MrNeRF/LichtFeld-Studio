/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_system_interface.hpp"
#include "core/logger.hpp"
#include "core/event_bridge/localization_manager.hpp"

#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_timer.h>

#include <cmath>

namespace lfs::vis::gui {

    RmlSystemInterface::RmlSystemInterface(SDL_Window* window) : window_(window) {}

    void RmlSystemInterface::beginFrame() {}

    RmlCursorRequest RmlSystemInterface::consumeCursorRequest() {
        return frame_cursor_request_;
    }

    double RmlSystemInterface::GetElapsedTime() {
        return static_cast<double>(SDL_GetTicks()) / 1000.0;
    }

    int RmlSystemInterface::TranslateString(Rml::String& translated, const Rml::String& input) {
        constexpr std::string_view kPrefix = "@tr:";
        const std::string_view view(input);
        if (!view.starts_with(kPrefix)) {
            translated = input;
            return 0;
        }

        translated = lfs::event::LocalizationManager::getInstance().get(view.substr(kPrefix.size()));
        return 1;
    }

    bool RmlSystemInterface::LogMessage(Rml::Log::Type type, const Rml::String& message) {
        if (type == Rml::Log::LT_WARNING &&
            (message.find("Data array index out of bounds") != Rml::String::npos ||
             message.find("Could not get value from data variable") != Rml::String::npos)) {
            return true;
        }

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

    void RmlSystemInterface::SetMouseCursor(const Rml::String& cursor_name) {
        frame_cursor_request_ = mapCursorRequest(cursor_name);
    }

    void RmlSystemInterface::JoinPath(Rml::String& translated_path,
                                      const Rml::String& document_path,
                                      const Rml::String& path) {
#ifndef _WIN32
        if (!path.empty() && path[0] == '/') {
            translated_path = path;
            return;
        }
#endif
        Rml::SystemInterface::JoinPath(translated_path, document_path, path);
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

    void RmlSystemInterface::ActivateKeyboard(const Rml::Vector2f caret_position,
                                              const float line_height) {
        if (!window_)
            return;

        SDL_Rect rect{};
        rect.x = static_cast<int>(std::lround(caret_position.x));
        rect.y = static_cast<int>(std::lround(caret_position.y));
        rect.w = 1;
        rect.h = std::max(1, static_cast<int>(std::lround(line_height)));
        SDL_SetTextInputArea(window_, &rect, 0);
    }

    void RmlSystemInterface::DeactivateKeyboard() {
        if (!window_)
            return;

        SDL_ClearComposition(window_);
        SDL_SetTextInputArea(window_, nullptr, 0);
    }

    RmlCursorRequest RmlSystemInterface::mapCursorRequest(const Rml::String& cursor_name) const {
        if (cursor_name == "text")
            return RmlCursorRequest::TextInput;
        if (cursor_name == "pointer")
            return RmlCursorRequest::Hand;
        if (cursor_name == "resize-horizontal" || cursor_name == "e-resize" ||
            cursor_name == "w-resize" || cursor_name == "ew-resize")
            return RmlCursorRequest::ResizeEW;
        if (cursor_name == "resize-vertical" || cursor_name == "n-resize" ||
            cursor_name == "s-resize" || cursor_name == "ns-resize")
            return RmlCursorRequest::ResizeNS;
        if (cursor_name == "nwse-resize" || cursor_name == "se-resize" ||
            cursor_name == "nw-resize")
            return RmlCursorRequest::ResizeNWSE;
        if (cursor_name == "nesw-resize" || cursor_name == "ne-resize" ||
            cursor_name == "sw-resize")
            return RmlCursorRequest::ResizeNESW;
        if (cursor_name == "move")
            return RmlCursorRequest::ResizeAll;
        if (cursor_name == "not-allowed")
            return RmlCursorRequest::NotAllowed;
        if (cursor_name == "default" || cursor_name == "auto")
            return RmlCursorRequest::Arrow;
        return RmlCursorRequest::Arrow;
    }

} // namespace lfs::vis::gui
