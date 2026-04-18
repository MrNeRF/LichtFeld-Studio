/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <RmlUi/Core/SystemInterface.h>

#include <cstdint>

struct SDL_Window;

namespace Rml {
    class Context;
}

namespace lfs::vis::gui {

    enum class RmlCursorRequest : uint8_t {
        None,
        Arrow,
        TextInput,
        Hand,
        Pipette,
        ResizeEW,
        ResizeNS,
        ResizeNWSE,
        ResizeNESW,
        ResizeAll,
        NotAllowed,
    };

    class LFS_VIS_API RmlSystemInterface final : public Rml::SystemInterface {
    public:
        explicit RmlSystemInterface(SDL_Window* window);

        void beginFrame();
        void trackContext(const Rml::Context* context, int window_x, int window_y);
        void releaseContext(const Rml::Context* context);
        RmlCursorRequest consumeCursorRequest();

        double GetElapsedTime() override;
        int TranslateString(Rml::String& translated, const Rml::String& input) override;
        bool LogMessage(Rml::Log::Type type, const Rml::String& message) override;
        void SetMouseCursor(const Rml::String& cursor_name) override;
        void SetClipboardText(const Rml::String& text) override;
        void GetClipboardText(Rml::String& text) override;
        void JoinPath(Rml::String& translated_path, const Rml::String& document_path,
                      const Rml::String& path) override;
        void ActivateKeyboard(Rml::Vector2f caret_position, float line_height) override;
        void DeactivateKeyboard() override;

    private:
        RmlCursorRequest mapCursorRequest(const Rml::String& cursor_name) const;

        SDL_Window* window_;
        const Rml::Context* current_context_ = nullptr;
        const Rml::Context* cursor_context_ = nullptr;
        int current_context_window_x_ = 0;
        int current_context_window_y_ = 0;
        RmlCursorRequest cursor_request_ = RmlCursorRequest::None;
    };

} // namespace lfs::vis::gui
