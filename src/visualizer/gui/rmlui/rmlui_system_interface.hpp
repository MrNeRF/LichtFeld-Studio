/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/SystemInterface.h>

struct SDL_Window;

namespace lfs::vis::gui {

    class RmlSystemInterface final : public Rml::SystemInterface {
    public:
        explicit RmlSystemInterface(SDL_Window* window);

        double GetElapsedTime() override;
        bool LogMessage(Rml::Log::Type type, const Rml::String& message) override;
        void SetClipboardText(const Rml::String& text) override;
        void GetClipboardText(Rml::String& text) override;
        void JoinPath(Rml::String& translated_path, const Rml::String& document_path,
                      const Rml::String& path) override;

    private:
        SDL_Window* window_;
    };

} // namespace lfs::vis::gui
