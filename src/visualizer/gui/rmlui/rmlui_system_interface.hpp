/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/SystemInterface.h>

struct GLFWwindow;

namespace lfs::vis::gui {

    class RmlSystemInterface final : public Rml::SystemInterface {
    public:
        explicit RmlSystemInterface(GLFWwindow* window);

        double GetElapsedTime() override;
        bool LogMessage(Rml::Log::Type type, const Rml::String& message) override;
        void SetClipboardText(const Rml::String& text) override;
        void GetClipboardText(Rml::String& text) override;

    private:
        GLFWwindow* window_;
    };

} // namespace lfs::vis::gui
