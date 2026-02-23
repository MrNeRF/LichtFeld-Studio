/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rmlui_system_interface.hpp"
#include "core/logger.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace lfs::vis::gui {

    RmlSystemInterface::RmlSystemInterface(GLFWwindow* window) : window_(window) {}

    double RmlSystemInterface::GetElapsedTime() { return glfwGetTime(); }

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
        if (window_)
            glfwSetClipboardString(window_, text.c_str());
    }

    void RmlSystemInterface::GetClipboardText(Rml::String& text) {
        if (window_) {
            const char* clipboard = glfwGetClipboardString(window_);
            if (clipboard)
                text = clipboard;
        }
    }

} // namespace lfs::vis::gui
