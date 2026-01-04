/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <deque>
#include <functional>
#include <string>
#include <imgui.h>

namespace lfs::vis::gui {

    class NotificationPopup {
    public:
        enum class Type { INFO,
                          WARNING,
                          FAILURE };
        using Callback = std::function<void()>;

        NotificationPopup();

        void render(const ImVec2& viewport_pos = {0, 0}, const ImVec2& viewport_size = {0, 0});
        void show(Type type, const std::string& title, const std::string& message,
                  Callback on_close = nullptr);

        static std::string formatDuration(float seconds);

    private:
        void setupEventHandlers();

        struct Notification {
            Type type;
            std::string title;
            std::string message;
            Callback on_close;
        };

        std::deque<Notification> pending_;
        Notification current_;
        bool popup_open_ = false;
    };

} // namespace lfs::vis::gui
