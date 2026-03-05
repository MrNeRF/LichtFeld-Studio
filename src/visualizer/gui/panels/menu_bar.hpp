/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "python/python_runtime.hpp"

#include <functional>
#include <future>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    class MenuBar {
    public:
        MenuBar();
        ~MenuBar();

        void render();
        void setFonts(const FontSet& fonts) { fonts_ = fonts; }

        void setOnShowPythonConsole(std::function<void()> callback);

        void renderPluginInstallPopup();

        bool hasMenuEntries() const;
        std::vector<python::MenuBarEntry> getMenuEntries() const;

        void triggerShowPythonConsole() {
            if (on_show_python_console_)
                on_show_python_console_();
        }

        // Thumbnail system for Python access
        void requestThumbnail(const std::string& video_id);
        void processThumbnails();
        bool isThumbnailReady(const std::string& video_id) const;
        uint64_t getThumbnailTexture(const std::string& video_id) const;

    private:
        void openURL(const char* url);

        struct Thumbnail {
            unsigned int texture = 0;
            enum class State { PENDING,
                               LOADING,
                               READY,
                               FAILED } state = State::PENDING;
            std::future<std::vector<uint8_t>> download_future;
        };

        void startThumbnailDownload(const std::string& video_id);
        void updateThumbnails();

        std::function<void()> on_show_python_console_;

        bool show_plugin_install_popup_ = false;
        std::string plugin_install_url_;
        std::string plugin_status_message_;
        bool plugin_status_is_error_ = false;

        FontSet fonts_;
        std::unordered_map<std::string, Thumbnail> thumbnails_;
    };

} // namespace lfs::vis::gui
