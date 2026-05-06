/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/vulkan_ui_texture.hpp"
#include "python/python_runtime.hpp"

#include <functional>
#include <future>
#include <memory>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui {

    class MenuBar {
    public:
        MenuBar();
        ~MenuBar();

        void render();

        void setOnShowPythonConsole(std::function<void()> callback);

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
        void clearThumbnails();

    private:
        struct Thumbnail {
            std::unique_ptr<VulkanUiTexture> texture;
            enum class State { PENDING,
                               LOADING,
                               READY,
                               FAILED } state = State::PENDING;
            std::future<std::vector<uint8_t>> download_future;
        };

        void startThumbnailDownload(const std::string& video_id);
        void updateThumbnails();

        std::function<void()> on_show_python_console_;
        std::unordered_map<std::string, Thumbnail> thumbnails_;
    };

} // namespace lfs::vis::gui
