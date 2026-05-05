/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "core/image_io.hpp"
#include "python/python_runtime.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

namespace lfs::vis::gui {

    MenuBar::MenuBar() = default;

    MenuBar::~MenuBar() = default;

    void MenuBar::requestThumbnail(const std::string& video_id) {
        startThumbnailDownload(video_id);
    }

    void MenuBar::processThumbnails() {
        updateThumbnails();
    }

    bool MenuBar::isThumbnailReady(const std::string& video_id) const {
        const auto it = thumbnails_.find(video_id);
        return it != thumbnails_.end() && it->second.state == Thumbnail::State::READY;
    }

    uint64_t MenuBar::getThumbnailTexture(const std::string& video_id) const {
        const auto it = thumbnails_.find(video_id);
        if (it != thumbnails_.end() && it->second.state == Thumbnail::State::READY) {
            return static_cast<uint64_t>(it->second.texture ? it->second.texture->textureId() : 0);
        }
        return 0;
    }

    void MenuBar::clearThumbnails() {
        thumbnails_.clear();
    }

    void MenuBar::startThumbnailDownload(const std::string& video_id) {
        if (video_id.empty() || thumbnails_.contains(video_id))
            return;

        auto& thumb = thumbnails_[video_id];
        thumb.state = Thumbnail::State::LOADING;

        thumb.download_future = std::async(std::launch::async, [video_id]() -> std::vector<uint8_t> {
            httplib::Client cli("https://img.youtube.com");
            cli.set_connection_timeout(5);
            cli.set_read_timeout(5);

            if (const auto res = cli.Get("/vi/" + video_id + "/mqdefault.jpg"))
                if (res->status == 200)
                    return {res->body.begin(), res->body.end()};
            return {};
        });
    }

    void MenuBar::updateThumbnails() {
        for (auto& [id, thumb] : thumbnails_) {
            if (thumb.state != Thumbnail::State::LOADING || !thumb.download_future.valid())
                continue;
            if (thumb.download_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
                continue;

            const auto data = thumb.download_future.get();
            if (data.empty()) {
                thumb.state = Thumbnail::State::FAILED;
                continue;
            }

            try {
                auto [pixels, w, h, c] = lfs::core::load_image_from_memory(data.data(), data.size());
                if (!pixels) {
                    thumb.state = Thumbnail::State::FAILED;
                    continue;
                }

                auto texture = std::make_unique<VulkanUiTexture>();
                const bool uploaded = texture->upload(
                    static_cast<const std::uint8_t*>(pixels),
                    w,
                    h,
                    c);
                lfs::core::free_image(pixels);
                if (uploaded) {
                    thumb.texture = std::move(texture);
                    thumb.state = Thumbnail::State::READY;
                } else {
                    thumb.state = Thumbnail::State::FAILED;
                }
            } catch (...) {
                thumb.state = Thumbnail::State::FAILED;
            }
        }
    }

    namespace {
        MenuBar* g_menu_bar_instance = nullptr;
        std::mutex g_menu_entries_mutex;
        std::vector<python::MenuBarEntry> g_menu_entries;
        std::atomic<bool> g_menu_entries_ready{false};
        std::atomic<bool> g_menu_entries_loading{false};

        void start_menu_entry_preload_once() {
            bool expected = false;
            if (!g_menu_entries_loading.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                return;
            }

            std::thread([] {
                auto entries = python::get_menu_bar_entries();
                {
                    std::lock_guard lock(g_menu_entries_mutex);
                    g_menu_entries = std::move(entries);
                }
                g_menu_entries_ready.store(true, std::memory_order_release);
                g_menu_entries_loading.store(false, std::memory_order_release);
            }).detach();
        }

        std::vector<python::MenuBarEntry> copy_menu_entries() {
            std::lock_guard lock(g_menu_entries_mutex);
            return g_menu_entries;
        }
    } // namespace

    bool MenuBar::hasMenuEntries() const {
        return g_menu_entries_ready.load(std::memory_order_acquire);
    }

    std::vector<python::MenuBarEntry> MenuBar::getMenuEntries() const {
        return copy_menu_entries();
    }

    void MenuBar::render() {
        if (g_menu_bar_instance != this) {
            g_menu_bar_instance = this;
            python::set_show_python_console_callback([]() {
                if (g_menu_bar_instance)
                    g_menu_bar_instance->triggerShowPythonConsole();
            });
        }

        if (!g_menu_entries_ready.load(std::memory_order_acquire))
            start_menu_entry_preload_once();

        if (python::are_plugins_loaded() && !g_menu_entries_ready.load(std::memory_order_acquire)) {
            g_menu_entries_loading.store(false, std::memory_order_release);
            start_menu_entry_preload_once();
        }
    }

    void MenuBar::setOnShowPythonConsole(std::function<void()> callback) {
        on_show_python_console_ = std::move(callback);
    }

} // namespace lfs::vis::gui
