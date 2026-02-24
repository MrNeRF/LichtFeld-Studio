/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "config.h"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/tensor_trace.hpp"
#include "core/training_snapshot.hpp"
#include "python/python_runtime.hpp"
#ifdef WIN32
#include <shellapi.h>
#include <winsock2.h>
#endif
#include "core/event_bridge/localization_manager.hpp"
#include "gui/string_keys.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"

using namespace lichtfeld::Strings;

#include <glad/glad.h>
#include <imgui.h>

#include <atomic>
#include <cfloat>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "python/gil.hpp"
#include "python/runner.hpp"
#include <Python.h>
#ifdef PLATFORM
#undef PLATFORM
#endif

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

namespace lfs::vis::gui {

    MenuBar::MenuBar() = default;

    MenuBar::~MenuBar() {
        for (const auto& [id, thumb] : thumbnails_) {
            if (thumb.texture)
                glDeleteTextures(1, &thumb.texture);
        }
    }

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
            return static_cast<uint64_t>(it->second.texture);
        }
        return 0;
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

                GLuint tex = 0;
                glGenTextures(1, &tex);
                glBindTexture(GL_TEXTURE_2D, tex);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
                glBindTexture(GL_TEXTURE_2D, 0);

                lfs::core::free_image(pixels);
                thumb.texture = tex;
                thumb.state = Thumbnail::State::READY;
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

    void MenuBar::render() {
        const auto& t = theme();

        // Register callbacks for Python-driven menu
        if (g_menu_bar_instance != this) {
            g_menu_bar_instance = this;
            python::set_show_python_console_callback([]() {
                if (g_menu_bar_instance)
                    g_menu_bar_instance->triggerShowPythonConsole();
            });
        }

        if (fonts_.regular)
            ImGui::PushFont(fonts_.regular);

        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, t.menu_background());
        ImGui::PushStyleColor(ImGuiCol_Header, t.menu_active());
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, t.menu_hover());
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, t.menu_active());
        ImGui::PushStyleColor(ImGuiCol_PopupBg, t.menu_popup_background());
        ImGui::PushStyleColor(ImGuiCol_Border, t.menu_border());
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, t.menu.popup_rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, t.menu.popup_border_size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, t.menu.popup_padding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, t.menu.frame_padding);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, t.menu.item_spacing);

        if (!g_menu_entries_ready.load(std::memory_order_acquire)) {
            start_menu_entry_preload_once();
        }

        if (python::are_plugins_loaded() && !g_menu_entries_ready.load(std::memory_order_acquire)) {
            g_menu_entries_loading.store(false, std::memory_order_release);
            start_menu_entry_preload_once();
        }

        if (ImGui::BeginMainMenuBar()) {
            if (g_menu_entries_ready.load(std::memory_order_acquire)) {
                auto entries = copy_menu_entries();
                for (const auto& entry : entries) {
                    if (ImGui::BeginMenu(LOC(entry.label.c_str()))) {
                        python::draw_menu_bar_entry(entry.idname);
                        ImGui::EndMenu();
                    }
                }
            }

            const float h = ImGui::GetWindowHeight();
            ImGui::GetWindowDrawList()->AddLine({0, h - 1}, {ImGui::GetWindowWidth(), h - 1},
                                                t.menu_bottom_border_u32(), 1.0f);

            ImGui::EndMainMenuBar();
        }

        ImGui::PopStyleVar(5);
        ImGui::PopStyleColor(6);
        if (fonts_.regular)
            ImGui::PopFont();

        renderPluginInstallPopup();
    }

    void MenuBar::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open " + std::string(url);
        system(cmd.c_str());
#endif
    }

    void MenuBar::setOnShowPythonConsole(std::function<void()> callback) {
        on_show_python_console_ = std::move(callback);
    }

    void MenuBar::renderPluginInstallPopup() {
        if (!show_plugin_install_popup_)
            return;

        lfs::python::ensure_initialized();

        const float scale = lfs::python::get_shared_dpi_scale();
        const auto& t = theme();

        ImGui::SetNextWindowSize({400.0f * scale, 0}, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f * scale);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f * scale, 16.0f * scale));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.surface, 0.98f));

        if (ImGui::Begin("Install Plugin", &show_plugin_install_popup_,
                         ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize)) {

            ImGui::Text("GitHub URL or shorthand:");
            ImGui::SetNextItemWidth(-1);

            char buf[512];
            strncpy(buf, plugin_install_url_.c_str(), sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText("##url", buf, sizeof(buf))) {
                plugin_install_url_ = buf;
            }

            ImGui::Spacing();
            ImGui::TextDisabled("Examples: owner/repo, github:owner/repo");
            ImGui::Spacing();

            if (!plugin_status_message_.empty()) {
                if (plugin_status_is_error_) {
                    ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%s", plugin_status_message_.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.3f, 1.0f), "%s", plugin_status_message_.c_str());
                }
                ImGui::Spacing();
            }

            if (widgets::ColoredButton("Install", widgets::ButtonStyle::Success, {100 * scale, 0})) {
                if (!plugin_install_url_.empty()) {
                    const lfs::python::GilAcquire gil;

                    PyObject* const lf_mod = PyImport_ImportModule("lichtfeld");
                    if (lf_mod) {
                        PyObject* const plugins_mod = PyObject_GetAttrString(lf_mod, "plugins");
                        if (plugins_mod) {
                            PyObject* const result = PyObject_CallMethod(plugins_mod, "install", "s", plugin_install_url_.c_str());
                            if (result) {
                                const char* const name = PyUnicode_AsUTF8(result);
                                plugin_status_message_ = std::string("Installed: ") + (name ? name : "");
                                plugin_status_is_error_ = false;
                                plugin_install_url_.clear();
                                Py_DECREF(result);
                            } else {
                                PyObject *type, *value, *tb;
                                PyErr_Fetch(&type, &value, &tb);
                                if (value) {
                                    PyObject* const str = PyObject_Str(value);
                                    if (str) {
                                        plugin_status_message_ = PyUnicode_AsUTF8(str);
                                        Py_DECREF(str);
                                    }
                                    Py_XDECREF(type);
                                    Py_XDECREF(value);
                                    Py_XDECREF(tb);
                                } else {
                                    plugin_status_message_ = "Install failed";
                                }
                                plugin_status_is_error_ = true;
                            }
                            Py_DECREF(plugins_mod);
                        }
                        Py_DECREF(lf_mod);
                    }
                    PyErr_Clear();
                }
            }
            ImGui::SameLine();
            if (widgets::ColoredButton("Cancel", widgets::ButtonStyle::Secondary, {100 * scale, 0})) {
                show_plugin_install_popup_ = false;
            }
        }
        ImGui::End();

        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }

} // namespace lfs::vis::gui
