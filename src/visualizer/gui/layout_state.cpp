/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/layout_state.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include <atomic>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::vis::gui {

    namespace {
        std::atomic<bool> g_persistence_enabled{true};
    }

    std::filesystem::path LayoutState::getConfigDir() {
        const auto paths = lfs::core::UserPaths::resolve();
        if (paths)
            return paths->configDir();
        LOG_WARN("Unable to resolve user settings path: {}; using local config directory", paths.error());
        return std::filesystem::current_path() / "config";
    }

    std::filesystem::path LayoutState::getConfigPath() {
        return getConfigDir() / "layout.json";
    }

    void LayoutState::save() const {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return;
        try {
            const auto path = getConfigPath();
            std::filesystem::create_directories(path.parent_path());

            nlohmann::json j;
            j["right_panel_width"] = right_panel_width;
            j["scene_panel_ratio"] = scene_panel_ratio;
            j["python_console_width"] = python_console_width;
            j["bottom_dock_height"] = bottom_dock_height;
            j["show_sequencer"] = show_sequencer;

            if (!file_association.empty())
                j["file_association"] = file_association;

            nlohmann::json windows;
            for (const auto& [name, visible] : window_visibility) {
                windows[name] = visible;
            }
            j["windows"] = windows;

            nlohmann::json vram_hud;
            vram_hud["x"] = vram_hud_x;
            vram_hud["y"] = vram_hud_y;
            vram_hud["width"] = vram_hud_width;
            vram_hud["height"] = vram_hud_height;
            vram_hud["active_tab"] = vram_hud_active_tab;
            vram_hud["collapsed"] = vram_hud_collapsed_paths;
            j["vram_hud"] = vram_hud;

            nlohmann::json perf_hud;
            perf_hud["visible"] = perf_hud_visible;
            perf_hud["expanded"] = perf_hud_expanded;
            j["perf_hud"] = perf_hud;

            std::ofstream file(path);
            if (file) {
                file << j.dump(2);
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save layout state: {}", e.what());
        } catch (...) {
            LOG_WARN("Failed to save layout state: unknown error");
        }
    }

    void LayoutState::load() {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return;
        try {
            const auto path = getConfigPath();
            if (!std::filesystem::exists(path))
                return;

            std::ifstream file(path);
            if (!file)
                return;

            const auto j = nlohmann::json::parse(file);
            right_panel_width = j.value("right_panel_width", right_panel_width);
            scene_panel_ratio = j.value("scene_panel_ratio", scene_panel_ratio);
            python_console_width = j.value("python_console_width", python_console_width);
            bottom_dock_height = j.value("bottom_dock_height", bottom_dock_height);
            show_sequencer = j.value("show_sequencer", show_sequencer);
            file_association = j.value("file_association", file_association);

            if (j.contains("windows") && j["windows"].is_object()) {
                for (const auto& [key, val] : j["windows"].items()) {
                    if (val.is_boolean()) {
                        window_visibility[key] = val.get<bool>();
                    }
                }
            }

            if (j.contains("vram_hud") && j["vram_hud"].is_object()) {
                const auto& vh = j["vram_hud"];
                vram_hud_x = vh.value("x", vram_hud_x);
                vram_hud_y = vh.value("y", vram_hud_y);
                vram_hud_width = vh.value("width", vram_hud_width);
                vram_hud_height = vh.value("height", vram_hud_height);
                vram_hud_active_tab = vh.value("active_tab", vram_hud_active_tab);
                if (vh.contains("collapsed") && vh["collapsed"].is_array()) {
                    vram_hud_collapsed_paths.clear();
                    for (const auto& entry : vh["collapsed"]) {
                        if (entry.is_string())
                            vram_hud_collapsed_paths.push_back(entry.get<std::string>());
                    }
                }
            }

            if (j.contains("perf_hud") && j["perf_hud"].is_object()) {
                const auto& ph = j["perf_hud"];
                perf_hud_visible = ph.value("visible", perf_hud_visible);
                perf_hud_expanded = ph.value("expanded", perf_hud_expanded);
            }

            LOG_INFO("Layout state loaded from {}", path.string());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load layout state: {}", e.what());
        }
    }

    void LayoutState::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

} // namespace lfs::vis::gui
