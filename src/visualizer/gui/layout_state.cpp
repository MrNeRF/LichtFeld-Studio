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
        LOG_WARN("Unable to resolve user settings path: {}; layout persistence is disabled", paths.error());
        return {};
    }

    std::filesystem::path LayoutState::getConfigPath() {
        const auto config_dir = getConfigDir();
        return config_dir.empty() ? std::filesystem::path{} : config_dir / "layout.json";
    }

    void LayoutState::load(const bool log_success) {
        if (!g_persistence_enabled.load(std::memory_order_acquire))
            return;
        try {
            const auto path = getConfigPath();
            if (path.empty())
                return;
            if (!std::filesystem::exists(path))
                return;

            std::ifstream file(path);
            if (!file)
                return;
            if (file.peek() == std::ifstream::traits_type::eof())
                return;

            const auto j = nlohmann::json::parse(file);
            right_panel_width = j.value("right_panel_width", right_panel_width);
            scene_panel_ratio = j.value("scene_panel_ratio", scene_panel_ratio);
            python_console_width = j.value("python_console_width", python_console_width);
            bottom_dock_height = j.value("bottom_dock_height", bottom_dock_height);
            left_dock_width = j.value("left_dock_width", left_dock_width);
            show_sequencer = j.value("show_sequencer", show_sequencer);
            active_main_tab = j.value("active_main_tab", active_main_tab);
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

            if (log_success)
                LOG_INFO("Layout state loaded from {}", path.string());
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load layout state: {}", e.what());
        }
    }

    void LayoutState::setPersistenceEnabled(const bool enabled) noexcept {
        g_persistence_enabled.store(enabled, std::memory_order_release);
    }

} // namespace lfs::vis::gui
