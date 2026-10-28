/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "internal/viewport.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "tools/tool_base.hpp"
#include <optional>
#include <vector>

namespace lfs::vis::tools {

    [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c) {
        return {c.x, c.y, c.z, c.w};
    }

    [[nodiscard]] lfs::rendering::OverlayColor toOverlay(const auto& c, float alpha) {
        return {c.x, c.y, c.z, alpha};
    }

    [[nodiscard]] inline lfs::rendering::OverlayColor toOverlay(const lfs::rendering::OverlayColor& c, float alpha) {
        return {c.r, c.g, c.b, alpha};
    }

    struct PanelProjection {
        lfs::vis::RenderingManager::ViewerPanelInfo info{};
        Viewport viewport;
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        bool orthographic = false;
        float ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
        float screen_scale_x = 1.0f;
        float screen_scale_y = 1.0f;
    };
    struct ProjectedScreen {
        glm::vec2 pos{-1000.0f, -1000.0f};
        bool valid = false;
    };

    struct PointPickOverlayFrame {
        glm::vec2 mouse_pos;
        PanelProjection panel;
        bool has_panel;
        bool over_gui;
    };

    [[nodiscard]] std::string pointPickShortcut(input::Action action);
    [[nodiscard]] lfs::rendering::ScreenOverlayRenderer* getOverlayRenderer(const ToolContext& ctx);
    void drawLabel(lfs::rendering::ScreenOverlayRenderer& overlay, const glm::vec2& pos,
                   const char* text, const lfs::rendering::OverlayColor& color, float size);
    [[nodiscard]] ProjectedScreen projectToScreenChecked(const PanelProjection& proj, const glm::vec3& world_pos);
    [[nodiscard]] glm::vec2 projectToScreen(const PanelProjection& proj, const glm::vec3& world_pos);
    [[nodiscard]] PointPickOverlayFrame pointPickOverlayFrame(const ToolContext& ctx);
    void drawPointMarkers(lfs::rendering::ScreenOverlayRenderer& overlay, const PanelProjection& panel_proj,
                          const std::vector<glm::vec3>& picked_points, std::optional<int> selected_point,
                          const lfs::rendering::OverlayColor& color, float label_size);
    void drawPointPickHud(lfs::rendering::ScreenOverlayRenderer& overlay, const PanelProjection& panel_proj,
                          std::string_view hint, const std::string& count_text, const std::string* status,
                          std::string_view option_text = {});
    [[nodiscard]] std::optional<glm::vec3> drawPointPickHover(lfs::rendering::ScreenOverlayRenderer& overlay,
                                                              RenderingManager* rendering_manager,
                                                              const PointPickOverlayFrame& frame,
                                                              size_t point_count, size_t max_points, bool sample_hover,
                                                              const lfs::rendering::OverlayColor& color, float label_size);

} // namespace lfs::vis::tools
