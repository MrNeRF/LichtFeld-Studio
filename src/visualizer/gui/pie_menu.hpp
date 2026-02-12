/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <imgui.h>

namespace lfs::vis::gui {

    struct PieMenuSubmode {
        std::string id;
        std::string label;
        std::string icon_name;
    };

    struct PieMenuItem {
        std::string id;
        std::string label;
        std::string icon_name;
        ToolType tool_type = ToolType::None;
        bool enabled = true;
        bool is_active = false;
        std::vector<PieMenuSubmode> submodes;
    };

    class PieMenu {
    public:
        void open(ImVec2 center);
        void close();

        void updateItems(const EditorContext& editor);
        void draw(ImDrawList* drawlist);

        void onKeyRelease();
        void onMouseMove(ImVec2 pos);
        void onMouseClick(ImVec2 pos);

        [[nodiscard]] bool isOpen() const { return open_; }
        [[nodiscard]] bool hasSelection() const { return selected_sector_ >= 0; }
        [[nodiscard]] const std::string& getSelectedId() const;
        [[nodiscard]] const std::string& getSelectedSubmodeId() const;
        [[nodiscard]] ToolType getSelectedToolType() const;

    private:
        static constexpr float INNER_RADIUS = 38.0f;
        static constexpr float OUTER_RADIUS = 90.0f;
        static constexpr float DEAD_ZONE_RADIUS = 25.0f;
        static constexpr float LABEL_RADIUS = 108.0f;
        static constexpr float ICON_SIZE = 20.0f;
        static constexpr float SUBMODE_GAP = 10.0f;
        static constexpr float SUBMODE_WIDTH = 28.0f;
        static constexpr float SUBMODE_MIN_ARC_DEG = 24.0f;
        static constexpr float GESTURE_MOUSE_THRESHOLD = 8.0f;
        static constexpr int GESTURE_TIME_MS = 250;

        int sectorFromAngle(float angle, int count) const;
        int submodeFromAngle(float angle, int parent_sector) const;
        float dpiScale() const;

        void drawSector(ImDrawList* drawlist, int index, float a0, float a1, float scale) const;
        void drawSubmodeRing(ImDrawList* drawlist, int sector, float scale) const;

        bool open_ = false;
        ImVec2 center_{0, 0};
        std::vector<PieMenuItem> items_;

        int hovered_sector_ = -1;
        int hovered_submode_ = -1;
        int selected_sector_ = -1;
        int selected_submode_ = -1;

        std::chrono::steady_clock::time_point open_time_;
        bool mouse_moved_significantly_ = false;

        static const std::string EMPTY_STRING;
    };

} // namespace lfs::vis::gui
