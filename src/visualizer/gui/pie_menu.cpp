/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/pie_menu.hpp"
#include "core/editor_context.hpp"
#include "gui/icon_cache.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "theme/theme.hpp"
#include "tools/tool_descriptor.hpp"
#include "tools/unified_tool_registry.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/glm.hpp>

namespace lfs::vis::gui {

    const std::string PieMenu::EMPTY_STRING;

    namespace {
        constexpr float PI = 3.14159265358979323846f;
        constexpr float TWO_PI = 2.0f * PI;

        float normalizeAngle(float a) {
            a = std::fmod(a, TWO_PI);
            if (a < 0.0f)
                a += TWO_PI;
            return a;
        }

        struct ToolEntry {
            const char* id;
            const char* label;
            const char* icon;
            ToolType tool_type;
        };

        constexpr ToolEntry TOOL_ORDER[] = {
            {"builtin.select", "Select", "selection", ToolType::Selection},
            {"builtin.translate", "Move", "translation", ToolType::Translate},
            {"builtin.rotate", "Rotate", "rotation", ToolType::Rotate},
            {"builtin.scale", "Scale", "scaling", ToolType::Scale},
            {"builtin.mirror", "Mirror", "mirror", ToolType::Mirror},
            {"builtin.align", "Align", "align", ToolType::Align},
            {"builtin.cropbox", "Crop Box", "cropbox", ToolType::None},
            {"builtin.ellipsoid", "Crop Ellipsoid", "blob", ToolType::None},
        };
        constexpr int TOOL_COUNT = sizeof(TOOL_ORDER) / sizeof(TOOL_ORDER[0]);

        struct SubmodeEntry {
            const char* id;
            const char* label;
            const char* icon;
        };

        constexpr SubmodeEntry SELECTION_SUBMODES[] = {
            {"centers", "Centers", ""},
            {"rectangle", "Rect", ""},
            {"polygon", "Poly", ""},
            {"lasso", "Lasso", ""},
            {"rings", "Rings", ""},
        };

        struct CropEntry {
            const char* id;
            const char* label;
            const char* icon;
        };

        constexpr CropEntry CROP_ITEMS[] = {
            {"crop.translate", "Move", "translation"},
            {"crop.rotate", "Rotate", "rotation"},
            {"crop.scale", "Scale", "scaling"},
            {"crop.apply", "Apply", "check"},
            {"crop.fit", "Fit", "arrows-maximize"},
            {"crop.fit_trim", "Fit Trim", "arrows-minimize"},
            {"crop.invert", "Invert", "contrast"},
            {"crop.reset", "Reset", "reset"},
            {"crop.delete", "Delete", "icon/scene/trash.png"},
        };
        constexpr int CROP_COUNT = sizeof(CROP_ITEMS) / sizeof(CROP_ITEMS[0]);

        constexpr SubmodeEntry MIRROR_SUBMODES[] = {
            {"x", "X", "mirror-x"},
            {"y", "Y", "mirror-y"},
            {"z", "Z", "mirror-z"},
        };

        float sectorAngleOffset(int count) {
            const float sector_size = TWO_PI / static_cast<float>(count);
            return normalizeAngle(-PI / 2.0f - sector_size / 2.0f);
        }

        lfs::rendering::OverlayColor toOverlay(const ThemeColor& c, const float alpha_scale = 1.0f) {
            return {c.x, c.y, c.z, c.w * alpha_scale};
        }

        std::vector<glm::vec2> sectorFan(const glm::vec2 center,
                                         const float radius,
                                         const float a0,
                                         const float a1,
                                         const int segments) {
            std::vector<glm::vec2> points;
            points.reserve(static_cast<size_t>(segments) + 2);
            points.push_back(center);
            for (int s = 0; s <= segments; ++s) {
                const float u = static_cast<float>(s) / static_cast<float>(segments);
                const float a = a0 + (a1 - a0) * u;
                points.push_back({center.x + std::cos(a) * radius,
                                  center.y + std::sin(a) * radius});
            }
            return points;
        }

        void addRingSector(lfs::rendering::ScreenOverlayRenderer& overlay,
                           const glm::vec2 center,
                           const float inner_radius,
                           const float outer_radius,
                           const float a0,
                           const float a1,
                           const int segments,
                           const lfs::rendering::OverlayColor color) {
            for (int s = 0; s < segments; ++s) {
                const float u0 = static_cast<float>(s) / static_cast<float>(segments);
                const float u1 = static_cast<float>(s + 1) / static_cast<float>(segments);
                const float p0 = a0 + (a1 - a0) * u0;
                const float p1 = a0 + (a1 - a0) * u1;
                const glm::vec2 i0{center.x + std::cos(p0) * inner_radius,
                                   center.y + std::sin(p0) * inner_radius};
                const glm::vec2 o0{center.x + std::cos(p0) * outer_radius,
                                   center.y + std::sin(p0) * outer_radius};
                const glm::vec2 i1{center.x + std::cos(p1) * inner_radius,
                                   center.y + std::sin(p1) * inner_radius};
                const glm::vec2 o1{center.x + std::cos(p1) * outer_radius,
                                   center.y + std::sin(p1) * outer_radius};
                overlay.addTriangleFilled(i0, o0, o1, color);
                overlay.addTriangleFilled(i0, o1, i1, color);
            }
        }

        void addCenteredText(lfs::rendering::ScreenOverlayRenderer& overlay,
                             const glm::vec2 center,
                             const std::string_view text,
                             const lfs::rendering::OverlayColor color,
                             const float size_px) {
            const glm::vec2 measured = overlay.measureText(text, size_px);
            overlay.addTextWithShadow({center.x - measured.x * 0.5f,
                                       center.y - measured.y * 0.5f},
                                      text, color, {0.0f, 0.0f, 0.0f, 0.45f}, size_px);
        }
    } // namespace

    float PieMenu::dpiScale() const {
        return getThemeDpiScale();
    }

    void PieMenu::open(glm::vec2 center) {
        center_ = center;
        open_ = true;
        hovered_sector_ = -1;
        hovered_submode_ = -1;
        selected_sector_ = -1;
        selected_submode_ = -1;
        mouse_moved_significantly_ = false;
        open_time_ = std::chrono::steady_clock::now();
    }

    void PieMenu::close() {
        open_ = false;
        hovered_sector_ = -1;
        hovered_submode_ = -1;
        selected_sector_ = -1;
        selected_submode_ = -1;
    }

    void PieMenu::updateItems(const EditorContext& editor) {
        items_.clear();

        const auto node_type = editor.getSelectedNodeType();
        if (node_type == core::NodeType::CROPBOX || node_type == core::NodeType::ELLIPSOID) {
            items_.reserve(CROP_COUNT);
            for (const auto& entry : CROP_ITEMS) {
                PieMenuItem item;
                item.id = entry.id;
                item.label = entry.label;
                item.icon_name = entry.icon;
                item.enabled = true;
                items_.push_back(std::move(item));
            }
            return;
        }

        items_.reserve(TOOL_COUNT);
        const auto& active_id = UnifiedToolRegistry::instance().getActiveTool();

        for (const auto& entry : TOOL_ORDER) {
            PieMenuItem item;
            item.id = entry.id;
            item.label = entry.label;
            item.icon_name = entry.icon;
            item.tool_type = entry.tool_type;

            if (entry.tool_type != ToolType::None) {
                item.enabled = editor.isToolAvailable(entry.tool_type);
                item.is_active = (active_id == entry.id);
            } else {
                item.enabled = editor.hasSelection() && !editor.isToolsDisabled();
                item.is_active = false;
            }

            if (entry.tool_type == ToolType::Selection) {
                for (const auto& sm : SELECTION_SUBMODES)
                    item.submodes.push_back({sm.id, sm.label, sm.icon});
            } else if (entry.tool_type == ToolType::Mirror) {
                for (const auto& sm : MIRROR_SUBMODES)
                    item.submodes.push_back({sm.id, sm.label, sm.icon});
            }

            items_.push_back(std::move(item));
        }
    }

    void PieMenu::onMouseMove(glm::vec2 pos) {
        const float dx = pos.x - center_.x;
        const float dy = pos.y - center_.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        const float scale = dpiScale();

        if (dist > GESTURE_MOUSE_THRESHOLD * scale)
            mouse_moved_significantly_ = true;

        if (items_.empty())
            return;

        const int n = static_cast<int>(items_.size());

        if (dist < DEAD_ZONE_RADIUS * scale) {
            hovered_sector_ = -1;
            hovered_submode_ = -1;
            return;
        }

        const float angle = normalizeAngle(std::atan2(dy, dx));
        const float sm_inner = (OUTER_RADIUS + SUBMODE_GAP) * scale;
        const float sm_outer = sm_inner + SUBMODE_WIDTH * scale;

        if (dist > sm_inner && dist < sm_outer && hovered_sector_ >= 0 &&
            hovered_sector_ < n && !items_[hovered_sector_].submodes.empty()) {
            hovered_submode_ = submodeFromAngle(angle, hovered_sector_);
        } else {
            hovered_sector_ = sectorFromAngle(angle, n);
            hovered_submode_ = -1;
        }
    }

    void PieMenu::onMouseClick(glm::vec2 pos) {
        onMouseMove(pos);
        if (hovered_sector_ >= 0 && hovered_sector_ < static_cast<int>(items_.size()) &&
            items_[hovered_sector_].enabled) {
            selected_sector_ = hovered_sector_;
            selected_submode_ = hovered_submode_;
        } else if (hovered_sector_ < 0) {
            close();
        }
    }

    void PieMenu::onKeyRelease() {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - open_time_);

        if (elapsed.count() < GESTURE_TIME_MS && !mouse_moved_significantly_)
            return;

        if (hovered_sector_ >= 0 && hovered_sector_ < static_cast<int>(items_.size()) &&
            items_[hovered_sector_].enabled) {
            selected_sector_ = hovered_sector_;
            selected_submode_ = hovered_submode_;
        } else {
            close();
        }
    }

    const std::string& PieMenu::getSelectedId() const {
        if (selected_sector_ >= 0 && selected_sector_ < static_cast<int>(items_.size()))
            return items_[selected_sector_].id;
        return EMPTY_STRING;
    }

    ToolType PieMenu::getSelectedToolType() const {
        if (selected_sector_ >= 0 && selected_sector_ < static_cast<int>(items_.size()))
            return items_[selected_sector_].tool_type;
        return ToolType::None;
    }

    const std::string& PieMenu::getSelectedSubmodeId() const {
        if (selected_sector_ >= 0 && selected_sector_ < static_cast<int>(items_.size()) &&
            selected_submode_ >= 0) {
            const auto& submodes = items_[selected_sector_].submodes;
            if (selected_submode_ < static_cast<int>(submodes.size()))
                return submodes[selected_submode_].id;
        }
        return EMPTY_STRING;
    }

    int PieMenu::sectorFromAngle(float angle, int count) const {
        assert(count > 0);
        const float sector_size = TWO_PI / static_cast<float>(count);
        const float offset = sectorAngleOffset(count);
        const float relative = normalizeAngle(angle - offset);
        return static_cast<int>(relative / sector_size) % count;
    }

    int PieMenu::submodeFromAngle(float angle, int parent_sector) const {
        assert(parent_sector >= 0 && parent_sector < static_cast<int>(items_.size()));
        const auto& submodes = items_[parent_sector].submodes;
        if (submodes.empty())
            return -1;

        const int n = static_cast<int>(items_.size());
        const int sm_count = static_cast<int>(submodes.size());
        const float sector_size = TWO_PI / static_cast<float>(n);
        const float offset = sectorAngleOffset(n);
        const float sector_mid = offset + (static_cast<float>(parent_sector) + 0.5f) * sector_size;

        const float min_arc = SUBMODE_MIN_ARC_DEG * PI / 180.0f;
        const float total_arc = std::max(sector_size, static_cast<float>(sm_count) * min_arc);
        const float sm_start = sector_mid - total_arc * 0.5f;
        const float sub_size = total_arc / static_cast<float>(sm_count);

        const float relative = normalizeAngle(angle - sm_start);
        if (relative > total_arc)
            return -1;

        return std::min(static_cast<int>(relative / sub_size), sm_count - 1);
    }

    void PieMenu::draw(lfs::rendering::ScreenOverlayRenderer& overlay) {
        if (!open_ || items_.empty())
            return;

        const auto& t = theme();
        const float scale = dpiScale();
        const float outer_r = OUTER_RADIUS * scale;
        const float dead_r = DEAD_ZONE_RADIUS * scale;
        const int n = static_cast<int>(items_.size());
        const float sector_size = TWO_PI / static_cast<float>(n);
        const float angle_offset = sectorAngleOffset(n);
        const auto border_col = toOverlay(t.palette.border, 0.50f);

        overlay.addCircleFilled(center_, outer_r + 2.0f * scale,
                                toOverlay(t.palette.background, 0.30f), 64);

        for (int i = 0; i < n; ++i) {
            const auto& item = items_[i];
            const float a0 = angle_offset + static_cast<float>(i) * sector_size;
            const float a1 = a0 + sector_size;

            lfs::rendering::OverlayColor fill;
            if (!item.enabled) {
                fill = toOverlay(t.palette.surface, 0.55f);
            } else if (i == hovered_sector_) {
                fill = toOverlay(t.palette.primary, 0.92f);
            } else if (item.is_active) {
                fill = toOverlay(t.palette.primary_dim, 0.80f);
            } else {
                fill = toOverlay(t.palette.surface, 0.92f);
            }

            const auto points = sectorFan(center_, outer_r, a0, a1, 24);
            overlay.addConvexPolyFilled(points, fill);
            overlay.addLine(center_,
                            {center_.x + std::cos(a0) * outer_r,
                             center_.y + std::sin(a0) * outer_r},
                            border_col, 1.0f * scale);

            const float mid_angle = (a0 + a1) * 0.5f;
            const float icon_r = (INNER_RADIUS + OUTER_RADIUS) * 0.5f * scale;
            const glm::vec2 icon_center{
                center_.x + std::cos(mid_angle) * icon_r,
                center_.y + std::sin(mid_angle) * icon_r};
            const auto text_col = item.enabled ? toOverlay(t.palette.text, 1.0f)
                                               : toOverlay(t.palette.text_dim, 0.40f);

            const std::uintptr_t icon_texture =
                IconCache::instance().getIcon(item.icon_name);
            if (icon_texture != 0) {
                const float icon_size = ICON_SIZE * scale;
                const glm::vec2 icon_min =
                    icon_center - glm::vec2(icon_size * 0.5f);
                overlay.addImage(icon_texture, icon_min,
                                 icon_min + glm::vec2(icon_size), text_col);
            } else if (!item.label.empty()) {
                const char initial[2] = {item.label[0], '\0'};
                addCenteredText(overlay, icon_center, initial, text_col, 16.0f * scale);
            }

            const float label_r = (OUTER_RADIUS + 18.0f) * scale;
            const glm::vec2 label_pos{
                center_.x + std::cos(mid_angle) * label_r,
                center_.y + std::sin(mid_angle) * label_r};
            addCenteredText(overlay, label_pos, item.label, text_col, 11.0f * scale);
        }

        overlay.addCircle(center_, outer_r, border_col, 64, 1.0f * scale);
        overlay.addCircleFilled(center_, dead_r, toOverlay(t.palette.background, 0.55f), 32);
        overlay.addCircle(center_, INNER_RADIUS * scale, border_col, 48, 1.0f * scale);

        if (hovered_sector_ < 0 || hovered_sector_ >= n ||
            items_[hovered_sector_].submodes.empty() || !items_[hovered_sector_].enabled) {
            return;
        }

        const auto& item = items_[hovered_sector_];
        const int sm_count = static_cast<int>(item.submodes.size());
        const float sector_mid = angle_offset + (static_cast<float>(hovered_sector_) + 0.5f) * sector_size;
        const float min_arc = SUBMODE_MIN_ARC_DEG * PI / 180.0f;
        const float total_arc = std::max(sector_size, static_cast<float>(sm_count) * min_arc);
        const float sub_size = total_arc / static_cast<float>(sm_count);
        const float sm_inner_radius = (OUTER_RADIUS + SUBMODE_GAP) * scale;
        const float sm_radius = sm_inner_radius + SUBMODE_WIDTH * scale;
        const float text_radius = (OUTER_RADIUS + SUBMODE_GAP + SUBMODE_WIDTH * 0.5f) * scale;
        const float start = sector_mid - total_arc * 0.5f;

        for (int si = 0; si < sm_count; ++si) {
            const float a0 = start + static_cast<float>(si) * sub_size;
            const float a1 = a0 + sub_size;
            const auto fill = si == hovered_submode_
                                  ? toOverlay(t.palette.primary, 0.85f)
                                  : toOverlay(t.palette.surface, 0.70f);
            addRingSector(overlay, center_, sm_inner_radius, sm_radius, a0, a1, 12, fill);
            const float mid = (a0 + a1) * 0.5f;
            const glm::vec2 item_center{
                center_.x + std::cos(mid) * text_radius,
                center_.y + std::sin(mid) * text_radius};
            const auto text_color = toOverlay(t.palette.text, 1.0f);
            const auto& submode = item.submodes[si];
            const std::uintptr_t icon_texture =
                submode.icon_name.empty()
                    ? 0
                    : IconCache::instance().getIcon(submode.icon_name);
            if (icon_texture != 0) {
                const float icon_size =
                    (sm_radius - sm_inner_radius) * 0.65f;
                const glm::vec2 icon_min =
                    item_center - glm::vec2(icon_size * 0.5f);
                overlay.addImage(icon_texture, icon_min,
                                 icon_min + glm::vec2(icon_size), text_color);
            } else {
                addCenteredText(overlay, item_center, submode.label,
                                text_color, 10.0f * scale);
            }
        }

        overlay.addCircle(center_, sm_radius, border_col, 64, 1.0f * scale);
        overlay.addCircle(center_, sm_inner_radius, border_col, 64, 1.0f * scale);
    }

} // namespace lfs::vis::gui
