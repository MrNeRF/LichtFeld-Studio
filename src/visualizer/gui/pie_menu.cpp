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
        constexpr int ARC_SEGMENTS = 32;

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

        using lfs::rendering::OverlayColor;
        using lfs::rendering::ScreenOverlayRenderer;

        OverlayColor toOverlay(const ImVec4& color, const float alpha) {
            return {color.x, color.y, color.z, alpha};
        }

        glm::vec2 polar(const ImVec2 center, const float radius, const float angle) {
            return {center.x + std::cos(angle) * radius, center.y + std::sin(angle) * radius};
        }

        std::vector<glm::vec2> arcPoints(const ImVec2 center, const float radius,
                                         const float a0, const float a1, const int segments) {
            std::vector<glm::vec2> pts;
            pts.reserve(static_cast<std::size_t>(segments) + 1);
            for (int s = 0; s <= segments; ++s) {
                const float t = static_cast<float>(s) / static_cast<float>(segments);
                pts.push_back(polar(center, radius, a0 + (a1 - a0) * t));
            }
            return pts;
        }

        // Filled annular band from a0 to a1 as a quad strip (a ring sector is not
        // convex, so a plain convex fill would be wrong).
        void fillArcBand(ScreenOverlayRenderer& overlay, const ImVec2 center,
                         const float inner_r, const float outer_r,
                         const float a0, const float a1, const int segments,
                         const OverlayColor color) {
            for (int s = 0; s < segments; ++s) {
                const float t0 = static_cast<float>(s) / static_cast<float>(segments);
                const float t1 = static_cast<float>(s + 1) / static_cast<float>(segments);
                const float b0 = a0 + (a1 - a0) * t0;
                const float b1 = a0 + (a1 - a0) * t1;
                const glm::vec2 i0 = polar(center, inner_r, b0);
                const glm::vec2 i1 = polar(center, inner_r, b1);
                const glm::vec2 o0 = polar(center, outer_r, b0);
                const glm::vec2 o1 = polar(center, outer_r, b1);
                overlay.addTriangleFilled(i0, o0, o1, color);
                overlay.addTriangleFilled(i0, o1, i1, color);
            }
        }

        // Triangle-fan disc. Emits Triangle commands so it stays in the same draw
        // layer as (and under) the sector fills; CircleFilled commands render in the
        // later UI-shape pass and would tint the fills.
        void fillCircleFan(ScreenOverlayRenderer& overlay, const ImVec2 center,
                           const float radius, const int segments, const OverlayColor color) {
            const glm::vec2 c{center.x, center.y};
            for (int s = 0; s < segments; ++s) {
                const float b0 = TWO_PI * static_cast<float>(s) / static_cast<float>(segments);
                const float b1 = TWO_PI * static_cast<float>(s + 1) / static_cast<float>(segments);
                overlay.addTriangleFilled(c, polar(center, radius, b0), polar(center, radius, b1), color);
            }
        }

        void strokeArc(ScreenOverlayRenderer& overlay, const ImVec2 center, const float radius,
                       const float a0, const float a1, const int segments,
                       const OverlayColor color, const float thickness) {
            const auto pts = arcPoints(center, radius, a0, a1, segments);
            overlay.addPolyline(pts, color, false, thickness);
        }

        void drawCenteredText(ScreenOverlayRenderer& overlay, const glm::vec2 center,
                              const std::string& text, const OverlayColor color,
                              const float font_size) {
            const glm::vec2 sz = overlay.measureText(text, font_size);
            overlay.addText({center.x - sz.x * 0.5f, center.y - sz.y * 0.5f}, text, color, font_size);
        }
    } // namespace

    float PieMenu::dpiScale() const {
        return getThemeDpiScale();
    }

    void PieMenu::open(ImVec2 center) {
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

    void PieMenu::onMouseMove(ImVec2 pos) {
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

    void PieMenu::onMouseClick(ImVec2 pos) {
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

    void PieMenu::drawSector(ScreenOverlayRenderer& overlay, int index, float a0, float a1, float scale) const {
        const auto& t = theme();
        const auto& item = items_[index];
        const float inner_r = INNER_RADIUS * scale;
        const float outer_r = OUTER_RADIUS * scale;

        OverlayColor fill_color;
        if (!item.enabled) {
            fill_color = toOverlay(t.palette.surface, 0.55f);
        } else if (index == hovered_sector_) {
            fill_color = toOverlay(t.palette.primary, 0.92f);
        } else if (item.is_active) {
            fill_color = toOverlay(t.palette.primary_dim, 0.80f);
        } else {
            fill_color = toOverlay(t.palette.surface, 0.92f);
        }

        fillArcBand(overlay, center_, inner_r, outer_r, a0, a1, ARC_SEGMENTS, fill_color);

        const OverlayColor border_col = toOverlay(t.palette.border, 0.50f);
        const float border_w = 1.0f * scale;
        overlay.addLine(polar(center_, inner_r, a0), polar(center_, outer_r, a0),
                        border_col, border_w);

        const float mid_angle = (a0 + a1) * 0.5f;
        const float icon_r = (inner_r + outer_r) * 0.5f;
        const glm::vec2 icon_center = polar(center_, icon_r, mid_angle);

        const OverlayColor text_col = item.enabled
                                          ? toOverlay(t.palette.text, 1.0f)
                                          : toOverlay(t.palette.text_dim, 0.40f);

        const std::uintptr_t icon_tex = IconCache::instance().getIcon(item.icon_name);
        const float icon_sz = ICON_SIZE * scale;
        const float font_size = t.fonts.base_size * scale;

        if (icon_tex != 0) {
            const glm::vec2 icon_min = {icon_center.x - icon_sz * 0.5f, icon_center.y - icon_sz * 0.5f};
            const glm::vec2 icon_max = {icon_min.x + icon_sz, icon_min.y + icon_sz};
            overlay.addImage(icon_tex, icon_min, icon_max, text_col);
        } else {
            drawCenteredText(overlay, icon_center, std::string(1, item.label[0]),
                             text_col, t.fonts.large_size * scale);
        }

        const float label_r = LABEL_RADIUS * scale;
        drawCenteredText(overlay, polar(center_, label_r, mid_angle), item.label,
                         text_col, font_size);
    }

    void PieMenu::drawSubmodeRing(ScreenOverlayRenderer& overlay, int sector, float scale) const {
        const auto& t = theme();
        const auto& item = items_[sector];
        const int sm_count = static_cast<int>(item.submodes.size());
        if (sm_count == 0)
            return;

        const int n = static_cast<int>(items_.size());
        const float sector_size = TWO_PI / static_cast<float>(n);
        const float offset = sectorAngleOffset(n);
        const float sector_mid = offset + (static_cast<float>(sector) + 0.5f) * sector_size;

        const float min_arc = SUBMODE_MIN_ARC_DEG * PI / 180.0f;
        const float total_arc = std::max(sector_size, static_cast<float>(sm_count) * min_arc);
        const float a0 = sector_mid - total_arc * 0.5f;
        const float sub_size = total_arc / static_cast<float>(sm_count);

        const float sm_inner = (OUTER_RADIUS + SUBMODE_GAP) * scale;
        const float sm_outer = sm_inner + SUBMODE_WIDTH * scale;
        const OverlayColor border_col = toOverlay(t.palette.border, 0.50f);
        const float border_w = 1.0f * scale;

        for (int si = 0; si < sm_count; ++si) {
            const float sa0 = a0 + static_cast<float>(si) * sub_size;
            const float sa1 = sa0 + sub_size;

            OverlayColor sm_fill;
            if (si == hovered_submode_) {
                sm_fill = toOverlay(t.palette.primary, 0.85f);
            } else {
                sm_fill = toOverlay(t.palette.surface, 0.70f);
            }

            fillArcBand(overlay, center_, sm_inner, sm_outer, sa0, sa1, ARC_SEGMENTS, sm_fill);

            overlay.addLine(polar(center_, sm_inner, sa0), polar(center_, sm_outer, sa0),
                            border_col, border_w);

            const float sm_mid = (sa0 + sa1) * 0.5f;
            const float sm_r = (sm_inner + sm_outer) * 0.5f;
            const glm::vec2 sm_center = polar(center_, sm_r, sm_mid);
            const OverlayColor sm_text_col = toOverlay(t.palette.text, 1.0f);
            const auto& submode = item.submodes[si];

            const std::uintptr_t sm_icon =
                submode.icon_name.empty() ? 0 : IconCache::instance().getIcon(submode.icon_name);
            if (sm_icon != 0) {
                const float sm_icon_sz = (sm_outer - sm_inner) * 0.65f;
                const glm::vec2 icon_min = {sm_center.x - sm_icon_sz * 0.5f,
                                            sm_center.y - sm_icon_sz * 0.5f};
                const glm::vec2 icon_max = {icon_min.x + sm_icon_sz, icon_min.y + sm_icon_sz};
                overlay.addImage(sm_icon, icon_min, icon_max, sm_text_col);
            } else {
                drawCenteredText(overlay, sm_center, submode.label, sm_text_col,
                                 t.fonts.base_size * scale);
            }
        }

        const float a1 = a0 + total_arc;

        overlay.addLine(polar(center_, sm_inner, a1), polar(center_, sm_outer, a1),
                        border_col, border_w);

        strokeArc(overlay, center_, sm_outer, a0, a1, ARC_SEGMENTS, border_col, border_w);
        strokeArc(overlay, center_, sm_inner, a0, a1, ARC_SEGMENTS, border_col, border_w);
    }

    void PieMenu::draw(ScreenOverlayRenderer& overlay) {
        if (!open_ || items_.empty())
            return;

        const auto& t = theme();
        const float scale = dpiScale();
        const float inner_r = INNER_RADIUS * scale;
        const float outer_r = OUTER_RADIUS * scale;
        const float dead_r = DEAD_ZONE_RADIUS * scale;

        const int n = static_cast<int>(items_.size());
        const float sector_size = TWO_PI / static_cast<float>(n);
        const float angle_offset = sectorAngleOffset(n);

        const glm::vec2 center{center_.x, center_.y};
        fillCircleFan(overlay, center_, outer_r + 2.0f * scale, 64,
                      toOverlay(t.palette.background, 0.30f));

        for (int i = 0; i < n; ++i) {
            const float a0 = angle_offset + static_cast<float>(i) * sector_size;
            const float a1 = a0 + sector_size;
            drawSector(overlay, i, a0, a1, scale);
        }

        const OverlayColor border_col = toOverlay(t.palette.border, 0.50f);
        overlay.addCircle(center, outer_r, border_col, 64, 1.0f * scale);
        overlay.addCircle(center, inner_r, border_col, 64, 1.0f * scale);

        overlay.addCircleFilled(center, dead_r, toOverlay(t.palette.background, 0.55f), 32);

        if (hovered_sector_ >= 0 && hovered_sector_ < n &&
            !items_[hovered_sector_].submodes.empty() && items_[hovered_sector_].enabled) {
            drawSubmodeRing(overlay, hovered_sector_, scale);
        }
    }

} // namespace lfs::vis::gui
