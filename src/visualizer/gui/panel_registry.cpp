/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_registry.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/ui_context.hpp"
#include "gui/ui_widgets.hpp"
#include "theme/theme.hpp"

#include <algorithm>
#include <optional>
#include <imgui.h>

namespace lfs::vis::gui {

    namespace {
        float floatingUiScale() {
            return std::max(1.0f, getThemeDpiScale());
        }

        float floatingResizeEdge() {
            return 6.0f * floatingUiScale();
        }

        float scaledFloatingDimensionForScale(const float value, const float scale) {
            if (value <= 0.0f)
                return value;
            return std::round(value * std::max(1.0f, scale));
        }

        void resetFloatingPanelSize(PanelInfo& panel, const float scale) {
            panel.initial_width = scaledFloatingDimensionForScale(panel.original_width, scale);
            panel.initial_height = scaledFloatingDimensionForScale(panel.original_height, scale);
            panel.float_user_height = 0.0f;
        }

        const char* panelSpaceName(const PanelSpace space) {
            switch (space) {
            case PanelSpace::SidePanel: return "side_panel";
            case PanelSpace::Floating: return "floating";
            case PanelSpace::ViewportOverlay: return "viewport_overlay";
            case PanelSpace::MainPanelTab: return "main_panel_tab";
            case PanelSpace::SceneHeader: return "scene_header";
            case PanelSpace::StatusBar: return "status_bar";
            }
            return "unknown";
        }

        bool requiresDirectWindowSurface(const PanelInfo& panel, const PanelSpace space) {
            return panel.parent_id.empty() &&
                   space == PanelSpace::Floating &&
                   !panel.has_option(PanelOption::SELF_MANAGED);
        }

        bool validatePanelContract(const PanelInfo& panel, const PanelSpace space) {
            if (!requiresDirectWindowSurface(panel, space))
                return true;

            if (panel.panel && panel.panel->supportsDirectDraw())
                return true;

            LOG_ERROR("Panel '{}' ({}) cannot use '{}' space without direct rendering. "
                      "Window panels must be self-managed or implement supportsDirectDraw().",
                      panel.label.empty() ? panel.id : panel.label,
                      panel.id,
                      panelSpaceName(space));
            return false;
        }

        bool pointInRoundedRect(const double x, const double y, const double w,
                                const double h, const double radius) {
            if (x < 0.0 || y < 0.0 || x >= w || y >= h)
                return false;

            const double clamped_radius = std::clamp(radius, 0.0, 0.5 * std::min(w, h));
            if (clamped_radius <= 0.0)
                return true;

            const auto inside_corner = [x, y](const double min_x, const double min_y,
                                              const double corner_radius,
                                              const double center_x,
                                              const double center_y) {
                if (x < min_x || x > min_x + corner_radius ||
                    y < min_y || y > min_y + corner_radius) {
                    return true;
                }

                const double dx = x - center_x;
                const double dy = y - center_y;
                return (dx * dx + dy * dy) <= (corner_radius * corner_radius);
            };

            return inside_corner(0.0, 0.0, clamped_radius, clamped_radius, clamped_radius) &&
                   inside_corner(w - clamped_radius, 0.0, clamped_radius,
                                 w - clamped_radius, clamped_radius) &&
                   inside_corner(w - clamped_radius, h - clamped_radius, clamped_radius,
                                 w - clamped_radius, h - clamped_radius) &&
                   inside_corner(0.0, h - clamped_radius, clamped_radius,
                                 clamped_radius, h - clamped_radius);
        }
    } // namespace

    PanelRegistry& PanelRegistry::instance() {
        static PanelRegistry registry;
        return registry;
    }

    uint64_t PanelRegistry::alloc_float_stack_order_locked() {
        return next_float_stack_order_++;
    }

    void PanelRegistry::ensure_float_stack_order_locked(PanelInfo& panel) {
        if (panel.space == PanelSpace::Floating && panel.float_stack_order == 0)
            panel.float_stack_order = alloc_float_stack_order_locked();
    }

    void PanelRegistry::bring_floating_panel_to_front_locked(PanelInfo& panel) {
        if (panel.space != PanelSpace::Floating)
            return;
        panel.float_stack_order = alloc_float_stack_order_locked();
    }

    bool PanelRegistry::register_panel(PanelInfo info) {
        std::lock_guard lock(mutex_);
        assert(info.panel);
        assert(!info.id.empty());

        if (!validatePanelContract(info, info.space))
            return false;

        info.original_width = info.initial_width;
        info.original_height = info.initial_height;
        if (info.space == PanelSpace::Floating)
            resetFloatingPanelSize(info, floatingUiScale());

        if (disabled_overrides_.contains(info.id))
            info.enabled = false;

        for (auto& p : panels_) {
            if (p.id == info.id) {
                if (info.space == PanelSpace::Floating && info.float_stack_order == 0 &&
                    p.float_stack_order != 0)
                    info.float_stack_order = p.float_stack_order;
                ensure_float_stack_order_locked(info);
                p = std::move(info);
                return true;
            }
        }

        ensure_float_stack_order_locked(info);
        panels_.push_back(std::move(info));
        std::stable_sort(panels_.begin(), panels_.end(), [](const PanelInfo& a, const PanelInfo& b) {
            if (a.order != b.order)
                return a.order < b.order;
            return a.label < b.label;
        });
        return true;
    }

    void PanelRegistry::unregister_panel(const std::string& id) {
        {
            std::lock_guard lock(mutex_);
            std::erase_if(panels_, [&id](const PanelInfo& p) { return p.id == id; });
        }
        {
            std::lock_guard poll_lock(poll_mutex_);
            poll_cache_.erase(id);
        }
    }

    void PanelRegistry::unregister_all_non_native() {
        std::vector<std::string> remaining;
        {
            std::lock_guard lock(mutex_);
            std::erase_if(panels_, [](const PanelInfo& p) { return !p.is_native; });
            remaining.reserve(panels_.size());
            for (const auto& p : panels_)
                remaining.push_back(p.id);
        }
        {
            std::lock_guard poll_lock(poll_mutex_);
            std::erase_if(poll_cache_, [&remaining](const auto& pair) {
                return std::none_of(remaining.begin(), remaining.end(),
                                    [&](const std::string& id) { return id == pair.first; });
            });
        }
    }

    bool PanelRegistry::check_poll(const PanelSnapshot& snap, const PanelDrawContext& ctx) {
        assert(snap.panel);
        if (snap.is_native)
            return snap.panel->poll(ctx);

        const uint64_t gen = ctx.scene_generation;
        const bool has_sel = ctx.has_selection;
        const bool training = ctx.is_training;

        {
            std::lock_guard poll_lock(poll_mutex_);
            auto cache_it = poll_cache_.find(snap.id);
            if (cache_it != poll_cache_.end()) {
                const auto& e = cache_it->second;
                bool valid = true;
                if ((snap.poll_dependencies & PollDependency::SCENE) != PollDependency::NONE)
                    valid &= (e.scene_generation == gen);
                if ((snap.poll_dependencies & PollDependency::SELECTION) != PollDependency::NONE)
                    valid &= (e.has_selection == has_sel);
                if ((snap.poll_dependencies & PollDependency::TRAINING) != PollDependency::NONE)
                    valid &= (e.is_training == training);
                if (valid)
                    return e.result;
            }
        }

        const bool result = snap.panel->poll(ctx);

        {
            std::lock_guard poll_lock(poll_mutex_);
            poll_cache_[snap.id] = {result, gen, has_sel, training, snap.poll_dependencies};
        }
        return result;
    }

    void PanelRegistry::draw_panels(PanelSpace space, const PanelDrawContext& ctx,
                                    const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            if (space == PanelSpace::Floating) {
                for (auto& p : panels_) {
                    if (p.space == PanelSpace::Floating) {
                        p.float_last_bounds_valid = false;
                    }
                }
            }
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y, p.order, p.float_stack_order});
                }
            }
        }

        if (space == PanelSpace::Floating) {
            std::stable_sort(snapshots.begin(), snapshots.end(),
                             [](const PanelSnapshot& a, const PanelSnapshot& b) {
                                 if (a.float_stack_order != b.float_stack_order)
                                     return a.float_stack_order < b.float_stack_order;
                                 if (a.order != b.order)
                                     return a.order < b.order;
                                 return a.label < b.label;
                             });
        }

        std::vector<bool> should_draw(snapshots.size(), false);
        for (size_t i = 0; i < snapshots.size(); ++i) {
            auto& snap = snapshots[i];
            try {
                should_draw[i] = check_poll(snap, ctx);
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
            }
        }

        struct FloatingDirectLayout {
            bool valid = false;
            float width = 0.0f;
            float height = 0.0f;
            float pos_x = 0.0f;
            float pos_y = 0.0f;
            float drawn_height = 0.0f;
            bool has_user_height = false;
            float forced_height = 0.0f;
            bool mouse_in_panel = false;
            bool mouse_in_titlebar = false;
            bool mouse_in_resize_grip = false;
            int8_t hover_dir_x = 0;
            int8_t hover_dir_y = 0;
        };

        std::vector<FloatingDirectLayout> floating_direct_layouts(snapshots.size());
        int hovered_floating_direct = -1;

        const float dpi = floatingUiScale();
        const float kTitleH = 30.0f * dpi;
        const float kResizeEdge = floatingResizeEdge();
        constexpr float kVisibleFrac = 0.1f;

        auto with_panel_input = [&](IPanel* panel, auto&& draw_fn) {
            panel->setInput(input);
            try {
                draw_fn();
            } catch (...) {
                panel->setInput(nullptr);
                throw;
            }
            panel->setInput(nullptr);
        };

        auto prepare_floating_direct_layout = [&](const PanelSnapshot& snap,
                                                  FloatingDirectLayout& layout) {
            layout = {};

            if (space != PanelSpace::Floating || !snap.panel->supportsDirectDraw() ||
                snap.has_option(PanelOption::SELF_MANAGED))
                return;

            auto* vp = ImGui::GetMainViewport();
            if (!vp)
                return;

            const float min_panel_width = 320.0f * dpi;
            const float max_panel_width = std::max(min_panel_width, vp->WorkSize.x);
            float w = snap.initial_width > 0 ? snap.initial_width : 560.0f * dpi;
            w = std::clamp(w, min_panel_width, max_panel_width);
            const float max_h = snap.initial_height > 0
                                    ? std::min(snap.initial_height, vp->WorkSize.y)
                                    : vp->WorkSize.y;
            float drawn_h = snap.panel->getDirectDrawHeight();
            if (drawn_h <= 0.0f) {
                float prev_h = -1.0f;
                for (int pass = 0; pass < 3; ++pass) {
                    with_panel_input(snap.panel, [&] {
                        snap.panel->preloadDirect(w, max_h, ctx, -1.0f, -1.0f, input);
                    });
                    drawn_h = snap.panel->getDirectDrawHeight();

                    const bool stable_height =
                        prev_h > 0.0f && std::abs(drawn_h - prev_h) <= 1.0f;
                    if (drawn_h > 0.0f && stable_height &&
                        !snap.panel->needsAnimationFrame())
                        break;

                    prev_h = drawn_h;
                }
            }

            float h = 0.0f;
            bool has_user_height = false;
            {
                std::lock_guard lock(mutex_);
                if (snap.index < panels_.size() && panels_[snap.index].id == snap.id &&
                    panels_[snap.index].float_user_height > 0) {
                    h = panels_[snap.index].float_user_height;
                    has_user_height = true;
                } else if (drawn_h > 0) {
                    h = std::min(drawn_h, max_h);
                } else if (snap.initial_height > 0) {
                    h = snap.initial_height;
                } else {
                    h = 400.0f * dpi;
                }
            }

            if (!has_user_height && drawn_h > 0 && h > drawn_h)
                h = drawn_h;

            float px = snap.float_x;
            float py = snap.float_y;
            if (std::isnan(px) || std::isnan(py)) {
                px = vp->WorkPos.x + (vp->WorkSize.x - w) * 0.5f;
                py = vp->WorkPos.y + (vp->WorkSize.y - h) * 0.5f;
            }

            const float vx = vp->WorkPos.x;
            const float vy = vp->WorkPos.y;
            const float vw = vp->WorkSize.x;
            const float vh = vp->WorkSize.y;
            px = std::clamp(px, vx - w * (1.0f - kVisibleFrac), vx + vw - w * kVisibleFrac);
            py = std::clamp(py, vy, vy + vh - kTitleH);

            layout.valid = true;
            layout.width = w;
            layout.height = h;
            layout.pos_x = px;
            layout.pos_y = py;
            layout.drawn_height = drawn_h;
            layout.has_user_height = has_user_height;
            layout.forced_height = (has_user_height && drawn_h > 0 && h > drawn_h) ? h : 0.0f;

            if (!input)
                return;

            const float mouse_x = input->mouse_x;
            const float mouse_y = input->mouse_y;
            layout.mouse_in_panel =
                mouse_x >= px && mouse_x < px + w && mouse_y >= py && mouse_y < py + h;
            layout.mouse_in_titlebar =
                mouse_x >= px && mouse_x < px + w && mouse_y >= py && mouse_y < py + kTitleH;

            const bool on_left =
                mouse_x >= px - kResizeEdge && mouse_x < px + kResizeEdge;
            const bool on_right =
                mouse_x >= px + w - kResizeEdge && mouse_x < px + w + kResizeEdge;
            const bool on_top =
                mouse_y >= py - kResizeEdge && mouse_y < py + kResizeEdge;
            const bool on_bottom =
                mouse_y >= py + h - kResizeEdge && mouse_y < py + h + kResizeEdge;
            const bool on_edge_x = on_left || on_right;
            const bool on_edge_y = on_top || on_bottom;
            const bool in_y_range =
                mouse_y >= py - kResizeEdge && mouse_y < py + h + kResizeEdge;
            const bool in_x_range =
                mouse_x >= px - kResizeEdge && mouse_x < px + w + kResizeEdge;

            layout.mouse_in_resize_grip =
                (on_edge_x && on_edge_y) ||
                (on_edge_x && !on_edge_y && in_y_range) ||
                (on_edge_y && !on_edge_x && in_x_range);
            layout.hover_dir_x = on_left ? int8_t(-1) : (on_right ? int8_t(1) : int8_t(0));
            layout.hover_dir_y = on_top ? int8_t(-1) : (on_bottom ? int8_t(1) : int8_t(0));
        };

        if (space == PanelSpace::Floating) {
            for (size_t i = 0; i < snapshots.size(); ++i) {
                if (should_draw[i])
                    prepare_floating_direct_layout(snapshots[i], floating_direct_layouts[i]);
            }

            if (input) {
                for (int i = static_cast<int>(snapshots.size()) - 1; i >= 0; --i) {
                    const auto& layout = floating_direct_layouts[static_cast<size_t>(i)];
                    if (!layout.valid)
                        continue;
                    if (layout.mouse_in_panel || layout.mouse_in_resize_grip) {
                        hovered_floating_direct = i;
                        break;
                    }
                }
            }
        }

        for (size_t snap_idx = 0; snap_idx < snapshots.size(); ++snap_idx) {
            auto& snap = snapshots[snap_idx];
            bool draw_succeeded = false;
            if (!should_draw[snap_idx])
                continue;

            try {
                ImGui::PushID(snap.id.c_str());

                switch (space) {
                case PanelSpace::Floating: {
                    if (snap.has_option(PanelOption::SELF_MANAGED)) {
                        with_panel_input(snap.panel, [&] { snap.panel->draw(ctx); });
                    } else if (snap.panel->supportsDirectDraw()) {
                        auto& layout = floating_direct_layouts[snap_idx];
                        if (!layout.valid)
                            prepare_floating_direct_layout(snap, layout);

                        float w = layout.width;
                        float h = layout.height;
                        float px = layout.pos_x;
                        float py = layout.pos_y;
                        float drawn_h = layout.drawn_height;
                        bool has_user_height = layout.has_user_height;

                        const float kMinPanelWidth = 320.0f * dpi;
                        const float kMinPanelHeight = 180.0f * dpi;

                        {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].id == snap.id) {
                                auto& pi = panels_[snap.index];
                                const bool active_this_panel = pi.float_dragging || pi.float_resizing;
                                const bool hovered_this_panel =
                                    static_cast<int>(snap_idx) == hovered_floating_direct;
                                const bool interactive = active_this_panel || hovered_this_panel;
                                const bool mouse_clicked_left = input && input->mouse_clicked[0];
                                const bool mouse_down_left = input && input->mouse_down[0];
                                const float mouse_x = input ? input->mouse_x : px;
                                const float mouse_y = input ? input->mouse_y : py;

                                bool any_active = std::any_of(panels_.begin(), panels_.end(),
                                                              [](const PanelInfo& p) { return p.float_dragging || p.float_resizing; });

                                if (interactive && (layout.mouse_in_panel || layout.mouse_in_resize_grip) &&
                                    mouse_clicked_left) {
                                    bring_floating_panel_to_front_locked(pi);
                                }

                                if (interactive && layout.mouse_in_resize_grip && !any_active &&
                                    mouse_clicked_left) {
                                    pi.float_resizing = true;
                                    pi.float_resize_start_w = w;
                                    pi.float_resize_start_h = h;
                                    pi.float_resize_start_mx = mouse_x;
                                    pi.float_resize_start_my = mouse_y;
                                    pi.float_resize_start_px = px;
                                    pi.float_resize_start_py = py;
                                    pi.float_resize_dir_x = layout.hover_dir_x;
                                    pi.float_resize_dir_y = layout.hover_dir_y;
                                } else if (interactive && layout.mouse_in_titlebar &&
                                           !layout.mouse_in_resize_grip && !any_active &&
                                           mouse_clicked_left) {
                                    pi.float_dragging = true;
                                    pi.float_drag_ox = mouse_x - px;
                                    pi.float_drag_oy = mouse_y - py;
                                }

                                if (pi.float_dragging) {
                                    if (mouse_down_left) {
                                        px = mouse_x - pi.float_drag_ox;
                                        py = mouse_y - pi.float_drag_oy;
                                    } else {
                                        pi.float_dragging = false;
                                    }
                                }
                                if (pi.float_resizing) {
                                    if (mouse_down_left) {
                                        const float dx = mouse_x - pi.float_resize_start_mx;
                                        const float dy = mouse_y - pi.float_resize_start_my;
                                        if (pi.float_resize_dir_x == 1) {
                                            w = std::max(kMinPanelWidth, pi.float_resize_start_w + dx);
                                            pi.initial_width = w;
                                        } else if (pi.float_resize_dir_x == -1) {
                                            w = std::max(kMinPanelWidth, pi.float_resize_start_w - dx);
                                            px = pi.float_resize_start_px + pi.float_resize_start_w - w;
                                            pi.initial_width = w;
                                        }
                                        if (pi.float_resize_dir_y == 1) {
                                            h = std::max(kMinPanelHeight, pi.float_resize_start_h + dy);
                                            pi.float_user_height = h;
                                        } else if (pi.float_resize_dir_y == -1) {
                                            h = std::max(kMinPanelHeight, pi.float_resize_start_h - dy);
                                            py = pi.float_resize_start_py + pi.float_resize_start_h - h;
                                            pi.float_user_height = h;
                                        }
                                    } else {
                                        pi.float_resizing = false;
                                        pi.float_resize_dir_x = 0;
                                        pi.float_resize_dir_y = 0;
                                    }
                                }

                                if (!pi.float_resizing && pi.float_user_height > 0 &&
                                    snap.panel->getDirectDrawHeight() <= 0) {
                                    pi.float_user_height = 0;
                                }
                                has_user_height = pi.float_user_height > 0.0f;

                                const auto* vp = ImGui::GetMainViewport();
                                const float vx = vp->WorkPos.x;
                                const float vy = vp->WorkPos.y;
                                const float vw = vp->WorkSize.x;
                                const float vh = vp->WorkSize.y;
                                px = std::clamp(px, vx - w * (1.0f - kVisibleFrac), vx + vw - w * kVisibleFrac);
                                py = std::clamp(py, vy, vy + vh - kTitleH);

                                pi.float_x = px;
                                pi.float_y = py;
                            }
                        }

                        {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].id == snap.id) {
                                const auto& pi = panels_[snap.index];
                                const bool interactive =
                                    pi.float_dragging || pi.float_resizing ||
                                    static_cast<int>(snap_idx) == hovered_floating_direct;

                                panels_[snap.index].float_last_bounds_valid = true;
                                panels_[snap.index].float_last_x = px;
                                panels_[snap.index].float_last_y = py;
                                panels_[snap.index].float_last_w = w;
                                panels_[snap.index].float_last_h = h;

                                if (pi.float_dragging || pi.float_resizing ||
                                    (interactive &&
                                     (layout.mouse_in_panel || layout.mouse_in_resize_grip))) {
                                    guiFocusState().want_capture_mouse = true;
                                }

                                if (interactive) {
                                    const int8_t dx =
                                        pi.float_resizing ? pi.float_resize_dir_x : layout.hover_dir_x;
                                    const int8_t dy =
                                        pi.float_resizing ? pi.float_resize_dir_y : layout.hover_dir_y;
                                    if (dx && dy) {
                                        const bool nw_se = (dx == dy);
                                        ImGui::SetMouseCursor(nw_se ? ImGuiMouseCursor_ResizeNWSE
                                                                    : ImGuiMouseCursor_ResizeNESW);
                                    } else if (dx) {
                                        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                                    } else if (dy) {
                                        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                                    }
                                }
                            }
                        }

                        const float forced = (has_user_height && drawn_h > 0 && h > drawn_h) ? h : 0.0f;
                        snap.panel->setForcedHeight(forced);
                        try {
                            with_panel_input(snap.panel, [&] {
                                if (snap.panel->wantsExternalFloatingShadow()) {
                                    widgets::DrawFloatingWindowShadow({px, py}, {w, h},
                                                                      theme().sizes.window_rounding);
                                }
                                snap.panel->drawDirect(px, py, w, h, ctx);
                            });
                        } catch (...) {
                            snap.panel->setForcedHeight(0.0f);
                            throw;
                        }
                        snap.panel->setForcedHeight(0.0f);
                    } else {
                        LOG_ERROR("Panel '{}' ({}) reached floating draw without direct rendering. "
                                  "Disabling the invalid panel instance.",
                                  snap.label, snap.id);
                        std::lock_guard lock(mutex_);
                        if (snap.index < panels_.size() && panels_[snap.index].id == snap.id)
                            panels_[snap.index].error_disabled = true;
                    }
                    break;
                }
                case PanelSpace::SidePanel: {
                    const ImGuiTreeNodeFlags flags = snap.has_option(PanelOption::DEFAULT_CLOSED)
                                                         ? ImGuiTreeNodeFlags_None
                                                         : ImGuiTreeNodeFlags_DefaultOpen;
                    if (snap.has_option(PanelOption::HIDE_HEADER)) {
                        snap.panel->draw(ctx);
                    } else if (ImGui::CollapsingHeader(snap.label.c_str(), flags)) {
                        snap.panel->draw(ctx);
                    }
                    break;
                }
                case PanelSpace::ViewportOverlay:
                case PanelSpace::SceneHeader:
                    snap.panel->draw(ctx);
                    break;
                case PanelSpace::StatusBar:
                    with_panel_input(snap.panel, [&] { snap.panel->draw(ctx); });
                    break;
                case PanelSpace::MainPanelTab:
                    break;
                }

                ImGui::PopID();
                draw_succeeded = true;
            } catch (const std::exception& e) {
                ImGui::PopID();
                LOG_ERROR("Panel '{}' draw error: {}", snap.label, e.what());
            }

            track_draw_result(snap, draw_succeeded);
        }
    }

    void PanelRegistry::preload_panels(PanelSpace space, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y, p.order, p.float_stack_order});
                }
            }
        }

        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preload poll error: {}", snap.label, e.what());
                continue;
            }

            try {
                snap.panel->preload(ctx);
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preload error: {}", snap.label, e.what());
            }
        }
    }

    bool PanelRegistry::isPositionOverFloatingPanel(const double x, const double y) const {
        const double kResizeEdge = static_cast<double>(floatingResizeEdge());
        const double rounding = std::max(0.0f, theme().sizes.window_rounding);

        std::lock_guard lock(mutex_);
        for (const auto& panel : panels_) {
            if (panel.space != PanelSpace::Floating || !panel.enabled || panel.error_disabled ||
                !panel.parent_id.empty() || !panel.float_last_bounds_valid) {
                continue;
            }

            const double panel_x = static_cast<double>(panel.float_last_x);
            const double panel_y = static_cast<double>(panel.float_last_y);
            const double panel_w = static_cast<double>(panel.float_last_w);
            const double panel_h = static_cast<double>(panel.float_last_h);

            if (x < panel_x - kResizeEdge || x >= panel_x + panel_w + kResizeEdge ||
                y < panel_y - kResizeEdge || y >= panel_y + panel_h + kResizeEdge) {
                continue;
            }

            if (x < panel_x || x >= panel_x + panel_w ||
                y < panel_y || y >= panel_y + panel_h) {
                return true;
            }

            const double local_x = x - panel_x;
            const double local_y = y - panel_y;
            if (pointInRoundedRect(local_x, local_y, panel_w, panel_h, rounding))
                return true;

            if (local_x < kResizeEdge || local_x >= panel_w - kResizeEdge ||
                local_y < kResizeEdge || local_y >= panel_h - kResizeEdge) {
                return true;
            }
        }

        return false;
    }

    float PanelRegistry::draw_panels_direct(PanelSpace space, float x, float y, float w,
                                            float max_h, const PanelDrawContext& ctx,
                                            const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y});
                }
            }
        }

        float y_offset = 0.0f;
        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
                continue;
            }

            const float remaining = max_h - y_offset;
            if (remaining <= 0)
                break;

            bool draw_succeeded = false;
            try {
                snap.panel->setInput(input);
                snap.panel->drawDirect(x, y + y_offset, w, remaining, ctx);
                snap.panel->setInput(nullptr);
                const float h = snap.panel->getDirectDrawHeight();
                y_offset += h > 0 ? h : remaining;
                draw_succeeded = true;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' drawDirect error: {}", snap.label, e.what());
            }

            track_draw_result(snap, draw_succeeded);
        }
        return y_offset;
    }

    void PanelRegistry::preload_panels_direct(PanelSpace space, float w, float max_h,
                                              const PanelDrawContext& ctx,
                                              float clip_y_min, float clip_y_max,
                                              const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y});
                }
            }
        }

        float y_offset = 0.0f;
        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preloadDirect poll error: {}", snap.label, e.what());
                continue;
            }

            const float remaining = max_h - y_offset;
            if (remaining <= 0)
                break;

            bool preload_succeeded = false;
            try {
                snap.panel->setInputClipY(clip_y_min, clip_y_max);
                snap.panel->setInput(input);
                snap.panel->preloadDirect(w, remaining, ctx, clip_y_min, clip_y_max, input);
                snap.panel->setInput(nullptr);
                snap.panel->setInputClipY(-1.0f, -1.0f);
                const float used = snap.panel->getDirectDrawHeight();
                y_offset += used > 0 ? used : remaining;
                preload_succeeded = true;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preloadDirect error: {}", snap.label, e.what());
            }

            track_draw_result(snap, preload_succeeded);
        }
    }

    void PanelRegistry::draw_single_panel(const std::string& id, const PanelDrawContext& ctx) {
        std::shared_ptr<IPanel> panel_holder;
        PanelSnapshot snap{};
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                if (panels_[i].id == id && panels_[i].enabled && !panels_[i].error_disabled) {
                    panel_holder = panels_[i].panel;
                    snap = {i, panels_[i].panel.get(), panels_[i].label, panels_[i].id,
                            panels_[i].parent_id, panels_[i].options, panels_[i].is_native,
                            panels_[i].poll_dependencies, panels_[i].initial_width, panels_[i].initial_height,
                            panels_[i].float_x, panels_[i].float_y};
                    found = true;
                    break;
                }
            }
        }

        if (!found)
            return;

        try {
            if (!check_poll(snap, ctx))
                return;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
            return;
        }

        bool draw_succeeded = false;
        try {
            ImGui::PushID(snap.id.c_str());
            snap.panel->draw(ctx);
            ImGui::PopID();
            draw_succeeded = true;
        } catch (const std::exception& e) {
            ImGui::PopID();
            LOG_ERROR("Panel '{}' error: {}", snap.label, e.what());
        }

        track_draw_result(snap, draw_succeeded);
    }

    bool PanelRegistry::has_panels(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty())
                return true;
        }
        return false;
    }

    std::vector<PanelSummary> PanelRegistry::get_panels_for_space(PanelSpace space) {
        std::lock_guard lock(mutex_);
        std::vector<PanelSummary> result;
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled && !p.error_disabled && p.parent_id.empty())
                result.push_back({p.label, p.id, p.space, p.order, p.enabled});
        }
        std::stable_sort(result.begin(), result.end(), [](const PanelSummary& a, const PanelSummary& b) {
            if (a.order != b.order)
                return a.order < b.order;
            return a.label < b.label;
        });
        return result;
    }

    std::optional<PanelDetails> PanelRegistry::get_panel(const std::string& id) {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.id == id)
                return PanelDetails{
                    p.label,
                    p.id,
                    p.parent_id,
                    p.space,
                    p.order,
                    p.enabled,
                    p.options,
                    p.poll_dependencies,
                    p.is_native,
                    p.initial_width,
                    p.initial_height,
                };
        }
        return std::nullopt;
    }

    std::vector<std::string> PanelRegistry::get_panel_names(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> names;
        for (const auto& p : panels_) {
            if (p.space == space)
                names.push_back(p.id);
        }
        return names;
    }

    void PanelRegistry::set_panel_enabled(const std::string& id, bool enabled) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.id == id) {
                p.enabled = enabled;
                if (enabled && p.space == PanelSpace::Floating) {
                    p.float_x = NAN;
                    p.float_y = NAN;
                    resetFloatingPanelSize(p, floatingUiScale());
                    bring_floating_panel_to_front_locked(p);
                } else if (!enabled) {
                    p.float_last_bounds_valid = false;
                    guiFocusState().want_capture_mouse = false;
                    guiFocusState().want_capture_keyboard = false;
                    guiFocusState().want_text_input = false;
                }
                return;
            }
        }
    }

    void PanelRegistry::set_panel_disabled_override(const std::string& id) {
        std::lock_guard lock(mutex_);
        disabled_overrides_.insert(id);
        for (auto& p : panels_) {
            if (p.id == id) {
                p.enabled = false;
                p.float_last_bounds_valid = false;
                guiFocusState().want_capture_mouse = false;
                guiFocusState().want_capture_keyboard = false;
                guiFocusState().want_text_input = false;
                return;
            }
        }
    }

    bool PanelRegistry::is_panel_enabled(const std::string& id) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.id == id)
                return p.enabled;
        }
        return false;
    }

    void PanelRegistry::rescale_floating_panels(float previous_scale, float new_scale) {
        previous_scale = std::max(previous_scale, 1.0f);
        new_scale = std::max(new_scale, 1.0f);
        if (std::abs(previous_scale - new_scale) < 0.001f)
            return;

        const float ratio = new_scale / previous_scale;

        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.space != PanelSpace::Floating)
                continue;

            if (p.initial_width > 0.0f)
                p.initial_width = std::round(p.initial_width * ratio);
            else if (p.original_width > 0.0f)
                p.initial_width = scaledFloatingDimensionForScale(p.original_width, new_scale);

            if (p.initial_height > 0.0f)
                p.initial_height = std::round(p.initial_height * ratio);
            else if (p.original_height > 0.0f)
                p.initial_height = scaledFloatingDimensionForScale(p.original_height, new_scale);

            if (p.float_user_height > 0.0f)
                p.float_user_height = std::max(1.0f, std::round(p.float_user_height * ratio));

            p.float_dragging = false;
            p.float_resizing = false;
            p.float_resize_dir_x = 0;
            p.float_resize_dir_y = 0;
            p.float_last_bounds_valid = false;
        }
    }

    bool PanelRegistry::needsAnimationFrame() const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (!p.enabled || p.error_disabled || !p.panel)
                continue;
            if (p.panel->needsAnimationFrame())
                return true;
        }
        return false;
    }

    bool PanelRegistry::set_panel_label(const std::string& id, const std::string& new_label) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.id == id) {
                p.label = new_label;
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::set_panel_order(const std::string& id, int new_order) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.id == id) {
                p.order = new_order;
                std::stable_sort(panels_.begin(), panels_.end(), [](const PanelInfo& a, const PanelInfo& b) {
                    if (a.order != b.order)
                        return a.order < b.order;
                    return a.label < b.label;
                });
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::set_panel_space(const std::string& id, PanelSpace new_space) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.id == id) {
                if (!validatePanelContract(p, new_space))
                    return false;
                const bool was_floating = p.space == PanelSpace::Floating;
                p.space = new_space;
                if (!was_floating && new_space == PanelSpace::Floating) {
                    p.float_x = NAN;
                    p.float_y = NAN;
                    resetFloatingPanelSize(p, floatingUiScale());
                    bring_floating_panel_to_front_locked(p);
                } else if (was_floating && new_space != PanelSpace::Floating) {
                    p.initial_width = p.original_width;
                    p.initial_height = p.original_height;
                    p.float_user_height = 0.0f;
                    p.float_last_bounds_valid = false;
                }
                ensure_float_stack_order_locked(p);
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::set_panel_parent(const std::string& id, const std::string& parent_id) {
        std::lock_guard lock(mutex_);

        if (!parent_id.empty()) {
            bool parent_found = false;
            for (const auto& p : panels_) {
                if (p.id == parent_id) {
                    parent_found = true;
                    break;
                }
            }
            if (!parent_found)
                LOG_WARN("Panel '{}': parent '{}' not registered (may register later)", id, parent_id);
        }

        for (auto& p : panels_) {
            if (p.id == id) {
                PanelInfo candidate = p;
                candidate.parent_id = parent_id;
                if (!validatePanelContract(candidate, candidate.space))
                    return false;
                p.parent_id = parent_id;
                return true;
            }
        }
        return false;
    }

    void PanelRegistry::draw_child_panels(const std::string& parent_id, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.parent_id == parent_id && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y});
                }
            }
        }

        for (auto& snap : snapshots) {
            bool draw_succeeded = false;
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
                continue;
            }

            try {
                ImGui::PushID(snap.id.c_str());

                if (snap.has_option(PanelOption::HIDE_HEADER)) {
                    snap.panel->draw(ctx);
                } else {
                    const ImGuiTreeNodeFlags flags = snap.has_option(PanelOption::DEFAULT_CLOSED)
                                                         ? ImGuiTreeNodeFlags_None
                                                         : ImGuiTreeNodeFlags_DefaultOpen;
                    if (ImGui::CollapsingHeader(snap.label.c_str(), flags)) {
                        snap.panel->draw(ctx);
                    }
                }

                ImGui::PopID();
                draw_succeeded = true;
            } catch (const std::exception& e) {
                ImGui::PopID();
                LOG_ERROR("Panel '{}' draw error: {}", snap.label, e.what());
            }

            track_draw_result(snap, draw_succeeded);
        }
    }

    float PanelRegistry::draw_single_panel_direct(const std::string& id, float x, float y,
                                                  float w, float h, const PanelDrawContext& ctx,
                                                  float clip_y_min, float clip_y_max,
                                                  const PanelInputState* input) {
        std::shared_ptr<IPanel> panel_holder;
        PanelSnapshot snap{};
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                if (panels_[i].id == id && panels_[i].enabled && !panels_[i].error_disabled) {
                    panel_holder = panels_[i].panel;
                    snap = {i, panels_[i].panel.get(), panels_[i].label, panels_[i].id,
                            panels_[i].parent_id, panels_[i].options, panels_[i].is_native,
                            panels_[i].poll_dependencies, panels_[i].initial_width, panels_[i].initial_height,
                            panels_[i].float_x, panels_[i].float_y};
                    found = true;
                    break;
                }
            }
        }

        if (!found)
            return 0.0f;

        try {
            if (!check_poll(snap, ctx))
                return 0.0f;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
            return 0.0f;
        }

        bool draw_succeeded = false;
        try {
            snap.panel->setInputClipY(clip_y_min, clip_y_max);
            snap.panel->setInput(input);
            snap.panel->drawDirect(x, y, w, h, ctx);
            snap.panel->setInput(nullptr);
            snap.panel->setInputClipY(-1.0f, -1.0f);
            draw_succeeded = true;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' drawDirect error: {}", snap.label, e.what());
        }

        track_draw_result(snap, draw_succeeded);
        const float used = snap.panel->getDirectDrawHeight();
        return used > 0 ? used : 0.0f;
    }

    float PanelRegistry::preload_single_panel_direct(const std::string& id, float w, float h,
                                                     const PanelDrawContext& ctx,
                                                     float clip_y_min, float clip_y_max,
                                                     const PanelInputState* input) {
        std::shared_ptr<IPanel> panel_holder;
        PanelSnapshot snap{};
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                if (panels_[i].id == id && panels_[i].enabled && !panels_[i].error_disabled) {
                    panel_holder = panels_[i].panel;
                    snap = {i, panels_[i].panel.get(), panels_[i].label, panels_[i].id,
                            panels_[i].parent_id, panels_[i].options, panels_[i].is_native,
                            panels_[i].poll_dependencies, panels_[i].initial_width, panels_[i].initial_height,
                            panels_[i].float_x, panels_[i].float_y};
                    found = true;
                    break;
                }
            }
        }

        if (!found)
            return 0.0f;

        try {
            if (!check_poll(snap, ctx))
                return 0.0f;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' preloadDirect poll error: {}", snap.label, e.what());
            return 0.0f;
        }

        bool preload_succeeded = false;
        try {
            snap.panel->setInputClipY(clip_y_min, clip_y_max);
            snap.panel->setInput(input);
            snap.panel->preloadDirect(w, h, ctx, clip_y_min, clip_y_max, input);
            snap.panel->setInput(nullptr);
            snap.panel->setInputClipY(-1.0f, -1.0f);
            preload_succeeded = true;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' preloadDirect error: {}", snap.label, e.what());
        }

        track_draw_result(snap, preload_succeeded);
        const float used = snap.panel->getDirectDrawHeight();
        return used > 0.0f ? used : 0.0f;
    }

    float PanelRegistry::preload_child_panels_direct(const std::string& parent_id, float w, float h,
                                                     const PanelDrawContext& ctx,
                                                     float clip_y_min, float clip_y_max,
                                                     const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.parent_id == parent_id && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y});
                }
            }
        }

        float y_offset = 0.0f;
        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preloadDirect poll error: {}", snap.label, e.what());
                continue;
            }

            const float remaining = h - y_offset;
            if (remaining <= 0)
                break;

            bool preload_succeeded = false;
            try {
                snap.panel->setInputClipY(clip_y_min, clip_y_max);
                snap.panel->setInput(input);
                snap.panel->preloadDirect(w, remaining, ctx, clip_y_min, clip_y_max, input);
                snap.panel->setInput(nullptr);
                snap.panel->setInputClipY(-1.0f, -1.0f);
                const float used = snap.panel->getDirectDrawHeight();
                y_offset += used > 0 ? used : remaining;
                preload_succeeded = true;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' preloadDirect error: {}", snap.label, e.what());
            }

            track_draw_result(snap, preload_succeeded);
        }

        return y_offset;
    }

    float PanelRegistry::draw_child_panels_direct(const std::string& parent_id, float x, float y,
                                                  float w, float h, const PanelDrawContext& ctx,
                                                  float clip_y_min, float clip_y_max,
                                                  const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.parent_id == parent_id && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.id,
                                         p.parent_id, p.options, p.is_native,
                                         p.poll_dependencies, p.initial_width, p.initial_height,
                                         p.float_x, p.float_y});
                }
            }
        }

        float y_offset = 0.0f;
        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' poll error: {}", snap.label, e.what());
                continue;
            }

            const float remaining = h - y_offset;
            if (remaining <= 0)
                break;

            bool draw_succeeded = false;
            try {
                snap.panel->setInputClipY(clip_y_min, clip_y_max);
                snap.panel->setInput(input);
                snap.panel->drawDirect(x, y + y_offset, w, remaining, ctx);
                snap.panel->setInput(nullptr);
                snap.panel->setInputClipY(-1.0f, -1.0f);
                const float used = snap.panel->getDirectDrawHeight();
                y_offset += used > 0 ? used : remaining;
                draw_succeeded = true;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' drawDirect error: {}", snap.label, e.what());
            }

            track_draw_result(snap, draw_succeeded);
        }
        return y_offset;
    }

    void PanelRegistry::track_draw_result(const PanelSnapshot& snap, bool draw_succeeded) {
        if (snap.is_native)
            return;
        std::lock_guard lock(mutex_);
        if (snap.index >= panels_.size() || panels_[snap.index].id != snap.id)
            return;
        if (!draw_succeeded) {
            panels_[snap.index].consecutive_errors++;
            if (panels_[snap.index].consecutive_errors >= PanelInfo::MAX_CONSECUTIVE_ERRORS) {
                panels_[snap.index].error_disabled = true;
                LOG_ERROR("Panel '{}' disabled after {} errors",
                          snap.label, panels_[snap.index].consecutive_errors);
            }
        } else {
            panels_[snap.index].consecutive_errors = 0;
        }
    }

    void PanelRegistry::invalidate_poll_cache(PollDependency changed) {
        std::lock_guard poll_lock(poll_mutex_);
        if (changed == PollDependency::ALL) {
            poll_cache_.clear();
            return;
        }
        std::erase_if(poll_cache_, [&](const auto& pair) {
            return (pair.second.poll_dependencies & changed) != PollDependency::NONE;
        });
    }

} // namespace lfs::vis::gui
