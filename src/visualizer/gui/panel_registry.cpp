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
#include "visualizer/app_store.hpp"

#include <SDL3/SDL_mouse.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>

namespace lfs::vis::gui {

    namespace {
        float floatingUiScale() {
            return std::max(1.0f, getThemeDpiScale());
        }

        float floatingResizeEdge() {
            return 6.0f * floatingUiScale();
        }

        bool shouldSuppressPanelForContext(const PanelInfo& panel, const PanelDrawContext& ctx) {
            return ctx.suppress_non_native_panels && !panel.is_native;
        }

        float scaledFloatingDimensionForScale(const float value, const float scale) {
            if (value <= 0.0f)
                return value;
            return std::round(value * std::max(1.0f, scale));
        }

        void resetFloatingPanelSize(PanelInfo& panel, FloatingPanelInteraction& interaction,
                                    const float scale) {
            panel.initial_width = scaledFloatingDimensionForScale(panel.original_width, scale);
            panel.initial_height = scaledFloatingDimensionForScale(panel.original_height, scale);
            interaction.user_height = 0.0f;
        }

        std::string panelDirectTimerName(const std::string& panel_id, const char* stage) {
            std::string name = "gui_render.panel_direct.";
            if (panel_id.empty()) {
                name += "unknown";
            } else {
                name.reserve(name.size() + panel_id.size() + 12);
                for (const unsigned char ch : panel_id) {
                    if (std::isalnum(ch) || ch == '.' || ch == '_' || ch == '-') {
                        name.push_back(static_cast<char>(ch));
                    } else {
                        name.push_back('_');
                    }
                }
            }
            name += '.';
            name += stage;
            return name;
        }

        const char* panelSpaceName(const PanelSpace space) {
            switch (space) {
            case PanelSpace::SidePanel: return "side_panel";
            case PanelSpace::Floating: return "floating";
            case PanelSpace::ViewportOverlay: return "viewport_overlay";
            case PanelSpace::MainPanelTab: return "main_panel_tab";
            case PanelSpace::SceneHeader: return "scene_header";
            case PanelSpace::BottomDock: return "bottom_dock";
            case PanelSpace::LeftDock: return "left_dock";
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

            if (panel.panel && panel.panel->renderCapabilities().direct)
                return true;

            LOG_ERROR("Panel '{}' ({}) cannot use '{}' space without direct rendering. "
                      "Window panels must be self-managed or support direct rendering.",
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

        FloatingPanelAnchor floatingAnchorRect(const PanelDrawContext& ctx) {
            // Floating panels may overlap docked panels. Constraining them to
            // the viewport makes their usable area shrink with the docks.
            if (ctx.screen_bounds && ctx.screen_bounds->valid()) {
                const auto& screen = *ctx.screen_bounds;
                const float screen_bottom = screen.y + screen.height;
                const float top = ctx.viewport && ctx.viewport->size.y > 0.0f
                                      ? std::max(screen.y, ctx.viewport->pos.y)
                                      : screen.y;
                return {
                    .x = screen.x,
                    .y = top,
                    .width = screen.width,
                    .height = std::max(0.0f, screen_bottom - top),
                };
            }

            if (ctx.viewport && ctx.viewport->size.x > 0.0f && ctx.viewport->size.y > 0.0f) {
                return {
                    .x = ctx.viewport->pos.x,
                    .y = ctx.viewport->pos.y,
                    .width = ctx.viewport->size.x,
                    .height = ctx.viewport->size.y,
                };
            }

            return {};
        }
    } // namespace

    FloatingPanelPlacement computeFloatingPanelPlacement(
        const FloatingPanelAnchor& anchor,
        const float panel_width,
        const float panel_height,
        const float stored_x,
        const float stored_y,
        const bool auto_center,
        const float title_height,
        const float visible_fraction) {
        if (anchor.width <= 0.0f || anchor.height <= 0.0f || panel_width <= 0.0f || panel_height <= 0.0f) {
            return {.x = stored_x, .y = stored_y};
        }

        float x = stored_x;
        float y = stored_y;
        if (auto_center || std::isnan(x) || std::isnan(y)) {
            x = anchor.x + (anchor.width - panel_width) * 0.5f;
            y = anchor.y + (anchor.height - panel_height) * 0.5f;
        }

        x = std::clamp(
            x,
            anchor.x - panel_width * (1.0f - visible_fraction),
            anchor.x + anchor.width - panel_width * visible_fraction);
        y = std::clamp(y, anchor.y, anchor.y + anchor.height - title_height);
        return {.x = x, .y = y};
    }

    PanelRegistry& PanelRegistry::instance() {
        static PanelRegistry registry;
        return registry;
    }

    uint64_t PanelRegistry::alloc_float_stack_order_locked() {
        return next_float_stack_order_++;
    }

    FloatingPanelInteraction& PanelRegistry::ensure_floating_interaction_locked(
        const PanelInfo& panel) {
        assert(panel.space == PanelSpace::Floating);
        auto& interaction = floating_interactions_[panel.id];
        if (interaction.stack_order == 0)
            interaction.stack_order = alloc_float_stack_order_locked();
        return interaction;
    }

    void PanelRegistry::bring_floating_panel_to_front_locked(const PanelInfo& panel) {
        if (panel.space != PanelSpace::Floating)
            return;
        auto& interaction = floating_interactions_[panel.id];
        interaction.stack_order = alloc_float_stack_order_locked();
    }

    bool PanelRegistry::register_panel(PanelInfo info) {
        std::shared_ptr<IPanel> instance;
        std::string pending_payload;
        {
            std::lock_guard lock(mutex_);
            assert(info.panel);
            assert(!info.id.empty());

            if (!validatePanelContract(info, info.space))
                return false;

            info.original_width = info.initial_width;
            info.original_height = info.initial_height;
            info.default_parent_id = info.parent_id;
            info.default_space = info.space;
            info.default_order = info.order;
            if (info.has_option(PanelOption::DEFAULT_CLOSED)) {
                info.enabled = false;
            }
            info.default_enabled = info.enabled;

            if (const auto requested =
                    requested_project_floating_state_.find(info.id);
                requested != requested_project_floating_state_.end()) {
                const auto& saved = requested->second;
                info.parent_id = saved.parent_id;
                info.space = saved.space;
                info.order = saved.order;
                info.enabled = saved.enabled;
            }

            if (disabled_overrides_.contains(info.id))
                info.enabled = false;

            if (!validatePanelContract(info, info.space)) {
                LOG_WARN(
                    "Ignoring invalid saved placement for panel '{}'",
                    info.id);
                info.parent_id = info.default_parent_id;
                info.space = info.default_space;
                info.order = info.default_order;
                info.enabled = info.default_enabled;
                if (disabled_overrides_.contains(info.id))
                    info.enabled = false;
                requested_project_floating_state_.erase(info.id);
            }

            const auto take_pending_payload = [&](const std::string& id) {
                if (const auto payload = requested_panel_payloads_.find(id);
                    payload != requested_panel_payloads_.end()) {
                    pending_payload = payload->second;
                }
            };

            for (auto& p : panels_) {
                if (p.id != info.id)
                    continue;
                uint64_t previous_stack_order = 0;
                if (const auto it = floating_interactions_.find(info.id);
                    it != floating_interactions_.end()) {
                    previous_stack_order = it->second.stack_order;
                }
                floating_interactions_.erase(info.id);
                if (info.space == PanelSpace::Floating) {
                    auto& interaction = ensure_floating_interaction_locked(info);
                    interaction.stack_order = previous_stack_order != 0
                                                  ? previous_stack_order
                                                  : interaction.stack_order;
                    resetFloatingPanelSize(info, interaction, floatingUiScale());
                    if (const auto requested =
                            requested_project_floating_state_.find(info.id);
                        requested != requested_project_floating_state_.end()) {
                        const auto& saved = requested->second;
                        interaction.x = saved.float_x;
                        interaction.y = saved.float_y;
                        interaction.user_height =
                            std::max(saved.float_user_height, 0.0f);
                        interaction.last_bounds_valid =
                            saved.float_last_bounds_valid;
                        interaction.last_x = saved.float_last_x;
                        interaction.last_y = saved.float_last_y;
                        interaction.last_width =
                            std::max(saved.float_last_w, 0.0f);
                        interaction.last_height =
                            std::max(saved.float_last_h, 0.0f);
                        interaction.auto_center = saved.float_auto_center;
                        if (saved.float_stack_order != 0)
                            interaction.stack_order = saved.float_stack_order;
                    }
                }
                take_pending_payload(info.id);
                p = std::move(info);
                instance = p.panel;
                ++registration_revision_;
                goto apply_registered_chrome;
            }

            if (info.space == PanelSpace::Floating) {
                auto& interaction = ensure_floating_interaction_locked(info);
                resetFloatingPanelSize(info, interaction, floatingUiScale());
                if (const auto requested =
                        requested_project_floating_state_.find(info.id);
                    requested != requested_project_floating_state_.end()) {
                    const auto& saved = requested->second;
                    interaction.x = saved.float_x;
                    interaction.y = saved.float_y;
                    interaction.user_height =
                        std::max(saved.float_user_height, 0.0f);
                    interaction.last_bounds_valid =
                        saved.float_last_bounds_valid;
                    interaction.last_x = saved.float_last_x;
                    interaction.last_y = saved.float_last_y;
                    interaction.last_width =
                        std::max(saved.float_last_w, 0.0f);
                    interaction.last_height =
                        std::max(saved.float_last_h, 0.0f);
                    interaction.auto_center = saved.float_auto_center;
                    if (saved.float_stack_order != 0)
                        interaction.stack_order = saved.float_stack_order;
                }
            }
            take_pending_payload(info.id);
            instance = info.panel;
            panels_.push_back(std::move(info));
            std::stable_sort(panels_.begin(), panels_.end(), [](const PanelInfo& a, const PanelInfo& b) {
                if (a.order != b.order)
                    return a.order < b.order;
                return a.label < b.label;
            });
            ++registration_revision_;
        }
apply_registered_chrome:
        if (instance && !pending_payload.empty())
            instance->applyChromeJson(pending_payload);
        return true;
    }

    void PanelRegistry::unregister_panel(const std::string& id) {
        {
            std::lock_guard lock(mutex_);
            if (std::erase_if(
                    panels_, [&id](const PanelInfo& p) {
                        return p.id == id;
                    }) != 0) {
                ++registration_revision_;
            }
            floating_interactions_.erase(id);
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
            std::vector<std::string> removed;
            for (const auto& panel : panels_) {
                if (!panel.is_native)
                    removed.push_back(panel.id);
            }
            std::erase_if(panels_, [](const PanelInfo& p) { return !p.is_native; });
            for (const auto& id : removed)
                floating_interactions_.erase(id);
            if (!removed.empty())
                ++registration_revision_;
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

    void PanelRegistry::reload_rml_resources() {
        std::vector<std::shared_ptr<IPanel>> panels;
        {
            std::lock_guard lock(mutex_);
            panels.reserve(panels_.size());
            for (const auto& panel : panels_) {
                if (panel.panel)
                    panels.push_back(panel.panel);
            }
        }

        for (const auto& panel : panels)
            panel->reloadRmlResources();

        invalidate_poll_cache();
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

    PanelSnapshot PanelRegistry::make_snapshot_locked(const size_t index,
                                                      const PanelInfo& panel) const {
        PanelSnapshot snapshot{
            index,
            panel.panel.get(),
            panel.label,
            panel.id,
            panel.space,
            panel.options,
            panel.is_native,
            panel.poll_dependencies,
            panel.initial_width,
            panel.initial_height,
            NAN,
            NAN,
            panel.order,
            0,
            panel.panel,
        };
        if (const auto it = floating_interactions_.find(panel.id);
            it != floating_interactions_.end()) {
            snapshot.float_x = it->second.x;
            snapshot.float_y = it->second.y;
            snapshot.float_stack_order = it->second.stack_order;
        }
        return snapshot;
    }

    std::vector<PanelSnapshot> PanelRegistry::collect_snapshots_locked(
        const PanelRenderTarget& target,
        const PanelDrawContext& ctx) const {
        std::vector<PanelSnapshot> snapshots;
        snapshots.reserve(panels_.size());
        for (size_t i = 0; i < panels_.size(); ++i) {
            const auto& panel = panels_[i];
            bool matches = false;
            switch (target.kind) {
            case PanelRenderTargetKind::Space:
                matches = panel.space == target.space && panel.parent_id.empty();
                break;
            case PanelRenderTargetKind::Panel:
                matches = panel.id == target.id;
                break;
            case PanelRenderTargetKind::Children:
                matches = panel.parent_id == target.id;
                break;
            }
            if (matches && panel.enabled && !panel.error_disabled &&
                !shouldSuppressPanelForContext(panel, ctx)) {
                snapshots.push_back(make_snapshot_locked(i, panel));
            }
        }
        return snapshots;
    }

    void PanelRegistry::render_space_panels(PanelSpace space, const PanelDrawContext& ctx,
                                            const PanelInputState* input) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            if (space == PanelSpace::Floating) {
                for (auto& [id, interaction] : floating_interactions_) {
                    (void)id;
                    interaction.last_bounds_valid = false;
                }
            }
            snapshots = collect_snapshots_locked(PanelRenderTarget::for_space(space), ctx);
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

        auto prepare_floating_direct_layout = [&](const PanelSnapshot& snap,
                                                  FloatingDirectLayout& layout) {
            layout = {};

            if (space != PanelSpace::Floating || !snap.panel->renderCapabilities().direct ||
                snap.has_option(PanelOption::SELF_MANAGED))
                return;

            const auto anchor = floatingAnchorRect(ctx);
            if (anchor.width <= 0.0f || anchor.height <= 0.0f)
                return;

            const float min_panel_width = 320.0f * dpi;
            const float max_panel_width = std::max(min_panel_width, anchor.width);
            float w = snap.initial_width > 0 ? snap.initial_width : 560.0f * dpi;
            w = std::clamp(w, min_panel_width, max_panel_width);
            const float max_h = snap.initial_height > 0
                                    ? std::min(snap.initial_height, anchor.height)
                                    : anchor.height;
            float drawn_h = snap.panel->renderDirect({
                                                         .mode = PanelDirectRenderMode::Measure,
                                                         .space = snap.space,
                                                     },
                                                     ctx)
                                .height;
            if (drawn_h <= 0.0f) {
                float prev_h = -1.0f;
                for (int pass = 0; pass < 3; ++pass) {
                    drawn_h = snap.panel->renderDirect({
                                                           .mode = PanelDirectRenderMode::Preload,
                                                           .space = snap.space,
                                                           .width = w,
                                                           .height = max_h,
                                                           .input = input,
                                                       },
                                                       ctx)
                                  .height;

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
                    panels_[snap.index].space == PanelSpace::Floating) {
                    const auto interaction = floating_interactions_.find(snap.id);
                    if (interaction != floating_interactions_.end() &&
                        interaction->second.user_height > 0) {
                        h = interaction->second.user_height;
                        has_user_height = true;
                    } else if (drawn_h > 0) {
                        h = std::min(drawn_h, max_h);
                    } else if (snap.initial_height > 0) {
                        h = snap.initial_height;
                    } else {
                        h = 400.0f * dpi;
                    }
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
            bool auto_center = true;
            {
                std::lock_guard lock(mutex_);
                if (snap.index < panels_.size() && panels_[snap.index].id == snap.id &&
                    panels_[snap.index].space == PanelSpace::Floating) {
                    if (const auto interaction = floating_interactions_.find(snap.id);
                        interaction != floating_interactions_.end()) {
                        auto_center = interaction->second.auto_center;
                    }
                }
            }
            const auto placement =
                computeFloatingPanelPlacement(anchor, w, h, px, py, auto_center, kTitleH, kVisibleFrac);
            px = placement.x;
            py = placement.y;

            layout.valid = true;
            layout.width = w;
            layout.height = h;
            layout.pos_x = px;
            layout.pos_y = py;
            layout.drawn_height = drawn_h;
            layout.has_user_height = has_user_height;
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
                if (should_draw[i]) {
                    prepare_floating_direct_layout(snapshots[i], floating_direct_layouts[i]);
                }
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

        int8_t floating_cursor_dir_x = 0;
        int8_t floating_cursor_dir_y = 0;
        if (hovered_floating_direct >= 0) {
            const auto& hovered =
                floating_direct_layouts[static_cast<std::size_t>(
                    hovered_floating_direct)];
            if (hovered.mouse_in_resize_grip) {
                floating_cursor_dir_x = hovered.hover_dir_x;
                floating_cursor_dir_y = hovered.hover_dir_y;
            }
        }

        for (size_t snap_idx = 0; snap_idx < snapshots.size(); ++snap_idx) {
            auto& snap = snapshots[snap_idx];
            bool draw_succeeded = false;
            if (!should_draw[snap_idx])
                continue;

            constexpr double kViewportOverlayPanelPerfThresholdMs = 0.05;
            const bool time_viewport_panel = space == PanelSpace::ViewportOverlay;
            const auto panel_start = time_viewport_panel
                                         ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};

            try {
                switch (space) {
                case PanelSpace::Floating: {
                    if (snap.has_option(PanelOption::SELF_MANAGED)) {
                        snap.panel->renderDirect({
                                                     .mode = PanelDirectRenderMode::Draw,
                                                     .space = snap.space,
                                                     .input = input,
                                                 },
                                                 ctx);
                    } else if (snap.panel->renderCapabilities().direct) {
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
                            if (snap.index < panels_.size() && panels_[snap.index].id == snap.id &&
                                panels_[snap.index].space == PanelSpace::Floating) {
                                auto& pi = panels_[snap.index];
                                auto& interaction = ensure_floating_interaction_locked(pi);
                                const bool active_this_panel = interaction.dragging || interaction.resizing;
                                const bool hovered_this_panel =
                                    static_cast<int>(snap_idx) == hovered_floating_direct;
                                const bool interactive = active_this_panel || hovered_this_panel;
                                const bool mouse_clicked_left = input && input->mouse_clicked[0];
                                const bool mouse_down_left = input && input->mouse_down[0];
                                const float mouse_x = input ? input->mouse_x : px;
                                const float mouse_y = input ? input->mouse_y : py;

                                const bool any_active = std::any_of(
                                    floating_interactions_.begin(), floating_interactions_.end(),
                                    [](const auto& entry) {
                                        return entry.second.dragging || entry.second.resizing;
                                    });

                                if (interactive && (layout.mouse_in_panel || layout.mouse_in_resize_grip) &&
                                    mouse_clicked_left) {
                                    bring_floating_panel_to_front_locked(pi);
                                }

                                if (interactive && layout.mouse_in_resize_grip && !any_active &&
                                    mouse_clicked_left) {
                                    requested_project_floating_state_.erase(pi.id);
                                    interaction.auto_center = false;
                                    interaction.resizing = true;
                                    interaction.resize_start_width = w;
                                    interaction.resize_start_height = h;
                                    interaction.resize_start_mouse_x = mouse_x;
                                    interaction.resize_start_mouse_y = mouse_y;
                                    interaction.resize_start_panel_x = px;
                                    interaction.resize_start_panel_y = py;
                                    interaction.resize_direction_x = layout.hover_dir_x;
                                    interaction.resize_direction_y = layout.hover_dir_y;
                                } else if (interactive && layout.mouse_in_titlebar &&
                                           !layout.mouse_in_resize_grip && !any_active &&
                                           mouse_clicked_left) {
                                    requested_project_floating_state_.erase(pi.id);
                                    interaction.auto_center = false;
                                    interaction.dragging = true;
                                    interaction.drag_offset_x = mouse_x - px;
                                    interaction.drag_offset_y = mouse_y - py;
                                }

                                if (interaction.dragging) {
                                    if (mouse_down_left) {
                                        px = mouse_x - interaction.drag_offset_x;
                                        py = mouse_y - interaction.drag_offset_y;
                                    } else {
                                        interaction.dragging = false;
                                    }
                                }
                                if (interaction.resizing) {
                                    floating_cursor_dir_x = interaction.resize_direction_x;
                                    floating_cursor_dir_y = interaction.resize_direction_y;
                                    if (mouse_down_left) {
                                        const float dx = mouse_x - interaction.resize_start_mouse_x;
                                        const float dy = mouse_y - interaction.resize_start_mouse_y;
                                        if (interaction.resize_direction_x == 1) {
                                            w = std::max(kMinPanelWidth, interaction.resize_start_width + dx);
                                            pi.initial_width = w;
                                        } else if (interaction.resize_direction_x == -1) {
                                            w = std::max(kMinPanelWidth, interaction.resize_start_width - dx);
                                            px = interaction.resize_start_panel_x + interaction.resize_start_width - w;
                                            pi.initial_width = w;
                                        }
                                        if (interaction.resize_direction_y == 1) {
                                            h = std::max(kMinPanelHeight, interaction.resize_start_height + dy);
                                            interaction.user_height = h;
                                        } else if (interaction.resize_direction_y == -1) {
                                            h = std::max(kMinPanelHeight, interaction.resize_start_height - dy);
                                            py = interaction.resize_start_panel_y + interaction.resize_start_height - h;
                                            interaction.user_height = h;
                                        }
                                    } else {
                                        interaction.resizing = false;
                                        interaction.resize_direction_x = 0;
                                        interaction.resize_direction_y = 0;
                                    }
                                }

                                if (!interaction.resizing && interaction.user_height > 0 &&
                                    drawn_h <= 0) {
                                    interaction.user_height = 0;
                                }
                                has_user_height = interaction.user_height > 0.0f;

                                const auto anchor = floatingAnchorRect(ctx);
                                const auto placement = computeFloatingPanelPlacement(
                                    anchor, w, h, px, py, false, kTitleH, kVisibleFrac);
                                px = placement.x;
                                py = placement.y;

                                interaction.x = px;
                                interaction.y = py;
                            }
                        }

                        {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].id == snap.id &&
                                panels_[snap.index].space == PanelSpace::Floating) {
                                const auto& pi = panels_[snap.index];
                                auto& interaction = ensure_floating_interaction_locked(pi);
                                const bool interactive =
                                    interaction.dragging || interaction.resizing ||
                                    static_cast<int>(snap_idx) == hovered_floating_direct;

                                interaction.last_bounds_valid = true;
                                interaction.last_x = px;
                                interaction.last_y = py;
                                interaction.last_width = w;
                                interaction.last_height = h;

                                if (interaction.dragging || interaction.resizing ||
                                    (interactive &&
                                     (layout.mouse_in_panel || layout.mouse_in_resize_grip))) {
                                    guiFocusState().want_capture_mouse = true;
                                }

                                (void)interactive;
                            }
                        }

                        const float forced = (has_user_height && drawn_h > 0 && h > drawn_h) ? h : 0.0f;
                        const auto result = snap.panel->renderDirect({
                                                                         .mode = PanelDirectRenderMode::Draw,
                                                                         .space = snap.space,
                                                                         .x = px,
                                                                         .y = py,
                                                                         .width = w,
                                                                         .height = h,
                                                                         .forced_height = forced,
                                                                         .input = input,
                                                                     },
                                                                     ctx);
                        drawn_h = result.height;
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
                    snap.panel->renderDirect({
                                                 .mode = PanelDirectRenderMode::Draw,
                                                 .space = snap.space,
                                             },
                                             ctx);
                    break;
                }
                case PanelSpace::ViewportOverlay:
                case PanelSpace::SceneHeader:
                    snap.panel->renderDirect({
                                                 .mode = PanelDirectRenderMode::Draw,
                                                 .space = snap.space,
                                             },
                                             ctx);
                    break;
                case PanelSpace::BottomDock:
                case PanelSpace::LeftDock:
                    break;
                case PanelSpace::StatusBar:
                    snap.panel->renderDirect({
                                                 .mode = PanelDirectRenderMode::Draw,
                                                 .space = snap.space,
                                                 .input = input,
                                             },
                                             ctx);
                    break;
                case PanelSpace::MainPanelTab:
                    break;
                }

                draw_succeeded = true;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' draw error: {}", snap.label, e.what());
            }

            track_draw_result(snap, draw_succeeded);
            if (time_viewport_panel) {
                const auto elapsed =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - panel_start)
                        .count();
                if (elapsed >= kViewportOverlayPanelPerfThresholdMs) {
                    LOG_PERF("gui_render.viewport_overlay.panel.{} took {:.2f}ms",
                             snap.id, elapsed);
                }
            }
        }

        if (space == PanelSpace::Floating) {
            std::lock_guard lock(mutex_);
            floating_cursor_dir_x_ = floating_cursor_dir_x;
            floating_cursor_dir_y_ = floating_cursor_dir_y;
        }
    }

    float PanelRegistry::render_panels(const PanelRenderOptions& options,
                                       const PanelDrawContext& ctx) {
        switch (options.mode) {
        case PanelRenderMode::Standard:
            if (options.target.kind == PanelRenderTargetKind::Space) {
                render_space_panels(options.target.space, ctx, options.input);
                return 0.0f;
            }
            return render_standard_panels(options, ctx);
        case PanelRenderMode::StandardPreload:
            return render_standard_panels(options, ctx);
        case PanelRenderMode::Direct:
        case PanelRenderMode::DirectCached:
        case PanelRenderMode::DirectPreload:
            return render_direct_panels(options, ctx);
        }
        return 0.0f;
    }

    float PanelRegistry::render_standard_panels(const PanelRenderOptions& options,
                                                const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots = collect_snapshots_locked(options.target, ctx);
        }

        for (auto& snap : snapshots) {
            try {
                if (!check_poll(snap, ctx))
                    continue;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' {} poll error: {}", snap.label,
                          options.mode == PanelRenderMode::StandardPreload ? "preload" : "draw", e.what());
                continue;
            }

            bool succeeded = false;
            try {
                const auto result = snap.panel->renderDirect({
                                                                 .mode = options.mode == PanelRenderMode::StandardPreload
                                                                             ? PanelDirectRenderMode::Preload
                                                                             : PanelDirectRenderMode::Draw,
                                                                 .space = snap.space,
                                                                 .input = options.input,
                                                             },
                                                             ctx);
                succeeded = result.handled;
            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' {} error: {}", snap.label,
                          options.mode == PanelRenderMode::StandardPreload ? "preload" : "draw", e.what());
            }

            if (options.mode != PanelRenderMode::StandardPreload)
                track_draw_result(snap, succeeded);
        }
        return 0.0f;
    }

    float PanelRegistry::render_direct_panels(const PanelRenderOptions& options,
                                              const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots = collect_snapshots_locked(options.target, ctx);
        }

        float y_offset = 0.0f;
        const bool single_panel = options.target.kind == PanelRenderTargetKind::Panel;
        for (auto& snap : snapshots) {
            const float remaining = options.height - y_offset;
            if (remaining <= 0.0f)
                break;

            bool succeeded = false;
            float used_height = 0.0f;
            try {
                const PanelSpace panel_space = snap.space;

                bool needs_live_draw = options.mode != PanelRenderMode::DirectCached;
                if (options.mode == PanelRenderMode::DirectCached) {
                    LOG_TIMER_THRESHOLD(panelDirectTimerName(snap.id, "draw_cached"), 0.25);
                    const auto result = snap.panel->renderDirect({
                                                                     .mode = PanelDirectRenderMode::Cached,
                                                                     .space = panel_space,
                                                                     .x = options.x,
                                                                     .y = options.y + y_offset,
                                                                     .width = options.width,
                                                                     .height = remaining,
                                                                     .clip_y_min = options.clip_y_min,
                                                                     .clip_y_max = options.clip_y_max,
                                                                     .input = options.input,
                                                                 },
                                                                 ctx);
                    needs_live_draw = !result.handled;
                    used_height = result.height;
                    if (!needs_live_draw)
                        succeeded = true;
                }

                if (needs_live_draw) {
                    {
                        LOG_TIMER_THRESHOLD(panelDirectTimerName(snap.id, "poll"), 0.25);
                        if (!check_poll(snap, ctx))
                            continue;
                    }

                    PanelDirectRenderResult result;
                    if (options.mode == PanelRenderMode::DirectPreload) {
                        LOG_TIMER_THRESHOLD(panelDirectTimerName(snap.id, "preload"), 0.25);
                        result = snap.panel->renderDirect({
                                                              .mode = PanelDirectRenderMode::Preload,
                                                              .space = panel_space,
                                                              .width = options.width,
                                                              .height = remaining,
                                                              .clip_y_min = options.clip_y_min,
                                                              .clip_y_max = options.clip_y_max,
                                                              .input = options.input,
                                                          },
                                                          ctx);
                    } else {
                        LOG_TIMER_THRESHOLD(panelDirectTimerName(snap.id, "draw"), 0.25);
                        result = snap.panel->renderDirect({
                                                              .mode = PanelDirectRenderMode::Draw,
                                                              .space = panel_space,
                                                              .x = options.x,
                                                              .y = options.y + y_offset,
                                                              .width = options.width,
                                                              .height = remaining,
                                                              .clip_y_min = options.clip_y_min,
                                                              .clip_y_max = options.clip_y_max,
                                                              .input = options.input,
                                                          },
                                                          ctx);
                    }
                    succeeded = result.handled;
                    used_height = result.height;
                }

            } catch (const std::exception& e) {
                LOG_ERROR("Panel '{}' {} error: {}", snap.label,
                          options.mode == PanelRenderMode::DirectPreload ? "DirectPreload"
                                                                         : (options.mode == PanelRenderMode::DirectCached
                                                                                ? "DirectCached"
                                                                                : "Direct"),
                          e.what());
            }

            track_draw_result(snap, succeeded);
            if (succeeded) {
                if (single_panel)
                    return used_height > 0.0f ? used_height : 0.0f;
                y_offset += used_height > 0.0f ? used_height : remaining;
            }
        }
        return y_offset;
    }

    bool PanelRegistry::isPositionOverFloatingPanel(const double x, const double y) const {
        const double kResizeEdge = static_cast<double>(floatingResizeEdge());
        const double rounding = std::max(0.0f, theme().sizes.window_rounding);

        std::lock_guard lock(mutex_);
        for (const auto& panel : panels_) {
            const auto interaction_it = floating_interactions_.find(panel.id);
            if (panel.space != PanelSpace::Floating || !panel.enabled || panel.error_disabled ||
                !panel.parent_id.empty() || interaction_it == floating_interactions_.end() ||
                !interaction_it->second.last_bounds_valid) {
                continue;
            }

            const auto& interaction = interaction_it->second;

            const double panel_x = static_cast<double>(interaction.last_x);
            const double panel_y = static_cast<double>(interaction.last_y);
            const double panel_w = static_cast<double>(interaction.last_width);
            const double panel_h = static_cast<double>(interaction.last_height);

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
                result.push_back({p.label, p.id, p.space, p.order, p.enabled, p.tab_closeable});
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
            if (p.id == id) {
                uint64_t stack_order = 0;
                if (const auto interaction = floating_interactions_.find(id);
                    interaction != floating_interactions_.end()) {
                    stack_order = interaction->second.stack_order;
                }
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
                    stack_order,
                };
            }
        }
        return std::nullopt;
    }

    std::vector<PanelProjectState>
    PanelRegistry::capture_project_state() const {
        std::lock_guard lock(mutex_);
        std::vector<PanelProjectState> result;
        result.reserve(panels_.size());
        for (const auto& panel : panels_) {
            const auto interaction_it =
                floating_interactions_.find(panel.id);
            const auto* interaction =
                interaction_it != floating_interactions_.end()
                    ? &interaction_it->second
                    : nullptr;
            PanelProjectState state{
                .id = panel.id,
                .parent_id = panel.parent_id,
                .space = panel.space,
                .order = panel.order,
                .enabled = panel.enabled,
                .float_x = interaction ? interaction->x : NAN,
                .float_y = interaction ? interaction->y : NAN,
                .float_user_height = interaction ? interaction->user_height : 0.0f,
                .float_last_bounds_valid =
                    interaction && interaction->last_bounds_valid,
                .float_last_x = interaction ? interaction->last_x : 0.0f,
                .float_last_y = interaction ? interaction->last_y : 0.0f,
                .float_last_w = interaction ? interaction->last_width : 0.0f,
                .float_last_h = interaction ? interaction->last_height : 0.0f,
                .float_auto_center =
                    !interaction || interaction->auto_center,
                .float_stack_order =
                    interaction ? interaction->stack_order : 0,
            };
            if (const auto requested =
                    requested_project_floating_state_.find(panel.id);
                panel.space == PanelSpace::Floating &&
                requested != requested_project_floating_state_.end()) {
                state.float_x = requested->second.float_x;
                state.float_y = requested->second.float_y;
                state.float_user_height =
                    requested->second.float_user_height;
                state.float_last_bounds_valid =
                    requested->second.float_last_bounds_valid;
                state.float_last_x =
                    requested->second.float_last_x;
                state.float_last_y =
                    requested->second.float_last_y;
                state.float_last_w =
                    requested->second.float_last_w;
                state.float_last_h =
                    requested->second.float_last_h;
                state.float_auto_center =
                    requested->second.float_auto_center;
            }
            result.push_back(std::move(state));
        }
        return result;
    }

    void PanelRegistry::apply_project_state(
        const std::vector<PanelProjectState>& state) {
        std::lock_guard lock(mutex_);
        reset_project_state_locked();
        for (const auto& saved : state) {
            const auto found = std::find_if(
                panels_.begin(), panels_.end(),
                [&](const PanelInfo& panel) {
                    return panel.id == saved.id;
                });
            if (found == panels_.end()) {
                requested_project_floating_state_.insert_or_assign(
                    saved.id, saved);
                continue;
            }

            PanelInfo candidate = *found;
            candidate.parent_id = saved.parent_id;
            candidate.space = saved.space;
            candidate.order = saved.order;
            if (!validatePanelContract(candidate, candidate.space)) {
                LOG_WARN(
                    "Ignoring invalid saved placement for panel '{}'",
                    saved.id);
                continue;
            }
            found->parent_id = saved.parent_id;
            found->space = saved.space;
            found->order = saved.order;
            found->enabled = saved.enabled;
            if (saved.space == PanelSpace::Floating) {
                auto& interaction =
                    ensure_floating_interaction_locked(*found);
                interaction.x = saved.float_x;
                interaction.y = saved.float_y;
                interaction.user_height =
                    std::max(saved.float_user_height, 0.0f);
                interaction.last_bounds_valid =
                    saved.float_last_bounds_valid;
                interaction.last_x = saved.float_last_x;
                interaction.last_y = saved.float_last_y;
                interaction.last_width =
                    std::max(saved.float_last_w, 0.0f);
                interaction.last_height =
                    std::max(saved.float_last_h, 0.0f);
                interaction.auto_center =
                    saved.float_auto_center;
                if (saved.float_stack_order != 0)
                    interaction.stack_order =
                        saved.float_stack_order;
                requested_project_floating_state_.insert_or_assign(
                    saved.id, saved);
                next_float_stack_order_ = std::max(
                    next_float_stack_order_,
                    interaction.stack_order + 1);
            } else {
                floating_interactions_.erase(saved.id);
            }
        }
        std::stable_sort(
            panels_.begin(), panels_.end(),
            [](const PanelInfo& lhs,
               const PanelInfo& rhs) {
                if (lhs.order != rhs.order)
                    return lhs.order < rhs.order;
                return lhs.label < rhs.label;
            });
    }

    void PanelRegistry::clear_project_state_retention() {
        std::lock_guard lock(mutex_);
        requested_project_floating_state_.clear();
    }

    std::unordered_map<std::string, std::string>
    PanelRegistry::capture_panel_payloads() const {
        std::vector<std::pair<std::string, std::shared_ptr<IPanel>>> owners;
        {
            std::lock_guard lock(mutex_);
            owners.reserve(panels_.size());
            for (const auto& panel : panels_) {
                if (panel.panel)
                    owners.emplace_back(panel.id, panel.panel);
            }
        }
        std::unordered_map<std::string, std::string> payloads;
        for (const auto& [id, panel] : owners) {
            auto json = panel->captureChromeJson();
            if (!json.empty())
                payloads.insert_or_assign(id, std::move(json));
        }
        return payloads;
    }

    void PanelRegistry::apply_panel_payloads(
        const std::unordered_map<std::string, std::string>& payloads) {
        std::vector<std::pair<std::shared_ptr<IPanel>, std::string>> apply_list;
        {
            std::lock_guard lock(mutex_);
            requested_panel_payloads_ = payloads;
            apply_list.reserve(panels_.size());
            for (const auto& panel : panels_) {
                if (!panel.panel)
                    continue;
                const auto found = payloads.find(panel.id);
                apply_list.emplace_back(
                    panel.panel,
                    found != payloads.end() ? found->second : std::string("{}"));
            }
        }
        for (const auto& [panel, json] : apply_list)
            panel->applyChromeJson(json);
    }

    void PanelRegistry::reset_project_state_locked() {
        requested_project_floating_state_.clear();
        requested_panel_payloads_.clear();
        floating_interactions_.clear();
        next_float_stack_order_ = 1;
        for (auto& panel : panels_) {
            panel.parent_id = panel.default_parent_id;
            panel.space = panel.default_space;
            panel.order = panel.default_order;
            panel.enabled = panel.default_enabled;
            if (disabled_overrides_.contains(panel.id))
                panel.enabled = false;
            if (panel.space == PanelSpace::Floating) {
                auto& interaction =
                    ensure_floating_interaction_locked(panel);
                resetFloatingPanelSize(
                    panel, interaction, floatingUiScale());
            }
        }
        std::stable_sort(
            panels_.begin(), panels_.end(),
            [](const PanelInfo& lhs, const PanelInfo& rhs) {
                if (lhs.order != rhs.order)
                    return lhs.order < rhs.order;
                return lhs.label < rhs.label;
            });
    }

    void PanelRegistry::reset_project_state() {
        std::lock_guard lock(mutex_);
        reset_project_state_locked();
    }

    uint64_t PanelRegistry::registration_revision() const {
        std::lock_guard lock(mutex_);
        return registration_revision_;
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
        bool changed = false;
        {
            std::lock_guard lock(mutex_);
            for (auto& p : panels_) {
                if (p.id == id) {
                    if (enabled)
                        disabled_overrides_.erase(id);
                    else
                        disabled_overrides_.insert(id);
                    changed = p.enabled != enabled;
                    if (!changed)
                        break;

                    p.enabled = enabled;
                    if (enabled && p.space == PanelSpace::Floating) {
                        auto& interaction = ensure_floating_interaction_locked(p);
                        if (const auto requested =
                                requested_project_floating_state_.find(p.id);
                            requested != requested_project_floating_state_.end()) {
                            const auto& saved = requested->second;
                            interaction.x = saved.float_x;
                            interaction.y = saved.float_y;
                            interaction.user_height =
                                std::max(saved.float_user_height, 0.0f);
                            interaction.last_bounds_valid =
                                saved.float_last_bounds_valid;
                            interaction.last_x = saved.float_last_x;
                            interaction.last_y = saved.float_last_y;
                            interaction.last_width =
                                std::max(saved.float_last_w, 0.0f);
                            interaction.last_height =
                                std::max(saved.float_last_h, 0.0f);
                            interaction.auto_center = saved.float_auto_center;
                        } else {
                            interaction.x = NAN;
                            interaction.y = NAN;
                            interaction.auto_center = true;
                            resetFloatingPanelSize(
                                p, interaction, floatingUiScale());
                            bring_floating_panel_to_front_locked(p);
                        }
                    } else if (!enabled) {
                        if (auto interaction = floating_interactions_.find(id);
                            interaction != floating_interactions_.end()) {
                            interaction->second.last_bounds_valid = false;
                        }
                        guiFocusState().want_capture_mouse = false;
                        guiFocusState().want_capture_keyboard = false;
                        guiFocusState().want_text_input = false;
                    }
                    break;
                }
            }
        }

        if (changed)
            lfs::vis::publish_viewport_toolbar_generation();
    }

    bool PanelRegistry::bring_panel_to_front(const std::string& id) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.id == id && p.enabled && !p.error_disabled && p.space == PanelSpace::Floating) {
                bring_floating_panel_to_front_locked(p);
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::is_panel_enabled(const std::string& id) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.id == id)
                return p.enabled;
        }
        return false;
    }

    bool PanelRegistry::apply_floating_resize_cursor() const {
        int8_t dir_x = 0;
        int8_t dir_y = 0;
        {
            std::lock_guard lock(mutex_);
            dir_x = floating_cursor_dir_x_;
            dir_y = floating_cursor_dir_y_;
        }

        SDL_Cursor* cursor = nullptr;
        if (dir_x != 0 && dir_y != 0) {
            static SDL_Cursor* const nwse =
                SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NWSE_RESIZE);
            static SDL_Cursor* const nesw =
                SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NESW_RESIZE);
            cursor = dir_x == dir_y ? nwse : nesw;
        } else if (dir_x != 0) {
            static SDL_Cursor* const ew =
                SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_EW_RESIZE);
            cursor = ew;
        } else if (dir_y != 0) {
            static SDL_Cursor* const ns =
                SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_NS_RESIZE);
            cursor = ns;
        }

        if (!cursor)
            return false;
        SDL_SetCursor(cursor);
        return true;
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

            auto& interaction = ensure_floating_interaction_locked(p);

            if (p.initial_width > 0.0f)
                p.initial_width = std::round(p.initial_width * ratio);
            else if (p.original_width > 0.0f)
                p.initial_width = scaledFloatingDimensionForScale(p.original_width, new_scale);

            if (p.initial_height > 0.0f)
                p.initial_height = std::round(p.initial_height * ratio);
            else if (p.original_height > 0.0f)
                p.initial_height = scaledFloatingDimensionForScale(p.original_height, new_scale);

            if (interaction.user_height > 0.0f)
                interaction.user_height = std::max(1.0f, std::round(interaction.user_height * ratio));

            interaction.dragging = false;
            interaction.resizing = false;
            interaction.resize_direction_x = 0;
            interaction.resize_direction_y = 0;
            interaction.last_bounds_valid = false;
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

    namespace {
        // Shared visibility predicate for animation demand and scheduled delays.
        // Must stay in lockstep: a mismatch causes either pinned frames or stalled wakes.
        [[nodiscard]] bool isPanelVisibleForAnimation(
            const PanelInfo& p, const PanelAnimationVisibility& visibility) {
            if (!p.parent_id.empty()) {
                return visibility.right_panel_visible &&
                       std::string_view(p.parent_id) == visibility.active_main_tab;
            }

            switch (p.space) {
            case PanelSpace::Floating:
                return visibility.ui_visible;
            case PanelSpace::SidePanel:
                return visibility.ui_visible;
            case PanelSpace::StatusBar:
                return visibility.ui_visible;
            case PanelSpace::ViewportOverlay:
                return true;
            case PanelSpace::SceneHeader:
                return visibility.right_panel_visible;
            case PanelSpace::MainPanelTab:
                return visibility.right_panel_visible &&
                       std::string_view(p.id) == visibility.active_main_tab;
            case PanelSpace::BottomDock:
                return visibility.ui_visible && visibility.bottom_dock_visible;
            case PanelSpace::LeftDock:
                return visibility.ui_visible && visibility.left_dock_visible;
            }
            return false;
        }

        void markVisibleAnimationDemand(PanelAnimationDemand& demand, const PanelInfo& p,
                                        const PanelAnimationVisibility& visibility) {
            if (!isPanelVisibleForAnimation(p, visibility))
                return;

            if (!p.parent_id.empty()) {
                demand.main_panel_tab = true;
                return;
            }

            switch (p.space) {
            case PanelSpace::Floating:
                demand.floating = true;
                return;
            case PanelSpace::SidePanel:
                demand.side_panel = true;
                return;
            case PanelSpace::StatusBar:
                demand.status_bar = true;
                return;
            case PanelSpace::ViewportOverlay:
                demand.viewport_overlay = true;
                return;
            case PanelSpace::SceneHeader:
                demand.scene_header = true;
                return;
            case PanelSpace::MainPanelTab:
                demand.main_panel_tab = true;
                return;
            case PanelSpace::BottomDock:
                demand.bottom_dock = true;
                return;
            case PanelSpace::LeftDock:
                demand.left_dock = true;
                return;
            }
        }

        [[nodiscard]] std::optional<double> minOptionalDelay(std::optional<double> a,
                                                             std::optional<double> b) {
            if (!a)
                return b;
            if (!b)
                return a;
            return std::min(*a, *b);
        }
    } // namespace

    PanelAnimationDemand PanelRegistry::animationDemandForVisiblePanels(
        const PanelAnimationVisibility visibility) const {
        PanelAnimationDemand demand;
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (!p.enabled || p.error_disabled || !p.panel)
                continue;
            if (p.panel->needsAnimationFrame())
                markVisibleAnimationDemand(demand, p, visibility);
        }
        return demand;
    }

    bool PanelRegistry::needsAnimationFrameForVisiblePanels(
        const PanelAnimationVisibility visibility) const {
        return animationDemandForVisiblePanels(visibility).any();
    }

    std::optional<double> PanelRegistry::nextScheduledAnimationDelayForVisiblePanels(
        const PanelAnimationVisibility visibility) const {
        std::optional<double> min_delay;
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (!p.enabled || p.error_disabled || !p.panel)
                continue;
            if (!isPanelVisibleForAnimation(p, visibility))
                continue;
            min_delay = minOptionalDelay(min_delay, p.panel->nextScheduledAnimationDelay());
        }
        return min_delay;
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
                requested_project_floating_state_.erase(p.id);
                p.space = new_space;
                if (!was_floating && new_space == PanelSpace::Floating) {
                    auto& interaction = ensure_floating_interaction_locked(p);
                    interaction.x = NAN;
                    interaction.y = NAN;
                    interaction.auto_center = true;
                    resetFloatingPanelSize(p, interaction, floatingUiScale());
                    bring_floating_panel_to_front_locked(p);
                } else if (was_floating && new_space != PanelSpace::Floating) {
                    p.initial_width = p.original_width;
                    p.initial_height = p.original_height;
                    floating_interactions_.erase(p.id);
                }
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
