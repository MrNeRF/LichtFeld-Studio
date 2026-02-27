/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_registry.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/ui_context.hpp"
#include "theme/theme.hpp"

#include <algorithm>
#include <optional>
#include <imgui.h>

namespace lfs::vis::gui {

    PanelRegistry& PanelRegistry::instance() {
        static PanelRegistry registry;
        return registry;
    }

    void PanelRegistry::register_panel(PanelInfo info) {
        std::lock_guard lock(mutex_);
        assert(info.panel);
        assert(!info.idname.empty());

        if (disabled_overrides_.contains(info.idname))
            info.enabled = false;

        for (auto& p : panels_) {
            if (p.idname == info.idname) {
                p = std::move(info);
                return;
            }
        }

        panels_.push_back(std::move(info));
        std::stable_sort(panels_.begin(), panels_.end(), [](const PanelInfo& a, const PanelInfo& b) {
            if (a.order != b.order)
                return a.order < b.order;
            return a.label < b.label;
        });
    }

    void PanelRegistry::unregister_panel(const std::string& idname) {
        {
            std::lock_guard lock(mutex_);
            std::erase_if(panels_, [&idname](const PanelInfo& p) { return p.idname == idname; });
        }
        {
            std::lock_guard poll_lock(poll_mutex_);
            poll_cache_.erase(idname);
        }
    }

    void PanelRegistry::unregister_all_non_native() {
        std::vector<std::string> remaining;
        {
            std::lock_guard lock(mutex_);
            std::erase_if(panels_, [](const PanelInfo& p) { return !p.is_native; });
            remaining.reserve(panels_.size());
            for (const auto& p : panels_)
                remaining.push_back(p.idname);
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
            auto cache_it = poll_cache_.find(snap.idname);
            if (cache_it != poll_cache_.end()) {
                const auto& e = cache_it->second;
                bool valid = true;
                if ((snap.poll_deps & PollDependency::SCENE) != PollDependency::NONE)
                    valid &= (e.scene_generation == gen);
                if ((snap.poll_deps & PollDependency::SELECTION) != PollDependency::NONE)
                    valid &= (e.has_selection == has_sel);
                if ((snap.poll_deps & PollDependency::TRAINING) != PollDependency::NONE)
                    valid &= (e.is_training == training);
                if (valid)
                    return e.result;
            }
        }

        const bool result = snap.panel->poll(ctx);

        {
            std::lock_guard poll_lock(poll_mutex_);
            poll_cache_[snap.idname] = {result, gen, has_sel, training, snap.poll_deps};
        }
        return result;
    }

    void PanelRegistry::draw_panels(PanelSpace space, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_idname.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.idname,
                                         p.parent_idname, p.options, p.is_native,
                                         p.poll_deps, p.initial_width, p.initial_height,
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
                ImGui::PushID(snap.idname.c_str());

                switch (space) {
                case PanelSpace::Floating: {
                    if (snap.has_option(PanelOption::SELF_MANAGED)) {
                        snap.panel->draw(ctx);
                    } else if (snap.panel->supportsDirectDraw()) {
                        float w = snap.initial_width > 0 ? snap.initial_width : 560.0f;
                        const float h = snap.panel->getDirectDrawHeight() > 0
                                            ? snap.panel->getDirectDrawHeight()
                                            : (snap.initial_height > 0 ? snap.initial_height : 400.0f);
                        const auto* vp = ImGui::GetMainViewport();

                        float px = snap.float_x;
                        float py = snap.float_y;
                        if (px < 0) {
                            px = vp->WorkPos.x + (vp->WorkSize.x - w) * 0.5f;
                            py = vp->WorkPos.y + (vp->WorkSize.y - h) * 0.5f;
                        }

                        ImGuiIO& io = ImGui::GetIO();
                        const ImVec2 mouse = io.MousePos;
                        const bool mouse_in_panel = mouse.x >= px && mouse.x < px + w &&
                                                    mouse.y >= py && mouse.y < py + h;
                        const bool mouse_in_titlebar = mouse.x >= px && mouse.x < px + w &&
                                                       mouse.y >= py && mouse.y < py + 28.0f;
                        constexpr float kResizeEdge = 6.0f;
                        constexpr float kMinPanelWidth = 300.0f;
                        const bool mouse_in_resize_grip = mouse.x >= px + w - kResizeEdge &&
                                                          mouse.x < px + w + kResizeEdge &&
                                                          mouse.y >= py && mouse.y < py + h;

                        {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                                auto& pi = panels_[snap.index];

                                bool any_active = std::any_of(panels_.begin(), panels_.end(),
                                                              [](const PanelInfo& p) { return p.float_dragging || p.float_resizing; });

                                if (mouse_in_resize_grip && !any_active && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                    pi.float_resizing = true;
                                    pi.float_resize_start_w = w;
                                    pi.float_resize_start_mx = mouse.x;
                                } else if (mouse_in_titlebar && !mouse_in_resize_grip && !any_active &&
                                           ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                                    pi.float_dragging = true;
                                    pi.float_drag_ox = mouse.x - px;
                                    pi.float_drag_oy = mouse.y - py;
                                }

                                if (pi.float_dragging) {
                                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                                        px = mouse.x - pi.float_drag_ox;
                                        py = mouse.y - pi.float_drag_oy;
                                    } else {
                                        pi.float_dragging = false;
                                    }
                                }
                                if (pi.float_resizing) {
                                    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                                        const float delta = mouse.x - pi.float_resize_start_mx;
                                        w = std::max(kMinPanelWidth, pi.float_resize_start_w + delta);
                                        pi.initial_width = w;
                                    } else {
                                        pi.float_resizing = false;
                                    }
                                }

                                pi.float_x = px;
                                pi.float_y = py;
                            }
                        }

                        if (mouse_in_panel || mouse_in_resize_grip)
                            io.WantCaptureMouse = true;

                        {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                                if (panels_[snap.index].float_resizing || mouse_in_resize_grip)
                                    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
                            }
                        }

                        snap.panel->drawDirect(px, py, w, h, ctx);
                    } else {
                        if (snap.initial_width > 0 || snap.initial_height > 0)
                            ImGui::SetNextWindowSize(ImVec2(snap.initial_width, snap.initial_height), ImGuiCond_Appearing);
                        bool open = true;
                        if (ImGui::Begin(snap.label.c_str(), &open)) {
                            snap.panel->draw(ctx);
                        }
                        ImGui::End();
                        if (!open) {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                                panels_[snap.index].enabled = false;
                            }
                        }
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

                case PanelSpace::Dockable: {
                    if (snap.has_option(PanelOption::SELF_MANAGED)) {
                        snap.panel->draw(ctx);
                    } else {
                        if (snap.initial_width > 0 || snap.initial_height > 0)
                            ImGui::SetNextWindowSize(ImVec2(snap.initial_width, snap.initial_height), ImGuiCond_Appearing);
                        bool open = true;
                        if (ImGui::Begin(snap.label.c_str(), &open)) {
                            snap.panel->draw(ctx);
                        }
                        ImGui::End();
                        if (!open) {
                            std::lock_guard lock(mutex_);
                            if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                                panels_[snap.index].enabled = false;
                            }
                        }
                    }
                    break;
                }
                case PanelSpace::StatusBar: {
                    constexpr float STATUS_BAR_HEIGHT = 22.0f;
                    constexpr float PADDING = 8.0f;
                    const auto* vp = ImGui::GetMainViewport();
                    const ImVec2 bar_pos{vp->WorkPos.x, vp->WorkPos.y + vp->WorkSize.y - STATUS_BAR_HEIGHT};
                    const ImVec2 bar_size{vp->WorkSize.x, STATUS_BAR_HEIGHT};

                    ImGui::SetNextWindowPos(bar_pos, ImGuiCond_Always);
                    ImGui::SetNextWindowSize(bar_size, ImGuiCond_Always);

                    constexpr ImGuiWindowFlags FLAGS =
                        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoSavedSettings |
                        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoFocusOnAppearing;

                    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
                    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0, 0, 0, 0));
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {PADDING, 3.0f});
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6.0f, 0.0f});
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, {1.0f, 1.0f});

                    if (ImGui::Begin("##StatusBar", nullptr, FLAGS)) {
                        snap.panel->draw(ctx);
                    }
                    ImGui::End();

                    ImGui::PopStyleVar(5);
                    ImGui::PopStyleColor(2);
                    break;
                }
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

    float PanelRegistry::draw_panels_direct(PanelSpace space, float x, float y, float w,
                                            float max_h, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.space == space && p.enabled && !p.error_disabled && p.parent_idname.empty()) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.idname,
                                         p.parent_idname, p.options, p.is_native,
                                         p.poll_deps, p.initial_width, p.initial_height,
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
                snap.panel->drawDirect(x, y + y_offset, w, remaining, ctx);
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

    void PanelRegistry::draw_single_panel(const std::string& idname, const PanelDrawContext& ctx) {
        std::shared_ptr<IPanel> panel_holder;
        PanelSnapshot snap{};
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                if (panels_[i].idname == idname && panels_[i].enabled && !panels_[i].error_disabled) {
                    panel_holder = panels_[i].panel;
                    snap = {i, panels_[i].panel.get(), panels_[i].label, panels_[i].idname,
                            panels_[i].parent_idname, panels_[i].options, panels_[i].is_native,
                            panels_[i].poll_deps, panels_[i].initial_width, panels_[i].initial_height,
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
            ImGui::PushID(snap.idname.c_str());
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
            if (p.space == space && p.enabled && !p.error_disabled && p.parent_idname.empty())
                return true;
        }
        return false;
    }

    std::vector<PanelSummary> PanelRegistry::get_panels_for_space(PanelSpace space) {
        std::lock_guard lock(mutex_);
        std::vector<PanelSummary> result;
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled && !p.error_disabled && p.parent_idname.empty())
                result.push_back({p.label, p.idname, p.space, p.order, p.enabled});
        }
        std::stable_sort(result.begin(), result.end(), [](const PanelSummary& a, const PanelSummary& b) {
            if (a.order != b.order)
                return a.order < b.order;
            return a.label < b.label;
        });
        return result;
    }

    std::optional<PanelSummary> PanelRegistry::get_panel(const std::string& idname) {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.idname == idname)
                return PanelSummary{p.label, p.idname, p.space, p.order, p.enabled};
        }
        return std::nullopt;
    }

    std::vector<std::string> PanelRegistry::get_panel_names(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> names;
        for (const auto& p : panels_) {
            if (p.space == space)
                names.push_back(p.idname);
        }
        return names;
    }

    void PanelRegistry::set_panel_enabled(const std::string& idname, bool enabled) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.enabled = enabled;
                if (enabled && p.space == PanelSpace::Floating) {
                    p.float_x = -1;
                    p.float_y = -1;
                }
                return;
            }
        }
    }

    void PanelRegistry::set_panel_disabled_override(const std::string& idname) {
        std::lock_guard lock(mutex_);
        disabled_overrides_.insert(idname);
        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.enabled = false;
                return;
            }
        }
    }

    bool PanelRegistry::is_panel_enabled(const std::string& idname) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.idname == idname)
                return p.enabled;
        }
        return false;
    }

    bool PanelRegistry::set_panel_label(const std::string& idname, const std::string& new_label) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.label = new_label;
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::set_panel_order(const std::string& idname, int new_order) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.idname == idname) {
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

    bool PanelRegistry::set_panel_space(const std::string& idname, PanelSpace new_space) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.space = new_space;
                return true;
            }
        }
        return false;
    }

    bool PanelRegistry::set_panel_parent(const std::string& idname, const std::string& parent_idname) {
        std::lock_guard lock(mutex_);

        if (!parent_idname.empty()) {
            bool parent_found = false;
            for (const auto& p : panels_) {
                if (p.idname == parent_idname) {
                    parent_found = true;
                    break;
                }
            }
            if (!parent_found)
                LOG_WARN("Panel '{}': parent '{}' not registered (may register later)", idname, parent_idname);
        }

        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.parent_idname = parent_idname;
                return true;
            }
        }
        return false;
    }

    void PanelRegistry::draw_child_panels(const std::string& parent_idname, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.parent_idname == parent_idname && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.idname,
                                         p.parent_idname, p.options, p.is_native,
                                         p.poll_deps, p.initial_width, p.initial_height,
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
                ImGui::PushID(snap.idname.c_str());

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

    float PanelRegistry::draw_single_panel_direct(const std::string& idname, float x, float y,
                                                  float w, float h, const PanelDrawContext& ctx) {
        std::shared_ptr<IPanel> panel_holder;
        PanelSnapshot snap{};
        bool found = false;
        {
            std::lock_guard lock(mutex_);
            for (size_t i = 0; i < panels_.size(); ++i) {
                if (panels_[i].idname == idname && panels_[i].enabled && !panels_[i].error_disabled) {
                    panel_holder = panels_[i].panel;
                    snap = {i, panels_[i].panel.get(), panels_[i].label, panels_[i].idname,
                            panels_[i].parent_idname, panels_[i].options, panels_[i].is_native,
                            panels_[i].poll_deps, panels_[i].initial_width, panels_[i].initial_height,
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
            snap.panel->drawDirect(x, y, w, h, ctx);
            draw_succeeded = true;
        } catch (const std::exception& e) {
            LOG_ERROR("Panel '{}' drawDirect error: {}", snap.label, e.what());
        }

        track_draw_result(snap, draw_succeeded);
        const float used = snap.panel->getDirectDrawHeight();
        return used > 0 ? used : 0.0f;
    }

    float PanelRegistry::draw_child_panels_direct(const std::string& parent_idname, float x, float y,
                                                  float w, float h, const PanelDrawContext& ctx) {
        std::vector<PanelSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            snapshots.reserve(panels_.size());
            for (size_t i = 0; i < panels_.size(); ++i) {
                auto& p = panels_[i];
                if (p.parent_idname == parent_idname && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.idname,
                                         p.parent_idname, p.options, p.is_native,
                                         p.poll_deps, p.initial_width, p.initial_height,
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
                snap.panel->drawDirect(x, y + y_offset, w, remaining, ctx);
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
        if (snap.index >= panels_.size() || panels_[snap.index].idname != snap.idname)
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
            return (pair.second.deps & changed) != PollDependency::NONE;
        });
    }

} // namespace lfs::vis::gui
