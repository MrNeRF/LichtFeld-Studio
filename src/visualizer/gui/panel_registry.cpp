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
                if (p.space == space && p.enabled && !p.error_disabled) {
                    snapshots.push_back({i, p.panel.get(), p.label, p.idname,
                                         p.options, p.is_native, p.poll_deps});
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
                    } else {
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

                    const auto& t = theme();
                    ImGui::PushStyleColor(ImGuiCol_WindowBg, withAlpha(t.palette.background, 0.95f));
                    ImGui::PushStyleColor(ImGuiCol_Border, withAlpha(t.palette.border, 0.6f));
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {PADDING, 3.0f});
                    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {6.0f, 0.0f});
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);

                    if (ImGui::Begin("##StatusBar", nullptr, FLAGS)) {
                        ImGui::GetWindowDrawList()->AddLine(
                            bar_pos, {bar_pos.x + bar_size.x, bar_pos.y},
                            toU32(withAlpha(t.palette.surface_bright, 0.4f)), 1.0f);
                        snap.panel->draw(ctx);
                    }
                    ImGui::End();

                    ImGui::PopStyleVar(4);
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

            if (!draw_succeeded && !snap.is_native) {
                std::lock_guard lock(mutex_);
                if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                    panels_[snap.index].consecutive_errors++;
                    if (panels_[snap.index].consecutive_errors >= PanelInfo::MAX_CONSECUTIVE_ERRORS) {
                        panels_[snap.index].error_disabled = true;
                        LOG_ERROR("Panel '{}' disabled after {} errors",
                                  snap.label, panels_[snap.index].consecutive_errors);
                    }
                }
            } else if (draw_succeeded) {
                std::lock_guard lock(mutex_);
                if (snap.index < panels_.size() && panels_[snap.index].idname == snap.idname) {
                    panels_[snap.index].consecutive_errors = 0;
                }
            }
        }
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
                            panels_[i].options, panels_[i].is_native, panels_[i].poll_deps};
                    found = true;
                    break;
                }
            }
        }

        if (!found)
            return;

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

        if (!draw_succeeded && !snap.is_native) {
            std::lock_guard lock(mutex_);
            if (snap.index < panels_.size() && panels_[snap.index].idname == idname) {
                panels_[snap.index].consecutive_errors++;
                if (panels_[snap.index].consecutive_errors >= PanelInfo::MAX_CONSECUTIVE_ERRORS) {
                    panels_[snap.index].error_disabled = true;
                    LOG_ERROR("Panel '{}' disabled after {} errors",
                              snap.label, panels_[snap.index].consecutive_errors);
                }
            }
        } else if (draw_succeeded) {
            std::lock_guard lock(mutex_);
            if (snap.index < panels_.size() && panels_[snap.index].idname == idname) {
                panels_[snap.index].consecutive_errors = 0;
            }
        }
    }

    bool PanelRegistry::has_panels(PanelSpace space) const {
        std::lock_guard lock(mutex_);
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled && !p.error_disabled)
                return true;
        }
        return false;
    }

    std::vector<PanelSummary> PanelRegistry::get_panels_for_space(PanelSpace space) {
        std::lock_guard lock(mutex_);
        std::vector<PanelSummary> result;
        for (const auto& p : panels_) {
            if (p.space == space && p.enabled && !p.error_disabled)
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
                names.push_back(p.label);
        }
        return names;
    }

    void PanelRegistry::set_panel_enabled(const std::string& idname, bool enabled) {
        std::lock_guard lock(mutex_);
        for (auto& p : panels_) {
            if (p.idname == idname) {
                p.enabled = enabled;
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
