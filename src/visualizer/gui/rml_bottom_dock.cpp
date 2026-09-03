/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_bottom_dock.hpp"

#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <algorithm>
#include <cassert>
#include <format>
#include <limits>

namespace lfs::vis::gui {

    void RmlBottomDock::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;
        rml_context_ = rml_manager_->createContext("bottom_dock", 400, 320);
        if (!rml_context_) {
            LOG_ERROR("RmlBottomDock: failed to create RML context");
            return;
        }

        auto ctor = rml_context_->CreateDataModel("bottom_dock_tabs");
        assert(ctor);
        if (auto h = ctor.RegisterStruct<TabSnapshot>()) {
            h.RegisterMember("id", &TabSnapshot::id);
            h.RegisterMember("label", &TabSnapshot::label);
            h.RegisterMember("dom_id", &TabSnapshot::dom_id);
            h.RegisterMember("closeable", &TabSnapshot::closeable);
        }
        ctor.RegisterArray<std::vector<TabSnapshot>>();
        ctor.Bind("tabs", &tabs_);
        ctor.Bind("active_tab", &active_tab_);
        ctor.Bind("tabs_overflow", &tabs_overflow_);
        ctor.Bind("can_scroll_tabs_left", &can_scroll_tabs_left_);
        ctor.Bind("can_scroll_tabs_right", &can_scroll_tabs_right_);
        ctor.BindEventCallback("tab_click", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
            if (!args.empty() && on_tab_changed) {
                const auto id = args[0].Get<Rml::String>();
                if (!id.empty())
                    on_tab_changed(std::string(id));
            }
        });
        ctor.BindEventCallback("tab_close", [this](Rml::DataModelHandle, Rml::Event& event,
                                                   const Rml::VariantList& args) {
            event.StopPropagation();
            if (!args.empty() && on_tab_closed) {
                const auto id = args[0].Get<Rml::String>();
                if (!id.empty())
                    on_tab_closed(std::string(id));
            }
        });
        ctor.BindEventCallback("scroll_tabs_left", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
            if (can_scroll_tabs_left_)
                scrollTabs(-1.0f);
        });
        ctor.BindEventCallback("scroll_tabs_right", [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
            if (can_scroll_tabs_right_)
                scrollTabs(1.0f);
        });
        tab_model_ = ctor.GetModelHandle();

        try {
            document_ = rml_documents::loadDocument(
                rml_context_, lfs::vis::getAssetPath("rmlui/bottom_dock.rml"));
            if (!document_) {
                LOG_ERROR("RmlBottomDock: failed to load bottom_dock.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlBottomDock: resource not found: {}", e.what());
            return;
        }
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_strip_viewport_el_ = document_->GetElementById("tab-strip-viewport");
        tab_separator_el_ = document_->GetElementById("tab-separator");
        updateTheme();
    }

    void RmlBottomDock::shutdown() {
        setVisible(false);
        tab_model_ = {};
        tabs_.clear();
        active_tab_.clear();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("bottom_dock");
        rml_context_ = nullptr;
        document_ = nullptr;
        tab_bar_el_ = nullptr;
        tab_strip_viewport_el_ = nullptr;
        tab_separator_el_ = nullptr;
        last_blurred_focus_ = nullptr;
        last_hover_element_ = nullptr;
    }

    void RmlBottomDock::setVisible(const bool visible) {
        if (visible) {
            if (visible_)
                return;
            visible_ = true;
            render_needed_ = true;
            input_dirty_ = true;
            return;
        }

        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        visible_ = false;
        wants_input_ = false;
        wants_keyboard_ = false;
        cursor_request_ = CursorRequest::None;
        rml_pointer_inside_ = false;
        last_over_interactive_ = false;
        last_hover_element_ = nullptr;
        render_needed_ = false;
        input_dirty_ = false;
    }

    void RmlBottomDock::reloadResources() {
        if (!rml_context_)
            return;
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }
        document_ = nullptr;
        tab_bar_el_ = nullptr;
        tab_strip_viewport_el_ = nullptr;
        base_rcss_.clear();
        has_theme_signature_ = false;
        render_needed_ = true;
        input_dirty_ = true;
        last_fbo_w_ = 0;
        last_fbo_h_ = 0;
        try {
            document_ = rml_documents::loadDocument(
                rml_context_, lfs::vis::getAssetPath("rmlui/bottom_dock.rml"));
            if (!document_) {
                LOG_ERROR("RmlBottomDock: failed to reload bottom_dock.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlBottomDock: resource not found during reload: {}", e.what());
            return;
        }
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_strip_viewport_el_ = document_->GetElementById("tab-strip-viewport");
        tab_separator_el_ = document_->GetElementById("tab-separator");
        tab_model_.DirtyVariable("tabs");
        tab_model_.DirtyVariable("active_tab");
        tab_model_.DirtyVariable("tabs_overflow");
        tab_model_.DirtyVariable("can_scroll_tabs_left");
        tab_model_.DirtyVariable("can_scroll_tabs_right");
        updateTheme();
    }

    bool RmlBottomDock::updateTheme() {
        if (!document_)
            return false;
        const auto signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && signature == last_theme_signature_)
            return false;
        last_theme_signature_ = signature;
        has_theme_signature_ = true;
        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/bottom_dock.rcss") + "\n" +
                         rml_theme::loadBaseRCSS("rmlui/panel_tabs.rcss");
        rml_theme::applyTheme(document_, base_rcss_,
                              rml_theme::loadBaseRCSS("rmlui/panel_tabs.theme.rcss") + "\n" +
                                  rml_theme::loadBaseRCSS("rmlui/bottom_dock.theme.rcss"));
        return true;
    }

    bool RmlBottomDock::syncTabData(const std::vector<TabSnapshot>& tabs,
                                    const std::string& active_tab) {
        bool dirty = false;
        if (tabs_ != tabs) {
            tabs_ = tabs;
            tab_model_.DirtyVariable("tabs");
            dirty = true;
        }
        if (active_tab_ != active_tab) {
            active_tab_ = active_tab;
            tab_model_.DirtyVariable("active_tab");
            dirty = true;
        }
        return dirty;
    }

    void RmlBottomDock::scrollTabs(const float delta) {
        if (!tab_strip_viewport_el_ || tabs_.empty() || delta == 0.0f)
            return;
        const float max_scroll = std::max(0.0f, tab_strip_viewport_el_->GetScrollWidth() -
                                                    tab_strip_viewport_el_->GetClientWidth());
        tab_scroll_left_ = std::clamp(tab_scroll_left_ + delta * tab_strip_viewport_el_->GetClientWidth(),
                                      0.0f, max_scroll);
        render_needed_ = true;
        input_dirty_ = true;
    }

    bool RmlBottomDock::syncTabScrollState() {
        if (!tab_bar_el_ || !tab_strip_viewport_el_)
            return false;
        bool dirty = false;
        const bool overflow = tab_strip_viewport_el_->GetScrollWidth() > tab_bar_el_->GetClientWidth() + 0.5f;
        if (tabs_overflow_ != overflow) {
            tabs_overflow_ = overflow;
            tab_model_.DirtyVariable("tabs_overflow");
            dirty = true;
        }
        const float max_scroll = overflow ? std::max(0.0f, tab_strip_viewport_el_->GetScrollWidth() -
                                                               tab_strip_viewport_el_->GetClientWidth())
                                          : 0.0f;
        tab_scroll_left_ = std::clamp(tab_scroll_left_, 0.0f, max_scroll);
        if (tab_strip_viewport_el_->GetScrollLeft() != tab_scroll_left_)
            tab_strip_viewport_el_->SetScrollLeft(tab_scroll_left_);
        const bool can_left = overflow && tab_scroll_left_ > 0.5f;
        const bool can_right = overflow && tab_scroll_left_ < max_scroll - 0.5f;
        if (can_scroll_tabs_left_ != can_left) {
            can_scroll_tabs_left_ = can_left;
            tab_model_.DirtyVariable("can_scroll_tabs_left");
            dirty = true;
        }
        if (can_scroll_tabs_right_ != can_right) {
            can_scroll_tabs_right_ = can_right;
            tab_model_.DirtyVariable("can_scroll_tabs_right");
            dirty = true;
        }
        return dirty;
    }

    void RmlBottomDock::syncTabNavigation() {
        if (!document_ || tabs_.empty())
            return;
        for (size_t i = 0; i < tabs_.size(); ++i) {
            if (tabs_[i].dom_id.empty())
                continue;
            auto* button = document_->GetElementById(tabs_[i].dom_id);
            if (!button)
                continue;
            const size_t count = tabs_.size();
            button->SetProperty("nav-left", "#" + tabs_[(i + count - 1) % count].dom_id);
            button->SetProperty("nav-right", "#" + tabs_[(i + 1) % count].dom_id);
        }
    }

    void RmlBottomDock::processInput(const BottomDockLayout& layout,
                                     const PanelInputState& input) {
        const auto previous_cursor = cursor_request_;
        wants_input_ = false;
        wants_keyboard_ = false;
        cursor_request_ = CursorRequest::None;
        if (!visible_ || !rml_context_ || !document_ || layout.size.x <= 0 || layout.size.y <= 0)
            return;
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, static_cast<int>(layout.pos.x - input.screen_x),
                                            static_cast<int>(layout.pos.y - input.screen_y));
        const float mx = input.mouse_x - layout.pos.x;
        const float my = input.mouse_y - layout.pos.y;
        const bool moved = mx != prev_mouse_x_ || my != prev_mouse_y_;
        prev_mouse_x_ = input.mouse_x;
        prev_mouse_y_ = input.mouse_y;
        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift, input.key_alt, input.key_super);
        const float chrome_h = layout.tab_bar_h + layout.separator_h;
        const bool inside = mx >= 0 && mx < layout.size.x && my >= 0 && my < chrome_h;
        if (inside) {
            if (moved)
                rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), mods);
            rml_pointer_inside_ = true;
        } else if (rml_pointer_inside_) {
            rml_context_->ProcessMouseLeave();
            rml_pointer_inside_ = false;
            last_over_interactive_ = false;
            input_dirty_ = true;
        }
        auto* hover = inside ? rml_context_->GetHoverElement() : nullptr;
        const bool over_interactive = hover && hover->GetTagName() != "body" &&
                                      hover->GetId() != "bottom-dock-body";
        if (over_interactive != last_over_interactive_ || (moved && over_interactive && hover != last_hover_element_))
            input_dirty_ = true;
        last_over_interactive_ = over_interactive;
        last_hover_element_ = hover;
        if (over_interactive) {
            wants_input_ = true;
            if (input.mouse_clicked[0]) {
                rml_context_->ProcessMouseButtonDown(0, mods);
                input_dirty_ = true;
            }
            if (input.mouse_released[0]) {
                rml_context_->ProcessMouseButtonUp(0, mods);
                input_dirty_ = true;
            }
            if (input.mouse_wheel != 0.0f) {
                rml_context_->ProcessMouseWheel(Rml::Vector2f(-input.mouse_wheel_x, -input.mouse_wheel), mods);
                input_dirty_ = true;
            }
        } else if (input.mouse_clicked[0]) {
            blurFocus();
        }
        if (input.viewport_keyboard_focus)
            blurFocus();
        auto* focused = rml_context_->GetFocusElement();
        wants_keyboard_ = rml_input::hasFocusedKeyboardTarget(focused);
        wants_input_ = wants_input_ || wants_keyboard_;
        if (rml_input::wantsTextInput(focused))
            guiFocusState().want_text_input = true;
        if (!moved && !input.mouse_clicked[0] && !input.mouse_released[0] && !input.mouse_wheel &&
            !wants_input_)
            cursor_request_ = previous_cursor;
    }

    void RmlBottomDock::render(const BottomDockLayout& layout,
                               const std::vector<TabSnapshot>& tabs,
                               const std::string& active_tab, float screen_x, float screen_y,
                               int screen_w, int screen_h) {
        (void)screen_w;
        (void)screen_h;
        if (!rml_context_ || !document_ || layout.size.x <= 0 || layout.size.y <= 0)
            return;
        setVisible(true);
        if (rml_manager_)
            rml_manager_->trackContextFrame(rml_context_, static_cast<int>(layout.pos.x - screen_x),
                                            static_cast<int>(layout.pos.y - screen_y));
        const bool theme_changed = updateTheme();
        const int w = static_cast<int>(layout.size.x);
        const int h = static_cast<int>(layout.tab_bar_h + layout.separator_h);
        const bool dims_changed = w != last_fbo_w_ || h != last_fbo_h_;
        const bool tabs_changed = syncTabData(tabs, active_tab);
        const bool needs_render = render_needed_ || input_dirty_ || theme_changed || dims_changed || tabs_changed;
        if (needs_render) {
            if (tab_bar_el_) {
                tab_bar_el_->SetProperty("top", "0px");
                tab_bar_el_->SetProperty("height", std::format("{:.0f}px", layout.tab_bar_h));
            }
            if (tab_separator_el_) {
                tab_separator_el_->SetProperty("top", std::format("{:.0f}px", layout.tab_bar_h));
                tab_separator_el_->SetProperty("height", std::format("{:.0f}px", layout.separator_h));
            }
            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            for (int pass = 0; pass < 3; ++pass) {
                rml_context_->Update();
                syncTabNavigation();
                if (!syncTabScrollState())
                    break;
            }
            last_fbo_w_ = w;
            last_fbo_h_ = h;
            render_needed_ = false;
            input_dirty_ = false;
        }
        if (!rml_manager_ || !rml_manager_->getVulkanRenderInterface())
            return;
        const float x = layout.pos.x - screen_x;
        const float y = layout.pos.y - screen_y;
        rml_manager_->queueCachedVulkanContext({
            .context = rml_context_,
            .cache = &direct_cache_,
            .cache_width = w,
            .cache_height = h,
            .offset_x = x,
            .offset_y = y,
            .draw_width = static_cast<float>(w),
            .draw_height = static_cast<float>(h),
            .refresh = needs_render || direct_cache_.texture == 0 ||
                       direct_cache_.width != w || direct_cache_.height != h,
            .foreground = false,
            .clip_enabled = true,
            .clip = {.x1 = x, .y1 = y, .x2 = x + w, .y2 = y + h},
        });
    }

    void RmlBottomDock::blurFocus() {
        if (!rml_context_)
            return;
        auto* focused = rml_context_->GetFocusElement();
        if (!focused) {
            last_blurred_focus_ = nullptr;
            return;
        }
        if (focused == last_blurred_focus_) {
            wants_keyboard_ = false;
            return;
        }
        focused->Blur();
        last_blurred_focus_ = focused;
        wants_keyboard_ = false;
        input_dirty_ = true;
    }

    bool RmlBottomDock::needsAnimationFrame() const {
        return visible_ && (render_needed_ || input_dirty_);
    }

    std::string RmlBottomDock::animationDemandDescription() const {
        if (!needsAnimationFrame())
            return {};
        const double delay = rml_context_ ? rml_context_->GetNextUpdateDelay() : std::numeric_limits<double>::infinity();
        return std::format("bottom_dock(render_needed={},input_dirty={},rml_delay={})",
                           render_needed_, input_dirty_, delay);
    }

    CursorRequest RmlBottomDock::getCursorRequest() const {
        return cursor_request_;
    }

} // namespace lfs::vis::gui
