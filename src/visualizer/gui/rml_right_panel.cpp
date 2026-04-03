/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rml_right_panel.hpp"
#include "core/logger.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/rmlui_render_interface.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <algorithm>
#include <cassert>
#include <format>

namespace lfs::vis::gui {

    void RmlRightPanel::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("right_panel", 400, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlRightPanel: failed to create RML context");
            return;
        }

        auto ctor = rml_context_->CreateDataModel("right_panel_tabs");
        assert(ctor);

        if (auto h = ctor.RegisterStruct<TabSnapshot>()) {
            h.RegisterMember("id", &TabSnapshot::id);
            h.RegisterMember("label", &TabSnapshot::label);
            h.RegisterMember("dom_id", &TabSnapshot::dom_id);
        }
        ctor.RegisterArray<std::vector<TabSnapshot>>();
        ctor.Bind("tabs", &tabs_);
        ctor.Bind("active_tab", &active_tab_);
        ctor.Bind("tabs_overflow", &tabs_overflow_);
        ctor.Bind("can_scroll_tabs_left", &can_scroll_tabs_left_);
        ctor.Bind("can_scroll_tabs_right", &can_scroll_tabs_right_);

        ctor.BindEventCallback("tab_click",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
                                   if (!args.empty()) {
                                       auto id = args[0].Get<Rml::String>();
                                       if (!id.empty() && on_tab_changed)
                                           on_tab_changed(std::string(id));
                                   }
                               });
        ctor.BindEventCallback("scroll_tabs_left",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
                                   if (can_scroll_tabs_left_)
                                       scrollTabs(-1.0f);
                               });
        ctor.BindEventCallback("scroll_tabs_right",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
                                   if (can_scroll_tabs_right_)
                                       scrollTabs(1.0f);
                               });

        tab_model_ = ctor.GetModelHandle();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/right_panel.rml");
            document_ = rml_context_->LoadDocument(rml_path.string());
            if (!document_) {
                LOG_ERROR("RmlRightPanel: failed to load right_panel.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlRightPanel: resource not found: {}", e.what());
            return;
        }

        resize_handle_el_ = document_->GetElementById("resize-handle");
        left_border_el_ = document_->GetElementById("left-border");
        splitter_el_ = document_->GetElementById("splitter");
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_strip_viewport_el_ = document_->GetElementById("tab-strip-viewport");
        tab_separator_el_ = document_->GetElementById("tab-separator");

        updateTheme();
    }

    void RmlRightPanel::shutdown() {
        tab_model_ = {};
        tabs_.clear();
        active_tab_.clear();
        fbo_.destroy();
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("right_panel");
        rml_context_ = nullptr;
        document_ = nullptr;
        resize_handle_el_ = nullptr;
        left_border_el_ = nullptr;
        splitter_el_ = nullptr;
        tab_bar_el_ = nullptr;
        tab_strip_viewport_el_ = nullptr;
        tab_separator_el_ = nullptr;
        tab_scroll_left_ = 0.0f;
        tabs_overflow_ = false;
        can_scroll_tabs_left_ = false;
        can_scroll_tabs_right_ = false;
    }

    std::string RmlRightPanel::generateThemeRCSS(const lfs::vis::Theme& t) const {
        using rml_theme::colorToRml;
        using rml_theme::colorToRmlAlpha;
        const auto& p = t.palette;

        const auto tab_hover = colorToRmlAlpha(p.surface_bright, 0.5f);
        const auto tab_active_bg = colorToRmlAlpha(p.surface_bright, 0.4f);
        const auto tab_accent = colorToRml(p.primary);
        const auto tab_text = colorToRml(p.text);
        const auto tab_text_dim = colorToRml(p.text_dim);
        const auto tab_nav_bg = colorToRmlAlpha(p.surface_bright, 0.2f);
        const auto tab_nav_hover = colorToRmlAlpha(p.surface_bright, 0.55f);
        const auto tab_nav_disabled = colorToRmlAlpha(p.text_dim, 0.35f);
        const auto splitter_bg = colorToRmlAlpha(p.border, 0.4f);
        const auto splitter_hover = colorToRmlAlpha(p.info, 0.6f);
        const auto splitter_active = colorToRmlAlpha(p.info, 0.8f);
        const auto border_color = colorToRmlAlpha(p.border, 0.6f);
        const auto separator_color = colorToRmlAlpha(p.border, 0.4f);
        const auto resize_hover = colorToRmlAlpha(p.info, 0.3f);
        const auto resize_active = colorToRmlAlpha(p.info, 0.5f);

        return std::format(
            "#splitter {{ background-color: {}; }}\n"
            "#splitter:hover {{ background-color: {}; }}\n"
            "#splitter.dragging {{ background-color: {}; }}\n"
            ".tab {{ background-color: transparent; color: {}; }}\n"
            ".tab:hover {{ background-color: {}; }}\n"
            ".tab:focus-visible {{ background-color: {}; color: {}; border-bottom-color: {}; }}\n"
            ".tab.active {{ background-color: {}; color: {}; "
            "border-bottom-color: {}; }}\n"
            ".tab-nav {{ background-color: {}; color: {}; }}\n"
            ".tab-nav:hover {{ background-color: {}; color: {}; }}\n"
            ".tab-nav.disabled {{ background-color: transparent; color: {}; opacity: 0.55; }}\n"
            ".tab-nav.disabled:hover {{ background-color: transparent; color: {}; opacity: 0.55; }}\n"
            "#left-border {{ background-color: {}; }}\n"
            "#tab-separator {{ background-color: {}; }}\n"
            "#resize-handle:hover {{ background-color: {}; }}\n"
            "#resize-handle.dragging {{ background-color: {}; }}\n",
            splitter_bg, splitter_hover, splitter_active,
            tab_text_dim,
            tab_hover,
            tab_hover, tab_text, tab_accent,
            tab_active_bg, tab_text, tab_accent,
            tab_nav_bg, tab_text_dim,
            tab_nav_hover, tab_text,
            tab_nav_disabled,
            tab_nav_disabled,
            border_color,
            separator_color,
            resize_hover,
            resize_active);
    }

    bool RmlRightPanel::updateTheme() {
        if (!document_)
            return false;

        const std::size_t theme_signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && theme_signature == last_theme_signature_)
            return false;
        last_theme_signature_ = theme_signature;
        has_theme_signature_ = true;

        if (base_rcss_.empty())
            base_rcss_ = rml_theme::loadBaseRCSS("rmlui/right_panel.rcss");

        rml_theme::applyTheme(document_, base_rcss_, rml_theme::generateAllThemeMedia([this](const auto& th) { return generateThemeRCSS(th); }));
        return true;
    }

    bool RmlRightPanel::syncTabData(const std::vector<TabSnapshot>& tabs,
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

    void RmlRightPanel::scrollTabs(float delta) {
        if (!tab_strip_viewport_el_ || !document_ || tabs_.empty() || delta == 0.0f)
            return;

        const float max_scroll = std::max(
            0.0f, tab_strip_viewport_el_->GetScrollWidth() - tab_strip_viewport_el_->GetClientWidth());
        const float current_scroll = std::clamp(tab_scroll_left_, 0.0f, max_scroll);
        const float viewport_left =
            tab_strip_viewport_el_->GetAbsoluteOffset(Rml::BoxArea::Border).x;
        const float epsilon = 0.5f;
        float next_scroll = current_scroll;

        if (delta > 0.0f) {
            next_scroll = max_scroll;
            for (const auto& tab : tabs_) {
                if (tab.dom_id.empty())
                    continue;
                auto* button = document_->GetElementById(tab.dom_id);
                if (!button)
                    continue;

                const float tab_left =
                    current_scroll +
                    (button->GetAbsoluteOffset(Rml::BoxArea::Border).x - viewport_left);
                if (tab_left > current_scroll + epsilon) {
                    next_scroll = tab_left;
                    break;
                }
            }
        } else {
            next_scroll = 0.0f;
            for (auto it = tabs_.rbegin(); it != tabs_.rend(); ++it) {
                if (it->dom_id.empty())
                    continue;
                auto* button = document_->GetElementById(it->dom_id);
                if (!button)
                    continue;

                const float tab_left =
                    current_scroll +
                    (button->GetAbsoluteOffset(Rml::BoxArea::Border).x - viewport_left);
                if (tab_left < current_scroll - epsilon) {
                    next_scroll = tab_left;
                    break;
                }
            }
        }

        next_scroll = std::clamp(next_scroll, 0.0f, max_scroll);
        if (next_scroll == tab_scroll_left_)
            return;

        tab_scroll_left_ = next_scroll;
        render_needed_ = true;
        input_dirty_ = true;
    }

    bool RmlRightPanel::syncTabScrollState() {
        if (!tab_bar_el_ || !tab_strip_viewport_el_)
            return false;

        bool dirty = false;

        const float full_bar_width = tab_bar_el_->GetClientWidth();
        const float content_width = tab_strip_viewport_el_->GetScrollWidth();
        const bool tabs_overflow = content_width > full_bar_width + 0.5f;
        if (tabs_overflow_ != tabs_overflow) {
            tabs_overflow_ = tabs_overflow;
            tab_model_.DirtyVariable("tabs_overflow");
            dirty = true;
        }

        const float viewport_width = tab_strip_viewport_el_->GetClientWidth();
        const float max_scroll = tabs_overflow_
                                     ? std::max(0.0f, content_width - viewport_width)
                                     : 0.0f;
        float next_scroll = std::clamp(tab_scroll_left_, 0.0f, max_scroll);

        if (next_scroll != tab_scroll_left_) {
            tab_scroll_left_ = next_scroll;
            dirty = true;
        }

        if (tab_strip_viewport_el_->GetScrollLeft() != tab_scroll_left_)
            tab_strip_viewport_el_->SetScrollLeft(tab_scroll_left_);

        const bool can_scroll_left = tabs_overflow_ && tab_scroll_left_ > 0.5f;
        const bool can_scroll_right = tabs_overflow_ && tab_scroll_left_ < max_scroll - 0.5f;
        if (can_scroll_tabs_left_ != can_scroll_left) {
            can_scroll_tabs_left_ = can_scroll_left;
            tab_model_.DirtyVariable("can_scroll_tabs_left");
            dirty = true;
        }
        if (can_scroll_tabs_right_ != can_scroll_right) {
            can_scroll_tabs_right_ = can_scroll_right;
            tab_model_.DirtyVariable("can_scroll_tabs_right");
            dirty = true;
        }

        return dirty;
    }

    void RmlRightPanel::syncTabNavigation() {
        if (!document_)
            return;

        const std::size_t count = tabs_.size();
        for (std::size_t i = 0; i < count; ++i) {
            const auto& tab = tabs_[i];
            if (tab.dom_id.empty())
                continue;

            auto* button = document_->GetElementById(tab.dom_id);
            if (!button)
                continue;

            const std::string left_id = "#" + tabs_[(i + count - 1) % count].dom_id;
            const std::string right_id = "#" + tabs_[(i + 1) % count].dom_id;
            button->SetProperty("nav-left", left_id);
            button->SetProperty("nav-right", right_id);
        }
    }

    static bool isOrHasAncestor(Rml::Element* el, const Rml::String& id) {
        while (el) {
            if (el->GetId() == id)
                return true;
            el = el->GetParentNode();
        }
        return false;
    }

    CursorRequest RmlRightPanel::getCursorRequest() const {
        return cursor_request_;
    }

    void RmlRightPanel::blurFocus() {
        if (!rml_context_)
            return;

        auto* const focused = rml_context_->GetFocusElement();
        if (!focused)
            return;

        focused->Blur();
        wants_keyboard_ = false;
        input_dirty_ = true;
    }

    bool RmlRightPanel::needsAnimationFrame() const {
        return render_needed_ || input_dirty_ || splitter_dragging_ || resize_dragging_;
    }

    void RmlRightPanel::processInput(const RightPanelLayout& layout, const PanelInputState& input) {
        wants_input_ = false;
        wants_keyboard_ = false;
        cursor_request_ = CursorRequest::None;

        const float delta_x = input.mouse_x - prev_mouse_x_;
        const float delta_y = input.mouse_y - prev_mouse_y_;
        const bool mouse_moved = (delta_x != 0.0f || delta_y != 0.0f);
        prev_mouse_x_ = input.mouse_x;
        prev_mouse_y_ = input.mouse_y;

        if (!rml_context_ || !document_)
            return;
        if (layout.size.x <= 0 || layout.size.y <= 0)
            return;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(layout.pos.x - input.screen_x),
                                            static_cast<int>(layout.pos.y - input.screen_y));
        }

        const float mx = input.mouse_x - layout.pos.x;
        const float my = input.mouse_y - layout.pos.y;

        if (mouse_moved)
            rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), 0);

        auto* hover = rml_context_->GetHoverElement();
        const bool over_interactive = hover && hover->GetTagName() != "body" &&
                                      hover->GetId() != "rp-body" &&
                                      hover->GetId() != "left-border" &&
                                      hover->GetId() != "tab-separator";

        if (over_interactive != last_over_interactive_) {
            input_dirty_ = true;
            last_over_interactive_ = over_interactive;
        } else if (mouse_moved && over_interactive) {
            input_dirty_ = true;
        }

        if (resize_dragging_) {
            wants_input_ = true;
            input_dirty_ = true;

            if (input.mouse_down[0]) {
                if (on_resize_delta && delta_x != 0.0f)
                    on_resize_delta(delta_x);
                cursor_request_ = CursorRequest::ResizeEW;
            } else {
                resize_dragging_ = false;
                if (resize_handle_el_)
                    resize_handle_el_->SetAttribute("class", "");
                if (on_resize_end)
                    on_resize_end();
            }
            return;
        }

        if (splitter_dragging_) {
            wants_input_ = true;
            input_dirty_ = true;

            if (input.mouse_down[0]) {
                if (on_splitter_delta && delta_y != 0.0f)
                    on_splitter_delta(delta_y);
                cursor_request_ = CursorRequest::ResizeNS;
            } else {
                splitter_dragging_ = false;
                if (splitter_el_)
                    splitter_el_->SetAttribute("class", "");
                if (on_splitter_end)
                    on_splitter_end();
            }
            return;
        }

        if (over_interactive) {
            wants_input_ = true;

            if (isOrHasAncestor(hover, "resize-handle")) {
                cursor_request_ = CursorRequest::ResizeEW;
                if (input.mouse_clicked[0]) {
                    resize_dragging_ = true;
                    input_dirty_ = true;
                    if (resize_handle_el_)
                        resize_handle_el_->SetAttribute("class", "dragging");
                }
            } else if (isOrHasAncestor(hover, "splitter")) {
                cursor_request_ = CursorRequest::ResizeNS;
                if (input.mouse_clicked[0]) {
                    splitter_dragging_ = true;
                    input_dirty_ = true;
                    if (splitter_el_)
                        splitter_el_->SetAttribute("class", "dragging");
                }
            } else {
                if (input.mouse_clicked[0]) {
                    input_dirty_ = true;
                    rml_context_->ProcessMouseButtonDown(0, 0);
                }
                if (input.mouse_released[0])
                    rml_context_->ProcessMouseButtonUp(0, 0);
            }
        } else if (input.mouse_clicked[0]) {
            if (auto* focused = rml_context_->GetFocusElement())
                focused->Blur();
        }

        if (input.viewport_keyboard_focus) {
            if (auto* focused = rml_context_->GetFocusElement())
                focused->Blur();
        }

        if (rml_input::hasFocusedKeyboardTarget(rml_context_->GetFocusElement()) &&
            !input.viewport_keyboard_focus) {
            const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                          input.key_alt, input.key_super);
            for (const int sc : input.keys_pressed) {
                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyDown(rml_key, mods);
                    input_dirty_ = true;
                }
            }
            for (const int sc : input.keys_released) {
                const auto rml_key = sdlScancodeToRml(static_cast<SDL_Scancode>(sc));
                if (rml_key != Rml::Input::KI_UNKNOWN) {
                    rml_context_->ProcessKeyUp(rml_key, mods);
                    input_dirty_ = true;
                }
            }
        }

        auto* focused = rml_context_->GetFocusElement();
        wants_keyboard_ = rml_input::hasFocusedKeyboardTarget(focused);
        wants_input_ = wants_input_ || wants_keyboard_;
    }

    void RmlRightPanel::render(const RightPanelLayout& layout,
                               const std::vector<TabSnapshot>& tabs,
                               const std::string& active_tab,
                               float screen_x, float screen_y,
                               int screen_w, int screen_h) {
        if (!rml_context_ || !document_)
            return;
        if (layout.size.x <= 0 || layout.size.y <= 0)
            return;
        if (rml_manager_) {
            rml_manager_->trackContextFrame(rml_context_,
                                            static_cast<int>(layout.pos.x - screen_x),
                                            static_cast<int>(layout.pos.y - screen_y));
        }

        const bool theme_changed = updateTheme();

        const int w = static_cast<int>(layout.size.x);
        const int h = static_cast<int>(layout.size.y);

        if (w <= 0 || h <= 0)
            return;

        const bool dims_changed = (w != last_fbo_w_ || h != last_fbo_h_);
        const bool layout_changed = (layout.scene_h != last_scene_h_ ||
                                     layout.splitter_h != last_splitter_h_);
        const bool tabs_changed = syncTabData(tabs, active_tab);

        const bool needs_render = render_needed_ || theme_changed || layout_changed ||
                                  tabs_changed || dims_changed || input_dirty_;

        if (needs_render && !rml_manager_->shouldDeferFboUpdate(fbo_)) {
            const float dp_ratio = rml_manager_->getDpRatio();
            const float tab_bar_h = PanelLayoutManager::TAB_BAR_H * dp_ratio;

            if (resize_handle_el_) {
                resize_handle_el_->SetProperty("top", "0px");
                resize_handle_el_->SetProperty("height", std::format("{:.0f}px", layout.size.y));
            }
            if (left_border_el_) {
                left_border_el_->SetProperty("top", "0px");
                left_border_el_->SetProperty("height", std::format("{:.0f}px", layout.size.y));
            }
            if (splitter_el_) {
                splitter_el_->SetProperty("top", std::format("{:.0f}px", layout.scene_h));
                splitter_el_->SetProperty("height", std::format("{:.0f}px", layout.splitter_h));
            }
            if (tab_bar_el_) {
                const float tab_top = layout.scene_h + layout.splitter_h;
                tab_bar_el_->SetProperty("top", std::format("{:.0f}px", tab_top));
                tab_bar_el_->SetProperty("height", std::format("{:.0f}px", tab_bar_h));
            }
            if (tab_separator_el_) {
                const float sep_top = layout.scene_h + layout.splitter_h + tab_bar_h;
                tab_separator_el_->SetProperty("top", std::format("{:.0f}px", sep_top));
            }

            rml_context_->SetDimensions(Rml::Vector2i(w, h));
            for (int pass = 0; pass < 3; ++pass) {
                rml_context_->Update();
                syncTabNavigation();
                if (!syncTabScrollState())
                    break;
            }
            syncTabNavigation();

            fbo_.ensure(w, h);
            if (!fbo_.valid())
                return;

            auto* render = rml_manager_->getRenderInterface();
            assert(render);
            render->SetViewport(w, h);

            GLint prev_fbo = 0;
            fbo_.bind(&prev_fbo);
            render->SetTargetFramebuffer(fbo_.fbo());

            render->BeginFrame();
            rml_context_->Render();
            render->EndFrame();

            render->SetTargetFramebuffer(0);
            fbo_.unbind(prev_fbo);

            last_fbo_w_ = w;
            last_fbo_h_ = h;
            last_scene_h_ = layout.scene_h;
            last_splitter_h_ = layout.splitter_h;
            render_needed_ = false;
            input_dirty_ = false;
        }

        if (fbo_.valid())
            fbo_.blitToScreen(layout.pos.x - screen_x,
                              layout.pos.y - screen_y,
                              layout.size.x, layout.size.y,
                              screen_w, screen_h);
    }

} // namespace lfs::vis::gui
