/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rml_right_panel.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/panel_layout.hpp"
#include "gui/panel_registry.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_manager.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "input/input_controller.hpp"
#include "input/sdl_key_mapping.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>

namespace lfs::vis::gui {

    namespace {
        bool setPixelProperty(Rml::Element* element, Rml::PropertyId id, float value) {
            const auto* current = element->GetLocalProperty(id);
            if (current && current->unit == Rml::Unit::PX && std::abs(current->Get<float>() - value) < 0.001f)
                return false;
            return element->SetProperty(id, Rml::Property(value, Rml::Unit::PX));
        }
    } // namespace

    void RmlRightPanel::init(RmlUIManager* mgr) {
        assert(mgr);
        rml_manager_ = mgr;

        rml_context_ = rml_manager_->createContext("right_panel", 400, 600);
        if (!rml_context_) {
            LOG_ERROR("RmlRightPanel: failed to create RML context");
            return;
        }

        bindTabModel();

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/right_panel.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
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
        splitter_el_ = document_->GetElementById("splitter");
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_strip_viewport_el_ = document_->GetElementById("tab-strip-viewport");

        updateTheme();
    }

    void RmlRightPanel::bindTabModel() {
        auto ctor = rml_context_->CreateDataModel("right_panel_tabs");
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
        ctor.Bind("can_scroll_tabs_up", &can_scroll_tabs_up_);
        ctor.Bind("can_scroll_tabs_down", &can_scroll_tabs_down_);
        ctor.BindEventCallback("scroll_tabs_up",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
                                   scrollTabs(-tab_strip_viewport_el_->GetClientHeight() * 0.75f);
                               });
        ctor.BindEventCallback("scroll_tabs_down",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList&) {
                                   scrollTabs(tab_strip_viewport_el_->GetClientHeight() * 0.75f);
                               });
        ctor.BindEventCallback("tab_close",
                               [this](Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& args) {
                                   event.StopPropagation();
                                   if (!args.empty() && on_tab_closed) {
                                       const auto id = args[0].Get<Rml::String>();
                                       if (!id.empty())
                                           on_tab_closed(id);
                                   }
                               });
        ctor.BindEventCallback("tab_click",
                               [this](Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& args) {
                                   if (!args.empty()) {
                                       auto id = args[0].Get<Rml::String>();
                                       if (!id.empty() && on_tab_changed)
                                           selectTab(std::string(id));
                                   }
                               });
        ctor.BindEventCallback("tab_keydown",
                               [this](Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& args) {
                                   const auto key = static_cast<Rml::Input::KeyIdentifier>(event.GetParameter("key_identifier", 0));
                                   if ((key == Rml::Input::KI_RETURN || key == Rml::Input::KI_SPACE) &&
                                       event.GetTargetElement()->IsClassSet("tab") && !args.empty()) {
                                       event.StopPropagation();
                                       const auto id = args[0].Get<Rml::String>();
                                       if (!id.empty() && on_tab_changed)
                                           selectTab(id);
                                   }
                               });
        tab_model_ = ctor.GetModelHandle();
    }

    void RmlRightPanel::shutdown() {
        tab_model_ = {};
        tabs_.clear();
        tab_text_metrics_.clear();
        tab_scroll_pending_ = false;
        tab_layout_dirty_ = true;
        tab_navigation_dirty_ = true;
        tabs_overflow_ = false;
        can_scroll_tabs_up_ = false;
        can_scroll_tabs_down_ = false;
        active_tab_.clear();
        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);
        if (rml_context_ && rml_manager_)
            rml_manager_->destroyContext("right_panel");
        rml_context_ = nullptr;
        document_ = nullptr;
        resize_handle_el_ = nullptr;
        splitter_el_ = nullptr;
        tab_bar_el_ = nullptr;
        tooltip_.setHover({}, nullptr);
        tab_strip_viewport_el_ = nullptr;
        tab_scroll_top_ = 0.0f;
        reveal_active_tab_ = true;
        tab_scroll_dragging_ = false;
        last_blurred_focus_ = nullptr;
        last_hover_element_ = nullptr;
    }

    void RmlRightPanel::reloadResources() {
        if (!rml_context_)
            return;

        if (rml_manager_)
            rml_manager_->releaseCachedVulkanContext(direct_cache_);

        if (document_) {
            rml_context_->UnloadDocument(document_);
            rml_context_->Update();
        }

        document_ = nullptr;
        resize_handle_el_ = nullptr;
        splitter_el_ = nullptr;
        tab_bar_el_ = nullptr;
        tooltip_.setHover({}, nullptr);
        tab_strip_viewport_el_ = nullptr;
        base_rcss_.clear();
        tab_text_metrics_.assign(tabs_.size(), {});
        tab_scroll_pending_ = false;
        tab_layout_dirty_ = true;
        tab_navigation_dirty_ = true;
        has_theme_signature_ = false;
        wants_input_ = false;
        wants_keyboard_ = false;
        splitter_dragging_ = false;
        resize_dragging_ = false;
        render_needed_ = true;
        input_dirty_ = true;
        last_fbo_w_ = 0;
        last_fbo_h_ = 0;
        last_scene_h_ = -1.0f;
        last_splitter_h_ = -1.0f;
        last_over_interactive_ = false;
        last_over_resize_handle_ = false;
        rml_pointer_inside_ = false;
        last_blurred_focus_ = nullptr;
        last_hover_element_ = nullptr;

        try {
            const auto rml_path = lfs::vis::getAssetPath("rmlui/right_panel.rml");
            document_ = rml_documents::loadDocument(rml_context_, rml_path);
            if (!document_) {
                LOG_ERROR("RmlRightPanel: failed to reload right_panel.rml");
                return;
            }
            document_->Show();
        } catch (const std::exception& e) {
            LOG_ERROR("RmlRightPanel: resource not found during reload: {}", e.what());
            return;
        }

        resize_handle_el_ = document_->GetElementById("resize-handle");
        splitter_el_ = document_->GetElementById("splitter");
        tab_bar_el_ = document_->GetElementById("tab-bar");
        tab_strip_viewport_el_ = document_->GetElementById("tab-strip-viewport");

        tab_model_.DirtyVariable("tabs");
        tab_model_.DirtyVariable("active_tab");
        reveal_active_tab_ = true;
        tab_scroll_dragging_ = false;
        updateTheme();
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

        rml_theme::applyTheme(document_, base_rcss_,
                              rml_theme::loadBaseRCSS("rmlui/right_panel.theme.rcss"));
        return true;
    }

    bool RmlRightPanel::syncTabData(const std::vector<TabSnapshot>& tabs,
                                    const std::string& active_tab) {
        bool dirty = false;

        if (tabs_ != tabs) {
            tabs_ = tabs;
            tab_text_metrics_.assign(tabs_.size(), {});
            tab_layout_dirty_ = true;
            tab_navigation_dirty_ = true;
            reveal_active_tab_ = true;
            tab_model_.DirtyVariable("tabs");
            dirty = true;
        }

        if (active_tab_ != active_tab) {
            active_tab_ = active_tab;
            tab_layout_dirty_ = true;
            reveal_active_tab_ = true;
            tab_model_.DirtyVariable("active_tab");
            dirty = true;
        }

        return dirty;
    }

    void RmlRightPanel::updateTabs() {
        rml_context_->Update();
        if (tab_strip_viewport_el_ && !tab_scroll_pending_ &&
            tab_strip_viewport_el_->GetScrollTop() != tab_scroll_top_) {
            tab_scroll_top_ = tab_strip_viewport_el_->GetScrollTop();
            tab_layout_dirty_ = true;
        }
        if (tab_navigation_dirty_) {
            syncTabNavigation();
            tab_navigation_dirty_ = false;
        }
        if (!tab_layout_dirty_)
            return;

        // Overflow controls change the viewport size; font and scroll changes
        // can then move labels. Hover redraws need only the context update above.
        for (int pass = 0; pass < 4; ++pass) {
            const bool layout_changed = layoutTabs();
            if (layout_changed)
                rml_context_->Update();
            const bool scroll_changed = syncTabScrollState();
            if (scroll_changed)
                rml_context_->Update();
            if (!layout_changed && !scroll_changed)
                break;
        }
        tab_layout_dirty_ = false;
        tab_scroll_pending_ = false;
    }

    bool RmlRightPanel::layoutTabs() {
        bool changed = false;
        const float dp = rml_context_->GetDensityIndependentPixelRatio();
        constexpr float text_clearance_dp = 2.0f;
        const float line_height = std::ceil(20.0f * dp);
        for (std::size_t i = 0; i < tabs_.size(); ++i) {
            const auto& tab = tabs_[i];
            auto* button = document_->GetElementById(tab.dom_id);
            if (!button)
                continue;
            auto* label = button->GetChild(0);
            if (!label)
                continue;
            // Rotation changes drawing and hit testing, but not layout. Reserve
            // the measured text length on the vertical axis, including plugin labels.
            auto& metrics = tab_text_metrics_[i];
            const auto font = label->GetFontFaceHandle();
            if (metrics.font != font || metrics.dp != dp) {
                metrics = {font, dp, std::ceil(Rml::ElementUtilities::GetStringWidth(label, tab.label) + text_clearance_dp * dp)};
            }
            const float text_width = metrics.width;
            const float close_height = tab.closeable ? 18.0f * dp : 0.0f;
            const float height = std::ceil(std::max(28.0f * dp, text_width + 18.0f * dp + close_height));
            changed |= setPixelProperty(button, Rml::PropertyId::Height, height);
            // A rotated line box must fit across the tab. Snap its origin and
            // dimensions in physical pixels so fractional scales do not clip or
            // blur glyph edges. Keep clearance for the font's ink overhang.
            const auto origin = button->GetAbsoluteOffset(Rml::BoxArea::Padding);
            const float content_width = button->GetBox().GetSize(Rml::BoxArea::Padding).x;
            const float content_height = button->GetBox().GetSize(Rml::BoxArea::Padding).y;
            const float left = std::round(origin.x + (content_width - line_height) * 0.5f) - origin.x;
            // Center the rotated text along the tab. Exclude its trailing ink
            // clearance from the text center.
            const float top = std::round(origin.y + (content_height - close_height + text_width - text_clearance_dp * dp) * 0.5f) - origin.y;
            changed |= setPixelProperty(label, Rml::PropertyId::Width, text_width);
            changed |= setPixelProperty(label, Rml::PropertyId::Height, line_height);
            changed |= setPixelProperty(label, Rml::PropertyId::LineHeight, line_height);
            changed |= setPixelProperty(label, Rml::PropertyId::Left, left);
            changed |= setPixelProperty(label, Rml::PropertyId::Top, top);
        }
        return changed;
    }

    void RmlRightPanel::scrollTabs(float delta) {
        if (!tab_strip_viewport_el_ || delta == 0.0f)
            return;
        if (!tab_scroll_pending_)
            tab_scroll_top_ = tab_strip_viewport_el_->GetScrollTop();
        const float max_scroll = std::max(0.0f, tab_strip_viewport_el_->GetScrollHeight() -
                                                    tab_strip_viewport_el_->GetClientHeight());
        setTabStripScroll(std::clamp(tab_scroll_top_ + delta, 0.0f, max_scroll));
    }

    void RmlRightPanel::cycleTab(int direction) {
        if (tabs_.empty() || !on_tab_changed)
            return;
        const auto current = std::find_if(tabs_.begin(), tabs_.end(),
                                          [this](const auto& tab) { return tab.id == active_tab_; });
        int index = current == tabs_.end() ? 0 : static_cast<int>(current - tabs_.begin());
        const int count = static_cast<int>(tabs_.size());
        index += direction;
        index = (index + count) % count;
        selectTab(tabs_[index].id);
    }

    void RmlRightPanel::selectTab(std::string id) {
        if (id.empty() || !on_tab_changed)
            return;
        syncTabData(tabs_, id);
        render_needed_ = true;
        input_dirty_ = true;
        on_tab_changed(id);
    }

    bool RmlRightPanel::syncTabScrollState() {
        if (!tab_strip_viewport_el_)
            return false;
        const float dp = rml_context_->GetDensityIndependentPixelRatio();
        const float rail_height = tab_bar_el_ ? tab_bar_el_->GetClientHeight() : tab_strip_viewport_el_->GetClientHeight();
        const bool overflow = tab_strip_viewport_el_->GetScrollHeight() > rail_height - 6.0f * dp + 0.5f;
        if (tabs_overflow_ != overflow) {
            tabs_overflow_ = overflow;
            tab_model_.DirtyVariable("tabs_overflow");
            return true;
        }
        const float viewport_height = tab_strip_viewport_el_->GetClientHeight();
        const float max_scroll = std::max(0.0f, tab_strip_viewport_el_->GetScrollHeight() - viewport_height);
        float next_scroll = std::clamp(tab_scroll_top_, 0.0f, max_scroll);
        const auto active = std::find_if(tabs_.begin(), tabs_.end(),
                                         [this](const auto& tab) { return tab.id == active_tab_; });
        auto* button = active != tabs_.end() ? document_->GetElementById(active->dom_id) : nullptr;
        const float active_top = button
                                     ? button->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                                           tab_strip_viewport_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y +
                                           tab_strip_viewport_el_->GetScrollTop()
                                     : 0.0f;
        const float active_height = button ? button->GetBox().GetSize(Rml::BoxArea::Border).y : 0.0f;
        if (reveal_active_tab_) {
            if (button) {
                if (active_top < next_scroll || active_height > viewport_height)
                    next_scroll = active_top;
                else if (active_top + active_height > next_scroll + viewport_height)
                    next_scroll = active_top + active_height - viewport_height;
            }
            next_scroll = std::clamp(next_scroll, 0.0f, max_scroll);
            reveal_active_tab_ = false;
        }
        const float previous_scroll = tab_strip_viewport_el_->GetScrollTop();
        if (previous_scroll != next_scroll)
            tab_strip_viewport_el_->SetScrollTop(next_scroll);
        // RmlUi rounds scroll offsets to physical pixels, including at the
        // fractional-scale scroll limit. Keep the model and rail at that offset.
        next_scroll = tab_strip_viewport_el_->GetScrollTop();
        tab_scroll_top_ = next_scroll;
        bool changed = previous_scroll != next_scroll;
        const auto set_flag = [&](bool& flag, bool value, const char* name) {
            if (flag != value) {
                flag = value;
                tab_model_.DirtyVariable(name);
                changed = true;
            }
        };
        set_flag(can_scroll_tabs_up_, next_scroll > 0.5f, "can_scroll_tabs_up");
        set_flag(can_scroll_tabs_down_, next_scroll < max_scroll - 0.5f, "can_scroll_tabs_down");
        const float viewport_top = tab_bar_el_
                                       ? tab_strip_viewport_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y -
                                             tab_bar_el_->GetAbsoluteOffset(Rml::BoxArea::Border).y
                                       : 0.0f;
        float open_top = rail_height;
        float open_bottom = rail_height;
        if (button) {
            open_top = std::clamp(viewport_top + active_top - next_scroll, 0.0f, rail_height);
            open_bottom = std::clamp(viewport_top + active_top + active_height - next_scroll, 0.0f, rail_height);
        }
        if (auto* edge = document_->GetElementById("tab-edge-before")) {
            changed |= setPixelProperty(edge, Rml::PropertyId::Top, viewport_top);
            changed |= setPixelProperty(edge, Rml::PropertyId::Height, std::max(0.0f, open_top - viewport_top));
        }
        if (auto* edge = document_->GetElementById("tab-edge-after")) {
            const float edge_top = std::max(viewport_top, open_bottom);
            changed |= setPixelProperty(edge, Rml::PropertyId::Top, edge_top);
            changed |= setPixelProperty(edge, Rml::PropertyId::Height, std::max(0.0f, rail_height - edge_top));
        }
        return changed;
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

            const std::string previous_id = "#" + tabs_[(i + count - 1) % count].dom_id;
            const std::string next_id = "#" + tabs_[(i + 1) % count].dom_id;
            button->SetProperty("nav-up", previous_id);
            button->SetProperty("nav-down", next_id);
        }
    }

    void RmlRightPanel::setTabStripScroll(const float value) {
        if (!std::isfinite(value))
            return;
        const float next = std::max(0.0f, value);
        if (next == tab_scroll_top_)
            return;
        tab_scroll_top_ = next;
        tab_scroll_pending_ = true;
        tab_layout_dirty_ = true;
        render_needed_ = true;
        input_dirty_ = true;
    }

    CursorRequest RmlRightPanel::getCursorRequest() const {
        return cursor_request_;
    }

    void RmlRightPanel::blurFocus() {
        if (!rml_context_)
            return;

        auto* const focused = rml_context_->GetFocusElement();
        if (!focused) {
            last_blurred_focus_ = nullptr;
            return;
        }

        // Frame-input capture is applied after the right-panel render. RmlUi can
        // retain the same focus element until its next Update(), so repeatedly
        // marking this flag here would keep needsAnimationFrame() true forever.
        if (focused == last_blurred_focus_) {
            wants_keyboard_ = false;
            return;
        }

        focused->Blur();
        last_blurred_focus_ = focused;
        wants_keyboard_ = false;
        input_dirty_ = true;
    }

    bool RmlRightPanel::needsAnimationFrame() const {
        return render_needed_ || input_dirty_ || splitter_dragging_ || resize_dragging_ || tab_scroll_dragging_ || tooltip_.revealDue();
    }

    std::string RmlRightPanel::animationDemandDescription() const {
        if (!needsAnimationFrame())
            return {};

        const double next_update_delay = rml_context_
                                             ? rml_context_->GetNextUpdateDelay()
                                             : std::numeric_limits<double>::infinity();
        return std::format(
            "right_panel(render_needed={},input_dirty={},splitter_dragging={},resize_dragging={},rml_delay={})",
            render_needed_, input_dirty_, splitter_dragging_, resize_dragging_,
            next_update_delay);
    }

    bool RmlRightPanel::inputOccluded(float x, float y) const {
        return PanelRegistry::instance().isPositionOverFloatingPanel(x, y) ||
               (rml_manager_ && rml_manager_->activeOverlayOccludesContext(rml_context_, x, y));
    }

    bool RmlRightPanel::interceptCategoryInput(const RightPanelLayout& layout, const PanelInputState& input) {
        if (splitter_dragging_ || resize_dragging_ || tab_scroll_dragging_ ||
            inputOccluded(input.mouse_x, input.mouse_y))
            return false;
        const float dp = rml_context_->GetDensityIndependentPixelRatio();
        const float x = input.mouse_x - layout.pos.x;
        const float y = input.mouse_y - layout.pos.y;
        if (x < 0 || x >= layout.size.x ||
            y < layout.scene_h + layout.splitter_h + PanelLayoutManager::RIGHT_PANEL_SECTION_GAP * dp || y >= layout.size.y)
            return false;
        const auto* controller = InputController::instance();
        if (!controller || controller->getBindings().isCapturing())
            return false;
        const auto& bindings = controller->getBindings();
        // A focused text field keeps keys that type into it; Ctrl or Cmd chords still switch.
        // Ctrl+Alt stays with the field because it is AltGr on Windows.
        const bool typing = guiFocusState().want_text_input || (rml_manager_ && rml_manager_->wantsTextInput());
        bool handled = false;
        for (const auto& event : input.input_events) {
            if (event.kind != FrameInputEventKind::KeyDown ||
                (event.dispatch && event.dispatch->consumed) || (event.modifiers & SDL_KMOD_MODE))
                continue;
            if (typing && (!(event.modifiers & (SDL_KMOD_CTRL | SDL_KMOD_GUI)) || (event.modifiers & SDL_KMOD_ALT)))
                continue;
            bool repeats = false;
            const auto printed = SDL_GetKeyFromScancode(event.scancode, event.modifiers & SDL_KMOD_SHIFT, false);
            const auto unshifted = SDL_GetKeyFromScancode(event.scancode, SDL_KMOD_NONE, false);
            const auto action = bindings.getActionForShortcut(
                input::ToolMode::GLOBAL,
                input::sdlScancodeToAppKey(event.scancode),
                input::sdlShortcutKeyToAppKey(event.scancode, event.keycode),
                input::sdlModsToAppMods(event.modifiers),
                input::sdlPrintedKeyToAppKey(printed),
                (event.modifiers & SDL_KMOD_SHIFT) && printed != unshifted, &repeats);
            if (action != input::Action::NEXT_SIDEBAR_CATEGORY &&
                action != input::Action::PREVIOUS_SIDEBAR_CATEGORY)
                continue;
            if (!event.repeat || repeats)
                cycleTab(action == input::Action::NEXT_SIDEBAR_CATEGORY ? 1 : -1);
            if (event.dispatch)
                event.dispatch->consumed = true;
            handled = true;
        }
        return handled;
    }

    void RmlRightPanel::processInput(const RightPanelLayout& layout, const PanelInputState& input) {
        if (rml_manager_ && rml_context_ && rml_manager_->routeInput(rml_context_, input, [this, layout](const PanelInputState& event) { processInput(layout, event); }, false, [this](float x, float y) { return inputOccluded(x, y); }, [this, layout](const PanelInputState& event) { return interceptCategoryInput(layout, event); }))
            return;
        const CursorRequest previous_cursor_request = cursor_request_;
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

        const bool pointer_event =
            input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2] ||
            input.mouse_released[0] || input.mouse_released[1] || input.mouse_released[2] ||
            input.mouse_wheel != 0.0f;
        const bool pointer_down =
            input.mouse_down[0] || input.mouse_down[1] || input.mouse_down[2];
        const bool keyboard_event =
            !input.keys_pressed.empty() || !input.input_events.empty();
        auto* const focused_before = rml_context_->GetFocusElement();
        if (pointer_event || keyboard_event)
            last_blurred_focus_ = nullptr;
        const bool viewport_focus_blurs_panel = input.viewport_keyboard_focus && focused_before;
        const bool layout_changed =
            static_cast<int>(layout.size.x) != last_fbo_w_ ||
            static_cast<int>(layout.size.y) != last_fbo_h_ ||
            layout.scene_h != last_scene_h_ ||
            layout.splitter_h != last_splitter_h_;
        if (!mouse_moved && !pointer_event && !pointer_down && !keyboard_event &&
            !resize_dragging_ && !splitter_dragging_ && !tab_scroll_dragging_ && !viewport_focus_blurs_panel &&
            !layout_changed) {
            wants_keyboard_ = rml_input::hasFocusedKeyboardTarget(focused_before);
            wants_input_ = wants_keyboard_ || last_over_interactive_ ||
                           previous_cursor_request != CursorRequest::None;
            cursor_request_ = previous_cursor_request;
            if (rml_input::wantsTextInput(focused_before))
                guiFocusState().want_text_input = true;
            return;
        }

        const float mx = input.mouse_x - layout.pos.x;
        const float my = input.mouse_y - layout.pos.y;
        const float dp_ratio = rml_manager_ ? rml_manager_->getDpRatio()
                                            : rml_context_->GetDensityIndependentPixelRatio();
        const auto resize_zone = resizeHitZone(layout.pos.x, dp_ratio);
        const auto splitter_zone = resizeHitZone(
            layout.pos.y + layout.scene_h + layout.splitter_h * 0.5f, dp_ratio);

        const int mods = sdlModsToRml(input.key_ctrl, input.key_shift,
                                      input.key_alt, input.key_super);

        const bool pointer_in_context =
            mx >= 0.0f && mx < layout.size.x && my >= 0.0f && my < layout.size.y;
        if (pointer_in_context) {
            if (mouse_moved)
                rml_context_->ProcessMouseMove(static_cast<int>(mx), static_cast<int>(my), mods);
            rml_pointer_inside_ = true;
        } else if (rml_pointer_inside_) {
            // RmlUi does not clear hover merely because it receives an
            // out-of-bounds position. Explicitly leave so it cannot retain a
            // cursor or request frames after the pointer entered the viewport.
            rml_context_->ProcessMouseLeave();
            rml_pointer_inside_ = false;
            last_over_interactive_ = false;
            input_dirty_ = true;
        }
        auto* hover = pointer_in_context ? rml_context_->GetHoverElement() : nullptr;
        const bool over_resize_handle_geom =
            resize_zone.contains(input.mouse_x) &&
            my >= 0.0f &&
            my < layout.size.y;
        const bool over_resize_handle = over_resize_handle_geom;
        if (resize_handle_el_ && !resize_dragging_ &&
            over_resize_handle != last_over_resize_handle_) {
            resize_handle_el_->SetAttribute("class", over_resize_handle ? "hover" : "");
            last_over_resize_handle_ = over_resize_handle;
            input_dirty_ = true;
        }
        const bool over_splitter =
            mx >= 0.0f && mx < layout.size.x &&
            splitter_zone.contains(input.mouse_y);
        const bool over_interactive = hover && hover->GetTagName() != "body" &&
                                      hover->GetId() != "rp-body" &&
                                      hover->GetId() != "left-border";

        if (over_interactive != last_over_interactive_) {
            input_dirty_ = true;
            last_over_interactive_ = over_interactive;
        } else if (mouse_moved && over_interactive && hover != last_hover_element_) {
            input_dirty_ = true;
        }
        last_hover_element_ = hover;
        tooltip_.setHover(resolveRmlTooltip(hover), hover);

        const bool over_tabs = tab_bar_el_ &&
                               mx >= 8.0f && mx < 8.0f + PanelLayoutManager::RIGHT_TAB_BAR_W * dp_ratio && my >= layout.scene_h + layout.splitter_h + PanelLayoutManager::RIGHT_PANEL_SECTION_GAP * dp_ratio &&
                               my < layout.size.y;
        const bool cycled = interceptCategoryInput(layout, input);
        if (cycled) {
            wants_input_ = true;
            input_dirty_ = true;
        } else if (over_tabs && input.mouse_wheel != 0.0f) {
            scrollTabs(-input.mouse_wheel * 30.0f * dp_ratio);
            wants_input_ = true;
        }
        if (over_tabs && input.mouse_clicked[2]) {
            if (tab_strip_viewport_el_ && !tab_scroll_pending_)
                tab_scroll_top_ = tab_strip_viewport_el_->GetScrollTop();
            tab_scroll_dragging_ = true;
            tab_drag_start_y_ = input.mouse_y;
            tab_drag_start_scroll_ = tab_scroll_top_;
        }
        if (tab_scroll_dragging_) {
            wants_input_ = true;
            input_dirty_ = true;
            if (input.mouse_down[2]) {
                setTabStripScroll(std::max(0.0f, tab_drag_start_scroll_ + tab_drag_start_y_ - input.mouse_y));
            } else {
                tab_scroll_dragging_ = false;
            }
            return;
        }

        if (resize_dragging_) {
            wants_input_ = true;
            input_dirty_ = true;

            if (input.mouse_down[0]) {
                if (on_resize_width)
                    on_resize_width(resize_start_width_ - (input.mouse_x - resize_start_mouse_x_));
                cursor_request_ = CursorRequest::ResizeEW;
            } else {
                resize_dragging_ = false;
                if (resize_handle_el_) {
                    resize_handle_el_->SetAttribute("class", "");
                    last_over_resize_handle_ = false;
                }
                if (on_resize_end)
                    on_resize_end();
            }
            return;
        }

        if (splitter_dragging_) {
            wants_input_ = true;
            input_dirty_ = true;

            if (input.mouse_down[0]) {
                if (on_splitter_height)
                    on_splitter_height(splitter_start_scene_h_ + input.mouse_y - splitter_start_mouse_y_,
                                       layout.size.y);
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

        if (over_interactive || over_resize_handle || over_splitter) {
            wants_input_ = true;
            if (over_resize_handle) {
                cursor_request_ = CursorRequest::ResizeEW;
                if (input.mouse_clicked[0]) {
                    resize_dragging_ = true;
                    resize_start_mouse_x_ = input.mouse_x;
                    resize_start_width_ = layout.size.x;
                    input_dirty_ = true;
                    if (resize_handle_el_)
                        resize_handle_el_->SetAttribute("class", "dragging");
                }
            } else if (over_splitter) {
                cursor_request_ = CursorRequest::ResizeNS;
                if (input.mouse_clicked[0]) {
                    splitter_dragging_ = true;
                    splitter_start_mouse_y_ = input.mouse_y;
                    splitter_start_scene_h_ = layout.scene_h;
                    input_dirty_ = true;
                    if (splitter_el_)
                        splitter_el_->SetAttribute("class", "dragging");
                }
            } else {
                if (input.mouse_clicked[0]) {
                    input_dirty_ = true;
                    rml_context_->ProcessMouseButtonDown(0, mods);
                }
                if (input.mouse_released[0]) {
                    rml_context_->ProcessMouseButtonUp(0, mods);
                    tooltip_.setHover(resolveRmlTooltip(hover), hover);
                }
            }
        } else if (input.mouse_clicked[0]) {
            blurFocus();
        }

        if (input.viewport_keyboard_focus)
            blurFocus();

        if (!cycled && rml_input::hasFocusedKeyboardTarget(rml_context_->GetFocusElement()) &&
            !input.viewport_keyboard_focus) {
            for (const auto& event : input.input_events)
                input_dirty_ |= rml_input::processKeyboardEvent(*rml_context_, event, rml_manager_->getTextInputHandler());
        }

        auto* focused = rml_context_->GetFocusElement();
        wants_keyboard_ = rml_input::hasFocusedKeyboardTarget(focused);
        wants_input_ = wants_input_ || wants_keyboard_;
        if (rml_input::wantsTextInput(focused))
            guiFocusState().want_text_input = true;
    }

    void RmlRightPanel::render(const RightPanelLayout& layout,
                               const std::vector<TabSnapshot>& tabs,
                               const std::string& active_tab,
                               float screen_x, float screen_y,
                               int screen_w, int screen_h) {
        (void)screen_w;
        (void)screen_h;
        if (!rml_context_ || !document_)
            return;
        if (layout.size.x <= 0 || layout.size.y <= 0)
            return;
        const bool theme_changed = updateTheme();

        const int w = static_cast<int>(layout.size.x);
        const int h = static_cast<int>(layout.size.y);

        if (w <= 0 || h <= 0)
            return;

        const float dp_ratio = rml_manager_->getDpRatio();
        const bool dims_changed = (w != last_fbo_w_ || h != last_fbo_h_ || dp_ratio != last_dp_ratio_);
        const bool layout_changed = (layout.scene_h != last_scene_h_ ||
                                     layout.splitter_h != last_splitter_h_);
        const bool tabs_changed = syncTabData(tabs, active_tab);

        const bool tooltip_changed = tooltip_.apply(document_,
                                                    static_cast<int>(prev_mouse_x_ - layout.pos.x),
                                                    static_cast<int>(prev_mouse_y_ - layout.pos.y), w, h);
        rml_manager_->setContextNeedsPassiveMouseMoveFrames(rml_context_, tooltip_.hasActiveState());
        rml_manager_->setContextTooltipRevealDeadline(rml_context_, tooltip_.revealDeadline());
        const bool needs_render = render_needed_ || theme_changed || layout_changed ||
                                  tabs_changed || dims_changed || input_dirty_ || tooltip_changed;

        if (needs_render) {
            const float tab_bar_w = PanelLayoutManager::RIGHT_TAB_BAR_W * dp_ratio;
            if (dims_changed || layout_changed)
                reveal_active_tab_ = true;

            const bool geometry_changed = dims_changed || layout_changed || theme_changed;
            if (theme_changed)
                tab_text_metrics_.assign(tabs_.size(), {});
            tab_layout_dirty_ |= geometry_changed;
            if (geometry_changed) {
                if (splitter_el_) {
                    setPixelProperty(splitter_el_, Rml::PropertyId::Top, std::round(layout.scene_h));
                    setPixelProperty(splitter_el_, Rml::PropertyId::Height, std::round(layout.splitter_h));
                }
                if (tab_bar_el_) {
                    const float tab_top = layout.scene_h + layout.splitter_h +
                                          PanelLayoutManager::RIGHT_PANEL_SECTION_GAP * dp_ratio;
                    setPixelProperty(tab_bar_el_, Rml::PropertyId::Top, std::round(tab_top));
                    setPixelProperty(tab_bar_el_, Rml::PropertyId::Height, std::round(std::max(0.0f, layout.size.y - tab_top - 8.0f)));
                    setPixelProperty(tab_bar_el_, Rml::PropertyId::Width, std::round(tab_bar_w));
                }
            }
            if (dims_changed)
                rml_context_->SetDimensions(Rml::Vector2i(w, h));
            updateTabs();

            // Data-model and overflow updates can move the tab strip after input was
            // processed for this frame. Re-resolve hover against the settled layout so
            // a long tab cannot briefly hover at its previous coordinates
            // and then drop the pseudo-class on the following frame.
            if (rml_pointer_inside_) {
                const float local_x = prev_mouse_x_ - layout.pos.x;
                const float local_y = prev_mouse_y_ - layout.pos.y;
                if (local_x >= 0.0f && local_x < layout.size.x &&
                    local_y >= 0.0f && local_y < layout.size.y) {
                    auto* const hover_before = rml_context_->GetHoverElement();
                    rml_context_->ProcessMouseMove(static_cast<int>(local_x),
                                                   static_cast<int>(local_y), 0);
                    if (rml_context_->GetHoverElement() != hover_before)
                        rml_context_->Update();
                }
            }

            last_dp_ratio_ = dp_ratio;
            last_fbo_w_ = w;
            last_fbo_h_ = h;
            last_scene_h_ = layout.scene_h;
            last_splitter_h_ = layout.splitter_h;
            render_needed_ = false;
            input_dirty_ = false;
        }

        rml_manager_->trackContextFrame(rml_context_,
                                        static_cast<int>(layout.pos.x - screen_x),
                                        static_cast<int>(layout.pos.y - screen_y));
        if (!rml_manager_->getVulkanRenderInterface())
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
            // Tooltips may overlap the independently rendered active panel.
            .foreground = tooltip_.hasActiveState(),
            .clip_enabled = true,
            .clip = {
                .x1 = x,
                .y1 = y,
                .x2 = x + static_cast<float>(w),
                .y2 = y + static_cast<float>(h),
            },
        });
    }

} // namespace lfs::vis::gui
