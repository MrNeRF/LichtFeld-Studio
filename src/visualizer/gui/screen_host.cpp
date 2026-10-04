/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/screen_host.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/rmlui/rml_document_utils.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/sdl_rml_key_mapping.hpp"
#include "gui/screen_host_logic.hpp"
#include "internal/resource_paths.hpp"
#include "screen/view3d_space.hpp"

#include <RmlUi/Core.h>
#include <SDL3/SDL_scancode.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    namespace {

        constexpr float kHeaderHeightDp = 28.0f;
        constexpr float kDividerDp = 2.0f;
        constexpr float kMinAreaWidthDp = 72.0f;
        constexpr float kMinContentHeightDp = 24.0f;
        constexpr float kCornerDp = 12.0f;
        constexpr float kDividerSlopDp = 4.0f;
        constexpr float kSplitMinDp = 72.0f;

        std::string px(const float value) { return std::format("{:.0f}px", value); }

        std::string iconPath(const std::string_view name) {
            if (name.empty())
                return {};
            return std::format("../icon/{}.png", name);
        }

        std::string localizedLabel(const screen::EditorType& type) {
            if (!type.label_key.empty()) {
                const std::string translated = LOC(type.label_key.c_str());
                if (!translated.empty() && translated != type.label_key)
                    return translated;
            }
            return type.label;
        }

        void setBox(Rml::Element* el, const screen::Rect& r, const screen::Rect& origin) {
            if (!el)
                return;
            el->SetProperty("left", px(r.x - origin.x));
            el->SetProperty("top", px(r.y - origin.y));
            el->SetProperty("width", px(std::max(0.0f, r.w)));
            el->SetProperty("height", px(std::max(0.0f, r.h)));
        }

        void setShown(Rml::Element* el, const bool shown) {
            if (el)
                el->SetProperty("display", shown ? "block" : "none");
        }

    } // namespace

    ScreenHost::ScreenHost() = default;
    ScreenHost::~ScreenHost() = default;

    void ScreenHost::init(Services services) {
        services_ = std::move(services);
        assert(services_.screens && services_.rml);

        view3d_ = std::make_unique<View3DEditor>(
            [this](const screen::AreaId id, const std::string_view action) {
                if (services_.view_command)
                    services_.view_command(id, action);
                chrome_dirty_ = true;
            });
        assert(services_.scene_manager);
        node_editor_ = std::make_unique<NodeEditor>(*services_.rml, *services_.scene_manager,
                                                    services_.context_menu);
        installPanelEditorTypes(services_.screens->editorTypes());

        chrome_context_ = services_.rml->createContext("screen_chrome", 800, 600);
        overlay_context_ = services_.rml->createContext("screen_overlay", 800, 600);
        if (!chrome_context_ || !overlay_context_) {
            LOG_ERROR("ScreenHost: failed to create RmlUi contexts");
            return;
        }

        auto ctor = chrome_context_->CreateDataModel("screen_chrome");
        assert(ctor);
        if (auto h = ctor.RegisterStruct<ChromeItem>()) {
            h.RegisterMember("id", &ChromeItem::id);
            h.RegisterMember("label", &ChromeItem::label);
            h.RegisterMember("icon", &ChromeItem::icon);
            h.RegisterMember("tooltip", &ChromeItem::tooltip);
            h.RegisterMember("kind", &ChromeItem::kind);
            h.RegisterMember("active", &ChromeItem::active);
            h.RegisterMember("closeable", &ChromeItem::closeable);
        }
        ctor.RegisterArray<std::vector<ChromeItem>>();
        if (auto h = ctor.RegisterStruct<ChromeArea>()) {
            h.RegisterMember("id", &ChromeArea::id);
            h.RegisterMember("left", &ChromeArea::left);
            h.RegisterMember("top", &ChromeArea::top);
            h.RegisterMember("width", &ChromeArea::width);
            h.RegisterMember("height", &ChromeArea::height);
            h.RegisterMember("header_height", &ChromeArea::header_height);
            h.RegisterMember("editor_icon", &ChromeArea::editor_icon);
            h.RegisterMember("editor_label", &ChromeArea::editor_label);
            h.RegisterMember("items", &ChromeArea::items);
            h.RegisterMember("right_items", &ChromeArea::right_items);
            h.RegisterMember("is_view", &ChromeArea::is_view);
            h.RegisterMember("hovered", &ChromeArea::hovered);
            h.RegisterMember("active", &ChromeArea::active);
            h.RegisterMember("maximized", &ChromeArea::maximized);
            h.RegisterMember("view_label", &ChromeArea::view_label);
        }
        ctor.RegisterArray<std::vector<ChromeArea>>();
        ctor.Bind("areas", &chrome_areas_);
        const auto queue = [this](Rml::Event& event, const int area, std::string action) {
            auto* element = event.GetCurrentElement();
            float x = last_mouse_x_;
            float y = last_mouse_y_;
            if (element) {
                const auto offset = element->GetAbsoluteOffset(Rml::BoxArea::Border);
                const auto size = element->GetBox().GetSize(Rml::BoxArea::Border);
                x = work_.x + offset.x;
                y = work_.y + offset.y + size.y;
            }
            pending_actions_.push_back({area, std::move(action), x, y});
        };
        ctor.BindEventCallback("area_action", [queue](Rml::DataModelHandle, Rml::Event& event,
                                                      const Rml::VariantList& args) {
            if (args.size() >= 2)
                queue(event, args[0].Get<int>(), args[1].Get<Rml::String>());
        });
        ctor.BindEventCallback("close_tab", [queue](Rml::DataModelHandle, Rml::Event& event,
                                                    const Rml::VariantList& args) {
            event.StopPropagation();
            if (args.size() >= 2) {
                auto id = args[1].Get<Rml::String>();
                if (id.starts_with("tab:"))
                    id = id.substr(4);
                queue(event, args[0].Get<int>(), "close:" + id);
            }
        });
        chrome_model_ = ctor.GetModelHandle();
        reloadResources();
    }

    void ScreenHost::shutdown() {
        node_editor_.reset();
        chrome_tooltip_.setHover({}, nullptr);
        if (services_.rml) {
            services_.rml->releaseCachedVulkanContext(chrome_cache_);
            services_.rml->releaseCachedVulkanContext(overlay_cache_);
            if (chrome_context_)
                services_.rml->destroyContext("screen_chrome");
            if (overlay_context_)
                services_.rml->destroyContext("screen_overlay");
        }
        chrome_context_ = nullptr;
        overlay_context_ = nullptr;
        chrome_document_ = nullptr;
        overlay_document_ = nullptr;
        chrome_model_ = {};
        chrome_areas_.clear();
    }

    void ScreenHost::reloadResources() {
        chrome_tooltip_.setHover({}, nullptr);
        if (!chrome_context_ || !overlay_context_)
            return;
        if (services_.rml) {
            services_.rml->releaseCachedVulkanContext(chrome_cache_);
            services_.rml->releaseCachedVulkanContext(overlay_cache_);
        }
        if (chrome_document_) {
            chrome_context_->UnloadDocument(chrome_document_);
            chrome_context_->Update();
        }
        if (overlay_document_) {
            overlay_context_->UnloadDocument(overlay_document_);
            overlay_context_->Update();
        }
        chrome_document_ = nullptr;
        overlay_document_ = nullptr;
        chrome_base_rcss_.clear();
        overlay_base_rcss_.clear();
        has_theme_signature_ = false;
        try {
            chrome_document_ =
                rml_documents::loadDocument(chrome_context_, lfs::vis::getAssetPath("rmlui/screen_chrome.rml"));
            overlay_document_ =
                rml_documents::loadDocument(overlay_context_, lfs::vis::getAssetPath("rmlui/screen_overlay.rml"));
        } catch (const std::exception& e) {
            LOG_ERROR("ScreenHost: resource not found: {}", e.what());
        }
        if (chrome_document_)
            chrome_document_->Show();
        if (overlay_document_)
            overlay_document_->Show();
        (void)updateTheme();
        chrome_model_.DirtyVariable("areas");
        chrome_dirty_ = true;
        overlay_dirty_ = true;
    }

    bool ScreenHost::updateTheme() {
        const auto signature = rml_theme::currentThemeSignature();
        if (has_theme_signature_ && signature == theme_signature_)
            return false;
        theme_signature_ = signature;
        has_theme_signature_ = true;
        if (chrome_document_) {
            if (chrome_base_rcss_.empty())
                chrome_base_rcss_ = rml_theme::loadBaseRCSS("rmlui/screen_chrome.rcss");
            rml_theme::applyTheme(chrome_document_, chrome_base_rcss_,
                                  rml_theme::loadBaseRCSS("rmlui/screen_chrome.theme.rcss"));
        }
        if (overlay_document_) {
            if (overlay_base_rcss_.empty())
                overlay_base_rcss_ = rml_theme::loadBaseRCSS("rmlui/screen_overlay.rcss");
            rml_theme::applyTheme(overlay_document_, overlay_base_rcss_,
                                  rml_theme::loadBaseRCSS("rmlui/screen_overlay.theme.rcss"));
        }
        chrome_dirty_ = true;
        overlay_dirty_ = true;
        return true;
    }

    AreaEditor& ScreenHost::editorFor(const std::string_view editor) {
        if (editor == screen::editors::kView3D)
            return *view3d_;
        if (editor == screen::editors::kProperties)
            return properties_;
        if (editor == screen::editors::kScene)
            return scene_;
        if (editor == screen::editors::kConsole)
            return console_;
        if (editor == screen::editors::kNodeEditor)
            return *node_editor_;
        return panel_;
    }

    void ScreenHost::layout(const screen::Rect& work, const float ui_scale) {
        auto& screen = services_.screens->screen();
        const bool scale_changed = ui_scale != ui_scale_;
        ui_scale_ = ui_scale;
        const float header_h = std::round(kHeaderHeightDp * ui_scale);
        const screen::LayoutMetrics metrics{
            .divider = std::max(2.0f, std::round(kDividerDp * ui_scale)),
            .min_width = kMinAreaWidthDp * ui_scale,
            .min_height = header_h + kMinContentHeightDp * ui_scale,
        };
        gestures_.setMetrics({
            .corner_size = kCornerDp * ui_scale,
            .drag_threshold = 8.0f * ui_scale,
            .divider_slop = kDividerSlopDp * ui_scale,
            .min_split_extent = kSplitMinDp * ui_scale,
        });
        auto geometry = screen.solve(work, metrics);
        const bool changed = scale_changed || !(work == work_) || screen.generation() != laid_out_generation_ ||
                             services_.screens->screenEpoch() != laid_out_epoch_ ||
                             geometry.areas.size() != geometry_.areas.size() ||
                             !std::equal(geometry.areas.begin(), geometry.areas.end(), geometry_.areas.begin(),
                                         [](const screen::AreaGeometry& a, const screen::AreaGeometry& b) {
                                             return a.area == b.area && a.rect == b.rect;
                                         });
        work_ = work;
        geometry_ = std::move(geometry);
        laid_out_generation_ = screen.generation();
        laid_out_epoch_ = services_.screens->screenEpoch();
        layout_changed_ = changed;
        if (!changed)
            return;

        frames_.clear();
        frames_.reserve(geometry_.areas.size());
        for (const auto& g : geometry_.areas) {
            const auto* area = screen.area(g.area);
            if (!area)
                continue;
            AreaFrame frame;
            frame.id = g.area;
            frame.rect = g.rect;
            const float h = std::min(header_h, g.rect.h);
            frame.header = {g.rect.x, g.rect.y, g.rect.w, h};
            frame.content = {g.rect.x, g.rect.y + h, g.rect.w, std::max(0.0f, g.rect.h - h)};
            frame.editor = area->editor;
            frames_.push_back(std::move(frame));
        }
        chrome_dirty_ = true;
        overlay_dirty_ = true;
    }

    const AreaFrame* ScreenHost::area(const screen::AreaId id) const {
        const auto it = std::find_if(frames_.begin(), frames_.end(), [id](const AreaFrame& f) { return f.id == id; });
        return it != frames_.end() ? &*it : nullptr;
    }

    screen::Rect ScreenHost::currentAreaRect(const screen::AreaId id) {
        refreshLayout();
        if (const auto* frame = area(id))
            return frame->rect;
        return {};
    }

    screen::AreaId ScreenHost::areaAt(const float x, const float y) const {
        for (const auto& f : frames_) {
            if (f.rect.contains(x, y))
                return f.id;
        }
        return {};
    }

    screen::AreaId ScreenHost::viewAt(const float x, const float y) const {
        for (const auto& f : frames_) {
            if (f.editor == screen::editors::kView3D && f.content.contains(x, y))
                return f.id;
        }
        return {};
    }

    std::optional<screen::Rect> ScreenHost::viewContent(const screen::AreaId id) const {
        const auto* f = area(id);
        if (!f || f->editor != screen::editors::kView3D)
            return std::nullopt;
        return f->content;
    }

    bool ScreenHost::pointerOverHeader(const float x, const float y) const {
        return std::any_of(frames_.begin(), frames_.end(),
                           [x, y](const AreaFrame& f) { return f.header.contains(x, y); });
    }

    bool ScreenHost::blocksPointer(const float x, const float y) const {
        return gestures_.active() || pointerOverHeader(x, y) ||
               screen_host_detail::cornerGestureZone(geometry_, geometry_.maximized.valid(),
                                                     gestures_.metrics().corner_size, x, y);
    }

    bool ScreenHost::blocksPress(const float x, const float y) const {
        if (blocksPointer(x, y))
            return true;
        if (geometry_.maximized.valid())
            return false;
        return screen::cornerAt(geometry_, x, y, gestures_.metrics().corner_size).has_value() ||
               geometry_.dividerAt(x, y, gestures_.metrics().divider_slop) != nullptr;
    }

    bool ScreenHost::cornerGestureAt(const float x, const float y) const {
        return screen_host_detail::cornerGestureZone(geometry_, geometry_.maximized.valid(),
                                                     gestures_.metrics().corner_size, x, y);
    }

    bool ScreenHost::resizeGestureAt(const float x, const float y) const {
        return gestures_.active() || cornerGestureAt(x, y) ||
               (!geometry_.maximized.valid() && geometry_.dividerAt(x, y, gestures_.metrics().divider_slop));
    }

    bool ScreenHost::isEditorVisible(const std::string_view editor) const {
        return std::any_of(frames_.begin(), frames_.end(), [editor](const AreaFrame& f) { return f.editor == editor; });
    }

    void ScreenHost::mutate(const std::function<void(screen::Screen&)>& fn) {
        services_.screens->edit([&](screen::Screen& screen) { fn(screen); });
        chrome_dirty_ = true;
        overlay_dirty_ = true;
        if (services_.screen_changed)
            services_.screen_changed();
    }

    bool ScreenHost::toggleMaximized(const screen::AreaId id) {
        bool done = false;
        mutate([&](screen::Screen& s) { done = s.toggleMaximized(id); });
        return done;
    }

    bool ScreenHost::toggleMaximizedAt(const float x, const float y) {
        const auto id = areaAt(x, y);
        return id.valid() && toggleMaximized(id);
    }

    screen::AreaId ScreenHost::splitArea(const screen::AreaId id, const screen::SplitAxis axis) {
        screen::AreaId added;
        mutate([&](screen::Screen& s) { added = s.split(id, axis, 0.5f); });
        return added;
    }

    bool ScreenHost::closeArea(const screen::AreaId id) {
        bool done = false;
        mutate([&](screen::Screen& s) { done = s.close(id); });
        return done;
    }

    bool ScreenHost::setEditor(const screen::AreaId id, const std::string_view editor) {
        bool done = false;
        mutate([&](screen::Screen& s) { done = s.setEditor(id, editor); });
        return done;
    }

    bool ScreenHost::toggleEditor(const std::string_view editor) {
        bool shown = false;
        mutate([&](screen::Screen& screen) {
            if (screen.findEditor(editor).valid()) {
                screen.closeEditor(editor);
            } else {
                shown = screen.openEditor(editor).valid();
            }
        });
        return shown;
    }

    void ScreenHost::openEditorMenu(const screen::AreaId id, const float x, const float y) {
        if (!services_.context_menu)
            return;
        const auto& screen = services_.screens->screen();
        const auto* area = screen.area(id);
        if (!area)
            return;
        std::vector<ContextMenuItem> items;
        items.push_back({.label = LOC("screen.editor_type"), .is_label = true});
        bool panels_started = false;
        for (const auto& type : services_.screens->editorTypes().list()) {
            const bool builtin = type.id == screen::editors::kView3D || type.id == screen::editors::kScene ||
                                 type.id == screen::editors::kProperties || type.id == screen::editors::kConsole ||
                                 type.id == screen::editors::kNodeEditor;
            items.push_back({.label = localizedLabel(type),
                             .action = "editor:" + type.id,
                             .separator_before = !builtin && !panels_started,
                             .is_active = area->editor == type.id,
                             .icon = iconPath(type.icon)});
            panels_started = panels_started || !builtin;
        }
        services_.context_menu->request(std::move(items), x, y, [this, id](const std::string_view action) {
            if (action.starts_with("editor:"))
                setEditor(id, action.substr(7));
        });
    }

    void ScreenHost::openAreaMenu(const screen::AreaId id, const float x, const float y) {
        if (!services_.context_menu)
            return;
        const auto& screen = services_.screens->screen();
        if (!screen.area(id))
            return;
        std::vector<ContextMenuItem> items;
        items.push_back({.label = LOC("screen.area"), .is_label = true});
        items.push_back({.label = LOC("screen.split_vertically"), .action = "area:split_columns"});
        items.push_back({.label = LOC("screen.split_horizontally"), .action = "area:split_rows"});
        items.push_back({.label = screen.maximized() == id ? LOC("screen.restore_area") : LOC("screen.maximize_area"),
                         .action = "area:maximize",
                         .separator_before = true,
                         .shortcut = "Ctrl Space"});
        if (screen.canClose(id))
            items.push_back({.label = LOC("screen.close_area"), .action = "area:close"});
        services_.context_menu->request(std::move(items), x, y, [this, id, x, y](const std::string_view action) {
            handleAction(static_cast<int>(id.value), std::string(action), x, y);
        });
    }

    void ScreenHost::handleAction(const int area_value, const std::string& action, const float x, const float y) {
        const screen::AreaId id{static_cast<std::uint32_t>(area_value)};
        auto& screen = services_.screens->screen();
        const auto* frame = area(id);
        if (!frame || !screen.area(id))
            return;
        if (action == "editor") {
            openEditorMenu(id, x, y);
        } else if (action == "area:maximize") {
            toggleMaximized(id);
        } else if (action == "area:close") {
            closeArea(id);
        } else if (action == "area:split_columns") {
            splitArea(id, screen::SplitAxis::Columns);
        } else if (action == "area:split_rows") {
            splitArea(id, screen::SplitAxis::Rows);
        } else if (action == "area:quad") {
            if (services_.view_command)
                services_.view_command(id, "area:quad");
        } else if (action.starts_with("menu:")) {
            const AreaFrame snapshot = *frame;
            auto& editor = editorFor(snapshot.editor);
            auto items = editor.menu(snapshot, screen, action.substr(5));
            if (!items.empty() && services_.context_menu) {
                services_.context_menu->request(std::move(items), x, y,
                                                [this, area_value, x, y](const std::string_view chosen) {
                                                    handleAction(area_value, std::string(chosen), x, y);
                                                });
            }
        } else {
            const AreaFrame snapshot = *frame;
            editorFor(snapshot.editor).headerAction(snapshot, screen, action, x, y);
        }
        chrome_dirty_ = true;
    }

    void ScreenHost::applyGesture(const screen::GestureCommand& command) {
        using Kind = screen::GestureCommand::Kind;
        switch (command.kind) {
        case Kind::None: return;
        case Kind::MoveDivider:
            mutate([&](screen::Screen& s) { s.moveDivider(command.divider, command.position); });
            break;
        case Kind::Split:
            mutate([&](screen::Screen& s) { s.split(command.area, command.axis, command.fraction, command.new_first); });
            break;
        case Kind::Join:
            mutate([&](screen::Screen& s) { s.join(command.area, command.other); });
            break;
        case Kind::Swap:
            mutate([&](screen::Screen& s) { s.swap(command.area, command.other); });
            break;
        }
        // Commands change the layout; later events this frame must hit the
        // new geometry.
        layout(work_, ui_scale_);
    }

    void ScreenHost::processInput(const PanelInputState& input, const bool pointer_free) {
        if (services_.rml && chrome_context_ &&
            services_.rml->routeInput(chrome_context_, input, [this, pointer_free](const PanelInputState& event) {
                processInput(event, services_.pointer_available
                                        ? services_.pointer_available(event.mouse_x, event.mouse_y)
                                        : pointer_free);
            }))
            return;
        const float x = input.mouse_x;
        const float y = input.mouse_y;
        const bool moved = x != last_mouse_x_ || y != last_mouse_y_;
        last_mouse_x_ = x;
        last_mouse_y_ = y;
        const auto previous_hover = hovered_area_;
        hovered_area_ = areaAt(x, y);
        if (hovered_area_ != previous_hover)
            chrome_dirty_ = true;

        auto& screen = services_.screens->screen();

        if (gestures_.active()) {
            if (hasKey(input.keys_pressed, SDL_SCANCODE_ESCAPE)) {
                gestures_.cancel();
                overlay_dirty_ = true;
            } else if (input.mouse_released[0] || !input.mouse_down[0]) {
                applyGesture(gestures_.release(geometry_, screen, x, y));
                overlay_dirty_ = true;
            } else if (moved) {
                applyGesture(gestures_.move(geometry_, screen, x, y));
                overlay_dirty_ = true;
            }
        }

        const bool over_corner = screen_host_detail::cornerGestureZone(
            geometry_, geometry_.maximized.valid(), gestures_.metrics().corner_size, x, y);
        const bool over_header = pointer_free && !gestures_.active() && !over_corner && pointerOverHeader(x, y);
        if (chrome_context_) {
            const int mods = sdlModsToRml(input.key_ctrl, input.key_shift, input.key_alt, input.key_super);
            if (over_header) {
                if (moved || !chrome_pointer_inside_) {
                    chrome_context_->ProcessMouseMove(static_cast<int>(x - work_.x), static_cast<int>(y - work_.y),
                                                      mods);
                    chrome_dirty_ = true;
                }
                chrome_pointer_inside_ = true;
                if (input.mouse_clicked[0]) {
                    chrome_context_->ProcessMouseButtonDown(0, mods);
                    chrome_dirty_ = true;
                }
                if (input.mouse_released[0]) {
                    chrome_context_->ProcessMouseButtonUp(0, mods);
                    chrome_dirty_ = true;
                }
                guiFocusState().want_capture_mouse = true;
                if (input.mouse_clicked[1]) {
                    if (const auto id = areaAt(x, y); id.valid())
                        openAreaMenu(id, x, y);
                }
            } else if (chrome_pointer_inside_) {
                chrome_context_->ProcessMouseLeave();
                chrome_pointer_inside_ = false;
                chrome_dirty_ = true;
            }
        }

        if (chrome_context_ && over_header) {
            auto* hover = chrome_context_->GetHoverElement();
            chrome_tooltip_.setHover(hover ? resolveRmlTooltip(hover) : std::string{}, hover);
        } else {
            chrome_tooltip_.setHover({}, nullptr);
        }
        if (services_.rml && chrome_context_) {
            services_.rml->setContextNeedsPassiveMouseMoveFrames(chrome_context_, chrome_tooltip_.hasActiveState());
            services_.rml->setContextTooltipRevealDeadline(chrome_context_, chrome_tooltip_.revealDeadline());
        }

        const auto* press = input.lastPress(0);
        if (!gestures_.active() && press) {
            const bool press_over_corner = screen_host_detail::cornerGestureZone(
                geometry_, geometry_.maximized.valid(), gestures_.metrics().corner_size, press->x, press->y);
            const bool press_over_divider = !geometry_.maximized.valid() &&
                                            geometry_.dividerAt(press->x, press->y,
                                                                gestures_.metrics().divider_slop) != nullptr;
            const bool press_over_header = !press_over_corner && pointerOverHeader(press->x, press->y);
            const bool screen_gesture_press = press_over_corner || press_over_divider;
            if ((pointer_free || screen_gesture_press) &&
                (screen_gesture_press || (!press->gui_owned && !press_over_header))) {
                const bool accepted = gestures_.press(geometry_, press->x, press->y, press->ctrl);
                if (accepted) {
                    overlay_dirty_ = true;
                    if (x != press->x || y != press->y) {
                        applyGesture(gestures_.move(geometry_, screen, x, y));
                        overlay_dirty_ = true;
                    }
                }
            }
        }

        const auto hover = pointer_free && !gestures_.active() && !over_header
                               ? gestures_.hoverPreview(geometry_, x, y)
                               : screen::GesturePreview{};
        if (hover.kind != hover_preview_.kind || !(hover.line == hover_preview_.line) ||
            hover.source != hover_preview_.source || hover.corner != hover_preview_.corner) {
            hover_preview_ = hover;
            overlay_dirty_ = true;
        }
        cursor_ = gestures_.active()
                      ? gestures_.cursor()
                      : (pointer_free && !over_header ? gestures_.hoverCursor(geometry_, x, y)
                                                      : screen::GestureCursor::Default);

        if (input.mouse_clicked[0] || input.mouse_clicked[1] || input.mouse_clicked[2])
            live_capture_area_ = pointer_free ? hovered_area_ : screen::AreaId{};
        if (!input.mouse_down[0] && !input.mouse_down[1] && !input.mouse_down[2])
            live_capture_area_ = {};

        auto actions = std::move(pending_actions_);
        pending_actions_.clear();
        for (const auto& a : actions)
            handleAction(a.area, a.action, a.x, a.y);
    }

    void ScreenHost::setExternallyManaged(std::vector<std::string> editors) {
        externally_managed_ = std::move(editors);
    }

    void ScreenHost::syncPanelEditors() {
        auto& reg = PanelRegistry::instance();
        auto& screen = services_.screens->screen();
        const auto& types = services_.screens->editorTypes();

        // Areas showing a panel that left the editor spaces (it floats now,
        // or was unregistered) close.
        for (const auto& f : frames_) {
            if (!screen_host_detail::shouldCloseMissingPanelEditor(types.contains(f.editor),
                                                                   panel_editor_ids_.contains(f.editor)))
                continue;
            const auto id = f.id;
            mutate([&](screen::Screen& s) { s.close(id); });
        }

        std::unordered_set<std::string> shown_now;
        for (const auto& type : types.list()) {
            const auto details = reg.get_panel(type.id);
            if (!details || !isPanelEditorSpace(details->space))
                continue;
            panel_editor_ids_.insert(type.id);
            if (std::find(externally_managed_.begin(), externally_managed_.end(), type.id) !=
                externally_managed_.end())
                continue;
            const bool enabled = details->enabled;
            const bool shown = screen.findEditor(type.id).valid();
            const auto seen = panel_enabled_seen_.find(type.id);
            const bool enabled_changed = seen == panel_enabled_seen_.end() ? enabled : seen->second != enabled;
            const bool was_shown = panel_editors_shown_.contains(type.id);
            if (enabled_changed) {
                if (enabled && !shown)
                    mutate([&](screen::Screen& s) { s.openEditor(type.id); });
                else if (!enabled && shown)
                    mutate([&](screen::Screen& s) { s.closeEditor(type.id); });
            } else if (shown != was_shown) {
                reg.set_panel_enabled(type.id, shown);
            }
            panel_enabled_seen_[type.id] = reg.is_panel_enabled(type.id);
            if (screen.findEditor(type.id).valid())
                shown_now.insert(type.id);
        }
        panel_editors_shown_ = std::move(shown_now);
    }

    void ScreenHost::draw(const UIContext& ui, const PanelDrawContext& draw, const PanelInputState& input,
                          const bool force_live, const PanelAnimationDemand demand) {
        syncPanelEditors();
        if (layout_changed_ || services_.screens->screen().generation() != laid_out_generation_ ||
            services_.screens->screenEpoch() != laid_out_epoch_)
            layout(work_, ui_scale_);

        const bool keyboard_activity = !input.keys_pressed.empty() || !input.input_events.empty();

        rebuildChrome();
        if (chrome_context_ && chrome_document_ && services_.rml && !work_.empty()) {
            (void)updateTheme();
            const int w = static_cast<int>(work_.w);
            const int h = static_cast<int>(work_.h);
            const bool dims_changed = chrome_cache_.width != w || chrome_cache_.height != h;
            const bool refresh = chrome_dirty_ || dims_changed || chrome_cache_.texture == 0 ||
                                 chrome_tooltip_.revealDue();
            if (refresh) {
                chrome_context_->SetDimensions(Rml::Vector2i(w, h));
                chrome_context_->Update();
                chrome_dirty_ = false;
            }
            if (chrome_tooltip_.apply(chrome_document_->GetElementById("screen-chrome"),
                                      static_cast<int>(input.mouse_x - work_.x),
                                      static_cast<int>(input.mouse_y - work_.y), w, h)) {
                chrome_context_->Update();
                chrome_dirty_ = false;
            }
            services_.rml->setContextNeedsPassiveMouseMoveFrames(chrome_context_, chrome_tooltip_.hasActiveState());
            services_.rml->setContextTooltipRevealDeadline(chrome_context_, chrome_tooltip_.revealDeadline());
            services_.rml->trackContextFrame(chrome_context_, static_cast<int>(work_.x), static_cast<int>(work_.y));
            services_.rml->queueCachedVulkanContext({
                .context = chrome_context_,
                .cache = &chrome_cache_,
                .cache_width = w,
                .cache_height = h,
                .offset_x = work_.x,
                .offset_y = work_.y,
                .draw_width = static_cast<float>(w),
                .draw_height = static_cast<float>(h),
                .refresh = refresh,
                .foreground = false,
                .clip_enabled = true,
                .clip = {.x1 = work_.x, .y1 = work_.y, .x2 = work_.x + w, .y2 = work_.y + h},
            });
        }

        const auto frames = frames_;
        for (const auto& frame : frames) {
            if (frame.editor == screen::editors::kView3D || frame.content.empty())
                continue;
            auto& editor = editorFor(frame.editor);
            bool space_demand = false;
            if (const auto space = editor.panelSpace()) {
                switch (*space) {
                case PanelSpace::MainPanelTab: space_demand = demand.main_panel_tab; break;
                case PanelSpace::SceneHeader: space_demand = demand.scene_header; break;
                default: space_demand = demand.bottom_editor || demand.left_editor; break;
                }
            }
            const bool pointer_here = frame.rect.contains(input.mouse_x, input.mouse_y);
            const bool live = force_live || layout_changed_ || pointer_here || live_capture_area_ == frame.id ||
                              keyboard_activity || space_demand || editor.needsAnimationFrame();
            // Panels see the pointer only over their own area (or while a
            // press that started there is held).
            const bool owns_pointer = pointer_here || live_capture_area_ == frame.id;
            PanelInputState area_input = input;
            if (!owns_pointer || gestures_.active()) {
                area_input.mouse_x = -1.0e9f;
                area_input.mouse_y = -1.0e9f;
                for (auto& v : area_input.mouse_clicked)
                    v = false;
                for (auto& v : area_input.mouse_released)
                    v = false;
                for (auto& v : area_input.mouse_down)
                    v = false;
                area_input.mouse_wheel = 0.0f;
                area_input.mouse_wheel_x = 0.0f;
                area_input.pinch_scale = 1.0f;
                area_input.mouse_button_events.clear();
            }
            editor.draw({.area = frame, .ui = ui, .draw = draw, .input = area_input, .live = live});
        }
        layout_changed_ = false;
    }

    void ScreenHost::rebuildChrome() {
        if (!chrome_model_)
            return;
        const auto& screen = services_.screens->screen();
        const auto& types = services_.screens->editorTypes();
        const bool several_views = screen.views().size() > 1;
        std::vector<ChromeArea> areas;
        areas.reserve(frames_.size());
        for (const auto& frame : frames_) {
            ChromeArea a;
            a.id = static_cast<int>(frame.id.value);
            a.left = px(frame.rect.x - work_.x);
            a.top = px(frame.rect.y - work_.y);
            a.width = px(frame.rect.w);
            a.height = px(frame.rect.h);
            a.header_height = px(frame.header.h);
            a.is_view = frame.editor == screen::editors::kView3D;
            a.hovered = hovered_area_ == frame.id;
            a.active = a.is_view && several_views && screen.activeView() == frame.id;
            a.maximized = screen.maximized() == frame.id;
            if (a.is_view) {
                if (const auto* view = screen.view(frame.id)) {
                    const char* direction_key = "view3d.user";
                    switch (screen::alignedViewAxis(view->camera.camera.R)) {
                    case screen::ViewAxis::Top: direction_key = "view3d.top"; break;
                    case screen::ViewAxis::Bottom: direction_key = "view3d.bottom"; break;
                    case screen::ViewAxis::Front: direction_key = "view3d.front"; break;
                    case screen::ViewAxis::Back: direction_key = "view3d.back"; break;
                    case screen::ViewAxis::Right: direction_key = "view3d.right"; break;
                    case screen::ViewAxis::Left: direction_key = "view3d.left"; break;
                    case screen::ViewAxis::None: break;
                    }
                    const char* projection_key = view->settings.equirectangular ? "view3d.panorama"
                                                 : view->settings.orthographic  ? "view3d.orthographic"
                                                                                : "view3d.perspective";
                    a.view_label = std::string(LOC(direction_key)) + " " + LOC(projection_key);
                }
            }
            if (const auto type = types.find(frame.editor)) {
                a.editor_icon = iconPath(type->icon.empty() ? "layout-rows" : type->icon);
                a.editor_label = localizedLabel(*type);
            } else {
                a.editor_icon = iconPath("puzzle");
                a.editor_label = frame.editor;
            }
            std::vector<HeaderItem> items;
            editorFor(frame.editor).header(frame, screen, items);
            bool right = false;
            for (const auto& item : items) {
                if (item.kind == HeaderItem::Kind::Spacer) {
                    right = true;
                    continue;
                }
                ChromeItem c;
                c.id = item.kind == HeaderItem::Kind::Menu ? "menu:" + item.id : item.id;
                c.label = item.label;
                c.icon = iconPath(item.icon);
                c.tooltip = item.tooltip;
                c.kind = static_cast<int>(item.kind);
                c.active = item.active;
                c.closeable = item.closeable;
                (right ? a.right_items : a.items).push_back(std::move(c));
            }
            a.right_items.push_back({.id = "area:maximize",
                                     .icon = iconPath(a.maximized ? "arrows-minimize" : "arrows-maximize"),
                                     .tooltip = std::format("{} (Ctrl Space)", a.maximized ? LOC("screen.restore_area")
                                                                                           : LOC("screen.maximize_area")),
                                     .kind = static_cast<int>(HeaderItem::Kind::Button),
                                     .active = a.maximized});
            areas.push_back(std::move(a));
        }
        if (areas != chrome_areas_) {
            chrome_areas_ = std::move(areas);
            chrome_model_.DirtyVariable("areas");
            chrome_dirty_ = true;
        }
    }

    void ScreenHost::queueOverlay() {
        if (!overlay_context_ || !overlay_document_ || !services_.rml || work_.empty())
            return;
        updateOverlay();
        if (!overlay_visible_)
            return;
        const int w = static_cast<int>(work_.w);
        const int h = static_cast<int>(work_.h);
        const bool dims_changed = overlay_cache_.width != w || overlay_cache_.height != h;
        const bool refresh = overlay_dirty_ || dims_changed || overlay_cache_.texture == 0;
        if (refresh) {
            overlay_context_->SetDimensions(Rml::Vector2i(w, h));
            overlay_context_->Update();
            overlay_dirty_ = false;
        }
        services_.rml->queueCachedVulkanContext({
            .context = overlay_context_,
            .cache = &overlay_cache_,
            .cache_width = w,
            .cache_height = h,
            .offset_x = work_.x,
            .offset_y = work_.y,
            .draw_width = static_cast<float>(w),
            .draw_height = static_cast<float>(h),
            .refresh = refresh,
            .foreground = true,
            .clip_enabled = true,
            .clip = {.x1 = work_.x, .y1 = work_.y, .x2 = work_.x + w, .y2 = work_.y + h},
        });
    }

    void ScreenHost::updateOverlay() {
        if (!overlay_dirty_)
            return;
        auto* doc = overlay_document_;
        const auto& screen = services_.screens->screen();
        auto* outline = doc->GetElementById("active-outline");
        auto* divider = doc->GetElementById("divider-highlight");
        auto* corner = doc->GetElementById("corner-hint");
        auto* split_new = doc->GetElementById("split-new");
        auto* split_line = doc->GetElementById("split-line");
        auto* join_target = doc->GetElementById("join-target");
        auto* join_arrow = doc->GetElementById("join-arrow");
        auto* swap_source = doc->GetElementById("swap-source");
        auto* swap_target = doc->GetElementById("swap-target");
        for (auto* el : {outline, divider, corner, split_new, split_line, join_target, swap_source, swap_target})
            setShown(el, false);

        bool any = false;
        if (screen.views().size() > 1 && !screen.maximized().valid()) {
            if (const auto rect = viewContent(screen.activeView())) {
                setBox(outline, *rect, work_);
                setShown(outline, true);
                any = true;
            }
        }

        using Kind = screen::GesturePreview::Kind;
        const auto& preview = gestures_.active() ? gestures_.preview() : hover_preview_;
        switch (preview.kind) {
        case Kind::None: break;
        case Kind::Divider:
            setBox(divider, preview.line, work_);
            setShown(divider, true);
            any = true;
            break;
        case Kind::Corner: {
            const float s = gestures_.metrics().corner_size * 1.5f;
            const auto& r = preview.first;
            const bool left = preview.corner == screen::Corner::TopLeft || preview.corner == screen::Corner::BottomLeft;
            const bool top = preview.corner == screen::Corner::TopLeft || preview.corner == screen::Corner::TopRight;
            const screen::Rect box{left ? r.x : r.right() - s, top ? r.y : r.bottom() - s, s, s};
            setBox(corner, box, work_);
            if (corner) {
                corner->SetClass("top-left", left && top);
                corner->SetClass("top-right", !left && top);
                corner->SetClass("bottom-left", left && !top);
                corner->SetClass("bottom-right", !left && !top);
            }
            setShown(corner, true);
            any = true;
            break;
        }
        case Kind::Split:
            setBox(split_new, preview.second, work_);
            setBox(split_line, preview.line, work_);
            if (split_new)
                split_new->SetClass("denied", !preview.allowed);
            setShown(split_new, true);
            setShown(split_line, preview.allowed);
            any = true;
            break;
        case Kind::Join:
            if (!preview.second.empty()) {
                setBox(join_target, preview.second, work_);
                if (join_target)
                    join_target->SetClass("denied", !preview.allowed);
                if (join_arrow) {
                    join_arrow->SetClass("left", preview.direction == screen::Side::Left);
                    join_arrow->SetClass("right", preview.direction == screen::Side::Right);
                    join_arrow->SetClass("up", preview.direction == screen::Side::Top);
                    join_arrow->SetClass("down", preview.direction == screen::Side::Bottom);
                }
                setShown(join_target, true);
                any = true;
            }
            break;
        case Kind::Swap:
            setBox(swap_source, preview.first, work_);
            setShown(swap_source, true);
            if (!preview.second.empty()) {
                setBox(swap_target, preview.second, work_);
                if (swap_target)
                    swap_target->SetClass("denied", !preview.allowed);
                setShown(swap_target, true);
            }
            any = true;
            break;
        }
        overlay_visible_ = any;
        screen_host_detail::clearDirtyWhenOverlayHidden(overlay_visible_, overlay_dirty_);
    }

    bool ScreenHost::needsAnimationFrame() const {
        return chrome_dirty_ || overlay_dirty_ || gestures_.active() || !pending_actions_.empty() ||
               chrome_tooltip_.needsFrame();
    }

    std::string ScreenHost::animationDemandDescription() const {
        if (!needsAnimationFrame())
            return {};
        return std::format("screen(chrome_dirty={},overlay_dirty={},gesture={},actions={})", chrome_dirty_,
                           overlay_dirty_, gestures_.active(), pending_actions_.size());
    }

} // namespace lfs::vis::gui
