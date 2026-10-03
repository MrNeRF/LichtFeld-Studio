/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/area_editors.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/rmlui/elements/node_canvas_element.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "input/frame_input_buffer.hpp"
#include "python/python_runtime.hpp"
#include "scene/scene_manager.hpp"
#include "screen/view3d_space.hpp"
#include "visualizer/nodes/modifier_manager.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <algorithm>
#include <chrono>
#include <format>
#include <utility>

namespace lfs::vis::gui {

    namespace {

        constexpr float kPreloadMaxHeight = 100000.0f;
        constexpr float kWheelStep = 30.0f;

        PanelInputState withoutPointer(const PanelInputState& input) {
            PanelInputState masked = input;
            masked.mouse_x = -1.0e9f;
            masked.mouse_y = -1.0e9f;
            for (auto& v : masked.mouse_clicked)
                v = false;
            for (auto& v : masked.mouse_released)
                v = false;
            for (auto& v : masked.mouse_down)
                v = false;
            masked.mouse_wheel = 0.0f;
            masked.mouse_wheel_x = 0.0f;
            masked.pinch_scale = 1.0f;
            masked.mouse_button_events.clear();
            return masked;
        }

        bool contains(const screen::Rect& r, const float x, const float y) { return r.contains(x, y); }

        // Panels ignore the pointer while a floating panel covers it.
        PanelInputState panelInput(const PanelInputState& input) {
            if (PanelRegistry::instance().isPositionOverFloatingPanel(input.mouse_x, input.mouse_y))
                return withoutPointer(input);
            return input;
        }

    } // namespace

    bool isPanelEditorSpace(const PanelSpace space) {
        return space == PanelSpace::BottomArea || space == PanelSpace::LeftArea;
    }

    // ---- 3D viewport -------------------------------------------------------

    namespace {

        std::string displayMode(const ViewSettings& s) {
            if (s.point_cloud_mode)
                return "points";
            if (s.show_rings)
                return "rings";
            if (s.show_center_markers)
                return "centers";
            return "splats";
        }

    } // namespace

    void View3DEditor::header(const AreaFrame& area, const screen::Screen& screen,
                              std::vector<HeaderItem>& items) const {
        const auto* view = screen.view(area.id);
        if (!view)
            return;
        const auto& s = view->settings;
        const std::string mode = displayMode(s);
        const float scale = std::max(area.header.h / 28.0f, 1.0f);
        const float width_dp = area.header.w / scale;
        const bool compact_display = width_dp < 560.0f;
        const bool compact_controls = width_dp < 420.0f;
        const bool icon_only_view = width_dp < 280.0f;
        items.push_back({.kind = HeaderItem::Kind::Menu,
                         .id = "view",
                         .label = icon_only_view ? "" : LOC("view3d.view"),
                         .icon = icon_only_view ? "camera-orbit" : "",
                         .tooltip = LOC("view3d.view")});
        const auto display_icon = mode == "points"    ? "dots-diagonal"
                                  : mode == "rings"   ? "ring"
                                  : mode == "centers" ? "circle-dot"
                                                      : "blob";
        if (compact_display) {
            items.push_back({.kind = HeaderItem::Kind::Menu,
                             .id = "display",
                             .icon = display_icon,
                             .tooltip = LOC(mode == "points"    ? "view3d.point_cloud"
                                            : mode == "rings"   ? "view3d.rings"
                                            : mode == "centers" ? "view3d.centers"
                                                                : "view3d.splats")});
        } else {
            items.push_back({.kind = HeaderItem::Kind::Toggle,
                             .id = "display:splats",
                             .icon = "blob",
                             .tooltip = LOC("view3d.splats"),
                             .active = mode == "splats"});
            items.push_back({.kind = HeaderItem::Kind::Toggle,
                             .id = "display:points",
                             .icon = "dots-diagonal",
                             .tooltip = LOC("view3d.point_cloud"),
                             .active = mode == "points"});
            items.push_back({.kind = HeaderItem::Kind::Toggle,
                             .id = "display:rings",
                             .icon = "ring",
                             .tooltip = LOC("view3d.rings"),
                             .active = mode == "rings"});
            items.push_back({.kind = HeaderItem::Kind::Toggle,
                             .id = "display:centers",
                             .icon = "circle-dot",
                             .tooltip = LOC("view3d.centers"),
                             .active = mode == "centers"});
        }
        if (!compact_controls) {
            items.push_back({.kind = HeaderItem::Kind::Toggle, .id = "depth", .icon = "depth-map", .tooltip = LOC("view3d.depth_map"), .active = s.depth_view});
            items.push_back({.kind = HeaderItem::Kind::Menu, .id = "overlays", .icon = "overlays", .tooltip = LOC("view3d.overlays"), .active = s.show_grid || s.show_coord_axes || s.show_pivot || s.show_camera_frustums});
            const auto tooltip = std::format("{} (Numpad 5)",
                                             LOC(s.orthographic ? "view3d.orthographic" : "view3d.perspective"));
            items.push_back({.kind = HeaderItem::Kind::Toggle,
                             .id = "projection",
                             .icon = s.orthographic ? "orthographic" : "perspective",
                             .tooltip = tooltip,
                             .active = s.orthographic});
        }
        items.push_back({.kind = HeaderItem::Kind::Toggle,
                         .id = "toggle_node_editor",
                         .icon = "layout-grid",
                         .tooltip = LOC("node_editor.toggle"),
                         .active = screen.findEditor(screen::editors::kNodeEditor).valid()});
    }

    std::vector<ContextMenuItem> View3DEditor::menu(const AreaFrame& area, const screen::Screen& screen,
                                                    const std::string_view menu_id) const {
        std::vector<ContextMenuItem> items;
        const auto* view = screen.view(area.id);
        if (!view)
            return items;
        const auto& s = view->settings;
        if (menu_id == "view") {
            items.push_back({.label = s.orthographic ? LOC("view3d.perspective") : LOC("view3d.orthographic"),
                             .action = "projection",
                             .shortcut = "Numpad 5"});
            items.push_back({.label = LOC("view3d.frame_selected"), .action = "frame_selected", .shortcut = "F"});
            items.push_back({.label = LOC("view3d.frame_all"), .action = "frame_all", .shortcut = "Home"});
            items.push_back({.label = LOC("view3d.viewpoint"), .separator_before = true, .is_label = true});
            const auto axis = screen::alignedViewAxis(view->camera.camera.R);
            const auto viewpoint = [&](const char* label_key, const char* action, screen::ViewAxis a,
                                       const char* shortcut) {
                items.push_back({.label = LOC(label_key),
                                 .action = action,
                                 .is_submenu_item = true,
                                 .is_active = axis == a,
                                 .shortcut = shortcut});
            };
            viewpoint("view3d.top", "axis:top", screen::ViewAxis::Top, "Numpad 7");
            viewpoint("view3d.bottom", "axis:bottom", screen::ViewAxis::Bottom, "Ctrl Numpad 7");
            viewpoint("view3d.front", "axis:front", screen::ViewAxis::Front, "Numpad 1");
            viewpoint("view3d.back", "axis:back", screen::ViewAxis::Back, "Ctrl Numpad 1");
            viewpoint("view3d.right", "axis:right", screen::ViewAxis::Right, "Numpad 3");
            viewpoint("view3d.left", "axis:left", screen::ViewAxis::Left, "Ctrl Numpad 3");
            items.push_back({.label = LOC("screen.area"), .separator_before = true, .is_label = true});
            items.push_back({.label = LOC("screen.split_vertically"), .action = "area:split_columns", .is_submenu_item = true});
            items.push_back({.label = LOC("screen.split_horizontally"), .action = "area:split_rows", .is_submenu_item = true});
            items.push_back({.label = LOC("screen.four_views"), .action = "area:quad", .is_submenu_item = true, .shortcut = "Ctrl Alt Q"});
            items.push_back({.label = screen.maximized() == area.id ? LOC("screen.restore_area") : LOC("screen.maximize_area"),
                             .action = "area:maximize",
                             .is_submenu_item = true,
                             .shortcut = "Ctrl Space"});
            if (screen.canClose(area.id))
                items.push_back({.label = LOC("screen.close_area"), .action = "area:close", .is_submenu_item = true});
        } else if (menu_id == "display") {
            const auto mode_item = [&](const char* key, const char* action, const char* icon, const bool active) {
                const auto icon_path = std::format("../icon/{}.png", icon);
                items.push_back({.label = LOC(key),
                                 .action = action,
                                 .is_active = active,
                                 .icon = icon_path});
            };
            mode_item("view3d.splats", "display:splats", "blob", displayMode(s) == "splats");
            mode_item("view3d.point_cloud", "display:points", "dots-diagonal", displayMode(s) == "points");
            mode_item("view3d.rings", "display:rings", "ring", displayMode(s) == "rings");
            mode_item("view3d.centers", "display:centers", "circle-dot", displayMode(s) == "centers");
            if (area.header.w / std::max(area.header.h / 28.0f, 1.0f) < 420.0f) {
                items.push_back({.label = LOC("view3d.depth_map"), .action = "depth", .separator_before = true, .is_active = s.depth_view});
                items.push_back({.label = LOC("view3d.overlays"),
                                 .action = "menu:overlays",
                                 .is_submenu_item = true});
                items.push_back({.label = LOC(s.orthographic ? "view3d.perspective" : "view3d.orthographic"),
                                 .action = "projection",
                                 .is_active = s.orthographic,
                                 .shortcut = "Numpad 5"});
            }
        } else if (menu_id == "overlays") {
            items.push_back({.label = LOC("view3d.viewport_overlays"), .is_label = true});
            items.push_back({.label = LOC("view3d.grid"), .action = "overlay:grid", .is_active = s.show_grid});
            const char* planes[] = {"view3d.grid_plane_yz", "view3d.grid_plane_xz", "view3d.grid_plane_xy"};
            for (int plane = 0; plane < 3; ++plane) {
                items.push_back({.label = LOC(planes[plane]),
                                 .action = "overlay:grid_plane:" + std::to_string(plane),
                                 .is_submenu_item = true,
                                 .is_active = s.show_grid && s.grid_plane == plane});
            }
            items.push_back({.label = LOC("view3d.coordinate_axes"),
                             .action = "overlay:axes",
                             .separator_before = true,
                             .is_active = s.show_coord_axes});
            items.push_back({.label = LOC("view3d.pivot_point"), .action = "overlay:pivot", .is_active = s.show_pivot});
            items.push_back({.label = LOC("view3d.camera_frustums"),
                             .action = "overlay:frustums",
                             .is_active = s.show_camera_frustums});
        }
        return items;
    }

    void View3DEditor::headerAction(const AreaFrame& area, screen::Screen& screen, const std::string_view action,
                                    float, float) {
        if (action == "toggle_node_editor") {
            if (screen.findEditor(screen::editors::kNodeEditor).valid())
                screen.closeEditor(screen::editors::kNodeEditor);
            else
                screen.openEditor(screen::editors::kNodeEditor);
            return;
        }
        if (command_)
            command_(area.id, action);
    }

    // ---- Properties ------------------------------------------------------

    void PropertiesEditor::syncTabs() const {
        tabs_ = PanelRegistry::instance().get_panels_for_space(PanelSpace::MainPanelTab);
    }

    void PropertiesEditor::setActiveTab(std::string id) {
        if (id == active_tab_)
            return;
        active_tab_ = std::move(id);
        scroll_ = 0.0f;
    }

    bool PropertiesEditor::focusTab(const std::string_view id_or_label) {
        syncTabs();
        const auto it = std::find_if(tabs_.begin(), tabs_.end(), [&](const PanelSummary& tab) {
            return tab.id == id_or_label || tab.label == id_or_label;
        });
        if (it == tabs_.end())
            return false;
        setActiveTab(it->id);
        return true;
    }

    void PropertiesEditor::header(const AreaFrame&, const screen::Screen&, std::vector<HeaderItem>& items) const {
        syncTabs();
        const bool active_valid = std::any_of(tabs_.begin(), tabs_.end(),
                                              [&](const PanelSummary& tab) { return tab.id == active_tab_; });
        const std::string active = active_valid || tabs_.empty() ? active_tab_ : tabs_.front().id;
        for (const auto& tab : tabs_) {
            items.push_back({.kind = HeaderItem::Kind::Tab,
                             .id = "tab:" + tab.id,
                             .label = tab.label,
                             .tooltip = tab.label,
                             .active = tab.id == active,
                             .closeable = tab.tab_closeable});
        }
    }

    void PropertiesEditor::headerAction(const AreaFrame&, screen::Screen&, const std::string_view action, float,
                                        float) {
        if (action.starts_with("tab:")) {
            setActiveTab(std::string(action.substr(4)));
        } else if (action.starts_with("close:")) {
            const std::string id(action.substr(6));
            PanelRegistry::instance().set_panel_enabled(id, false);
            if (id == active_tab_)
                active_tab_.clear();
        }
    }

    void PropertiesEditor::draw(const AreaDrawContext& ctx) {
        auto& reg = PanelRegistry::instance();
        syncTabs();
        const bool active_valid = std::any_of(tabs_.begin(), tabs_.end(),
                                              [&](const PanelSummary& tab) { return tab.id == active_tab_; });
        if (!active_valid)
            setActiveTab(tabs_.empty() ? std::string{} : tabs_.front().id);
        if (active_tab_.empty()) {
            content_height_ = 0.0f;
            scroll_ = 0.0f;
            return;
        }

        const auto& c = ctx.area.content;
        const float pad = 8.0f * lfs::python::get_shared_dpi_scale();
        const float x = c.x + pad;
        const float w = std::max(0.0f, c.w - 2.0f * pad);
        const float top = c.y + pad * 0.5f;
        const float h = std::max(0.0f, c.bottom() - top);
        const PanelInputState input = panelInput(ctx.input);
        const float clip_min = top;
        const float clip_max = top + h;

        float scroll_limit = std::max(0.0f, content_height_ - h);
        if (ctx.live) {
            const float main_h = reg.render_panels({.target = PanelRenderTarget::for_panel(active_tab_),
                                                    .mode = PanelRenderMode::DirectPreload,
                                                    .width = w,
                                                    .height = kPreloadMaxHeight,
                                                    .clip_y_min = clip_min,
                                                    .clip_y_max = clip_max,
                                                    .input = &input},
                                                   ctx.draw);
            const float child_h = reg.render_panels({.target = PanelRenderTarget::for_children(active_tab_),
                                                     .mode = PanelRenderMode::DirectPreload,
                                                     .width = w,
                                                     .height = kPreloadMaxHeight,
                                                     .clip_y_min = clip_min,
                                                     .clip_y_max = clip_max,
                                                     .input = &input},
                                                    ctx.draw);
            scroll_limit = std::max(0.0f, main_h + child_h - h);
        }
        if (contains(c, input.mouse_x, input.mouse_y) && input.mouse_wheel != 0.0f)
            scroll_ -= input.mouse_wheel * kWheelStep;
        scroll_ = std::clamp(scroll_, 0.0f, scroll_limit);

        const auto mode = ctx.live ? PanelRenderMode::Direct : PanelRenderMode::DirectCached;
        const float y = top - scroll_;
        const float main_h = reg.render_panels({.target = PanelRenderTarget::for_panel(active_tab_),
                                                .mode = mode,
                                                .x = x,
                                                .y = y,
                                                .width = w,
                                                .height = kPreloadMaxHeight,
                                                .clip_y_min = clip_min,
                                                .clip_y_max = clip_max,
                                                .input = &input},
                                               ctx.draw);
        const float child_h = reg.render_panels({.target = PanelRenderTarget::for_children(active_tab_),
                                                 .mode = mode,
                                                 .x = x,
                                                 .y = y + main_h,
                                                 .width = w,
                                                 .height = kPreloadMaxHeight,
                                                 .clip_y_min = clip_min,
                                                 .clip_y_max = clip_max,
                                                 .input = &input},
                                                ctx.draw);
        content_height_ = main_h + child_h;
        scroll_ = std::clamp(scroll_, 0.0f, std::max(0.0f, content_height_ - h));
    }

    // ---- Scene -----------------------------------------------------------

    void ScenePanelEditor::draw(const AreaDrawContext& ctx) {
        auto& reg = PanelRegistry::instance();
        const auto& c = ctx.area.content;
        const float pad = 8.0f * lfs::python::get_shared_dpi_scale();
        const float x = c.x + pad;
        const float y = c.y + pad * 0.5f;
        const float w = std::max(0.0f, c.w - 2.0f * pad);
        const float h = std::max(0.0f, c.bottom() - y - pad);
        const PanelInputState input = panelInput(ctx.input);
        if (ctx.live) {
            reg.render_panels({.target = PanelRenderTarget::for_space(PanelSpace::SceneHeader),
                               .mode = PanelRenderMode::DirectPreload,
                               .width = w,
                               .height = h,
                               .input = &input},
                              ctx.draw);
        }
        reg.render_panels({.target = PanelRenderTarget::for_space(PanelSpace::SceneHeader),
                           .mode = ctx.live ? PanelRenderMode::Direct : PanelRenderMode::DirectCached,
                           .x = x,
                           .y = y,
                           .width = w,
                           .height = h,
                           .input = &input},
                          ctx.draw);
    }

    // ---- Python console ----------------------------------------------------

    void ConsoleEditor::draw(const AreaDrawContext& ctx) {
        const auto& c = ctx.area.content;
        const PanelInputState input = panelInput(ctx.input);
        panels::DrawDockedPythonConsole(ctx.ui, c.x, c.y, c.w, c.h, &input);
    }

    // ---- Node editor -------------------------------------------------------

    NodeEditor::NodeEditor(RmlUIManager& rml, SceneManager& scene_manager,
                           GlobalContextMenu* context_menu)
        : scene_manager_(&scene_manager),
          context_menu_(context_menu),
          host_(std::make_unique<RmlPanelHost>(&rml, "node_editor", "rmlui/node_editor.rml")) {
        host_->setKeyboardHandler([this](const FrameInputEvent& event) {
            if (!canvas_ || event.kind != FrameInputEventKind::KeyDown || event.repeat)
                return false;
            return canvas_->handleKey(event.scancode, (event.modifiers & SDL_KMOD_SHIFT) != 0,
                                      (event.modifiers & SDL_KMOD_CTRL) != 0 ||
                                          (event.modifiers & SDL_KMOD_GUI) != 0,
                                      (event.modifiers & SDL_KMOD_ALT) != 0);
        });
    }

    NodeEditor::~NodeEditor() = default;

    void NodeEditor::bindCanvas() {
        if (!host_->ensureDocumentLoaded())
            return;
        auto* next = dynamic_cast<NodeCanvasElement*>(host_->getDocument()->GetElementById("node-editor-canvas"));
        if (canvas_ == next)
            return;
        canvas_ = next;
        if (canvas_) {
            canvas_->setContext(scene_manager_, context_menu_);
            canvas_->setSidebarVisible(sidebar_visible_);
            canvas_->setPreviewSelection(preview_selection_);
        }
    }

    void NodeEditor::header(const AreaFrame&, const screen::Screen&,
                            std::vector<HeaderItem>& items) const {
        items.push_back({.kind = HeaderItem::Kind::Label,
                         .id = "title",
                         .label = LOC("editor.node_editor")});
        items.push_back({.kind = HeaderItem::Kind::Button, .id = "add", .label = LOC("node_editor.add")});
        items.push_back({.kind = HeaderItem::Kind::Toggle, .id = "modifiers-visible", .label = LOC(canvas_ && canvas_->modifiersVisible() ? "node_editor.modifiers_on" : "node_editor.modifiers_off"), .active = canvas_ && canvas_->modifiersVisible()});
        items.push_back({.kind = HeaderItem::Kind::Button, .id = "frame", .label = LOC("node_editor.frame")});
        items.push_back({.kind = HeaderItem::Kind::Button, .id = "arrange", .label = LOC("node_editor.arrange")});
        items.push_back({.kind = HeaderItem::Kind::Button, .id = "open", .label = LOC("node_editor.open")});
        items.push_back({.kind = HeaderItem::Kind::Button, .id = "save", .label = LOC("node_editor.save")});
        items.push_back({.kind = HeaderItem::Kind::Spacer});
        if (canvas_)
            items.push_back({.kind = HeaderItem::Kind::Label, .id = "status", .label = canvas_->statusText()});
        items.push_back({.kind = HeaderItem::Kind::Toggle,
                         .id = "preview_selection",
                         .icon = "focus-selection",
                         .tooltip = LOC("node_editor.preview_selection"),
                         .active = canvas_ ? canvas_->previewSelection() : preview_selection_});
        items.push_back({.kind = HeaderItem::Kind::Toggle,
                         .id = "sidebar",
                         .icon = "layout-columns-right",
                         .tooltip = LOC("node_editor.toggle_sidebar"),
                         .active = canvas_ ? canvas_->sidebarVisible() : sidebar_visible_});
    }

    void NodeEditor::headerAction(const AreaFrame&, screen::Screen&, const std::string_view action,
                                  const float x, const float y) {
        if (action == "preview_selection") {
            preview_selection_ = !(canvas_ ? canvas_->previewSelection() : preview_selection_);
            if (canvas_)
                canvas_->setPreviewSelection(preview_selection_);
        } else if (action == "sidebar") {
            sidebar_visible_ = !(canvas_ ? canvas_->sidebarVisible() : sidebar_visible_);
            if (canvas_)
                canvas_->setSidebarVisible(sidebar_visible_);
        } else if (canvas_) {
            canvas_->headerAction(action, x, y);
        }
    }

    void NodeEditor::draw(const AreaDrawContext& ctx) {
        const auto start = std::chrono::steady_clock::now();
        bindCanvas();
        const auto& content = ctx.area.content;
        if (canvas_ && area_id_ != ctx.area.id.value) {
            area_id_ = ctx.area.id.value;
            canvas_->invalidateView();
            host_->markContentDirty();
        }
        if (canvas_) {
            canvas_->setPanelScreenOffset(content.x, content.y);
            canvas_->setPanelSize(content.w, content.h);
        }
        if (canvas_ && canvas_->needsModelUpdate())
            host_->markContentDirty();
        const PanelInputState input = panelInput(ctx.input);
        host_->setInput(&input);
        host_->drawDirect(content.x, content.y, content.w, content.h);
        scene_manager_->modifierManager().recordCanvasFrame(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }

    bool NodeEditor::needsAnimationFrame() const {
        return (canvas_ && canvas_->needsModelUpdate()) || (host_ && host_->needsAnimationFrame());
    }

    // ---- Panel editors -----------------------------------------------------

    void PanelEditor::header(const AreaFrame& area, const screen::Screen&, std::vector<HeaderItem>& items) const {
        if (const auto details = PanelRegistry::instance().get_panel(area.editor))
            items.push_back({.kind = HeaderItem::Kind::Label, .id = "title", .label = details->label, .tooltip = details->label});
    }

    void PanelEditor::draw(const AreaDrawContext& ctx) {
        auto& reg = PanelRegistry::instance();
        const auto& c = ctx.area.content;
        const PanelInputState input = panelInput(ctx.input);
        if (ctx.live) {
            reg.render_panels({.target = PanelRenderTarget::for_panel(ctx.area.editor),
                               .mode = PanelRenderMode::DirectPreload,
                               .width = c.w,
                               .height = c.h,
                               .clip_y_min = c.y,
                               .clip_y_max = c.bottom(),
                               .input = &input},
                              ctx.draw);
        }
        reg.render_panels({.target = PanelRenderTarget::for_panel(ctx.area.editor),
                           .mode = ctx.live ? PanelRenderMode::Direct : PanelRenderMode::DirectCached,
                           .x = c.x,
                           .y = c.y,
                           .width = c.w,
                           .height = c.h,
                           .clip_y_min = c.y,
                           .clip_y_max = c.bottom(),
                           .input = &input},
                          ctx.draw);
    }

    void installPanelEditorTypes(screen::EditorTypeRegistry& registry) {
        const auto make = [](const PanelDetails& details) {
            const bool left = details.space == PanelSpace::LeftArea;
            screen::EditorType type;
            type.id = details.id;
            type.label = details.label;
            if (details.id == "native.sequencer")
                type.icon = "sequencer/film-strip";
            else if (details.id == "lfs.histogram")
                type.icon = "histogram";
            else
                type.icon = left ? "archive" : "layout-rows";
            type.placement = left ? screen::EditorPlacement{.anchor = screen::EditorPlacement::Anchor::ScreenEdge,
                                                            .edge_side = screen::Side::Left,
                                                            .edge_fraction = 0.2f}
                                  : screen::EditorPlacement{.anchor = screen::EditorPlacement::Anchor::ActiveView,
                                                            .side = screen::Side::Bottom,
                                                            .fraction = 0.32f};
            return type;
        };
        registry.setDynamicProvider(
            [make](const std::string_view id) -> std::optional<screen::EditorType> {
                const auto details = PanelRegistry::instance().get_panel(std::string(id));
                if (!details || !isPanelEditorSpace(details->space) || !details->parent_id.empty())
                    return std::nullopt;
                return make(*details);
            },
            [make] {
                std::vector<screen::EditorType> out;
                auto& reg = PanelRegistry::instance();
                std::vector<PanelDetails> panels;
                for (const PanelSpace space : {PanelSpace::LeftArea, PanelSpace::BottomArea}) {
                    for (const auto& id : reg.get_panel_names(space)) {
                        if (auto details = reg.get_panel(id); details && details->parent_id.empty())
                            panels.push_back(std::move(*details));
                    }
                }
                std::stable_sort(panels.begin(), panels.end(), [](const PanelDetails& a, const PanelDetails& b) {
                    return a.order != b.order ? a.order < b.order : a.label < b.label;
                });
                for (const auto& details : panels)
                    out.push_back(make(details));
                return out;
            });
    }

} // namespace lfs::vis::gui
