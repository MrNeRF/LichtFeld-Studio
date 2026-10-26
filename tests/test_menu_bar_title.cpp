/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "collision_fixture.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/path_utils.hpp"
#include "core/services.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "input/input_bindings.hpp"
#include "input/input_controller.hpp"
#include "ipc/view_context.hpp"
#include "python/python_runtime.hpp"
#include "rendering/render_pass.hpp"
#include "scene/scene_manager.hpp"
#include "screen/screen_service.hpp"
#include "screen/view3d_space.hpp"
#include "test_view_targets.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/visualizer.hpp"
#include <SDL3/SDL_keycode.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementText.h>
#include <RmlUi/Core/ElementUtilities.h>
#include <RmlUi/Core/Factory.h>
#include <RmlUi/Core/RenderInterface.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <thread>
#include <vector>

namespace lfs::vis::gui {
    // Attach a CPU Rml context instead of creating the Vulkan-backed GUI manager.
    // Model registration, resources, layout, updates and hit tests are production code.
    class RmlMenuBarTestAccess {
    public:
        static void enterWalk(RmlMenuBar& bar) {
            bar.dispatchToolbarAction("set_camera_navigation_mode", "walk");
        }
        static void toggleProjection(RmlMenuBar& bar) {
            bar.dispatchToolbarAction("toggle_projection", "");
        }
        static void dropdown(RmlMenuBar& bar, const std::vector<MenuItemDesc>& items) {
            bar.dropdown_container_ = bar.document_->GetElementById("dropdown-container");
            bar.dropdown_popup_ = bar.document_->GetElementById("dropdown-popup");
            bar.dropdown_overlay_ = bar.document_->GetElementById("dropdown-overlay");
            bar.dropdown_items_ = bar.buildDropdownItems(items);
            bar.open_menu_index_ = 0;
            bar.open_menu_idname_ = "file";
            bar.open_submenu_index_ = -1;
            bar.open_child_submenu_index_ = -1;
            bar.open_grandchild_submenu_index_ = -1;
            bar.keyboard_path_.fill(-1);
            bar.keyboard_depth_ = 0;
            bar.keyboard_opened_menu_ = false;
            bar.syncAccessKeyVisibility();
            bar.moveKeyboardHighlight(1, true);
            bar.dropdown_container_->SetClass("visible", true);
            bar.dropdown_popup_->SetProperty("left", "1400px");
            bar.dropdown_popup_->SetProperty("top", "30px");
            bar.menu_model_.DirtyVariable("dropdown_items");
            bar.rml_context_->Update();
        }
        static void openPath(RmlMenuBar& bar, int root, int child, int grandchild) {
            bar.setOpenSubmenu(root, child, grandchild);
            bar.sizeOpenDropdowns();
        }
        static bool keyEvent(RmlMenuBar& bar, lfs::vis::FrameInputEventKind kind,
                             SDL_Scancode code, bool window_focused = true) {
            lfs::vis::gui::PanelInputState input{};
            input.screen_w = 1600;
            input.screen_h = 300;
            lfs::vis::InputEventDispatch dispatch;
            lfs::vis::FrameInputEvent event{.kind = kind, .scancode = code, .dispatch = &dispatch};
            if (window_focused) {
                input.input_events.push_back(event);
                bar.processInput(input);
            } else {
                bar.handleMenuKey(event, false, false);
            }
            return dispatch.consumed;
        }
        static void heldKey(RmlMenuBar& bar, SDL_Scancode code) {
            lfs::vis::gui::PanelInputState input{};
            input.screen_w = 1600;
            input.screen_h = 300;
            lfs::vis::InputEventDispatch dispatch;
            input.input_events.push_back(
                {.kind = lfs::vis::FrameInputEventKind::KeyDown, .scancode = code, .repeat = true, .dispatch = &dispatch});
            bar.processInput(input);
        }
        static void key(RmlMenuBar& bar, SDL_Scancode code) {
            SCOPED_TRACE(static_cast<int>(code));
            EXPECT_TRUE(keyEvent(bar, lfs::vis::FrameInputEventKind::KeyDown, code))
                << SDL_GetScancodeName(code) << " open=" << bar.isOpen()
                << " depth=" << bar.keyboard_depth_;
        }
        static int depth(const RmlMenuBar& bar) { return bar.keyboard_depth_; }
        static int path(const RmlMenuBar& bar, int depth) { return bar.keyboard_path_[depth]; }
        static int menuIndex(const RmlMenuBar& bar) { return bar.open_menu_index_; }
        static int activeLabel(const RmlMenuBar& bar) { return bar.active_index_; }
        static void altTap(RmlMenuBar& bar, bool mouse_down = false) {
            for (const auto kind : {lfs::vis::FrameInputEventKind::KeyDown, lfs::vis::FrameInputEventKind::KeyUp}) {
                lfs::vis::gui::PanelInputState input{};
                input.screen_w = 1600;
                input.screen_h = 300;
                input.mouse_down[0] = mouse_down;
                lfs::vis::InputEventDispatch dispatch;
                input.input_events.push_back({.kind = kind, .scancode = SDL_SCANCODE_LALT, .dispatch = &dispatch});
                bar.processInput(input);
                if (kind == lfs::vis::FrameInputEventKind::KeyUp)
                    EXPECT_EQ(dispatch.consumed, !mouse_down);
            }
        }
        static void enableLeafLabel(RmlMenuBar& bar, int root, int child, int grandchild, int leaf) {
            bar.dropdown_items_[root].children[child].children[grandchild].children[leaf].enabled = true;
        }
        static void keepContextHeight(RmlMenuBar& bar) { bar.last_ctx_h_ = bar.rml_context_->GetDimensions().y; }
        static void windowFocusLost(RmlMenuBar& bar) { bar.onWindowFocusLost(); }
        static bool renderNeeded(const RmlMenuBar& bar) { return bar.render_needed_; }
        static void setRenderNeeded(RmlMenuBar& bar, bool needed) { bar.render_needed_ = needed; }
        static void hover(RmlMenuBar& bar, float x, float y, bool force_input) {
            lfs::vis::gui::PanelInputState input{};
            input.screen_w = 1600;
            input.screen_h = 300;
            input.mouse_x = x;
            input.mouse_y = y;
            input.mouse_wheel = force_input ? 1.0f : 0.0f;
            bar.processInput(input);
        }
        static void click(RmlMenuBar& bar, float x, float y) {
            hover(bar, x, y, true);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.down = true;
            event.button.x = x;
            event.button.y = y;
            bar.rml_manager_->dispatchInputEvent(event);
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.type = SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.down = false;
            bar.rml_manager_->dispatchInputEvent(event);
        }
        static void outsideClick(RmlMenuBar& bar) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = 1500.0f;
            event.button.y = 299.0f;
            bar.rml_manager_->dispatchInputEvent(event);
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.type = SDL_EVENT_MOUSE_BUTTON_UP;
            bar.rml_manager_->dispatchInputEvent(event);
        }
        static void frame(RmlMenuBar& bar, std::vector<lfs::vis::FrameInputEvent> events = {}) {
            lfs::vis::gui::PanelInputState input{};
            input.screen_w = 1600;
            input.screen_h = 300;
            input.input_events = std::move(events);
            bar.processInput(input);
        }
        static void dispatchKey(RmlMenuBar& bar, SDL_Scancode code) {
            SDL_Event event{};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.type = SDL_EVENT_KEY_DOWN;
            event.key.scancode = code;
            event.key.down = true;
            bar.rml_manager_->dispatchInputEvent(event);
        }
        static Rml::Element* menuHit(RmlMenuBar& bar, float x, float y) {
            return bar.dropdownElementAtPoint(x, y);
        }
        static int grandchildIndex(RmlMenuBar& bar, Rml::Element* element) {
            return bar.grandchildSubmenuIndexForElement(element);
        }
        static void bind(RmlMenuBar& bar, Rml::Context* context) {
            bar.rml_context_ = context;
            bar.bindModel();
            bar.camera_buttons_ = {
                {.button_id = "menu-camera-orbit", .action = "set_camera_navigation_mode", .value = "orbit"},
                {.button_id = "menu-camera-trackball", .action = "set_camera_navigation_mode", .value = "trackball"},
                {.button_id = "menu-camera-fpv", .action = "set_camera_navigation_mode", .value = "fpv"},
                {.button_id = "menu-camera-drone", .action = "set_camera_navigation_mode", .value = "drone"},
            };
            bar.snap_buttons_ = {{.button_id = "menu-view-snap", .action = "toggle_camera_view_snap"}};
            for (const auto* name : {"menu_camera_buttons", "menu_snap_buttons"})
                bar.menu_model_.DirtyVariable(name);
        }
        static void attach(RmlMenuBar& bar, Rml::ElementDocument* doc, RmlUIManager& manager) {
            bar.rml_manager_ = &manager;
            bar.document_ = doc;
            bar.menu_items_ = doc->GetElementById("menu-items");
            bar.menu_toolbar_ = doc->GetElementById("menu-toolbar");
            bar.menu_window_controls_ = doc->GetElementById("menu-window-controls");
            bar.dropdown_container_ = doc->GetElementById("dropdown-container");
            bar.dropdown_popup_ = doc->GetElementById("dropdown-popup");
            bar.dropdown_overlay_ = doc->GetElementById("dropdown-overlay");
            bar.project_title_container_ = doc->GetElementById("project-title");
            bar.project_title_el_ = doc->GetElementById("project-title-content");
            bar.body_el_ = doc->GetElementById("body");
        }
        static bool walkEnabled(RmlMenuBar& bar) {
            bar.rebuildToolbarButtons();
            return bar.camera_buttons_.back().enabled;
        }
        static bool accessVisible(const RmlMenuBar& bar) {
            return bar.body_el_ && bar.body_el_->IsClassSet("access-keys-visible");
        }
        static bool letterKey(RmlMenuBar& bar, SDL_Keycode key, SDL_Keymod modifiers = SDL_KMOD_NONE,
                              bool repeat = false) {
            lfs::vis::gui::PanelInputState input{};
            input.screen_w = 1600;
            input.screen_h = 300;
            lfs::vis::InputEventDispatch dispatch;
            input.input_events.push_back({.kind = lfs::vis::FrameInputEventKind::KeyDown,
                                          .scancode = SDL_SCANCODE_UNKNOWN,
                                          .keycode = key,
                                          .modifiers = modifiers,
                                          .repeat = repeat,
                                          .dispatch = &dispatch});
            bar.processInput(input);
            return dispatch.consumed;
        }
        static void fullToolbar(RmlMenuBar& bar) {
            bar.camera_buttons_.resize(4);
            bar.snap_buttons_.resize(7);
            int index = 0;
            for (auto* buttons : {&bar.camera_buttons_, &bar.snap_buttons_}) {
                for (auto& button : *buttons) {
                    button.button_id = "toolbar-test-" + std::to_string(index++);
                    button.action = "test-action";
                }
            }
            for (const auto* name : {"menu_camera_buttons", "menu_snap_buttons"})
                bar.menu_model_.DirtyVariable(name);
        }
        static void toolbarLayout(RmlMenuBar& bar, int width, float dp) {
            bar.updateToolbarLayout(width, dp);
        }
        static Rml::Element* toolbarHit(const RmlMenuBar& bar, float x, float y) {
            return bar.toolbarButtonAtPoint(x, y);
        }
        static float heightDp(const RmlMenuBar& bar) { return bar.bar_height_; }
        static RmlTooltipController& tooltip(RmlMenuBar& bar) { return bar.tooltip_; }
        static void portalLabel(RmlMenuBar& bar, std::string label) {
            bar.portal_connection_label_ = std::move(label);
            bar.menu_model_.DirtyVariable("portal_connection_label");
        }
        static void rebuildPortalStatus(RmlMenuBar& bar) { bar.rebuildPortalStatus(); }
        static void layout(RmlMenuBar& bar, int width, float dp) {
            bar.updateCompactLayout(width, dp);
            bar.rml_context_->Update();
            bar.updateProjectTitleLayout(width, dp);
        }
        static bool hit(const RmlMenuBar& bar, float x, float y) {
            return bar.projectTitleAtPoint(x, y);
        }
        static void toolbar(RmlMenuBar& bar, bool visible, float right) {
            bar.toolbar_fits_ = visible;
            bar.applied_toolbar_right_ = right;
            bar.menu_toolbar_->SetProperty("visibility", visible ? "visible" : "hidden");
            bar.menu_toolbar_->SetProperty("right", std::to_string(right) + "px");
        }
    };
} // namespace lfs::vis::gui

namespace {
    using lfs::vis::ProjectDisplayInfo;
    using lfs::vis::gui::resolveRmlTooltip;
    using lfs::vis::gui::RmlMenuBarTestAccess;

    class TitleRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
        void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
        Rml::TextureHandle LoadTexture(Rml::Vector2i& size, const Rml::String&) override {
            size = {16, 16};
            return 1;
        }
        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    std::string resource(const char* name) {
        std::ifstream file(std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/rmlui/resources" / name);
        std::ostringstream result;
        result << file.rdbuf();
        return result.str();
    }

    std::string textContent(Rml::Element* element) {
        if (auto* text = dynamic_cast<Rml::ElementText*>(element))
            return text->GetText();
        std::string result;
        for (int i = 0; i < element->GetNumChildren(); ++i)
            result += textContent(element->GetChild(i));
        return result;
    }

    struct Rect {
        float left, top, right, bottom;
    };
    Rect bounds(Rml::Element* el) {
        const auto offset = el->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = el->GetBox().GetSize(Rml::BoxArea::Border);
        return {offset.x, offset.y, offset.x + size.x, offset.y + size.y};
    }

    class MenuBarTitleTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() {
            lfs::event::LocalizationManager::getInstance().reset();
            Rml::Shutdown();
        }
        void SetUp() override {
            context_ = Rml::CreateContext("menu_bar_title_test", {1600, 300}, &renderer_);
            ASSERT_NE(context_, nullptr);
            RmlMenuBarTestAccess::bind(bar_, context_);
            document_ = context_->LoadDocument((std::filesystem::path(PROJECT_ROOT_PATH) /
                                                "src/visualizer/gui/rmlui/resources/menubar.rml")
                                                   .string());
            ASSERT_NE(document_, nullptr);
            document_->SetStyleSheetContainer(Rml::Factory::InstanceStyleSheetString(
                resource("components.rcss") + "\n" + resource("menubar.rcss")));
            for (const auto* id : {"project-title", "project-title-content", "menu-items", "menu-toolbar", "menu-window-controls"})
                ASSERT_NE(document_->GetElementById(id), nullptr) << id;
            RmlMenuBarTestAccess::attach(bar_, document_, manager_);
            bar_.updateLabels({"File", "Edit", "View", "Help"}, {"file", "edit", "view", "help"});
            bar_.updateProjectDisplay(ProjectDisplayInfo{.path = "/projects/scene.licht", .title = "Scene"});
            document_->Show();
            resize(1600);
        }
        void TearDown() override {
            // Release input registrations before destroying the externally owned context.
            manager_.shutdown();
            ASSERT_TRUE(Rml::RemoveContext("menu_bar_title_test"));
        }
        Rml::Element* el(const char* id) { return document_->GetElementById(id); }
        void resize(int width, float dp = 1.0f, bool toolbar = true) {
            context_->SetDensityIndependentPixelRatio(dp);
            context_->SetDimensions({width, 300});
            context_->Update();
            const float right = width - bounds(el("menu-window-controls")).left + 12 * dp;
            RmlMenuBarTestAccess::toolbar(bar_, toolbar, right);
            context_->Update();
            RmlMenuBarTestAccess::layout(bar_, width, dp);
            context_->Update();
        }
        std::string titleText() { return textContent(el("project-title-content")->GetChild(0)); }
        struct MenuKeyState {
            bool open = false;
            int index = -1;
            int depth = -1;
            int path0 = -1;
            bool operator==(const MenuKeyState&) const = default;
        };
        MenuKeyState menuKeyState() {
            return {bar_.isOpen(), RmlMenuBarTestAccess::menuIndex(bar_), RmlMenuBarTestAccess::depth(bar_),
                    RmlMenuBarTestAccess::path(bar_, 0)};
        }
        Rml::Element* rowByCallback(int callback) {
            Rml::ElementList rows;
            el("dropdown-popup")->GetElementsByClassName(rows, "menu-item");
            for (auto* row : rows)
                if (row->GetAttribute<int>("data-callback-index", -1) == callback)
                    return row;
            return nullptr;
        }
        std::string accessMark(Rml::Element* row) {
            Rml::ElementList marks;
            row->GetElementsByClassName(marks, "access-key");
            if (marks.size() != 1)
                return {};
            return textContent(marks.front());
        }
        inline static TitleRenderInterface renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        lfs::vis::gui::RmlUIManager manager_;
        lfs::vis::gui::RmlMenuBar bar_;
    };

    TEST_F(MenuBarTitleTest, WalkRejectionDoesNotEscapeNativeDispatch) {
        lfs::vis::input::InputBindings::setPersistenceEnabled(false);
        lfs::vis::SceneManager manager;
        lfs::vis::services().set(&manager);
        Viewport viewport(200, 200);
        lfs::vis::TestViewTargets targets(viewport);
        lfs::vis::InputController controller(nullptr, targets);
        controller.initialize();
        controller.setCameraNavigationMode(lfs::vis::InputController::CameraNavigationMode::Orbit);
        EXPECT_NO_THROW(RmlMenuBarTestAccess::enterWalk(bar_));
        EXPECT_EQ(controller.walk_error, "No collision nodes are loaded");
        const auto floor = manager.getScene().addCollision("Floor", collision_test::box({-8, -1, -8}, {16, 1, 16}));
        manager.getScene().setNodeTransform(floor, lfs::rendering::VISUALIZER_TO_DATA_WORLD_AXES_4);
        manager.getScene().setNodeVisibility(floor, false);
        EXPECT_TRUE(RmlMenuBarTestAccess::walkEnabled(bar_));
        viewport.camera.t = {0, 20, 0};
        EXPECT_NO_THROW(RmlMenuBarTestAccess::enterWalk(bar_));
        for (int frame = 0; frame < 20; ++frame)
            EXPECT_NO_THROW(RmlMenuBarTestAccess::frame(bar_));
        EXPECT_EQ(controller.walk_error, "No Walk support below the camera");
        EXPECT_EQ(controller.cameraNavigationMode(), lfs::vis::InputController::CameraNavigationMode::Orbit);
        EXPECT_EQ(viewport.camera.t, glm::vec3(0, 20, 0));
        lfs::vis::services().clear();
        lfs::vis::input::InputBindings::setPersistenceEnabled(true);
    }

    TEST_F(MenuBarTitleTest, ContextMenuFitsShortcutColumns) {
        auto* doc = context_->LoadDocumentFromMemory(
            "<rml><body><div id='menu' class='context-menu visible has-shortcuts'>"
            "<div><button class='context-menu-item has-shortcut'><span class='context-menu-text'>"
            "Perspective</span><span class='context-menu-shortcut'>Numpad 5</span></button></div>"
            "<div><button class='context-menu-item has-shortcut submenu-item active'>"
            "<span class='context-menu-check'>&#x2713;</span><span class='context-menu-text'>"
            "Bottom</span><span class='context-menu-shortcut'>Ctrl Numpad 7</span></button></div>"
            "<div><button class='context-menu-item has-shortcut'><span class='context-menu-text'>"
            "Maximize Area</span><span class='context-menu-shortcut'>Ctrl Space</span></button></div>"
            "<div><button class='context-menu-item'>Split Horizontally</button></div>"
            "</div></body></rml>");
        ASSERT_NE(doc, nullptr);
        doc->SetStyleSheetContainer(Rml::Factory::InstanceStyleSheetString(
            resource("components.rcss") + "\n" + resource("global_context_menu.rcss")));
        doc->Show();
        auto* menu = doc->GetElementById("menu");
        for (const float dp : {1.0f, 1.5f}) {
            context_->SetDensityIndependentPixelRatio(dp);
            for (const int width : {850, 1600}) {
                context_->SetDimensions({width, 900});
                menu->SetProperty("max-width", std::to_string(width - 8 * dp) + "px");
                for (const int left : {20, width - 40}) {
                    SCOPED_TRACE(::testing::Message() << width << " dp=" << dp << " left=" << left);
                    menu->SetProperty("left", std::to_string(left) + "px");
                    context_->Update();
                    const auto menu_width = bounds(menu).right - bounds(menu).left;
                    EXPECT_LE(menu_width, width - 8 * dp);
                    Rml::ElementList rows;
                    menu->GetElementsByClassName(rows, "context-menu-item");
                    for (auto* row : rows) {
                        EXPECT_LE(bounds(row).right, bounds(menu).right);
                        auto* label = row->QuerySelector(".context-menu-text");
                        auto* shortcut = row->QuerySelector(".context-menu-shortcut");
                        if (!shortcut)
                            continue;
                        EXPECT_LE(bounds(label).right + 8 * dp, bounds(shortcut).left);
                        EXPECT_LE(bounds(shortcut).right + 12 * dp, bounds(menu).right);
                        EXPECT_LT(bounds(label).bottom - bounds(label).top, 20 * dp);
                        EXPECT_LT(bounds(shortcut).bottom - bounds(shortcut).top, 20 * dp);
                    }
                }
            }
        }
        // Equal byte counts with different glyph widths must size differently.
        context_->SetDensityIndependentPixelRatio(1);
        menu->SetProperty("left", "20px");
        auto* label = menu->QuerySelector(".context-menu-text");
        label->SetInnerRML("WWWWWWWWWWWWWWWWWWWW");
        context_->Update();
        const auto wide = bounds(menu).right - bounds(menu).left;
        label->SetInnerRML("iiiiiiiiiiiiiiiiiiii");
        context_->Update();
        EXPECT_LT(bounds(menu).right - bounds(menu).left, wide);
        doc->Close();
    }

    // Fails if the shortcuts of a menu do not form one left-aligned column at its right edge, so
    // "Ctrl+A", "Ctrl+D" and "Ctrl+I" start at different places, or if a toggle's checkmark pushes
    // the column off the edge or the checkmarks do not line up.
    TEST_F(MenuBarTitleTest, ShortcutsFormOneColumnAtTheEdge) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::ShortcutItem, .label = "Hide UI", .shortcut = "F12", .callback_index = 1},
            {.type = Type::Toggle, .label = "Performance HUD", .shortcut = "F10", .callback_index = 2},
            {.type = Type::Toggle, .label = "Logging", .selected = true, .callback_index = 3},
            {.type = Type::SubMenuBegin, .label = "Select"},
            {.type = Type::ShortcutItem, .label = "Select All", .shortcut = "Ctrl+A", .callback_index = 4},
            {.type = Type::ShortcutItem, .label = "Deselect All", .shortcut = "Ctrl+D", .callback_index = 5},
            {.type = Type::ShortcutItem, .label = "Invert Selection", .shortcut = "Ctrl+I", .callback_index = 6},
            {.type = Type::SubMenuEnd},
        };
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::openPath(bar_, 3, -1, -1);
        Rml::ElementList submenus;
        el("dropdown-popup")->GetElementsByClassName(submenus, "submenu-popup");
        std::erase_if(submenus, [](Rml::Element* popup) { return popup->GetDisplay() == Rml::Style::Display::None; });
        ASSERT_EQ(submenus.size(), 1u);
        for (auto* popup : {el("dropdown-popup"), submenus.front()}) {
            std::vector<float> text_lefts;
            float widest_right = 0.0f;
            float content_right = 0.0f;
            std::vector<float> checkmark_lefts;
            Rml::ElementList rows;
            popup->GetElementsByClassName(rows, "menu-item");
            for (auto* row : rows) {
                if (!row->IsVisible(true) || row->GetParentNode()->GetParentNode() != popup)
                    continue;
                const auto& box = row->GetBox();
                content_right = bounds(row).right - box.GetEdge(Rml::BoxArea::Padding, Rml::BoxEdge::Right) -
                                box.GetEdge(Rml::BoxArea::Border, Rml::BoxEdge::Right);
                for (int i = 0; i < row->GetNumChildren(); ++i) {
                    auto* const child = row->GetChild(i);
                    if (!child->IsVisible())
                        continue;
                    if (child->IsClassSet("label")) {
                        EXPECT_LE(child->GetScrollWidth(), child->GetClientWidth() + 0.5f) << textContent(child);
                    }
                    if (child->IsClassSet("checkmark"))
                        checkmark_lefts.push_back(bounds(child).left);
                    if (child->IsClassSet("shortcut") && !textContent(child).empty()) {
                        // A text node has no box width; its glyph run ends at start + string width.
                        const float text_left = bounds(child->GetChild(0)).left;
                        text_lefts.push_back(text_left);
                        widest_right = std::max(widest_right, text_left + static_cast<float>(Rml::ElementUtilities::GetStringWidth(child, textContent(child))));
                    }
                }
            }
            ASSERT_GE(text_lefts.size(), 2u);
            for (const float left : text_lefts)
                EXPECT_NEAR(left, text_lefts.front(), 0.5f);
            EXPECT_NEAR(widest_right, content_right, 1.0f);
            for (const float left : checkmark_lefts)
                EXPECT_NEAR(left, checkmark_lefts.front(), 0.5f);
        }
    }

    TEST_F(MenuBarTitleTest, ArrowKeyMovesOpenMenuOncePerPress) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::Item, .label = "One", .callback_index = 1},
            {.type = Type::Item, .label = "Two", .callback_index = 2},
            {.type = Type::Item, .label = "Three", .callback_index = 3},
        };
        RmlMenuBarTestAccess::dropdown(bar_, items);
        // A frame makes the open menu the keyboard owner.
        RmlMenuBarTestAccess::frame(bar_);
        // One press reaches the menu through SDL dispatch and again in the next frame.
        RmlMenuBarTestAccess::dispatchKey(bar_, SDL_SCANCODE_DOWN);
        lfs::vis::InputEventDispatch dispatch;
        RmlMenuBarTestAccess::frame(bar_, {{.kind = lfs::vis::FrameInputEventKind::KeyDown,
                                            .scancode = SDL_SCANCODE_DOWN,
                                            .dispatch = &dispatch}});
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 1);
    }

    TEST_F(MenuBarTitleTest, HeldArrowKeysRepeatButOtherKeysActOnce) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::Item, .label = "One", .callback_index = 1},
            {.type = Type::Item, .label = "Two", .callback_index = 2},
            {.type = Type::Item, .label = "Three", .callback_index = 3},
        };
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::heldKey(bar_, SDL_SCANCODE_DOWN);
        RmlMenuBarTestAccess::heldKey(bar_, SDL_SCANCODE_DOWN);
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 2);
        RmlMenuBarTestAccess::heldKey(bar_, SDL_SCANCODE_RETURN);
        EXPECT_TRUE(bar_.isOpen());
    }

    TEST_F(MenuBarTitleTest, KeyboardNavigationSequences) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::SubMenuBegin, .label = "Group"},
            {.type = Type::SubMenuBegin, .label = "More"},
            {.type = Type::SubMenuBegin, .label = "Deep"},
            {.type = Type::Item, .label = "First", .callback_index = 1},
            {.type = Type::SubMenuBegin, .label = "Named group"},
            {.type = Type::Item, .label = "Nested", .callback_index = 5},
            {.type = Type::SubMenuEnd},
            {.type = Type::Separator},
            {.type = Type::Item, .label = "Disabled", .enabled = false, .callback_index = 2},
            {.type = Type::Item, .label = "Last", .callback_index = 3},
            {.type = Type::SubMenuEnd},
            {.type = Type::SubMenuEnd},
            {.type = Type::SubMenuEnd},
            {.type = Type::Item, .label = "Unavailable", .enabled = false},
            {.type = Type::Toggle, .label = "Toggle", .callback_index = 4},
            {.type = Type::Operator, .label = "Run", .operator_id = "menu_keyboard_test_missing"},
        };
        RmlMenuBarTestAccess::dropdown(bar_, items);
        struct Step {
            SDL_Scancode key;
            int depth;
            int selected;
        };
        const std::vector<Step> steps{
            {SDL_SCANCODE_RIGHT, 1, 0},
            {SDL_SCANCODE_RIGHT, 2, 0},
            {SDL_SCANCODE_RIGHT, 3, 0},
            {SDL_SCANCODE_DOWN, 3, 2},
            {SDL_SCANCODE_DOWN, 3, 4},
            {SDL_SCANCODE_DOWN, 3, 0},
            {SDL_SCANCODE_END, 3, 4},
            {SDL_SCANCODE_HOME, 3, 0},
            {SDL_SCANCODE_ESCAPE, 2, 0},
            {SDL_SCANCODE_LEFT, 1, 0},
            {SDL_SCANCODE_ESCAPE, 0, 0},
            {SDL_SCANCODE_DOWN, 0, 2},
        };
        RmlMenuBarTestAccess::enableLeafLabel(bar_, 0, 0, 0, 1);
        for (const auto& step : steps) {
            RmlMenuBarTestAccess::key(bar_, step.key);
            EXPECT_EQ(RmlMenuBarTestAccess::depth(bar_), step.depth);
            EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, step.depth), step.selected);
        }
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_RIGHT);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_LEFT);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 0);
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_RETURN);
        EXPECT_FALSE(bar_.isOpen());
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_SPACE);
        EXPECT_FALSE(bar_.isOpen());
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_RETURN);
        EXPECT_FALSE(bar_.isOpen());
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(bar_.isOpen());
        RmlMenuBarTestAccess::altTap(bar_, true);
        EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), -1);
        RmlMenuBarTestAccess::altTap(bar_);
        EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), 0);
        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown,
                                                    SDL_SCANCODE_A));
        EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), -1);
        RmlMenuBarTestAccess::altTap(bar_);
        RmlMenuBarTestAccess::outsideClick(bar_);
        EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), -1);
        const struct {
            bool focused;
            bool window_focus_lost;
            lfs::vis::FrameInputEventKind kind;
            int expected_active;
        } alt_tab_steps[] = {
            {true, false, lfs::vis::FrameInputEventKind::KeyDown, -1},
            {false, true, lfs::vis::FrameInputEventKind::KeyUp, -1},
        };
        for (const auto& step : alt_tab_steps) {
            if (step.window_focus_lost)
                RmlMenuBarTestAccess::windowFocusLost(bar_);
            EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, step.kind, SDL_SCANCODE_LALT,
                                                        step.focused));
            EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), step.expected_active);
        }
        RmlMenuBarTestAccess::altTap(bar_);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        EXPECT_TRUE(bar_.isOpen());
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(bar_.isOpen());
    }

    // A pointer-opened submenu keeps keyboard depth on the parent row. Escape
    // must close only that submenu; the next Escape closes the menu.
    TEST_F(MenuBarTitleTest, EscapeClosesMouseOpenedSubmenuOneLevel) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::SubMenuBegin, .label = "Import"},
            {.type = Type::SubMenuBegin, .label = "More"},
            {.type = Type::Item, .label = "Deep", .callback_index = 2},
            {.type = Type::SubMenuEnd},
            {.type = Type::Item, .label = "Image", .callback_index = 1},
            {.type = Type::SubMenuEnd},
            {.type = Type::Item, .label = "Open", .callback_index = 3},
        };
        const auto visible_submenus = [&] {
            Rml::ElementList popups;
            el("dropdown-popup")->GetElementsByClassName(popups, "submenu-popup");
            int visible = 0;
            for (auto* popup : popups)
                if (popup->GetDisplay() != Rml::Style::Display::None)
                    ++visible;
            return visible;
        };
        const auto hover_label = [&](const char* label) {
            Rml::ElementList rows;
            el("dropdown-popup")->GetElementsByClassName(rows, "menu-item");
            for (auto* row : rows) {
                if (row->GetDisplay() == Rml::Style::Display::None)
                    continue;
                if (textContent(row).find(label) == std::string::npos)
                    continue;
                const auto rect = bounds(row);
                RmlMenuBarTestAccess::hover(bar_, (rect.left + rect.right) * 0.5f,
                                            (rect.top + rect.bottom) * 0.5f, true);
                return;
            }
            FAIL() << label;
        };

        RmlMenuBarTestAccess::dropdown(bar_, items);
        hover_label("Import");
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(RmlMenuBarTestAccess::depth(bar_), 0);
        EXPECT_EQ(visible_submenus(), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(visible_submenus(), 0);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(bar_.isOpen());

        RmlMenuBarTestAccess::dropdown(bar_, items);
        hover_label("Import");
        hover_label("More");
        EXPECT_EQ(RmlMenuBarTestAccess::depth(bar_), 1);
        EXPECT_EQ(visible_submenus(), 2);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(visible_submenus(), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(visible_submenus(), 0);
    }

    TEST_F(MenuBarTitleTest, HoveringSameRowDoesNotDirtyTheMenuModel) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        const std::vector<Item> items{
            {.type = Type::SubMenuBegin, .label = "Group"},
            {.type = Type::SubMenuBegin, .label = "More"},
            {.type = Type::SubMenuBegin, .label = "Deep"},
            {.type = Type::Item, .label = "Leaf", .callback_index = 1},
            {.type = Type::SubMenuEnd},
            {.type = Type::SubMenuEnd},
            {.type = Type::SubMenuEnd},
        };
        RmlMenuBarTestAccess::dropdown(bar_, items);
        RmlMenuBarTestAccess::openPath(bar_, 0, 0, 0);
        Rml::ElementList rows;
        el("dropdown-popup")->GetElementsByClassName(rows, "menu-item");
        Rml::Element* leaf = nullptr;
        for (auto* row : rows) {
            if (row->GetAttribute<int>("data-callback-index", -1) == 1)
                leaf = row;
        }
        ASSERT_NE(leaf, nullptr);
        const auto rect = bounds(leaf);
        const float x = (rect.left + rect.right) / 2;
        const float y = (rect.top + rect.bottom) / 2;
        RmlMenuBarTestAccess::hover(bar_, x, y, false);
        RmlMenuBarTestAccess::setRenderNeeded(bar_, false);
        RmlMenuBarTestAccess::hover(bar_, x, y, true);
        EXPECT_FALSE(RmlMenuBarTestAccess::renderNeeded(bar_));
    }

    class MenuBarProjectionTest : public MenuBarTitleTest {
    protected:
        void SetUp() override {
            MenuBarTitleTest::SetUp();
            rendering_.retainVisibleViews({views_.activeView()});
            view_.settings.focal_length_mm = 50.0f;
            view_.settings.orthographic = false;
            view_.camera.camera.t = {0.0f, 0.0f, 7.5f};
            view_.camera.camera.pivot = {0.0f, 0.0f, 0.0f};
        }
        void toggle() {
            ASSERT_TRUE(lfs::vis::screen::applyViewCommand(view_, "projection", height_));
            rendering_.markCameraPoseChanged(views_.activeView());
        }
        float renderedScale() const { return view_.settings.ortho_scale; }
        lfs::vis::screen::View3DSpace view_;
        lfs::vis::screen::View3DSpace secondary_;
        lfs::vis::screen::ScreenService views_;
        lfs::vis::RenderingManager rendering_{views_};
        int height_ = 900;
    };

    TEST_F(MenuBarProjectionTest, EnterUsesMatchingScaleAfterRestoredOrWheelOverride) {
        for (const float stale_scale : {100.0f, 1.0f}) {
            view_.settings.ortho_scale = stale_scale;
            toggle();
            const auto settings = view_.settings;
            ASSERT_TRUE(settings.orthographic);
            EXPECT_FLOAT_EQ(renderedScale(), settings.ortho_scale);
            EXPECT_NE(renderedScale(), stale_scale);
            toggle();
        }
    }

    TEST_F(MenuBarProjectionTest, FreshViewKeepsMatchingScaleAndPose) {
        const auto eye = view_.camera.camera.t;
        const auto pivot = view_.camera.camera.pivot;
        const float expected = 900.0f / (15.0f * std::tan(
                                                     glm::radians(lfs::rendering::focalLengthToVFov(50.0f)) * 0.5f));
        toggle();
        EXPECT_FLOAT_EQ(renderedScale(), expected);
        EXPECT_EQ(view_.camera.camera.t, eye);
        EXPECT_EQ(view_.camera.camera.pivot, pivot);
        toggle();
        EXPECT_FALSE(view_.settings.orthographic);
        EXPECT_FLOAT_EQ(view_.settings.focal_length_mm, 50.0f);
    }

    TEST_F(MenuBarProjectionTest, InvalidViewUsesFallbackWithoutStaleOverride) {
        height_ = 0;
        view_.settings.ortho_scale = 1.0f;
        toggle();
        EXPECT_TRUE(view_.settings.orthographic);
        EXPECT_FLOAT_EQ(renderedScale(), 100.0f);
    }

    TEST_F(MenuBarProjectionTest, TogglePreservesIndependentSecondaryZoom) {
        auto& secondary = secondary_.settings;
        secondary.ortho_scale = 240.0f;
        view_.settings.ortho_scale = 1.0f;
        toggle();
        EXPECT_FLOAT_EQ(renderedScale(), view_.settings.ortho_scale);
        ASSERT_TRUE(secondary.ortho_scale);
        EXPECT_FLOAT_EQ(secondary.ortho_scale, 240.0f);
        toggle();
        EXPECT_FLOAT_EQ(secondary.ortho_scale, 240.0f);
    }

    TEST_F(MenuBarProjectionTest, RepeatedToggleKeepsFreshViewScaleStable) {
        constexpr int updates = 20000;
        std::vector<double> samples;
        for (int batch = 0; batch < 7; ++batch) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < updates; ++i)
                toggle();
            const auto elapsed = std::chrono::steady_clock::now() - start;
            samples.push_back(std::chrono::duration<double, std::nano>(elapsed).count() / updates);
            const auto plan = rendering_.frameDemandLedger().plan(lfs::vis::FrameClock::now());
            EXPECT_EQ(plan.render_views, 1u);
            EXPECT_TRUE(rendering_.frameDemandLedger().plan(lfs::vis::FrameClock::now()).empty());
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_ns_per_toggle", std::to_string(samples[samples.size() / 2]));
        EXPECT_FALSE(view_.settings.orthographic);
        EXPECT_FLOAT_EQ(renderedScale(), view_.settings.ortho_scale);
    }

    class ToolbarSystemInterface final : public Rml::SystemInterface {
    public:
        double GetElapsedTime() override { return time; }
        double time = 1.0;
    };

    class MenuBarToolbarTest : public MenuBarTitleTest {
    protected:
        static void SetUpTestSuite() {
            Rml::SetSystemInterface(&clock_);
            MenuBarTitleTest::SetUpTestSuite();
        }
        static void TearDownTestSuite() {
            MenuBarTitleTest::TearDownTestSuite();
            Rml::SetSystemInterface(nullptr);
        }
        void SetUp() override {
            clock_.time = 1.0;
            MenuBarTitleTest::SetUp();
            lfs::vis::gui::rml_theme::applyTheme(document_, resource("menubar.rcss"),
                                                 resource("menubar.theme.rcss"));
            context_->Update();
        }
        void expectImmediateSelection(Rml::Element* button) {
            ASSERT_NE(button, nullptr);
            const auto plain = button->GetProperty("background-color")->ToString();
            for (bool selected : {true, true, false, false, true, false}) {
                SCOPED_TRACE(::testing::Message() << button->GetId() << " selected=" << selected);
                button->SetClass("selected", selected);
                context_->Update();
                const auto immediate = button->GetProperty("background-color")->ToString();
                // A cached menu must already have its final color on the change frame.
                clock_.time += 1.0;
                context_->Update();
                const auto settled = button->GetProperty("background-color")->ToString();
                EXPECT_EQ(immediate, settled);
                if (selected)
                    EXPECT_NE(settled, plain);
                else
                    EXPECT_EQ(settled, plain);
            }
        }
        inline static ToolbarSystemInterface clock_;
    };

    TEST_F(MenuBarToolbarTest, SplitSelectionIsImmediateInBothDirections) {
        expectImmediateSelection(el("menu-window-split-view"));
    }

    TEST_F(MenuBarToolbarTest, UiVisibilitySelectionIsImmediateInBothDirections) {
        expectImmediateSelection(el("menu-window-toggle-ui"));
    }

    TEST_F(MenuBarToolbarTest, ModeSelectionRemainsImmediateInBothDirections) {
        Rml::ElementList buttons;
        document_->GetElementsByClassName(buttons, "menu-toolbar-btn");
        ASSERT_FALSE(buttons.empty());
        for (auto* button : buttons)
            expectImmediateSelection(button);
    }

    TEST_F(MenuBarToolbarTest, WindowControlsUseImmediateHoverFeedback) {
        for (const auto* id : {"menu-window-minimize", "menu-window-maximize", "menu-window-close"}) {
            const auto transition = el(id)->GetProperty("transition")->ToString();
            EXPECT_EQ(transition, "none");
        }
    }

    TEST_F(MenuBarTitleTest, ToolbarStaysReachableAcrossWindowSizesAndScales) {
        bar_.updateLabels({"File", "Edit", "Select", "Tools", "View", "Help"},
                          {"file", "edit", "select", "tools", "view", "help"});
        RmlMenuBarTestAccess::fullToolbar(bar_);
        for (float dp : {1.0f, 1.5f, 1.75f, 2.0f, 1.0f}) {
            for (int width : {1920, 1600, 900, 800, 640, 1600}) {
                SCOPED_TRACE(::testing::Message() << width << " dp=" << dp);
                context_->SetDensityIndependentPixelRatio(dp);
                context_->SetDimensions({width, 300});
                RmlMenuBarTestAccess::layout(bar_, width, dp);
                context_->Update();
                RmlMenuBarTestAccess::toolbarLayout(bar_, width, dp);
                context_->Update();
                const auto toolbar = bounds(el("menu-toolbar"));
                const auto menus = bounds(el("menu-items"));
                const auto controls = bounds(el("menu-window-controls"));
                EXPECT_TRUE(el("menu-toolbar")->IsVisible());
                EXPECT_GE(menus.top, 0);
                EXPECT_LE(menus.bottom, 30 * dp + 0.5f);
                EXPECT_GE(controls.top, 0);
                EXPECT_LE(controls.bottom, 30 * dp + 0.5f);
                EXPECT_GE(toolbar.left, 0);
                EXPECT_LE(toolbar.right, width);
                EXPECT_LE(toolbar.bottom, RmlMenuBarTestAccess::heightDp(bar_) * dp + 0.5f);
                EXPECT_TRUE(toolbar.top >= menus.bottom - 0.5f ||
                            (toolbar.left >= menus.right && toolbar.right <= controls.left));
                for (int i = 0; i < 11; ++i) {
                    auto* button = el(("toolbar-test-" + std::to_string(i)).c_str());
                    ASSERT_NE(button, nullptr);
                    auto rect = bounds(button);
                    EXPECT_TRUE(button->IsVisible());
                    EXPECT_EQ(RmlMenuBarTestAccess::toolbarHit(bar_, (rect.left + rect.right) / 2,
                                                               (rect.top + rect.bottom) / 2),
                              button);
                }
                const auto stable = toolbar;
                for (int frame = 0; frame < 4; ++frame) {
                    RmlMenuBarTestAccess::toolbarLayout(bar_, width, dp);
                    context_->Update();
                    EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).left, stable.left);
                    EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).top, stable.top);
                }
            }
        }
    }

    TEST_F(MenuBarTitleTest, ResizeUpdatesShellHeightAndInputBeforeDrawing) {
        bar_.updateLabels({"File", "Edit", "Select", "Tools", "View", "Help"},
                          {"file", "edit", "select", "tools", "view", "help"});
        RmlMenuBarTestAccess::fullToolbar(bar_);
        context_->Update();
        lfs::vis::gui::PanelInputState input{};
        input.screen_w = 800;
        input.screen_h = 600;
        bar_.processInput(input);
        EXPECT_FLOAT_EQ(bar_.barHeight(), 60);
        for (int i = 0; i < 11; ++i) {
            const auto button = bounds(el(("toolbar-test-" + std::to_string(i)).c_str()));
            input.mouse_x = (button.left + button.right) / 2;
            input.mouse_y = (button.top + button.bottom) / 2;
            bar_.processInput(input);
            EXPECT_TRUE(bar_.wantsInput());
        }
        input.screen_w = 1600;
        bar_.processInput(input);
        EXPECT_FLOAT_EQ(bar_.barHeight(), 30);
    }

    TEST_F(MenuBarTitleTest, GermanToolsStayOnTheTitleRowAt1000) {
        auto* body = el("body");
        auto* portal = el("menu-portal-connection");
        Rml::Element* portal_label = nullptr;
        for (int i = 0; i < portal->GetNumChildren(); ++i) {
            if (portal->GetChild(i)->GetTagName() == "span")
                portal_label = portal->GetChild(i);
        }
        ASSERT_NE(portal_label, nullptr);
        const auto layout_at = [&](int width) {
            context_->SetDimensions({width, 700});
            context_->Update();
            RmlMenuBarTestAccess::toolbarLayout(bar_, width, 1.0f);
            context_->Update();
        };

        bar_.updateLabels({"Hilfe", "Datei", "Bearbeiten", "Auswahl", "Werkzeuge", "Ansicht"},
                          {"lfs_plugins.help_menu.HelpMenu", "file", "edit", "select", "tools", "view"});
        bar_.updateProjectDisplay("Untitled", "/projects/scene.licht", false);
        RmlMenuBarTestAccess::portalLabel(bar_, "Portal: Nicht verbunden");
        RmlMenuBarTestAccess::fullToolbar(bar_);
        layout_at(1000);
        EXPECT_FALSE(el("menu-toolbar")->IsClassSet("second-row"));
        EXPECT_FLOAT_EQ(RmlMenuBarTestAccess::heightDp(bar_), 30);
        EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).top, 0);
        EXPECT_TRUE(body->IsClassSet("toolbar-squeeze"));
        EXPECT_FALSE(portal_label->IsVisible());
        EXPECT_TRUE(el("menu-window-maximize")->IsVisible());
        EXPECT_LT(bounds(el("toolbar-test-0")).bottom, 30.5f);
        EXPECT_GT(bounds(el("toolbar-test-10")).right, bounds(el("menu-items")).right);

        layout_at(1600);
        EXPECT_FALSE(body->IsClassSet("toolbar-squeeze"));
        EXPECT_FALSE(el("menu-toolbar")->IsClassSet("second-row"));
        EXPECT_TRUE(portal_label->IsVisible());
        EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).top, 0);

        bar_.updateLabels({"Help", "File", "Edit", "Select", "Tools", "View"},
                          {"lfs_plugins.help_menu.HelpMenu", "file", "edit", "select", "tools", "view"});
        RmlMenuBarTestAccess::portalLabel(bar_, "Portal: Not connected");
        layout_at(1000);
        EXPECT_FALSE(body->IsClassSet("toolbar-squeeze"));
        EXPECT_FALSE(el("menu-toolbar")->IsClassSet("second-row"));
        EXPECT_TRUE(portal_label->IsVisible());
        EXPECT_FLOAT_EQ(RmlMenuBarTestAccess::heightDp(bar_), 30);
    }

    TEST_F(MenuBarTitleTest, WideToolbarKeepsOriginalPlacementAndUpdateCost) {
        RmlMenuBarTestAccess::fullToolbar(bar_);
        resize(1600);
        const float menus_right = bounds(el("menu-items")).right;
        const float toolbar_width = el("menu-toolbar")->GetOffsetWidth();
        const float min_right = 1600 - bounds(el("menu-window-controls")).left + 4;
        for (float viewport_edge : {0.0f, 1200.0f, 1550.0f}) {
            bar_.setViewportRightEdge(viewport_edge);
            const float expected_right = std::min(std::max(min_right, viewport_edge > 0 ? 1600 - viewport_edge + 8 : min_right),
                                                  1600 - toolbar_width - menus_right - 12);
            RmlMenuBarTestAccess::toolbarLayout(bar_, 1600, 1);
            context_->Update();
            EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).right, 1600 - expected_right);
            EXPECT_FLOAT_EQ(bounds(el("menu-toolbar")).top, 0);
            EXPECT_FLOAT_EQ(RmlMenuBarTestAccess::heightDp(bar_), 30);
        }
        constexpr int iterations = 100000;
        for (int sample = 0; sample < 5; ++sample) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
                RmlMenuBarTestAccess::toolbarLayout(bar_, 1600, 1);
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            std::cout << "toolbar layout ns/update: " << double(ns) / iterations << '\n';
        }
    }

    TEST_F(MenuBarTitleTest, HiddenToolbarKeepsItsWidthAcrossScaleChanges) {
        resize(1600, 1.0f, true);
        const float base_width = el("menu-toolbar")->GetOffsetWidth();
        ASSERT_GT(base_width, 0);
        RmlMenuBarTestAccess::toolbar(bar_, false, 400);
        context_->Update();
        for (float dp : {2.0f, 1.5f, 1.0f}) {
            context_->SetDensityIndependentPixelRatio(dp);
            context_->Update();
            EXPECT_FALSE(el("menu-toolbar")->IsVisible());
            EXPECT_NEAR(el("menu-toolbar")->GetOffsetWidth(), base_width * dp, 1.0f);
        }
        resize(1600, 1.0f, true);
        EXPECT_TRUE(el("menu-toolbar")->IsVisible());
        EXPECT_FLOAT_EQ(el("menu-toolbar")->GetOffsetWidth(), base_width);
    }

    TEST_F(MenuBarTitleTest, ConstrainsAndCentersTitleBetweenMenusAndControls) {
        for (float dp : {1.0f, 1.5f}) {
            for (int width : {1600, 1200, 1000}) {
                for (bool toolbar : {true, false}) {
                    SCOPED_TRACE(::testing::Message() << width << " dp=" << dp << " toolbar=" << toolbar);
                    bar_.updateProjectDisplay(std::string(250, 'W'), "/projects/scene.licht", false);
                    resize(width * dp, dp, toolbar);
                    ASSERT_TRUE(el("project-title")->IsVisible());
                    auto title = bounds(el("project-title-content"));
                    const auto menus = bounds(el("menu-items"));
                    const auto controls = bounds(el(toolbar ? "menu-toolbar" : "menu-window-controls"));
                    EXPECT_GE(title.left, menus.right + 12 * dp - 0.5f);
                    EXPECT_LE(title.right, controls.left - 12 * dp + 0.5f);
                    EXPECT_NEAR((title.left + title.right) / 2, (menus.right + controls.left) / 2, 0.5f);
                    bar_.updateProjectDisplay("Short", "/projects/scene.licht", false);
                    context_->Update();
                    title = bounds(el("project-title-content"));
                    EXPECT_NEAR((title.left + title.right) / 2, (menus.right + controls.left) / 2, 0.5f);
                }
            }
        }
    }

    TEST_F(MenuBarTitleTest, NarrowBarsKeepEveryMenuVisible) {
        bar_.updateLabels({"Help", "File", "Edit", "Select", "Tools", "View"},
                          {"lfs_plugins.help_menu.HelpMenu", "file", "edit", "select", "tools", "view"});
        RmlMenuBarTestAccess::portalLabel(bar_, "Portal: Not connected");
        for (float dp : {1.0f, 1.5f, 2.0f}) {
            // The narrowest window SDL allows, and a little wider.
            for (int width : {640, 720}) {
                SCOPED_TRACE(::testing::Message() << width << " dp=" << dp);
                resize(width, dp, false);
                const auto controls = bounds(el("menu-window-controls"));
                Rml::ElementList labels;
                document_->GetElementsByClassName(labels, "menu-label");
                std::erase_if(labels, [](Rml::Element* label) { return !label->IsVisible(); });
                ASSERT_EQ(labels.size(), 6u);
                for (auto* label : labels)
                    EXPECT_LE(bounds(label).right, controls.left + 0.5f);
                EXPECT_LE(bounds(el("menu-window-close")).right, width + 0.5f);
            }
        }
    }

    TEST_F(MenuBarTitleTest, ToolbarKeepsNavigationAndViewSnapOnly) {
        context_->Update();
        for (const auto* id : {"menu-camera-orbit", "menu-camera-trackball", "menu-camera-fpv",
                               "menu-camera-drone", "menu-view-snap"})
            EXPECT_NE(el(id), nullptr) << id;

        Rml::ElementList buttons;
        document_->GetElementsByClassName(buttons, "menu-toolbar-btn");
        std::vector<Rml::Element*> rendered_buttons;
        for (auto* button : buttons) {
            if (button->GetAttribute<Rml::String>("id", "").empty())
                continue; // Rml keeps the two data-for templates in the tree.
            rendered_buttons.push_back(button);
            const auto action = button->GetAttribute<Rml::String>("data-action", "");
            const auto id = button->GetAttribute<Rml::String>("id", "");
            if (id == "menu-view-snap") {
                EXPECT_EQ(action, "toggle_camera_view_snap");
            } else {
                EXPECT_EQ(action, "set_camera_navigation_mode");
                EXPECT_EQ(id, "menu-camera-" + button->GetAttribute<Rml::String>("data-value", ""));
            }
            EXPECT_NE(action, "set_render_mode");
            EXPECT_NE(action, "set_projection");
            EXPECT_NE(action, "toggle_depth_view");
        }
        EXPECT_EQ(rendered_buttons.size(), 5u);
        for (const auto* id : {"menu-render-mode", "menu-projection-perspective", "menu-projection-orthographic",
                               "menu-depth-view"})
            EXPECT_EQ(el(id), nullptr) << id;
    }

    TEST_F(MenuBarTitleTest, ClippingStyleContractForThePaintBackend) {
        // The geometry sink does not paint glyphs. These computed-style checks
        // verify that the real stylesheet asks Rml to clip/ellipsize; they cannot
        // prove the backend's final pixels.
        for (auto* element : {el("project-title-content"), el("project-title-content")->GetChild(0)}) {
            EXPECT_EQ(element->GetComputedValues().overflow_x(), Rml::Style::Overflow::Hidden);
            EXPECT_EQ(element->GetComputedValues().text_overflow(), Rml::Style::TextOverflow::Ellipsis);
        }
    }

    TEST_F(MenuBarTitleTest, PortalStatusKeepsLongNameInTooltipAtNarrowWidth) {
        auto& store = lfs::vis::app_store();
        const auto previous = store.account_state.get();
        const std::string name = "Katharina Theodora Extremely Long Display Name Example";
        store.account_state.set(lfs::vis::AppStore::AccountState{
            .signed_in = true,
            .authorized = true,
            .label = "KE",
            .display_name = name,
        });
        RmlMenuBarTestAccess::rebuildPortalStatus(bar_);
        resize(1280);
        EXPECT_EQ(textContent(el("menu-portal-connection")), "KE · Portal");
        const auto tooltip = el("menu-portal-connection")->GetAttribute<Rml::String>("title", "");
        EXPECT_NE(tooltip.find("Portal connected as " + name), std::string::npos);
        EXPECT_EQ(textContent(el("menu-portal-connection")).find(name), std::string::npos);
        store.account_state.set(previous);
    }

    TEST_F(MenuBarTitleTest, PortalStatusShowsSpecificTransitionLabels) {
        auto& store = lfs::vis::app_store();
        const auto previous_account = store.account_state.get();
        const auto previous_gallery = store.gallery_state.get();
        store.gallery_state.set({});
        const auto label_for = [&](lfs::vis::AppStore::AccountState state) {
            store.account_state.set(std::move(state));
            RmlMenuBarTestAccess::rebuildPortalStatus(bar_);
            context_->Update();
            return textContent(el("menu-portal-connection"));
        };
        EXPECT_EQ(label_for({}), "Portal: Not connected");
        EXPECT_EQ(label_for({.authorized = true}), "Portal connected, switched off");
        EXPECT_EQ(label_for({.linking = true, .label = "ABCD-EFGH"}), "Portal: Connecting… ABCD-EFGH");
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .disconnecting = true, .label = "KT", .display_name = "Kay Test"}),
                  "Portal: Disconnecting…");
        auto approval_gallery = previous_gallery;
        approval_gallery.relink_required = true;
        store.gallery_state.set(approval_gallery);
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .label = "KT", .display_name = "Kay Test"}),
                  "Portal: Approval needed");
        store.gallery_state.set({});
        EXPECT_EQ(label_for({.signed_in = true, .authorized = true, .label = "KT", .display_name = "Kay Test"}),
                  "KT · Portal");
        store.account_state.set(previous_account);
        store.gallery_state.set(previous_gallery);
    }

    TEST_F(MenuBarTitleTest, AccountsForPendingToolbarPlacementInTheSameFrame) {
        // SetProperty is pending until Update. The layout code must use the new
        // placement selected by the caller, not the toolbar's stale Rml box.
        RmlMenuBarTestAccess::toolbar(bar_, true, 600);
        RmlMenuBarTestAccess::layout(bar_, 1600, 1.0f);
        context_->Update();
        const auto title = bounds(el("project-title"));
        EXPECT_LE(title.right, bounds(el("menu-toolbar")).left - 12 + 0.5f);
    }

    TEST_F(MenuBarTitleTest, HidesWhenNarrowAndReappearsAfterExpansion) {
        for (float dp : {1.0f, 1.5f}) {
            resize(1600 * dp, dp);
            const auto old = bounds(el("project-title-content"));
            resize(550 * dp, dp);
            EXPECT_FALSE(el("project-title")->IsVisible());
            EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, (old.left + old.right) / 2, 15 * dp));
            resize(1600 * dp, dp);
            EXPECT_TRUE(el("project-title")->IsVisible());
            const auto title = bounds(el("project-title-content"));
            EXPECT_TRUE(RmlMenuBarTestAccess::hit(bar_, (title.left + title.right) / 2, 15 * dp));
        }
    }

    TEST_F(MenuBarTitleTest, HitTestUsesVisibleTextBoundsAndRequiresAPath) {
        auto title = bounds(el("project-title-content"));
        const float x = (title.left + title.right) / 2, y = (title.top + title.bottom) / 2;
        EXPECT_TRUE(RmlMenuBarTestAccess::hit(bar_, x, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, title.left - 1, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, title.right, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, title.top - 1));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, title.bottom));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, 10, y));
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, 1590, y));
        bar_.updateProjectDisplay("Scene", "", false);
        context_->Update();
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, y));
        bar_.updateProjectDisplay("", "/projects/scene.licht", false);
        resize(1600);
        EXPECT_FALSE(el("project-title")->IsVisible());
        EXPECT_FALSE(RmlMenuBarTestAccess::hit(bar_, x, y));
    }

    TEST_F(MenuBarTitleTest, UsesMetadataThenFileStemThenUntitled) {
        const std::vector<std::pair<ProjectDisplayInfo, std::string>> cases = {
            {{.path = "/projects/saved.name.licht", .title = "Metadata title"}, "Metadata title"},
            {{.path = "/projects/saved.name.licht"}, "saved.name"},
            {{.path = "/projects/saved.name.licht", .title = ""}, "saved.name"},
            {{.title = "Unsaved title"}, "Unsaved title"},
            {{}, "Untitled"},
            {{.title = ""}, "Untitled"},
        };
        for (const auto& [info, expected] : cases) {
            SCOPED_TRACE(expected);
            bar_.updateProjectDisplay(info);
            context_->Update();
            EXPECT_EQ(titleText(), expected);
        }
    }

    TEST_F(MenuBarTitleTest, DirtyMarkerTracksBothTransitionsWithoutChangingTitle) {
        for (bool dirty : {false, true, false}) {
            bar_.updateProjectDisplay(ProjectDisplayInfo{.path = "/projects/scene.licht", .title = "Scene", .dirty = dirty});
            context_->Update();
            auto* marker = el("project-title-content")->QuerySelector(".project-dirty");
            ASSERT_NE(marker, nullptr);
            EXPECT_EQ(marker->IsVisible(), dirty);
            if (dirty)
                EXPECT_NE(textContent(marker).find('*'), std::string::npos);
            EXPECT_EQ(titleText(), "Scene");
        }
    }

    TEST_F(MenuBarTitleTest, SettingHelpOnlyAppearsOverItsLabel) {
        auto* row = document_->CreateElement("div").release();
        row->SetClass("setting-row", true);
        row->SetAttribute("title", "Setting help");
        row->SetInnerRML("<span class=\"setting-row__label-col\"><span>Label</span></span>"
                         "<div class=\"setting-row__control-col\"><input type=\"text\" /></div>");
        document_->AppendChild(Rml::ElementPtr(row));
        auto* label = row->GetChild(0);
        auto* control = row->GetChild(1)->GetChild(0);
        EXPECT_EQ(resolveRmlTooltip(label->GetChild(0)), "Setting help");
        EXPECT_TRUE(resolveRmlTooltip(control).empty());
        EXPECT_TRUE(resolveRmlTooltip(row).empty());
        control->SetAttribute("title", "Control help");
        EXPECT_EQ(resolveRmlTooltip(control), "Control help");
        label->SetClass("setting-row__label-col", false);
        label->SetClass("prop-label", true);
        EXPECT_EQ(lfs::vis::gui::propertyTooltipTarget(row), label);
        EXPECT_EQ(lfs::vis::gui::propertyTooltipTarget(control), control);
        EXPECT_EQ(resolveRmlTooltip(label->GetChild(0)), "Setting help");
        EXPECT_TRUE(resolveRmlTooltip(row).empty());
        control->RemoveAttribute("title");
        EXPECT_TRUE(resolveRmlTooltip(control).empty());
        label->SetClass("prop-label", false);
        EXPECT_EQ(lfs::vis::gui::propertyTooltipTarget(row), nullptr);
    }

    TEST_F(MenuBarTitleTest, ShortcutTooltipTracksBindingAndUnbinding) {
        using namespace lfs::vis::input;
        EXPECT_EQ(toolModeFromName("global"), ToolMode::GLOBAL);
        EXPECT_EQ(toolModeFromName("selection"), ToolMode::SELECTION);
        EXPECT_EQ(toolModeFromName("translate"), ToolMode::TRANSLATE);
        EXPECT_EQ(toolModeFromName("rotate"), ToolMode::ROTATE);
        EXPECT_EQ(toolModeFromName("scale"), ToolMode::SCALE);
        EXPECT_EQ(toolModeFromName("align"), ToolMode::ALIGN);
        EXPECT_EQ(toolModeFromName("crop_box"), ToolMode::CROP_BOX);
        EXPECT_EQ(toolModeFromName("SeLeCtIoN"), ToolMode::SELECTION);
        EXPECT_EQ(toolModeFromName("unknown"), ToolMode::GLOBAL);
        InputBindings bindings;
        lfs::python::set_keymap_bindings(&bindings);
        auto button = document_->CreateElement("button");
        button->SetAttribute("title", "Home");
        button->SetAttribute("data-action", "camera_reset_home");
        button->SetAttribute("data-shortcut", "stale");
        auto* element = document_->AppendChild(std::move(button));
        const auto initial = bindings.getLocalizedTriggerDescription(Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home (" + initial + ")");
        bindings.setBinding(ToolMode::GLOBAL, Action::CAMERA_RESET_HOME, KeyTrigger{KEY_F6});
        const auto rebound = bindings.getLocalizedTriggerDescription(Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home (" + rebound + ")");
        bindings.clearBinding(ToolMode::GLOBAL, Action::CAMERA_RESET_HOME);
        EXPECT_EQ(resolveRmlTooltip(element), "Home");

        auto selection_button = document_->CreateElement("button");
        selection_button->SetAttribute("title", "Depth filter");
        selection_button->SetAttribute("data-action", "toggle_depth_view");
        selection_button->SetAttribute("data-keymap-action", "toggle_selection_depth_filter");
        selection_button->SetAttribute("data-keymap-mode", "selection");
        auto* selection_element = document_->AppendChild(std::move(selection_button));
        const auto selection_shortcut = bindings.getLocalizedTriggerDescription(
            Action::TOGGLE_SELECTION_DEPTH_FILTER, ToolMode::SELECTION);
        EXPECT_EQ(resolveRmlTooltip(selection_element), "Depth filter (" + selection_shortcut + ")");
        bindings.clearBinding(ToolMode::SELECTION, Action::TOGGLE_SELECTION_DEPTH_FILTER);
        EXPECT_EQ(resolveRmlTooltip(selection_element), "Depth filter");

        auto& locale = lfs::event::LocalizationManager::getInstance();
        ASSERT_TRUE(locale.initialize((std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/resources/locales")
                                          .string()));
        ASSERT_TRUE(locale.setLanguage("de"));
        auto orbit_button = document_->CreateElement("button");
        orbit_button->SetAttribute("title", "Orbit");
        orbit_button->SetAttribute("data-action", "camera_orbit");
        auto* orbit_element = document_->AppendChild(std::move(orbit_button));
        const auto localized = bindings.getLocalizedTriggerDescription(Action::CAMERA_ORBIT);
        EXPECT_NE(localized.find("Ziehen"), std::string::npos);
        EXPECT_EQ(resolveRmlTooltip(orbit_element), "Orbit (" + localized + ")");
        EXPECT_TRUE(locale.setLanguage("en"));
        lfs::python::set_keymap_bindings(nullptr);
    }

    // Catches a visible tooltip that never hides once the pointer leaves its
    // context for another one (the next hover then skips the show delay): a
    // pointer move over the viewport must still produce a frame while any
    // context has an active tooltip, and must not when none has.
    TEST_F(MenuBarTitleTest, PointerMoveElsewhereRendersWhileATooltipIsActive) {
        Rml::Context* viewport = Rml::CreateContext("menu_bar_title_viewport", {1600, 260}, &renderer_);
        ASSERT_NE(viewport, nullptr);
        ASSERT_NE(viewport->LoadDocumentFromMemory("<rml><body></body></rml>"), nullptr);
        viewport->Update();
        viewport->ProcessMouseMove(800, 160, 0);
        viewport->Update();
        lfs::vis::gui::RmlUIManager manager;
        manager.trackContextFrame(context_, 0, 0);
        manager.trackContextFrame(viewport, 0, 40);

        EXPECT_FALSE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        manager.setContextNeedsPassiveMouseMoveFrames(context_, true);
        EXPECT_TRUE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        manager.setContextNeedsPassiveMouseMoveFrames(context_, false);
        EXPECT_FALSE(manager.passiveMouseMoveNeedsRender(800.0f, 200.0f));
        ASSERT_TRUE(Rml::RemoveContext("menu_bar_title_viewport"));
    }

    TEST_F(MenuBarTitleTest, PointerEventsAreScopedToTheMainWindowAndPressResumesInput) {
        RmlMenuBarTestAccess::keepContextHeight(bar_);
        auto* button = el("menu-items")->GetChild(0);
        ASSERT_NE(button, nullptr);
        auto rect = bounds(button);
        lfs::vis::gui::PanelInputState input{};
        input.screen_w = 1600;
        input.screen_h = 300;
        input.mouse_x = (rect.left + rect.right) / 2;
        input.mouse_y = (rect.top + rect.bottom) / 2;
        bar_.processInput(input);
        context_->Update();
        ASSERT_TRUE(button->IsPseudoClassSet("hover"));
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_MOUSE_LEAVE;
        event.window.windowID = 2;
        bar_.onWindowPointerEvent(event, 1);
        bar_.processInput(input);
        context_->Update();
        EXPECT_TRUE(button->IsPseudoClassSet("hover"));
        RmlMenuBarTestAccess::windowFocusLost(bar_);
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.motion.windowID = 2;
        bar_.onWindowPointerEvent(event, 1);
        bar_.processInput(input);
        context_->Update();
        EXPECT_FALSE(button->IsPseudoClassSet("hover"));
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.windowID = 1;
        bar_.onWindowPointerEvent(event, 1);
        rect = bounds(button);
        input.mouse_x = (rect.left + rect.right) / 2;
        input.mouse_y = (rect.top + rect.bottom) / 2;
        event.button.button = SDL_BUTTON_LEFT;
        event.button.down = true;
        event.button.x = input.mouse_x;
        event.button.y = input.mouse_y;
        manager_.dispatchInputEvent(event);
        context_->Update();
        EXPECT_TRUE(bar_.isOpen());
    }

    TEST_F(MenuBarTitleTest, RestoreBackKeepsItsOriginalOuterExtent) {
        auto* back = document_->QuerySelector(".window-restore-back");
        ASSERT_NE(back, nullptr);
        EXPECT_EQ(back->GetProperty("width")->ToString(), "11dp");
        EXPECT_EQ(back->GetProperty("height")->ToString(), "11dp");
    }

    TEST_F(MenuBarTitleTest, LosingWindowFocusClearsPointerFeedback) {
        RmlMenuBarTestAccess::keepContextHeight(bar_);
        auto* button = el("menu-window-close");
        ASSERT_NE(button, nullptr);
        const auto rect = bounds(button);
        context_->ProcessMouseMove(static_cast<int>((rect.left + rect.right) / 2),
                                   static_cast<int>((rect.top + rect.bottom) / 2), 0);
        context_->Update();
        ASSERT_TRUE(button->IsPseudoClassSet("hover"));
        RmlMenuBarTestAccess::setRenderNeeded(bar_, false);
        RmlMenuBarTestAccess::windowFocusLost(bar_);
        context_->Update();
        EXPECT_FALSE(button->IsPseudoClassSet("hover"));
        EXPECT_TRUE(RmlMenuBarTestAccess::renderNeeded(bar_));
        EXPECT_FALSE(RmlMenuBarTestAccess::tooltip(bar_).needsFrame());
        const lfs::vis::gui::PanelInputState input{.mouse_x = (rect.left + rect.right) / 2,
                                                   .mouse_y = (rect.top + rect.bottom) / 2,
                                                   .screen_w = 1600,
                                                   .screen_h = 300};
        bar_.processInput(input);
        context_->Update();
        EXPECT_FALSE(button->IsPseudoClassSet("hover"));
        const auto resumed_rect = bounds(button);
        auto moved_input = input;
        moved_input.mouse_x = (resumed_rect.left + resumed_rect.right) / 2;
        moved_input.mouse_y = (resumed_rect.top + resumed_rect.bottom) / 2;
        bar_.onPointerMotion();
        bar_.processInput(moved_input);
        context_->Update();
        EXPECT_TRUE(button->IsPseudoClassSet("hover"));
        bar_.onPointerLeave();
        bar_.processInput(input);
        context_->Update();
        EXPECT_FALSE(button->IsPseudoClassSet("hover"));
    }

    TEST_F(MenuBarTitleTest, TooltipPreservesFullPathAsTextIncludingMarkupCharacters) {
        const std::filesystem::path path =
            std::filesystem::path{"/projects"} / "long folder" / "<b>scan &amp; capture.licht";
        const std::string expected_path = lfs::core::path_to_utf8(path.lexically_normal());
        bar_.updateProjectDisplay(ProjectDisplayInfo{.path = path, .title = "Scan"});
        context_->Update();
        EXPECT_EQ(el("project-title-content")->GetAttribute<Rml::String>("title", ""), expected_path);
        resize(1600);
        const auto title = bounds(el("project-title-content"));
        bar_.processInput({.mouse_x = (title.left + title.right) / 2, .mouse_y = 15, .screen_w = 1600, .screen_h = 300});
        auto& tooltip = RmlMenuBarTestAccess::tooltip(bar_);
        const auto deadline = tooltip.revealDeadline();
        ASSERT_TRUE(deadline);
        std::this_thread::sleep_until(*deadline);
        ASSERT_TRUE(tooltip.apply(document_, 700, 15, 1600, 300));
        context_->Update();
        auto* rendered = el("frame-tooltip");
        ASSERT_NE(rendered, nullptr);
        EXPECT_TRUE(rendered->IsVisible());
        EXPECT_EQ(rendered->QuerySelector("b"), nullptr);
        EXPECT_EQ(textContent(rendered), expected_path);
    }

    TEST_F(MenuBarTitleTest, AccessKeys) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        RmlMenuBarTestAccess::altTap(bar_);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_DOWN);
        const auto opened_by_down = menuKeyState();
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        EXPECT_FALSE(RmlMenuBarTestAccess::accessVisible(bar_));
        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown, SDL_SCANCODE_LALT));
        EXPECT_TRUE(RmlMenuBarTestAccess::accessVisible(bar_));
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_F, SDL_KMOD_LALT));
        EXPECT_EQ(menuKeyState(), opened_by_down);
        const auto opened = menuKeyState();
        EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_F, SDL_KMOD_LALT, true));
        EXPECT_EQ(menuKeyState(), opened);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        RmlMenuBarTestAccess::altTap(bar_);
        EXPECT_TRUE(RmlMenuBarTestAccess::accessVisible(bar_));
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_F));
        EXPECT_EQ(menuKeyState(), opened_by_down);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(RmlMenuBarTestAccess::accessVisible(bar_));

        RmlMenuBarTestAccess::dropdown(bar_, {{.type = Type::Item, .label = "About", .callback_index = 7},
                                              {.type = Type::Item, .label = "Preferences", .callback_index = 8}});
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_P));
        EXPECT_FALSE(bar_.isOpen());

        RmlMenuBarTestAccess::dropdown(bar_, {{.type = Type::Item, .label = "X", .callback_index = 1},
                                              {.type = Type::Item, .label = "X", .callback_index = 2},
                                              {.type = Type::Item, .label = "Other", .callback_index = 3}});
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_X));
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 1);
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_X));
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 0);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_RETURN);
        EXPECT_FALSE(bar_.isOpen());

        RmlMenuBarTestAccess::dropdown(
            bar_, {{.type = Type::SubMenuBegin, .label = "Export"},
                   {.type = Type::Item, .label = "Image", .callback_index = 4},
                   {.type = Type::SubMenuEnd}});
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_E));
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(RmlMenuBarTestAccess::depth(bar_), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        RmlMenuBarTestAccess::dropdown(bar_, {{.type = Type::Item, .label = "Alpha", .enabled = false, .callback_index = 5},
                                              {.type = Type::Item, .label = "Beta", .callback_index = 6}});
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 1);
        EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_A));
        EXPECT_TRUE(bar_.isOpen());
        EXPECT_EQ(RmlMenuBarTestAccess::path(bar_, 0), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown, SDL_SCANCODE_RALT));
        EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_E, SDL_KMOD_RALT | SDL_KMOD_MODE));
        EXPECT_FALSE(bar_.isOpen());
        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyUp, SDL_SCANCODE_RALT));

        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_F, SDL_KMOD_RALT));
        EXPECT_EQ(menuKeyState(), opened_by_down);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        bar_.updateLabels({"Cameras"}, {"plugin.Cameras"});
        context_->Update();
        {
            lfs::vis::input::InputBindings::setPersistenceEnabled(false);
            Viewport viewport(32, 32);
            lfs::vis::TestViewTargets targets(viewport);
            lfs::vis::InputController controller(nullptr, targets);
            controller.initialize();
            EXPECT_FALSE(RmlMenuBarTestAccess::walkEnabled(bar_));
            EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown, SDL_SCANCODE_LALT));
            EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_C, SDL_KMOD_LALT));
            EXPECT_FALSE(bar_.isOpen());
            EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyUp, SDL_SCANCODE_LALT));
        }
        lfs::vis::input::InputBindings::setPersistenceEnabled(true);
        EXPECT_TRUE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_C, SDL_KMOD_LALT));
        EXPECT_TRUE(bar_.isOpen());
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        RmlMenuBarTestAccess::dropdown(bar_, {{.type = Type::Item, .label = "About", .callback_index = 7}});
        EXPECT_FALSE(RmlMenuBarTestAccess::accessVisible(bar_));
        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown, SDL_SCANCODE_LALT));
        EXPECT_TRUE(RmlMenuBarTestAccess::accessVisible(bar_));
        RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyUp, SDL_SCANCODE_LALT);
        EXPECT_FALSE(RmlMenuBarTestAccess::accessVisible(bar_));
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);

        auto* text_context = Rml::CreateContext("menu-access-text", {200, 80}, &renderer_);
        ASSERT_NE(text_context, nullptr);
        auto* text_document = text_context->LoadDocumentFromMemory(
            "<rml><head><style>body { font-family: Inter; font-size: 14px; } input { width: 120px; height: 24px; }</style></head>"
            "<body><input type=\"text\" id=\"field\"/></body></rml>");
        ASSERT_NE(text_document, nullptr);
        text_document->Show();
        auto* field = text_document->GetElementById("field");
        ASSERT_NE(field, nullptr);
        ASSERT_TRUE(field->Focus());
        text_context->Update();
        manager_.activateInput(text_context, [](const lfs::vis::gui::PanelInputState&) {});
        EXPECT_TRUE(manager_.wantsTextInput());
        EXPECT_FALSE(RmlMenuBarTestAccess::keyEvent(bar_, lfs::vis::FrameInputEventKind::KeyDown, SDL_SCANCODE_LALT));
        EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_F, SDL_KMOD_LALT));
        EXPECT_FALSE(bar_.isOpen());
        manager_.deactivateInput(text_context);
        manager_.shutdown();
        EXPECT_TRUE(Rml::RemoveContext("menu-access-text"));
    }

    TEST_F(MenuBarTitleTest, AccessKeyAssignment) {
        using Item = lfs::vis::gui::MenuItemDesc;
        using Type = Item::Type;
        RmlMenuBarTestAccess::dropdown(bar_, {{.type = Type::Item, .label = "Öffnen", .callback_index = 1},
                                              {.type = Type::Item, .label = "Speichern", .callback_index = 2},
                                              {.type = Type::Item, .label = "Schließen", .callback_index = 3},
                                              {.type = Type::Item, .label = "ファイル", .callback_index = 4},
                                              {.type = Type::Item, .label = "A < B & C", .callback_index = 5},
                                              {.type = Type::Item, .label = "X", .callback_index = 6},
                                              {.type = Type::Item, .label = "X", .enabled = false, .callback_index = 7}});
        EXPECT_EQ(accessMark(rowByCallback(1)), "f");
        EXPECT_EQ(accessMark(rowByCallback(2)), "S");
        EXPECT_EQ(accessMark(rowByCallback(3)), "c");
        EXPECT_EQ(textContent(rowByCallback(4)), "ファイル");
        EXPECT_TRUE(accessMark(rowByCallback(4)).empty());
        auto* markup = rowByCallback(5);
        EXPECT_EQ(accessMark(markup), "A");
        EXPECT_NE(textContent(markup).find('<'), std::string::npos);
        EXPECT_NE(textContent(markup).find('&'), std::string::npos);
        EXPECT_EQ(markup->QuerySelector("b"), nullptr);
        EXPECT_EQ(accessMark(rowByCallback(6)), "X");
        EXPECT_EQ(accessMark(rowByCallback(7)), "X");

        bar_.updateLabels({"ファイル", "編集"},
                          {"lfs_plugins.file_menu.FileMenu", "lfs_plugins.edit_menu.EditMenu"});
        context_->Update();
        auto* file = el("menu-items")->GetChild(0);
        ASSERT_NE(file, nullptr);
        EXPECT_EQ(textContent(file), "ファイル(F)");
        Rml::ElementList marks;
        file->GetElementsByClassName(marks, "access-key");
        ASSERT_EQ(marks.size(), 1u);
        EXPECT_EQ(textContent(marks.front()), "F");
    }

    TEST_F(MenuBarTitleTest, TheBrandLogoIsTheHelpMenu) {
        bar_.updateLabels({"Help", "File", "Edit"},
                          {"lfs_plugins.help_menu.HelpMenu", "lfs_plugins.file_menu.FileMenu",
                           "lfs_plugins.edit_menu.EditMenu"});
        context_->Update();
        auto* const logo = el("menu-items")->GetChild(0);
        ASSERT_NE(logo, nullptr);
        EXPECT_TRUE(logo->IsClassSet("logo"));
        auto* const image = logo->GetChild(0);
        EXPECT_EQ(image->GetTagName(), "img");
        // The logo is a full menu box: it holds the whole image and ends before File starts.
        const auto box = bounds(logo);
        const auto icon = bounds(image);
        EXPECT_FLOAT_EQ(icon.right - icon.left, 20);
        EXPECT_GT(icon.left, box.left);
        EXPECT_LT(icon.right, box.right);
        EXPECT_GE(bounds(el("menu-items")->GetChild(1)).left, box.right);
        EXPECT_EQ(textContent(el("menu-items")), "FileEdit");
        EXPECT_FALSE(RmlMenuBarTestAccess::letterKey(bar_, SDLK_H, SDL_KMOD_LALT));
        EXPECT_FALSE(bar_.isOpen());
    }

    TEST_F(MenuBarTitleTest, TheBrandLogoOpensLikeEveryOtherMenu) {
        bar_.updateLabels({"Help", "File", "Edit"},
                          {"lfs_plugins.help_menu.HelpMenu", "lfs_plugins.file_menu.FileMenu",
                           "lfs_plugins.edit_menu.EditMenu"});
        context_->Update();
        const auto center = [](const Rect& r) { return std::pair{(r.left + r.right) / 2, (r.top + r.bottom) / 2}; };
        const auto logo = bounds(el("menu-items")->GetChild(0));
        const auto file = bounds(el("menu-items")->GetChild(1));
        const auto [logo_x, logo_y] = center(logo);
        const auto [file_x, file_y] = center(file);

        RmlMenuBarTestAccess::click(bar_, logo_x, logo_y);
        ASSERT_TRUE(bar_.isOpen());
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 0);
        EXPECT_EQ(RmlMenuBarTestAccess::activeLabel(bar_), 0);
        EXPECT_NEAR(bounds(el("dropdown-popup")).left, logo.left, 0.5f);
        EXPECT_NEAR(bounds(el("dropdown-popup")).top, logo.bottom, 0.5f);

        RmlMenuBarTestAccess::hover(bar_, file_x, file_y, true);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 1);
        RmlMenuBarTestAccess::hover(bar_, logo_x, logo_y, true);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 0);
        RmlMenuBarTestAccess::click(bar_, logo_x, logo_y);
        EXPECT_FALSE(bar_.isOpen());

        RmlMenuBarTestAccess::click(bar_, file_x, file_y);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_LEFT);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 0);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_RIGHT);
        EXPECT_EQ(RmlMenuBarTestAccess::menuIndex(bar_), 1);
        RmlMenuBarTestAccess::key(bar_, SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(bar_.isOpen());
    }

    TEST(MenuAccessKeyLocales, LettersAreUniqueAndNotAltBound) {
        const auto locales = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales";
        const auto profile = lfs::vis::input::InputBindings::createDefaultProfile();
        const char* ids[] = {"file", "edit", "select", "tools", "view"};
        for (const auto& entry : std::filesystem::directory_iterator(locales)) {
            if (entry.path().extension() != ".json")
                continue;
            const auto json = nlohmann::json::parse(std::ifstream(entry.path()));
            std::string seen;
            for (const char* id : ids) {
                const auto key = std::string("access_key.") + id;
                ASSERT_TRUE(json.contains("menu") && json["menu"].contains(key)) << entry.path().filename() << " " << key;
                const auto value = json["menu"][key].get<std::string>();
                ASSERT_EQ(value.size(), 1u) << entry.path().filename() << " " << value;
                const char letter = value[0];
                ASSERT_TRUE(letter >= 'A' && letter <= 'Z') << entry.path().filename() << " " << value;
                ASSERT_EQ(seen.find(letter), std::string::npos) << entry.path().filename() << " " << letter;
                seen.push_back(letter);
                constexpr int kModifierMask = lfs::vis::input::MODIFIER_SHIFT | lfs::vis::input::MODIFIER_CTRL |
                                              lfs::vis::input::MODIFIER_ALT | lfs::vis::input::MODIFIER_SUPER;
                for (const auto& binding : profile.bindings) {
                    const auto* trigger = std::get_if<lfs::vis::input::KeyTrigger>(&binding.trigger);
                    if (trigger && trigger->key == static_cast<int>(letter) &&
                        (trigger->modifiers & kModifierMask) == lfs::vis::input::MODIFIER_ALT)
                        FAIL() << entry.path().filename().string() << " Alt+" << letter;
                }
            }
        }
    }
} // namespace
