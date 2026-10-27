/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Keep this fixture independent of transform, LSP and project-archive tests so
// the focused sidebar target only needs symbols exported by the GUI DLL.
#include "core/event_bridge/event_bridge.hpp"
#include "core/services.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rml_right_panel.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "input/input_controller.hpp"
#include "visualizer_impl.hpp"
#include "window/window_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL.h>
#include <gtest/gtest.h>

namespace lfs::vis {
    namespace {
        class NullRenderInterface final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                        Rml::Span<const int>) override {
                return Rml::CompiledGeometryHandle(1);
            }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f,
                                Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = Rml::Vector2i(1, 1);
                return Rml::TextureHandle(1);
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                               Rml::Vector2i) override {
                return Rml::TextureHandle(1);
            }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };

    } // namespace
    class SidebarInputDispatchTest : public ::testing::Test {
    protected:
        void SetUp() override {
            persistence_enabled_ = input::InputBindings::isPersistenceEnabled();
            input::InputBindings::setPersistenceEnabled(false);
            ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO));
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.safe_mode = true;
            viewer_ = std::make_unique<VisualizerImpl>(options);
            window_ = viewer_->getWindowManager();
            gui_ = viewer_->getGuiManager();
            lfs::event::EventBridge::instance().clear_all();
            window_->window_ = SDL_CreateWindow("Input dispatch", 400, 300, SDL_WINDOW_HIDDEN);
            ASSERT_NE(window_->window_, nullptr);
            ASSERT_TRUE(manager().initWithRenderInterface(window_->window_, 1.f,
                                                          std::make_unique<NullRenderInterface>(), nullptr));
            gui_->startup_overlay_.dismiss();
            controller_ = std::make_unique<InputController>(window_->window_, viewport_);
            controller_->initialize();
            window_->setInputController(controller_.get());
            context_ = manager().createContext("dispatch-field", 400, 300);
            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>body { width:400px; height:300px; font-family:Inter; font-size:14px; }"
                "input { position:absolute; left:10px; top:10px; width:180px; height:40px; }"
                "</style></head><body><input id='field' type='text'/></body></rml>");
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();
            field_ = dynamic_cast<Rml::ElementFormControlInput*>(document_->GetElementById("field"));
            ASSERT_NE(field_, nullptr);
            revert_.bind(field_);
            manager().trackContextFrame(context_, 0, 0);
            manager().routeInput(context_, {}, [this](const gui::PanelInputState& input) {
                for (const auto& button : input.mouse_button_events) {
                    context_->ProcessMouseMove(button.x, button.y, 0);
                    if (button.down)
                        context_->ProcessMouseButtonDown(button.button, 0);
                    else
                        context_->ProcessMouseButtonUp(button.button, 0);
                }
                for (const auto& event : input.input_events) {
                    if (event.kind == FrameInputEventKind::KeyDown && event.scancode == SDL_SCANCODE_ESCAPE &&
                        gui::rml_input::cancelFocusedElement(*context_)) {
                        event.dispatch->consumed = true;
                        continue;
                    }
                    gui::rml_input::processKeyboardEvent(*context_, event, manager().getTextInputHandler());
                }
            });
        }
        void TearDown() override {
            manager().vulkan_render_interface_ = nullptr;
            gui::PanelRegistry::instance().unregister_panel("test.input_float");
            revert_.clear();
            gui_->startup_overlay_.shutdown();
            controller_.reset();
            viewer_.reset();
            Rml::SetSystemInterface(nullptr);
            Rml::SetRenderInterface(nullptr);
            services().clear();
            gui::guiFocusState().reset();
            input::InputBindings::setPersistenceEnabled(persistence_enabled_);
        }
        gui::RmlUIManager& manager() { return gui_->rmlui_manager_; }
        void setEventWatchActive(bool active) { window_->watching_event_ = active; }
        void setKeyboardContext(Rml::Context* context, bool external_enabled) {
            manager().keyboard_context_ = context;
            manager().native_windows_enabled_ = external_enabled;
        }
        void attachInputHost(gui::RmlPanelHost& host, int width, int height, const char* markup) {
            // Input uses the real host and null geometry renderer, without a GPU frame.
            manager().vulkan_render_interface_ = &input_renderer_;
            host.rml_context_ = manager().createContext(host.context_name_, width, height);
            std::string styled_markup = markup;
            styled_markup.insert(styled_markup.find("<style>") + 7, gui::rml_theme::getComponentsRCSS());
            host.document_ = host.rml_context_->LoadDocumentFromMemory(styled_markup);
            ASSERT_NE(host.document_, nullptr);
            host.document_->Show();
            host.rml_context_->Update();
            host.last_fbo_w_ = width;
            host.last_fbo_h_ = height;
        }
        void forwardHost(gui::RmlPanelHost& host, float x, float y, gui::PanelInputState input = {}) {
            host.setInput(&input);
            host.forwardInput(x, y);
            host.setInput(nullptr);
        }
        gui::RmlRightPanel& sidebar() { return gui_->rml_right_panel_; }
        void setupSidebar() {
            auto& panel = sidebar();
            panel.init(&manager());
            panel.rml_context_->SetDimensions({400, 300});
            panel.tab_bar_el_->SetProperty("top", "0px");
            panel.tab_bar_el_->SetProperty("height", "300px");
            std::vector<gui::TabSnapshot> tabs;
            for (int i = 0; i < 14; ++i) {
                const auto id = "test.sidebar." + std::to_string(i);
                tabs.push_back({.id = id, .label = "Temporary Tab " + std::to_string(i), .dom_id = "native-tab-" + std::to_string(i)});
            }
            panel.syncTabData(tabs, tabs.front().id);
            panel.updateTabs();
            panel.on_tab_changed = [this](const auto& id) { sidebar_picks_.push_back(id); };
            gui::PanelInputState input;
            input.mouse_x = 120;
            input.mouse_y = 200;
            gui::RightPanelLayout layout;
            layout.size = {400, 300};
            layout.splitter_h = 0;
            panel.processInput(layout, input);
        }
        void settleSidebar() { sidebar().updateTabs(); }
        void focusSidebar() {
            auto& panel = sidebar();
            ASSERT_TRUE(panel.document_->GetElementById("native-tab-0")->Focus());
            setKeyboardContext(panel.rml_context_, false);
        }
        gui::RmlUIManager::InputDispatchResult sidebarWheel(float x, float y, float amount, SDL_Keymod mods) {
            SDL_SetModState(mods);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_WHEEL;
            event.wheel.windowID = SDL_GetWindowID(window_->getWindow());
            event.wheel.mouse_x = x;
            event.wheel.mouse_y = y;
            event.wheel.y = amount;
            const auto result = manager().dispatchInputEvent(event);
            SDL_SetModState(SDL_KMOD_NONE);
            return result;
        }
        gui::RmlUIManager::InputDispatchResult sidebarKey(SDL_Scancode code, SDL_Keymod mods = SDL_KMOD_NONE,
                                                          bool repeat = false) {
            SDL_Event event{};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.windowID = SDL_GetWindowID(window_->getWindow());
            event.key.scancode = code;
            event.key.mod = mods;
            event.key.repeat = repeat;
            return manager().dispatchInputEvent(event);
        }
        std::vector<std::string> sidebar_picks_;
        void registerInputFloat(float x = 100.f) {
            class Panel final : public gui::IPanel {
                void draw(const gui::PanelDrawContext&) override {}
                gui::PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }
            };
            gui::PanelInfo info;
            info.id = "test.input_float";
            info.label = "Input float";
            info.space = gui::PanelSpace::Floating;
            info.panel = std::make_shared<Panel>();
            ASSERT_TRUE(gui::PanelRegistry::instance().register_panel(std::move(info)));
            gui::PanelRegistry::instance().apply_project_state({{
                .id = "test.input_float",
                .space = gui::PanelSpace::Floating,
                .enabled = true,
                .float_x = x,
                .float_y = 60.f,
                .float_user_height = 100.f,
                .float_last_bounds_valid = true,
                .float_last_x = x,
                .float_last_y = 60.f,
                .float_last_w = 180.f,
                .float_last_h = 100.f,
                .float_auto_center = false,
            }});
            ASSERT_TRUE(gui::PanelRegistry::instance().isPositionOverFloatingPanel(x + 40, 100));
        }
        RenderInterface_VK input_renderer_;
        void dispatch(SDL_Event event) {
            event.common.timestamp = ++timestamp_;
            event.key.windowID = SDL_GetWindowID(window_->window_);
            window_->dispatchQueuedEvent(event);
        }
        void key(SDL_Scancode code, SDL_Keymod mods = SDL_KMOD_NONE, bool repeat = false) {
            SDL_Event event{};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.down = true;
            event.key.scancode = code;
            event.key.mod = mods;
            event.key.repeat = repeat;
            dispatch(event);
        }
        Viewport viewport_{400, 300};
        std::unique_ptr<VisualizerImpl> viewer_;
        std::unique_ptr<InputController> controller_;
        WindowManager* window_ = nullptr;
        gui::GuiManager* gui_ = nullptr;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::ElementFormControlInput* field_ = nullptr;
        uint64_t timestamp_ = 0;
        gui::rml_input::TextInputEscapeRevertController revert_;
        bool persistence_enabled_ = true;
    };
} // namespace lfs::vis
namespace lfs::vis {
    TEST_F(SidebarInputDispatchTest, GamepadDiscoveryLifetimeFollowsVisibleDiagnostics) {
        window_->setInputController(nullptr);
        EXPECT_FALSE(window_->gamepads().snapshot().available);
        window_->setGamepadDiagnosticsActive(true);
        ASSERT_TRUE(window_->gamepads().snapshot().available);
        window_->pollEvents();
        EXPECT_TRUE(window_->gamepads().snapshot().available);
        window_->setGamepadDiagnosticsActive(false);
        EXPECT_FALSE(window_->gamepads().snapshot().available);
        window_->pollEvents();
        EXPECT_FALSE(window_->gamepads().snapshot().available);
    }

    TEST_F(SidebarInputDispatchTest, GamepadHandleChangesWaitUntilTheEventWatchReturns) {
        window_->setInputController(nullptr);
        setEventWatchActive(true);
        window_->setGamepadDiagnosticsActive(true);
        EXPECT_FALSE(window_->gamepads().snapshot().available);
        setEventWatchActive(false);
        window_->pollEvents();
        ASSERT_TRUE(window_->gamepads().snapshot().available);
        setEventWatchActive(true);
        window_->setGamepadDiagnosticsActive(false);
        EXPECT_TRUE(window_->gamepads().snapshot().available);
        setEventWatchActive(false);
        window_->pollEvents();
        EXPECT_FALSE(window_->gamepads().snapshot().available);
    }

    TEST_F(SidebarInputDispatchTest, WindowPumpPreservesShortTriggerCrossingsAcrossDiagnosticReads) {
        window_->setInputController(nullptr);
        window_->setGamepadDiagnosticsActive(true);
        SDL_VirtualJoystickDesc desc;
        SDL_INIT_INTERFACE(&desc);
        desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
        desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
        desc.nbuttons = SDL_GAMEPAD_BUTTON_COUNT;
        desc.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
        desc.button_mask = (1u << SDL_GAMEPAD_BUTTON_COUNT) - 1;
        desc.name = "LichtFeld trigger dispatch test";
        const auto id = SDL_AttachVirtualJoystick(&desc);
        ASSERT_NE(id, 0u);
        window_->pollEvents();
        ASSERT_TRUE(window_->gamepads().selectNavigation(id));
        ASSERT_TRUE(window_->gamepads().select(id));
        SDL_Event event{};
        event.type = SDL_EVENT_GAMEPAD_AXIS_MOTION;
        event.gaxis.which = id;
        event.gaxis.axis = SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
        event.gaxis.value = 24000;
        ASSERT_TRUE(SDL_PushEvent(&event));
        event.gaxis.value = 0;
        ASSERT_TRUE(SDL_PushEvent(&event));
        window_->pollEvents();
        window_->pollEvents();
        EXPECT_FLOAT_EQ(window_->gamepads().navigationSample().axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER], 0.f);
        EXPECT_TRUE(window_->gamepads().navigationSample().trigger_pressed[1]);
        EXPECT_FLOAT_EQ(window_->gamepads().takeDiagnosticSnapshot().sample.axes[SDL_GAMEPAD_AXIS_RIGHT_TRIGGER], 0.f);
        (void)window_->gamepads().takeDiagnosticSnapshot();
        EXPECT_TRUE(window_->gamepads().navigationSample().trigger_pressed[1]);
        EXPECT_TRUE(SDL_DetachVirtualJoystick(id));
        window_->setGamepadDiagnosticsActive(false);
    }

    TEST_F(SidebarInputDispatchTest, SidebarCtrlWheelRemainsAvailableToScrollableContent) {
        gui::RmlPanelHost host(&manager(), "sidebar-content", "");
        attachInputHost(host, 400, 300,
                        "<rml><head><style>body { width:400px; height:300px; }"
                        "#scroll { height:250px; overflow:auto; } #long { height:2000px; }</style></head>"
                        "<body><div id='scroll'><div id='long'>Content</div></div></body></rml>");
        forwardHost(host, 40, 0);
        setupSidebar();
        auto* scroll = host.getDocument()->GetElementById("scroll");
        ASSERT_EQ(scroll->GetScrollTop(), 0);
        sidebarWheel(120, 200, -1, SDL_KMOD_CTRL);
        EXPECT_TRUE(sidebar_picks_.empty());
        EXPECT_GT(scroll->GetScrollTop(), 0);
        const float before = scroll->GetScrollTop();
        sidebarWheel(20, 200, -1, SDL_KMOD_CTRL);
        settleSidebar();
        EXPECT_GT(sidebar().tabStripScroll(), 0);
        EXPECT_EQ(scroll->GetScrollTop(), before);
        EXPECT_TRUE(sidebar_picks_.empty());
        sidebarWheel(120, 200, -1, SDL_KMOD_NONE);
        EXPECT_GT(scroll->GetScrollTop(), 0);
    }

    TEST_F(SidebarInputDispatchTest, SidebarCtrlTabWorksWithTextFocusAndWithoutFocus) {
        setupSidebar();
        ASSERT_TRUE(field_->Focus());
        setKeyboardContext(context_, false);
        SDL_Event event{};
        event.type = SDL_EVENT_KEY_DOWN;
        event.key.windowID = SDL_GetWindowID(window_->getWindow());
        event.key.scancode = SDL_SCANCODE_TAB;
        event.key.mod = SDL_KMOD_CTRL;
        EXPECT_TRUE(manager().dispatchInputEvent(event).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 1);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.1");
        event.type = SDL_EVENT_KEY_UP;
        EXPECT_TRUE(manager().dispatchInputEvent(event).consumed);
        field_->Blur();
        setKeyboardContext(nullptr, false);
        event.type = SDL_EVENT_KEY_DOWN;
        EXPECT_TRUE(manager().dispatchInputEvent(event).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 2);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.2");
    }

    TEST_F(SidebarInputDispatchTest, SidebarTypingKeysStayWithAFocusedTextField) {
        setupSidebar();
        auto& bindings = controller_->getBindings();
        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_N});
        ASSERT_TRUE(field_->Focus());
        setKeyboardContext(context_, false);
        sidebarKey(SDL_SCANCODE_N);
        EXPECT_TRUE(sidebar_picks_.empty());
        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_N, input::MODIFIER_CTRL | input::MODIFIER_ALT});
        sidebarKey(SDL_SCANCODE_N, static_cast<SDL_Keymod>(SDL_KMOD_CTRL | SDL_KMOD_ALT));
        EXPECT_TRUE(sidebar_picks_.empty());
        field_->Blur();
        setKeyboardContext(nullptr, false);
        gui::guiFocusState().reset(); // the next frame starts without text focus
        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_N});
        EXPECT_TRUE(sidebarKey(SDL_SCANCODE_N).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 1);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.1");
    }

    TEST_F(SidebarInputDispatchTest, SidebarKeymapRebindingClearingAndCaptureAreRespected) {
        setupSidebar();
        auto& bindings = controller_->getBindings();
        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_F8, input::MODIFIER_ALT});
        bindings.setBinding(input::ToolMode::GLOBAL, input::Action::PREVIOUS_SIDEBAR_CATEGORY,
                            input::KeyTrigger{input::KEY_F9});
        sidebarKey(SDL_SCANCODE_TAB, SDL_KMOD_CTRL);
        EXPECT_TRUE(sidebar_picks_.empty());
        ASSERT_TRUE(sidebarKey(SDL_SCANCODE_F8, SDL_KMOD_ALT).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 1);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.1");
        ASSERT_TRUE(sidebarKey(SDL_SCANCODE_F9).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 2);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.0");
        ASSERT_TRUE(sidebarKey(SDL_SCANCODE_F9).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 3);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.13");
        const auto before = sidebar_picks_.size();
        sidebarKey(SDL_SCANCODE_F8, SDL_KMOD_NONE);
        sidebarKey(SDL_SCANCODE_F8, SDL_KMOD_ALT, true);
        EXPECT_EQ(sidebar_picks_.size(), before);
        bindings.clearBinding(input::ToolMode::GLOBAL, input::Action::NEXT_SIDEBAR_CATEGORY);
        sidebarKey(SDL_SCANCODE_F8, SDL_KMOD_ALT);
        EXPECT_EQ(sidebar_picks_.size(), before);
        bindings.startCapture(input::ToolMode::GLOBAL, input::Action::PREVIOUS_SIDEBAR_CATEGORY);
        sidebarKey(SDL_SCANCODE_F9);
        EXPECT_EQ(sidebar_picks_.size(), before);
        bindings.cancelCapture();
    }

    TEST_F(SidebarInputDispatchTest, SidebarConsecutiveKeymapCyclesAdvanceWithoutRendering) {
        setupSidebar();
        EXPECT_TRUE(sidebarKey(SDL_SCANCODE_TAB, SDL_KMOD_CTRL).consumed);
        EXPECT_TRUE(sidebarKey(SDL_SCANCODE_TAB, SDL_KMOD_CTRL).consumed);
        ASSERT_EQ(sidebar_picks_.size(), 2);
        EXPECT_EQ(sidebar_picks_[0], "test.sidebar.1");
        EXPECT_EQ(sidebar_picks_[1], "test.sidebar.2");
        EXPECT_TRUE(sidebarKey(SDL_SCANCODE_TAB, SDL_KMOD_CTRL | SDL_KMOD_SHIFT).consumed);
        EXPECT_EQ(sidebar_picks_.back(), "test.sidebar.1");
    }

    TEST_F(SidebarInputDispatchTest, SidebarFloatingOcclusionBlocksNewGesturesButKeepsOwnedDrag) {
        setupSidebar();
        registerInputFloat(0);
        sidebarWheel(20, 100, -1, SDL_KMOD_CTRL);
        settleSidebar();
        EXPECT_FLOAT_EQ(sidebar().tabStripScroll(), 0);
        EXPECT_TRUE(sidebar_picks_.empty());
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.button = SDL_BUTTON_MIDDLE;
        event.button.x = 20;
        event.button.y = 100;
        dispatch(event);
        EXPECT_FALSE(sidebar().isResizeInteractionActive());
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        dispatch(event);
        gui::PanelRegistry::instance().unregister_panel("test.input_float");
        registerInputFloat();
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.y = 200;
        dispatch(event);
        ASSERT_TRUE(sidebar().isResizeInteractionActive());
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.motion.x = 140;
        event.motion.y = 100;
        dispatch(event);
        settleSidebar();
        EXPECT_GT(sidebar().tabStripScroll(), 0);
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.button = SDL_BUTTON_MIDDLE;
        event.button.x = 140;
        event.button.y = 100;
        dispatch(event);
        EXPECT_FALSE(sidebar().isResizeInteractionActive());
    }

    TEST_F(SidebarInputDispatchTest, SidebarKeyboardOverflowThenWheelKeepsTheCurrentOffset) {
        setupSidebar();
        focusSidebar();
        for (int i = 0; i < 13; ++i)
            key(SDL_SCANCODE_DOWN);
        settleSidebar();
        const float before = sidebar().tabStripScroll();
        ASSERT_GT(before, 100);
        sidebarWheel(20, 200, 1, SDL_KMOD_NONE);
        settleSidebar();
        EXPECT_NEAR(sidebar().tabStripScroll(), before - 30, 0.5f);
    }
} // namespace lfs::vis
