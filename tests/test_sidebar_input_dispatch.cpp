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
#include "gui/vulkan_ui_texture.hpp"
#include "input/input_controller.hpp"
#include "python/python_runtime.hpp"
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
            // These entries exercise desktop input/stacking with real SDL windows,
            // without a GPU renderer. Remove them before normal context teardown.
            for (auto* context : desktop_contexts_) {
                manager().context_windows_.erase(context);
                manager().native_windows_.erase(context);
            }
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
        void mapContextWindow(Rml::Context* context, SDL_Window* window) {
            manager().context_windows_[context] = window;
        }
        bool applyPin(SDL_Window* window, bool pinned) {
            return manager().applyNativeWindowPin(window, pinned);
        }
        std::pair<Rml::Context*, SDL_Window*> attachDesktopContext(const char* name, const char* panel_id) {
            auto* context = manager().createContext(name, 200, 100);
            if (!context)
                return {nullptr, nullptr};
            auto* window = SDL_CreateWindow("Desktop pin regression", 200, 100, SDL_WINDOW_HIDDEN);
            if (!window)
                return {context, nullptr};
            auto native = std::make_unique<gui::RmlUIManager::NativeWindow>();
            native->window = window;
            native->name = name;
            native->graphics = std::make_unique<VulkanContext>();
            native->graphics->setSceneRuntimeOwnership(false);
            manager().native_windows_[context] = std::move(native);
            mapContextWindow(context, window);
            SDL_SetStringProperty(SDL_GetWindowProperties(window), "lfs.panel_id", panel_id);
            desktop_contexts_.push_back(context);
            return {context, window};
        }
        VulkanContext* desktopGraphics(Rml::Context* context) {
            return manager().native_windows_.at(context)->graphics.get();
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
        bool hostOwnsPointer(const gui::RmlPanelHost& host) { return host.mouse_captured_[0]; }
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
        std::vector<Rml::Context*> desktop_contexts_;
        void registerInputFloat(float x = 100.f, std::function<void()> suspend = {}) {
            class Panel final : public gui::IPanel {
            public:
                explicit Panel(std::function<void()> suspend) : suspend_(std::move(suspend)) {}
                void suspendInput() override {
                    if (suspend_)
                        suspend_();
                }
                void draw(const gui::PanelDrawContext&) override {}
                gui::PanelRenderCapabilities renderCapabilities() const override { return {.direct = true}; }

            private:
                std::function<void()> suspend_;
            };
            gui::PanelInfo info;
            info.id = "test.input_float";
            info.label = "Input float";
            info.space = gui::PanelSpace::Floating;
            info.panel = std::make_shared<Panel>(std::move(suspend));
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
    TEST_F(SidebarInputDispatchTest, DesktopPinChangesOnlyTopmostState) {
        std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
            SDL_CreateWindow("Pin state regression", 200, 100, SDL_WINDOW_HIDDEN | SDL_WINDOW_RESIZABLE),
            SDL_DestroyWindow);
        ASSERT_NE(window, nullptr);
        const auto flags = SDL_GetWindowFlags(window.get());
        const bool pinned = applyPin(window.get(), true);
        EXPECT_EQ(pinned, bool(SDL_GetWindowFlags(window.get()) & SDL_WINDOW_ALWAYS_ON_TOP));
        EXPECT_EQ(SDL_GetWindowFlags(window.get()) & ~SDL_WINDOW_ALWAYS_ON_TOP, flags);
        EXPECT_EQ(applyPin(window.get(), true), pinned);
        ASSERT_TRUE(applyPin(window.get(), false));
        EXPECT_EQ(SDL_GetWindowFlags(window.get()), flags);
        int width = 0, height = 0;
        ASSERT_TRUE(SDL_GetWindowSize(window.get(), &width, &height));
        EXPECT_EQ(width, 200);
        EXPECT_EQ(height, 100);
    }

    TEST_F(SidebarInputDispatchTest, MainExclusiveInputBlocksDesktopPanelUntilDismissed) {
        auto* external_window = SDL_CreateWindow("Modal input regression", 200, 100, SDL_WINDOW_HIDDEN);
        ASSERT_NE(external_window, nullptr);
        auto* external_context = manager().createContext("external-modal-input", 200, 100);
        ASSERT_NE(external_context, nullptr);
        mapContextWindow(external_context, external_window);
        auto* document = external_context->LoadDocumentFromMemory(
            "<rml><head><style>body {width:200px;height:100px;} input {width:180px;height:80px;}</style></head>"
            "<body><input id='external-field' type='text'/></body></rml>");
        ASSERT_NE(document, nullptr);
        document->Show();
        external_context->Update();
        auto* field = document->GetElementById("external-field");
        ASSERT_NE(field, nullptr);
        int external_events = 0;
        int passive_updates = 0;
        const auto handler = [&](const gui::PanelInputState& input) {
            external_events += static_cast<int>(input.input_events.size()) + (input.mouse_wheel != 0.0f ? 1 : 0);
            ++passive_updates;
        };
        manager().activateInput(external_context, handler, false);
        manager().trackContextFrame(external_context, 0, 0);
        const auto send = [&](Uint32 type) {
            SDL_Event event{};
            event.type = type;
            if (type == SDL_EVENT_KEY_DOWN) {
                event.key.windowID = SDL_GetWindowID(external_window);
                event.key.scancode = SDL_SCANCODE_A;
            } else if (type == SDL_EVENT_MOUSE_WHEEL) {
                event.wheel.windowID = SDL_GetWindowID(external_window);
                event.wheel.y = 1;
                event.wheel.mouse_x = 20;
                event.wheel.mouse_y = 20;
            } else {
                event.button.windowID = SDL_GetWindowID(external_window);
                event.button.button = SDL_BUTTON_LEFT;
                event.button.x = 20;
                event.button.y = 20;
            }
            manager().dispatchInputEvent(event);
        };
        const auto send_all = [&] {
            send(SDL_EVENT_MOUSE_BUTTON_DOWN);
            send(SDL_EVENT_MOUSE_BUTTON_UP);
            send(SDL_EVENT_MOUSE_WHEEL);
            send(SDL_EVENT_KEY_DOWN);
        };
        EXPECT_TRUE(field->Focus());
        setKeyboardContext(external_context, true);
        send_all();
        EXPECT_GE(external_events, 4);
        manager().activateInput(context_, [](const gui::PanelInputState&) {});
        EXPECT_TRUE(field->Focus());
        setKeyboardContext(external_context, true);
        const int before_popup = external_events;
        send_all();
        EXPECT_GE(external_events - before_popup, 4);
        manager().activateInput(context_, [](const gui::PanelInputState&) {}, true, {}, true);
        const int before_modal = external_events;
        const int before_passive = passive_updates;
        send_all();
        manager().routeInput(external_context, {}, handler);
        EXPECT_EQ(external_events, before_modal);
        EXPECT_EQ(passive_updates, before_passive);
        manager().deactivateInput(context_);
        manager().routeInput(external_context, {}, handler);
        EXPECT_TRUE(field->Focus());
        setKeyboardContext(external_context, true);
        send_all();
        EXPECT_GE(external_events - before_modal, 4);
        manager().destroyContext("external-modal-input");
        SDL_DestroyWindow(external_window);
    }

    TEST_F(SidebarInputDispatchTest, DesktopPinSuspensionRestoresChoiceAndRejectsUnsupportedBackends) {
        VulkanContext main_textures;
        main_textures.setSceneRuntimeOwnership(false);
        const gui::ScopedVulkanUiTextureContext main_texture_scope(&main_textures);
        auto suspensions = std::make_shared<int>(0);
        registerInputFloat(100.f, [this, suspensions] {
            ++*suspensions;
            auto* context = manager().getContext("pin-state");
            ASSERT_NE(context, nullptr);
            EXPECT_EQ(gui::getVulkanUiTextureContext(), desktopGraphics(context));
            manager().deactivateInput(context);
        });
        const auto [context, window] = attachDesktopContext("pin-state", "test.input_float");
        ASSERT_NE(context, nullptr);
        ASSERT_NE(window, nullptr);
        const auto topmost = [&] { return bool(SDL_GetWindowFlags(window) & SDL_WINDOW_ALWAYS_ON_TOP); };
        manager().activateInput(context, [](const gui::PanelInputState&) {}, false);
        const auto generation = lfs::python::redraw_request_generation();
        const bool supported = manager().setNativeWindowPinned("test.input_float", true);
        EXPECT_EQ(supported, topmost());
        EXPECT_EQ(manager().nativeWindowPinned("test.input_float"), supported);
        EXPECT_EQ(lfs::python::redraw_request_generation() > generation, supported);
        const auto unchanged = lfs::python::redraw_request_generation();
        EXPECT_EQ(manager().setNativeWindowPinned("test.input_float", true), supported);
        EXPECT_EQ(lfs::python::redraw_request_generation(), unchanged);
        manager().activateInput(context_, [](const gui::PanelInputState&) {}, true, {}, true);
        EXPECT_FALSE(topmost());
        EXPECT_EQ(*suspensions, 1);
        ASSERT_TRUE(manager().setNativeWindowPinned("test.input_float", false));
        ASSERT_TRUE(manager().setNativeWindowPinned("test.input_float", true));
        EXPECT_EQ(manager().nativeWindowPinned("test.input_float"), true);
        EXPECT_FALSE(topmost());
        auto* second_modal = manager().createContext("second-pin-modal", 200, 100);
        ASSERT_NE(second_modal, nullptr);
        manager().activateInput(second_modal, [](const gui::PanelInputState&) {}, true, {}, true);
        manager().deactivateInput(context_);
        EXPECT_FALSE(topmost());
        manager().destroyContext("second-pin-modal");
        EXPECT_EQ(topmost(), supported);
        EXPECT_EQ(*suspensions, 1);
        EXPECT_EQ(manager().nativeWindowPinned("test.input_float"), supported);
        manager().deactivateInput(context);
        manager().activateInput(context_, [](const gui::PanelInputState&) {}, true, {}, true);
        EXPECT_EQ(*suspensions, 2);
        EXPECT_EQ(gui::getVulkanUiTextureContext(), &main_textures);
        manager().deactivateInput(context_);
        EXPECT_EQ(topmost(), supported);
        EXPECT_TRUE(manager().setNativeWindowPinned("test.input_float", false));
        EXPECT_FALSE(topmost());
    }

    TEST_F(SidebarInputDispatchTest, InternalPanelsCannotBePinned) {
        registerInputFloat();
        ASSERT_TRUE(gui::PanelRegistry::instance().get_panel("test.input_float").has_value());
        const auto [context, window] = attachDesktopContext("other-desktop", "test.other_desktop");
        ASSERT_NE(context, nullptr);
        ASSERT_NE(window, nullptr);
        EXPECT_FALSE(manager().nativeWindowPinned("test.input_float").has_value());
        EXPECT_FALSE(manager().setNativeWindowPinned("test.input_float", true));
        EXPECT_FALSE(manager().nativeWindowPinned("unknown").has_value());
        EXPECT_FALSE(manager().setNativeWindowPinned("unknown", true));
    }

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

    TEST_F(SidebarInputDispatchTest, CollapsingSidebarReleasesOwnedPointerDragAndInput) {
        setupSidebar();
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.button = SDL_BUTTON_MIDDLE;
        event.button.x = 20.0f;
        event.button.y = 200.0f;
        dispatch(event);
        ASSERT_TRUE(sidebar().isResizeInteractionActive());
        sidebar().suspendInput();
        EXPECT_FALSE(sidebar().isResizeInteractionActive());
        EXPECT_FALSE(sidebar().wantsInput());
        EXPECT_FALSE(sidebar().wantsKeyboard());
        EXPECT_EQ(sidebar().getCursorRequest(), gui::CursorRequest::None);
    }

    TEST_F(SidebarInputDispatchTest, SuspendingContentHostCancelsPointerWithoutClosingDocument) {
        for (const bool minimized : {false, true}) {
            gui::RmlPanelHost host(&manager(), minimized ? "minimized-content" : "suspended-content", "");
            attachInputHost(host, 400, 300,
                            "<rml><head><style>body {width:400px;height:300px;}"
                            "#drag {width:200px;height:100px;drag:drag;}</style></head>"
                            "<body><div id='drag'>Drag</div></body></rml>");
            struct PayloadListener final : Rml::EventListener {
                explicit PayloadListener(gui::RmlUIManager& owner) : owner(owner) {}
                gui::RmlUIManager& owner;
                uint64_t token = 0;
                int cancelled = 0;
                void ProcessEvent(Rml::Event& event) override {
                    if (event.GetType() == "dragstart")
                        token = owner.beginDragPayload("project", "test-dock-project");
                    else if (event.GetType() == "dragend") {
                        cancelled += event.GetParameter("cancelled", false);
                        owner.endDragPayload(token);
                    }
                }
            } listener(manager());
            auto* drag = host.getDocument()->GetElementById("drag");
            drag->AddEventListener("dragstart", &listener);
            drag->AddEventListener("dragend", &listener);
            forwardHost(host, 0, 0);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = 40;
            event.button.y = 40;
            dispatch(event);
            EXPECT_TRUE(hostOwnsPointer(host));
            SDL_Event motion{};
            motion.type = SDL_EVENT_MOUSE_MOTION;
            motion.motion.x = 100;
            motion.motion.y = 60;
            dispatch(motion);
            EXPECT_NE(listener.token, 0);
            EXPECT_TRUE(manager().dragPayload().has_value());
            if (minimized) {
                host.setFloating(true);
                host.setWindowPresentation(gui::FloatingWindowState::Minimized);
            } else {
                host.suspendInput();
            }
            EXPECT_FALSE(manager().dragPayload().has_value());

            EXPECT_FALSE(hostOwnsPointer(host));
            EXPECT_TRUE(host.isDocumentLoaded());
            EXPECT_FALSE(host.wantsKeyboard());
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            dispatch(event);
            forwardHost(host, 0, 0);
            EXPECT_FALSE(hostOwnsPointer(host));
            EXPECT_EQ(listener.cancelled, 1);
            EXPECT_FALSE(manager().takeReleasedDragPayload().has_value());
            drag->RemoveEventListener("dragstart", &listener);
            drag->RemoveEventListener("dragend", &listener);
        }
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
