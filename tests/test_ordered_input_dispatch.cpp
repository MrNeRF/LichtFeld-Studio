/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "operator/operator_registry.hpp"
#include "visualizer_impl.hpp"
#include "window/window_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL.h>
#include <future>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

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

        class InputEventRecorder final : public Rml::EventListener {
        public:
            void ProcessEvent(Rml::Event& event) override {
                log_.push_back(event.GetTargetElement()->GetId() + ":" + event.GetType());
            }
            int countOf(const Rml::String& entry) const {
                return static_cast<int>(std::count(log_.begin(), log_.end(), entry));
            }
            Rml::String joined() const {
                Rml::String result;
                for (const auto& entry : log_)
                    result += entry + " ";
                return result;
            }

        private:
            std::vector<Rml::String> log_;
        };

    } // namespace

    class RmlPointerReplayTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            render_interface_ = new NullRenderInterface();
            ASSERT_TRUE(Rml::Initialise());
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
            delete render_interface_;
            render_interface_ = nullptr;
        }

        void SetUp() override {
            context_ = Rml::CreateContext("pointer_delivery", Rml::Vector2i(400, 300),
                                          render_interface_);
            ASSERT_NE(context_, nullptr) << "headless RmlUi context could not be created";

            document_ = context_->LoadDocumentFromMemory(
                "<rml><head><style>body { width:400px; height:300px; }</style></head><body/></rml>");
            ASSERT_NE(document_, nullptr);
            document_->Show();
            context_->Update();
        }

        void TearDown() override {
            if (context_)
                Rml::RemoveContext(context_->GetName());
            context_ = nullptr;
            document_ = nullptr;
        }

        static inline NullRenderInterface* render_interface_ = nullptr;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        InputEventRecorder recorder_;
    };

} // namespace lfs::vis

namespace lfs::vis {
    namespace {
        FrameInputBuffer orderedTypingFrame(const bool backspace) {
            FrameInputBuffer frame;
            SDL_Event event{};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = backspace ? "a" : "(";
            frame.processEvent(event);
            if (!backspace) {
                event.text.text = ")";
                frame.processEvent(event);
            }
            event = {};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.scancode = backspace ? SDL_SCANCODE_BACKSPACE : SDL_SCANCODE_RETURN;
            frame.processEvent(event);
            event.type = SDL_EVENT_KEY_UP;
            frame.processEvent(event);
            if (backspace) {
                event = {};
                event.type = SDL_EVENT_TEXT_INPUT;
                event.text.text = "b";
                frame.processEvent(event);
            }
            return frame;
        }
    } // namespace

    TEST_F(RmlPointerReplayTest, DocumentReceivesKeysWithoutTextFocus) {
        document_->SetId("keyboard-document");
        document_->AddEventListener("keydown", &recorder_);
        ASSERT_TRUE(document_->Focus());
        ASSERT_EQ(context_->GetFocusElement(), document_);
        const FrameInputEvent event{.kind = FrameInputEventKind::KeyDown, .scancode = SDL_SCANCODE_A};
        EXPECT_TRUE(gui::rml_input::processKeyboardEvent(*context_, event));
        EXPECT_EQ(recorder_.countOf("keyboard-document:keydown"), 1) << recorder_.joined();
        document_->RemoveEventListener("keydown", &recorder_);
    }

    TEST_F(RmlPointerReplayTest, TextFieldConsumesSdlTextAndKeysInArrivalOrder) {
        ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                          .string()));
        auto element = document_->CreateElement("input");
        element->SetProperty("font-family", "Inter");
        element->SetProperty("font-size", "14px");
        element->SetProperty("width", "200px");
        element->SetProperty("height", "24px");
        auto* field = dynamic_cast<Rml::ElementFormControl*>(element.get());
        ASSERT_NE(field, nullptr);
        struct ValueOnKey : Rml::EventListener {
            explicit ValueOnKey(Rml::ElementFormControl& value) : field(value) {}
            Rml::ElementFormControl& field;
            std::string before_edit;
            void ProcessEvent(Rml::Event& event) override {
                const auto key = event.GetParameter<int>("key_identifier", 0);
                if (key == Rml::Input::KI_RETURN || key == Rml::Input::KI_BACK)
                    before_edit = field.GetValue();
            }
        } listener(*field);
        field->AddEventListener("keydown", &listener);
        document_->AppendChild(std::move(element));
        context_->Update();
        for (const bool backspace : {false, true}) {
            field->SetValue("");
            context_->Update();
            ASSERT_TRUE(field->Focus());
            const auto input = gui::buildPanelInputFromSDL(orderedTypingFrame(backspace));
            for (const auto& event : input.input_events)
                gui::rml_input::processKeyboardEvent(*context_, event);
            EXPECT_EQ(listener.before_edit, backspace ? "a" : "()");
            EXPECT_EQ(field->GetValue(), backspace ? "b" : "()");
        }
        field->RemoveEventListener("keydown", &listener);
    }

    TEST_F(RmlPointerReplayTest, GlobalInputDispatchOwnsTextHandlerAndKeyReleases) {
        ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                          .string()));
        auto* other = Rml::CreateContext("other-input-context", {400, 300}, render_interface_);
        ASSERT_NE(other, nullptr);
        auto* other_document = other->CreateDocument();
        other_document->Show();
        gui::RmlTextInputHandler handler;
        Rml::SetTextInputHandler(&handler);
        {
            gui::RmlUIManager dispatcher;
            std::array<Rml::Context*, 2> contexts{context_, other};
            std::array<Rml::ElementFormControl*, 2> fields{};
            for (int i = 0; i < 2; ++i) {
                auto* doc = i ? other_document : document_;
                auto field = doc->CreateElement("input");
                field->SetProperty("font-family", "Inter");
                field->SetProperty("font-size", "14px");
                field->SetProperty("width", "200px");
                field->SetProperty("height", "24px");
                fields[i] = dynamic_cast<Rml::ElementFormControl*>(field.get());
                fields[i]->SetValue(i ? "" : "original");
                field->SetId(i ? "field-b" : "field-a");
                field->AddEventListener("keyup", &recorder_);
                doc->AppendChild(std::move(field));
                contexts[i]->Update();
            }
            // B registers first, matching the render order that used to retarget A's Ctrl+A.
            for (int i : {1, 0}) {
                dispatcher.routeInput(contexts[i], {}, [&, i](const gui::PanelInputState& input) {
                    if (input.mouse_clicked[0]) {
                        if (int(input.mouse_x) == i)
                            fields[i]->Focus();
                        else if (auto* focus = contexts[i]->GetFocusElement())
                            focus->Blur();
                    }
                    for (const auto& event : input.input_events)
                        gui::rml_input::processKeyboardEvent(*contexts[i], event, &handler);
                });
            }
            ASSERT_TRUE(fields[0]->Focus());
            SDL_Event event{};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.scancode = SDL_SCANCODE_A;
            event.key.mod = SDL_KMOD_CTRL;
            event.key.down = true;
            EXPECT_TRUE(dispatcher.dispatchInputEvent(event).consumed);
            event = {};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = "x";
            EXPECT_TRUE(dispatcher.dispatchInputEvent(event).consumed);
            event = {};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = 1;
            dispatcher.dispatchInputEvent(event);
            event = {};
            event.type = SDL_EVENT_TEXT_EDITING;
            event.edit.text = "y";
            event.edit.start = 1;
            dispatcher.dispatchInputEvent(event);
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = "y";
            dispatcher.dispatchInputEvent(event);
            EXPECT_EQ(fields[0]->GetValue(), "x");
            EXPECT_EQ(fields[1]->GetValue(), "y");
            event = {};
            event.type = SDL_EVENT_KEY_UP;
            event.key.scancode = SDL_SCANCODE_A;
            EXPECT_TRUE(dispatcher.dispatchInputEvent(event).consumed);
            EXPECT_EQ(recorder_.countOf("field-a:keyup"), 1);
            EXPECT_EQ(recorder_.countOf("field-b:keyup"), 0);
            fields[1]->Blur();
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.scancode = SDL_SCANCODE_LALT;
            event.key.down = true;
            EXPECT_FALSE(dispatcher.dispatchInputEvent(event).consumed);
            fields[0]->Focus();
            event.type = SDL_EVENT_KEY_UP;
            event.key.down = false;
            const auto release = dispatcher.dispatchInputEvent(event);
            EXPECT_FALSE(release.consumed);
            EXPECT_TRUE(release.owned_release);
            auto button = document_->CreateElement("button");
            auto* action = document_->AppendChild(std::move(button));
            ASSERT_TRUE(action->Focus());
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.down = true;
            event.key.scancode = SDL_SCANCODE_Z;
            event.key.mod = SDL_KMOD_CTRL;
            EXPECT_FALSE(dispatcher.dispatchInputEvent(event).consumed);
            for (auto* field : fields)
                field->RemoveEventListener("keyup", &recorder_);
        }
        Rml::SetTextInputHandler(nullptr);
        Rml::RemoveContext("other-input-context");
    }

    TEST_F(RmlPointerReplayTest, TerminalTextDispatchSurvivesRepeatedFocusChangesWithoutRendering) {
        static Rml::ElementInstancerGeneric<gui::TerminalElement> instancer;
        Rml::Factory::RegisterElementInstancer("terminal-view", &instancer);
        auto element = document_->CreateElement("terminal-view");
        auto* terminal = element.get();
        terminal->SetProperty("position", "absolute");
        terminal->SetProperty("left", "0px");
        terminal->SetProperty("top", "0px");
        terminal->SetProperty("width", "100px");
        terminal->SetProperty("height", "100px");
        document_->AppendChild(std::move(element));
        auto button = document_->CreateElement("button");
        auto* tab = document_->AppendChild(std::move(button));
        context_->Update();
        ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_VIDEO));
        auto* window = SDL_CreateWindow("Input collection", 100, 100, SDL_WINDOW_HIDDEN);
        ASSERT_NE(window, nullptr);
        {
            gui::RmlUIManager dispatcher(window);
            std::string received;
            dispatcher.routeInput(context_, {}, [&](const gui::PanelInputState& input) {
                if (!input.mouse_button_events.empty()) {
                    context_->ProcessMouseMove(50, 50, 0);
                    if (input.mouse_clicked[0])
                        context_->ProcessMouseButtonDown(0, 0);
                    if (input.mouse_released[0])
                        context_->ProcessMouseButtonUp(0, 0);
                }
                for (const auto& event : input.input_events)
                    if (event.kind == FrameInputEventKind::Text && context_->GetFocusElement() == terminal)
                        received += event.text;
            });
            for (int i = 0; i < 30; ++i) {
                ASSERT_TRUE(tab->Focus());
                dispatcher.syncTextInput();
                EXPECT_FALSE(gui::guiFocusState().want_text_input);
                EXPECT_FALSE(SDL_TextInputActive(window));
                EXPECT_FALSE(gui::guiFocusState().want_text_input);
                SDL_Event event{};
                event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
                event.button.button = SDL_BUTTON_LEFT;
                event.button.x = event.button.y = 50;
                dispatcher.dispatchInputEvent(event);
                event.type = SDL_EVENT_MOUSE_BUTTON_UP;
                dispatcher.dispatchInputEvent(event);
                EXPECT_TRUE(gui::guiFocusState().want_text_input);
                event = {};
                event.type = SDL_EVENT_TEXT_INPUT;
                event.text.text = "x";
                EXPECT_TRUE(dispatcher.dispatchInputEvent(event).consumed);
                EXPECT_EQ(received, std::string(i + 1, 'x'));
                EXPECT_TRUE(SDL_TextInputActive(window));
            }
            ASSERT_TRUE(tab->Focus());
            dispatcher.syncTextInput();
            EXPECT_FALSE(SDL_TextInputActive(window));
        }
        SDL_DestroyWindow(window);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
    }

    TEST_F(RmlPointerReplayTest, TerminalClickClaimsTextInputFocus) {
        static Rml::ElementInstancerGeneric<gui::TerminalElement> instancer;
        Rml::Factory::RegisterElementInstancer("terminal-view", &instancer);
        auto element = document_->CreateElement("terminal-view");
        auto* terminal = element.get();
        terminal->SetProperty("position", "absolute");
        terminal->SetProperty("left", "0px");
        terminal->SetProperty("top", "0px");
        terminal->SetProperty("width", "100px");
        terminal->SetProperty("height", "100px");
        document_->AppendChild(std::move(element));
        context_->Update();
        EXPECT_TRUE(gui::rml_input::wantsTextInput(context_->GetElementAtPoint({50, 50})));
        context_->ProcessMouseMove(50, 50, 0);
        context_->ProcessMouseButtonDown(0, 0);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_EQ(context_->GetFocusElement(), terminal);
        EXPECT_TRUE(gui::rml_input::wantsTextInput(context_->GetFocusElement()));
    }

    TEST_F(RmlPointerReplayTest, FocusTransitionsSplitKeyboardReplayBetweenFields) {
        ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                          .string()));
        std::array<Rml::ElementFormControl*, 2> fields{};
        for (int i = 0; i < 2; ++i) {
            auto element = document_->CreateElement("input");
            element->SetProperty("font-family", "Inter");
            element->SetProperty("font-size", "14px");
            element->SetProperty("width", "200px");
            element->SetProperty("height", "24px");
            fields[i] = dynamic_cast<Rml::ElementFormControl*>(element.get());
            document_->AppendChild(std::move(element));
        }
        context_->Update();
        FrameInputBuffer frame;
        SDL_Event event{};
        const auto click = [&](float x) {
            event = {};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = x;
            frame.processEvent(event);
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            frame.processEvent(event);
        };
        const auto text = [&](const char* value) {
            event = {};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = value;
            frame.processEvent(event);
        };
        click(0);
        text("x");
        click(1);
        text("y");
        // Replay independently for each panel, as hosts do. The other panel's
        // click blurs this field at that point, without consuming future text.
        for (int panel = 0; panel < 2; ++panel) {
            gui::rml_input::replayInputEvents(frame.input_events, frame.mouse_button_events, [&](const FrameMouseButtonEvent& button) {
                    if (!button.down) return;
                    if (int(button.x) == panel) fields[panel]->Focus();
                    else if (auto* focused = context_->GetFocusElement()) focused->Blur(); }, [&](const FrameInputEvent& input) { gui::rml_input::processKeyboardEvent(*context_, input); });
        }
        EXPECT_EQ(fields[0]->GetValue(), "x");
        EXPECT_EQ(fields[1]->GetValue(), "y");
    }

    TEST_F(RmlPointerReplayTest, RepeatedModalActionsDoNotCancelAfterFieldBlur) {
        auto element = document_->CreateElement("input");
        auto* field = element.get();
        document_->AppendChild(std::move(element));
        ASSERT_TRUE(field->Focus());
        const FrameInputEvent escape{.scancode = SDL_SCANCODE_ESCAPE};
        const FrameInputEvent repeated{.scancode = SDL_SCANCODE_ESCAPE, .repeat = true};
        ASSERT_FALSE(gui::rml_input::isRepeatedDialogAction(escape));
        EXPECT_TRUE(gui::rml_input::cancelFocusedElement(*context_));
        EXPECT_NE(context_->GetFocusElement(), field);
        EXPECT_TRUE(gui::rml_input::isRepeatedDialogAction(repeated));
        EXPECT_TRUE(gui::rml_input::isRepeatedDialogAction({.scancode = SDL_SCANCODE_RETURN, .repeat = true}));
        EXPECT_FALSE(gui::rml_input::isRepeatedDialogAction({.scancode = SDL_SCANCODE_BACKSPACE, .repeat = true}));
        EXPECT_FALSE(gui::rml_input::isRepeatedDialogAction({.scancode = SDL_SCANCODE_LEFT, .repeat = true}));
        auto multiline = document_->CreateElement("textarea");
        EXPECT_FALSE(gui::rml_input::isRepeatedDialogAction({.scancode = SDL_SCANCODE_RETURN, .repeat = true}, multiline.get()));
        EXPECT_TRUE(gui::rml_input::isRepeatedDialogAction(repeated, multiline.get()));
        auto button = document_->CreateElement("button");
        const FrameInputEvent space{.scancode = SDL_SCANCODE_SPACE, .repeat = true};
        EXPECT_TRUE(gui::rml_input::isRepeatedDialogAction(space, button.get()));
        EXPECT_FALSE(gui::rml_input::isRepeatedDialogAction(space, field));
        EXPECT_FALSE(gui::rml_input::isRepeatedDialogAction(space, multiline.get()));
        button->SetAttribute("type", "button");
        button->SetProperty("tab-index", "auto");
        button->SetProperty("width", "100px");
        button->SetProperty("height", "24px");
        auto* action = document_->AppendChild(std::move(button));
        action->SetId("repeat-action");
        action->AddEventListener("click", &recorder_);
        ASSERT_TRUE(action->Focus());
        if (!gui::rml_input::isRepeatedDialogAction(space, context_->GetFocusElement()))
            gui::rml_input::processKeyboardEvent(*context_, space);
        EXPECT_EQ(recorder_.countOf("repeat-action:click"), 0);
        context_->Update();
        gui::rml_input::processKeyboardEvent(*context_, {.scancode = SDL_SCANCODE_SPACE});
        gui::rml_input::processKeyboardEvent(*context_, {.kind = FrameInputEventKind::KeyUp, .scancode = SDL_SCANCODE_SPACE});
        EXPECT_EQ(recorder_.countOf("repeat-action:click"), 1);
        action->RemoveEventListener("click", &recorder_);
    }

    TEST_F(RmlPointerReplayTest, PythonEditorConsumesSdlTextAndKeysInArrivalOrder) {
        static Rml::ElementInstancerGeneric<gui::PythonEditorElement> instancer;
        Rml::Factory::RegisterElementInstancer("python-editor", &instancer);
        editor::PythonEditor editor;
        auto element = document_->CreateElement("python-editor");
        auto* view = dynamic_cast<gui::PythonEditorElement*>(element.get());
        ASSERT_NE(view, nullptr);
        view->setEditor(&editor);
        document_->AppendChild(std::move(element));
        context_->Update();
        for (const bool backspace : {false, true}) {
            editor.setText("");
            ASSERT_TRUE(view->Focus());
            const auto input = gui::buildPanelInputFromSDL(orderedTypingFrame(backspace));
            for (const auto& event : input.input_events)
                gui::rml_input::processKeyboardEvent(*context_, event);
            EXPECT_EQ(editor.getText(), backspace ? "b" : "()\n");
        }
        editor.setText("a");
        ASSERT_TRUE(view->Focus());
        gui::rml_input::processKeyboardEvent(*context_, {.scancode = SDL_SCANCODE_A, .modifiers = SDL_KMOD_CTRL});
        gui::rml_input::processKeyboardEvent(*context_, {.kind = FrameInputEventKind::Text, .text = "£"});
        gui::rml_input::processKeyboardEvent(*context_, {.kind = FrameInputEventKind::Text, .text = "x"});
        EXPECT_EQ(editor.getText(), "£x");
        view->setEditor(nullptr);
    }
} // namespace lfs::vis

namespace lfs::vis {
    class WindowInputDispatchTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO));
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.safe_mode = true;
            viewer_ = std::make_unique<VisualizerImpl>(options);
            window_ = viewer_->getWindowManager();
            gui_ = viewer_->getGuiManager();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            window_->window_ = SDL_CreateWindow("Input dispatch", 400, 300, SDL_WINDOW_HIDDEN);
            ASSERT_NE(window_->window_, nullptr);
            ASSERT_TRUE(manager().initWithRenderInterface(window_->window_, 1.f,
                                                          std::make_unique<NullRenderInterface>(), nullptr));
            gui_->startup_overlay_.dismiss();
            controller_ = std::make_unique<InputController>(window_->window_, viewport_);
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
            revert_.clear();
            gui_->startup_overlay_.shutdown();
            controller_.reset();
            viewer_.reset();
            Rml::SetSystemInterface(nullptr);
            Rml::SetRenderInterface(nullptr);
            services().clear();
            gui::guiFocusState().reset();
        }
        gui::RmlUIManager& manager() { return gui_->rmlui_manager_; }
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
        void text(const char* value) {
            SDL_Event event{};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = value;
            dispatch(event);
        }
        void click(int x = 25, int y = 25) {
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = x;
            event.button.y = y;
            dispatch(event);
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            dispatch(event);
        }
        void nativeClickThenText(const char* value) {
            // SDL invokes this same watch while translating native events. Text
            // is generated only if the preceding press enabled native text input.
            window_->pumping_events_ = true;
            SDL_AddEventWatch(WindowManager::watchEvent, window_);
            SDL_Event event{};
            event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
            event.button.windowID = SDL_GetWindowID(window_->window_);
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = event.button.y = 25;
            ASSERT_TRUE(SDL_PushEvent(&event));
            EXPECT_TRUE(SDL_TextInputActive(window_->window_));
            event.type = SDL_EVENT_MOUSE_BUTTON_UP;
            ASSERT_TRUE(SDL_PushEvent(&event));
            if (SDL_TextInputActive(window_->window_)) {
                event = {};
                event.type = SDL_EVENT_TEXT_INPUT;
                event.text.windowID = SDL_GetWindowID(window_->window_);
                event.text.text = value;
                ASSERT_TRUE(SDL_PushEvent(&event));
            }
            SDL_RemoveEventWatch(WindowManager::watchEvent, window_);
            window_->pumping_events_ = false;
            SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
            window_->dispatched_events_.clear();
        }
        SDL_Window* nativeWindow() { return window_->window_; }
        gui::RmlModalOverlay& modal() { return *gui_->rml_modal_overlay_; }
        gui::GlobalContextMenu& menu() { return *gui_->global_context_menu_; }
        void openStartupLanguage() {
            auto& overlay = gui_->startup_overlay_;
            overlay.init(&manager());
            overlay.visible_ = true;
            overlay.shown_frames_ = 3;
            overlay.rml_context_->SetDimensions({800, 600});
            overlay.rml_context_->Update();
            overlay.forwardInput({}, 0, 0, 800, 600);
            auto* select = overlay.document_->GetElementById("lang-select");
            ASSERT_NE(select, nullptr);
            ASSERT_TRUE(select->Focus());
            overlay.content_dirty_ = false;
            key(SDL_SCANCODE_SPACE);
            ASSERT_TRUE(overlay.isLanguageSelectOpen());
            EXPECT_TRUE(overlay.content_dirty_);
        }
        bool startupVisible() { return gui_->startup_overlay_.isVisible(); }
        bool languageOpen() { return gui_->startup_overlay_.isLanguageSelectOpen(); }
        Viewport viewport_{400, 300};
        std::unique_ptr<VisualizerImpl> viewer_;
        WindowManager* window_ = nullptr;
        gui::GuiManager* gui_ = nullptr;
        std::unique_ptr<InputController> controller_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::ElementFormControlInput* field_ = nullptr;
        uint64_t timestamp_ = 0;
        gui::rml_input::TextInputEscapeRevertController revert_;
    };

    TEST_F(WindowInputDispatchTest, OpeningAndClosingModalChangesOwnerWithinOneBatch) {
        click();
        text("underlying");
        lfs::event::ScopedHandler handlers;
        std::string submitted;
        int shortcuts = 0;
        handlers.subscribe<core::events::cmd::ProjectSave>([&](const auto&) {
            core::ModalRequest request;
            request.title = "Input";
            request.has_input = true;
            request.input_default = "placeholder";
            request.buttons = {{"OK", "primary"}};
            request.on_result = [&](const auto& result) { submitted = result.input_value; };
            gui_->enqueueModal(std::move(request));
        });
        handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++shortcuts; });
        key(SDL_SCANCODE_S, SDL_KMOD_CTRL);
        ASSERT_TRUE(modal().isOpen());
        // A background panel must not steal the modal's focus or text handler.
        field_->Focus();
        key(SDL_SCANCODE_A, SDL_KMOD_CTRL);
        text("modal");
        EXPECT_EQ(field_->GetValue(), "underlying");
        key(SDL_SCANCODE_RETURN);
        EXPECT_EQ(submitted, "modal");
        EXPECT_FALSE(modal().isOpen());
        key(SDL_SCANCODE_G);
        EXPECT_EQ(shortcuts, 1);
        EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
    }

    TEST_F(WindowInputDispatchTest, MenuOwnershipStartsAndEndsWithoutRendering) {
        click();
        text("field");
        int shortcuts = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++shortcuts; });
        menu().request({}, 200, 100, {});
        key(SDL_SCANCODE_G);
        text("hidden");
        EXPECT_EQ(field_->GetValue(), "field");
        EXPECT_EQ(shortcuts, 0);
        key(SDL_SCANCODE_ESCAPE);
        key(SDL_SCANCODE_G);
        EXPECT_EQ(shortcuts, 1);
    }

    TEST_F(WindowInputDispatchTest, SaveAndBindingCapturePrecedeTextConsumption) {
        click();
        int saves = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ProjectSave>([&](const auto&) { ++saves; });
        key(SDL_SCANCODE_S, SDL_KMOD_CTRL);
        EXPECT_EQ(saves, 1);
        controller_->getBindings().startCapture(input::ToolMode::GLOBAL, input::Action::TOOL_ALIGN);
        key(SDL_SCANCODE_B);
        const auto captured = controller_->getBindings().getAndClearCaptured();
        ASSERT_TRUE(captured);
        const auto* trigger = std::get_if<input::KeyTrigger>(&*captured);
        ASSERT_NE(trigger, nullptr);
        EXPECT_EQ(trigger->key, input::KEY_B);
    }

    TEST_F(WindowInputDispatchTest, ClickEscapeThenViewportShortcutHasNoPhantomTextFocus) {
        int shortcuts = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++shortcuts; });
        click();
        text("changed");
        key(SDL_SCANCODE_ESCAPE);
        key(SDL_SCANCODE_G);
        EXPECT_FALSE(manager().wantsTextInput());
        EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
        EXPECT_EQ(shortcuts, 1);
    }

    TEST_F(WindowInputDispatchTest, NativePressEnablesTextBeforeTranslationAndBlurStopsIt) {
        for (int i = 0; i < 30; ++i) {
            EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
            nativeClickThenText("abcdefghijklmnopqrstuvwxy");
            EXPECT_EQ(field_->GetValue(), "abcdefghijklmnopqrstuvwxy");
            key(SDL_SCANCODE_ESCAPE);
            EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
            EXPECT_EQ(field_->GetValue(), "");
        }
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, EscapeClosesLanguageDropdownWithoutDismissingStartup) {
        openStartupLanguage();
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(languageOpen());
        EXPECT_TRUE(startupVisible());
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(startupVisible());
    }

    TEST_F(WindowInputDispatchTest, DispatchReusesEventStorage) {
        click();
        const FrameInputEvent* storage = nullptr;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            if (input.input_events.empty())
                return;
            if (storage)
                EXPECT_EQ(storage, input.input_events.data());
            storage = input.input_events.data();
            for (const auto& event : input.input_events)
                gui::rml_input::processKeyboardEvent(*context_, event, manager().getTextInputHandler());
        });
        for (int i = 0; i < 100; ++i)
            text("x");
        EXPECT_EQ(field_->GetValue(), std::string(100, 'x'));
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, WorkerWakeDoesNotWaitForTheNativeInputCallback) {
        std::future<void> wake;
        bool called = false;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            for (const auto& button : input.mouse_button_events) {
                context_->ProcessMouseMove(button.x, button.y, 0);
                if (button.down)
                    context_->ProcessMouseButtonDown(button.button, 0);
                else
                    context_->ProcessMouseButtonUp(button.button, 0);
            }
            for (const auto& event : input.input_events) {
                if (event.kind != FrameInputEventKind::Text)
                    continue;
                called = true;
                wake = std::async(std::launch::async, [&] { window_->wakeEventLoop(); });
                // A native callback can need work from a thread holding the
                // Python lock. Waking the viewer must not wait for this callback.
                EXPECT_EQ(wake.wait_for(std::chrono::milliseconds(200)), std::future_status::ready);
            }
        });
        nativeClickThenText("x");
        EXPECT_TRUE(called);
        if (wake.valid())
            wake.get();
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, FreshPressIsNotMaskedByPreviousFramePointerState) {
        click();
        gui::PanelInputState masked;
        masked.mouse_x = masked.mouse_y = -1.e9f;
        int presses = 0;
        manager().routeInput(context_, masked, [&](const gui::PanelInputState& input) {
            for (const auto& event : input.mouse_button_events) {
                if (event.down) {
                    ++presses;
                    field_->Blur();
                }
            }
        });
        int shortcuts = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleGTComparison>([&](const auto&) { ++shortcuts; });
        click(300, 200);
        key(SDL_SCANCODE_G);
        EXPECT_EQ(presses, 1);
        EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
        EXPECT_EQ(shortcuts, 1);
    }

    TEST_F(WindowInputDispatchTest, EditorFocusRequestPrecedesTheNextPointerTransition) {
        editor::PythonEditor editor;
        auto element = document_->CreateElement("python-editor-view");
        auto* view = dynamic_cast<gui::PythonEditorElement*>(element.get());
        ASSERT_NE(view, nullptr);
        view->setEditor(&editor);
        view->SetProperty("position", "absolute");
        view->SetProperty("left", "210px");
        view->SetProperty("top", "10px");
        view->SetProperty("width", "160px");
        view->SetProperty("height", "60px");
        document_->AppendChild(std::move(element));
        context_->Update();
        click(225, 25);
        click();
        ASSERT_EQ(context_->GetFocusElement(), field_);
        editor.focus();
        EXPECT_EQ(context_->GetFocusElement(), view);
        click();
        text("next");
        EXPECT_EQ(field_->GetValue(), "next");
        EXPECT_FALSE(editor.isFocused());
        view->setEditor(nullptr);
    }
} // namespace lfs::vis
