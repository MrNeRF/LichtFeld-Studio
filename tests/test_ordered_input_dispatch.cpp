/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_input.hpp"
#include "gui/gui_manager.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rml_sequencer_overlay.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/scene_panel_native.hpp"
#include "gui/translation_gizmo.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "test_view_targets.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer_impl.hpp"
#include "window/window_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <cstring>
#include <future>
#include <glm/gtc/type_ptr.hpp>
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

    class OrderedPointerReplayTest : public ::testing::Test {
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

    TEST_F(OrderedPointerReplayTest, DocumentReceivesKeysWithoutTextFocus) {
        document_->SetId("keyboard-document");
        document_->AddEventListener("keydown", &recorder_);
        ASSERT_TRUE(document_->Focus());
        ASSERT_EQ(context_->GetFocusElement(), document_);
        const FrameInputEvent event{.kind = FrameInputEventKind::KeyDown, .scancode = SDL_SCANCODE_A};
        EXPECT_TRUE(gui::rml_input::processKeyboardEvent(*context_, event));
        EXPECT_EQ(recorder_.countOf("keyboard-document:keydown"), 1) << recorder_.joined();
        document_->RemoveEventListener("keydown", &recorder_);
    }

    TEST_F(OrderedPointerReplayTest, TextFieldConsumesSdlTextAndKeysInArrivalOrder) {
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

    TEST_F(OrderedPointerReplayTest, GlobalInputDispatchOwnsTextHandlerAndKeyReleases) {
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

    TEST_F(OrderedPointerReplayTest, TerminalTextDispatchSurvivesRepeatedFocusChangesWithoutRendering) {
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

    TEST_F(OrderedPointerReplayTest, TerminalClickClaimsTextInputFocus) {
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

    TEST_F(OrderedPointerReplayTest, FocusTransitionsSplitKeyboardReplayBetweenFields) {
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

    TEST_F(OrderedPointerReplayTest, RepeatedModalActionsDoNotCancelAfterFieldBlur) {
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

    TEST_F(OrderedPointerReplayTest, PythonEditorConsumesSdlTextAndKeysInArrivalOrder) {
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
        virtual bool keepSceneHandlers() const { return false; }
        void SetUp() override {
            ASSERT_TRUE(SDL_Init(SDL_INIT_VIDEO));
            ViewerOptions options;
            options.show_startup_overlay = false;
            options.safe_mode = true;
            viewer_ = std::make_unique<VisualizerImpl>(options);
            window_ = viewer_->getWindowManager();
            gui_ = viewer_->getGuiManager();
            if (!keepSceneHandlers()) {
                lfs::event::EventBridge::instance().clear_all();
                lfs::core::event::bus().clear_all();
            }
            window_->window_ = SDL_CreateWindow("Input dispatch", 400, 300, SDL_WINDOW_HIDDEN);
            ASSERT_NE(window_->window_, nullptr);
            ASSERT_TRUE(manager().initWithRenderInterface(window_->window_, 1.f,
                                                          std::make_unique<NullRenderInterface>(), nullptr));
            gui_->startup_overlay_.dismiss();
            controller_ = std::make_unique<InputController>(window_->window_, view_targets_);
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
            gui::cancelTranslationGizmoDrag();
            gui::cancelRotationGizmoDrag();
            gui::cancelScaleGizmoDrag();
            gui::cancelBoundsGizmoDrag();
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
        void watchNative(SDL_Event event) {
            window_->pumping_events_ = true;
            WindowManager::watchEvent(window_, &event);
            window_->pumping_events_ = false;
        }
        void poll() { window_->pollEvents(); }
        FrameInputBuffer& frameInput() { return window_->frame_input_; }
        void wait(double timeout) { window_->waitEvents(timeout); }
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
        void setGizmoViewportLayout(const gui::ViewportLayout& layout, const bool hidden = true) {
            gui_->ui_hidden_ = hidden;
            gui_->viewport_layout_ = layout;
        }
        bool startupVisible() { return gui_->startup_overlay_.isVisible(); }
        bool languageOpen() { return gui_->startup_overlay_.isLanguageSelectOpen(); }

        void dragXAxisCenterlineAtViewTilt(const float tilt_degrees, const bool scale, const float y_offset = 0.0f,
                                           const float start_x = 236.0f) {
            auto& scene_manager = *viewer_->getSceneManager();
            auto& scene = viewer_->getScene();
            const core::NodeId node_id = scene.addGroup(
                std::string("Axis centerline drag ") + (scale ? "scale " : "translate ") + std::to_string(tilt_degrees));
            ASSERT_NE(node_id, core::NULL_NODE);
            scene_manager.selectNode(node_id);
            viewer_->getEditorContext().update(&scene_manager, viewer_->getTrainerManager());
            viewer_->getEditorContext().setActiveTool(scale ? ToolType::Scale : ToolType::Translate);
            UnifiedToolRegistry::instance().setActiveTool(scale ? "builtin.scale" : "builtin.translate");
            auto& gizmo = gui_->gizmo();
            gizmo.setOperation(scale ? gui::GizmoOperation::Scale : gui::GizmoOperation::Translate);
            gizmo.setTransformSpace(TransformSpace::World);
            gizmo.setPivotMode(PivotMode::Origin);

            auto& camera = viewer_->getViewport().camera;
            const float tilt = glm::radians(tilt_degrees);
            camera.t = {12.0f * std::sin(tilt), 0.0f, 12.0f * std::cos(tilt)};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);

            gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
            const gui::ViewportLayout layout{.view = viewer_->activeView().id, .pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            gui_->ui_hidden_ = true;
            gui_->viewport_layout_ = layout;
            gizmo.updateToolState(ui, false);
            auto& frame = window_->frame_input_;
            const glm::vec2 start{start_x, 150.0f + y_offset};
            frame.mouse_x = start.x;
            frame.mouse_y = start.y;
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            gui::guiFocusState().want_capture_mouse = false;
            gizmo.renderNodeTransformGizmo(ui, layout);
            const glm::mat4 before = scene.getNodeTransform(node_id);

            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS,
                                           start.x, start.y);
            frame.mouse_down[0] = true;
            frame.mouse_clicked[0] = true;
            frame.mouse_released[0] = false;
            frame.mouse_x = start.x;
            gizmo.renderNodeTransformGizmo(ui, layout);
            frame.mouse_x = start.x + 30.0f;
            gizmo.renderNodeTransformGizmo(ui, layout);
            const glm::mat4 after = scene.getNodeTransform(node_id);
            if (scale) {
                EXPECT_GT(std::abs(after[0][0] - before[0][0]), 1e-4f);
            } else {
                EXPECT_GT(std::abs(after[3].x - before[3].x), 1e-4f);
            }

            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                           frame.mouse_x, frame.mouse_y);
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            gizmo.renderNodeTransformGizmo(ui, layout);
        }

        void dragXYPlaneAtNormalView(const bool scale) {
            auto& scene_manager = *viewer_->getSceneManager();
            auto& scene = viewer_->getScene();
            const core::NodeId node_id = scene.addGroup(scale ? "Scale plane handle drag" : "Translate plane handle drag");
            ASSERT_NE(node_id, core::NULL_NODE);
            scene_manager.selectNode(node_id);
            viewer_->getEditorContext().update(&scene_manager, viewer_->getTrainerManager());
            viewer_->getEditorContext().setActiveTool(scale ? ToolType::Scale : ToolType::Translate);
            UnifiedToolRegistry::instance().setActiveTool(scale ? "builtin.scale" : "builtin.translate");
            auto& gizmo = gui_->gizmo();
            gizmo.setOperation(scale ? gui::GizmoOperation::Scale : gui::GizmoOperation::Translate);
            gizmo.setTransformSpace(TransformSpace::World);
            gizmo.setPivotMode(PivotMode::Origin);
            auto& camera = viewer_->getViewport().camera;
            camera.t = {0.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);

            gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
            const gui::ViewportLayout layout{.view = viewer_->activeView().id, .pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            gui_->ui_hidden_ = true;
            gui_->viewport_layout_ = layout;
            gizmo.updateToolState(ui, false);
            auto& frame = window_->frame_input_;
            constexpr glm::vec2 start{230.0f, 120.0f};
            frame.mouse_x = start.x;
            frame.mouse_y = start.y;
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            gui::guiFocusState().want_capture_mouse = false;
            gizmo.renderNodeTransformGizmo(ui, layout);
            if (scale) {
                EXPECT_TRUE(gui::isScaleGizmoHovered());
            }
            const glm::mat4 before = scene.getNodeTransform(node_id);
            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS,
                                           start.x, start.y);
            frame.mouse_down[0] = true;
            frame.mouse_clicked[0] = true;
            frame.mouse_released[0] = false;
            gizmo.renderNodeTransformGizmo(ui, layout);
            frame.mouse_x = start.x + 12.0f;
            frame.mouse_y = start.y + (scale ? -12.0f : 12.0f);
            gizmo.renderNodeTransformGizmo(ui, layout);
            const glm::mat4 after = scene.getNodeTransform(node_id);
            if (scale) {
                EXPECT_GT(std::abs(after[0][0] - before[0][0]), 1e-4f);
                EXPECT_GT(std::abs(after[1][1] - before[1][1]), 1e-4f);
            } else {
                EXPECT_GT(std::abs(after[3].x - before[3].x), 1e-4f);
                EXPECT_GT(std::abs(after[3].y - before[3].y), 1e-4f);
                EXPECT_NEAR(after[3].x, 0.32914072f, 1e-6f);
                EXPECT_NEAR(after[3].y, 0.32914030f, 1e-6f);
            }

            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                           frame.mouse_x, frame.mouse_y);
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            gizmo.renderNodeTransformGizmo(ui, layout);
        }

        enum class TransformDragMode { Translate,
                                       Rotate,
                                       Scale,
                                       BoundsScale };
        enum class TransformDragFinish { Escape,
                                         RightClick,
                                         LeftRelease,
                                         ReleaseAtStart,
                                         ZeroLengthRelease };

        static bool sameMatrixBits(const glm::mat4& lhs, const glm::mat4& rhs) {
            return std::memcmp(glm::value_ptr(lhs), glm::value_ptr(rhs), sizeof(glm::mat4)) == 0;
        }

        void exerciseTransformDrag(const TransformDragMode mode, const TransformDragFinish finish) {
            auto& scene_manager = *viewer_->getSceneManager();
            auto& scene = viewer_->getScene();
            const bool bounds_scale = mode == TransformDragMode::BoundsScale;
            const std::string node_name = "Cancel drag " + std::to_string(static_cast<int>(mode));
            const core::NodeId node_id = bounds_scale
                                             ? scene.addSplat(node_name, lfs::test::licht::make_splat(3))
                                             : scene.addGroup(node_name);
            ASSERT_NE(node_id, core::NULL_NODE);
            scene_manager.changeContentType(SceneManager::ContentType::SplatFiles);
            scene_manager.selectNode(node_id);
            viewer_->getEditorContext().update(&scene_manager, viewer_->getTrainerManager());

            gui::GizmoOperation operation = gui::GizmoOperation::Translate;
            ToolType tool = ToolType::Translate;
            const char* tool_id = "builtin.translate";
            if (mode == TransformDragMode::Rotate) {
                operation = gui::GizmoOperation::Rotate;
                tool = ToolType::Rotate;
                tool_id = "builtin.rotate";
            } else if (mode == TransformDragMode::Scale || bounds_scale) {
                operation = gui::GizmoOperation::Scale;
                tool = ToolType::Scale;
                tool_id = "builtin.scale";
            }
            viewer_->getEditorContext().setActiveTool(tool);
            UnifiedToolRegistry::instance().setActiveTool(tool_id);
            auto& gizmo = gui_->gizmo();
            gizmo.setOperation(operation);
            gizmo.setTransformSpace(TransformSpace::World);
            gizmo.setPivotMode(PivotMode::Origin);

            auto& camera = viewer_->getViewport().camera;
            camera.t = {0.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);

            gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
            const gui::ViewportLayout layout{.view = viewer_->activeView().id, .pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            gui_->ui_hidden_ = true;
            gui_->viewport_layout_ = layout;
            gizmo.updateToolState(ui, false);
            auto& frame = window_->frame_input_;
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;

            const auto hovered = [&] {
                switch (mode) {
                case TransformDragMode::Translate: return gui::isTranslationGizmoHovered();
                case TransformDragMode::Rotate: return gui::isRotationGizmoHovered();
                case TransformDragMode::Scale: return gui::isScaleGizmoHovered();
                case TransformDragMode::BoundsScale: return gui::isBoundsGizmoHovered();
                }
                return false;
            };
            glm::vec2 start{200.0f, 150.0f};
            if (mode == TransformDragMode::Translate)
                start += glm::vec2(55.0f, 3.0f);
            else if (mode == TransformDragMode::Rotate)
                start += glm::vec2(0.0f, -75.0f);
            else if (mode == TransformDragMode::Scale)
                start += glm::vec2(55.0f, 0.0f);
            else
                start += glm::vec2(65.0f, 0.0f);

            const auto probe = [&](const glm::vec2 pos) {
                frame.mouse_x = pos.x;
                frame.mouse_y = pos.y;
                gui::guiFocusState().want_capture_mouse = false;
                gizmo.renderNodeTransformGizmo(ui, layout);
                return hovered();
            };
            bool found = probe(start);
            if (!found) {
                for (int y = 48; y <= 252 && !found; y += 6) {
                    for (int x = 48; x <= 352 && !found; x += 6)
                        found = probe({static_cast<float>(x), static_cast<float>(y)});
                }
                if (found)
                    start = {frame.mouse_x, frame.mouse_y};
            }
            ASSERT_TRUE(found) << "no transform handle found for mode " << static_cast<int>(mode);

            const glm::mat4 before = scene.getNodeTransform(node_id);
            op::undoHistory().clear();
            const size_t undo_before = op::undoHistory().undoCount();
            gui::guiFocusState().want_capture_mouse = false;
            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS,
                                           start.x, start.y);
            frame.mouse_x = start.x;
            frame.mouse_y = start.y;
            frame.mouse_down[0] = true;
            frame.mouse_clicked[0] = true;
            frame.mouse_released[0] = false;
            gizmo.renderNodeTransformGizmo(ui, layout);

            if (finish == TransformDragFinish::ZeroLengthRelease) {
                controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                               start.x, start.y);
                frame.mouse_down[0] = false;
                frame.mouse_released[0] = true;
                gizmo.renderNodeTransformGizmo(ui, layout);
                frame.mouse_released[0] = false;
                gizmo.renderNodeTransformGizmo(ui, layout);
                EXPECT_TRUE(sameMatrixBits(before, scene.getNodeTransform(node_id)));
                EXPECT_EQ(op::undoHistory().undoCount(), undo_before);
                return;
            }

            if (finish == TransformDragFinish::ReleaseAtStart) {
                frame.mouse_x = start.x + 28.0f;
                frame.mouse_y = start.y;
                frame.mouse_clicked[0] = false;
                gizmo.renderNodeTransformGizmo(ui, layout);
                ASSERT_FALSE(sameMatrixBits(before, scene.getNodeTransform(node_id)))
                    << "pointer motion did not change the selected node transform";

                controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                               start.x, start.y);
                frame.mouse_x = start.x;
                frame.mouse_y = start.y;
                frame.mouse_down[0] = false;
                frame.mouse_released[0] = true;
                gizmo.renderNodeTransformGizmo(ui, layout);
                frame.mouse_released[0] = false;
                gizmo.renderNodeTransformGizmo(ui, layout);

                EXPECT_TRUE(sameMatrixBits(before, scene.getNodeTransform(node_id)));
                EXPECT_EQ(op::undoHistory().undoCount(), undo_before);
                return;
            }

            bool changed = false;
            for (const glm::vec2 delta : {glm::vec2(28.0f, 0.0f), glm::vec2(0.0f, 28.0f),
                                          glm::vec2(20.0f, 20.0f), glm::vec2(-24.0f, 0.0f)}) {
                frame.mouse_x = start.x + delta.x;
                frame.mouse_y = start.y + delta.y;
                frame.mouse_clicked[0] = false;
                gizmo.renderNodeTransformGizmo(ui, layout);
                if (!sameMatrixBits(before, scene.getNodeTransform(node_id))) {
                    changed = true;
                    break;
                }
            }
            ASSERT_TRUE(changed) << "pointer drag did not change the selected node transform";

            if (finish == TransformDragFinish::Escape) {
                controller_->handleKey(input::KEY_ESCAPE, input::ACTION_PRESS, input::KEYMOD_NONE);
            } else if (finish == TransformDragFinish::RightClick) {
                controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::RIGHT), input::ACTION_PRESS,
                                               frame.mouse_x, frame.mouse_y);
            } else {
                controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                               frame.mouse_x, frame.mouse_y);
                frame.mouse_down[0] = false;
                frame.mouse_released[0] = true;
                gizmo.renderNodeTransformGizmo(ui, layout);
            }

            const glm::mat4 after = scene.getNodeTransform(node_id);
            if (finish == TransformDragFinish::LeftRelease) {
                EXPECT_FALSE(sameMatrixBits(before, after));
                EXPECT_EQ(op::undoHistory().undoCount(), undo_before + 1);
            } else {
                EXPECT_TRUE(sameMatrixBits(before, after));
                EXPECT_EQ(op::undoHistory().undoCount(), undo_before);
            }
        }

        Viewport viewport_{400, 300};
        TestViewTargets view_targets_{viewport_};
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

    TEST_F(WindowInputDispatchTest, NodeDeletionPreservesTransformToolThroughReselectionAndHistory) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = viewer_->getScene();
        auto& editor = viewer_->getEditorContext();
        auto& gizmo = gui_->gizmo();
        gizmo.setupEvents();
        int index = 0;
        for (const auto tool : {ToolType::Translate, ToolType::Rotate, ToolType::Scale}) {
            for (const auto pivot : {PivotMode::Origin, PivotMode::BoundsCenter}) {
                for (const bool remove_selected : {true, false}) {
                    SCOPED_TRACE(static_cast<int>(tool));
                    SCOPED_TRACE(remove_selected);
                    const auto survivor_name = "Survivor " + std::to_string(index);
                    const auto removed_name = "Removed " + std::to_string(index++);
                    const auto survivor = scene.addSplat(survivor_name, lfs::test::licht::make_splat(3));
                    const auto removed = scene.addSplat(removed_name, lfs::test::licht::make_splat(3));
                    sm.changeContentType(SceneManager::ContentType::SplatFiles);
                    sm.selectNode(remove_selected ? removed : survivor);
                    editor.update(&sm, viewer_->getTrainerManager());
                    core::events::tools::SetToolbarTool{.tool_mode = static_cast<int>(tool)}.emit();
                    gizmo.setPivotMode(pivot);
                    const auto operation = gizmo.getOperation();
                    const auto tool_id = app_store().active_tool.get();
                    ASSERT_EQ(editor.getActiveTool(), tool);
                    ASSERT_FALSE(tool_id.empty());
                    const auto check_tool = [&] {
                        EXPECT_EQ(editor.getActiveTool(), tool);
                        EXPECT_EQ(app_store().active_tool.get(), tool_id);
                        EXPECT_EQ(UnifiedToolRegistry::instance().getActiveTool(), tool_id);
                        EXPECT_EQ(gizmo.getOperation(), operation);
                        EXPECT_EQ(gizmo.getPivotMode(), pivot);
                    };
                    op::undoHistory().clear();
                    ASSERT_TRUE(sm.removeNodeWithResult(removed, false));
                    ASSERT_EQ(scene.getNode(removed_name), nullptr);
                    check_tool();
                    sm.selectNode(survivor);
                    editor.update(&sm, viewer_->getTrainerManager());
                    ASSERT_TRUE(editor.canTransformSelectedNode());
                    check_tool();
                    ASSERT_TRUE(op::undoHistory().undo().success);
                    ASSERT_NE(scene.getNode(removed_name), nullptr);
                    check_tool();
                    ASSERT_TRUE(op::undoHistory().redo().success);
                    ASSERT_EQ(scene.getNode(removed_name), nullptr);
                    sm.selectNode(survivor_name);
                    editor.update(&sm, viewer_->getTrainerManager());
                    check_tool();
                }
            }
        }
        op::undoHistory().clear();
        core::events::state::SceneCleared{}.emit();
        EXPECT_EQ(editor.getActiveTool(), ToolType::None);
        EXPECT_TRUE(app_store().active_tool.get().empty());
        EXPECT_EQ(gizmo.getOperation(), gui::GizmoOperation::Translate);
    }

    TEST_F(WindowInputDispatchTest, UnrelatedDeletionPreservesCropToolAndVolumeDeletionLeavesIt) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = viewer_->getScene();
        auto& editor = viewer_->getEditorContext();
        auto& gizmo = gui_->gizmo();
        gizmo.setupEvents();
        int index = 0;
        for (const bool ellipsoid : {false, true}) {
            for (const auto operation : {gui::GizmoOperation::Translate, gui::GizmoOperation::Rotate,
                                         gui::GizmoOperation::Scale}) {
                SCOPED_TRACE(ellipsoid);
                SCOPED_TRACE(static_cast<int>(operation));
                const auto suffix = std::to_string(index++);
                const auto parent = scene.addSplat("Crop parent " + suffix, lfs::test::licht::make_splat(3));
                const auto other = scene.addSplat("Other " + suffix, lfs::test::licht::make_splat(3));
                const auto volume = ellipsoid ? scene.addEllipsoid("Volume " + suffix, parent)
                                              : scene.addCropBox("Volume " + suffix, parent);
                sm.changeContentType(SceneManager::ContentType::SplatFiles);
                sm.selectNode(volume);
                editor.update(&sm, viewer_->getTrainerManager());
                gizmo.setOperation(operation);
                ASSERT_EQ(editor.getActiveOperator(), "builtin.cropbox");
                ASSERT_TRUE(sm.removeNodeWithResult(other, false));
                EXPECT_EQ(editor.getActiveOperator(), "builtin.cropbox");
                EXPECT_EQ(app_store().active_tool.get(), "builtin.cropbox");
                EXPECT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "builtin.cropbox");
                EXPECT_EQ(gizmo.getOperation(), operation);
                EXPECT_EQ(sm.getSelectedNodeIds(), std::vector<core::NodeId>{volume});
                ASSERT_TRUE(sm.removeNodeWithResult(volume, false));
                EXPECT_EQ(sm.getSelectedNodeIds(), std::vector<core::NodeId>{parent});
                EXPECT_FALSE(editor.hasActiveOperator());
                EXPECT_TRUE(UnifiedToolRegistry::instance().getActiveTool().empty());
                op::undoHistory().clear();
            }
        }
    }

#define TRANSFORM_DRAG_CANCEL_TEST(test_name, drag_mode, finish_mode) \
    TEST_F(WindowInputDispatchTest, test_name) {                      \
        exerciseTransformDrag(TransformDragMode::drag_mode,           \
                              TransformDragFinish::finish_mode);      \
    }

    TRANSFORM_DRAG_CANCEL_TEST(EscapeCancelsTranslateAndAddsNoUndo, Translate, Escape)
    TRANSFORM_DRAG_CANCEL_TEST(RightClickCancelsTranslateAndAddsNoUndo, Translate, RightClick)
    TRANSFORM_DRAG_CANCEL_TEST(LeftReleaseCommitsTranslateAndAddsOneUndo, Translate, LeftRelease)
    TRANSFORM_DRAG_CANCEL_TEST(EscapeCancelsRotateAndAddsNoUndo, Rotate, Escape)
    TRANSFORM_DRAG_CANCEL_TEST(RightClickCancelsRotateAndAddsNoUndo, Rotate, RightClick)
    TRANSFORM_DRAG_CANCEL_TEST(LeftReleaseCommitsRotateAndAddsOneUndo, Rotate, LeftRelease)
    TRANSFORM_DRAG_CANCEL_TEST(EscapeCancelsScaleAndAddsNoUndo, Scale, Escape)
    TRANSFORM_DRAG_CANCEL_TEST(RightClickCancelsScaleAndAddsNoUndo, Scale, RightClick)
    TRANSFORM_DRAG_CANCEL_TEST(LeftReleaseCommitsScaleAndAddsOneUndo, Scale, LeftRelease)
    TRANSFORM_DRAG_CANCEL_TEST(EscapeCancelsBoundsScaleAndAddsNoUndo, BoundsScale, Escape)
    TRANSFORM_DRAG_CANCEL_TEST(RightClickCancelsBoundsScaleAndAddsNoUndo, BoundsScale, RightClick)
    TRANSFORM_DRAG_CANCEL_TEST(LeftReleaseCommitsBoundsScaleAndAddsOneUndo, BoundsScale, LeftRelease)

    TEST_F(WindowInputDispatchTest, SameFrameReleaseAppliesFinalTranslatePosition) {
        exerciseTransformDrag(TransformDragMode::Translate, TransformDragFinish::ReleaseAtStart);
    }

    TEST_F(WindowInputDispatchTest, SameFrameReleaseAppliesFinalRotatePosition) {
        exerciseTransformDrag(TransformDragMode::Rotate, TransformDragFinish::ReleaseAtStart);
    }

    TEST_F(WindowInputDispatchTest, SameFrameReleaseAppliesFinalScalePosition) {
        exerciseTransformDrag(TransformDragMode::Scale, TransformDragFinish::ReleaseAtStart);
    }

    TEST_F(WindowInputDispatchTest, SameFrameReleaseAppliesFinalBoundsPosition) {
        exerciseTransformDrag(TransformDragMode::BoundsScale, TransformDragFinish::ReleaseAtStart);
    }

    TEST_F(WindowInputDispatchTest, ZeroLengthGizmoClickDragsAddNoUndoEntry) {
        for (const auto mode : {TransformDragMode::Translate, TransformDragMode::Rotate,
                                TransformDragMode::Scale, TransformDragMode::BoundsScale}) {
            SCOPED_TRACE(static_cast<int>(mode));
            exerciseTransformDrag(mode, TransformDragFinish::ZeroLengthRelease);
        }
    }

#undef TRANSFORM_DRAG_CANCEL_TEST

    // Catches gizmo hover that outlives the crop gizmo: the deleted volume's last hover
    // made every later viewport press look like a gizmo grab, so orbit never started.
    TEST_F(WindowInputDispatchTest, ApplyingCropWhileHoveringItsGizmoKeepsViewportOrbit) {
        auto& scene_manager = *viewer_->getSceneManager();
        auto& scene = viewer_->getScene();
        const core::NodeId splat_id = scene.addSplat("Crop orbit", lfs::test::licht::make_splat(3));
        ASSERT_NE(splat_id, core::NULL_NODE);
        scene_manager.changeContentType(SceneManager::ContentType::SplatFiles);
        const core::NodeId cropbox_id = scene.addCropBox("Crop orbit_cropbox", splat_id);
        ASSERT_NE(cropbox_id, core::NULL_NODE);
        scene_manager.selectNode(cropbox_id);
        viewer_->getEditorContext().update(&scene_manager, viewer_->getTrainerManager());
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        auto& gizmo = gui_->gizmo();
        gizmo.setCropToolShape("box");
        gizmo.setOperation(gui::GizmoOperation::Translate);

        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);

        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.view = viewer_->activeView().id, .pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        setGizmoViewportLayout(layout);
        gizmo.updateToolState(ui, false);
        auto& frame = frameInput();
        frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
        bool hovered = false;
        for (int y = 48; y <= 252 && !hovered; y += 4) {
            for (int x = 48; x <= 352 && !hovered; x += 4) {
                frame.mouse_x = static_cast<float>(x);
                frame.mouse_y = static_cast<float>(y);
                gizmo.renderCropBoxGizmo(ui, layout);
                hovered = gui::isTranslationGizmoHovered();
            }
        }
        ASSERT_TRUE(hovered) << "no crop gizmo handle found under the pointer";

        gizmo.applyActiveCropTool();
        ASSERT_EQ(scene.getNodeById(cropbox_id), nullptr);
        ASSERT_EQ(UnifiedToolRegistry::instance().getActiveTool(), "");
        gizmo.updateToolState(ui, false);
        gizmo.renderCropBoxGizmo(ui, layout);
        gizmo.renderNodeTransformGizmo(ui, layout);
        gui::guiFocusState().want_capture_mouse = false;

        Viewport orbit_viewport(400, 300);
        TestViewTargets orbit_views{orbit_viewport};
        InputController orbit_controller(nullptr, orbit_views);
        orbit_controller.updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);
        orbit_controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE), input::ACTION_PRESS,
                                           60.0, 250.0);
        EXPECT_TRUE(orbit_controller.isCameraDragging());
    }

    // A translate drag ends with the pointer still over the handle. That hover must not
    // swallow a middle-button orbit, and it must not outlive the pointer leaving the gizmo.
    TEST_F(WindowInputDispatchTest, TranslateReleaseHoverKeepsMiddleOrbitAndClearsOnLeave) {
        auto& scene_manager = *viewer_->getSceneManager();
        auto& scene = viewer_->getScene();
        const core::NodeId node_id = scene.addGroup("Orbit after translate");
        ASSERT_NE(node_id, core::NULL_NODE);
        scene_manager.selectNode(node_id);
        viewer_->getEditorContext().update(&scene_manager, viewer_->getTrainerManager());
        viewer_->getEditorContext().setActiveTool(ToolType::Translate);
        UnifiedToolRegistry::instance().setActiveTool("builtin.translate");
        auto& gizmo = gui_->gizmo();
        gizmo.setOperation(gui::GizmoOperation::Translate);
        gizmo.setTransformSpace(TransformSpace::World);
        gizmo.setPivotMode(PivotMode::Origin);

        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);

        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.view = viewer_->activeView().id, .pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        setGizmoViewportLayout(layout, false);
        gui_->screenHost().init({.screens = &viewer_->screens(), .rml = &manager(), .scene_manager = &scene_manager});
        gui_->screenHost().layout({0.0f, 0.0f, 400.0f, 300.0f}, 1.0f);
        gizmo.updateToolState(ui, false);
        auto& frame = frameInput();
        const auto render_at = [&](const float x, const float y, const bool down, const bool clicked) {
            frame.mouse_x = x;
            frame.mouse_y = y;
            frame.mouse_down[0] = down;
            frame.mouse_clicked[0] = clicked;
            frame.mouse_released[0] = false;
            gui::beginTranslationGizmoFrame();
            gui::guiFocusState().want_capture_mouse = false;
            gizmo.renderNodeTransformGizmo(ui, layout);
        };

        render_at(236.0f, 150.0f, false, false);
        ASSERT_TRUE(gui::isTranslationGizmoHovered());
        render_at(236.0f, 150.0f, true, true);
        render_at(266.0f, 150.0f, true, false);
        ASSERT_TRUE(gui::isTranslationGizmoActive());
        render_at(266.0f, 150.0f, false, false);
        ASSERT_FALSE(gui::isTranslationGizmoActive());
        ASSERT_TRUE(gui::isTranslationGizmoHovered());
        gui::guiFocusState().want_capture_mouse = false;

        Viewport orbit_viewport(400, 300);
        TestViewTargets orbit_views{orbit_viewport};
        InputController orbit_controller(nullptr, orbit_views);
        orbit_controller.updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);
        orbit_controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE), input::ACTION_PRESS,
                                           266.0, 150.0);
        EXPECT_TRUE(orbit_controller.isCameraDragging());
        orbit_controller.handleMouseButton(static_cast<int>(input::AppMouseButton::MIDDLE), input::ACTION_RELEASE,
                                           266.0, 150.0);

        context_->ProcessMouseMove(60, 260, 0);
        EXPECT_TRUE(gui_->passiveMouseMoveNeedsRender(60.0f, 260.0f));
        render_at(60.0f, 260.0f, false, false);
        EXPECT_FALSE(gui::isTranslationGizmoHovered());
        gui::guiFocusState().want_capture_mouse = false;
        EXPECT_FALSE(gui_->passiveMouseMoveNeedsRender(60.0f, 260.0f));
    }

    TEST_F(WindowInputDispatchTest, TranslateXAxisCenterlineDragSurvivesSmallViewTilts) {
        for (const float tilt : {0.0f, 1.0f, 2.0f, 3.0f, 5.0f}) {
            SCOPED_TRACE(tilt);
            dragXAxisCenterlineAtViewTilt(tilt, false);
        }
    }

    TEST_F(WindowInputDispatchTest, ScaleXAxisCenterlineDragSurvivesSmallViewTilts) {
        for (const float tilt : {0.0f, 1.0f, 2.0f, 3.0f, 5.0f}) {
            SCOPED_TRACE(tilt);
            dragXAxisCenterlineAtViewTilt(tilt, true);
        }
    }

    TEST_F(WindowInputDispatchTest, TranslateXAxisThreePixelOffsetStillDrags) {
        dragXAxisCenterlineAtViewTilt(0.0f, false, 3.0f, 255.0f);
    }

    TEST_F(WindowInputDispatchTest, TranslateXYPlaneHandleStillDragsAtNormalView) {
        dragXYPlaneAtNormalView(false);
    }

    TEST_F(WindowInputDispatchTest, ScaleXYPlaneHandleStillDragsAtNormalView) {
        dragXYPlaneAtNormalView(true);
    }

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
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++shortcuts; });
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
        key(SDL_SCANCODE_V);
        EXPECT_EQ(shortcuts, 1);
        EXPECT_FALSE(SDL_TextInputActive(nativeWindow()));
    }

    TEST_F(WindowInputDispatchTest, MenuOwnershipStartsAndEndsWithoutRendering) {
        click();
        text("field");
        int shortcuts = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++shortcuts; });
        menu().request({}, 200, 100, {});
        key(SDL_SCANCODE_V);
        text("hidden");
        EXPECT_EQ(field_->GetValue(), "field");
        EXPECT_EQ(shortcuts, 0);
        key(SDL_SCANCODE_ESCAPE);
        key(SDL_SCANCODE_V);
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
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++shortcuts; });
        click();
        text("changed");
        key(SDL_SCANCODE_ESCAPE);
        key(SDL_SCANCODE_V);
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
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++shortcuts; });
        click(300, 200);
        key(SDL_SCANCODE_V);
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

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, WorkerEventsPrecedeLaterWatchedEventsAndAreNotReplayed) {
        click();
        SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
        const auto event = [&](const char* value, Uint64 stamp) {
            SDL_Event result{};
            result.type = SDL_EVENT_TEXT_INPUT;
            result.text.windowID = SDL_GetWindowID(nativeWindow());
            result.text.text = value;
            result.common.timestamp = stamp;
            return result;
        };
        auto a = event("a", 100);
        watchNative(a);
        ASSERT_EQ(SDL_PeepEvents(&a, 1, SDL_ADDEVENT, 0, 0), 1);
        auto worker = std::async(std::launch::async, [&] {
            auto b = event("b", 101);
            return SDL_PeepEvents(&b, 1, SDL_ADDEVENT, 0, 0);
        });
        ASSERT_EQ(worker.get(), 1);
        auto c = event("c", 102);
        watchNative(c);
        ASSERT_EQ(SDL_PeepEvents(&c, 1, SDL_ADDEVENT, 0, 0), 1);
        poll();
        EXPECT_EQ(field_->GetValue(), "abc");
    }

    TEST_F(WindowInputDispatchTest, QueuedWakeReturnsWithoutAnAdditionalIdleWait) {
        SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST);
        window_->wakeEventLoop();
        const auto start = std::chrono::steady_clock::now();
        wait(.2);
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(100));
    }

    TEST_F(WindowInputDispatchTest, SequencerMenuAndViewportEditsReceiveKeysWithoutDomFocus) {
        SequencerController controller;
        gui::RmlSequencerOverlay overlay(controller, &manager());
        overlay.showContextMenu(200, 100, std::nullopt, 0, gui::SequencerViewportEditMode::None);
        ASSERT_TRUE(overlay.isContextMenuOpen());
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(overlay.isContextMenuOpen());
        overlay.hideContextMenu();
        overlay.updateEditOverlay(0, 1, 1, 400, 0);
        key(SDL_SCANCODE_U);
        auto action = overlay.consumeAction();
        ASSERT_TRUE(action);
        EXPECT_EQ(action->action, gui::RmlSequencerOverlay::Action::APPLY_EDIT);
        key(SDL_SCANCODE_ESCAPE);
        action = overlay.consumeAction();
        ASSERT_TRUE(action);
        EXPECT_EQ(action->action, gui::RmlSequencerOverlay::Action::REVERT_EDIT);
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, ExplicitOcclusionBlocksViewportPressAndItsRelease) {
        gui::RmlViewportOverlay overlay;
        overlay.init(&manager());
        overlay.setViewportBounds({0, 0}, {400, 300}, {0, 0});
        auto* context = Rml::GetContext("viewport_overlay");
        ASSERT_NE(context, nullptr);
        auto* document = context->LoadDocumentFromMemory(
            "<rml><head><style>body { width:400px; height:300px; font-family:Inter; }"
            "input { position:absolute; left:220px; top:100px; width:100px; height:40px; }"
            "</style></head><body><input id='covered' type='text'/></body></rml>");
        ASSERT_NE(document, nullptr);
        document->Show();
        context->Update();
        auto* field = document->GetElementById("covered");
        struct PointerListener : Rml::EventListener {
            std::vector<Rml::String> events;
            void ProcessEvent(Rml::Event& event) override { events.push_back(event.GetType()); }
        } listener;
        field->AddEventListener("mousedown", &listener);
        field->AddEventListener("mouseup", &listener);
        bool blocked = true;
        overlay.processInput({}, [&](float x, float y) { return blocked && x >= 200 && y >= 80; });
        SDL_Event press{};
        press.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        press.button.button = SDL_BUTTON_LEFT;
        press.button.x = 240;
        press.button.y = 115;
        dispatch(press);
        EXPECT_NE(context->GetFocusElement(), field);
        blocked = false;
        press.type = SDL_EVENT_MOUSE_BUTTON_UP;
        dispatch(press);
        EXPECT_TRUE(listener.events.empty());
        click(240, 115);
        EXPECT_EQ(context->GetFocusElement(), field);
        EXPECT_EQ(listener.events.size(), 2);
        field->RemoveEventListener("mousedown", &listener);
        field->RemoveEventListener("mouseup", &listener);
    }

    TEST_F(WindowInputDispatchTest, SequencerEditShortcutsRespectTextFocusAndVisibility) {
        SequencerController controller;
        gui::RmlSequencerOverlay overlay(controller, &manager());
        overlay.updateEditOverlay(0, 1, 1, 400, 0);
        click();
        key(SDL_SCANCODE_U);
        text("u");
        EXPECT_EQ(field_->GetValue(), "u");
        EXPECT_FALSE(overlay.consumeAction());
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(overlay.consumeAction());
        key(SDL_SCANCODE_U);
        ASSERT_TRUE(overlay.consumeAction());
        overlay.hideEditOverlay();
        key(SDL_SCANCODE_U);
        EXPECT_FALSE(overlay.consumeAction());
        EXPECT_FALSE(manager().wantsCaptureKeyboard());
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, SequencerOwnerCanBeDestroyedAfterUiShutdown) {
        SequencerController controller;
        gui::RmlSequencerOverlay overlay(controller, &manager());
        overlay.showContextMenu(200, 100, std::nullopt, 0, gui::SequencerViewportEditMode::None);
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(overlay.isContextMenuOpen());
        revert_.clear();
        manager().shutdown();
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, SequencerEditShortcutsFallThroughFromUnconsumedContainerFocus) {
        SequencerController controller;
        gui::RmlSequencerOverlay overlay(controller, &manager());
        overlay.updateEditOverlay(0, 1, 1, 400, 0);
        auto* container = document_->AppendChild(document_->CreateElement("div"));
        ASSERT_TRUE(container->Focus());
        ASSERT_EQ(context_->GetFocusElement(), container);
        key(SDL_SCANCODE_U);
        auto action = overlay.consumeAction();
        ASSERT_TRUE(action);
        EXPECT_EQ(action->action, gui::RmlSequencerOverlay::Action::APPLY_EDIT);
        key(SDL_SCANCODE_ESCAPE);
        action = overlay.consumeAction();
        ASSERT_TRUE(action);
        EXPECT_EQ(action->action, gui::RmlSequencerOverlay::Action::REVERT_EDIT);
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, AcceptedDragKeepsMotionAndReleaseAcrossOcclusion) {
        auto* handle = document_->AppendChild(document_->CreateElement("div"));
        handle->SetProperty("position", "absolute");
        handle->SetProperty("left", "220px");
        handle->SetProperty("top", "100px");
        handle->SetProperty("width", "50px");
        handle->SetProperty("height", "50px");
        handle->SetProperty("drag", "drag");
        context_->Update();
        struct DragListener : Rml::EventListener {
            int starts = 0, moves = 0, ends = 0;
            float x = 0;
            void ProcessEvent(Rml::Event& event) override {
                if (event.GetType() == "dragstart")
                    ++starts;
                else if (event.GetType() == "drag") {
                    ++moves;
                    x = event.GetParameter("mouse_x", 0.0f);
                } else if (event.GetType() == "dragend")
                    ++ends;
            }
        } listener;
        for (const auto* type : {"dragstart", "drag", "dragend"})
            handle->AddEventListener(type, &listener);
        const auto forward = [&](const gui::PanelInputState& input) {
            context_->ProcessMouseMove(input.mouse_x, input.mouse_y, 0);
            for (const auto& button : input.mouse_button_events) {
                if (button.down)
                    context_->ProcessMouseButtonDown(button.button, 0);
                else
                    context_->ProcessMouseButtonUp(button.button, 0);
            }
        };
        const auto blocked = [](float x, float) { return x >= 280; };
        manager().routeInput(context_, {}, forward, false, blocked);
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.button = SDL_BUTTON_LEFT;
        event.button.x = 240;
        event.button.y = 115;
        dispatch(event);
        const auto move = [&](float x, float y) {
            SDL_Event motion{};
            motion.type = SDL_EVENT_MOUSE_MOTION;
            motion.motion.x = x;
            motion.motion.y = y;
            dispatch(motion);
        };
        move(260, 130);
        EXPECT_EQ(listener.starts, 1);
        move(300, 160);
        EXPECT_EQ(listener.x, 300);
        // A render pass may still provide hover-masked coordinates mid-drag.
        gui::PanelInputState masked;
        masked.mouse_x = masked.mouse_y = -1e9f;
        manager().routeInput(context_, masked, forward, false, blocked);
        EXPECT_EQ(listener.x, 300);
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.x = 300;
        event.button.y = 160;
        dispatch(event);
        EXPECT_EQ(listener.ends, 1);
        const int moves_after_release = listener.moves;
        move(240, 220);
        EXPECT_EQ(listener.moves, moves_after_release);
        for (const auto* type : {"dragstart", "drag", "dragend"})
            handle->RemoveEventListener(type, &listener);
    }
} // namespace lfs::vis

namespace lfs::vis {
    TEST_F(WindowInputDispatchTest, ExpiredExclusiveOverlayDoesNotBlockAreaPresses) {
        auto* overlay = manager().createContext("expired-overlay", 400, 300);
        manager().routeInput(overlay, {}, [](const gui::PanelInputState&) {}, true);
        manager().beginFrameCursorTracking();
        int presses = 0;
        int releases = 0;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            presses += input.mouse_clicked[0];
            releases += input.mouse_released[0];
        });
        click();
        EXPECT_EQ(presses, 1);
        EXPECT_EQ(releases, 1);
    }

    TEST_F(WindowInputDispatchTest, AcceptedPointerReleaseSurvivesExclusiveFocusChange) {
        int releases = 0;
        float motion_x = 0;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            motion_x = input.mouse_x;
            releases += input.mouse_released[0];
        });
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.button = SDL_BUTTON_LEFT;
        event.button.x = event.button.y = 25;
        dispatch(event);
        auto* modal = manager().createContext("pointer-modal", 400, 300);
        manager().activateInput(modal, [](const gui::PanelInputState&) {});
        SDL_Event motion{};
        motion.type = SDL_EVENT_MOUSE_MOTION;
        motion.motion.x = 300;
        motion.motion.y = 200;
        dispatch(motion);
        EXPECT_EQ(motion_x, 300);
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        event.button.x = 300;
        event.button.y = 200;
        dispatch(event);
        EXPECT_EQ(releases, 1);
        manager().deactivateInput(modal);
    }
} // namespace lfs::vis

namespace lfs::vis {
    class PointerCancellationDispatchTest : public WindowInputDispatchTest {
    protected:
        struct Listener : Rml::EventListener {
            int starts = 0, moves = 0, ends = 0, drops = 0, clicks = 0, cancelled_ends = 0;
            std::function<void(Rml::Event&)> callback;
            void ProcessEvent(Rml::Event& event) override {
                const auto& type = event.GetType();
                starts += type == "dragstart";
                moves += type == "drag";
                ends += type == "dragend";
                cancelled_ends += type == "dragend" && event.GetParameter("cancelled", false);
                drops += type == "dragdrop";
                clicks += type == "click";
                if (callback)
                    callback(event);
            }
        } listener_;
        Rml::ObserverPtr<Rml::Element> handle_{nullptr};
        float delivered_x_ = 0;
        bool delivered_down_ = false;

        void SetUp() override {
            WindowInputDispatchTest::SetUp();
            auto* handle = document_->AppendChild(document_->CreateElement("div"));
            handle_ = handle->GetObserverPtr();
            for (const auto& [name, value] : std::initializer_list<std::pair<const char*, const char*>>{
                     {"position", "absolute"},
                     {"left", "220px"},
                     {"top", "100px"},
                     {"width", "50px"},
                     {"height", "50px"},
                     {"drag", "drag-drop"}})
                handle->SetProperty(name, value);
            for (const auto* type : {"dragstart", "drag", "dragend", "click"})
                handle->AddEventListener(type, &listener_);
            document_->AddEventListener("dragdrop", &listener_, true);
            context_->Update();
            registerPointer();
        }
        void TearDown() override {
            if (handle_) {
                handle_->GetOwnerDocument()->RemoveEventListener("dragdrop", &listener_, true);
                for (const auto* type : {"dragstart", "drag", "dragend", "click"})
                    handle_->RemoveEventListener(type, &listener_);
            }
            WindowInputDispatchTest::TearDown();
        }
        void registerPointer() {
            manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
                delivered_x_ = input.mouse_x;
                delivered_down_ = input.mouse_down[0];
                context_->ProcessMouseMove(input.mouse_x, input.mouse_y, 0);
                for (const auto& button : input.mouse_button_events) {
                    if (button.down)
                        context_->ProcessMouseButtonDown(button.button, 0);
                    else
                        context_->ProcessMouseButtonUp(button.button, 0);
                } }, false, [](float x, float) { return x >= 280; });
        }
        void pointer(Uint32 type, float x = 240, float y = 115) {
            SDL_Event event{};
            event.type = type;
            if (type == SDL_EVENT_MOUSE_MOTION) {
                event.motion.x = x;
                event.motion.y = y;
            } else {
                event.button.button = SDL_BUTTON_LEFT;
                event.button.x = x;
                event.button.y = y;
            }
            dispatch(event);
        }
        void drainLifecycle() {
            SDL_Event event{};
            event.type = SDL_EVENT_WINDOW_FOCUS_GAINED;
            dispatch(event);
        }
        void startDrag() {
            pointer(SDL_EVENT_MOUSE_BUTTON_DOWN);
            pointer(SDL_EVENT_MOUSE_MOTION, 250, 130);
            ASSERT_EQ(listener_.starts, 1);
        }
        void expectCancelled() {
            EXPECT_EQ(listener_.ends, 1);
            EXPECT_EQ(listener_.cancelled_ends, 1);
            EXPECT_EQ(listener_.drops, 0);
            EXPECT_EQ(listener_.clicks, 0);
        }
        void expectNoDragAfterReopen() {
            registerPointer();
            EXPECT_FALSE(delivered_down_);
            const int moves = listener_.moves;
            pointer(SDL_EVENT_MOUSE_MOTION, 310, 210);
            EXPECT_LT(delivered_x_, 0);
            EXPECT_FALSE(delivered_down_);
            pointer(SDL_EVENT_MOUSE_MOTION, 250, 140);
            EXPECT_FALSE(delivered_down_);
            EXPECT_EQ(listener_.moves, moves);
            expectCancelled();
        }
        void releaseAndReopen() {
            pointer(SDL_EVENT_MOUSE_BUTTON_UP, 300, 200);
            expectNoDragAfterReopen();
        }
    };

    TEST_F(PointerCancellationDispatchTest, DeactivationCancelsBeforeReleaseAndReopen) {
        startDrag();
        manager().deactivateInput(context_);
        drainLifecycle();
        expectCancelled();
        releaseAndReopen();
    }

    TEST_F(PointerCancellationDispatchTest, FocusLossCancelsBeforeReleaseAndRefocus) {
        startDrag();
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
        dispatch(event);
        expectCancelled();
        event.type = SDL_EVENT_WINDOW_FOCUS_GAINED;
        dispatch(event);
        // The physical release happened outside the application and was lost.
        expectNoDragAfterReopen();
    }

    TEST_F(PointerCancellationDispatchTest, HidingOwningDocumentCancelsDrag) {
        startDrag();
        document_->Hide();
        drainLifecycle();
        expectCancelled();
        document_->Show();
        context_->Update();
        releaseAndReopen();
    }

    TEST_F(PointerCancellationDispatchTest, UnloadingOwningDocumentEndsDragBeforeRemoval) {
        startDrag();
        revert_.clear();
        document_->Close();
        drainLifecycle();
        expectCancelled();
        context_->Update();
        releaseAndReopen();
    }

    TEST_F(PointerCancellationDispatchTest, UnloadingSecondaryPressDocumentEndsPrimaryDragOnce) {
        startDrag();
        auto* other = context_->LoadDocumentFromMemory(
            "<rml><head><style>body { position:absolute; left:0px; top:180px; width:100px; height:100px; }"
            "</style></head><body/></rml>");
        ASSERT_NE(other, nullptr);
        other->Show();
        context_->Update();
        ASSERT_EQ(context_->GetElementAtPoint({40, 200})->GetOwnerDocument(), other);
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.button = SDL_BUTTON_RIGHT;
        event.button.x = 40;
        event.button.y = 200;
        dispatch(event);
        other->Close();
        drainLifecycle();
        expectCancelled();
        context_->Update();
        event.type = SDL_EVENT_MOUSE_BUTTON_UP;
        dispatch(event);
        releaseAndReopen();
    }

    TEST_F(PointerCancellationDispatchTest, DestroyingContextRetiresGestureBeforeRemoval) {
        startDrag();
        revert_.clear();
        manager().destroyContext("dispatch-field");
        drainLifecycle();
        EXPECT_EQ(listener_.ends, 0);
        EXPECT_EQ(listener_.drops, 0);
        pointer(SDL_EVENT_MOUSE_BUTTON_UP, 300, 200);
        EXPECT_FALSE(handle_);
    }

    TEST_F(PointerCancellationDispatchTest, HidingAnotherDocumentKeepsAcceptedDrag) {
        auto* other = context_->LoadDocumentFromMemory("<rml><body/></rml>");
        other->Show();
        startDrag();
        other->Hide();
        EXPECT_EQ(listener_.ends, 0);
        pointer(SDL_EVENT_MOUSE_BUTTON_UP, 300, 200);
        EXPECT_EQ(listener_.ends, 1);
    }

    TEST_F(PointerCancellationDispatchTest, HideDuringDragStartCancelsAfterRmlReturns) {
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart")
                document_->Hide();
        };
        startDrag();
        expectCancelled();
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, CancelCallbackCanDestroyItsContext) {
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragend") {
                revert_.clear();
                manager().destroyContext("dispatch-field");
                EXPECT_NE(Rml::GetContext("dispatch-field"), nullptr);
            }
        };
        startDrag();
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
        dispatch(event);
        EXPECT_EQ(manager().getContext("dispatch-field"), nullptr);
        EXPECT_FALSE(handle_);
        expectCancelled();
    }

    TEST_F(PointerCancellationDispatchTest, DestroyDuringDragStartWaitsForRmlCaller) {
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart") {
                revert_.clear();
                manager().destroyContext("dispatch-field");
                EXPECT_NE(Rml::GetContext("dispatch-field"), nullptr);
            }
        };
        startDrag();
        EXPECT_EQ(manager().getContext("dispatch-field"), nullptr);
        EXPECT_FALSE(handle_);
        EXPECT_EQ(listener_.ends, 0);
        EXPECT_EQ(listener_.drops, 0);
    }

    TEST_F(PointerCancellationDispatchTest, EarlierRootListenerNeverReceivesCancelledDrop) {
        auto* root = context_->GetRootElement();
        root->AddEventListener("dragdrop", &listener_, true);
        startDrag();
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
        dispatch(event);
        expectCancelled();
        root->RemoveEventListener("dragdrop", &listener_, true);
    }

    TEST_F(PointerCancellationDispatchTest, MouseoutRestoringHoverCannotClickOnCancel) {
        bool restored = false;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "mouseout" && !restored) {
                restored = true;
                context_->ProcessMouseMove(240, 115, 0);
            }
        };
        auto* root = context_->GetRootElement();
        pointer(SDL_EVENT_MOUSE_BUTTON_DOWN);
        root->AddEventListener("mouseout", &listener_, true);
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
        dispatch(event);
        EXPECT_TRUE(restored);
        pointer(SDL_EVENT_MOUSE_BUTTON_UP);
        EXPECT_EQ(listener_.clicks, 0);
        EXPECT_EQ(listener_.drops, 0);
        root->RemoveEventListener("mouseout", &listener_, true);
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, HidingPayloadSourceCannotReleaseIntoViewport) {
        uint64_t token = 0;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart") {
                token = manager().beginDragPayload("project", "drag-project", "Project");
            } else if (event.GetType() == "dragend")
                manager().endDragPayload(token);
        };
        startDrag();
        document_->Hide();
        drainLifecycle();
        EXPECT_NE(token, 0);
        EXPECT_FALSE(manager().dragPayload());
        EXPECT_FALSE(manager().takeReleasedDragPayload());
        expectCancelled();
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, DeactivatingPayloadSourceCannotReleaseIntoViewport) {
        uint64_t token = 0;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart") {
                token = manager().beginDragPayload("project", "drag-project", "Project");
            } else if (event.GetType() == "dragend")
                manager().endDragPayload(token);
        };
        startDrag();
        manager().deactivateInput(context_);
        drainLifecycle();
        EXPECT_NE(token, 0);
        EXPECT_FALSE(manager().dragPayload());
        EXPECT_FALSE(manager().takeReleasedDragPayload());
        expectCancelled();
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, HideBeforePayloadCreationRejectsPayload) {
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart") {
                document_->Hide();
                EXPECT_EQ(manager().beginDragPayload("project", "drag-project", "Project"), 0);
            }
        };
        startDrag();
        expectCancelled();
        EXPECT_FALSE(manager().dragPayload());
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, HideDuringReleaseDragStartRejectsPayload) {
        uint64_t token = 0;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart") {
                document_->Hide();
                token = manager().beginDragPayload("project", "drag-project", "Project");
                EXPECT_EQ(token, 0);
            } else if (event.GetType() == "dragend")
                EXPECT_FALSE(manager().endDragPayload(token));
        };
        pointer(SDL_EVENT_MOUSE_BUTTON_DOWN);
        // The release position starts the drag before the button-up is delivered.
        pointer(SDL_EVENT_MOUSE_BUTTON_UP, 250, 130);
        EXPECT_EQ(listener_.starts, 1);
        EXPECT_EQ(listener_.ends, 1);
        EXPECT_FALSE(manager().dragPayload());
        EXPECT_FALSE(manager().takeReleasedDragPayload());
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, ReleaseDragStartWithoutHideReleasesPayload) {
        uint64_t token = 0;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart")
                token = manager().beginDragPayload("project", "drag-project", "Project");
            else if (event.GetType() == "dragend")
                EXPECT_TRUE(manager().endDragPayload(token));
        };
        pointer(SDL_EVENT_MOUSE_BUTTON_DOWN);
        pointer(SDL_EVENT_MOUSE_BUTTON_UP, 250, 130);
        EXPECT_EQ(listener_.starts, 1);
        EXPECT_EQ(listener_.ends, 1);
        EXPECT_NE(token, 0);
        auto released = manager().takeReleasedDragPayload();
        ASSERT_TRUE(released);
        EXPECT_EQ(released->token, token);
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, CancelCallbackCanCloseDocument) {
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragend") {
                revert_.clear();
                document_->Close();
            }
        };
        startDrag();
        manager().deactivateInput(context_);
        drainLifecycle();
        expectCancelled();
        context_->Update();
        EXPECT_FALSE(handle_);
        listener_.callback = {};
    }

    TEST_F(PointerCancellationDispatchTest, ReopenedPanelSuppressesCancelledPressUntilRelease) {
        startDrag();
        document_->Hide();
        drainLifecycle();
        document_->Show();
        context_->Update();
        gui::PanelInputState held;
        held.mouse_down[0] = true;
        manager().routeInput(context_, held, [&](const gui::PanelInputState& input) {
            EXPECT_FALSE(input.mouse_down[0]);
            EXPECT_TRUE(input.mouse_button_events.empty());
        });
        pointer(SDL_EVENT_MOUSE_BUTTON_UP);
        expectCancelled();
    }

    TEST_F(PointerCancellationDispatchTest, RetiredCancellationCannotAffectReplacementContext) {
        startDrag();
        document_->Hide();
        revert_.clear();
        manager().destroyContext("dispatch-field");
        auto* replacement = manager().createContext("dispatch-field", 400, 300);
        ASSERT_NE(replacement, nullptr);
        auto* document = replacement->LoadDocumentFromMemory("<rml><body>replacement</body></rml>");
        document->Show();
        drainLifecycle();
        EXPECT_EQ(manager().getContext("dispatch-field"), replacement);
        EXPECT_TRUE(document->IsVisible());
        EXPECT_EQ(listener_.ends, 0);
        EXPECT_FALSE(handle_);
    }

    TEST_F(PointerCancellationDispatchTest, UnrelatedDeactivationPreservesNormalPayloadRelease) {
        uint64_t token = 0;
        listener_.callback = [&](Rml::Event& event) {
            if (event.GetType() == "dragstart")
                token = manager().beginDragPayload("project", "drag-project", "Project");
            else if (event.GetType() == "dragend")
                manager().endDragPayload(token);
        };
        startDrag();
        auto* other = manager().createContext("other-payload-context", 400, 300);
        manager().activateInput(other, [](const gui::PanelInputState&) {}, false);
        manager().deactivateInput(other);
        drainLifecycle();
        ASSERT_TRUE(manager().dragPayload());
        EXPECT_EQ(manager().dragPayload()->token, token);
        pointer(SDL_EVENT_MOUSE_BUTTON_UP, 300, 200);
        auto released = manager().takeReleasedDragPayload();
        ASSERT_TRUE(released);
        EXPECT_EQ(released->token, token);
        EXPECT_EQ(listener_.ends, 1);
        EXPECT_EQ(listener_.cancelled_ends, 0);
        listener_.callback = {};
    }
} // namespace lfs::vis

namespace lfs::vis {
    class ScenePanelRefreshTest : public WindowInputDispatchTest {
    protected:
        bool keepSceneHandlers() const override { return true; }

        void checkMutation(const bool cached, const bool add_group, const bool from_menu = false) {
            auto& scene_manager = *viewer_->getSceneManager();
            auto& scene = scene_manager.getScene();
            const auto original = scene.addGroup("Original");
            gui::NativeScenePanel panel(&manager());
            gui::PanelDrawContext ctx;
            ctx.scene = &scene;
            ctx.scene_generation = python::get_scene_generation();
            ctx.frame_serial = 1;
            panel.preload(ctx);
            auto* context = manager().getContext("scene_panel_native");
            ASSERT_NE(context, nullptr);
            auto* tree = dynamic_cast<gui::SceneGraphElement*>(
                context->GetDocument(0)->GetElementById("tree-container"));
            ASSERT_NE(tree, nullptr);
            ASSERT_EQ(tree->nodeCount(), 1u);
            auto& ledger = viewer_->getRenderingManager()->frameDemandLedger();
            const auto consume = [&] {
                static_cast<void>(app_store().store().drain_dirty_into_frame());
                return ledger.plan(FrameClock::now());
            };
            static_cast<void>(consume());

            // The frame context was captured before the action mutated the scene.
            if (from_menu) {
                const auto action = add_group ? std::string("scene_panel:add_group_root")
                                              : "scene_panel:duplicate:" + std::to_string(original);
                menu().request({{"Create node", action}}, 0, 0);
                auto* menu_context = manager().getContext("global_context_menu");
                ASSERT_NE(menu_context, nullptr);
                auto* item = menu_context->GetDocument(0)->QuerySelector("[data-ctx-action]");
                ASSERT_NE(item, nullptr);
                item->DispatchEvent("click", {});
            } else {
                if (add_group)
                    scene_manager.addGroupNode("Added");
                else
                    ASSERT_FALSE(scene_manager.duplicateNodeTree(original).empty());
                EXPECT_TRUE(consume().present);
            }
            gui::PanelInputState input;
            input.mouse_clicked[0] = from_menu;
            gui::PanelDirectRenderRequest request;
            request.input = cached ? nullptr : &input;
            request.mode = cached ? gui::PanelDirectRenderMode::Cached : gui::PanelDirectRenderMode::Preload;
            request.width = 400;
            request.height = 300;
            panel.renderDirect(request, ctx);
            EXPECT_EQ(tree->nodeCount(), 2u);
            if (from_menu)
                EXPECT_TRUE(consume().present);

            // A fresh idle frame must neither retain stale rows nor request more frames.
            ctx.scene_generation = python::get_scene_generation();
            ++ctx.frame_serial;
            request.mode = gui::PanelDirectRenderMode::Cached;
            request.input = nullptr;
            panel.renderDirect(request, ctx);
            EXPECT_EQ(tree->nodeCount(), 2u);
            EXPECT_TRUE(consume().empty());
            EXPECT_FALSE(ledger.nextDeadline(FrameClock::now()).has_value());
        }
    };

    TEST_F(ScenePanelRefreshTest, UnchangedPanelStaysIdle) {
        auto& scene = viewer_->getSceneManager()->getScene();
        scene.addGroup("Existing");
        gui::NativeScenePanel panel(&manager());
        gui::PanelDrawContext ctx;
        ctx.scene = &scene;
        ctx.scene_generation = python::get_scene_generation();
        panel.preload(ctx);
        auto& ledger = viewer_->getRenderingManager()->frameDemandLedger();
        static_cast<void>(app_store().store().drain_dirty_into_frame());
        static_cast<void>(ledger.plan(FrameClock::now()));
        const auto start = std::chrono::steady_clock::now();
        constexpr int repeats = 10000;
        for (int i = 0; i < repeats; ++i) {
            panel.renderDirect({.mode = gui::PanelDirectRenderMode::Cached,
                                .width = 400,
                                .height = 300},
                               ctx);
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        RecordProperty("idle_update_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() / repeats);
        EXPECT_TRUE(ledger.plan(FrameClock::now()).empty());
        EXPECT_FALSE(ledger.nextDeadline(FrameClock::now()).has_value());
        const auto sync_start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i)
            panel.preload(ctx);
        const auto sync_elapsed = std::chrono::steady_clock::now() - sync_start;
        RecordProperty("live_sync_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(sync_elapsed).count() / repeats);
        EXPECT_TRUE(ledger.plan(FrameClock::now()).empty());
    }

    TEST_F(ScenePanelRefreshTest, ContextMenuDuplicateRefreshesLiveTree) {
        checkMutation(false, false, true);
    }
    TEST_F(ScenePanelRefreshTest, ContextMenuDuplicateRefreshesCachedTree) {
        checkMutation(true, false, true);
    }
    TEST_F(ScenePanelRefreshTest, ContextMenuAddGroupRefreshesCachedTree) {
        checkMutation(true, true, true);
    }
    TEST_F(ScenePanelRefreshTest, DuplicateRefreshesLiveTreeWithCapturedFrameContext) {
        checkMutation(false, false);
    }
    TEST_F(ScenePanelRefreshTest, DuplicateRefreshesCachedTreeWithCapturedFrameContext) {
        checkMutation(true, false);
    }
    TEST_F(ScenePanelRefreshTest, AddGroupRefreshesCachedTreeWithCapturedFrameContext) {
        checkMutation(true, true);
    }
} // namespace lfs::vis
