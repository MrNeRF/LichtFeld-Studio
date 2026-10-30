/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "core/point_cloud.hpp"
#include "core/services.hpp"
#include "gui/bounds_gizmo.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/global_context_menu.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/line_renderer.hpp"
#include "gui/line_renderer_overlays.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panel_layout.hpp"
#include "gui/rml_menu_bar.hpp"
#include "gui/rml_modal_overlay.hpp"
#include "gui/rml_right_panel.hpp"
#include "gui/rml_sequencer_overlay.hpp"
#include "gui/rmlui/elements/python_editor_element.hpp"
#include "gui/rmlui/elements/scene_graph_element.hpp"
#include "gui/rmlui/elements/terminal_element.hpp"
#include "gui/rmlui/rml_input_utils.hpp"
#include "gui/rmlui/rml_panel_host.hpp"
#include "gui/rmlui/rml_theme.hpp"
#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/scene_panel_native.hpp"
#include "gui/translation_gizmo.hpp"
#include "gui/viewport_gizmo_geometry.hpp"
#include "input/frame_input_buffer.hpp"
#include "input/input_controller.hpp"
#include "input/key_codes.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/render_constants.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "tools/unified_tool_registry.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"
#include "window/window_manager.hpp"
#include <RmlUi/Core.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/Elements/ElementFormControlSelect.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <glm/gtc/type_ptr.hpp>
#include <gtest/gtest.h>
#include <set>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <cstdlib>
#endif

#ifndef _WIN32
namespace lfs_test_lsp {
    std::filesystem::path write_mock_lsp_server();
}
#endif

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

    TEST_F(RmlPointerReplayTest, DotOffersCompletionAfterThePanelConsumesTheEdit) {
#ifdef _WIN32
        GTEST_SKIP() << "Mock language server fixture is POSIX-only.";
#else
        const auto script = lfs_test_lsp::write_mock_lsp_server();
        setenv("LFS_PYTHON_LSP", script.c_str(), 1);
        setenv("LFS_PYTHON_LSP_WORKSPACE", (script.parent_path() / "workspace").c_str(), 1);
        static Rml::ElementInstancerGeneric<gui::PythonEditorElement> instancer;
        Rml::Factory::RegisterElementInstancer("python-editor", &instancer);
        editor::PythonEditor editor;
        auto element = document_->CreateElement("python-editor");
        auto* view = dynamic_cast<gui::PythonEditorElement*>(element.get());
        ASSERT_NE(view, nullptr);
        view->setEditor(&editor);
        document_->AppendChild(std::move(element));
        context_->Update();
        ASSERT_TRUE(view->Focus());
        for (const char* piece : {"l", "f", "."})
            gui::rml_input::processKeyboardEvent(*context_, {.kind = FrameInputEventKind::Text, .text = piece});
        EXPECT_EQ(editor.getText(), "lf.");
        // The docked console consumes the edit before the deferred Rml render.
        EXPECT_TRUE(editor.consumeTextChanged());
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!editor.hasActiveCompletion() && std::chrono::steady_clock::now() < deadline) {
            editor.renderRml(*view, 200.0f, 80.0f, 14.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        EXPECT_TRUE(editor.hasActiveCompletion());
        view->setEditor(nullptr);
        unsetenv("LFS_PYTHON_LSP");
        unsetenv("LFS_PYTHON_LSP_WORKSPACE");
#endif
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
            }
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
            manager().vulkan_render_interface_ = nullptr;
            gui::PanelRegistry::instance().unregister_panel("test.input_float");
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
        bool hasDesktopHost(const Rml::Context* context) { return manager().native_windows_.contains(context); }
        void failExternalGraphics(int& attempts) {
            manager().native_windows_enabled_ = true;
            manager().create_native_graphics_ = [&attempts]() -> std::unique_ptr<VulkanContext> {
                ++attempts;
                return nullptr;
            };
        }
        void setKeyboardContext(Rml::Context* context, bool external_enabled) {
            manager().keyboard_context_ = context;
            manager().native_windows_enabled_ = external_enabled;
        }
        void mapContextWindow(Rml::Context* context, SDL_Window* window) {
            manager().context_windows_[context] = window;
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
        // GuiManager keeps these listeners alive until their Rml context is shut down.
        gui::RmlViewportOverlay& viewportOverlay() { return gui_->rml_viewport_overlay_; }
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
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
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
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
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
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
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

        void exerciseCropToolGizmoEscape(const bool ellipsoid, const bool crop_tool = true) {
            auto& sm = *viewer_->getSceneManager();
            auto& scene = sm.getScene();
            auto& gizmo = gui_->gizmo();
            auto& camera = viewer_->getViewport().camera;
            camera.t = {0.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);

            const auto shape = ellipsoid ? "ellipsoid" : "box";
            const auto parent = scene.addSplat(std::string("Escape crop parent ") + shape,
                                               lfs::test::licht::make_splat(3));
            ASSERT_NE(parent, core::NULL_NODE);
            const auto volume = ellipsoid ? scene.addEllipsoid(std::string("Escape crop ") + shape, parent)
                                          : scene.addCropBox(std::string("Escape crop ") + shape, parent);
            ASSERT_NE(volume, core::NULL_NODE);
            sm.changeContentType(SceneManager::ContentType::SplatFiles);
            sm.selectNode(volume);
            viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
            if (crop_tool) {
                UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
                gizmo.setCropToolShape(shape);
            } else {
                viewer_->getEditorContext().setActiveTool(ToolType::Translate);
                UnifiedToolRegistry::instance().setActiveTool("builtin.translate");
            }
            gizmo.setOperation(gui::GizmoOperation::Translate);
            gizmo.setTransformSpace(TransformSpace::World);
            const gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
            gizmo.updateToolState(ui, false);
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);

            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            auto& frame = frameInput();
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            const auto render = [&] {
                if (ellipsoid)
                    gizmo.renderEllipsoidGizmo(ui, layout);
                else
                    gizmo.renderCropBoxGizmo(ui, layout);
            };
            bool found = false;
            for (int y = 45; y <= 255 && !found; y += 3) {
                for (int x = 45; x <= 355 && !found; x += 3) {
                    frame.mouse_x = static_cast<float>(x);
                    frame.mouse_y = static_cast<float>(y);
                    gui::guiFocusState().want_capture_mouse = false;
                    render();
                    found = gui::isTranslationGizmoHovered();
                }
            }
            if (!crop_tool) {
                EXPECT_FALSE(found) << "Selecting a crop must not expose legacy transform handles";
                EXPECT_FALSE(gui::isTranslationGizmoActive());
                gizmo.deactivateAllTools();
                return;
            }
            ASSERT_TRUE(found) << "no crop " << shape << " translation handle found";

            const glm::mat4 before_transform = scene.getNodeTransform(volume);
            const auto before_box = scene.getNodeById(volume)->cropbox
                                        ? std::optional{*scene.getNodeById(volume)->cropbox}
                                        : std::nullopt;
            const auto before_ellipsoid = scene.getNodeById(volume)->ellipsoid
                                              ? std::optional{*scene.getNodeById(volume)->ellipsoid}
                                              : std::nullopt;
            op::undoHistory().clear();
            const size_t undo_before = op::undoHistory().undoCount();
            const glm::vec2 start(frame.mouse_x, frame.mouse_y);
            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_PRESS,
                                           start.x, start.y);
            frame.mouse_down[0] = true;
            frame.mouse_clicked[0] = true;
            frame.mouse_released[0] = false;
            render();
            frame.mouse_clicked[0] = false;
            frame.mouse_x = start.x + 24.0f;
            frame.mouse_y = start.y + 12.0f;
            render();
            ASSERT_TRUE(gui::isTranslationGizmoActive());

            controller_->handleKey(input::KEY_ESCAPE, input::ACTION_PRESS, input::KEYMOD_NONE);
            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                           frame.mouse_x, frame.mouse_y);
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            render();
            frame.mouse_released[0] = false;

            EXPECT_TRUE(sameMatrixBits(before_transform, scene.getNodeTransform(volume)));
            ASSERT_NE(scene.getNodeById(volume), nullptr);
            if (ellipsoid) {
                ASSERT_NE(scene.getNodeById(volume)->ellipsoid, nullptr);
                EXPECT_EQ(scene.getNodeById(volume)->ellipsoid->radii, before_ellipsoid->radii);
            } else {
                ASSERT_NE(scene.getNodeById(volume)->cropbox, nullptr);
                EXPECT_EQ(scene.getNodeById(volume)->cropbox->min, before_box->min);
                EXPECT_EQ(scene.getNodeById(volume)->cropbox->max, before_box->max);
            }
            EXPECT_EQ(op::undoHistory().undoCount(), undo_before);
            gizmo.deactivateAllTools();
            op::undoHistory().clear();
        }

        void exerciseViewportGizmoEscape() {
            auto& gizmo = gui_->gizmo();
            auto& camera = viewer_->getViewport().camera;
            camera.t = {0.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
            const glm::vec3 before_position = camera.t;
            const glm::vec3 before_pivot = camera.pivot;
            const glm::mat4 before_rotation(camera.R);
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);
            auto& frame = frameInput();
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            gui::guiFocusState().want_capture_mouse = false;

            bool found = false;
            for (int y = 12; y < 118 && !found; y += 2) {
                for (int x = 280; x < 398 && !found; x += 2) {
                    camera.t = before_position;
                    camera.pivot = before_pivot;
                    camera.R = glm::mat3(before_rotation);
                    frame.mouse_x = static_cast<float>(x);
                    frame.mouse_y = static_cast<float>(y);
                    frame.mouse_down[0] = frame.mouse_clicked[0] = true;
                    gizmo.renderViewportGizmo(layout);
                    found = gizmo.isViewportGizmoDragging();
                    frame.mouse_down[0] = frame.mouse_clicked[0] = false;
                    if (!found)
                        gizmo.renderViewportGizmo(layout);
                }
            }
            ASSERT_TRUE(found) << "no empty area of the viewport orientation gizmo found";

            op::undoHistory().clear();
            const size_t undo_before = op::undoHistory().undoCount();
            frame.mouse_x += 18.0f;
            frame.mouse_y += 11.0f;
            frame.mouse_dx = 18.0f;
            frame.mouse_dy = 11.0f;
            frame.mouse_down[0] = true;
            gizmo.renderViewportGizmo(layout);
            frame.mouse_dx = frame.mouse_dy = 0.0f;
            EXPECT_FALSE(sameMatrixBits(before_rotation, glm::mat4(camera.R)));

            controller_->handleKey(input::KEY_ESCAPE, input::ACTION_PRESS, input::KEYMOD_NONE);
            controller_->handleMouseButton(static_cast<int>(input::AppMouseButton::LEFT), input::ACTION_RELEASE,
                                           frame.mouse_x, frame.mouse_y);
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            gizmo.renderViewportGizmo(layout);
            frame.mouse_released[0] = false;

            EXPECT_TRUE(sameMatrixBits(before_rotation, glm::mat4(camera.R)));
            EXPECT_EQ(camera.t, before_position);
            EXPECT_EQ(camera.pivot, before_pivot);
            EXPECT_EQ(op::undoHistory().undoCount(), undo_before);
            EXPECT_FALSE(gizmo.isViewportGizmoDragging());
        }

        // The axis widget as drawn for the 400x300 test viewport, looking down -Z.
        gui::ViewportGizmoLayout prepareViewportGizmo() {
            auto& camera = viewer_->getViewport().camera;
            camera.t = {0.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
            controller_->setViewer(viewer_.get());
            controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);
            auto& frame = frameInput();
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            frame.mouse_dx = frame.mouse_dy = 0.0f;
            gui::guiFocusState().want_capture_mouse = false;
            return *gui::buildViewportGizmoLayout(viewer_->getViewport().getRotationMatrix(), {0.0f, 0.0f},
                                                  {400.0f, 300.0f}, 1.0f);
        }

        void pressViewportGizmo(const gui::ViewportLayout& layout, const glm::vec2& point) {
            auto& frame = frameInput();
            frame.mouse_x = point.x;
            frame.mouse_y = point.y;
            frame.mouse_down[0] = frame.mouse_clicked[0] = true;
            gui_->gizmo().renderViewportGizmo(layout);
            frame.mouse_clicked[0] = false;
        }

        // A point on the sphere outside every marker.
        static glm::vec2 viewportGizmoSpherePoint(const gui::ViewportGizmoLayout& gizmo_layout) {
            for (float y = gizmo_layout.top_left.y; y < gizmo_layout.top_left.y + gizmo_layout.size; y += 2.0f)
                for (float x = gizmo_layout.top_left.x; x < gizmo_layout.top_left.x + gizmo_layout.size; x += 2.0f)
                    if (gizmo_layout.discContains({x, y}) && gui::hitTestViewportGizmoMarkers(gizmo_layout.markers, {x, y}) < 0)
                        return {x, y};
            ADD_FAILURE() << "no free point on the axis widget sphere";
            return gizmo_layout.center;
        }

        void exerciseViewportGizmoEdgeDrag() {
            auto& gizmo = gui_->gizmo();
            auto& camera = viewer_->getViewport().camera;
            const auto gizmo_layout = prepareViewportGizmo();
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            auto& frame = frameInput();
            pressViewportGizmo(layout, viewportGizmoSpherePoint(gizmo_layout));
            ASSERT_TRUE(gizmo.isViewportGizmoDragging());

            // At the window edge SDL clamps the position, so only relative motion moves on.
            // Panels claiming the pointer must not stall the drag either.
            for (const bool ui_wants_mouse : {false, true}) {
                gui::guiFocusState().want_capture_mouse = ui_wants_mouse;
                const glm::mat4 before(camera.R);
                frame.mouse_dx = 25.0f;
                gizmo.renderViewportGizmo(layout);
                EXPECT_FALSE(sameMatrixBits(before, glm::mat4(camera.R))) << ui_wants_mouse;
            }
            frame.mouse_dx = 0.0f;
            frame.mouse_down[0] = false;
            gizmo.renderViewportGizmo(layout);
            EXPECT_FALSE(gizmo.isViewportGizmoDragging());
            gui::guiFocusState().want_capture_mouse = false;
        }

        void exerciseViewportGizmoMarkerPress() {
            auto& gizmo = gui_->gizmo();
            auto& camera = viewer_->getViewport().camera;
            const auto gizmo_layout = prepareViewportGizmo();
            const glm::vec2 x_marker = gizmo_layout.markers[0].screen_pos;
            ASSERT_EQ(gui::hitTestViewportGizmoMarkers(gizmo_layout.markers, x_marker), 0);
            const glm::mat4 before(camera.R);
            auto x_view = camera;
            x_view.setAxisAlignedView(0, false);
            const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            auto& frame = frameInput();

            // A click views along the axis on release.
            pressViewportGizmo(layout, x_marker);
            EXPECT_TRUE(sameMatrixBits(before, glm::mat4(camera.R)));
            frame.mouse_down[0] = false;
            gizmo.renderViewportGizmo(layout);
            EXPECT_TRUE(sameMatrixBits(glm::mat4(x_view.R), glm::mat4(camera.R)));

            // Pressing a marker and moving orbits like the sphere.
            prepareViewportGizmo();
            pressViewportGizmo(layout, x_marker);
            frame.mouse_x += 12.0f;
            frame.mouse_dx = 12.0f;
            gizmo.renderViewportGizmo(layout);
            frame.mouse_dx = 0.0f;
            EXPECT_TRUE(gizmo.isViewportGizmoDragging());
            EXPECT_FALSE(sameMatrixBits(before, glm::mat4(camera.R)));
            EXPECT_FALSE(sameMatrixBits(glm::mat4(x_view.R), glm::mat4(camera.R)));
            frame.mouse_down[0] = false;
            gizmo.renderViewportGizmo(layout);
            EXPECT_FALSE(gizmo.isViewportGizmoDragging());
        }

        // Each change of what the widget shows under the pointer wakes one frame.
        void exerciseViewportGizmoHoverDemand() {
            const auto gizmo_layout = prepareViewportGizmo();
            gui_->viewport_layout_ = {.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
            auto& gizmo = gui_->gizmo();
            const glm::vec2 away(200.0f, 200.0f);
            context_->ProcessMouseMove(static_cast<int>(away.x), static_cast<int>(away.y), 0);
            gui_->drawn_viewport_gizmo_hover_ = gizmo.viewportGizmoHover(away.x, away.y);
            ASSERT_FALSE(gui_->passiveMouseMoveNeedsRender(away.x, away.y));
            for (const glm::vec2 point : {viewportGizmoSpherePoint(gizmo_layout), gizmo_layout.markers[0].screen_pos, away}) {
                context_->ProcessMouseMove(static_cast<int>(point.x), static_cast<int>(point.y), 0);
                EXPECT_TRUE(gui_->passiveMouseMoveNeedsRender(point.x, point.y)) << point.x << "," << point.y;
                gui_->drawn_viewport_gizmo_hover_ = gizmo.viewportGizmoHover(point.x, point.y);
                EXPECT_FALSE(gui_->passiveMouseMoveNeedsRender(point.x, point.y)) << point.x << "," << point.y;
            }
        }

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

    TEST_F(WindowInputDispatchTest, MainWindowFocusLossPreservesFocusedFieldWithExternalWindowsOnOrOff) {
        for (const bool enabled : {false, true}) {
            SCOPED_TRACE(enabled);
            ASSERT_TRUE(field_->Focus());
            field_->SetValue("half-typed");
            setKeyboardContext(context_, enabled);
            SDL_Event event{};
            event.type = SDL_EVENT_WINDOW_FOCUS_LOST;
            event.window.windowID = SDL_GetWindowID(window_->getWindow());
            manager().dispatchInputEvent(event);
            EXPECT_EQ(context_->GetFocusElement(), field_);
            EXPECT_EQ(field_->GetValue(), "half-typed");
        }
    }

    TEST_F(WindowInputDispatchTest, UnvalidatedPanelsCannotCreateExternalHosts) {
        int attempts = 0;
        failExternalGraphics(attempts);
        for (const auto* name : {"lfs.asset_manager", "lfs.histogram", "plugin.custom", "modal"}) {
            auto* context = manager().createContext(name, 400, 300, true);
            ASSERT_NE(context, nullptr);
            EXPECT_FALSE(hasDesktopHost(context));
            EXPECT_TRUE(manager().externalWindowsAllowed());
            manager().destroyContext(name);
        }
        EXPECT_EQ(attempts, 0);
        EXPECT_TRUE(gui::RmlUIManager::validatedExternalPanel("lfs.preferences"));
        EXPECT_TRUE(gui::RmlUIManager::validatedExternalPanel("lfs.plugin_marketplace"));
        EXPECT_TRUE(gui::RmlUIManager::validatedExternalPanel("video_extractor"));
    }

    TEST_F(WindowInputDispatchTest, FailedExternalHostDisablesFurtherAttemptsForTheSession) {
        int attempts = 0;
        failExternalGraphics(attempts);
        auto* first = manager().createContext("video_extractor", 400, 300, true);
        ASSERT_NE(first, nullptr);
        EXPECT_FALSE(hasDesktopHost(first));
        EXPECT_FALSE(manager().externalWindowsAllowed());
        for (int frame = 0; frame < 10; ++frame) {
            const std::string name = "lfs.preferences";
            auto* fallback = manager().createContext(name, 400, 300, true);
            ASSERT_NE(fallback, nullptr);
            EXPECT_FALSE(hasDesktopHost(fallback));
            EXPECT_FALSE(manager().externalWindowsAllowed());
            manager().destroyContext(name);
        }
        EXPECT_EQ(attempts, 1);
        manager().destroyContext("video_extractor");
    }

    TEST_F(WindowInputDispatchTest, SecondaryWindowMouseDoesNotReachMainWindowContext) {
        auto* secondary = SDL_CreateWindow("Secondary input", 400, 300, SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        ASSERT_NE(secondary, nullptr);
        auto* context = manager().createContext("secondary-input", 400, 300);
        ASSERT_NE(context, nullptr);
        mapContextWindow(context, secondary);
        int main_presses = 0;
        int secondary_presses = 0;
        float secondary_x = 0.0f;
        float press_x = 0.0f;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            main_presses += static_cast<int>(input.mouse_button_events.size());
        });
        manager().routeInput(context, {}, [&](const gui::PanelInputState& input) {
            secondary_presses += static_cast<int>(input.mouse_button_events.size());
            if (!input.mouse_button_events.empty()) {
                secondary_x = input.mouse_x;
                press_x = input.mouse_button_events.front().x;
            }
        });
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.windowID = SDL_GetWindowID(secondary);
        event.button.button = SDL_BUTTON_LEFT;
        event.button.x = event.button.y = 20;
        manager().dispatchInputEvent(event);
        EXPECT_EQ(main_presses, 0);
        EXPECT_EQ(secondary_presses, 1);
        const auto scale = input::windowPixelScale(secondary);
        EXPECT_FLOAT_EQ(secondary_x, 20 * scale.x);
        EXPECT_FLOAT_EQ(press_x, 20 * scale.x);
        manager().destroyContext("secondary-input");
        SDL_DestroyWindow(secondary);
    }

    TEST_F(WindowInputDispatchTest, MainWindowTextCannotReachSecondaryExclusiveContext) {
        auto* secondary = SDL_CreateWindow("Secondary keyboard", 400, 300, SDL_WINDOW_HIDDEN);
        ASSERT_NE(secondary, nullptr);
        auto* context = manager().createContext("secondary-keyboard", 400, 300);
        ASSERT_NE(context, nullptr);
        mapContextWindow(context, secondary);
        int text_events = 0;
        manager().activateInput(context, [&](const gui::PanelInputState& input) {
            text_events += static_cast<int>(input.input_events.size());
        });
        SDL_Event event{};
        event.type = SDL_EVENT_TEXT_INPUT;
        event.text.windowID = SDL_GetWindowID(window_->getWindow());
        event.text.text = "main only";
        manager().dispatchInputEvent(event);
        EXPECT_EQ(text_events, 0);
        event.text.windowID = SDL_GetWindowID(secondary);
        manager().dispatchInputEvent(event);
        EXPECT_EQ(text_events, 1);
        manager().destroyContext("secondary-keyboard");
        SDL_DestroyWindow(secondary);
    }

    TEST_F(WindowInputDispatchTest, EmbeddedPointerCoordinatesConvertToPixelsOnlyOnce) {
        context_->SetDensityIndependentPixelRatio(2.0f);
        float move_x = -1, press_x = -1;
        manager().routeInput(context_, {}, [&](const gui::PanelInputState& input) {
            move_x = input.mouse_x;
            if (!input.mouse_button_events.empty())
                press_x = input.mouse_button_events.front().x;
        });
        SDL_Event event{};
        event.type = SDL_EVENT_MOUSE_MOTION;
        event.motion.windowID = SDL_GetWindowID(window_->getWindow());
        event.motion.x = 37;
        event.motion.y = 23;
        manager().dispatchInputEvent(event);
        const auto scale = input::windowPixelScale(window_->getWindow());
        EXPECT_FLOAT_EQ(move_x, 37 * scale.x);
        event = {};
        event.type = SDL_EVENT_MOUSE_BUTTON_DOWN;
        event.button.windowID = SDL_GetWindowID(window_->getWindow());
        event.button.button = SDL_BUTTON_LEFT;
        event.button.x = 37;
        event.button.y = 23;
        manager().dispatchInputEvent(event);
        EXPECT_FLOAT_EQ(move_x, 37 * scale.x);
        EXPECT_FLOAT_EQ(press_x, 37 * scale.x);
    }

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

    TEST_F(WindowInputDispatchTest, EscapeCancelsCropBoxGizmoDragAndAddsNoUndo) {
        exerciseCropToolGizmoEscape(false);
    }

    TEST_F(WindowInputDispatchTest, EscapeCancelsCropEllipsoidGizmoDragAndAddsNoUndo) {
        exerciseCropToolGizmoEscape(true);
    }

    TEST_F(WindowInputDispatchTest, SelectedCropBoxHasNoLegacyTransformHandles) {
        exerciseCropToolGizmoEscape(false, false);
    }

    TEST_F(WindowInputDispatchTest, SelectedEllipsoidHasNoLegacyTransformHandles) {
        exerciseCropToolGizmoEscape(true, false);
    }

    // The polygon crop tool edits its outline from vertex handles on the cap facing the
    // camera: a primary drag moves a vertex, winning over the bounds corner under it, and
    // the key map's insert and remove clicks add a vertex on an edge and delete one. Each
    // is one undo step, and the other vertices keep their world positions.
    TEST_F(WindowInputDispatchTest, CropPolygonVertexHandlesMoveInsertAndRemoveVertices) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = sm.getScene();
        auto& gizmo = gui_->gizmo();
        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);

        const auto parent = scene.addSplat("Polygon parent", lfs::test::licht::make_splat(3));
        core::CropPolygonData data;
        data.vertices = {{-3.0f, -2.0f}, {3.0f, -2.0f}, {3.0f, 2.0f}, {-3.0f, 2.0f}};
        data.half_depth = 0.5f;
        const auto polygon = scene.addCropPolygon("Polygon", parent, data);
        ASSERT_NE(polygon, core::NULL_NODE);
        sm.changeContentType(SceneManager::ContentType::SplatFiles);
        sm.selectNode(polygon);
        viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        gizmo.setCropToolShape("polygon");
        gizmo.setOperation(gui::GizmoOperation::Scale);
        const gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        gizmo.updateToolState(ui, false);
        controller_->setViewer(viewer_.get());
        controller_->updateViewportBounds(0.0f, 0.0f, 400.0f, 300.0f);
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};

        auto& frame = frameInput();
        frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
        frame.mouse_button_events.clear();
        frame.key_mods = SDL_KMOD_NONE;
        const auto render = [&] {
            gui::guiFocusState().want_capture_mouse = false;
            gizmo.renderEllipsoidGizmo(ui, layout);
        };
        // The handles as the tool projects them, on the cap facing the camera.
        const auto handles = [&] {
            const auto settings = viewer_->getRenderingManager()->getSettings();
            const auto& viewport = viewer_->getViewport();
            const glm::mat4 view = viewport.getViewMatrix();
            const glm::mat4 view_projection =
                rendering::createProjectionMatrixFromFocal(
                    glm::ivec2(400, 300), settings.focal_length_mm, settings.orthographic,
                    viewport.ortho_scale_override.value_or(settings.ortho_scale)) *
                view;
            const glm::mat4 to_world = scene_coords::nodeVisualizerWorldTransform(scene, polygon);
            const glm::vec3 eye(glm::inverse(view)[3]);
            const auto* const outline = scene.getCropPolygonData(polygon);
            const float z = glm::dot(glm::vec3(to_world[2]), eye - glm::vec3(to_world[3])) >= 0.0f
                                ? outline->half_depth
                                : -outline->half_depth;
            std::vector<glm::vec2> points;
            for (const glm::vec2& vertex : outline->vertices) {
                const glm::vec4 clip = view_projection * to_world * glm::vec4(vertex, z, 1.0f);
                points.emplace_back((clip.x / clip.w * 0.5f + 0.5f) * 400.0f, (0.5f - clip.y / clip.w * 0.5f) * 300.0f);
            }
            return points;
        };
        const auto world = [&] {
            std::vector<glm::vec3> points;
            for (const glm::vec2& vertex : scene.getCropPolygonData(polygon)->vertices)
                points.emplace_back(scene.getWorldTransform(polygon) * glm::vec4(vertex, 0.0f, 1.0f));
            return points;
        };
        const auto expectSamePoint = [](const glm::vec3& actual, const glm::vec3& expected) {
            EXPECT_NEAR(glm::distance(actual, expected), 0.0f, 1e-4f);
        };
        // Hover first, as a pointer does, then press with the given modifiers.
        const auto press = [&](const glm::vec2 point, const SDL_Keymod mods) {
            frame.mouse_x = point.x;
            frame.mouse_y = point.y;
            frame.key_mods = mods;
            render();
            frame.mouse_down[0] = frame.mouse_clicked[0] = true;
            frame.mouse_button_events = {{.button = 0, .down = true, .x = point.x, .y = point.y}};
            render();
            frame.mouse_clicked[0] = false;
            frame.mouse_button_events.clear();
        };
        const auto release = [&] {
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            frame.mouse_button_events = {{.button = 0, .down = false, .x = frame.mouse_x, .y = frame.mouse_y}};
            render();
            frame.mouse_released[0] = false;
            frame.mouse_button_events.clear();
            frame.key_mods = SDL_KMOD_NONE;
        };

        render();
        op::undoHistory().clear();
        const auto start = handles();
        ASSERT_EQ(start.size(), 4u);
        for (const glm::vec2& point : start) {
            ASSERT_GT(point.x, 20.0f);
            ASSERT_LT(point.x, 380.0f);
            ASSERT_GT(point.y, 20.0f);
            ASSERT_LT(point.y, 280.0f);
        }
        const auto drawn = world();

        // A primary drag on vertex 0 moves it with the pointer, not the bounds corner.
        press(start[0], SDL_KMOD_NONE);
        EXPECT_FALSE(gui::isBoundsGizmoActive());
        frame.mouse_x = start[0].x - 30.0f;
        frame.mouse_y = start[0].y + 20.0f;
        render();
        release();
        const auto moved = handles();
        ASSERT_EQ(moved.size(), 4u);
        EXPECT_NEAR(moved[0].x, start[0].x - 30.0f, 0.5f);
        EXPECT_NEAR(moved[0].y, start[0].y + 20.0f, 0.5f);
        const auto after_move = world();
        for (size_t i = 1; i < 4; ++i)
            expectSamePoint(after_move[i], drawn[i]);
        const glm::mat4 transform = scene.getNodeById(polygon)->local_transform.get();
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(glm::length(glm::vec3(transform[axis])), 1.0f, 1e-5f);
        EXPECT_EQ(op::undoHistory().undoCount(), 1u);

        // The insert click puts a vertex on the edge from vertex 1 to vertex 2, under the pointer.
        const glm::vec2 on_edge = glm::mix(moved[1], moved[2], 0.25f);
        press(on_edge, SDL_KMOD_SHIFT);
        release();
        const auto inserted = handles();
        ASSERT_EQ(inserted.size(), 5u);
        EXPECT_NEAR(inserted[2].x, on_edge.x, 0.5f);
        EXPECT_NEAR(inserted[2].y, on_edge.y, 0.5f);
        const auto after_insert = world();
        for (size_t i = 0; i < 5; ++i) {
            if (i != 2)
                expectSamePoint(after_insert[i], after_move[i < 2 ? i : i - 1]);
        }
        EXPECT_EQ(op::undoHistory().undoCount(), 2u);

        // The remove click deletes it again.
        press(inserted[2], SDL_KMOD_CTRL);
        release();
        ASSERT_EQ(handles().size(), 4u);
        const auto after_remove = world();
        for (size_t i = 0; i < 4; ++i)
            expectSamePoint(after_remove[i], after_move[i]);
        EXPECT_EQ(op::undoHistory().undoCount(), 3u);

        for (int step = 0; step < 3; ++step)
            ASSERT_TRUE(op::undoHistory().undo().success);
        EXPECT_EQ(scene.getCropPolygonData(polygon)->vertices, data.vertices);
        EXPECT_EQ(scene.getNodeById(polygon)->local_transform.get(), glm::mat4(1.0f));
        gizmo.deactivateAllTools();
        op::undoHistory().clear();
    }

    TEST_F(WindowInputDispatchTest, EscapeCancelsViewportGizmoDragAndRestoresCamera) {
        exerciseViewportGizmoEscape();
    }

    TEST_F(WindowInputDispatchTest, ViewportGizmoOrbitKeepsTurningAtTheWindowEdgeAndOverUi) {
        exerciseViewportGizmoEdgeDrag();
    }

    TEST_F(WindowInputDispatchTest, ViewportGizmoMarkerClickSnapsAndMarkerDragOrbits) {
        exerciseViewportGizmoMarkerPress();
    }

    TEST_F(WindowInputDispatchTest, ViewportGizmoHoverChangesRequestAFrame) {
        exerciseViewportGizmoHoverDemand();
    }

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

    TEST_F(WindowInputDispatchTest, CropScaleHandleChangesTransformAndPreservesExtentsUnderNestedParents) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = sm.getScene();
        auto& gizmo = gui_->gizmo();
        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        auto& frame = frameInput();
        for (const bool nested : {false, true}) {
            SCOPED_TRACE(nested);
            const auto outer = scene.addGroup(nested ? "Outer" : "Root");
            const auto inner = scene.addGroup(nested ? "Nested inner" : "Root inner", outer);
            if (nested) {
                auto transform = glm::rotate(glm::mat4(1.0f), 0.2f, glm::vec3(1, 0, 0));
                transform = glm::rotate(transform, -0.12f, glm::vec3(0, 1, 0));
                transform = glm::rotate(transform, 0.25f, glm::vec3(0, 0, 1));
                scene.setNodeTransform(outer, glm::scale(transform, glm::vec3(1.25f, 0.82f, 1.1f)));
            }
            const auto model = scene.addSplat(nested ? "Nested model" : "Root model", lfs::test::licht::make_splat(3), inner);
            const auto volume = scene.addCropBox(nested ? "Nested box" : "Root box", model);
            core::CropBoxData data;
            data.min = glm::vec3(-0.05f);
            data.max = glm::vec3(0.05f);
            if (!nested) {
                data.min.y = -1e-5f;
                data.max.y = 1e-5f;
            }
            scene.setCropBoxData(volume, data);
            sm.changeContentType(SceneManager::ContentType::SplatFiles);
            sm.selectNode(volume);
            viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
            UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
            gizmo.setCropToolShape("box");
            gizmo.setOperation(gui::GizmoOperation::Scale);
            gizmo.setTransformSpace(TransformSpace::World);
            gizmo.updateToolState(ui, false);
            frame.mouse_down[0] = frame.mouse_clicked[0] = frame.mouse_released[0] = false;
            bool found = false;
            for (int y = 105; y <= 195 && !found; ++y) {
                frame.mouse_x = 290.0f;
                frame.mouse_y = static_cast<float>(y);
                gizmo.renderCropBoxGizmo(ui, layout);
                found = gui::isScaleGizmoHovered();
            }
            ASSERT_TRUE(found);
            const glm::vec2 start(frame.mouse_x, frame.mouse_y);
            const auto drag_to = [&](const glm::vec2 end, const bool reverse = false) {
                frame.mouse_x = start.x;
                frame.mouse_y = start.y;
                frame.mouse_down[0] = frame.mouse_clicked[0] = true;
                gizmo.renderCropBoxGizmo(ui, layout);
                EXPECT_TRUE(gui::isScaleGizmoActive());
                frame.mouse_clicked[0] = false;
                if (reverse) {
                    frame.mouse_x = start.x - 200.0f;
                    gizmo.renderCropBoxGizmo(ui, layout);
                }
                frame.mouse_x = end.x;
                frame.mouse_y = end.y;
                gizmo.renderCropBoxGizmo(ui, layout);
                frame.mouse_down[0] = false;
                frame.mouse_released[0] = true;
                gizmo.renderCropBoxGizmo(ui, layout);
                frame.mouse_released[0] = false;
            };
            if (!nested) {
                op::undoHistory().clear();
                drag_to(start + glm::vec2(24.0f, 0.0f), true);
                const auto grown = *scene.getNodeById(volume)->cropbox;
                EXPECT_EQ(grown.min, data.min);
                EXPECT_EQ(grown.max, data.max);
                // The numeric scale field multiplies these same basis columns.
                const auto scaled_transform = sm.getNodeTransform(scene.getNodeById(volume)->name);
                EXPECT_NEAR(glm::length(glm::vec3(scaled_transform[0])), 1.25f, 1e-5f);
                EXPECT_NEAR(glm::length(glm::vec3(scaled_transform[1])), 1.0f, 1e-5f);
                ASSERT_TRUE(op::undoHistory().undo().success);
                EXPECT_EQ(scene.getNodeById(volume)->cropbox->min, data.min);
                EXPECT_NEAR(glm::length(glm::vec3(sm.getNodeTransform(scene.getNodeById(volume)->name)[0])), 1.0f, 1e-5f);
                ASSERT_TRUE(op::undoHistory().redo().success);
                EXPECT_EQ(scene.getNodeById(volume)->cropbox->min, grown.min);
                EXPECT_EQ(sm.getNodeTransform(scene.getNodeById(volume)->name), scaled_transform);
                ASSERT_TRUE(op::undoHistory().undo().success);
            }
            const auto world_scale = [&] {
                const glm::mat4 world = scene.getWorldTransform(volume);
                return glm::vec3(glm::length(glm::vec3(world[0])), glm::length(glm::vec3(world[1])),
                                 glm::length(glm::vec3(world[2])));
            };
            const glm::vec3 scale_before = world_scale();
            // Each drag can shrink to 1 %, so three drags fall far below the floor without the clamp.
            for (int drag = 0; drag < 3; ++drag)
                drag_to(start - glm::normalize(start - glm::vec2(200.0f, 150.0f)) * 200.0f);
            const auto* node = scene.getNodeById(volume);
            ASSERT_NE(node, nullptr);
            EXPECT_EQ(node->cropbox->min, data.min);
            EXPECT_EQ(node->cropbox->max, data.max);
            const glm::vec3 scale = world_scale();
            EXPECT_NEAR(scale.x, gui::gizmo_ops::MIN_VOLUME_AXIS_SCALE, 1e-6f);
            EXPECT_NEAR(scale.y, nested ? gui::gizmo_ops::MIN_VOLUME_AXIS_SCALE : scale_before.y, 1e-5f);
            EXPECT_NEAR(scale.z, scale_before.z, 1e-5f);
            gizmo.deactivateAllTools();
            op::undoHistory().clear();
        }
    }

    TEST_F(WindowInputDispatchTest, CropScaleHandleFloorIsReversibleWithinOneDrag) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = sm.getScene();
        auto& gizmo = gui_->gizmo();
        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
        const auto model = scene.addSplat("Floor model", lfs::test::licht::make_splat(3));
        const auto volume = scene.addCropBox("Floor box", model);
        core::CropBoxData data;
        data.min = glm::vec3(-0.5f);
        data.max = glm::vec3(0.5f);
        scene.setCropBoxData(volume, data);
        scene.setNodeTransform(volume, glm::scale(glm::mat4(1.0f), glm::vec3(0.05f, 1.0f, 1.0f)));
        sm.changeContentType(SceneManager::ContentType::SplatFiles);
        sm.selectNode(volume);
        viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        gizmo.setCropToolShape("box");
        gizmo.setOperation(gui::GizmoOperation::Scale);
        gizmo.setTransformSpace(TransformSpace::World);
        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        gizmo.updateToolState(ui, false);
        auto& frame = frameInput();
        bool found = false;
        for (int y = 105; y <= 195 && !found; ++y) {
            frame.mouse_x = 290.0f;
            frame.mouse_y = static_cast<float>(y);
            gizmo.renderCropBoxGizmo(ui, layout);
            found = gui::isScaleGizmoHovered();
        }
        ASSERT_TRUE(found);
        const glm::vec2 start(frame.mouse_x, frame.mouse_y);
        const auto x_scale = [&] { return glm::length(glm::vec3(scene.getWorldTransform(volume)[0])); };
        frame.mouse_down[0] = frame.mouse_clicked[0] = true;
        gizmo.renderCropBoxGizmo(ui, layout);
        ASSERT_TRUE(gui::isScaleGizmoActive());
        frame.mouse_clicked[0] = false;
        // Past the centre the gizmo factor bottoms out at 1 %, far below the 0.001 floor here.
        const glm::vec2 past_center = start - glm::normalize(start - glm::vec2(200.0f, 150.0f)) * 200.0f;
        frame.mouse_x = past_center.x;
        frame.mouse_y = past_center.y;
        gizmo.renderCropBoxGizmo(ui, layout);
        frame.mouse_x = start.x;
        frame.mouse_y = start.y;
        gizmo.renderCropBoxGizmo(ui, layout);
        frame.mouse_down[0] = false;
        frame.mouse_released[0] = true;
        gizmo.renderCropBoxGizmo(ui, layout);
        frame.mouse_released[0] = false;
        // Chaining per-frame deltas onto the clamped transform would end at 0.1 instead.
        EXPECT_NEAR(x_scale(), 0.05f, 1e-5f);
        gizmo.deactivateAllTools();
        op::undoHistory().clear();
    }

    TEST_F(WindowInputDispatchTest, EllipsoidScaleHandleMatchesNumericTransformScale) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = sm.getScene();
        auto& gizmo = gui_->gizmo();
        auto& camera = viewer_->getViewport().camera;
        camera.t = {0.0f, 0.0f, 12.0f};
        camera.pivot = {0.0f, 0.0f, 0.0f};
        camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
        const auto model = scene.addSplat("Scale model", lfs::test::licht::make_splat(3));
        const auto volume = scene.addEllipsoid("Scale ellipsoid", model);
        core::EllipsoidData data;
        data.radii = glm::vec3(0.05f);
        scene.setEllipsoidData(volume, data);
        sm.changeContentType(SceneManager::ContentType::SplatFiles);
        sm.selectNode(volume);
        viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        gizmo.setCropToolShape("ellipsoid");
        gizmo.setOperation(gui::GizmoOperation::Scale);
        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        gizmo.updateToolState(ui, false);
        const auto original = sm.getNodeTransform(scene.getNodeById(volume)->name);
        auto& frame = frameInput();
        bool found = false;
        for (int y = 105; y <= 195 && !found; ++y) {
            frame.mouse_x = 290.0f;
            frame.mouse_y = static_cast<float>(y);
            gizmo.renderEllipsoidGizmo(ui, layout);
            found = gui::isScaleGizmoHovered();
        }
        ASSERT_TRUE(found);
        frame.mouse_down[0] = frame.mouse_clicked[0] = true;
        gizmo.renderEllipsoidGizmo(ui, layout);
        ASSERT_TRUE(gui::isScaleGizmoActive());
        frame.mouse_clicked[0] = false;
        frame.mouse_x += 24.0f;
        gizmo.renderEllipsoidGizmo(ui, layout);
        frame.mouse_down[0] = false;
        frame.mouse_released[0] = true;
        gizmo.renderEllipsoidGizmo(ui, layout);
        frame.mouse_released[0] = false;
        const auto* node = scene.getNodeById(volume);
        EXPECT_EQ(node->ellipsoid->radii, data.radii);
        const auto expected = glm::scale(original, glm::vec3(1.25f, 1.0f, 1.0f));
        const auto actual = sm.getNodeTransform(node->name);
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                EXPECT_NEAR(actual[column][row], expected[column][row], 1e-5f);

        const glm::vec2 start(290.0f, frame.mouse_y);
        const glm::vec2 end = start - glm::normalize(start - glm::vec2(200.0f, 150.0f)) * 200.0f;
        for (int drag = 0; drag < 3; ++drag) {
            frame.mouse_x = start.x;
            frame.mouse_y = start.y;
            frame.mouse_down[0] = frame.mouse_clicked[0] = true;
            gizmo.renderEllipsoidGizmo(ui, layout);
            ASSERT_TRUE(gui::isScaleGizmoActive());
            frame.mouse_clicked[0] = false;
            frame.mouse_x = end.x;
            frame.mouse_y = end.y;
            gizmo.renderEllipsoidGizmo(ui, layout);
            frame.mouse_down[0] = false;
            frame.mouse_released[0] = true;
            gizmo.renderEllipsoidGizmo(ui, layout);
            frame.mouse_released[0] = false;
        }
        const glm::mat4 world = scene.getWorldTransform(volume);
        EXPECT_NEAR(glm::length(glm::vec3(world[0])), gui::gizmo_ops::MIN_VOLUME_AXIS_SCALE, 1e-6f);
        EXPECT_EQ(scene.getNodeById(volume)->ellipsoid->radii, data.radii);
        gizmo.deactivateAllTools();
        op::undoHistory().clear();
    }

    TEST_F(WindowInputDispatchTest, CropToolGizmoDrawsInEveryIndependentSplitPanel) {
        auto& sm = *viewer_->getSceneManager();
        auto& scene = sm.getScene();
        auto& gizmo = gui_->gizmo();
        auto* const rendering = viewer_->getRenderingManager();
        ASSERT_NE(rendering, nullptr);
        auto& primary = viewer_->getViewport();
        rendering->restoreSplitViewMode(SplitViewMode::IndependentDual, primary);
        ASSERT_TRUE(rendering->isIndependentSplitViewActive());
        for (const auto panel : {SplitViewPanelId::Left, SplitViewPanelId::Right}) {
            auto& camera = rendering->resolvePanelViewport(primary, panel).camera;
            camera.t = {panel == SplitViewPanelId::Left ? 0.0f : 6.0f, 0.0f, 12.0f};
            camera.pivot = {0.0f, 0.0f, 0.0f};
            camera.R = rendering::makeVisualizerLookAtRotation(camera.t, camera.pivot);
        }
        rendering->setFocusedSplitPanel(SplitViewPanelId::Left);

        const auto model = scene.addSplat("Split model", lfs::test::licht::make_splat(3));
        const auto box = scene.addCropBox("Split box", model);
        core::CropBoxData data;
        data.min = glm::vec3(-0.5f);
        data.max = glm::vec3(0.5f);
        scene.setCropBoxData(box, data);
        const auto ellipsoid = scene.addEllipsoid("Split ellipsoid", model);
        sm.changeContentType(SceneManager::ContentType::SplatFiles);

        gui::UIContext ui{.viewer = viewer_.get(), .editor = &viewer_->getEditorContext()};
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        UnifiedToolRegistry::instance().setActiveTool("builtin.cropbox");
        for (const auto& [shape, volume] : {std::pair{"box", box}, std::pair{"ellipsoid", ellipsoid}}) {
            sm.selectNode(volume);
            viewer_->getEditorContext().update(&sm, viewer_->getTrainerManager());
            gizmo.setCropToolShape(shape);
            for (const auto operation :
                 {gui::GizmoOperation::Translate, gui::GizmoOperation::Rotate, gui::GizmoOperation::Scale}) {
                SCOPED_TRACE(std::string(shape) + " " + std::to_string(static_cast<int>(operation)));
                gizmo.setOperation(operation);
                gizmo.updateToolState(ui, false);
                (void)gui::consumeLineRendererCommands();
                gizmo.renderCropBoxGizmo(ui, layout);
                gizmo.renderEllipsoidGizmo(ui, layout);
                // Every split panel draws the gizmo, each clipped to its own rectangle.
                std::set<std::pair<int, int>> clips;
                for (const auto& command : gui::consumeLineRendererCommands()) {
                    if (command.clip_rect)
                        clips.emplace(command.clip_rect->x, command.clip_rect->x + command.clip_rect->width);
                }
                ASSERT_EQ(clips.size(), 2u);
                EXPECT_EQ(clips.begin()->first, 0);
                EXPECT_LE(clips.begin()->second, std::next(clips.begin())->first);
                EXPECT_EQ(std::next(clips.begin())->second, 400);
            }
        }
        gizmo.deactivateAllTools();
        rendering->restoreSplitViewMode(SplitViewMode::Disabled, primary);
        op::undoHistory().clear();
    }

    TEST(OverlayDrawListClip, PrimitivesStayInsideTheirClipRect) {
        (void)gui::consumeLineRendererCommands();
        const auto color = gui::overlayColor(255, 0, 0, 255);
        gui::NativeOverlayDrawList draw_list;
        draw_list.PushClipRect({0.0f, 0.0f}, {200.0f, 300.0f});
        draw_list.AddLine({50.0f, 150.0f}, {350.0f, 150.0f}, color, 3.0f);
        draw_list.AddTriangleFilled({150.0f, 100.0f}, {350.0f, 150.0f}, {150.0f, 200.0f}, color);
        draw_list.AddCircleFilled({195.0f, 60.0f}, 20.0f, color);
        draw_list.AddCircle({195.0f, 240.0f}, 20.0f, color, 16, 2.0f);
        draw_list.PopClipRect();

        VulkanViewportPassParams params;
        params.viewport_pos = {0.0f, 0.0f};
        params.viewport_size = {400.0f, 300.0f};
        gui::detail::appendLineRendererOverlays(params);

        ASSERT_FALSE(params.ui_shape_overlay_triangles.empty());
        for (const auto& vertex : params.ui_shape_overlay_triangles)
            EXPECT_LE(vertex.screen_position.x, 200.0f + 1e-3f);
        ASSERT_FALSE(params.overlay_triangles.empty());
        for (const auto& vertex : params.overlay_triangles)
            EXPECT_LE((vertex.position.x + 1.0f) * 0.5f * params.viewport_size.x, 200.0f + 1e-3f);
    }

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
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
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
        InputController orbit_controller(nullptr, orbit_viewport);
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
        const gui::ViewportLayout layout{.pos = {0.0f, 0.0f}, .size = {400.0f, 300.0f}};
        gizmo.updateToolState(ui, false);
        auto& frame = frameInput();
        const auto render_at = [&](const float x, const float y, const bool down, const bool clicked) {
            frame.mouse_x = x;
            frame.mouse_y = y;
            frame.mouse_down[0] = down;
            frame.mouse_clicked[0] = clicked;
            frame.mouse_released[0] = false;
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
        InputController orbit_controller(nullptr, orbit_viewport);
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

    TEST_F(WindowInputDispatchTest, ProductionMenubarPopupOwnsShortcutUntilEscape) {
        gui::RmlMenuBar bar;
        bar.init(&manager());
        struct ShutdownMenuBar {
            gui::RmlMenuBar& bar;
            ~ShutdownMenuBar() { bar.shutdown(); }
        } shutdown{bar};
        bar.updateLabels({"File"}, {"file"});
        auto* const context = manager().getContext("menu_bar");
        ASSERT_NE(context, nullptr);
        gui::PanelInputState layout_input;
        layout_input.screen_w = 400;
        layout_input.screen_h = 300;
        bar.processInput(layout_input);
        context->Update();
        auto* const items = context->GetDocument(0)->GetElementById("menu-items");
        ASSERT_NE(items, nullptr);
        ASSERT_GT(items->GetNumChildren(), 0);
        auto* const first = items->GetChild(0);
        const auto offset = first->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = first->GetBox().GetSize(Rml::BoxArea::Border);

        click(static_cast<int>(offset.x + size.x * 0.5f), static_cast<int>(offset.y + size.y * 0.5f));
        ASSERT_TRUE(bar.isOpen());
        manager().syncTextInput();
        ASSERT_TRUE(manager().wantsModalKeyboardCapture());

        int shortcuts = 0;
        lfs::event::ScopedHandler handlers;
        handlers.subscribe<core::events::cmd::ToggleSplitView>([&](const auto&) { ++shortcuts; });
        key(SDL_SCANCODE_V);
        EXPECT_EQ(shortcuts, 0);
        key(SDL_SCANCODE_ESCAPE);
        EXPECT_FALSE(bar.isOpen());
        EXPECT_FALSE(manager().wantsModalKeyboardCapture());
        key(SDL_SCANCODE_V);
        EXPECT_EQ(shortcuts, 1);

        ASSERT_TRUE(field_->Focus());
        field_->Blur();
        manager().syncTextInput();
        for (const bool unload_document : {true, false}) {
            SCOPED_TRACE(unload_document ? "document unload" : "window focus loss");
            auto* popup_document = context_->LoadDocumentFromMemory(
                "<rml><head><style>body { width:400px; height:300px; }</style></head>"
                "<body><div id='popup' class='dropdown-popup'/></body></rml>");
            ASSERT_NE(popup_document, nullptr);
            popup_document->Show();
            context_->Update();
            manager().syncTextInput();
            ASSERT_TRUE(manager().wantsModalKeyboardCapture());

            if (unload_document) {
                popup_document->DispatchEvent("unload", {});
                auto* popup = popup_document->GetElementById("popup");
                popup->GetParentNode()->RemoveChild(popup);
            } else {
                auto* popup = popup_document->GetElementById("popup");
                popup->GetParentNode()->RemoveChild(popup);
                SDL_Event focus_lost{};
                focus_lost.type = SDL_EVENT_WINDOW_FOCUS_LOST;
                dispatch(focus_lost);
            }
            manager().syncTextInput();
            EXPECT_FALSE(manager().wantsModalKeyboardCapture());
            const int previous_shortcuts = shortcuts;
            key(SDL_SCANCODE_V);
            EXPECT_EQ(shortcuts, previous_shortcuts + 1);
        }
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

    TEST_F(WindowInputDispatchTest, FloatingSelectDoesNotPressDockedHostBehindIt) {
        gui::RmlPanelHost docked(&manager(), "test-input-docked", "");
        gui::RmlPanelHost floating(&manager(), "test-input-floating", "");
        attachInputHost(docked, 400, 300,
                        "<rml><head><style>body { width:400px; height:300px; }"
                        "div { width:400px; height:300px; }</style></head>"
                        "<body><div id='chart'/></body></rml>");
        attachInputHost(floating, 180, 100,
                        "<rml><head><style>body { width:180px; height:100px; font-family:Inter; font-size:14px; }"
                        "select { position:absolute; left:20px; top:20px; width:140px; height:30px; }"
                        "selectbox { position:absolute; top:30px; width:140px; height:120px; }"
                        "option { display:block; height:40px; }</style></head>"
                        "<body><select id='scale'><option value='100'>100%</option>"
                        "<option value='125'>125%</option><option value='150'>150%</option>"
                        "</select></body></rml>");
        // The host's render surface leaves room for dropdowns outside its panel bounds.
        floating.getContext()->SetDimensions({400, 300});
        floating.getContext()->Update();
        floating.setFloating(true);
        registerInputFloat();
        auto* select = dynamic_cast<Rml::ElementFormControlSelect*>(floating.getDocument()->GetElementById("scale"));
        ASSERT_NE(select, nullptr);
        auto* chart = docked.getDocument()->GetElementById("chart");
        InputEventRecorder recorder;
        chart->AddEventListener("mousedown", &recorder);
        chart->AddEventListener("mouseup", &recorder);
        chart->AddEventListener("click", &recorder);
        const auto frame = [&] {
            manager().beginFrameCursorTracking();
            forwardHost(docked, 0, 0);
            forwardHost(floating, 100, 60);
        };
        frame();
        frame();

        click(140, 95);
        EXPECT_TRUE(select->IsSelectBoxVisible());
        EXPECT_TRUE(gui::rml_input::hasOpenKeyboardPopup(floating.getContext()));
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 0) << recorder.joined();
        EXPECT_EQ(recorder.countOf("chart:mouseup"), 0) << recorder.joined();
        EXPECT_EQ(recorder.countOf("chart:click"), 0) << recorder.joined();

        floating.getContext()->Update();
        floating.getContext()->Render();
        frame();
        frame();
        ASSERT_TRUE(select->IsSelectBoxVisible());
        auto* option = select->GetOption(2);
        const auto position = option->GetAbsoluteOffset(Rml::BoxArea::Border);
        const int option_x = 100 + static_cast<int>(position.x) + 5;
        const int option_y = 60 + static_cast<int>(position.y) + 5;
        ASSERT_GT(option_y, 160); // The dropdown extends beyond the floating panel.
        click(option_x, option_y);
        EXPECT_EQ(select->GetValue(), "150");
        EXPECT_FALSE(select->IsSelectBoxVisible());
        EXPECT_FALSE(gui::rml_input::hasOpenKeyboardPopup(floating.getContext()));
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 0) << recorder.joined();
        EXPECT_EQ(recorder.countOf("chart:mouseup"), 0) << recorder.joined();
        frame();
        frame();

        // Blank panel space also occludes, while a point outside still reaches the chart.
        click(260, 145);
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 0);
        click(330, 240);
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 1);
        EXPECT_EQ(recorder.countOf("chart:mouseup"), 1);
        EXPECT_EQ(recorder.countOf("chart:click"), 1);

        gui::PanelRegistry::instance().set_panel_enabled("test.input_float", false);
        manager().deactivateInput(floating.getContext());
        click(140, 95);
        EXPECT_EQ(recorder.countOf("chart:click"), 2);
        chart->RemoveEventListener("mousedown", &recorder);
        chart->RemoveEventListener("mouseup", &recorder);
        chart->RemoveEventListener("click", &recorder);
    }

    TEST_F(WindowInputDispatchTest, FloatingCloseButtonDisablesItsPanel) {
        gui::RmlPanelHost host(&manager(), "test.input_float", "");
        attachInputHost(host, 180, 100,
                        "<rml><head><style>body { width:180px; height:100px; }</style></head>"
                        "<body id='window-frame'><div id='title-bar'><span id='close-btn'>x</span></div>"
                        "<div id='body'>log</div></body></rml>");
        host.setFloating(true);
        host.getContext()->Update();
        registerInputFloat();
        forwardHost(host, 100, 60);
        ASSERT_TRUE(gui::PanelRegistry::instance().is_panel_enabled("test.input_float"));

        click(120, 140);
        EXPECT_TRUE(gui::PanelRegistry::instance().is_panel_enabled("test.input_float"));

        auto* button = host.getDocument()->GetElementById("close-btn");
        ASSERT_NE(button, nullptr);
        const auto origin = button->GetAbsoluteOffset(Rml::BoxArea::Border);
        const auto size = button->GetBox().GetSize(Rml::BoxArea::Border);
        ASSERT_GT(size.x, 0.f);
        ASSERT_GT(size.y, 0.f);
        click(100 + static_cast<int>(origin.x + size.x * 0.5f),
              60 + static_cast<int>(origin.y + size.y * 0.5f));
        EXPECT_FALSE(gui::PanelRegistry::instance().is_panel_enabled("test.input_float"));
    }

    TEST_F(WindowInputDispatchTest, FloatingOcclusionPreservesDockedGestureOwnership) {
        gui::RmlPanelHost docked(&manager(), "test-input-drag", "");
        attachInputHost(docked, 400, 300,
                        "<rml><head><style>body { width:400px; height:300px; }"
                        "div { width:400px; height:300px; }</style></head>"
                        "<body><div id='chart'/></body></rml>");
        registerInputFloat();
        forwardHost(docked, 0, 0);
        auto* chart = docked.getDocument()->GetElementById("chart");
        InputEventRecorder recorder;
        chart->AddEventListener("mousedown", &recorder);
        chart->AddEventListener("mouseup", &recorder);
        const auto button = [&](bool down, float x, float y) {
            SDL_Event event{};
            event.type = down ? SDL_EVENT_MOUSE_BUTTON_DOWN : SDL_EVENT_MOUSE_BUTTON_UP;
            event.button.button = SDL_BUTTON_LEFT;
            event.button.x = x;
            event.button.y = y;
            dispatch(event);
        };
        button(true, 140, 95);
        button(false, 330, 240);
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 0);
        EXPECT_EQ(recorder.countOf("chart:mouseup"), 0);
        button(true, 330, 240);
        button(false, 140, 95);
        EXPECT_EQ(recorder.countOf("chart:mousedown"), 1);
        EXPECT_EQ(recorder.countOf("chart:mouseup"), 1);

        SDL_Event motion{};
        motion.type = SDL_EVENT_MOUSE_MOTION;
        motion.motion.y = 240.f;
        const auto start = std::chrono::steady_clock::now();
        constexpr int updates = 10000;
        for (int i = 0; i < updates; ++i) {
            motion.motion.x = 320.f + static_cast<float>(i % 10);
            dispatch(motion);
        }
        const auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start);
        RecordProperty("pointer_update_us", std::to_string(elapsed.count() / updates));
        EXPECT_EQ(docked.getContext()->GetHoverElement(), chart);
        chart->RemoveEventListener("mousedown", &recorder);
        chart->RemoveEventListener("mouseup", &recorder);
    }

    TEST_F(WindowInputDispatchTest, EnterThatOpensATextInputLeavesItFocused) {
        // ScrubFieldController opens its text input from the field's Enter key-down.
        struct OpenEditor final : Rml::EventListener {
            Rml::Element* editor = nullptr;
            void ProcessEvent(Rml::Event& event) override {
                if (event.GetParameter<int>("key_identifier", 0) == Rml::Input::KI_RETURN)
                    editor->Focus();
            }
        } open_editor;
        gui::RmlPanelHost host(&manager(), "test-input-enter", "");
        attachInputHost(host, 200, 100,
                        "<rml><head><style>body { width:200px; height:100px; }</style></head>"
                        "<body><div id='field' class='scrub-field'>1</div>"
                        "<input id='editor' type='text'/></body></rml>");
        auto* const field = host.getDocument()->GetElementById("field");
        auto* const editor = host.getDocument()->GetElementById("editor");
        open_editor.editor = editor;
        field->AddEventListener(Rml::EventId::Keydown, &open_editor);
        setKeyboardContext(host.getContext(), false);
        const auto press_enter = [&] {
            forwardHost(host, 0, 0);
            key(SDL_SCANCODE_RETURN);
        };
        ASSERT_TRUE(field->Focus());
        press_enter();
        EXPECT_EQ(host.getContext()->GetFocusElement(), editor);
        // Enter in an input that already had focus still ends the edit.
        press_enter();
        EXPECT_NE(host.getContext()->GetFocusElement(), editor);
        field->RemoveEventListener(Rml::EventId::Keydown, &open_editor);
    }

    TEST_F(WindowInputDispatchTest, CapturedReleaseOutsidePanelEndsScrubDrag) {
        // Body listeners match ScrubFieldController: press the field, move and release on the body.
        struct Scrub final : Rml::EventListener {
            bool active = false;
            float value = 1.f, x = 0.f;
            int ups = 0, clicks = 0;
            void ProcessEvent(Rml::Event& event) override {
                const float mx = event.GetParameter("mouse_x", 0.f);
                if (event.GetId() == Rml::EventId::Mousedown) {
                    active = true;
                    x = mx;
                } else if (event.GetId() == Rml::EventId::Mousemove && active) {
                    value += (mx - x) * 0.01f;
                    x = mx;
                } else if (event.GetId() == Rml::EventId::Mouseup && active) {
                    active = false;
                    ++ups;
                } else if (event.GetId() == Rml::EventId::Click) {
                    ++clicks;
                }
            }
        } scrub;
        gui::RmlPanelHost host(&manager(), "test-scrub-release", "");
        attachInputHost(host, 200, 160,
                        "<rml><head><style>body{width:200px;height:160px;}"
                        "#field{position:absolute;left:20px;top:40px;width:100px;height:28px;}"
                        "</style></head><body id='body'><div id='field'/></body></rml>");
        forwardHost(host, 0, 0);
        auto* field = host.getDocument()->GetElementById("field");
        auto* body = host.getDocument()->GetElementById("body");
        ASSERT_NE(field, nullptr);
        ASSERT_NE(body, nullptr);
        field->AddEventListener(Rml::EventId::Mousedown, &scrub);
        field->AddEventListener(Rml::EventId::Click, &scrub);
        body->AddEventListener(Rml::EventId::Mousemove, &scrub);
        body->AddEventListener(Rml::EventId::Mouseup, &scrub);
        const auto at = [&](uint32_t type, float x, float y) {
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
        };
        at(SDL_EVENT_MOUSE_BUTTON_DOWN, 70, 54);
        at(SDL_EVENT_MOUSE_MOTION, 150, 54);
        const float dragged = scrub.value;
        ASSERT_GT(dragged, 1.f);
        ASSERT_TRUE(scrub.active);
        at(SDL_EVENT_MOUSE_MOTION, 320, 200);
        ASSERT_FLOAT_EQ(scrub.value, dragged);
        at(SDL_EVENT_MOUSE_BUTTON_UP, 320, 200);
        at(SDL_EVENT_MOUSE_MOTION, 70, 54);
        EXPECT_FALSE(scrub.active);
        EXPECT_EQ(scrub.ups, 1);
        EXPECT_FLOAT_EQ(scrub.value, dragged);
        EXPECT_EQ(scrub.clicks, 0);
        at(SDL_EVENT_MOUSE_BUTTON_DOWN, 70, 54);
        at(SDL_EVENT_MOUSE_BUTTON_UP, 70, 54);
        EXPECT_EQ(scrub.clicks, 1);
        EXPECT_FALSE(scrub.active);
        field->RemoveEventListener(Rml::EventId::Mousedown, &scrub);
        field->RemoveEventListener(Rml::EventId::Click, &scrub);
        body->RemoveEventListener(Rml::EventId::Mousemove, &scrub);
        body->RemoveEventListener(Rml::EventId::Mouseup, &scrub);
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
        auto& overlay = viewportOverlay();
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

    TEST_F(WindowInputDispatchTest, ViewportClickDismissesOpenDropdown) {
        gui::RmlPanelHost panel(&manager(), "test-dropdown-panel", "");
        attachInputHost(panel, 200, 300,
                        "<rml><head><style>body { width:200px; height:300px; font-family:Inter; font-size:14px; }"
                        "select { position:absolute; left:20px; top:20px; width:140px; height:30px; }"
                        "</style></head><body><select id='tone'><option value='none'>None</option>"
                        "<option value='aces'>ACES</option></select></body></rml>");
        auto* panel_select = dynamic_cast<Rml::ElementFormControlSelect*>(
            panel.getDocument()->GetElementById("tone"));
        ASSERT_NE(panel_select, nullptr);

        auto& overlay = viewportOverlay();
        overlay.init(&manager());
        overlay.setViewportBounds({0, 0}, {400, 300}, {0, 0});
        auto* overlay_context = Rml::GetContext("viewport_overlay");
        ASSERT_NE(overlay_context, nullptr);
        auto* overlay_document = overlay_context->LoadDocumentFromMemory(
            "<rml><head><style>body { width:400px; height:300px; font-family:Inter; font-size:14px; }"
            "select { position:absolute; left:10px; top:10px; width:80px; height:24px; }"
            "</style></head><body><select id='depth'><option value='palette'>Color</option>"
            "<option value='gray'>Gray</option></select></body></rml>");
        ASSERT_NE(overlay_document, nullptr);
        overlay_document->Show();
        overlay_context->Update();
        struct DocumentGuard {
            Rml::Context* context = nullptr;
            Rml::ElementDocument* document = nullptr;
            ~DocumentGuard() {
                if (context && document)
                    context->UnloadDocument(document);
            }
        } guard{overlay_context, overlay_document};
        auto* overlay_select = dynamic_cast<Rml::ElementFormControlSelect*>(
            overlay_document->GetElementById("depth"));
        ASSERT_NE(overlay_select, nullptr);

        const auto pump = [&] {
            manager().beginFrameCursorTracking();
            forwardHost(panel, 200, 0);
            overlay.processInput({});
        };

        pump();
        click(230, 35);
        EXPECT_TRUE(panel_select->IsSelectBoxVisible());
        panel.getContext()->Update();
        panel.getContext()->Render();
        click(230, 35);
        EXPECT_FALSE(panel_select->IsSelectBoxVisible());

        ASSERT_TRUE(panel.getDocument()->Focus());
        panel_select->ShowSelectBox();
        panel.getContext()->Update();
        panel.getContext()->Render();
        ASSERT_TRUE(panel_select->IsSelectBoxVisible());
        EXPECT_NE(panel.getContext()->GetFocusElement(), panel_select);
        click(60, 180);
        EXPECT_FALSE(panel_select->IsSelectBoxVisible());

        pump();
        click(30, 18);
        EXPECT_TRUE(overlay_select->IsSelectBoxVisible());
        overlay_context->Update();
        overlay_context->Render();
        click(30, 18);
        EXPECT_FALSE(overlay_select->IsSelectBoxVisible());

        ASSERT_TRUE(overlay_document->Focus());
        overlay_select->ShowSelectBox();
        overlay_context->Update();
        overlay_context->Render();
        ASSERT_TRUE(overlay_select->IsSelectBoxVisible());
        EXPECT_NE(overlay_context->GetFocusElement(), overlay_select);
        click(60, 180);
        EXPECT_FALSE(overlay_select->IsSelectBoxVisible());
    }

    // Fails if a tab picked while the strip overflows stays scrolled out of view.
    TEST_F(WindowInputDispatchTest, ActiveTabScrollsIntoTheOverflowingStrip) {
        gui::RmlRightPanel right_panel;
        right_panel.init(&manager());
        struct Shutdown {
            gui::RmlRightPanel& panel;
            ~Shutdown() { panel.shutdown(); }
        } shutdown{right_panel};
        gui::RightPanelLayout layout;
        // The four tabs overflow the strip, but each one fits inside it.
        layout.size = {150.0f, 160.0f};
        std::vector<gui::TabSnapshot> tabs;
        for (const char* id : {"first", "second", "third", "fourth"})
            tabs.push_back({.id = id, .label = id, .dom_id = std::string("rp-tab-") + id});
        auto* const document = Rml::GetContext("right_panel")->GetDocument(0);
        auto* const strip = document->GetElementById("tab-strip-viewport");
        ASSERT_NE(strip, nullptr);
        for (const char* active : {"first", "fourth", "second"}) {
            SCOPED_TRACE(active);
            right_panel.render(layout, tabs, active, 0.0f, 0.0f, 400, 300);
            auto* const tab = document->GetElementById(std::string("rp-tab-") + active);
            ASSERT_NE(tab, nullptr);
            EXPECT_GT(strip->GetScrollHeight(), strip->GetClientHeight());
            ASSERT_LE(tab->GetOffsetHeight(), strip->GetClientHeight());
            const float top = tab->GetAbsoluteTop() - strip->GetAbsoluteTop();
            EXPECT_GE(top, -0.5f);
            EXPECT_LE(top + tab->GetOffsetHeight(), strip->GetClientHeight() + 0.5f);
        }
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
        Rml::ObserverPtr<Rml::Element> handle_;
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

        void checkMutation(const bool cached, const bool add_group, const bool from_menu = false,
                           const bool point_cloud = false) {
            auto& scene_manager = *viewer_->getSceneManager();
            auto& scene = scene_manager.getScene();
            const auto original = point_cloud
                                      ? scene.addPointCloud("Original", std::make_shared<core::PointCloud>(
                                                                            core::Tensor::zeros({2, 3}, core::Device::CPU),
                                                                            core::Tensor::ones({2, 3}, core::Device::CPU)))
                                      : scene.addGroup("Original");
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

    TEST_F(ScenePanelRefreshTest, CameraMenusOnlyOfferActionsWithTargets) {
        auto& scene_manager = *viewer_->getSceneManager();
        auto& scene = scene_manager.getScene();
        const auto empty = scene.addCameraGroup("Empty", core::NULL_NODE, 99);
        const auto empty_nested = scene.addCameraGroup("Empty nested", core::NULL_NODE, 0);
        scene.addGroup("Not a camera", empty_nested);
        const auto full = scene.addCameraGroup("Full", core::NULL_NODE, 0);
        const auto add_camera = [&](const std::string& name, const core::NodeId parent,
                                    const std::filesystem::path& image_path, const int uid) {
            return scene.addCamera(name, parent, std::make_shared<core::Camera>(core::Tensor::eye(3, core::Device::CPU), core::Tensor::zeros({3}, core::Device::CPU), 100.f, 110.f, 32.f, 24.f, core::Tensor{}, core::Tensor{}, core::CameraModelType::PINHOLE, name, image_path, std::filesystem::path{}, 64, 48, uid));
        };
        const auto no_image = add_camera("No image", core::NULL_NODE, {}, 41);
        const auto with_image = add_camera("With image", core::NULL_NODE, "image.png", 42);
        const auto child = add_camera("Child", full, {}, 43);
        gui::NativeScenePanel panel(&manager());
        gui::PanelDrawContext ctx;
        ctx.scene = &scene;
        const auto open_menu = [&](const core::NodeId id) {
            menu().request({}, 0, 0);
            ctx.scene_generation = python::get_scene_generation();
            ++ctx.frame_serial;
            panel.preload(ctx);
            panel.renderDirect({.mode = gui::PanelDirectRenderMode::Preload,
                                .width = 400,
                                .height = 600},
                               ctx);
            auto* context = manager().getContext("scene_panel_native");
            if (!context) {
                ADD_FAILURE() << "Missing Scene panel";
                return std::string{};
            }
            auto* document = context->GetDocument(0);
            document->Show();
            auto* tree = dynamic_cast<gui::SceneGraphElement*>(document->GetElementById("tree-container"));
            if (!tree) {
                ADD_FAILURE() << "Missing scene tree";
                return std::string{};
            }
            tree->SetProperty("height", "500px");
            context->SetDimensions({400, 600});
            context->Update();
            static_cast<void>(tree->syncFromScene(ctx));
            context->Update();
            auto* row = document->QuerySelector("[data-node-id='" + std::to_string(id) + "']");
            if (!row) {
                ADD_FAILURE() << "Missing Scene row " << id << " height=" << tree->GetClientHeight();
                return std::string{};
            }
            row->DispatchEvent("mousedown", {{"button", Rml::Variant(1)}, {"mouse_x", Rml::Variant(10.f)}, {"mouse_y", Rml::Variant(10.f)}});
            auto* menu_context = manager().getContext("global_context_menu");
            if (!menu_context || !menu_context->GetDocument(0)) {
                ADD_FAILURE() << "Missing context menu";
                return std::string{};
            }
            return menu_context->GetDocument(0)->GetInnerRML();
        };
        auto html = open_menu(no_image);
        EXPECT_EQ(html.find("scene_panel:go_to_image:"), std::string::npos);
        EXPECT_NE(html.find("scene_panel:go_to_camera:"), std::string::npos);
        html = open_menu(with_image);
        EXPECT_NE(html.find("scene_panel:go_to_image:42"), std::string::npos);
        EXPECT_NE(html.find("scene_panel:go_to_camera:42"), std::string::npos);
        for (const auto id : {empty, empty_nested}) {
            html = open_menu(id);
            EXPECT_EQ(html.find("scene_panel:enable_all_train:"), std::string::npos);
            EXPECT_EQ(html.find("scene_panel:disable_all_train:"), std::string::npos);
            EXPECT_EQ(html.find("scene_panel:duplicate:"), std::string::npos);
        }
        html = open_menu(full);
        EXPECT_NE(html.find("scene_panel:enable_all_train:"), std::string::npos);
        EXPECT_NE(html.find("scene_panel:disable_all_train:"), std::string::npos);
        EXPECT_EQ(html.find("scene_panel:duplicate:"), std::string::npos);
        auto* menu_context = manager().getContext("global_context_menu");
        ASSERT_NE(menu_context, nullptr);
        auto* doc = menu_context->GetDocument(0);
        ASSERT_NE(doc, nullptr);
        auto* disable = doc->QuerySelector("[data-ctx-action='scene_panel:disable_all_train:" + std::to_string(full) + "']");
        ASSERT_NE(disable, nullptr);
        disable->DispatchEvent("click", {});
        panel.preload(ctx);
        EXPECT_FALSE(scene.getNodeById(child)->training_enabled);
        open_menu(full);
        auto* enable = doc->QuerySelector("[data-ctx-action='scene_panel:enable_all_train:" + std::to_string(full) + "']");
        ASSERT_NE(enable, nullptr);
        enable->DispatchEvent("click", {});
        panel.preload(ctx);
        EXPECT_TRUE(scene.getNodeById(child)->training_enabled);

        scene_manager.selectNodesById({empty, empty_nested});
        html = open_menu(empty);
        EXPECT_EQ(html.find("scene_panel:enable_all_selected_train"), std::string::npos);
        EXPECT_EQ(html.find("scene_panel:disable_all_selected_train"), std::string::npos);
        scene_manager.selectNodesById({empty, with_image});
        html = open_menu(empty);
        EXPECT_NE(html.find("scene_panel:enable_all_selected_train"), std::string::npos);
        EXPECT_NE(html.find("scene_panel:disable_all_selected_train"), std::string::npos);
    }

    TEST_F(ScenePanelRefreshTest, ContextMenuDuplicateRefreshesLiveTree) {
        checkMutation(false, false, true);
    }
    TEST_F(ScenePanelRefreshTest, ContextMenuDuplicateRefreshesCachedTree) {
        checkMutation(true, false, true);
    }
    TEST_F(ScenePanelRefreshTest, ContextMenuPointCloudDuplicateRefreshesLiveTreeThenStaysIdle) {
        checkMutation(false, false, true, true);
    }
    TEST_F(ScenePanelRefreshTest, ContextMenuPointCloudDuplicateRefreshesCachedTreeThenStaysIdle) {
        checkMutation(true, false, true, true);
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
