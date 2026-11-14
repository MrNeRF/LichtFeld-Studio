/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Elements/ElementFormControlInput.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/RenderInterface.h>

#include "gui/rmlui/rml_input_utils.hpp"

#include <filesystem>
#include <gtest/gtest.h>

namespace {

    class StubRenderInterface final : public Rml::RenderInterface {
    public:
        Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>,
                                                    Rml::Span<const int>) override {
            return 1;
        }

        void RenderGeometry(Rml::CompiledGeometryHandle,
                            Rml::Vector2f,
                            Rml::TextureHandle) override {}

        void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}

        Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions,
                                       const Rml::String&) override {
            dimensions = {16, 16};
            return 1;
        }

        Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>,
                                           Rml::Vector2i) override {
            return 1;
        }

        void ReleaseTexture(Rml::TextureHandle) override {}
        void EnableScissorRegion(bool) override {}
        void SetScissorRegion(Rml::Rectanglei) override {}
    };

    class ChangeListener final : public Rml::EventListener {
    public:
        void ProcessEvent(Rml::Event& event) override {
            ++count;
            value = event.GetParameter<Rml::String>("value", "");
        }

        int count = 0;
        Rml::String value;
    };

    constexpr const char* kDocumentRml = R"RML(
<rml>
    <head>
        <title>native-text-input-keys</title>
        <style>
            input { font-family: Inter; font-size: 12px; }
        </style>
    </head>
    <body>
        <input id="text-input" type="text" value="hello"/>
    </body>
</rml>
)RML";

    class RmlTextInputNativeKeysTest : public ::testing::Test {
    protected:
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            const auto font_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                   "src/visualizer/gui/assets/fonts/Inter-Regular.ttf";
            ASSERT_TRUE(Rml::LoadFontFace(font_path.string()));
        }

        static void TearDownTestSuite() {
            Rml::Shutdown();
        }

        void SetUp() override {
            context_ = Rml::CreateContext("native_text_input_keys", {320, 80},
                                          &render_interface_, &text_input_handler_);
            ASSERT_TRUE(context_);
            document_ = context_->LoadDocumentFromMemory(kDocumentRml);
            ASSERT_TRUE(document_);
            document_->Show();
            context_->Update();
            input_ = rmlui_dynamic_cast<Rml::ElementFormControlInput*>(
                document_->GetElementById("text-input"));
            ASSERT_NE(input_, nullptr);
            input_->Focus();
            context_->Update();
        }

        void TearDown() override {
            ASSERT_TRUE(Rml::RemoveContext("native_text_input_keys"));
            context_ = nullptr;
            document_ = nullptr;
            input_ = nullptr;
        }

        void loadLockableControls() {
            document_->Close();
            const auto resource_path = std::filesystem::path(PROJECT_ROOT_PATH) /
                                       "src/visualizer/gui/rmlui/resources/test.rml";
            // RmlUi parses the source as a URL, so a Windows drive colon takes its '|' form.
            const auto source_url = Rml::StringUtilities::Replace(resource_path.generic_string(), ':', '|');
            document_ = context_->LoadDocumentFromMemory(R"RML(
<rml><head><link type="text/rcss" href="components.rcss"/></head><body>
    <input id="search" type="text"/>
    <div id="locked" class="disabled-overlay">
        <select id="strategy"><option>First</option><option>Second</option></select>
        <button id="lock">Lock</button>
        <div><input id="iterations" type="text" value="30000"/></div>
        <input id="toggle" type="checkbox"/>
        <button id="step">+</button>
    </div>
    <input id="after" type="text"/>
</body></rml>)RML",
                                                         source_url);
            ASSERT_NE(document_, nullptr);
            context_->SetDimensions({640, 480});
            document_->Show();
            context_->Update();
            input_ = rmlui_dynamic_cast<Rml::ElementFormControlInput*>(
                document_->GetElementById("iterations"));
            ASSERT_NE(input_, nullptr);
        }

        inline static StubRenderInterface render_interface_;
        lfs::vis::gui::RmlTextInputHandler text_input_handler_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        Rml::ElementFormControlInput* input_ = nullptr;
    };

    TEST_F(RmlTextInputNativeKeysTest, DeletionUpdatesValueAndDispatchesChange) {
        ChangeListener listener;
        input_->AddEventListener(Rml::EventId::Change, &listener);

        input_->SetSelectionRange(5, 5);
        context_->ProcessKeyDown(Rml::Input::KI_BACK, 0);
        EXPECT_EQ(input_->GetValue(), "hell");
        EXPECT_EQ(listener.count, 1);
        EXPECT_EQ(listener.value, "hell");

        input_->SetValue("hello");
        input_->SetSelectionRange(0, 0);
        context_->ProcessKeyDown(Rml::Input::KI_DELETE, 0);
        EXPECT_EQ(input_->GetValue(), "ello");
        EXPECT_EQ(listener.count, 2);
        EXPECT_EQ(listener.value, "ello");

        input_->RemoveEventListener(Rml::EventId::Change, &listener);
    }

    TEST_F(RmlTextInputNativeKeysTest, EditingShortcutsUsePrintedLetters) {
        using namespace lfs::vis;
        struct Shortcut {
            SDL_Scancode scancode;
            SDL_Keycode keycode;
            const char* result;
        };
        // A and Q exchange physical positions on AZERTY.
        for (const Shortcut shortcut : {
                 Shortcut{SDL_SCANCODE_Q, SDLK_A, ""},
                 Shortcut{SDL_SCANCODE_A, SDLK_A, ""},
                 Shortcut{SDL_SCANCODE_A, SDLK_Q, "ab"},
             }) {
            input_->SetValue("abc");
            input_->SetSelectionRange(3, 3);
            FrameInputBuffer buffer;
            SDL_Event key{};
            key.type = SDL_EVENT_KEY_DOWN;
            key.key.scancode = shortcut.scancode;
            key.key.key = shortcut.keycode;
            key.key.mod = SDL_KMOD_CTRL;
            buffer.processEvent(key);
            ASSERT_EQ(buffer.input_events.size(), 1u);
            gui::rml_input::processKeyboardEvent(*context_, buffer.input_events.front(), &text_input_handler_);
            context_->ProcessKeyDown(Rml::Input::KI_BACK, 0);
            EXPECT_EQ(input_->GetValue(), shortcut.result) << static_cast<int>(shortcut.scancode);
        }
    }

    TEST_F(RmlTextInputNativeKeysTest, SdlTextInputPreservesUtf8) {
        using namespace lfs::vis;
        for (const char* text : {"é", "é", "漢字"}) {
            input_->SetValue("");
            input_->SetSelectionRange(0, 0);
            SDL_Event event{};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = text;
            FrameInputBuffer buffer;
            buffer.processEvent(event);
            ASSERT_EQ(buffer.input_events.size(), 1u);
            gui::rml_input::processKeyboardEvent(*context_, buffer.input_events.front(),
                                                 &text_input_handler_);
            EXPECT_EQ(input_->GetValue(), text);
        }
    }

    TEST_F(RmlTextInputNativeKeysTest, ComposedTextPreservesUtf8) {
        using namespace lfs::vis;
        input_->SetValue("");
        input_->SetSelectionRange(0, 0);
        SDL_Event editing{};
        editing.type = SDL_EVENT_TEXT_EDITING;
        editing.edit.text = "漢";
        editing.edit.start = 1;
        FrameInputBuffer buffer;
        buffer.processEvent(editing);
        ASSERT_EQ(buffer.input_events.size(), 1u);
        gui::rml_input::processKeyboardEvent(*context_, buffer.input_events.front(),
                                             &text_input_handler_);
        SDL_Event commit{};
        commit.type = SDL_EVENT_TEXT_INPUT;
        commit.text.text = "漢字";
        FrameInputBuffer committed;
        committed.processEvent(commit);
        ASSERT_EQ(committed.input_events.size(), 1u);
        gui::rml_input::processKeyboardEvent(*context_, committed.input_events.front(),
                                             &text_input_handler_);
        EXPECT_EQ(input_->GetValue(), "漢字");
    }

    TEST_F(RmlTextInputNativeKeysTest, DisabledOverlaySkipsAllControlsInBothTabDirections) {
        loadLockableControls();
        auto* search = document_->GetElementById("search");
        auto* after = document_->GetElementById("after");
        ASSERT_TRUE(search->Focus());
        context_->ProcessKeyDown(Rml::Input::KI_TAB, 0);
        EXPECT_EQ(context_->GetFocusElement(), after);
        context_->ProcessKeyDown(Rml::Input::KI_TAB, Rml::Input::KM_SHIFT);
        EXPECT_EQ(context_->GetFocusElement(), search);
        for (const char* id : {"strategy", "lock", "iterations", "toggle", "step"}) {
            EXPECT_FALSE(document_->GetElementById(id)->Focus()) << id;
        }
        EXPECT_EQ(input_->GetValue(), "30000");
    }

    TEST_F(RmlTextInputNativeKeysTest, OpenPopupRequiresDisplayedVisibleAncestor) {
        auto container = document_->CreateElement("div");
        container->SetId("popup-container");
        container->SetProperty("display", "none");
        auto popup = document_->CreateElement("div");
        popup->SetAttribute("class", "dropdown-popup");
        container->AppendChild(std::move(popup));
        document_->AppendChild(std::move(container));
        context_->Update();

        EXPECT_FALSE(lfs::vis::gui::rml_input::hasOpenKeyboardPopup(context_));
        document_->GetElementById("popup-container")->SetProperty("display", "block");
        context_->Update();
        EXPECT_TRUE(lfs::vis::gui::rml_input::hasOpenKeyboardPopup(context_));

        document_->GetElementById("popup-container")->SetProperty("visibility", "hidden");
        context_->Update();
        EXPECT_FALSE(lfs::vis::gui::rml_input::hasOpenKeyboardPopup(context_));
        document_->GetElementById("popup-container")->SetProperty("visibility", "visible");

        document_->GetElementById("popup-container")->SetProperty("display", "none");
        context_->Update();
        EXPECT_FALSE(lfs::vis::gui::rml_input::hasOpenKeyboardPopup(context_));
    }

    TEST_F(RmlTextInputNativeKeysTest, DisabledOverlayBlocksMouseAndUnlockRestoresKeyboardEditing) {
        loadLockableControls();
        auto* search = document_->GetElementById("search");
        ASSERT_TRUE(search->Focus());
        const auto position = input_->GetAbsoluteOffset(Rml::BoxArea::Content);
        context_->ProcessMouseMove(static_cast<int>(position.x + 2), static_cast<int>(position.y + 2), 0);
        context_->ProcessMouseButtonDown(0, 0);
        context_->ProcessMouseButtonUp(0, 0);
        EXPECT_NE(context_->GetFocusElement(), input_);
        document_->GetElementById("locked")->SetClass("disabled-overlay", false);
        context_->Update();
        ASSERT_TRUE(search->Focus());
        context_->ProcessKeyDown(Rml::Input::KI_TAB, 0);
        EXPECT_EQ(context_->GetFocusElement(), document_->GetElementById("strategy"));
        ASSERT_TRUE(input_->Focus());
        input_->SetSelectionRange(0, 5);
        context_->ProcessTextInput("5000");
        EXPECT_EQ(input_->GetValue(), "5000");
    }

    TEST_F(RmlTextInputNativeKeysTest, LockingFocusedControlRejectsQueuedKeyboardInput) {
        using namespace lfs::vis;
        loadLockableControls();
        auto* locked = document_->GetElementById("locked");
        for (const auto kind : {FrameInputEventKind::Text, FrameInputEventKind::TextEditing,
                                FrameInputEventKind::KeyDown, FrameInputEventKind::KeyUp}) {
            locked->SetClass("disabled-overlay", false);
            context_->Update();
            ASSERT_TRUE(input_->Focus());
            input_->SetValue("30000");
            input_->SetSelectionRange(0, 5);
            locked->SetClass("disabled-overlay", true);
            context_->Update();
            InputEventDispatch dispatch;
            FrameInputEvent event{.kind = kind, .scancode = SDL_SCANCODE_BACKSPACE, .text = "5000", .dispatch = &dispatch};
            EXPECT_TRUE(gui::rml_input::processKeyboardEvent(*context_, event));
            EXPECT_TRUE(dispatch.consumed);
            EXPECT_EQ(input_->GetValue(), "30000");
            EXPECT_NE(context_->GetFocusElement(), input_);
        }
    }

} // namespace
