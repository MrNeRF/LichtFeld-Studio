/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "user_shader_test_access.hpp"

#include "core/scene.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/panel_input_utils.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/terminal/terminal_input.hpp"
#include "gui/terminal/terminal_widget.hpp"
#include "python/runner.hpp"
#include "user_shader_linker.hpp"
#include <RmlUi/Core.h>
#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <gtest/gtest.h>
#include <memory>
#include <zep/buffer.h>
#include <zep/display.h>
#include <zep/editor.h>
#include <zep/mode_standard.h>
#include <zep/window.h>

namespace {
    using namespace Zep;

    class PythonEditorInputTest : public ::testing::Test {
    protected:
        ZepEditor editor{new ZepDisplayNull(), {}, ZepEditorFlags::DisableThreads};
        ZepBuffer* buffer = nullptr;
        ZepMode* mode = nullptr;

        void SetUp() override {
            editor.SetGlobalMode(ZepMode_Standard::StaticName());
            buffer = editor.InitWithText("script.py", "alpha beta\nsecond line\nlast");
            mode = buffer->GetMode();
        }
        void select(long anchor, long cursor) {
            mode->SetSelection(GlyphIterator(buffer, anchor), GlyphIterator(buffer, cursor));
        }
        void key(uint32_t value, uint32_t mods = 0) { mode->AddKeyPress(value, mods); }
        std::string text() { return buffer->GetBufferText(buffer->Begin(), buffer->End()); }
        long cursor() { return editor.GetActiveWindow()->GetBufferCursor().Index(); }
    };

    namespace shader = lfs::vis::shaders;
    using lfs::vis::editor::DocumentKind;
    using lfs::vis::editor::PythonEditor;

    class ShaderEditorTest : public ::testing::Test {
    protected:
        lfs::core::Scene scene;
        lfs::test::Schedule schedule;
        shader::ShaderService service{scene, [this] { return std::make_unique<lfs::test::ScheduledLinker>(schedule, true); }, [] {}};
        PythonEditor editor;
        void TearDown() override { schedule.gate(false); }
        std::shared_ptr<const shader::DisplayRecord> display;
        shader::SnapshotPtr publication() {
            auto snapshot = service.candidate();
            return snapshot ? snapshot : service.restrictFallback(display ? display->snapshot : nullptr);
        }
        void present() {
            const auto snapshot = publication();
            display = std::make_shared<const shader::DisplayRecord>(shader::DisplayRecord{snapshot, bool(snapshot)});
            service.observeDisplay(display);
        }
        bool settle(const lfs::core::Uuid& asset, std::uint64_t revision) {
            service.demandLive(std::array{std::string("projection_forward_custom")});
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                present();
                const auto status = service.status(asset);
                if (status && status->revision == revision && !status->compiling && status->pending.empty() &&
                    (status->state == shader::ShaderState::Applied || status->state == shader::ShaderState::Failed))
                    return true;
                std::this_thread::yield();
            }
            return false;
        }
    };

    TEST_F(ShaderEditorTest, EditorChangesAndClosureWakeObserversWithoutShaderWork) {
        editor.setShaderChanges(service.changes());
        const auto doc = editor.newDocument("draft", DocumentKind::Shader);
        auto& console = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        console.setShaderChanges(service.changes());
        struct ResetSignal {
            lfs::vis::gui::panels::PythonConsoleState& console;
            ~ResetSignal() { console.setShaderChanges({}); }
        } reset{console};
        const std::pair<const char*, std::function<void()>> rows[] = {
            {"source supersession", [&] { EXPECT_TRUE(editor.setDocumentText(doc, "changed")); }},
            {"kind supersession", [&] { EXPECT_TRUE(editor.setDocumentMode(doc, DocumentKind::Python)); }},
            {"document closed", [&] { editor.restoreWorkspaceSessionState({}); EXPECT_FALSE(editor.inspectDocument(doc)); }},
            {"restored document", [&] { editor.setText("after restore"); }},
            {"output only", [&] { console.addInfo("observer output"); }}};
        for (const auto& [name, change] : rows) {
            SCOPED_TRACE(name);
            const auto before = service.generation();
            std::promise<void> entering;
            auto entered = entering.get_future();
            auto waiting = std::async(std::launch::async, [&] {
                entering.set_value();
                return service.changes()->wait(before, std::chrono::steady_clock::now() + std::chrono::seconds(5));
            });
            entered.wait();
            change(); // Also covers notify between observation and entering the CV wait.
            EXPECT_TRUE(waiting.get());
            EXPECT_GT(service.generation(), before);
            EXPECT_TRUE(service.work().work.empty());
        }
        (void)editor.pollShaders(service);
        const auto generation = service.generation();
        for (int i = 0; i < 100; ++i)
            EXPECT_TRUE(editor.pollShaders(service).empty());
        EXPECT_EQ(service.generation(), generation);
    }

    TEST_F(ShaderEditorTest, DocumentOperationsKeepFocusAndWaitForDrawDemand) {
        const auto python = editor.document().id;
        const auto shader_id = editor.newDocument("draft", DocumentKind::Shader);
        ASSERT_TRUE(editor.activateDocument(python));
        ASSERT_TRUE(editor.setDocumentText(shader_id, "valid shader"));
        EXPECT_EQ(editor.document().id, python);
        ASSERT_TRUE(editor.runShader(service, shader_id));
        const auto pending = editor.inspectDocument(shader_id, &service);
        ASSERT_TRUE(pending);
        ASSERT_TRUE(pending->shader_status);
        EXPECT_EQ(pending->shader_status->state, shader::ShaderState::Pending);
        EXPECT_FALSE(pending->shader_status->compiling);
        EXPECT_TRUE(pending->shader_status->pending.empty());
        EXPECT_TRUE(service.inspect().work.empty());
        ASSERT_EQ(pending->owned_bindings.size(), 1u);
        EXPECT_FALSE(pending->applied);
        EXPECT_EQ(editor.document().id, python);
        ASSERT_TRUE(settle(*pending->document.shader_asset, pending->document.asset_revision));
        EXPECT_TRUE(editor.inspectDocument(shader_id, &service)->applied);
        ASSERT_TRUE(editor.setDocumentMode(shader_id, DocumentKind::Python));
        ASSERT_TRUE(editor.setDocumentMode(shader_id, DocumentKind::Shader));
        EXPECT_EQ(editor.inspectDocument(shader_id)->text, "valid shader");
        EXPECT_EQ(editor.document().id, python);
        const auto asset = *pending->document.shader_asset;
        // A rebind to the same asset stays the Run's to stop.
        ASSERT_TRUE(service.command(shader::BindShader{{}, asset}));
        EXPECT_EQ(editor.inspectDocument(shader_id, &service)->owned_bindings.size(), 1u);
        editor.stopShader(service, shader_id);
        EXPECT_FALSE(service.boundAsset({}));
    }

    TEST_F(ShaderEditorTest, RejectedDocumentEditsDoNotChangeTextOrRunOwnership) {
        const auto id = editor.newDocument("valid", DocumentKind::Shader);
        ASSERT_TRUE(editor.runShader(service));
        const auto before = editor.inspectDocument(id, &service);
        ASSERT_TRUE(before);
        const std::string oversized(shader::kMaxShaderSourceBytes + 1, 'x');
        EXPECT_FALSE(editor.setDocumentText(id, oversized));
        EXPECT_FALSE(editor.runShader(service, id, oversized));
        EXPECT_FALSE(editor.setDocumentTarget(id, {shader::ShaderSlot::Kind::Node, {}}, service));
        EXPECT_EQ(editor.inspectDocument(id, &service)->document, before->document);
        EXPECT_EQ(editor.inspectDocument(id, &service)->text, before->text);
        EXPECT_EQ(editor.inspectDocument(id, &service)->owned_bindings, before->owned_bindings);
        EXPECT_EQ(editor.documents().size(), 2u);
        const auto other = editor.newDocument("print('other')");
        EXPECT_FALSE(editor.runShader(service, other, "mutated"));
        EXPECT_EQ(editor.inspectDocument(other)->text, "print('other')");
    }

    TEST_F(ShaderEditorTest, BuffersKeepKindIdentityTextAndSelectionAcrossToggles) {
        const std::string python = "alpha = 'λ'\nprint(alpha)\n";
        const std::string slang = "import lfs;\r\n// λ  \nLfsElement effect(LfsElement e, LfsContext c) { return e; }  \r\n";
        editor.setText(python);
        const auto first = editor.document().id;
        const auto second = editor.newDocument(slang, DocumentKind::Shader, "untitled://second");
        EXPECT_NE(first, second);
        EXPECT_EQ(editor.getTextStripped(), slang);
        auto workspace = editor.captureWorkspaceSessionState({}, true);
        auto& active = *std::ranges::find_if(workspace.open_files, [&](const auto& file) { return file.document.id == second; });
        active.editor.cursor_byte = 18;
        active.editor.selection_anchor_byte = 3;
        editor.restoreWorkspaceSessionState(workspace);
        editor.setDocumentKind(DocumentKind::Python);
        editor.setDocumentKind(DocumentKind::Shader);
        const auto after = editor.captureWorkspaceSessionState({}, true);
        const auto selected = std::ranges::find_if(after.open_files, [&](const auto& file) { return file.document.id == second; });
        ASSERT_NE(selected, after.open_files.end());
        EXPECT_EQ(selected->editor.cursor_byte, 18);
        EXPECT_EQ(selected->editor.selection_anchor_byte, 3);
        EXPECT_EQ(editor.getText(), slang);
        EXPECT_FALSE(editor.pythonAnalysisActive());
        EXPECT_TRUE(editor.activateDocument(first));
        EXPECT_EQ(editor.document().kind, DocumentKind::Python);
        EXPECT_EQ(editor.getText(), python);
        EXPECT_TRUE(editor.activateDocument(second));
        EXPECT_EQ(editor.document().kind, DocumentKind::Shader);
        EXPECT_EQ(editor.getText(), slang);
    }

    TEST_F(ShaderEditorTest, DiagnosticsUseCodePointsAndRejectNewerEditsAndOtherBuffers) {
        const auto shader_id = editor.newDocument("// λ\nλ broken\n", DocumentKind::Shader);
        const auto revision = editor.document().source_revision;
        const auto python_id = editor.newDocument("print('untouched')\n");
        const std::vector<shader::Diagnostic> diagnostics{{shader::DiagnosticSeverity::Error, "E1", "effect.slang", 2, 3, "bad <token> & value"}};
        EXPECT_TRUE(editor.applyDiagnostics(shader_id, revision, diagnostics));
        const auto ranges = editor.diagnosticRanges(shader_id);
        ASSERT_EQ(ranges.size(), 1);
        EXPECT_EQ(ranges[0].start, 9); // multibyte lambda before the marked 'b'
        EXPECT_EQ(ranges[0].end, 10);
        EXPECT_TRUE(editor.diagnosticRanges(python_id).empty());
        const auto snapshot = editor.inspectDocument(shader_id);
        ASSERT_TRUE(snapshot);
        ASSERT_EQ(snapshot->markers.size(), 1u);
        EXPECT_EQ(snapshot->markers[0].start, ranges[0].start);
        EXPECT_EQ(snapshot->markers[0].end, ranges[0].end);
        ASSERT_TRUE(editor.setDocumentText(shader_id, "corrected\n"));
        EXPECT_EQ(editor.document().id, python_id);
        EXPECT_GT(editor.inspectDocument(shader_id)->document.source_revision, revision);
        EXPECT_TRUE(editor.diagnosticRanges(shader_id).empty());
        EXPECT_FALSE(editor.applyDiagnostics(shader_id, revision, diagnostics));
        EXPECT_TRUE(editor.diagnosticRanges(shader_id).empty());
    }

    TEST_F(ShaderEditorTest, RunWhilePendingPublishesOnlyTheNewSourceAndStopRemovesSuccessfulBind) {
        schedule.gate(true);
        editor.setDocumentKind(DocumentKind::Shader);
        editor.setText("first draft");
        ASSERT_TRUE(editor.runShader(service));
        const auto asset = *editor.document().shader_asset;
        service.demandLive(std::array{std::string("projection_forward_custom")});
        ASSERT_TRUE(schedule.waitStarted(1));
        EXPECT_TRUE(service.status(asset)->compiling);
        editor.setText("second draft");
        ASSERT_TRUE(editor.runShader(service));
        EXPECT_EQ(service.assets().size(), 1);
        EXPECT_EQ(service.assets()[0].source, "second draft");
        const auto revision = editor.document().asset_revision;
        schedule.gate(false);
        ASSERT_TRUE(settle(asset, revision));
        EXPECT_EQ(service.status(asset)->displayed_revision, revision);
        EXPECT_FALSE(service.status(asset)->compiling);
        EXPECT_TRUE(editor.hasShaderBinding(service));
        editor.stopShader(service);
        present();
        EXPECT_FALSE(service.boundAsset({}));
        EXPECT_FALSE(editor.hasShaderBinding(service));
        EXPECT_EQ(service.status(asset)->state, shader::ShaderState::Unbound);
    }

    TEST_F(ShaderEditorTest, PendingStopAndFailedCompileKeepOtherBuffersAndLastGoodSafe) {
        const auto id = editor.newDocument("valid\n", DocumentKind::Shader);
        ASSERT_TRUE(editor.runShader(service));
        const auto asset = *editor.document().shader_asset;
        ASSERT_TRUE(settle(asset, editor.document().asset_revision));
        const auto good = publication();
        editor.setText("// λ\nλ broken\n");
        ASSERT_TRUE(editor.runShader(service));
        const auto failed_revision = editor.document().asset_revision;
        editor.newDocument("print('other')");
        ASSERT_TRUE(settle(asset, failed_revision));
        const auto updates = editor.pollShaders(service);
        ASSERT_FALSE(updates.empty());
        EXPECT_EQ(publication(), good);
        EXPECT_EQ(editor.diagnosticRanges(id).size(), 1);
        EXPECT_TRUE(editor.diagnosticRanges(editor.document().id).empty());
        const auto failed = editor.inspectDocument(id, &service);
        ASSERT_TRUE(failed);
        ASSERT_EQ(failed->diagnostics.size(), 1u);
        EXPECT_FALSE(failed->applied);
        EXPECT_EQ(failed->shader_status->displayed_revision, 1u);
        const auto other = editor.document().id;
        const auto associated = service.assets().front();
        ASSERT_TRUE(editor.setDocumentMode(other, DocumentKind::Shader, associated));
        const auto draft = editor.inspectDocument(other, &service);
        ASSERT_TRUE(draft);
        EXPECT_EQ(draft->text, "print('other')");
        EXPECT_TRUE(draft->diagnostics.empty());
        EXPECT_TRUE(draft->owned_bindings.empty());
        ASSERT_TRUE(editor.activateDocument(id));
        schedule.gate(true);
        editor.setText("pending correction");
        ASSERT_TRUE(editor.runShader(service));
        service.demandLive(std::array{std::string("projection_forward_custom")});
        ASSERT_TRUE(schedule.waitStarted(3));
        editor.stopShader(service);
        schedule.gate(false);
        EXPECT_FALSE(service.boundAsset({}));
        EXPECT_EQ(publication(), nullptr);
    }

    TEST_F(ShaderEditorTest, ParameterEditsUseServiceAndDoNotCompile) {
        editor.setDocumentKind(DocumentKind::Shader);
        editor.setText("valid");
        ASSERT_TRUE(editor.runShader(service));
        const auto asset = *editor.document().shader_asset;
        ASSERT_TRUE(settle(asset, editor.document().asset_revision));
        const auto links = service.status(asset)->links;
        const std::array value{0.75f};
        ASSERT_TRUE(lfs::vis::ShaderCommands{service}.setParameter(asset, "strength", value));
        EXPECT_EQ(service.status(asset)->values[0].x, 0.75f);
        EXPECT_EQ(service.status(asset)->links, links);
        EXPECT_FALSE(lfs::vis::ShaderCommands{service}.removeAsset(asset));
        ASSERT_TRUE(lfs::vis::ShaderCommands{service}.bind({}, asset));
        editor.stopShader(service);
        EXPECT_FALSE(service.boundAsset({}));
        EXPECT_FALSE(editor.hasShaderBinding(service));
        ASSERT_TRUE(lfs::vis::ShaderCommands{service}.renameAsset(asset, "Renamed fixture"));
        editor.openShaderAsset(service.assets().front());
        EXPECT_EQ(editor.getText(), "valid");
        ASSERT_TRUE(editor.runShader(service));
        ASSERT_TRUE(editor.setDocumentMode(editor.document().id, DocumentKind::Shader, service.assets().front()));
        EXPECT_TRUE(editor.stopShader(service).empty());
        EXPECT_EQ(service.boundAsset({}), asset);
        ASSERT_TRUE(lfs::vis::ShaderCommands{service}.unbind({}));
        EXPECT_TRUE(lfs::vis::ShaderCommands{service}.removeAsset(asset));
    }

    TEST(PythonConsoleInputTest, ShaderCommandsRespectRemappingUnbindingAndOneDispatch) {
        using namespace lfs::vis;
        input::InputBindings::setPersistenceEnabled(false);
        input::InputBindings bindings;
        auto& state = gui::panels::PythonConsoleState::getInstance();
        state.getTerminal()->setFocused(false);
        state.getOutputTerminal()->setFocused(false);
        state.getEditor()->newDocument("valid", DocumentKind::Python);
        EXPECT_EQ(bindings.getActionForKey(input::ToolMode::EDITOR, input::KEY_F8, 0), input::Action::NONE);
        bindings.setBinding(input::ToolMode::EDITOR, input::Action::EDITOR_TOGGLE_KIND, input::KeyTrigger{input::KEY_F8});
        InputEventDispatch dispatch;
        const FrameInputEvent toggle{.scancode = SDL_SCANCODE_F8, .dispatch = &dispatch};
        ASSERT_TRUE(state.handleShortcut(toggle, &bindings));
        EXPECT_EQ(state.getEditor()->document().kind, DocumentKind::Shader);
        EXPECT_FALSE(state.handleShortcut(toggle, &bindings));
        EXPECT_EQ(state.getEditor()->document().kind, DocumentKind::Shader);
        bindings.clearBinding(input::ToolMode::EDITOR, input::Action::EDITOR_TOGGLE_KIND);
        EXPECT_FALSE(state.handleShortcut({.scancode = SDL_SCANCODE_F8}, &bindings));
        EXPECT_EQ(state.getEditorText(), "valid");
        EXPECT_FALSE(state.formatDocument());
        EXPECT_FALSE(state.formatDocument(true));
        state.getEditor()->refreshSyntaxDiagnostics();
        EXPECT_TRUE(state.getEditor()->syntaxSymbols().empty());
        EXPECT_FALSE(state.getEditor()->foldAllSyntaxBlocks());
        EXPECT_FALSE(state.getEditor()->pythonAnalysisActive());
        state.runScriptAsync("raise AssertionError('must not execute')");
        EXPECT_FALSE(state.isScriptRunning());
        state.getEditor()->setDocumentKind(DocumentKind::Python);
        input::InputBindings::setPersistenceEnabled(true);
    }

    TEST_F(ShaderEditorTest, CopyOnlyStopsInFlightWorkAndToolbarStillUnbinds) {
        using namespace lfs::vis;
        struct InputEnvironment {
            bool persistence = input::InputBindings::isPersistenceEnabled();
            bool own_video = (SDL_WasInit(SDL_INIT_VIDEO) == 0);
            std::string previous_driver = SDL_GetHint(SDL_HINT_VIDEO_DRIVER) ? SDL_GetHint(SDL_HINT_VIDEO_DRIVER) : "";
            bool rml_ready = false;
            ~InputEnvironment() {
                input::InputBindings::setPersistenceEnabled(persistence);
                if (rml_ready)
                    Rml::Shutdown();
                if (own_video) {
                    SDL_QuitSubSystem(SDL_INIT_VIDEO);
                    if (previous_driver.empty())
                        SDL_ResetHint(SDL_HINT_VIDEO_DRIVER);
                    else
                        SDL_SetHintWithPriority(SDL_HINT_VIDEO_DRIVER, previous_driver.c_str(), SDL_HINT_OVERRIDE);
                }
            }
        } environment;
        input::InputBindings::setPersistenceEnabled(false);
        if (environment.own_video) {
            ASSERT_TRUE(SDL_SetHintWithPriority(SDL_HINT_VIDEO_DRIVER, "dummy", SDL_HINT_OVERRIDE));
            ASSERT_TRUE(SDL_InitSubSystem(SDL_INIT_VIDEO)) << SDL_GetError();
        }
        environment.rml_ready = Rml::Initialise();
        ASSERT_TRUE(environment.rml_ready);
        Rml::Element element("div");
        auto& state = gui::panels::PythonConsoleState::getInstance();
        auto* document = state.getEditor();
        state.getTerminal()->setFocused(false);
        state.getOutputTerminal()->setFocused(false);
        document->newDocument("valid shader", DocumentKind::Shader);
        document->setVimModeEnabled(false);
        input::InputBindings bindings;
        const FrameInputEvent copy{.scancode = SDL_SCANCODE_C, .modifiers = SDL_KMOD_CTRL};
        const auto key = [&](Rml::Input::KeyIdentifier id) {
            Rml::Dictionary parameters;
            parameters["key_identifier"] = static_cast<int>(id);
            parameters["ctrl_key"] = 1;
            Rml::Event event(&element, Rml::EventId::Keydown, "keydown",
                             parameters, true);
            document->processRmlEvent(element, event);
        };
        Rml::Event focus(&element, Rml::EventId::Focus, "focus", {}, true);
        document->processRmlEvent(element, focus);
        ASSERT_TRUE(document->runShader(service));
        const auto asset = *document->document().shader_asset;
        ASSERT_TRUE(settle(asset, document->document().asset_revision));
        key(Rml::Input::KI_A);
        ASSERT_TRUE(SDL_SetClipboardText("before copy"));
        ASSERT_FALSE(state.handleShortcut(copy, &bindings, &service));
        key(Rml::Input::KI_C);
        char* clipboard = SDL_GetClipboardText();
        EXPECT_STREQ(clipboard, "valid shader");
        SDL_free(clipboard);
        EXPECT_EQ(service.boundAsset({}), asset);
        EXPECT_TRUE(state.documentCommand(input::Action::EDITOR_STOP, &service));
        EXPECT_FALSE(service.boundAsset({}));

        schedule.gate(true);
        document->setText("pending shader");
        ASSERT_TRUE(document->runShader(service));
        service.demandLive(std::array{std::string("projection_forward_custom")});
        ASSERT_TRUE(schedule.waitStarted(2));
        EXPECT_TRUE(state.handleShortcut(copy, &bindings, &service));
        EXPECT_FALSE(service.boundAsset({}));
        schedule.gate(false);

        document->setDocumentKind(DocumentKind::Python);
        ASSERT_TRUE(lfs::python::ensure_initialized());
        state.runScriptAsync("import time\nprint('copy interrupt ready', flush=True)\n"
                             "for _ in range(1000): time.sleep(0.005)\n");
        const auto wait_until = [](const auto& predicate) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(6);
            while (!predicate() && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return predicate();
        };
        EXPECT_TRUE(wait_until([&] { return state.getOutputText().contains("copy interrupt ready"); }));
        EXPECT_TRUE(state.isScriptRunning());
        EXPECT_TRUE(state.handleShortcut(copy, &bindings, &service));
        EXPECT_TRUE(wait_until([&] { return !state.isScriptRunning(); }));
        EXPECT_NE(state.getOutputText().find("KeyboardInterrupt"), std::string::npos);
        EXPECT_FALSE(state.handleShortcut(copy, &bindings, &service));
        document->newDocument("");
    }

    TEST(PythonConsoleInputTest, DirtyShaderSaveAsReloadPreservesExactBytes) {
        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        const std::string source = "import lfs;\r\n// λ\t  \n\r\nLfsElement effect(LfsElement e, LfsContext c) { return e; }  ";
        const auto path = std::filesystem::temp_directory_path() / ("shader-" + lfs::core::generate_uuid_v4().to_string() + ".slang");
        state.getEditor()->newDocument(source, DocumentKind::Shader);
        state.setModified(true);
        ASSERT_TRUE(state.isModified());
        ASSERT_TRUE(state.saveDocument(path));
        EXPECT_FALSE(state.isModified());
        std::ifstream file(path, std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(file)), {});
        EXPECT_EQ(bytes, source);
        state.getEditor()->newDocument("python = True\n");
        ASSERT_TRUE(state.loadDocument(path));
        EXPECT_EQ(state.getEditor()->document().kind, DocumentKind::Shader);
        EXPECT_EQ(state.getEditorText(), source);
        EXPECT_EQ(state.getScriptPath(), path);
        EXPECT_FALSE(state.isModified());
        state.getEditor()->setText(source + "// edited\n");
        EXPECT_TRUE(state.isModified());
        ASSERT_TRUE(state.saveDocument(path));
        EXPECT_FALSE(state.isModified());
        file.close();
        std::filesystem::remove(path);
        state.getEditor()->newDocument("");
    }

    TEST_F(PythonEditorInputTest, MouseRangeDeletesInEitherDirectionAndUndoes) {
        const auto original = text();
        for (const auto keycode : {ExtKeys::DEL, ExtKeys::BACKSPACE}) {
            for (const bool backwards : {false, true}) {
                select(backwards ? 5 : 0, backwards ? 0 : 5);
                key(keycode);
                EXPECT_EQ(text(), " beta\nsecond line\nlast");
                key('z', ModifierKey::Ctrl);
                EXPECT_EQ(text(), original);
                key('z', ModifierKey::Ctrl | ModifierKey::Shift);
                EXPECT_EQ(text(), " beta\nsecond line\nlast");
                key('z', ModifierKey::Ctrl);
                EXPECT_EQ(text(), original);
            }
        }
    }

    TEST_F(PythonEditorInputTest, TypingReplacesSelectionAsOneUndoableEdit) {
        select(5, 0);
        key('q');
        EXPECT_EQ(text(), "q beta\nsecond line\nlast");
        key('z', ModifierKey::Ctrl);
        EXPECT_EQ(text(), "alpha beta\nsecond line\nlast");
        key('y', ModifierKey::Ctrl);
        EXPECT_EQ(text(), "q beta\nsecond line\nlast");
    }

    TEST_F(PythonEditorInputTest, SelectAllReplacesIncludingLastCharacter) {
        key('a', ModifierKey::Ctrl);
        key('q');
        EXPECT_EQ(text(), "q");
        key('z', ModifierKey::Ctrl);
        key('a', ModifierKey::Ctrl);
        key(ExtKeys::DEL);
        EXPECT_TRUE(text().empty());
    }

    TEST_F(PythonEditorInputTest, PlainHomeTogglesIndentAndEndUsesTheSameNavigationPath) {
        key('a', ModifierKey::Ctrl);
        for (char c : std::string("    alpha"))
            key(c);
        key(ExtKeys::HOME);
        EXPECT_EQ(cursor(), 4);
        key(ExtKeys::HOME);
        EXPECT_EQ(cursor(), 0);
        key(ExtKeys::END);
        EXPECT_EQ(cursor(), 9);
        key(ExtKeys::HOME, ModifierKey::Shift);
        key('q');
        EXPECT_EQ(text(), "q");
    }

    TEST_F(PythonEditorInputTest, HomeAndEndNavigateToFileBoundaries) {
        key(ExtKeys::END, ModifierKey::Ctrl);
        EXPECT_EQ(cursor(), text().size());
        key(ExtKeys::HOME, ModifierKey::Ctrl);
        EXPECT_EQ(cursor(), 0);
    }

    TEST_F(PythonEditorInputTest, ShiftHomeAndEndExtendAndReverseTheSelection) {
        select(3, 3);
        key(ExtKeys::END, ModifierKey::Shift);
        key(ExtKeys::HOME, ModifierKey::Shift);
        key('q');
        EXPECT_EQ(text(), "qha beta\nsecond line\nlast");
        key('z', ModifierKey::Ctrl);
        select(3, 3);
        key(ExtKeys::END, ModifierKey::Shift);
        key('q');
        EXPECT_EQ(text(), "alpq\nsecond line\nlast");
    }

    TEST_F(PythonEditorInputTest, ControlShiftHomeAndEndSelectToFileBoundaries) {
        select(11, 11);
        key(ExtKeys::HOME, ModifierKey::Ctrl | ModifierKey::Shift);
        key('q');
        EXPECT_EQ(text(), "qsecond line\nlast");
        key('z', ModifierKey::Ctrl);
        select(11, 11);
        key(ExtKeys::END, ModifierKey::Ctrl | ModifierKey::Shift);
        key('q');
        EXPECT_EQ(text(), "alpha beta\nq");
    }

    TEST_F(PythonEditorInputTest, ShiftArrowContinuesAMouseSelection) {
        select(0, 5);
        key(ExtKeys::RIGHT, ModifierKey::Shift);
        key(ExtKeys::DEL);
        EXPECT_EQ(text(), "beta\nsecond line\nlast");
    }

    TEST_F(PythonEditorInputTest, SelectionReturnReplacesAndTabIndentsAsOneEdit) {
        const auto original = text();
        for (const auto keycode : {ExtKeys::RETURN, ExtKeys::TAB}) {
            for (const bool keyboard : {false, true}) {
                key(ExtKeys::HOME, ModifierKey::Ctrl);
                if (keyboard) {
                    for (int i = 0; i < 5; ++i)
                        key(ExtKeys::RIGHT, ModifierKey::Shift);
                } else {
                    select(0, 5);
                }
                key(keycode);
                EXPECT_EQ(text(), keycode == ExtKeys::RETURN ? "\n beta\nsecond line\nlast" : "    " + original);
                key('z', ModifierKey::Ctrl);
                EXPECT_EQ(text(), original);
            }
        }
        select(0, 15);
        key(ExtKeys::TAB);
        EXPECT_EQ(text(), "    alpha beta\n    second line\nlast");
        key('z', ModifierKey::Ctrl);
        EXPECT_EQ(text(), original);
    }

    TEST_F(PythonEditorInputTest, MultibyteReplacementKeepsCursorAfterInsertedText) {
        key('a', ModifierKey::Ctrl);
        key('a');
        select(0, 1);
        key(0xa3);
        EXPECT_EQ(text(), "£");
        EXPECT_EQ(cursor(), 2);
        key('x');
        EXPECT_EQ(text(), "£x");
        key('z', ModifierKey::Ctrl);
        EXPECT_EQ(text(), "a");
    }

    TEST(PythonConsoleInputTest, ShortcutsUseEventModifiersAndIgnoreRepeats) {
        using namespace lfs::vis;
        auto& state = gui::panels::PythonConsoleState::getInstance();
        state.getTerminal()->setFocused(false);
        state.getOutputTerminal()->setFocused(false);
        state.setEditorText("keep");
        EXPECT_FALSE(state.handleShortcut({.scancode = SDL_SCANCODE_N}));
        EXPECT_EQ(state.getEditorText(), "keep");
        EXPECT_FALSE(state.handleShortcut({.scancode = SDL_SCANCODE_LCTRL, .modifiers = SDL_KMOD_CTRL}));
        EXPECT_EQ(state.getEditorText(), "keep");
        EXPECT_FALSE(state.handleShortcut({.scancode = SDL_SCANCODE_N, .modifiers = SDL_KMOD_CTRL, .repeat = true}));
        EXPECT_TRUE(state.handleShortcut({.scancode = SDL_SCANCODE_N, .modifiers = SDL_KMOD_CTRL}));
        EXPECT_TRUE(state.getEditorText().empty());
    }

    TEST(PythonConsoleInputTest, ClearEmptiesOutputAndPreservesScript) {
        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        state.setActiveTab(0);
        state.setEditorText("print(1)");
        state.getTerminal()->write("terminal");
        state.getOutputTerminal()->write("output\n");
        ASSERT_FALSE(state.getOutputTerminal()->getAllText().empty());
        state.clear();
        EXPECT_TRUE(state.getOutputTerminal()->getAllText().empty());
        EXPECT_EQ(state.getTerminal()->getAllText(), "terminal");
        EXPECT_EQ(state.getEditorText(), "print(1)");
        state.getTerminal()->reset();
    }

    TEST(PythonConsoleInputTest, ClearOnlyEmptiesTheVisibleTerminalIncludingScrollback) {
        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        state.setEditorText("print(1)");
        state.getOutputTerminal()->write("output");
        auto* terminal = state.getTerminal();
        terminal->resize(20, 2);
        terminal->write("first\r\nsecond\r\nthird\r\n");
        terminal->scrollUp(1);
        ASSERT_GT(terminal->snapshot().scroll_offset, 0);
        state.setActiveTab(1);
        state.clear();
        EXPECT_TRUE(terminal->getAllText().empty());
        EXPECT_EQ(terminal->snapshot().scroll_offset, 0);
        EXPECT_EQ(state.getOutputTerminal()->getAllText(), "output");
        EXPECT_EQ(state.getEditorText(), "print(1)");
        state.setActiveTab(0);
        state.clear();
    }

#ifndef _WIN32
    TEST(PythonConsoleInputTest, ClearRunningTerminalDiscardsScrollbackWithoutClosingSession) {
        lfs::vis::terminal::TerminalWidget terminal(20, 2);
        const auto fds = terminal.spawnEmbedded();
        ASSERT_TRUE(fds.valid());
        const std::unique_ptr<FILE, int (*)(FILE*)> session(fdopen(fds.read_fd, "r+"), &std::fclose);
        ASSERT_NE(session, nullptr);
        terminal.write("first\r\nsecond\r\nthird\r\n");
        terminal.scrollUp(1);
        ASSERT_GT(terminal.snapshot().scroll_offset, 0);
        terminal.beginSelection(0, 0);
        terminal.updateSelection(0, 3);
        terminal.endSelection();
        ASSERT_TRUE(terminal.hasSelection());
        terminal.clear();
        EXPECT_TRUE(terminal.getAllText().empty());
        EXPECT_EQ(terminal.snapshot().scroll_offset, 0);
        EXPECT_FALSE(terminal.hasSelection());
        EXPECT_TRUE(terminal.is_running());
        terminal.scrollUp(100);
        EXPECT_EQ(terminal.snapshot().scroll_offset, 0);
    }
#endif

    TEST(PythonConsoleInputTest, GrowingAWiderViewRestoresScrollbackWithoutHanging) {
        // The docked console's first layout: 80x24 -> 89x16 pushes lines to scrollback, 89x25 pops
        // them back into a wider view. Before the fix their padding cells had width 0 and the
        // second resize never returned.
        lfs::vis::terminal::TerminalWidget terminal(80, 24);
        for (int i = 0; i < 40; ++i)
            terminal.write("line " + std::to_string(i) + "\r\n");
        terminal.resize(89, 16);
        terminal.resize(89, 25);
        const auto snapshot = terminal.snapshot();
        EXPECT_EQ(snapshot.cols, 89);
        EXPECT_EQ(snapshot.rows, 25);
        EXPECT_NE(terminal.getAllText().find("line 39"), std::string::npos);
        EXPECT_NE(terminal.getAllText().find("line 20"), std::string::npos);
    }

    TEST(PythonConsoleInputTest, NarrowingAndWideningKeepsALongLineWhole) {
        // Output written before the docked console's first layout narrows it from 80 columns.
        lfs::vis::terminal::TerminalWidget terminal(80, 24);
        std::string line;
        for (int i = 0; i < 68; ++i)
            line += static_cast<char>('a' + i % 26);
        terminal.write(line + "\r\nnext\r\n");
        terminal.resize(49, 16);
        EXPECT_EQ(terminal.getAllText(), line + "\nnext");
        EXPECT_EQ(terminal.snapshot().visible_rows[1].cells[0].text, line.substr(49, 1));
        terminal.resize(80, 24);
        EXPECT_EQ(terminal.getAllText(), line + "\nnext");
    }

    TEST(PythonConsoleInputTest, ResizingKeepsWrappedScrollbackLinesWhole) {
        // Growing the view pops rows back from scrollback one at a time, without libvterm's line info.
        lfs::vis::terminal::TerminalWidget terminal(10, 3);
        const std::string line = "abcdefghij0123456789ABCDE";
        terminal.write(line + "\r\na\r\nb\r\nc\r\nd\r\n");
        ASSERT_EQ(terminal.getAllText(), line + "\na\nb\nc\nd");
        constexpr int sizes[][2] = {{10, 12}, {30, 12}, {6, 2}, {40, 20}};
        for (const auto& size : sizes) {
            terminal.resize(size[0], size[1]);
            EXPECT_EQ(terminal.getAllText(), line + "\na\nb\nc\nd") << size[0] << "x" << size[1];
        }
    }

    TEST(PythonConsoleInputTest, ResizingKeepsWrappedLinesAndTheCursorInPlace) {
        // libvterm 0.3.3 alone reads above the screen for a line that starts in scrollback, moves a cursor
        // below a full row onto that row, aborts when the cursor ends a wrapped line that no longer fits,
        // and keeps only the first row of such a line.
        lfs::vis::terminal::TerminalWidget terminal(10, 3);
        terminal.write("abcdefghij0123456789ABCDE\r\nz\r\n");
        terminal.resize(20, 5);
        EXPECT_EQ(terminal.getAllText(), "abcdefghij0123456789ABCDE\nz");
        terminal.write("ABCDEFGHIJKLMNOPQRST\r\n");
        terminal.resize(8, 5);
        terminal.write("end\r\n" + std::string(31, 'x'));
        terminal.resize(8, 2);
        terminal.resize(6, 2);
        terminal.write("!");
        EXPECT_EQ(terminal.getAllText(),
                  "abcdefghij0123456789ABCDE\nz\nABCDEFGHIJKLMNOPQRST\nend\n" + std::string(31, 'x') + "!");
    }

    TEST(PythonConsoleInputTest, AllTextJoinsRowsWrappedAtTheTerminalWidth) {
        // MCP editor output parses this text, so a long line must not gain newlines at the console width.
        lfs::vis::terminal::TerminalWidget terminal(6, 2);
        terminal.write("hello world\r\nnext\r\n");
        EXPECT_EQ(terminal.getAllText(), "hello world\nnext");
        terminal.write("[0.123456789, 1.5]");
        EXPECT_EQ(terminal.getAllText(), "hello world\nnext\n[0.123456789, 1.5]");
    }

    TEST(PythonConsoleInputTest, OutputBeforeThePanelFirstDrawsKeepsItsLines) {
        // MCP can run a script before the console panel draws; its lines must not stair-step.
        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        auto* const output = state.getOutputTerminal();
        ASSERT_NE(output, nullptr);
        output->clear();
        state.addInfo("first");
        state.addInfo("second");
        EXPECT_EQ(state.getOutputText(), "first\nsecond");
        output->clear();
    }

    TEST(PythonConsoleInputTest, ClearDoesNothingOnPackagesTab) {
        auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
        state.setEditorText("print(1)");
        state.getOutputTerminal()->write("output");
        state.getTerminal()->write("terminal");
        state.setActiveTab(2);
        state.clear();
        EXPECT_EQ(state.getOutputTerminal()->getAllText(), "output");
        EXPECT_EQ(state.getTerminal()->getAllText(), "terminal");
        EXPECT_EQ(state.getEditorText(), "print(1)");
        state.getTerminal()->reset();
        state.setActiveTab(0);
        state.clear();
    }

#ifndef _WIN32
    TEST(PythonConsoleInputTest, ClearPreservesWrappedInputAndCombiningCharacters) {
        lfs::vis::terminal::TerminalWidget terminal(12, 5);
        const auto fds = terminal.spawnEmbedded();
        ASSERT_TRUE(fds.valid());
        const std::unique_ptr<FILE, int (*)(FILE*)> session(fdopen(fds.read_fd, "r+"), &std::fclose);
        terminal.write("old output\r\n>>> abcdefghijklmne\xcc\x81op");
        const auto before = terminal.snapshot();
        terminal.clear();
        const auto after = terminal.snapshot();
        EXPECT_EQ(after.cursor_row, before.cursor_row - 1);
        EXPECT_EQ(after.cursor_col, before.cursor_col);
        for (int row = 0; row <= after.cursor_row; ++row)
            for (int col = 0; col < after.cols; ++col)
                EXPECT_EQ(after.visible_rows[row].cells[col].text, before.visible_rows[row + 1].cells[col].text);
        terminal.clear();
        const auto again = terminal.snapshot();
        EXPECT_EQ(again.cursor_row, after.cursor_row);
        for (int row = 0; row <= after.cursor_row; ++row)
            for (int col = 0; col < after.cols; ++col)
                EXPECT_EQ(again.visible_rows[row].cells[col].text, after.visible_rows[row].cells[col].text);
        terminal.reset();
        terminal.resize(12, 2);
        terminal.write("old output\r\n>>> " + std::string(45, 'w') + "e\xcc\x81");
        const auto wrapped = terminal.snapshot();
        terminal.clear();
        EXPECT_TRUE(terminal.getAllText().starts_with(">>> "));
        EXPECT_EQ(terminal.getAllText().find("old output"), std::string::npos);
        const auto restored = terminal.snapshot();
        for (int row = 0; row < restored.rows; ++row)
            for (int col = 0; col < restored.cols; ++col)
                EXPECT_EQ(restored.visible_rows[row].cells[col].text, wrapped.visible_rows[row].cells[col].text);
        EXPECT_TRUE(terminal.is_running());
    }

    TEST(PythonConsoleInputTest, ClearDoesNotWriteToRunningProgramsInput) {
        lfs::vis::terminal::TerminalWidget terminal(40, 4);
        const auto fds = terminal.spawnEmbedded();
        ASSERT_TRUE(fds.valid());
        const std::unique_ptr<FILE, int (*)(FILE*)> session(fdopen(fds.read_fd, "r+"), &std::fclose);
        ASSERT_NE(session, nullptr);
        termios settings{};
        ASSERT_EQ(tcgetattr(fds.read_fd, &settings), 0);
        cfmakeraw(&settings);
        ASSERT_EQ(tcsetattr(fds.read_fd, TCSANOW, &settings), 0);
        terminal.write("old output\r\n>>> ");
        terminal.clear();
        EXPECT_EQ(terminal.getAllText(), ">>>");
        EXPECT_EQ(terminal.snapshot().cursor_col, 4);
        pollfd descriptor{fds.read_fd, POLLIN, 0};
        EXPECT_EQ(poll(&descriptor, 1, 20), 0);
        terminal.write("x = input()\r\n");
        terminal.clear();
        EXPECT_TRUE(terminal.getAllText().empty());
        EXPECT_EQ(poll(&descriptor, 1, 20), 0);
        terminal.sendText("answer\n");
        ASSERT_GT(poll(&descriptor, 1, 1000), 0);
        char result[32]{};
        const auto count = read(fds.read_fd, result, sizeof(result));
        ASSERT_GT(count, 0);
        EXPECT_EQ(std::string(result, count), "answer\n");
        EXPECT_TRUE(terminal.is_running());
    }

    TEST(PythonConsoleInputTest, TerminalWritesTextAndKeysInSdlArrivalOrder) {
        using namespace lfs::vis;
        terminal::TerminalWidget terminal;
        const auto fds = terminal.spawnEmbedded();
        ASSERT_TRUE(fds.valid());
        const std::unique_ptr<FILE, int (*)(FILE*)> session(fdopen(fds.read_fd, "r+"), &std::fclose);
        ASSERT_NE(session, nullptr);
        termios settings{};
        ASSERT_EQ(tcgetattr(fds.read_fd, &settings), 0);
        cfmakeraw(&settings);
        ASSERT_EQ(tcsetattr(fds.read_fd, TCSANOW, &settings), 0);
        ASSERT_NE(fcntl(fds.read_fd, F_SETFL, O_NONBLOCK), -1);
        FrameInputBuffer frame;
        const auto text = [&](const char* value) {
            SDL_Event event{};
            event.type = SDL_EVENT_TEXT_INPUT;
            event.text.text = value;
            frame.processEvent(event);
        };
        const auto key = [&](SDL_Scancode sc) {
            SDL_Event event{};
            event.type = SDL_EVENT_KEY_DOWN;
            event.key.scancode = sc;
            frame.processEvent(event);
            event.type = SDL_EVENT_KEY_UP;
            frame.processEvent(event);
        };
        text("(");
        text(")");
        key(SDL_SCANCODE_RETURN);
        text("a");
        key(SDL_SCANCODE_BACKSPACE);
        text("b");
        SDL_Event control{};
        control.type = SDL_EVENT_KEY_DOWN;
        control.key.scancode = SDL_SCANCODE_L;
        control.key.mod = SDL_KMOD_CTRL;
        frame.processEvent(control);
        control.type = SDL_EVENT_KEY_UP;
        control.key.scancode = SDL_SCANCODE_LCTRL;
        control.key.mod = SDL_KMOD_NONE;
        frame.processEvent(control);
        frame.key_mods = SDL_KMOD_NONE;
        const auto input = gui::buildPanelInputFromSDL(frame);
        for (const auto& event : input.input_events)
            terminal::processInputEvent(terminal, event);
        const std::string expected = "()\ra\x7f"
                                     "b\x0c";
        std::string received;
        pollfd descriptor{fds.read_fd, POLLIN, 0};
        while (received.size() < expected.size()) {
            ASSERT_GT(poll(&descriptor, 1, 1000), 0);
            char chunk[32]{};
            const auto count = read(fds.read_fd, chunk, sizeof(chunk));
            ASSERT_GT(count, 0);
            received.append(chunk, count);
        }
        EXPECT_EQ(received, expected);
    }
#endif

    TEST(PythonConsoleInputTest, KeyModifiersSurviveReleaseInSameFrame) {
        lfs::vis::FrameInputBuffer frame;
        SDL_Event event{};
        event.type = SDL_EVENT_KEY_DOWN;
        event.key.scancode = SDL_SCANCODE_Z;
        event.key.mod = SDL_KMOD_CTRL | SDL_KMOD_SHIFT;
        frame.processEvent(event);
        event.type = SDL_EVENT_KEY_UP;
        event.key.scancode = SDL_SCANCODE_LCTRL;
        event.key.mod = SDL_KMOD_NONE;
        frame.processEvent(event);
        frame.key_mods = SDL_KMOD_NONE;
        const auto panel = lfs::vis::gui::buildPanelInputFromSDL(frame);
        ASSERT_EQ(panel.input_events.size(), 2);
        EXPECT_EQ(panel.input_events[0].scancode, SDL_SCANCODE_Z);
        EXPECT_EQ(panel.input_events[0].modifiers, SDL_KMOD_CTRL | SDL_KMOD_SHIFT);
        EXPECT_EQ(panel.input_events[0].kind, lfs::vis::FrameInputEventKind::KeyDown);
        EXPECT_EQ(panel.input_events[1].kind, lfs::vis::FrameInputEventKind::KeyUp);
        frame.beginFrame();
        EXPECT_TRUE(frame.input_events.empty());
    }
} // namespace
