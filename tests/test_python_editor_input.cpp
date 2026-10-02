/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panel_input_utils.hpp"
#include "gui/panels/python_console_panel.hpp"
#include "gui/terminal/terminal_input.hpp"
#include "gui/terminal/terminal_widget.hpp"
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
