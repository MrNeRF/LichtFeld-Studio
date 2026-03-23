/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "pty_process.hpp"

#ifdef _WIN32
#undef small // Windows rpcndr.h defines 'small' as 'char'; conflicts with libvterm
#endif

#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include <vterm.h>
#include <imgui.h>

namespace lfs::vis::terminal {

    class TerminalWidget {
    public:
        explicit TerminalWidget(int cols = 80, int rows = 24);
        ~TerminalWidget();

        TerminalWidget(const TerminalWidget&) = delete;
        TerminalWidget& operator=(const TerminalWidget&) = delete;

        // Spawn shell. Empty = default shell, "python3" for Python REPL
        void spawn(const std::string& shell = "");

        // Create PTY pair without forking. Returns fds for the Python-side I/O.
        EmbeddedFds spawnEmbedded();

        // Render terminal in available space. Returns true if content changed.
        bool render(ImFont* mono_font = nullptr);

        // Handle keyboard input (call when terminal is focused)
        void processInput();

        // State
        [[nodiscard]] bool is_running() const { return pty_.is_running(); }
        [[nodiscard]] bool has_output() const { return has_new_output_; }
        [[nodiscard]] bool isFocused() const { return is_focused_; }

        // Scrollback
        void scrollUp(int lines = 1);
        void scrollDown(int lines = 1);
        void scrollToBottom();

        // Copy/paste
        [[nodiscard]] std::string getSelection() const;
        [[nodiscard]] std::string getAllText() const;
        void paste(const std::string& text);

        // Clear screen (sends Ctrl+L to PTY, or resets vterm if no PTY)
        void clear();

        // Direct text output (for read-only output terminal without PTY)
        void write(const char* data, size_t len);
        void write(const std::string& text) { write(text.data(), text.size()); }

        // Reset terminal state
        void reset();

        // Read-only mode (disables keyboard input, for output-only terminals)
        void setReadOnly(bool readonly) { read_only_ = readonly; }
        [[nodiscard]] bool isReadOnly() const { return read_only_; }

        // Send text to PTY (for executing code programmatically)
        void sendToPty(const std::string& text);

        // Send interrupt signal (Ctrl+C) to stop running process
        void interrupt();

    private:
        void pump();
        void initVterm();
        void destroyVterm();
        void drawCell(ImDrawList* dl, int row, int col, const ImVec2& origin);
        void handleResize(int new_cols, int new_rows);

        // libvterm callbacks
        static int onDamage(VTermRect rect, void* user);
        static int onMoveCursor(VTermPos pos, VTermPos oldpos, int visible, void* user);
        static int onBell(void* user);
        static int onResize(int rows, int cols, void* user);
        static int onPushline(int cols, const VTermScreenCell* cells, void* user);
        static int onPopline(int cols, VTermScreenCell* cells, void* user);

        ImU32 vtermColorToImU32(VTermColor color) const;

        PtyProcess pty_;
        VTerm* vt_ = nullptr;
        VTermScreen* screen_ = nullptr;

        int cols_;
        int rows_;
        float char_width_ = 0.0f;
        float char_height_ = 0.0f;

        // Cursor
        VTermPos cursor_pos_ = {0, 0};
        bool cursor_visible_ = true;
        float cursor_blink_time_ = 0.0f;

        // Selection
        bool is_selecting_ = false;
        VTermPos selection_start_ = {0, 0};
        VTermPos selection_end_ = {0, 0};

        // Scrollback buffer
        struct ScrollbackLine {
            std::vector<VTermScreenCell> cells;
        };
        std::deque<ScrollbackLine> scrollback_;
        int scroll_offset_ = 0;
        static constexpr int MAX_SCROLLBACK = 10000;

        // State
        bool has_new_output_ = false;
        bool needs_redraw_ = true;
        bool is_focused_ = false;
        bool read_only_ = false;
        mutable std::mutex mutex_;

        // Read buffer
        char read_buffer_[4096];
    };

} // namespace lfs::vis::terminal
