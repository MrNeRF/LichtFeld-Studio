/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "pty_process.hpp"

#ifdef _WIN32
#undef small // Windows rpcndr.h defines 'small' as 'char'; conflicts with libvterm
#endif

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <vterm.h>

namespace lfs::vis::terminal {

    enum class TerminalKey {
        Enter,
        Backspace,
        Tab,
        Escape,
        Up,
        Down,
        Right,
        Left,
        Home,
        End,
        PageUp,
        PageDown,
        Delete,
        Insert,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
    };

    using TerminalColor = uint32_t;

    struct TerminalCellSnapshot {
        std::string text;
        TerminalColor foreground = 0;
        TerminalColor background = 0;
        bool selected = false;
        bool reverse = false;
        bool bold = false;
        bool underline = false;
    };

    struct TerminalRowSnapshot {
        std::vector<TerminalCellSnapshot> cells;
    };

    struct TerminalSnapshot {
        int cols = 0;
        int rows = 0;
        int cursor_col = 0;
        int cursor_row = 0;
        bool cursor_visible = false;
        bool focused = false;
        int scroll_offset = 0;
        std::vector<TerminalRowSnapshot> visible_rows;
    };

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

        // Backend-neutral update/render surface for RmlUi and tests.
        void update();
        void resize(int cols, int rows);
        [[nodiscard]] TerminalSnapshot snapshot() const;
        [[nodiscard]] int cols() const { return cols_; }
        [[nodiscard]] int rows() const { return rows_; }

        void setFocused(bool focused);
        void sendText(std::string_view text);
        void sendCodepoint(uint32_t codepoint);
        void sendKey(TerminalKey key);
        void sendControl(char letter);
        void beginSelection(int row, int col);
        void updateSelection(int row, int col);
        void endSelection();
        [[nodiscard]] bool hasSelection() const;

        // State
        [[nodiscard]] bool is_running() const { return pty_.is_running(); }
        [[nodiscard]] bool has_output() const { return has_new_output_.load(); }
        [[nodiscard]] bool isFocused() const { return is_focused_.load(); }
        [[nodiscard]] bool needsRedraw() const { return rendered_generation_.load() != redraw_generation_.load(); }
        [[nodiscard]] uint64_t redrawGeneration() const { return redraw_generation_.load(); }
        void markRendered(uint64_t generation);

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
        void setReadOnly(bool readonly) { read_only_.store(readonly); }
        [[nodiscard]] bool isReadOnly() const { return read_only_.load(); }

        // Send text to PTY (for executing code programmatically)
        void sendToPty(const std::string& text);

        // Send interrupt signal (Ctrl+C) to stop running process
        void interrupt();

    private:
        void pump();
        void initVterm();
        void destroyVterm();
        void handleResize(int new_cols, int new_rows);
        void markDirty();

        // libvterm callbacks
        static int onDamage(VTermRect rect, void* user);
        static int onMoveCursor(VTermPos pos, VTermPos oldpos, int visible, void* user);
        static int onBell(void* user);
        static int onResize(int rows, int cols, void* user);
        static int onPushline(int cols, const VTermScreenCell* cells, void* user);
        static int onPopline(int cols, VTermScreenCell* cells, void* user);

        [[nodiscard]] TerminalColor vtermColorToPackedColor(VTermColor color) const;
        [[nodiscard]] bool isCellSelected(int row, int col) const;
        [[nodiscard]] bool getVisibleCell(int visible_row,
                                          int col,
                                          int effective_scroll_offset,
                                          VTermScreenCell& cell) const;

        PtyProcess pty_;
        VTerm* vt_ = nullptr;
        VTermScreen* screen_ = nullptr;

        int cols_;
        int rows_;

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
        std::atomic<bool> has_new_output_{false};
        std::atomic<bool> is_focused_{false};
        std::atomic<bool> read_only_{false};
        std::atomic<uint64_t> redraw_generation_{1};
        std::atomic<uint64_t> rendered_generation_{0};
        mutable std::mutex mutex_;

        // Read buffer
        char read_buffer_[4096];
    };

} // namespace lfs::vis::terminal
