/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "terminal_widget.hpp"
#include <algorithm>
#include <core/logger.hpp>
#include <cstring>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#else
#include <pty.h>
#include <unistd.h>
#endif

namespace lfs::vis::terminal {

    namespace {

        constexpr TerminalColor packRgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
            return static_cast<TerminalColor>(r) |
                   (static_cast<TerminalColor>(g) << 8u) |
                   (static_cast<TerminalColor>(b) << 16u) |
                   (static_cast<TerminalColor>(a) << 24u);
        }

        constexpr TerminalColor TRANSPARENT_COLOR = 0;
        constexpr TerminalColor BG_COLOR = packRgba(30, 30, 30, 255);
        constexpr TerminalColor SELECTION_COLOR = packRgba(100, 100, 200, 200);
        constexpr TerminalColor DEFAULT_FG = packRgba(229, 229, 229, 255);

        struct TerminalKeyMapping {
            TerminalKey key;
            const char* seq;
        };

        constexpr TerminalKeyMapping TERMINAL_KEY_MAPPINGS[] = {
            {TerminalKey::Enter, "\r"},
            {TerminalKey::Backspace, "\x7f"},
            {TerminalKey::Tab, "\t"},
            {TerminalKey::Escape, "\x1b"},
            {TerminalKey::Up, "\x1b[A"},
            {TerminalKey::Down, "\x1b[B"},
            {TerminalKey::Right, "\x1b[C"},
            {TerminalKey::Left, "\x1b[D"},
            {TerminalKey::Home, "\x1b[H"},
            {TerminalKey::End, "\x1b[F"},
            {TerminalKey::PageUp, "\x1b[5~"},
            {TerminalKey::PageDown, "\x1b[6~"},
            {TerminalKey::Delete, "\x1b[3~"},
            {TerminalKey::Insert, "\x1b[2~"},
            {TerminalKey::F1, "\x1bOP"},
            {TerminalKey::F2, "\x1bOQ"},
            {TerminalKey::F3, "\x1bOR"},
            {TerminalKey::F4, "\x1bOS"},
            {TerminalKey::F5, "\x1b[15~"},
            {TerminalKey::F6, "\x1b[17~"},
            {TerminalKey::F7, "\x1b[18~"},
            {TerminalKey::F8, "\x1b[19~"},
            {TerminalKey::F9, "\x1b[20~"},
            {TerminalKey::F10, "\x1b[21~"},
            {TerminalKey::F11, "\x1b[23~"},
            {TerminalKey::F12, "\x1b[24~"},
        };

        size_t encodeUtf8(uint32_t codepoint, char (&out)[4]) {
            if (codepoint < 0x80) {
                out[0] = static_cast<char>(codepoint);
                return 1;
            }
            if (codepoint < 0x800) {
                out[0] = static_cast<char>(0xC0 | (codepoint >> 6));
                out[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                return 2;
            }
            if (codepoint < 0x10000) {
                out[0] = static_cast<char>(0xE0 | (codepoint >> 12));
                out[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                out[2] = static_cast<char>(0x80 | (codepoint & 0x3F));
                return 3;
            }
            if (codepoint > 0x10FFFF) {
                return 0;
            }
            out[0] = static_cast<char>(0xF0 | (codepoint >> 18));
            out[1] = static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            out[2] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            out[3] = static_cast<char>(0x80 | (codepoint & 0x3F));
            return 4;
        }

        const char* terminalKeySequence(TerminalKey key) {
            for (const auto& mapping : TERMINAL_KEY_MAPPINGS) {
                if (mapping.key == key)
                    return mapping.seq;
            }
            return nullptr;
        }

        std::string cellText(const VTermScreenCell& cell) {
            std::string text;
            for (int i = 0; i < VTERM_MAX_CHARS_PER_CELL && cell.chars[i]; ++i) {
                char tmp[4];
                const size_t n = encodeUtf8(cell.chars[i], tmp);
                if (n > 0)
                    text.append(tmp, n);
            }
            return text;
        }

    } // namespace

    TerminalWidget::TerminalWidget(int cols, int rows) : cols_(cols),
                                                         rows_(rows) {
        initVterm();
    }

    TerminalWidget::~TerminalWidget() {
        destroyVterm();
    }

    void TerminalWidget::initVterm() {
        vt_ = vterm_new(rows_, cols_);
        vterm_set_utf8(vt_, true);
        screen_ = vterm_obtain_screen(vt_);

        static VTermScreenCallbacks callbacks = {
            .damage = onDamage,
            .moverect = nullptr,
            .movecursor = onMoveCursor,
            .settermprop = nullptr,
            .bell = onBell,
            .resize = onResize,
            .sb_pushline = onPushline,
            .sb_popline = onPopline,
            .sb_clear = nullptr,
            .sb_pushline4 = nullptr,
        };

        vterm_screen_set_callbacks(screen_, &callbacks, this);
        vterm_screen_reset(screen_, 1);
    }

    void TerminalWidget::destroyVterm() {
        if (vt_) {
            vterm_free(vt_);
            vt_ = nullptr;
            screen_ = nullptr;
        }
    }

    void TerminalWidget::spawn(const std::string& shell) {
        pty_.close();
        destroyVterm();
        initVterm();
        scrollback_.clear();
        scroll_offset_ = 0;
        markDirty();

        if (!pty_.spawn(shell, cols_, rows_)) {
            LOG_ERROR("Failed to spawn: {}", shell.empty() ? "default shell" : shell);
        }
    }

    EmbeddedFds TerminalWidget::spawnEmbedded() {
        pty_.close();
        destroyVterm();
        initVterm();
        scrollback_.clear();
        scroll_offset_ = 0;
        markDirty();

#ifdef _WIN32
        // stdin pair: Python reads from stdin_read, terminal writes to stdin_write
        HANDLE stdin_read = INVALID_HANDLE_VALUE;
        HANDLE stdin_write = INVALID_HANDLE_VALUE;
        if (!CreatePipe(&stdin_read, &stdin_write, nullptr, 0)) {
            LOG_ERROR("CreatePipe for stdin failed");
            return {};
        }

        // stdout pair: Python writes to stdout_write, terminal reads from stdout_read
        HANDLE stdout_read = INVALID_HANDLE_VALUE;
        HANDLE stdout_write = INVALID_HANDLE_VALUE;
        if (!CreatePipe(&stdout_read, &stdout_write, nullptr, 0)) {
            LOG_ERROR("CreatePipe for stdout failed");
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            return {};
        }

        // Terminal side: reads from stdout_read, writes to stdin_write
        if (!pty_.attachPipes(stdout_read, stdin_write)) {
            CloseHandle(stdin_read);
            CloseHandle(stdin_write);
            CloseHandle(stdout_read);
            CloseHandle(stdout_write);
            return {};
        }

        // Python side: convert HANDLEs to C file descriptors
        const int py_read_fd = _open_osfhandle(reinterpret_cast<intptr_t>(stdin_read), _O_RDONLY);
        const int py_write_fd = _open_osfhandle(reinterpret_cast<intptr_t>(stdout_write), _O_WRONLY);

        if (py_read_fd < 0 || py_write_fd < 0) {
            LOG_ERROR("_open_osfhandle failed");
            pty_.close();
            if (py_read_fd < 0)
                CloseHandle(stdin_read);
            if (py_write_fd < 0)
                CloseHandle(stdout_write);
            return {};
        }

        return {py_read_fd, py_write_fd};
#else
        int master = -1, slave = -1;
        struct winsize ws = {};
        ws.ws_col = static_cast<unsigned short>(cols_);
        ws.ws_row = static_cast<unsigned short>(rows_);

        if (openpty(&master, &slave, nullptr, nullptr, &ws) < 0) {
            LOG_ERROR("openpty failed");
            return {};
        }

        if (!pty_.attach(master)) {
            ::close(master);
            ::close(slave);
            return {};
        }

        return {slave, slave};
#endif
    }

    void TerminalWidget::pump() {
        if (!pty_.is_running())
            return;

        ssize_t n;
        while ((n = pty_.read(read_buffer_, sizeof(read_buffer_))) > 0) {
            std::lock_guard lock(mutex_);
            vterm_input_write(vt_, read_buffer_, static_cast<size_t>(n));
            has_new_output_.store(true);
            markDirty();
        }
    }

    void TerminalWidget::update() {
        pump();
    }

    void TerminalWidget::resize(int cols, int rows) {
        handleResize(std::max(1, cols), std::max(1, rows));
    }

    TerminalSnapshot TerminalWidget::snapshot() const {
        std::lock_guard lock(mutex_);

        TerminalSnapshot result;
        result.cols = cols_;
        result.rows = rows_;
        result.cursor_col = cursor_pos_.col;
        result.cursor_row = cursor_pos_.row;
        result.cursor_visible = cursor_visible_;
        result.focused = is_focused_.load();
        result.scroll_offset = std::min(scroll_offset_, static_cast<int>(scrollback_.size()));
        result.visible_rows.resize(static_cast<size_t>(rows_));

        const int scrollback_size = static_cast<int>(scrollback_.size());
        const int eff_offset = std::min(scroll_offset_, scrollback_size);
        const auto fill_cell = [&](TerminalCellSnapshot& out,
                                   const VTermScreenCell& cell,
                                   const bool selected) {
            out.text = cellText(cell);
            out.selected = selected;
            out.reverse = cell.attrs.reverse != 0;
            out.bold = cell.attrs.bold != 0;
            out.underline = cell.attrs.underline != 0;

            out.background = vtermColorToPackedColor(cell.bg);
            if (out.reverse)
                out.background = vtermColorToPackedColor(cell.fg);
            if (out.selected)
                out.background = SELECTION_COLOR;

            out.foreground = vtermColorToPackedColor(cell.fg);
            if (out.reverse) {
                out.foreground = vtermColorToPackedColor(cell.bg);
                if (out.foreground == TRANSPARENT_COLOR)
                    out.foreground = BG_COLOR;
            }
            if (out.foreground == TRANSPARENT_COLOR)
                out.foreground = DEFAULT_FG;
        };

        for (int row = 0; row < rows_; ++row) {
            auto& snapshot_row = result.visible_rows[static_cast<size_t>(row)];
            snapshot_row.cells.resize(static_cast<size_t>(cols_));

            if (row < eff_offset) {
                const int idx = eff_offset - 1 - row;
                const auto& line = scrollback_[static_cast<size_t>(idx)];
                for (int col = 0; col < cols_; ++col) {
                    VTermScreenCell cell{};
                    if (col < static_cast<int>(line.cells.size()))
                        cell = line.cells[static_cast<size_t>(col)];
                    auto& out = snapshot_row.cells[static_cast<size_t>(col)];
                    fill_cell(out, cell, isCellSelected(row, col));
                }
                continue;
            }

            const int screen_row = row - eff_offset;
            if (screen_row >= rows_)
                continue;

            for (int col = 0; col < cols_; ++col) {
                VTermScreenCell cell;
                vterm_screen_get_cell(screen_, {screen_row, col}, &cell);

                auto& out = snapshot_row.cells[static_cast<size_t>(col)];
                fill_cell(out, cell, isCellSelected(row, col));
            }
        }

        return result;
    }

    void TerminalWidget::setFocused(bool focused) {
        is_focused_.store(focused);
        if (focused)
            cursor_blink_time_ = 0.0f;
        markDirty();
    }

    void TerminalWidget::sendText(std::string_view text) {
        if (read_only_.load() || text.empty() || !pty_.is_running())
            return;
        if (pty_.write(text.data(), text.size()) < 0) {
            LOG_ERROR("PTY write failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::sendCodepoint(uint32_t codepoint) {
        char utf8[4];
        const size_t len = encodeUtf8(codepoint, utf8);
        if (len == 0)
            return;
        sendText(std::string_view(utf8, len));
    }

    void TerminalWidget::sendKey(TerminalKey key) {
        if (const char* seq = terminalKeySequence(key)) {
            sendText(seq);
        }
    }

    void TerminalWidget::sendControl(char letter) {
        if (read_only_.load() || !pty_.is_running())
            return;

        char upper = letter;
        if (upper >= 'a' && upper <= 'z')
            upper = static_cast<char>(upper - 'a' + 'A');
        if (upper < 'A' || upper > 'Z')
            return;

        const char control = static_cast<char>(1 + (upper - 'A'));
        if (pty_.write(&control, 1) < 0) {
            LOG_ERROR("PTY write failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::beginSelection(int row, int col) {
        std::lock_guard lock(mutex_);
        row = std::clamp(row, 0, std::max(0, rows_ - 1));
        col = std::clamp(col, 0, std::max(0, cols_ - 1));
        selection_start_ = {row, col};
        selection_end_ = selection_start_;
        is_selecting_ = true;
        markDirty();
    }

    void TerminalWidget::updateSelection(int row, int col) {
        std::lock_guard lock(mutex_);
        row = std::clamp(row, 0, std::max(0, rows_ - 1));
        col = std::clamp(col, 0, std::max(0, cols_ - 1));
        selection_end_ = {row, col};
        markDirty();
    }

    void TerminalWidget::endSelection() {
        std::lock_guard lock(mutex_);
        is_selecting_ = false;
        markDirty();
    }

    bool TerminalWidget::hasSelection() const {
        std::lock_guard lock(mutex_);
        return selection_start_.row != selection_end_.row || selection_start_.col != selection_end_.col;
    }

    void TerminalWidget::markRendered(const uint64_t generation) {
        rendered_generation_.store(generation);
        if (redraw_generation_.load() == generation)
            has_new_output_.store(false);
    }

    void TerminalWidget::handleResize(int new_cols, int new_rows) {
        {
            std::lock_guard lock(mutex_);
            if (new_cols == cols_ && new_rows == rows_)
                return;

            cols_ = new_cols;
            rows_ = new_rows;
            vterm_set_size(vt_, rows_, cols_);
        }

        pty_.resize(new_cols, new_rows);
        markDirty();
    }

    void TerminalWidget::markDirty() {
        redraw_generation_.fetch_add(1);
    }

    TerminalColor TerminalWidget::vtermColorToPackedColor(VTermColor color) const {
        if (VTERM_COLOR_IS_DEFAULT_FG(&color) || VTERM_COLOR_IS_DEFAULT_BG(&color)) {
            return TRANSPARENT_COLOR;
        }
        if (VTERM_COLOR_IS_INDEXED(&color)) {
            vterm_screen_convert_color_to_rgb(screen_, &color);
        }
        if (VTERM_COLOR_IS_RGB(&color)) {
            return packRgba(color.rgb.red, color.rgb.green, color.rgb.blue, 255);
        }
        return DEFAULT_FG;
    }

    bool TerminalWidget::isCellSelected(int row, int col) const {
        if (selection_start_.row == selection_end_.row && selection_start_.col == selection_end_.col) {
            return false;
        }

        VTermPos start = selection_start_, end = selection_end_;
        if (start.row > end.row || (start.row == end.row && start.col > end.col)) {
            std::swap(start, end);
        }

        return (row > start.row || (row == start.row && col >= start.col)) &&
               (row < end.row || (row == end.row && col <= end.col));
    }

    bool TerminalWidget::getVisibleCell(int visible_row,
                                        int col,
                                        int effective_scroll_offset,
                                        VTermScreenCell& cell) const {
        if (visible_row < 0 || visible_row >= rows_ || col < 0 || col >= cols_) {
            return false;
        }

        std::memset(&cell, 0, sizeof(VTermScreenCell));
        if (visible_row < effective_scroll_offset) {
            const int idx = effective_scroll_offset - 1 - visible_row;
            if (idx < 0 || idx >= static_cast<int>(scrollback_.size())) {
                return false;
            }
            const auto& line = scrollback_[static_cast<size_t>(idx)];
            if (col >= static_cast<int>(line.cells.size())) {
                return true;
            }
            cell = line.cells[static_cast<size_t>(col)];
            return true;
        }

        const int screen_row = visible_row - effective_scroll_offset;
        if (screen_row < 0 || screen_row >= rows_) {
            return false;
        }
        vterm_screen_get_cell(screen_, {screen_row, col}, &cell);
        return true;
    }

    void TerminalWidget::scrollUp(int lines) {
        std::lock_guard lock(mutex_);
        scroll_offset_ = std::min(scroll_offset_ + lines, static_cast<int>(scrollback_.size()));
        markDirty();
    }

    void TerminalWidget::scrollDown(int lines) {
        std::lock_guard lock(mutex_);
        scroll_offset_ = std::max(0, scroll_offset_ - lines);
        markDirty();
    }

    void TerminalWidget::scrollToBottom() {
        std::lock_guard lock(mutex_);
        scroll_offset_ = 0;
        markDirty();
    }

    std::string TerminalWidget::getSelection() const {
        std::lock_guard lock(mutex_);

        if (selection_start_.row == selection_end_.row && selection_start_.col == selection_end_.col) {
            return {};
        }

        VTermPos start = selection_start_, end = selection_end_;
        if (start.row > end.row || (start.row == end.row && start.col > end.col)) {
            std::swap(start, end);
        }

        std::string result;
        const int scrollback_size = static_cast<int>(scrollback_.size());
        const int eff_offset = std::min(scroll_offset_, scrollback_size);
        for (int row = start.row; row <= end.row; ++row) {
            const int col_start = (row == start.row) ? start.col : 0;
            const int col_end = (row == end.row) ? end.col : cols_ - 1;

            for (int col = col_start; col <= col_end; ++col) {
                VTermScreenCell cell{};
                if (!getVisibleCell(row, col, eff_offset, cell)) {
                    result += ' ';
                    continue;
                }
                const std::string text = cellText(cell);
                if (text.empty())
                    result += ' ';
                else
                    result += text;
            }
            if (row < end.row)
                result += '\n';
        }
        return result;
    }

    std::string TerminalWidget::getAllText() const {
        std::lock_guard lock(mutex_);

        std::string result;
        result.reserve(scrollback_.size() * cols_ + rows_ * cols_);

        for (auto it = scrollback_.rbegin(); it != scrollback_.rend(); ++it) {
            const auto& line = *it;
            std::string row_text;
            for (size_t col = 0; col < line.cells.size(); ++col) {
                const auto& cell = line.cells[col];
                if (cell.chars[0]) {
                    char utf8[4];
                    const size_t len = encodeUtf8(cell.chars[0], utf8);
                    if (len > 0)
                        row_text.append(utf8, len);
                } else {
                    row_text += ' ';
                }
            }
            while (!row_text.empty() && row_text.back() == ' ')
                row_text.pop_back();
            result += row_text;
            result += '\n';
        }

        for (int row = 0; row < rows_; ++row) {
            std::string row_text;
            for (int col = 0; col < cols_; ++col) {
                VTermScreenCell cell;
                vterm_screen_get_cell(screen_, {row, col}, &cell);
                if (cell.chars[0]) {
                    char utf8[4];
                    const size_t len = encodeUtf8(cell.chars[0], utf8);
                    if (len > 0)
                        row_text.append(utf8, len);
                } else {
                    row_text += ' ';
                }
            }
            while (!row_text.empty() && row_text.back() == ' ')
                row_text.pop_back();
            result += row_text;
            if (row < rows_ - 1)
                result += '\n';
        }

        while (!result.empty() && (result.back() == '\n' || result.back() == ' '))
            result.pop_back();

        return result;
    }

    void TerminalWidget::paste(const std::string& text) {
        if (pty_.write("\x1b[200~", 6) < 0 ||
            pty_.write(text.c_str(), text.size()) < 0 ||
            pty_.write("\x1b[201~", 6) < 0) {
            LOG_ERROR("PTY paste failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::clear() {
        if (pty_.is_running()) {
            if (pty_.write("\x0c", 1) < 0) {
                LOG_ERROR("PTY clear failed");
            }
        } else {
            reset();
        }
    }

    void TerminalWidget::write(const char* data, size_t len) {
        std::lock_guard lock(mutex_);

        if (read_only_.load()) {
            const char* p = data;
            const char* const end = data + len;
            while (p < end) {
                const char* nl = static_cast<const char*>(std::memchr(p, '\n', end - p));
                if (!nl) {
                    vterm_input_write(vt_, p, end - p);
                    break;
                }
                if (nl > p)
                    vterm_input_write(vt_, p, nl - p);
                vterm_input_write(vt_, "\r\n", 2);
                p = nl + 1;
            }
        } else {
            vterm_input_write(vt_, data, len);
        }

        has_new_output_.store(true);
        markDirty();
    }

    void TerminalWidget::reset() {
        std::lock_guard lock(mutex_);
        vterm_screen_reset(screen_, 1);
        scrollback_.clear();
        scroll_offset_ = 0;
        cursor_pos_ = {0, 0};
        markDirty();
    }

    void TerminalWidget::sendToPty(const std::string& text) {
        if (!pty_.is_running())
            return;
        if (pty_.write(text.c_str(), text.size()) < 0 ||
            pty_.write("\n", 1) < 0) {
            LOG_ERROR("PTY send failed");
            return;
        }
        scrollToBottom();
    }

    void TerminalWidget::interrupt() {
        if (pty_.is_running()) {
            pty_.interrupt();
        }
    }

    int TerminalWidget::onDamage(VTermRect, void* user) {
        static_cast<TerminalWidget*>(user)->markDirty();
        return 0;
    }

    int TerminalWidget::onMoveCursor(VTermPos pos, VTermPos, int visible, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->cursor_pos_ = pos;
        self->cursor_visible_ = visible != 0;
        self->markDirty();
        return 0;
    }

    int TerminalWidget::onBell(void*) {
        return 0;
    }

    int TerminalWidget::onResize(int rows, int cols, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->rows_ = rows;
        self->cols_ = cols;
        self->markDirty();
        return 0;
    }

    int TerminalWidget::onPushline(int cols, const VTermScreenCell* cells, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        self->scrollback_.push_front({std::vector<VTermScreenCell>(cells, cells + cols)});
        while (self->scrollback_.size() > MAX_SCROLLBACK) {
            self->scrollback_.pop_back();
        }
        return 0;
    }

    int TerminalWidget::onPopline(int cols, VTermScreenCell* cells, void* user) {
        auto* self = static_cast<TerminalWidget*>(user);
        if (self->scrollback_.empty())
            return 0;

        const auto& line = self->scrollback_.front();
        const int copy_cols = std::min(cols, static_cast<int>(line.cells.size()));
        std::memcpy(cells, line.cells.data(), copy_cols * sizeof(VTermScreenCell));

        for (int i = copy_cols; i < cols; ++i) {
            std::memset(&cells[i], 0, sizeof(VTermScreenCell));
        }

        self->scrollback_.pop_front();
        return 1;
    }

} // namespace lfs::vis::terminal
