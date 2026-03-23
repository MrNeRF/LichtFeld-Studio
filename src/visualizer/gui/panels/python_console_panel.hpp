/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "gui/ui_context.hpp"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <imgui.h>

namespace lfs::vis::editor {
    class PythonEditor;
} // namespace lfs::vis::editor

namespace lfs::vis::terminal {
    class TerminalWidget;
} // namespace lfs::vis::terminal

namespace lfs::vis::gui::panels {

    class LFS_VIS_API PythonConsoleState {
    public:
        static PythonConsoleState& getInstance();

        void addOutput(const std::string& text, uint32_t color = 0xFFFFFFFF);
        void addError(const std::string& text);
        void addInput(const std::string& text);
        void addInfo(const std::string& text);
        void clear();

        void addToHistory(const std::string& cmd);
        void resetHistoryIndex() { history_index_ = -1; }
        void historyUp();
        void historyDown();
        int historyIndex() const { return history_index_; }

        terminal::TerminalWidget* getTerminal();
        terminal::TerminalWidget* getOutputTerminal();
        editor::PythonEditor* getEditor();
        void setEditorText(const std::string& text);
        void focusEditor();
        [[nodiscard]] std::string getEditorText() const;
        [[nodiscard]] std::string getEditorTextStripped() const;
        [[nodiscard]] std::string getOutputText() const;

        // Tab selection (0 = Output, 1 = Terminal)
        int getActiveTab() const { return active_tab_; }
        void setActiveTab(int tab) { active_tab_ = tab; }

        // Focus tracking for input routing
        bool isTerminalFocused() const { return terminal_focused_; }
        void setTerminalFocused(bool focused) { terminal_focused_ = focused; }

        // Script file management
        void setScriptPath(const std::filesystem::path& path) { script_path_ = path; }
        const std::filesystem::path& getScriptPath() const { return script_path_; }
        void setModified(bool modified) { is_modified_ = modified; }
        bool isModified() const { return is_modified_; }

        // Font scaling (steps match loaded monospace font sizes)
        float getFontScale() const { return font_scale_; }
        void setFontScale(float scale) { font_scale_ = std::clamp(scale, FONT_STEPS[0], FONT_STEPS[FONT_STEP_COUNT - 1]); }
        void increaseFontScale();
        void decreaseFontScale();
        void resetFontScale() { font_scale_ = 1.0f; }

        // Script execution
        bool isScriptRunning() const { return script_running_.load(); }
        void interruptScript();
        void runScriptAsync(const std::string& code);

    private:
        PythonConsoleState();
        ~PythonConsoleState();

        std::vector<std::string> command_history_;
        int history_index_ = -1;
        mutable std::mutex mutex_;
        static constexpr size_t MAX_MESSAGES = 1000;

        std::unique_ptr<terminal::TerminalWidget> terminal_;
        std::unique_ptr<terminal::TerminalWidget> output_terminal_;
        std::unique_ptr<editor::PythonEditor> editor_;
        int active_tab_ = 0;
        bool terminal_focused_ = false;

        // Script file tracking
        std::filesystem::path script_path_;
        bool is_modified_ = false;

        // Font scaling steps must match gui_manager MONO_SCALES
        static constexpr int FONT_STEP_COUNT = 5;
        static constexpr float FONT_STEPS[FONT_STEP_COUNT] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
        float font_scale_ = 1.0f;

        // Script execution
        std::atomic<bool> script_running_{false};
        std::atomic<unsigned long> script_thread_id_{0};
        std::thread script_thread_;
    };

    // Draw the Python console window (floating)
    void DrawPythonConsole(const UIContext& ctx, bool* open);

    // Draw the Python console as a docked panel (fixed position/size)
    void DrawDockedPythonConsole(const UIContext& ctx, float x, float y, float w, float h);

} // namespace lfs::vis::gui::panels
