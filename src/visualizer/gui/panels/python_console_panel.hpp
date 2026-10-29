/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "gui/ui_context.hpp"
#include "input/input_bindings.hpp"
#include "shaders/shader_service.hpp"
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lfs::vis {
    struct FrameInputEvent;
}

namespace lfs::vis::editor {
    class PythonEditor;
} // namespace lfs::vis::editor

namespace lfs::vis::terminal {
    class TerminalWidget;
} // namespace lfs::vis::terminal

namespace lfs::vis::gui {
    struct PanelInputState;
} // namespace lfs::vis::gui

namespace lfs::vis::gui::panels {

    class LFS_VIS_API PythonConsoleState {
    public:
        static PythonConsoleState& getInstance();
        static PythonConsoleState* tryGetInstance();

        void addError(const std::string& text);
        void addInfo(const std::string& text);
        void clear();
        bool handleShortcut(const FrameInputEvent& event, const input::InputBindings* bindings = nullptr,
                            shaders::ShaderService* service = nullptr);
        bool documentCommand(input::Action action, shaders::ShaderService* service = nullptr, bool in_flight_only = false);
        bool formatDocument(bool cleanup = false);
        bool loadDocument(const std::filesystem::path& path);
        bool saveDocument(const std::filesystem::path& path);

        void addToHistory(const std::string& cmd);

        terminal::TerminalWidget* getTerminal();
        terminal::TerminalWidget* getOutputTerminal();
        void setShaderChanges(std::shared_ptr<shaders::ShaderChanges> changes);
        void notifyShaderObservers() const {
            if (shader_changes_)
                shader_changes_->notify();
        }
        editor::PythonEditor* getEditor();
        const editor::PythonEditor* getEditor() const;
        void setEditorText(const std::string& text);
        void focusEditor();
        [[nodiscard]] std::string getEditorText() const;
        [[nodiscard]] std::string getEditorTextStripped() const;
        [[nodiscard]] std::string getOutputText() const;

        void setVisible(bool visible) { visible_.store(visible); }
        bool outputVisible() const { return visible_.load() && active_tab_.load() == 0; }

        // Tab selection (0 = Output, 1 = Terminal)
        int getActiveTab() const { return active_tab_; }
        void setActiveTab(int tab) { active_tab_ = tab; }

        // Focus tracking for input routing
        bool isTerminalFocused() const { return terminal_focused_; }
        void setTerminalFocused(bool focused) { terminal_focused_ = focused; }

        // Script file management
        void setScriptPath(const std::filesystem::path& path);
        [[nodiscard]] std::filesystem::path getScriptPath() const;
        void setModified(bool modified);
        bool isModified() const;

        // Font scaling (steps match loaded monospace font sizes)
        float getFontScale() const { return font_scale_; }
        void increaseFontScale();
        void decreaseFontScale();
        void resetFontScale() { font_scale_ = 1.0f; }
        void setFontScale(float scale);

        [[nodiscard]] static float splitterRatio();
        static void setSplitterRatio(float ratio);

        // Script execution
        bool isScriptRunning() const { return script_running_.load(); }
        void interruptScript();
        void runScriptAsync(const std::string& code, std::optional<core::Uuid> document = {});
        [[nodiscard]] std::uint64_t scriptRun() const { return script_run_; }
        // The document the latest run started from; a project switch may close it while the run goes on.
        [[nodiscard]] const std::optional<core::Uuid>& scriptDocument() const { return script_document_; }

    private:
        std::shared_ptr<shaders::ShaderChanges> shader_changes_;
        PythonConsoleState();
        ~PythonConsoleState();

        std::vector<std::string> command_history_;
        int history_index_ = -1;
        mutable std::mutex mutex_;

        std::unique_ptr<terminal::TerminalWidget> terminal_;
        std::unique_ptr<terminal::TerminalWidget> output_terminal_;
        std::unique_ptr<editor::PythonEditor> editor_;
        std::atomic<int> active_tab_{0};
        std::atomic<bool> visible_{false};
        bool terminal_focused_ = false;

        // Font scaling steps must match gui_manager MONO_SCALES
        static constexpr int FONT_STEP_COUNT = 5;
        static constexpr float FONT_STEPS[FONT_STEP_COUNT] = {0.7f, 1.0f, 1.3f, 1.7f, 2.2f};
        float font_scale_ = 1.0f;

        // Script execution
        std::atomic<bool> script_running_{false};
        std::atomic<unsigned long> script_thread_id_{0};
        std::thread script_thread_;
        std::uint64_t script_run_ = 0;
        std::optional<core::Uuid> script_document_;
    };

    // Draw the Python console as a docked panel (fixed position/size)
    void DrawDockedPythonConsole(const UIContext& ctx, float x, float y, float w, float h,
                                 const PanelInputState* input = nullptr);

    void ShutdownPythonConsoleRml();

} // namespace lfs::vis::gui::panels
