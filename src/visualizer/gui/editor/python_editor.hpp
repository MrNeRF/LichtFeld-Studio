/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <imgui.h>

namespace lfs::vis {
    struct Theme;
}

namespace lfs::vis::editor {

    class PythonEditor {
    public:
        PythonEditor();
        ~PythonEditor();

        PythonEditor(const PythonEditor&) = delete;
        PythonEditor& operator=(const PythonEditor&) = delete;

        // Render the editor. Returns true if execution was requested this frame.
        bool render(const ImVec2& size);

        std::string getText() const;
        std::string getTextStripped() const;
        void setText(const std::string& text);
        void clear();

        bool shouldExecute() const { return execute_requested_; }
        bool consumeTextChanged();

        void updateTheme(const Theme& theme);

        void addToHistory(const std::string& cmd);
        void historyUp();
        void historyDown();

        void focus();
        void unfocus();
        bool isFocused() const;
        bool hasActiveCompletion() const;
        void setVimModeEnabled(bool enabled);
        bool isVimModeEnabled() const;

        void setReadOnly(bool readonly);
        bool isReadOnly() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        bool execute_requested_ = false;
        std::vector<std::string> history_;
        int history_index_ = -1;
        std::string current_input_;
    };

} // namespace lfs::vis::editor
