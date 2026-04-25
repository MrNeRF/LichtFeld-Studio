/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Rml {
    class Element;
    class Event;
} // namespace Rml

namespace lfs::vis {
    struct Theme;
}

namespace lfs::vis::editor {

    struct PythonEditorSymbol {
        std::string label;
        std::string detail;
        std::size_t byte_offset = 0;
        std::size_t line = 0;
        int depth = 0;
    };

    struct PythonEditorFold {
        std::string label;
        std::string detail;
        std::size_t byte_offset = 0;
        std::size_t line = 0;
        std::size_t end_line = 0;
        bool collapsed = false;
    };

    class PythonEditor {
    public:
        PythonEditor();
        ~PythonEditor();

        PythonEditor(const PythonEditor&) = delete;
        PythonEditor& operator=(const PythonEditor&) = delete;

        // Render the editor inside a RmlUi custom element. Returns true if execution was requested this frame.
        bool renderRml(Rml::Element& element, float width, float height, float font_size_px = 0.0f);
        void processRmlEvent(Rml::Element& element, Rml::Event& event);

        std::string getText() const;
        std::string getTextStripped() const;
        void setText(const std::string& text);
        void clear();

        bool shouldExecute() const { return execute_requested_; }
        bool consumeExecuteRequested();
        bool consumeTextChanged();
        [[nodiscard]] bool hasSyntaxErrors() const;
        [[nodiscard]] bool syntaxDiagnosticsAvailable() const;
        [[nodiscard]] std::string syntaxSummary() const;
        [[nodiscard]] std::string syntaxStructureSummary() const;
        [[nodiscard]] std::vector<PythonEditorSymbol> syntaxSymbols() const;
        [[nodiscard]] std::vector<PythonEditorSymbol> syntaxBreadcrumbs() const;
        [[nodiscard]] std::vector<PythonEditorFold> syntaxFolds() const;
        [[nodiscard]] bool syntaxStructureCurrent() const;
        [[nodiscard]] std::size_t syntaxFoldCount() const;
        [[nodiscard]] std::string currentSyntaxScope() const;
        void refreshSyntaxDiagnostics();
        bool selectEnclosingSyntaxBlock();
        bool expandSyntaxSelection();
        bool selectCurrentSyntaxFold();
        bool toggleCurrentSyntaxFold();
        bool foldAllSyntaxBlocks();
        bool unfoldAllSyntaxBlocks();
        bool jumpToParentSyntaxBlock();
        bool jumpToChildSyntaxBlock();
        bool jumpToSyntaxSymbol(std::size_t index);
        bool jumpToSyntaxBreadcrumb(std::size_t index);
        bool jumpToSyntaxFold(std::size_t index);
        bool toggleSyntaxFold(std::size_t index);

        void updateTheme(const Theme& theme);

        void addToHistory(const std::string& cmd);
        void historyUp();
        void historyDown();

        void focus();
        void unfocus();
        bool isFocused() const;
        bool hasActiveCompletion() const;
        bool needsRmlFrame() const;
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
