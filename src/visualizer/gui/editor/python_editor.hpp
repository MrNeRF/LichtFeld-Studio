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

        bool consumeExecuteRequested();
        bool consumeTextChanged();
        [[nodiscard]] std::vector<PythonEditorSymbol> syntaxSymbols() const;
        [[nodiscard]] std::vector<PythonEditorSymbol> syntaxBreadcrumbs() const;
        [[nodiscard]] std::vector<PythonEditorFold> syntaxFolds() const;
        void refreshSyntaxDiagnostics();
        bool foldAllSyntaxBlocks();
        bool unfoldAllSyntaxBlocks();
        bool jumpToSyntaxSymbol(std::size_t index);
        bool jumpToSyntaxBreadcrumb(std::size_t index);
        bool toggleSyntaxFold(std::size_t index);

        void focus();
        void unfocus();
        bool isFocused() const;
        bool hasActiveCompletion() const;
        bool needsRmlFrame() const;
        void setVimModeEnabled(bool enabled);
        bool isVimModeEnabled() const;

        void setReadOnly(bool readonly);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        bool execute_requested_ = false;
    };

} // namespace lfs::vis::editor
