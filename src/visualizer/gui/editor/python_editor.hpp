/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/editor/document_kind.hpp"
#include "shaders/shader_service.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Rml {
    class Element;
    class Event;
} // namespace Rml

namespace lfs::vis::editor {

    struct EditorDocument {
        core::Uuid id;
        DocumentKind kind = DocumentKind::Python;
        std::uint64_t source_revision = 1;
        std::optional<core::Uuid> shader_asset;
        std::uint64_t asset_revision = 0;
        shaders::ShaderSlot target;
        bool operator==(const EditorDocument&) const = default;
    };

    struct EditorDiagnosticRange {
        std::size_t start = 0;
        std::size_t end = 0;
    };

    struct EditorShaderUpdate {
        core::Uuid document;
        shaders::ShaderStatus status;
        bool diagnostics_changed = false;
    };

    struct EditorDocumentState {
        EditorDocument document;
        std::string text, locator;
        bool modified = false;
        std::vector<EditorDiagnosticRange> markers;
        std::vector<shaders::Diagnostic> diagnostics;
        std::vector<shaders::ShaderSlot> owned_bindings;
        std::optional<shaders::ShaderStatus> shader_status;
        bool applied = false;
    };

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

    struct PythonEditorSessionFold {
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
        std::size_t start_line = 0;
        std::size_t end_line = 0;
        std::string kind;
        bool collapsed = false;

        friend bool operator==(
            const PythonEditorSessionFold&,
            const PythonEditorSessionFold&) = default;
    };

    struct PythonEditorSessionState {
        std::size_t cursor_byte = 0;
        std::optional<std::size_t> selection_anchor_byte;
        float scroll_x = 0.0f;
        float scroll_y = 0.0f;
        std::vector<PythonEditorSessionFold> folds;

        friend bool operator==(
            const PythonEditorSessionState&,
            const PythonEditorSessionState&) = default;
    };

    struct PythonEditorSessionFile {
        std::string locator;
        std::string text;
        bool modified = false;
        PythonEditorSessionState editor;
        EditorDocument document;

        friend bool operator==(
            const PythonEditorSessionFile&,
            const PythonEditorSessionFile&) = default;
    };

    struct PythonEditorWorkspaceSessionState {
        std::vector<PythonEditorSessionFile> open_files;
        std::optional<std::string> active_file;
        bool vim_mode = false;

        friend bool operator==(
            const PythonEditorWorkspaceSessionState&,
            const PythonEditorWorkspaceSessionState&) = default;
    };

    class LFS_VIS_API PythonEditor {
    public:
        PythonEditor();
        ~PythonEditor();

        PythonEditor(const PythonEditor&) = delete;
        PythonEditor& operator=(const PythonEditor&) = delete;

        // Render the editor inside a RmlUi custom element.
        void renderRml(Rml::Element& element, float width, float height, float font_size_px = 0.0f);
        void processRmlEvent(Rml::Element& element, Rml::Event& event);

        std::string getText() const;
        std::string getTextStripped(std::optional<core::Uuid> document = {}) const;
        void setText(const std::string& text);
        void clear();
        void setActiveSessionLocator(std::string locator);
        void clearActiveSessionLocator();

        [[nodiscard]] EditorDocument document() const;
        [[nodiscard]] std::vector<EditorDocument> documents() const;
        [[nodiscard]] Result<EditorDocumentState>
        inspectDocument(const core::Uuid& id, const shaders::ShaderService* service = nullptr) const;
        [[nodiscard]] Status setDocumentText(const core::Uuid& id, const std::string& text);
        [[nodiscard]] Status setDocumentMode(const core::Uuid& id, DocumentKind kind,
                                             const std::optional<shaders::ShaderAsset>& asset = {});
        [[nodiscard]] Status setDocumentTarget(const core::Uuid& id, const shaders::ShaderSlot& target,
                                               const shaders::ShaderService& service);
        [[nodiscard]] std::string activeSessionLocator() const;
        void setDocumentKind(DocumentKind kind);
        core::Uuid newDocument(const std::string& text, DocumentKind kind = DocumentKind::Python,
                               std::string locator = {});
        bool activateDocument(const core::Uuid& id);
        void setShaderTarget(const shaders::ShaderSlot& target);
        void openShaderAsset(const shaders::ShaderAsset& asset);
        [[nodiscard]] Status removeShaderAsset(shaders::ShaderService& service);
        [[nodiscard]] Status runShader(shaders::ShaderService& service, std::optional<core::Uuid> document = {},
                                       const std::optional<std::string>& source = {});
        std::vector<shaders::ShaderSlot> stopShader(shaders::ShaderService& service, std::optional<core::Uuid> document = {});
        void setShaderChanges(std::shared_ptr<shaders::ShaderChanges> changes);
        [[nodiscard]] bool hasShaderBinding(const shaders::ShaderService& service) const;
        [[nodiscard]] std::vector<EditorShaderUpdate> pollShaders(shaders::ShaderService& service);
        bool applyDiagnostics(const core::Uuid& document, std::uint64_t source_revision,
                              std::span<const shaders::Diagnostic> diagnostics);
        [[nodiscard]] std::vector<EditorDiagnosticRange> diagnosticRanges(const core::Uuid& document) const;
        [[nodiscard]] bool pythonAnalysisActive() const;

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
        [[nodiscard]] PythonEditorWorkspaceSessionState
        captureWorkspaceSessionState(
            std::string_view active_locator,
            bool active_modified) const;
        void restoreWorkspaceSessionState(
            const PythonEditorWorkspaceSessionState& state);

        void setReadOnly(bool readonly);
        bool isModified() const;
        void setModified(bool modified);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::editor
