/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis::editor {

    class PythonLspClient {
    public:
        struct TextEdit {
            int start_line = 0;
            int start_character = 0;
            int end_line = 0;
            int end_character = 0;
            std::string new_text;
        };

        struct CompletionItem {
            std::string label;
            std::string detail;
            std::string description;
            std::string sort_text;
            std::string filter_text;
            std::string insert_text;
            int kind = 0;
            bool deprecated = false;
            std::optional<TextEdit> text_edit;
            std::vector<TextEdit> additional_text_edits;
        };

        struct CompletionList {
            int document_version = 0;
            int line = 0;
            int character = 0;
            bool is_incomplete = false;
            std::vector<CompletionItem> items;
        };

        struct SemanticToken {
            int line = 0;
            int start_character = 0;
            int length = 0;
            std::string type;
            uint32_t modifiers = 0;
        };

        struct SemanticTokenList {
            int document_version = 0;
            std::vector<SemanticToken> tokens;
        };

        PythonLspClient();
        ~PythonLspClient();

        PythonLspClient(const PythonLspClient&) = delete;
        PythonLspClient& operator=(const PythonLspClient&) = delete;

        int updateDocument(const std::string& text);
        void requestCompletion(int document_version,
                               int line,
                               int character,
                               bool manual,
                               std::string trigger_character = {});
        void requestSemanticTokens(int document_version);
        std::optional<CompletionList> takeLatestCompletion();
        std::optional<SemanticTokenList> takeLatestSemanticTokens();

        [[nodiscard]] bool isReady() const;
        [[nodiscard]] bool isAvailable() const;
        [[nodiscard]] std::string statusText() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::editor
