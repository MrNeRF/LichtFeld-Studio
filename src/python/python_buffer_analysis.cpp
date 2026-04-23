/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_buffer_analysis.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-python.h>

namespace lfs::python {
    namespace {
        constexpr std::size_t MAX_ISSUES = 8;
        constexpr std::string_view SYMBOL_QUERY = R"QUERY(
(function_definition
  name: (identifier) @name) @function
(class_definition
  name: (identifier) @name) @class
(import_statement) @import
(import_from_statement) @import
)QUERY";

        struct ParserDeleter {
            void operator()(TSParser* parser) const {
                if (parser) {
                    ts_parser_delete(parser);
                }
            }
        };

        struct TreeDeleter {
            void operator()(TSTree* tree) const {
                if (tree) {
                    ts_tree_delete(tree);
                }
            }
        };

        struct QueryDeleter {
            void operator()(TSQuery* query) const {
                if (query) {
                    ts_query_delete(query);
                }
            }
        };

        struct QueryCursorDeleter {
            void operator()(TSQueryCursor* cursor) const {
                if (cursor) {
                    ts_query_cursor_delete(cursor);
                }
            }
        };

        using ParserPtr = std::unique_ptr<TSParser, ParserDeleter>;
        using TreePtr = std::unique_ptr<TSTree, TreeDeleter>;
        using QueryPtr = std::unique_ptr<TSQuery, QueryDeleter>;
        using QueryCursorPtr = std::unique_ptr<TSQueryCursor, QueryCursorDeleter>;

        [[nodiscard]] bool fits_tree_sitter_u32(const std::size_t value) {
            return value <= std::numeric_limits<std::uint32_t>::max();
        }

        [[nodiscard]] TSPoint to_ts_point(const PythonBufferPoint point) {
            return TSPoint{.row = point.row, .column = point.column};
        }

        std::string make_issue_message(const PythonBufferIssue& issue) {
            const auto line = issue.line + 1;
            const auto column = issue.column + 1;

            if (issue.kind == "missing") {
                return std::format("Missing Python syntax element '{}' at line {}, column {}",
                                   issue.node_type,
                                   line,
                                   column);
            }

            if (issue.node_type.empty() || issue.node_type == "ERROR") {
                return std::format("Python syntax error at line {}, column {}", line, column);
            }

            return std::format("Python syntax error near '{}' at line {}, column {}",
                               issue.node_type,
                               line,
                               column);
        }

        PythonBufferIssue make_issue(TSNode node, std::string kind) {
            const TSPoint start = ts_node_start_point(node);
            const TSPoint end = ts_node_end_point(node);

            PythonBufferIssue issue;
            issue.start_byte = ts_node_start_byte(node);
            issue.end_byte = ts_node_end_byte(node);
            issue.line = start.row;
            issue.column = start.column;
            issue.end_line = end.row;
            issue.end_column = end.column;
            issue.kind = std::move(kind);
            if (const char* type = ts_node_type(node)) {
                issue.node_type = type;
            }
            issue.message = make_issue_message(issue);
            return issue;
        }

        void collect_issues(TSNode node, std::vector<PythonBufferIssue>& issues) {
            if (issues.size() >= MAX_ISSUES || !ts_node_has_error(node)) {
                return;
            }

            if (ts_node_is_missing(node)) {
                issues.push_back(make_issue(node, "missing"));
                return;
            }

            if (ts_node_is_error(node)) {
                issues.push_back(make_issue(node, "error"));
                return;
            }

            const uint32_t child_count = ts_node_child_count(node);
            for (uint32_t i = 0; i < child_count && issues.size() < MAX_ISSUES; ++i) {
                collect_issues(ts_node_child(node, i), issues);
            }
        }

        bool is_blank(std::string_view code) {
            return std::ranges::all_of(code, [](unsigned char ch) {
                return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v';
            });
        }

        [[nodiscard]] std::string_view trim_ascii(std::string_view text) {
            while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) {
                text.remove_prefix(1);
            }
            while (!text.empty() &&
                   (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '\n')) {
                text.remove_suffix(1);
            }
            return text;
        }

        [[nodiscard]] std::string node_text(TSNode node, std::string_view code) {
            const std::size_t start = std::min<std::size_t>(ts_node_start_byte(node), code.size());
            const std::size_t end = std::min<std::size_t>(ts_node_end_byte(node), code.size());
            if (start >= end) {
                return {};
            }
            return std::string(code.substr(start, end - start));
        }

        [[nodiscard]] std::string single_line_node_text(TSNode node, std::string_view code) {
            std::string text = node_text(node, code);
            if (const auto newline = text.find('\n'); newline != std::string::npos) {
                text.erase(newline);
            }
            return std::string(trim_ascii(text));
        }

        [[nodiscard]] std::string_view node_type(TSNode node) {
            if (const char* type = ts_node_type(node)) {
                return type;
            }
            return {};
        }

        [[nodiscard]] bool is_symbol_container_type(std::string_view type) {
            return type == "function_definition" || type == "class_definition";
        }

        [[nodiscard]] bool is_block_type(std::string_view type) {
            static constexpr std::array BLOCK_TYPES{
                "decorated_definition",
                "function_definition",
                "class_definition",
                "if_statement",
                "elif_clause",
                "else_clause",
                "for_statement",
                "while_statement",
                "with_statement",
                "try_statement",
                "except_clause",
                "finally_clause",
                "match_statement",
                "case_clause",
            };
            return std::ranges::find(BLOCK_TYPES, type) != BLOCK_TYPES.end();
        }

        [[nodiscard]] int structural_depth(TSNode node) {
            int depth = 0;
            for (TSNode parent = ts_node_parent(node); !ts_node_is_null(parent);
                 parent = ts_node_parent(parent)) {
                if (is_symbol_container_type(node_type(parent))) {
                    ++depth;
                }
            }
            return depth;
        }

        [[nodiscard]] PythonSymbol make_symbol(
            const PythonSymbolKind kind,
            TSNode symbol_node,
            const std::optional<TSNode> name_node,
            std::string_view code) {
            const TSPoint start = ts_node_start_point(symbol_node);
            const TSPoint end = ts_node_end_point(symbol_node);

            PythonSymbol symbol;
            symbol.kind = kind;
            symbol.start_byte = ts_node_start_byte(symbol_node);
            symbol.end_byte = ts_node_end_byte(symbol_node);
            symbol.line = start.row;
            symbol.column = start.column;
            symbol.end_line = end.row;
            symbol.end_column = end.column;
            symbol.depth = structural_depth(symbol_node);

            if (name_node.has_value()) {
                symbol.name = node_text(*name_node, code);
            } else {
                symbol.name = single_line_node_text(symbol_node, code);
            }
            symbol.detail = single_line_node_text(symbol_node, code);
            return symbol;
        }

        [[nodiscard]] std::vector<PythonSymbol> extract_symbols(TSTree* tree, std::string_view code) {
            if (tree == nullptr || code.empty()) {
                return {};
            }

            std::uint32_t query_error_offset = 0;
            TSQueryError query_error = TSQueryErrorNone;
            QueryPtr query(ts_query_new(tree_sitter_python(),
                                        SYMBOL_QUERY.data(),
                                        static_cast<std::uint32_t>(SYMBOL_QUERY.size()),
                                        &query_error_offset,
                                        &query_error));
            if (!query) {
                return {};
            }

            QueryCursorPtr cursor(ts_query_cursor_new());
            if (!cursor) {
                return {};
            }

            std::vector<PythonSymbol> symbols;
            ts_query_cursor_exec(cursor.get(), query.get(), ts_tree_root_node(tree));
            ts_query_cursor_set_byte_range(cursor.get(), 0, static_cast<std::uint32_t>(code.size()));

            TSQueryMatch match;
            while (ts_query_cursor_next_match(cursor.get(), &match)) {
                std::optional<TSNode> symbol_node;
                std::optional<TSNode> name_node;
                std::optional<PythonSymbolKind> kind;

                for (std::uint32_t i = 0; i < match.capture_count; ++i) {
                    const TSQueryCapture capture = match.captures[i];
                    std::uint32_t capture_name_length = 0;
                    const char* capture_name =
                        ts_query_capture_name_for_id(query.get(), capture.index, &capture_name_length);
                    const std::string_view name(capture_name, capture_name_length);

                    if (name == "function") {
                        symbol_node = capture.node;
                        kind = PythonSymbolKind::Function;
                    } else if (name == "class") {
                        symbol_node = capture.node;
                        kind = PythonSymbolKind::Class;
                    } else if (name == "import") {
                        symbol_node = capture.node;
                        kind = PythonSymbolKind::Import;
                    } else if (name == "name") {
                        name_node = capture.node;
                    }
                }

                if (symbol_node.has_value() && kind.has_value()) {
                    symbols.push_back(make_symbol(*kind, *symbol_node, name_node, code));
                }
            }

            std::ranges::sort(symbols, [](const PythonSymbol& lhs, const PythonSymbol& rhs) {
                if (lhs.start_byte == rhs.start_byte) {
                    return lhs.end_byte > rhs.end_byte;
                }
                return lhs.start_byte < rhs.start_byte;
            });
            return symbols;
        }

        [[nodiscard]] PythonBufferAnalysis analyze_tree(std::string_view code, TSTree* tree) {
            PythonBufferAnalysis analysis;

            if (code.empty() || is_blank(code)) {
                analysis.status = PythonBufferStatus::Empty;
                analysis.summary = "Python buffer is empty";
                return analysis;
            }

            if (tree == nullptr) {
                analysis.summary = "Failed to parse Python buffer";
                return analysis;
            }

            const TSNode root = ts_tree_root_node(tree);
            if (!ts_node_has_error(root)) {
                analysis.status = PythonBufferStatus::Clean;
                analysis.summary = "Python buffer is syntactically clean";
                return analysis;
            }

            analysis.status = PythonBufferStatus::SyntaxError;
            collect_issues(root, analysis.issues);
            if (analysis.issues.empty()) {
                analysis.issues.push_back(make_issue(root, "error"));
            }
            analysis.summary = analysis.issues.front().message;
            return analysis;
        }

        [[nodiscard]] std::string join_scope_parts(const std::vector<std::string>& parts) {
            std::string scope;
            for (const auto& part : parts) {
                if (part.empty()) {
                    continue;
                }
                if (!scope.empty()) {
                    scope += ".";
                }
                scope += part;
            }
            return scope;
        }
    } // namespace

    PythonBufferPoint python_buffer_point_at_byte(std::string_view code, std::size_t byte_offset) {
        byte_offset = std::min(byte_offset, code.size());

        std::size_t row = 0;
        std::size_t line_start = 0;
        for (std::size_t index = 0; index < byte_offset; ++index) {
            if (code[index] == '\n') {
                ++row;
                line_start = index + 1;
            }
        }

        return {
            .row = static_cast<std::uint32_t>(
                std::min(row, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))),
            .column = static_cast<std::uint32_t>(std::min(
                byte_offset - line_start,
                static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))),
        };
    }

    struct PythonSyntaxDocument::Impl {
        ParserPtr parser;
        TreePtr tree;
        PythonBufferAnalysis current_analysis;
        std::vector<PythonSymbol> current_symbols;
        std::size_t code_size = 0;

        [[nodiscard]] bool ensure_parser() {
            if (parser != nullptr) {
                return true;
            }

            parser.reset(ts_parser_new());
            if (!parser) {
                current_analysis = {};
                current_analysis.summary = "Failed to create Python syntax parser";
                return false;
            }

            if (!ts_parser_set_language(parser.get(), tree_sitter_python())) {
                current_analysis = {};
                current_analysis.summary = "Failed to initialize Python syntax parser";
                parser.reset();
                return false;
            }

            return true;
        }

        void refresh_analysis(std::string_view code) {
            code_size = code.size();
            current_analysis = analyze_tree(code, tree.get());
            current_symbols = current_analysis.clean() ? extract_symbols(tree.get(), code)
                                                       : std::vector<PythonSymbol>{};
        }

        [[nodiscard]] bool reset(std::string_view code) {
            tree.reset();
            current_symbols.clear();
            code_size = code.size();

            if (code.empty() || is_blank(code)) {
                current_analysis = analyze_tree(code, nullptr);
                return true;
            }

            if (!ensure_parser()) {
                return false;
            }

            if (!fits_tree_sitter_u32(code.size())) {
                current_analysis = {};
                current_analysis.summary = "Python buffer is too large to parse";
                return false;
            }

            tree.reset(ts_parser_parse_string(
                parser.get(), nullptr, code.data(), static_cast<std::uint32_t>(code.size())));
            if (!tree) {
                current_analysis = {};
                current_analysis.summary = "Failed to parse Python buffer";
                return false;
            }

            refresh_analysis(code);
            return current_analysis.status != PythonBufferStatus::ParserUnavailable;
        }

        [[nodiscard]] bool apply_edits_and_reparse(
            std::string_view code,
            std::span<const PythonBufferEdit> edits) {
            if (code.empty() || is_blank(code) || tree == nullptr || edits.empty()) {
                return reset(code);
            }

            if (!ensure_parser()) {
                return false;
            }

            if (!fits_tree_sitter_u32(code.size())) {
                current_analysis = {};
                current_analysis.summary = "Python buffer is too large to parse";
                return false;
            }

            for (const auto& edit : edits) {
                if (!fits_tree_sitter_u32(edit.start_byte) ||
                    !fits_tree_sitter_u32(edit.old_end_byte) ||
                    !fits_tree_sitter_u32(edit.new_end_byte)) {
                    return reset(code);
                }

                const TSInputEdit ts_edit{
                    .start_byte = static_cast<std::uint32_t>(edit.start_byte),
                    .old_end_byte = static_cast<std::uint32_t>(edit.old_end_byte),
                    .new_end_byte = static_cast<std::uint32_t>(edit.new_end_byte),
                    .start_point = to_ts_point(edit.start_point),
                    .old_end_point = to_ts_point(edit.old_end_point),
                    .new_end_point = to_ts_point(edit.new_end_point),
                };
                ts_tree_edit(tree.get(), &ts_edit);
            }

            TreePtr new_tree(ts_parser_parse_string(
                parser.get(), tree.get(), code.data(), static_cast<std::uint32_t>(code.size())));
            if (!new_tree) {
                return reset(code);
            }

            tree = std::move(new_tree);
            refresh_analysis(code);
            return current_analysis.status != PythonBufferStatus::ParserUnavailable;
        }
    };

    PythonSyntaxDocument::PythonSyntaxDocument()
        : impl_(std::make_unique<Impl>()) {}

    PythonSyntaxDocument::~PythonSyntaxDocument() = default;
    PythonSyntaxDocument::PythonSyntaxDocument(PythonSyntaxDocument&&) noexcept = default;
    PythonSyntaxDocument& PythonSyntaxDocument::operator=(PythonSyntaxDocument&&) noexcept = default;

    bool PythonSyntaxDocument::reset(std::string_view code) {
        return impl_->reset(code);
    }

    bool PythonSyntaxDocument::applyEditsAndReparse(
        std::string_view code,
        std::span<const PythonBufferEdit> edits) {
        return impl_->apply_edits_and_reparse(code, edits);
    }

    const PythonBufferAnalysis& PythonSyntaxDocument::analysis() const {
        return impl_->current_analysis;
    }

    const std::vector<PythonSymbol>& PythonSyntaxDocument::symbols() const {
        return impl_->current_symbols;
    }

    std::string PythonSyntaxDocument::scopeAt(const std::size_t byte_offset) const {
        std::vector<std::string> scope_parts;
        for (const auto& symbol : impl_->current_symbols) {
            if (symbol.kind == PythonSymbolKind::Import || symbol.start_byte > byte_offset ||
                byte_offset > symbol.end_byte) {
                continue;
            }
            scope_parts.push_back(symbol.name);
        }
        return join_scope_parts(scope_parts);
    }

    std::optional<PythonByteRange> PythonSyntaxDocument::enclosingBlockRange(
        const std::size_t byte_offset) const {
        if (impl_->tree == nullptr || impl_->code_size == 0) {
            return std::nullopt;
        }

        const std::size_t query_byte = std::min(byte_offset, impl_->code_size - 1);
        TSNode node = ts_node_descendant_for_byte_range(
            ts_tree_root_node(impl_->tree.get()),
            static_cast<std::uint32_t>(query_byte),
            static_cast<std::uint32_t>(query_byte));

        for (; !ts_node_is_null(node); node = ts_node_parent(node)) {
            TSNode selected = node;
            const std::string_view type = node_type(node);
            if (is_symbol_container_type(type)) {
                const TSNode parent = ts_node_parent(node);
                if (!ts_node_is_null(parent) && node_type(parent) == "decorated_definition") {
                    selected = parent;
                }
            } else if (!is_block_type(type)) {
                continue;
            }

            const std::size_t start = ts_node_start_byte(selected);
            const std::size_t end = ts_node_end_byte(selected);
            if (start < end) {
                return PythonByteRange{.start_byte = start, .end_byte = end};
            }
        }

        return std::nullopt;
    }

    bool PythonSyntaxDocument::hasTree() const {
        return impl_->tree != nullptr;
    }

    PythonBufferAnalysis analyze_python_buffer(std::string_view code) {
        PythonSyntaxDocument document;
        document.reset(code);
        return document.analysis();
    }

} // namespace lfs::python
