/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_buffer_analysis.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <limits>
#include <memory>

#include <tree_sitter/api.h>
#include <tree_sitter/tree-sitter-python.h>

namespace lfs::python {
    namespace {
        constexpr std::size_t MAX_ISSUES = 8;

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

        using ParserPtr = std::unique_ptr<TSParser, ParserDeleter>;
        using TreePtr = std::unique_ptr<TSTree, TreeDeleter>;

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
    } // namespace

    PythonBufferAnalysis analyze_python_buffer(std::string_view code) {
        PythonBufferAnalysis analysis;

        if (code.empty() || is_blank(code)) {
            analysis.status = PythonBufferStatus::Empty;
            analysis.summary = "Python buffer is empty";
            return analysis;
        }

        ParserPtr parser(ts_parser_new());
        if (!parser) {
            analysis.summary = "Failed to create Python syntax parser";
            return analysis;
        }

        if (!ts_parser_set_language(parser.get(), tree_sitter_python())) {
            analysis.summary = "Failed to initialize Python syntax parser";
            return analysis;
        }

        if (code.size() > std::numeric_limits<uint32_t>::max()) {
            analysis.summary = "Python buffer is too large to parse";
            return analysis;
        }

        TreePtr tree(ts_parser_parse_string(
            parser.get(),
            nullptr,
            code.data(),
            static_cast<uint32_t>(code.size())));
        if (!tree) {
            analysis.summary = "Failed to parse Python buffer";
            return analysis;
        }

        const TSNode root = ts_tree_root_node(tree.get());
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

} // namespace lfs::python
