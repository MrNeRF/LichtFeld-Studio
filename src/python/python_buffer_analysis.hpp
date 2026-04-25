/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::python {

    struct PythonBufferPoint {
        std::uint32_t row = 0;
        std::uint32_t column = 0;
    };

    struct PythonBufferEdit {
        std::size_t start_byte = 0;
        std::size_t old_end_byte = 0;
        std::size_t new_end_byte = 0;
        PythonBufferPoint start_point;
        PythonBufferPoint old_end_point;
        PythonBufferPoint new_end_point;
    };

    enum class PythonBufferStatus {
        Empty,
        Clean,
        SyntaxError,
        ParserUnavailable,
    };

    struct PythonBufferIssue {
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
        std::size_t line = 0;
        std::size_t column = 0;
        std::size_t end_line = 0;
        std::size_t end_column = 0;
        std::string kind;
        std::string node_type;
        std::string message;
    };

    struct PythonBufferAnalysis {
        PythonBufferStatus status = PythonBufferStatus::ParserUnavailable;
        std::string summary;
        std::vector<PythonBufferIssue> issues;

        [[nodiscard]] bool clean() const {
            return status == PythonBufferStatus::Empty || status == PythonBufferStatus::Clean;
        }
    };

    enum class PythonSymbolKind {
        Function,
        Class,
        Import,
        Variable,
    };

    struct PythonSymbol {
        PythonSymbolKind kind = PythonSymbolKind::Function;
        std::string name;
        std::string detail;
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
        std::size_t line = 0;
        std::size_t column = 0;
        std::size_t end_line = 0;
        std::size_t end_column = 0;
        int depth = 0;
    };

    struct PythonByteRange {
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
    };

    struct PythonFoldRange {
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
        std::size_t line = 0;
        std::size_t end_line = 0;
        std::string kind;
    };

    enum class PythonHighlightKind {
        Keyword,
        Comment,
        String,
        Number,
        Constant,
        Decorator,
        Function,
        Type,
        Property,
    };

    struct PythonSyntaxHighlight {
        PythonHighlightKind kind = PythonHighlightKind::Keyword;
        std::size_t start_byte = 0;
        std::size_t end_byte = 0;
    };

    class PythonSyntaxDocument {
    public:
        PythonSyntaxDocument();
        ~PythonSyntaxDocument();

        PythonSyntaxDocument(const PythonSyntaxDocument&) = delete;
        PythonSyntaxDocument& operator=(const PythonSyntaxDocument&) = delete;
        PythonSyntaxDocument(PythonSyntaxDocument&&) noexcept;
        PythonSyntaxDocument& operator=(PythonSyntaxDocument&&) noexcept;

        bool reset(std::string_view code);
        bool applyEditsAndReparse(std::string_view code, std::span<const PythonBufferEdit> edits);

        [[nodiscard]] const PythonBufferAnalysis& analysis() const;
        [[nodiscard]] const std::vector<PythonSymbol>& symbols() const;
        [[nodiscard]] const std::vector<PythonFoldRange>& foldRanges() const;
        [[nodiscard]] const std::vector<PythonSyntaxHighlight>& highlights() const;
        [[nodiscard]] std::string scopeAt(std::size_t byte_offset) const;
        [[nodiscard]] std::optional<PythonByteRange> enclosingBlockRange(std::size_t byte_offset) const;
        [[nodiscard]] std::vector<PythonByteRange> enclosingBlockRanges(std::size_t byte_offset) const;
        [[nodiscard]] bool hasTree() const;
        [[nodiscard]] bool structureCurrent() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    PythonBufferPoint python_buffer_point_at_byte(std::string_view code, std::size_t byte_offset);
    PythonBufferAnalysis analyze_python_buffer(std::string_view code);

} // namespace lfs::python
