/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::python {

    enum class PythonBufferStatus {
        Empty,
        Clean,
        SyntaxError,
        ParserUnavailable,
    };

    struct PythonBufferIssue {
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

    PythonBufferAnalysis analyze_python_buffer(std::string_view code);

} // namespace lfs::python
