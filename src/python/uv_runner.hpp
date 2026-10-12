/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "python_runtime_api.hpp"

#include "subprocess.hpp"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace lfs::python {

    class LFS_PYTHON_RUNTIME_API UvRunner {
    public:
        using OutputCallback = std::function<void(const std::string&, bool is_stderr, bool is_line_update)>;
        using RawOutputCallback = std::function<void(const std::string&)>;
        using CompletionCallback = std::function<void(bool success, int exit_code)>;

        UvRunner() = default;
        ~UvRunner();

        UvRunner(const UvRunner&) = delete;
        UvRunner& operator=(const UvRunner&) = delete;

        // Start UV command with given arguments (non-blocking)
        // Returns false if UV not found or failed to start
        bool start(const std::vector<std::string>& args);

        // Poll for output (call from render loop, non-blocking)
        // Returns true if still running
        bool poll();

        // Cancel running operation
        void cancel();

        // State queries
        [[nodiscard]] bool is_running() const { return running_.load(); }
        [[nodiscard]] bool is_complete() const { return complete_.load(); }
        [[nodiscard]] int exit_code() const { return exit_code_; }

        void set_output_callback(OutputCallback cb) { output_callback_ = std::move(cb); }
        void set_raw_output_callback(RawOutputCallback cb) { raw_output_callback_ = std::move(cb); }
        void set_completion_callback(CompletionCallback cb) { completion_callback_ = std::move(cb); }

    private:
        void process_buffer();
        void emit_line(const std::string& line, bool is_line_update);

        SubProcess process_;
        std::string output_buffer_;
        OutputCallback output_callback_;
        RawOutputCallback raw_output_callback_;
        CompletionCallback completion_callback_;
        std::atomic<bool> running_{false};
        std::atomic<bool> complete_{false};
        int exit_code_ = -1;
    };

} // namespace lfs::python
