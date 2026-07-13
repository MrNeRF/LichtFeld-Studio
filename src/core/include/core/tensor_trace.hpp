/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "core/cuda_safe_format.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::core::debug {

    class LFS_CORE_API TensorOpTracer {
    public:
        struct OpRecord {
            std::string op_name;
            std::string tensor_name;
            std::string input_shapes;
            std::string output_shape;
            std::string file;
            int line = 0;
            float duration_ms = 0.0f;
            int depth = 0;
        };

        static TensorOpTracer& instance();

        void set_enabled(const bool enabled) { enabled_ = enabled; }
        [[nodiscard]] bool is_enabled() const { return enabled_; }

        [[nodiscard]] bool should_trace(const Tensor& t) const {
            return enabled_ || t.is_tracked();
        }

        [[nodiscard]] bool should_trace(const Tensor& t1, const Tensor& t2) const {
            return enabled_ || t1.is_tracked() || t2.is_tracked();
        }

        void push(const char* op, const Tensor& input, const SourceSite& loc) {
            if (!should_trace(input))
                return;
            std::lock_guard lock(mutex_);
            stack_.push_back({op, input.name(), input.shape().str(), "",
                              extract_filename(loc.file_name()), static_cast<int>(loc.line()),
                              0.0f, static_cast<int>(stack_.size())});
            start_times_.push_back(clock::now());
        }

        void push(const char* op, const Tensor& in1, const Tensor& in2, const SourceSite& loc) {
            if (!should_trace(in1, in2))
                return;
            std::lock_guard lock(mutex_);
            const std::string name = in1.is_tracked() ? in1.name() : in2.name();
            const std::string shapes = in1.shape().str() + ", " + in2.shape().str();
            stack_.push_back({op, name, shapes, "",
                              extract_filename(loc.file_name()), static_cast<int>(loc.line()),
                              0.0f, static_cast<int>(stack_.size())});
            start_times_.push_back(clock::now());
        }

        void push(const char* op, const TensorShape& shape, const SourceSite& loc) {
            if (!enabled_)
                return;
            std::lock_guard lock(mutex_);
            stack_.push_back({op, "", shape.str(), "",
                              extract_filename(loc.file_name()), static_cast<int>(loc.line()),
                              0.0f, static_cast<int>(stack_.size())});
            start_times_.push_back(clock::now());
        }

        void push(const char* op, const TensorShape& in1, const TensorShape& in2, const SourceSite& loc) {
            if (!enabled_)
                return;
            std::lock_guard lock(mutex_);
            const std::string shapes = in1.str() + ", " + in2.str();
            stack_.push_back({op, "", shapes, "",
                              extract_filename(loc.file_name()), static_cast<int>(loc.line()),
                              0.0f, static_cast<int>(stack_.size())});
            start_times_.push_back(clock::now());
        }

        void pop(const TensorShape& output_shape) {
            std::lock_guard lock(mutex_);
            if (stack_.empty())
                return;
            finalize_record(output_shape.str());
        }

        void pop() {
            std::lock_guard lock(mutex_);
            if (stack_.empty())
                return;
            finalize_record("");
        }

        void print_stack() const {
            std::lock_guard lock(mutex_);
            LOG_DEBUG("=== Tensor Op Stack ({} ops) ===", stack_.size());
            for (const auto& op : stack_) {
                LOG_DEBUG("{}{}({}){}{}",
                          std::string(op.depth * 2, ' '), op.op_name, op.input_shapes,
                          format_name_tag(op.tensor_name), format_location(op.file, op.line));
            }
        }

        void print_history(const size_t limit = 50) const {
            std::lock_guard lock(mutex_);
            const size_t start_idx = history_.size() > limit ? history_.size() - limit : 0;
            LOG_DEBUG("=== Tensor Op History (last {}) ===", std::min(limit, history_.size()));
            for (size_t i = start_idx; i < history_.size(); ++i) {
                const auto& op = history_[i];
                LOG_DEBUG("  {}({}) -> {} [{:.3f}ms]{}{}",
                          op.op_name, op.input_shapes, op.output_shape, op.duration_ms,
                          format_name_tag(op.tensor_name), format_location(op.file, op.line));
            }
        }

        void clear_history() {
            std::lock_guard lock(mutex_);
            history_.clear();
        }

        [[nodiscard]] const std::vector<OpRecord>& get_history() const { return history_; }

    private:
        using clock = std::chrono::high_resolution_clock;

        TensorOpTracer() = default;

        void finalize_record(const std::string& output) {
            auto& record = stack_.back();
            record.output_shape = output;
            record.duration_ms = std::chrono::duration<float, std::milli>(clock::now() - start_times_.back()).count();
            history_.push_back(std::move(record));
            stack_.pop_back();
            start_times_.pop_back();
        }

        static std::string extract_filename(const char* path) {
            const std::string_view sv(path);
            const auto pos = sv.find_last_of("/\\");
            return std::string(pos != std::string_view::npos ? sv.substr(pos + 1) : sv);
        }

        static std::string format_location(const std::string& file, const int line) {
            return file.empty() ? "" : detail::format_cuda_safe(" @ {}:{}", file, line);
        }

        static std::string format_name_tag(const std::string& name) {
            return name.empty() ? "" : detail::format_cuda_safe(" [{}]", name);
        }

        bool enabled_ = false;
        std::vector<OpRecord> stack_;
        std::vector<clock::time_point> start_times_;
        std::vector<OpRecord> history_;
        mutable std::mutex mutex_;
    };

    // RAII guard - traces if global enabled OR tensor tracked
    class OpTraceGuard {
    public:
        OpTraceGuard(const char* op, const Tensor& input, const SourceSite loc)
            : tracer_(TensorOpTracer::instance()),
              active_(tracer_.should_trace(input)) {
            if (active_)
                tracer_.push(op, input, loc);
        }

        OpTraceGuard(const char* op, const Tensor& in1, const Tensor& in2,
                     const SourceSite loc)
            : tracer_(TensorOpTracer::instance()),
              active_(tracer_.should_trace(in1, in2)) {
            if (active_)
                tracer_.push(op, in1, in2, loc);
        }

        OpTraceGuard(const char* op, const TensorShape& shape, const SourceSite loc)
            : tracer_(TensorOpTracer::instance()),
              active_(tracer_.is_enabled()) {
            if (active_)
                tracer_.push(op, shape, loc);
        }

        OpTraceGuard(const char* op, const TensorShape& in1, const TensorShape& in2,
                     const SourceSite loc)
            : tracer_(TensorOpTracer::instance()),
              active_(tracer_.is_enabled()) {
            if (active_)
                tracer_.push(op, in1, in2, loc);
        }

        void set_output(const TensorShape& shape) {
            if (active_) {
                has_output_ = true;
                output_shape_ = shape;
            }
        }

        ~OpTraceGuard() {
            if (!active_)
                return;
            has_output_ ? tracer_.pop(output_shape_) : tracer_.pop();
        }

    private:
        TensorOpTracer& tracer_;
        bool active_;
        bool has_output_ = false;
        TensorShape output_shape_;
    };

} // namespace lfs::core::debug

#ifdef TENSOR_OP_TRACING
#define TRACE_OP(name, shape)                                                        \
    lfs::core::debug::OpTraceGuard _trace_guard_##__LINE__(                         \
        (name), (shape), LFS_SOURCE_SITE_CURRENT())
#define TRACE_OP2(name, s1, s2)                                                     \
    lfs::core::debug::OpTraceGuard _trace_guard_##__LINE__(                         \
        (name), (s1), (s2), LFS_SOURCE_SITE_CURRENT())
#define TRACE_OP_OUTPUT(shape)  _trace_guard_##__LINE__.set_output(shape)
#define TRACE_PRINT_STACK()     lfs::core::debug::TensorOpTracer::instance().print_stack()
#define TRACE_PRINT_HISTORY(n)  lfs::core::debug::TensorOpTracer::instance().print_history(n)
#else
#define TRACE_OP(name, shape)   ((void)0)
#define TRACE_OP2(name, s1, s2) ((void)0)
#define TRACE_OP_OUTPUT(shape)  ((void)0)
#define TRACE_PRINT_STACK()     ((void)0)
#define TRACE_PRINT_HISTORY(n)  ((void)0)
#endif
