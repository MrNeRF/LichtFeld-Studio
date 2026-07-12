/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"

#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <format>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

#if __has_include(<stacktrace>)
#include <stacktrace>
#endif

#if defined(__cpp_lib_stacktrace) && __cpp_lib_stacktrace >= 202011L
#define LFS_HAS_STD_STACKTRACE 1
#elif defined(__unix__) || defined(__APPLE__)
#include <execinfo.h>
#define LFS_HAS_POSIX_BACKTRACE 1
#endif

namespace lfs::core {
    namespace {

        struct BreadcrumbSlot {
            std::atomic<uint64_t> sequence{0};
            std::atomic<const char*> tag{nullptr};
            std::atomic<const char*> file{nullptr};
            std::atomic<uint32_t> line{0};
            std::atomic<uintptr_t> stream{0};
            std::atomic<uint64_t> thread_id{0};
        };

        std::array<BreadcrumbSlot, CUDA_BREADCRUMB_CAPACITY> g_breadcrumbs;
        std::atomic<uint64_t> g_breadcrumb_sequence{0};
        std::once_flag g_sync_debug_log_once;

        [[nodiscard]] uint64_t current_thread_id() noexcept {
            static thread_local const uint64_t id =
                static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            return id;
        }

        [[nodiscard]] bool env_flag_enabled(const char* name) noexcept {
            const char* value = std::getenv(name);
            return value && value[0] != '\0' && std::string_view(value) != "0";
        }

        [[nodiscard]] std::string cuda_error_text(const cudaError_t error) {
            const char* name = cudaGetErrorName(error);
            const char* description = cudaGetErrorString(error);
            return std::format("{} ({}): {}",
                               name ? name : "unknown CUDA error",
                               static_cast<int>(error),
                               description ? description : "description unavailable");
        }

        void append_runtime_context(std::ostringstream& out) {
            int device = -1;
            int device_count = -1;
            const cudaError_t device_status = cudaGetDevice(&device);
            const cudaError_t count_status = cudaGetDeviceCount(&device_count);

            out << "Thread: " << current_thread_id() << '\n';
            out << "CUDA device: ";
            if (device_status == cudaSuccess) {
                out << device;
            } else {
                out << "unavailable (cudaGetDevice failed: " << cuda_error_text(device_status) << ')';
            }
            out << " / device count: ";
            if (count_status == cudaSuccess) {
                out << device_count;
            } else {
                out << "unavailable (cudaGetDeviceCount failed: " << cuda_error_text(count_status) << ')';
            }
            out << '\n';

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            const cudaError_t memory_status = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (memory_status == cudaSuccess) {
                out << std::format("VRAM: free={} MiB, used={} MiB, total={} MiB\n",
                                   free_bytes >> 20,
                                   (total_bytes - free_bytes) >> 20,
                                   total_bytes >> 20);
            } else {
                out << "VRAM: unavailable (cudaMemGetInfo failed: "
                    << cuda_error_text(memory_status) << ")\n";
            }
        }

        void append_breadcrumbs(std::ostringstream& out) {
            out << "CUDA breadcrumbs (most recent first):\n";
            const auto breadcrumbs = cuda_breadcrumbs_most_recent_first();
            if (breadcrumbs.empty()) {
                out << "  <none>\n";
                return;
            }
            for (const auto& entry : breadcrumbs) {
                out << std::format("  #{} {} at {}:{} thread={} stream={:#x}\n",
                                   entry.sequence,
                                   entry.tag ? entry.tag : "<untagged>",
                                   entry.file ? entry.file : "<unknown>",
                                   entry.line,
                                   entry.thread_id,
                                   entry.stream);
            }
        }

        [[nodiscard]] std::string format_cuda_failure_report(
            const cudaError_t result,
            const CudaCheckState& state,
            const char* expression,
            const std::string_view message,
            const std::source_location& location,
            const cudaError_t post_sync_error,
            const cudaError_t post_peek_error) {
            std::ostringstream out;
            out << "========== LFS FAILURE REPORT ==========\n";
            out << "Family: CUDA runtime error\n";
            out << "Error: " << cuda_error_text(result) << '\n';
            out << "Failed expression: " << expression << '\n';
            out << std::format("Detection site: {}:{} ({})\n",
                               location.file_name(), location.line(), location.function_name());
            if (!message.empty()) {
                out << "Context: " << message << '\n';
            }
            if (state.stream != 0) {
                out << std::format("Stream: {:#x}\n", state.stream);
            }
            if (!state.pre_call_sampled) {
                out << "Attribution: pre-call CUDA state was not sampled by this status adapter.\n";
            } else if (state.pre_call_error != cudaSuccess || state.pre_call_sync_error != cudaSuccess) {
                out << "Attribution: pre-existing CUDA error detected BEFORE this call — "
                       "this site is NOT the origin.\n";
                if (state.pre_call_error != cudaSuccess) {
                    out << "Pre-call cudaPeekAtLastError: " << cuda_error_text(state.pre_call_error) << '\n';
                }
                if (state.pre_call_sync_error != cudaSuccess) {
                    out << "Pre-call synchronization: " << cuda_error_text(state.pre_call_sync_error) << '\n';
                }
            } else {
                out << "Attribution: no pre-existing CUDA error was visible before this call.\n";
            }
            if (post_sync_error != cudaSuccess) {
                out << "Post-call synchronization: " << cuda_error_text(post_sync_error) << '\n';
            }
            if (post_peek_error != cudaSuccess) {
                out << "Post-call cudaPeekAtLastError: " << cuda_error_text(post_peek_error) << '\n';
            }
            append_runtime_context(out);
            out << "Host stack trace:\n"
                << capture_host_stacktrace(2);
            append_breadcrumbs(out);
            out << "Hint: CUDA reports async errors at the next sync point. Set "
                   "LFS_CUDA_SYNC_DEBUG=1 to synchronize after every op and pinpoint the true origin.\n";
            out << "========================================";
            return out.str();
        }

    } // namespace

    void record_cuda_breadcrumb(const char* tag,
                                const char* file,
                                const uint32_t line,
                                const cudaStream_t stream) noexcept {
        const uint64_t sequence = g_breadcrumb_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        BreadcrumbSlot& slot = g_breadcrumbs[(sequence - 1) % CUDA_BREADCRUMB_CAPACITY];
        slot.sequence.store(0, std::memory_order_relaxed);
        slot.tag.store(tag, std::memory_order_relaxed);
        slot.file.store(file, std::memory_order_relaxed);
        slot.line.store(line, std::memory_order_relaxed);
        slot.stream.store(reinterpret_cast<uintptr_t>(stream), std::memory_order_relaxed);
        slot.thread_id.store(current_thread_id(), std::memory_order_relaxed);
        slot.sequence.store(sequence, std::memory_order_release);
    }

    std::vector<CudaBreadcrumb> cuda_breadcrumbs_most_recent_first() {
        const uint64_t newest = g_breadcrumb_sequence.load(std::memory_order_acquire);
        const uint64_t count = std::min<uint64_t>(newest, CUDA_BREADCRUMB_CAPACITY);
        std::vector<CudaBreadcrumb> result;
        result.reserve(static_cast<size_t>(count));
        for (uint64_t offset = 0; offset < count; ++offset) {
            const uint64_t expected = newest - offset;
            const BreadcrumbSlot& slot = g_breadcrumbs[(expected - 1) % CUDA_BREADCRUMB_CAPACITY];
            const uint64_t before = slot.sequence.load(std::memory_order_acquire);
            if (before != expected) {
                continue;
            }
            CudaBreadcrumb entry{
                .sequence = expected,
                .tag = slot.tag.load(std::memory_order_relaxed),
                .file = slot.file.load(std::memory_order_relaxed),
                .line = slot.line.load(std::memory_order_relaxed),
                .stream = slot.stream.load(std::memory_order_relaxed),
                .thread_id = slot.thread_id.load(std::memory_order_relaxed),
            };
            if (slot.sequence.load(std::memory_order_acquire) == expected) {
                result.push_back(entry);
            }
        }
        return result;
    }

    void clear_cuda_breadcrumbs_for_testing() noexcept {
        g_breadcrumb_sequence.store(0, std::memory_order_release);
        for (auto& slot : g_breadcrumbs) {
            slot.sequence.store(0, std::memory_order_relaxed);
        }
    }

    bool cuda_sync_debug_enabled() noexcept {
        static const bool enabled = env_flag_enabled("LFS_CUDA_SYNC_DEBUG");
        return enabled;
    }

    void initialize_cuda_diagnostics() noexcept {
        try {
            if (cuda_sync_debug_enabled()) {
                std::call_once(g_sync_debug_log_once, [] {
                    std::fprintf(
                        stderr,
                        "LFS_CUDA_SYNC_DEBUG=1 active: synchronizing before and after every checked CUDA operation\n");
                });
            }
        } catch (...) {
            // Diagnostic initialization must not turn a checked CUDA call into
            // a process termination when the logger itself is unavailable.
        }
    }

    CudaCheckState prepare_cuda_check(const char*,
                                      const std::source_location&,
                                      const cudaStream_t stream) noexcept {
        initialize_cuda_diagnostics();
        CudaCheckState state;
        state.stream = reinterpret_cast<uintptr_t>(stream);
        state.pre_call_sampled = true;
        if (cuda_sync_debug_enabled()) {
            state.pre_call_sync_error = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        }
        // This sample must precede the checked call; sampling after it cannot distinguish a
        // sticky predecessor from an error produced by the expression itself.
        state.pre_call_error = cudaPeekAtLastError();
        return state;
    }

    CudaCheckCompletion complete_cuda_check(
        const cudaError_t result,
        const CudaCheckState& state) noexcept {
        CudaCheckCompletion completion;
        if (cuda_sync_debug_enabled()) {
            completion.post_sync_error =
                state.stream != 0
                    ? cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(state.stream))
                    : cudaDeviceSynchronize();
            completion.post_peek_error = cudaPeekAtLastError();
        }

        completion.effective_error = result != cudaSuccess
                                         ? result
                                     : completion.post_sync_error != cudaSuccess
                                         ? completion.post_sync_error
                                         : completion.post_peek_error;
        return completion;
    }

    [[noreturn]] void report_cuda_check_failure(
        const CudaCheckCompletion& completion,
        const CudaCheckState& state,
        const char* expression,
        const std::string_view message,
        const std::source_location& location) {
        const std::string report = format_cuda_failure_report(
            completion.effective_error, state, expression, message, location,
            completion.post_sync_error, completion.post_peek_error);
        Logger::get().log_internal(LogLevel::Error, location, report);
        throw std::runtime_error(std::format(
            "CUDA call failed: {} at {}:{}", expression, location.file_name(), location.line()));
    }

    void finish_cuda_check(const cudaError_t result,
                           const CudaCheckState& state,
                           const char* expression,
                           const std::string_view message,
                           const std::source_location& location) {
        const CudaCheckCompletion completion = complete_cuda_check(result, state);
        if (completion.effective_error == cudaSuccess) [[likely]] {
            return;
        }
        report_cuda_check_failure(completion, state, expression, message, location);
    }

    void ensure_cuda_success(const cudaError_t result,
                             const std::string_view expression,
                             const std::string_view message,
                             const std::source_location& location,
                             const CudaFailureDisposition disposition) {
        if (result == cudaSuccess) [[likely]] {
            return;
        }
        if (disposition == CudaFailureDisposition::Throw) {
            const std::string expression_copy(expression);
            finish_cuda_check(result, CudaCheckState{}, expression_copy.c_str(), message, location);
            return;
        }
        try {
            const std::string expression_copy(expression);
            const std::string report = format_cuda_failure_report(
                result, CudaCheckState{}, expression_copy.c_str(), message, location,
                cudaSuccess, cudaSuccess);
            Logger::get().log_internal(LogLevel::Error, location, report);
        } catch (...) {
            // Recovery, teardown, and allocator fallback paths use LogOnly and
            // must never acquire a new failure mode from diagnostics themselves.
        }
    }

    void validate_cuda_device_pointer(const void* pointer,
                                      const std::string_view name,
                                      const std::source_location& location) {
        if (!pointer) {
            detail::assertion_failed(
                "LFS boundary contract", "pointer != nullptr",
                std::format("CUDA pointer '{}' must not be null", name), location);
        }

        cudaPointerAttributes attributes{};
        const auto state = prepare_cuda_check(
            "cudaPointerGetAttributes(&attributes, pointer)", location);
        const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
        finish_cuda_check(result, state, "cudaPointerGetAttributes(&attributes, pointer)",
                          std::format("validating CUDA pointer '{}' ({})", name, pointer), location);
        if (attributes.type != cudaMemoryTypeDevice) {
            detail::assertion_failed(
                "LFS boundary contract", "attributes.type == cudaMemoryTypeDevice",
                std::format("CUDA pointer '{}' has memory type {} instead of device type {}",
                            name, static_cast<int>(attributes.type),
                            static_cast<int>(cudaMemoryTypeDevice)),
                location);
        }
    }

    void validate_cuda_device_pointer_optional(const void* pointer,
                                               const std::string_view name,
                                               const std::source_location& location) {
        if (pointer) {
            validate_cuda_device_pointer(pointer, name, location);
        }
    }

    std::string capture_host_stacktrace(const size_t skip_frames) {
#if defined(LFS_HAS_STD_STACKTRACE)
        std::ostringstream out;
        const auto trace = std::stacktrace::current(skip_frames + 1);
        if (trace.empty()) {
            return "  <unavailable>\n";
        }
        size_t index = 0;
        for (const auto& entry : trace) {
            out << "  #" << index++ << ' ' << entry << '\n';
        }
        return out.str();
#elif defined(LFS_HAS_POSIX_BACKTRACE)
        std::array<void*, 128> frames{};
        const int count = ::backtrace(frames.data(), static_cast<int>(frames.size()));
        if (count <= 0) {
            return "  <unavailable>\n";
        }
        char** symbols = ::backtrace_symbols(frames.data(), count);
        if (!symbols) {
            return "  <unavailable>\n";
        }
        std::ostringstream out;
        for (int i = static_cast<int>(skip_frames + 1); i < count; ++i) {
            out << "  #" << (i - static_cast<int>(skip_frames + 1)) << ' ' << symbols[i] << '\n';
        }
        std::free(symbols);
        return out.str();
#else
        (void)skip_frames;
        return "  <unavailable on this platform>\n";
#endif
    }

    std::string format_failure_report(
        const std::string_view family,
        const std::string_view contract,
        const std::string_view expression,
        const std::string_view message,
        const std::source_location& location,
        const std::string_view stacktrace) {
        std::ostringstream out;
        out << "========== LFS FAILURE REPORT ==========\n";
        out << "Family: " << family << '\n';
        out << "Contract: " << contract << '\n';
        out << "Failed expression: " << expression << '\n';
        out << std::format("Detection site: {}:{} ({})\n",
                           location.file_name(), location.line(), location.function_name());
        if (!message.empty()) {
            out << "Context: " << message << '\n';
        }
        append_runtime_context(out);
        out << "Host stack trace:\n"
            << stacktrace;
        append_breadcrumbs(out);
        out << "Hint: CUDA reports async errors at the next sync point. Set "
               "LFS_CUDA_SYNC_DEBUG=1 to synchronize after every op and pinpoint the true origin.\n";
        out << "========================================";
        return out.str();
    }

    std::string format_contract_failure_report(
        const std::string_view contract,
        const std::string_view expression,
        const std::string_view message,
        const std::source_location& location,
        const std::string_view stacktrace) {
        return format_failure_report(
            "tensor contract violation", contract, expression, message, location, stacktrace);
    }

    void report_tensor_exception(const std::string_view message,
                                 const std::source_location& location) {
        const std::string report = format_contract_failure_report(
            "Tensor exception", "throw TensorError", message, location, capture_host_stacktrace(1));
        Logger::get().log_internal(LogLevel::Error, location, report);
    }

    namespace detail {

        [[noreturn]] void assertion_failed(
            const std::string_view contract,
            const std::string_view expression,
            const std::string_view message,
            const std::source_location location) {
            const std::string report = format_contract_failure_report(
                contract, expression, message, location, capture_host_stacktrace(1));
            Logger::get().log_internal(LogLevel::Error, location, report);

            std::string error = std::format("{} failed: {}", contract, expression);
            if (!message.empty()) {
                error += " — ";
                error += message;
            }
            error += std::format(" ({}:{})", location.file_name(), location.line());
            throw std::runtime_error(error);
        }

    } // namespace detail

} // namespace lfs::core
