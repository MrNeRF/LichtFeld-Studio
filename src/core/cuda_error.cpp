/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"

#include "core/environment.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <format>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

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
        std::once_flag g_failure_report_provider_once;
        std::atomic<bool> g_cuda_unavailable{false};

        [[nodiscard]] uint64_t current_thread_id() noexcept {
            static thread_local const uint64_t id =
                static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
            return id;
        }

        [[nodiscard]] std::string cuda_error_text(const cudaError_t error) {
            const char* name = cudaGetErrorName(error);
            const char* description = cudaGetErrorString(error);
            return std::format("{} ({}): {}",
                               name ? name : "unknown CUDA error",
                               static_cast<int>(error),
                               description ? description : "description unavailable");
        }

        void append_runtime_context(std::ostream& out) {
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

        void append_breadcrumbs(std::ostream& out) {
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

        void append_cuda_failure_report_sections(
            std::ostream& out,
            const FailureReportSectionPosition position,
            const FailureReport&) {
            if (position == FailureReportSectionPosition::BeforeStackTrace) {
                append_runtime_context(out);
                return;
            }
            append_breadcrumbs(out);
            out << "Hint: CUDA reports async errors at the next sync point. Set "
                   "LFS_CUDA_SYNC_DEBUG=1 to synchronize after every op and pinpoint the true origin.\n";
        }

        void ensure_cuda_failure_report_provider_registered() {
            std::call_once(g_failure_report_provider_once, [] {
                register_failure_report_section_provider(
                    "CUDA runtime error", append_cuda_failure_report_sections);
            });
        }

        [[nodiscard]] std::string format_cuda_detail_sections(
            const CudaCheckState& state,
            const cudaError_t post_sync_error,
            const cudaError_t post_peek_error) {
            std::ostringstream out;
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
            return out.str();
        }

        void emit_cuda_failure_report(const cudaError_t effective_error,
                                      const CudaCheckState& state,
                                      const char* expression,
                                      const std::string_view message,
                                      const SourceSite& location,
                                      const cudaError_t post_sync_error,
                                      const cudaError_t post_peek_error) noexcept {
            try {
                if (is_cuda_unavailable_error(effective_error)) {
                    if (!latch_cuda_unavailable(effective_error)) {
                        return;
                    }
                }

                ensure_cuda_failure_report_provider_registered();
                const std::string error = cuda_error_text(effective_error);
                const std::string detail_sections = format_cuda_detail_sections(
                    state, post_sync_error, post_peek_error);
                emit_failure_report(FailureReport{
                    .family = "CUDA runtime error",
                    .error = error,
                    .expression = expression,
                    .message = message,
                    .detail_sections = detail_sections,
                    .location = location,
                    .deduplication_code = static_cast<long long>(effective_error),
                    .stacktrace_skip_frames = 2,
                });
            } catch (...) {
            }
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
        static const bool enabled = environment::flag("LFS_CUDA_SYNC_DEBUG");
        return enabled;
    }

    void initialize_cuda_diagnostics() noexcept {
        try {
            ensure_cuda_failure_report_provider_registered();
            if (cuda_sync_debug_enabled()) {
                std::call_once(g_sync_debug_log_once, [] {
                    std::fprintf(
                        stderr,
                        "LFS_CUDA_SYNC_DEBUG=1 active: synchronizing before and after every checked CUDA operation\n");
                });
            }
        } catch (...) {
            // Diagnostic initialization must not turn a checked CUDA call into
            // a process termination.
        }
    }

    bool is_cuda_unavailable_error(const cudaError_t error) noexcept {
        switch (error) {
        case cudaErrorInitializationError:
        case cudaErrorInsufficientDriver:
        case cudaErrorNoDevice:
        case cudaErrorDevicesUnavailable:
        case cudaErrorSystemNotReady:
        case cudaErrorSystemDriverMismatch:
        case cudaErrorCompatNotSupportedOnDevice:
        case cudaErrorStartupFailure:
            return true;
        default:
            return false;
        }
    }

    bool cuda_is_unavailable() noexcept {
        return g_cuda_unavailable.load(std::memory_order_relaxed);
    }

    bool latch_cuda_unavailable(const cudaError_t error) noexcept {
        bool expected = false;
        if (!g_cuda_unavailable.compare_exchange_strong(expected, true)) {
            return false;
        }
        try {
            Logger::get().log_internal(
                LogLevel::Error, LFS_SOURCE_SITE_CURRENT(),
                std::format(
                    "CUDA unavailable — GPU features disabled. A driver restart may be required. ({})",
                    cuda_error_text(error)));
        } catch (...) {
        }
        return true;
    }

    void reset_cuda_diagnostics_for_testing() noexcept {
        g_cuda_unavailable.store(false, std::memory_order_relaxed);
        reset_failure_report_dedup_for_testing();
    }

    CudaCheckState prepare_cuda_check(const char*,
                                      const SourceSite,
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

    CudaCheckState sample_cuda_pre_call_state(const cudaStream_t stream) noexcept {
        return prepare_cuda_check("", LFS_SOURCE_SITE_CURRENT(), stream);
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
        const SourceSite location) {
        emit_cuda_failure_report(
            completion.effective_error, state, expression, message, location,
            completion.post_sync_error, completion.post_peek_error);
        throw std::runtime_error(std::format(
            "CUDA call failed: {} at {}:{}", expression, location.file_name(), location.line()));
    }

    void finish_cuda_check(const cudaError_t result,
                           const CudaCheckState& state,
                           const char* expression,
                           const std::string_view message,
                           const SourceSite location) {
        const CudaCheckCompletion completion = complete_cuda_check(result, state);
        if (completion.effective_error == cudaSuccess) [[likely]] {
            return;
        }
        report_cuda_check_failure(completion, state, expression, message, location);
    }

    void ensure_cuda_success(const cudaError_t result,
                             const CudaCheckState& state,
                             const std::string_view expression,
                             const std::string_view message,
                             const SourceSite location,
                             const CudaFailureDisposition disposition) {
        if (result == cudaSuccess) [[likely]] {
            return;
        }
        if (disposition == CudaFailureDisposition::Throw) {
            const std::string expression_copy(expression);
            finish_cuda_check(result, state, expression_copy.c_str(), message, location);
            return;
        }
        try {
            const std::string expression_copy(expression);
            emit_cuda_failure_report(
                result, state, expression_copy.c_str(), message, location,
                cudaSuccess, cudaSuccess);
        } catch (...) {
            // Recovery, teardown, and allocator fallback paths use LogOnly and
            // must never acquire a new failure mode from diagnostics themselves.
        }
    }

    void ensure_cuda_success(const cudaError_t result,
                             const std::string_view expression,
                             const std::string_view message,
                             const SourceSite location,
                             const CudaFailureDisposition disposition) {
        ensure_cuda_success(
            result, CudaCheckState{}, expression, message, location, disposition);
    }

    void validate_cuda_device_pointer(const void* pointer,
                                      const std::string_view name,
                                      const SourceSite location) {
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
                                               const SourceSite location) {
        if (pointer) {
            validate_cuda_device_pointer(pointer, name, location);
        }
    }

} // namespace lfs::core
