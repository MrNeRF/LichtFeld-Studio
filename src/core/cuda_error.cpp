/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"

#include "core/device_fault.hpp"
#include "core/environment.hpp"
#include "core/error_codes.hpp"
#include "core/failure_report.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <format>
#include <mutex>
#include <optional>
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
            std::atomic<uint64_t> a0{0};
            std::atomic<uint64_t> a1{0};
            std::atomic<uint64_t> a2{0};
        };

        struct RegisteredCudaAddressRange {
            uint64_t base = 0;
            size_t bytes = 0;
            std::string label;
        };

        struct DeadCudaAddressRange {
            RegisteredCudaAddressRange range;
            std::chrono::steady_clock::time_point unmapped_at;
        };

        struct CudaAddressAnnotation {
            uint64_t candidate = 0;
            uint64_t base = 0;
            size_t bytes = 0;
            std::string label;
            std::optional<int64_t> unmapped_milliseconds;
        };

        std::array<BreadcrumbSlot, CUDA_BREADCRUMB_CAPACITY> g_breadcrumbs;
        std::atomic<uint64_t> g_breadcrumb_sequence{0};
        std::once_flag g_sync_debug_log_once;
        std::once_flag g_failure_report_provider_once;
        std::atomic<bool> g_cuda_unavailable{false};
        std::atomic<int> g_last_cuda_check_failure{cudaSuccess};
        thread_local uint64_t g_lfs_launch_watermark = 0;
        std::mutex g_cuda_address_ranges_mutex;
        std::vector<RegisteredCudaAddressRange> g_live_cuda_address_ranges;
        constexpr size_t CUDA_DEAD_ADDRESS_RANGE_CAPACITY = 32;
        constexpr size_t CUDA_ANNOTATION_BREADCRUMB_COUNT = 8;
        std::array<std::optional<DeadCudaAddressRange>, CUDA_DEAD_ADDRESS_RANGE_CAPACITY>
            g_dead_cuda_address_ranges;
        size_t g_next_dead_cuda_address_range = 0;
        size_t g_dead_cuda_address_range_count = 0;

        constexpr std::string_view kModeTokenCudaSync = "cuda-sync";
        constexpr std::string_view kModeTokenDeviceTrap = "device-trap";
        constexpr std::string_view kModeTokenVkFatal = "vk-fatal";

        [[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value) noexcept {
            const auto is_space = [](const char ch) noexcept {
                return std::isspace(static_cast<unsigned char>(ch)) != 0;
            };
            while (!value.empty() && is_space(value.front())) {
                value.remove_prefix(1);
            }
            while (!value.empty() && is_space(value.back())) {
                value.remove_suffix(1);
            }
            return value;
        }

        // Recognizes the legacy boolean spellings shared with environment::flag().
        [[nodiscard]] std::optional<bool> parse_legacy_bool_token(const std::string_view value) noexcept {
            using environment::detail::equals_ignore_ascii_case;
            if (equals_ignore_ascii_case(value, "1") || equals_ignore_ascii_case(value, "true") ||
                equals_ignore_ascii_case(value, "yes") || equals_ignore_ascii_case(value, "on")) {
                return true;
            }
            if (equals_ignore_ascii_case(value, "0") || equals_ignore_ascii_case(value, "false") ||
                equals_ignore_ascii_case(value, "no") || equals_ignore_ascii_case(value, "off")) {
                return false;
            }
            return std::nullopt;
        }

        struct ModeListParse {
            unsigned modes = 0;
            std::vector<std::string_view> unknown_tokens;
        };

        [[nodiscard]] ModeListParse parse_mode_list_tokens(const std::string_view value) {
            using environment::detail::equals_ignore_ascii_case;
            ModeListParse result;
            size_t pos = 0;
            while (pos <= value.size()) {
                const size_t comma = value.find(',', pos);
                const std::string_view raw_token = comma == std::string_view::npos
                                                       ? value.substr(pos)
                                                       : value.substr(pos, comma - pos);
                const std::string_view token = trim_ascii_whitespace(raw_token);
                if (!token.empty()) {
                    if (equals_ignore_ascii_case(token, kModeTokenCudaSync)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::CudaSync);
                    } else if (equals_ignore_ascii_case(token, kModeTokenDeviceTrap)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::DeviceTrap);
                    } else if (equals_ignore_ascii_case(token, kModeTokenVkFatal)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::VkFatal);
                    } else {
                        result.unknown_tokens.push_back(token);
                    }
                }
                if (comma == std::string_view::npos) {
                    break;
                }
                pos = comma + 1;
            }
            return result;
        }

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

        [[nodiscard]] bool address_is_inside(
            const uint64_t address, const RegisteredCudaAddressRange& range) noexcept {
            return address >= range.base &&
                   address - range.base < static_cast<uint64_t>(range.bytes);
        }

        [[nodiscard]] std::string format_range_size(const size_t bytes) {
            constexpr size_t KIB = size_t{1024};
            constexpr size_t MIB = KIB * size_t{1024};
            constexpr size_t GIB = MIB * size_t{1024};
            if (bytes >= GIB && bytes % GIB == 0) {
                return std::format("{} GiB", bytes / GIB);
            }
            if (bytes >= MIB && bytes % MIB == 0) {
                return std::format("{} MiB", bytes / MIB);
            }
            if (bytes >= KIB && bytes % KIB == 0) {
                return std::format("{} KiB", bytes / KIB);
            }
            return std::format("{} {}", bytes, bytes == 1 ? "byte" : "bytes");
        }

        [[nodiscard]] std::vector<CudaAddressAnnotation> find_cuda_address_annotations(
            const std::vector<CudaBreadcrumb>& breadcrumbs) {
            std::array<uint64_t, CUDA_ANNOTATION_BREADCRUMB_COUNT * 3> candidates{};
            size_t candidate_count = 0;
            const size_t breadcrumb_count =
                std::min(breadcrumbs.size(), CUDA_ANNOTATION_BREADCRUMB_COUNT);
            for (size_t index = 0; index < breadcrumb_count; ++index) {
                const std::array<uint64_t, 3> args{
                    breadcrumbs[index].a0,
                    breadcrumbs[index].a1,
                    breadcrumbs[index].a2,
                };
                for (const uint64_t argument : args) {
                    if (argument == 0 ||
                        std::find(candidates.begin(), candidates.begin() + candidate_count, argument) !=
                            candidates.begin() + candidate_count) {
                        continue;
                    }
                    candidates[candidate_count++] = argument;
                }
            }

            std::vector<CudaAddressAnnotation> annotations;
            const auto now = std::chrono::steady_clock::now();
            std::scoped_lock lock(g_cuda_address_ranges_mutex);
            for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
                const uint64_t candidate = candidates[candidate_index];
                const auto live = std::find_if(
                    g_live_cuda_address_ranges.begin(), g_live_cuda_address_ranges.end(),
                    [candidate](const RegisteredCudaAddressRange& range) {
                        return address_is_inside(candidate, range);
                    });
                if (live != g_live_cuda_address_ranges.end()) {
                    annotations.push_back(CudaAddressAnnotation{
                        .candidate = candidate,
                        .base = live->base,
                        .bytes = live->bytes,
                        .label = live->label,
                    });
                    continue;
                }

                for (size_t dead_offset = 0;
                     dead_offset < g_dead_cuda_address_range_count;
                     ++dead_offset) {
                    const size_t dead_index =
                        (g_next_dead_cuda_address_range + CUDA_DEAD_ADDRESS_RANGE_CAPACITY - 1 -
                         dead_offset) %
                        CUDA_DEAD_ADDRESS_RANGE_CAPACITY;
                    const auto& dead = g_dead_cuda_address_ranges[dead_index];
                    if (!dead || !address_is_inside(candidate, dead->range)) {
                        continue;
                    }
                    const int64_t elapsed = std::max<int64_t>(
                        0,
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - dead->unmapped_at)
                            .count());
                    annotations.push_back(CudaAddressAnnotation{
                        .candidate = candidate,
                        .base = dead->range.base,
                        .bytes = dead->range.bytes,
                        .label = dead->range.label,
                        .unmapped_milliseconds = elapsed,
                    });
                    break;
                }
            }
            return annotations;
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

        void append_breadcrumbs(std::ostream& out,
                                const std::vector<CudaBreadcrumb>& breadcrumbs) {
            out << "CUDA breadcrumbs (most recent first):\n";
            if (breadcrumbs.empty()) {
                out << "  <none>\n";
                return;
            }
            for (const auto& entry : breadcrumbs) {
                out << std::format("  #{} {} at {}:{} thread={} stream={:#x}",
                                   entry.sequence,
                                   entry.tag ? entry.tag : "<untagged>",
                                   entry.file ? entry.file : "<unknown>",
                                   entry.line,
                                   entry.thread_id,
                                   entry.stream);
                const std::array<uint64_t, 3> args{entry.a0, entry.a1, entry.a2};
                size_t arg_count = args.size();
                while (arg_count > 0 && args[arg_count - 1] == 0) {
                    --arg_count;
                }
                if (arg_count > 0) {
                    out << " args=";
                    for (size_t index = 0; index < arg_count; ++index) {
                        if (index != 0) {
                            out << ',';
                        }
                        out << std::format("{:#x}", args[index]);
                    }
                }
                out << '\n';
            }
        }

        void append_address_annotations(
            std::ostream& out, const std::vector<CudaBreadcrumb>& breadcrumbs) {
            const auto annotations = find_cuda_address_annotations(breadcrumbs);
            if (annotations.empty()) {
                return;
            }
            out << "Address annotations:\n";
            for (const auto& annotation : annotations) {
                out << std::format("  {:#x} → inside {} range '{}' [{:#x} +{}]",
                                   annotation.candidate,
                                   annotation.unmapped_milliseconds ? "DEAD" : "LIVE",
                                   annotation.label,
                                   annotation.base,
                                   format_range_size(annotation.bytes));
                if (annotation.unmapped_milliseconds) {
                    out << std::format(", unmapped {} ms ago", *annotation.unmapped_milliseconds);
                }
                out << '\n';
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
            const auto breadcrumbs = cuda_breadcrumbs_most_recent_first();
            append_breadcrumbs(out, breadcrumbs);
            append_address_annotations(out, breadcrumbs);
            out << "Hint: CUDA reports async errors at the next sync point. Set "
                   "LFS_CUDA_SYNC_DEBUG=cuda-sync to synchronize after every op and pinpoint the true origin.\n";
        }

        void ensure_cuda_failure_report_provider_registered() {
            std::call_once(g_failure_report_provider_once, [] {
                register_failure_report_section_provider(
                    lfs::to_string(ErrorDomain::CUDA), append_cuda_failure_report_sections);
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
                                      const cudaError_t post_peek_error,
                                      const bool latch_unavailable) noexcept {
            try {
                if (latch_unavailable && is_cuda_unavailable_error(effective_error)) {
                    if (!latch_cuda_unavailable(effective_error)) {
                        return;
                    }
                }

                ensure_cuda_failure_report_provider_registered();
                const std::string error = cuda_error_text(effective_error);
                const std::string detail_sections = format_cuda_detail_sections(
                    state, post_sync_error, post_peek_error);
                emit_failure_report(FailureReport{
                    .family = lfs::to_string(ErrorDomain::CUDA),
                    .error = error,
                    .expression = expression,
                    .message = message,
                    .detail_sections = detail_sections,
                    .location = location,
                    .deduplication_code = static_cast<long long>(effective_error),
                    .stacktrace_skip_frames = 2,
                });
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): publishing a CUDA failure report must not
                // throw a second failure out of the reporting path itself.
            }
        }

    } // namespace

    uint64_t record_cuda_breadcrumb(const char* tag,
                                    const char* file,
                                    const uint32_t line,
                                    const cudaStream_t stream) noexcept {
        return record_cuda_breadcrumb(tag, file, line, stream, 0, 0, 0);
    }

    uint64_t record_cuda_breadcrumb(const char* tag,
                                    const char* file,
                                    const uint32_t line,
                                    const cudaStream_t stream,
                                    const uint64_t a0,
                                    const uint64_t a1,
                                    const uint64_t a2) noexcept {
        const uint64_t sequence = g_breadcrumb_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        BreadcrumbSlot& slot = g_breadcrumbs[(sequence - 1) % CUDA_BREADCRUMB_CAPACITY];
        slot.sequence.store(0, std::memory_order_relaxed);
        slot.tag.store(tag, std::memory_order_relaxed);
        slot.file.store(file, std::memory_order_relaxed);
        slot.line.store(line, std::memory_order_relaxed);
        slot.stream.store(reinterpret_cast<uintptr_t>(stream), std::memory_order_relaxed);
        slot.thread_id.store(current_thread_id(), std::memory_order_relaxed);
        slot.a0.store(a0, std::memory_order_relaxed);
        slot.a1.store(a1, std::memory_order_relaxed);
        slot.a2.store(a2, std::memory_order_relaxed);
        slot.sequence.store(sequence, std::memory_order_release);
        return sequence;
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
                .a0 = slot.a0.load(std::memory_order_relaxed),
                .a1 = slot.a1.load(std::memory_order_relaxed),
                .a2 = slot.a2.load(std::memory_order_relaxed),
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

    void register_cuda_address_range(const void* base,
                                     const std::size_t bytes,
                                     std::string label) {
        if (base == nullptr || bytes == 0) {
            return;
        }
        const uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
        std::scoped_lock lock(g_cuda_address_ranges_mutex);
        const auto existing = std::find_if(
            g_live_cuda_address_ranges.begin(), g_live_cuda_address_ranges.end(),
            [address](const RegisteredCudaAddressRange& range) {
                return range.base == address;
            });
        if (existing != g_live_cuda_address_ranges.end()) {
            existing->bytes = bytes;
            existing->label = std::move(label);
            return;
        }
        g_live_cuda_address_ranges.push_back(RegisteredCudaAddressRange{
            .base = address,
            .bytes = bytes,
            .label = std::move(label),
        });
    }

    void unregister_cuda_address_range(const void* base) {
        if (base == nullptr) {
            return;
        }
        const uint64_t address = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(base));
        std::scoped_lock lock(g_cuda_address_ranges_mutex);
        const auto existing = std::find_if(
            g_live_cuda_address_ranges.begin(), g_live_cuda_address_ranges.end(),
            [address](const RegisteredCudaAddressRange& range) {
                return range.base == address;
            });
        if (existing == g_live_cuda_address_ranges.end()) {
            return;
        }

        RegisteredCudaAddressRange dead_range = std::move(*existing);
        g_live_cuda_address_ranges.erase(existing);
        g_dead_cuda_address_ranges[g_next_dead_cuda_address_range] = DeadCudaAddressRange{
            .range = std::move(dead_range),
            .unmapped_at = std::chrono::steady_clock::now(),
        };
        g_next_dead_cuda_address_range =
            (g_next_dead_cuda_address_range + 1) % CUDA_DEAD_ADDRESS_RANGE_CAPACITY;
        g_dead_cuda_address_range_count = std::min(
            g_dead_cuda_address_range_count + 1, CUDA_DEAD_ADDRESS_RANGE_CAPACITY);
    }

    ParsedDiagnosticModes parse_diagnostic_modes(
        const std::optional<std::string_view> sync_debug_value,
        const std::optional<std::string_view> vk_validation_fatal_value) noexcept {
        ParsedDiagnosticModes result;
        try {
            if (sync_debug_value) {
                const std::string_view trimmed = trim_ascii_whitespace(*sync_debug_value);
                if (!trimmed.empty()) {
                    if (const auto legacy = parse_legacy_bool_token(trimmed)) {
                        if (*legacy) {
                            result.modes |= static_cast<unsigned>(DiagnosticMode::CudaSync);
                        }
                    } else {
                        const ModeListParse parsed = parse_mode_list_tokens(trimmed);
                        result.modes |= parsed.modes;
                        if (!parsed.unknown_tokens.empty()) {
                            result.unknown_tokens_present = true;
                            for (size_t i = 0; i < parsed.unknown_tokens.size(); ++i) {
                                if (i != 0) {
                                    result.unknown_tokens += ", ";
                                }
                                result.unknown_tokens += parsed.unknown_tokens[i];
                            }
                        }
                    }
                }
            }
            if (vk_validation_fatal_value) {
                const std::string_view trimmed = trim_ascii_whitespace(*vk_validation_fatal_value);
                if (!trimmed.empty()) {
                    result.legacy_alias_present = true;
                    if (const auto legacy = parse_legacy_bool_token(trimmed); legacy && *legacy) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::VkFatal);
                    }
                }
            }
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): parsing must never turn a startup env-var read
            // into a crash; fall back to whatever modes were resolved before the failure.
        }
        return result;
    }

    unsigned diagnostic_modes() noexcept {
        static const unsigned modes = [] {
            const ParsedDiagnosticModes parsed = parse_diagnostic_modes(
                environment::value("LFS_CUDA_SYNC_DEBUG"),
                environment::value("LFS_VK_VALIDATION_FATAL"));
            try {
                if (parsed.legacy_alias_present) {
                    std::fprintf(
                        stderr,
                        "LFS_VK_VALIDATION_FATAL is deprecated; set LFS_CUDA_SYNC_DEBUG=vk-fatal "
                        "(or add vk-fatal to its mode list) instead.\n");
                }
                if (parsed.unknown_tokens_present) {
                    std::fprintf(
                        stderr,
                        "LFS_CUDA_SYNC_DEBUG: ignoring unknown mode token(s) [%s]; valid modes are "
                        "cuda-sync, device-trap, vk-fatal.\n",
                        parsed.unknown_tokens.c_str());
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): a diagnostics warning about bad env tokens
                // must not itself become a failure mode.
            }
            return parsed.modes;
        }();
        return modes;
    }

    bool diagnostic_mode_enabled(const DiagnosticMode mode) noexcept {
        return (diagnostic_modes() & static_cast<unsigned>(mode)) != 0;
    }

    bool cuda_sync_debug_enabled() noexcept {
        return diagnostic_mode_enabled(DiagnosticMode::CudaSync);
    }

    void initialize_cuda_diagnostics() noexcept {
        try {
            ensure_cuda_failure_report_provider_registered();
            if (cuda_sync_debug_enabled()) {
                std::call_once(g_sync_debug_log_once, [] {
                    std::fprintf(
                        stderr,
                        "LFS_CUDA_SYNC_DEBUG active (cuda-sync mode): synchronizing before and after "
                        "every checked CUDA operation\n");
                });
            }
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): diagnostic initialization must not turn a
            // checked CUDA call into a process termination.
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
            // LFS-CENSUS-OK(empty-catch): the CUDA-unavailable notice is best-effort;
            // logging failure must not mask the original driver condition.
        }
        return true;
    }

    void reset_cuda_diagnostics_for_testing() noexcept {
        g_cuda_unavailable.store(false, std::memory_order_relaxed);
        g_last_cuda_check_failure.store(cudaSuccess, std::memory_order_relaxed);
        g_lfs_launch_watermark = 0;
        try {
            std::scoped_lock lock(g_cuda_address_ranges_mutex);
            g_live_cuda_address_ranges.clear();
            for (auto& range : g_dead_cuda_address_ranges) {
                range.reset();
            }
            g_next_dead_cuda_address_range = 0;
            g_dead_cuda_address_range_count = 0;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): test-only forensic-table reset must never
            // propagate; a partial reset is still a usable baseline.
        }
        reset_failure_report_dedup_for_testing();
        // Spec §0.3 / §1.6: device-fault host registry joins the testing reset.
        reset_device_fault_registry_for_testing();
    }

    uint64_t current_cuda_breadcrumb_sequence() noexcept {
        return g_breadcrumb_sequence.load(std::memory_order_acquire);
    }

    cudaError_t exchange_last_cuda_check_failure(const cudaError_t native) noexcept {
        return static_cast<cudaError_t>(
            g_last_cuda_check_failure.exchange(static_cast<int>(native), std::memory_order_relaxed));
    }

    CudaAwaitTicket cuda_record_range(const cudaStream_t stream, const char* operation_tag) noexcept {
        return CudaAwaitTicket{
            .stream = reinterpret_cast<uintptr_t>(stream),
            .first_sequence = current_cuda_breadcrumb_sequence(),
            .operation_tag = operation_tag,
        };
    }

    void handle_cuda_launch_check_slow_path(const cudaError_t status,
                                            const cudaStream_t stream,
                                            const char* tag,
                                            const SourceSite location,
                                            const uint64_t last_sequence) {
        cudaError_t effective = status;
        if (cuda_sync_debug_enabled()) {
            const cudaError_t sync_status =
                stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            const cudaError_t post_peek = cudaPeekAtLastError();
            effective = status != cudaSuccess        ? status
                        : sync_status != cudaSuccess ? sync_status
                                                     : post_peek;
        }
        if (effective == cudaSuccess) {
            return;
        }
        const uint64_t first_sequence = g_lfs_launch_watermark + 1;
        g_lfs_launch_watermark = last_sequence;
        const cudaError_t predecessor = exchange_last_cuda_check_failure(effective);
        report_cuda_launch_check_failure(CudaFailureSeed{
            .native = effective,
            .predecessor = predecessor,
            .stream = reinterpret_cast<uintptr_t>(stream),
            .first_sequence = first_sequence,
            .last_sequence = last_sequence,
            .expression = tag,
            .source = location,
        });
    }

    void handle_cuda_await_failure(const cudaError_t status,
                                   const CudaAwaitTicket& ticket,
                                   const char* tag,
                                   const SourceSite location,
                                   const uint64_t last_sequence) {
        const cudaError_t predecessor = exchange_last_cuda_check_failure(status);
        report_cuda_await_failure(CudaFailureSeed{
            .native = status,
            .predecessor = predecessor,
            .stream = ticket.stream,
            .first_sequence = ticket.first_sequence,
            .last_sequence = last_sequence,
            .expression = tag,
            .source = location,
        });
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
            completion.post_sync_error, completion.post_peek_error, true);
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
                cudaSuccess, cudaSuccess,
                disposition != CudaFailureDisposition::LogOnlyNoLatch);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): recovery, teardown, and allocator fallback
            // paths use LogOnly and must never acquire a new failure mode from
            // diagnostics themselves.
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
