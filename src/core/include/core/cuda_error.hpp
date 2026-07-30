/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/cuda_safe_format.hpp"
#include "core/export.hpp"
#include "core/source_site.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace lfs::core {

    inline constexpr size_t CUDA_BREADCRUMB_CAPACITY = 64;

    struct LFS_CORE_API CudaBreadcrumb {
        uint64_t sequence = 0;
        const char* tag = nullptr;
        const char* file = nullptr;
        uint32_t line = 0;
        uintptr_t stream = 0;
        uint64_t thread_id = 0;
        uint64_t a0 = 0;
        uint64_t a1 = 0;
        uint64_t a2 = 0;
    };

    struct LFS_CORE_API CudaCheckState {
        cudaError_t pre_call_error = cudaSuccess;
        cudaError_t pre_call_sync_error = cudaSuccess;
        uintptr_t stream = 0;
        bool pre_call_sampled = false;
    };

    struct LFS_CORE_API CudaCheckCompletion {
        cudaError_t effective_error = cudaSuccess;
        cudaError_t post_sync_error = cudaSuccess;
        cudaError_t post_peek_error = cudaSuccess;
    };

    // Trivially-copyable, non-owning cold-path payload for a detected CUDA
    // launch-check or async-await failure. FROZEN VERBATIM shape (Section
    // 7.2.1 of .codex_tmp/error-architecture-analysis.md) — do not add,
    // remove, reorder, or rename fields. `expression`/`source`'s file/
    // function pointers must refer to static storage (string literals,
    // __builtin_FILE()-backed pointers) — never a temporary or
    // heap-allocated string. Has no default constructor (SourceSite has
    // none); always aggregate-initialize every field, never
    // default-construct a bare `CudaFailureSeed seed;`.
    struct LFS_CORE_API CudaFailureSeed {
        cudaError_t native;
        cudaError_t predecessor;
        std::uintptr_t stream;
        std::uint64_t first_sequence;
        std::uint64_t last_sequence;
        const char* expression;
        SourceSite source;
    };
    static_assert(std::is_trivially_copyable_v<CudaFailureSeed>,
                  "CudaFailureSeed crosses the nvcc/host boundary by value; "
                  "it must stay trivially copyable");

    // A submitted-but-not-yet-awaited CUDA operation range. Value type,
    // single-use by convention: obtained from cuda_record_range(), consumed
    // by exactly one LFS_CUDA_AWAIT. Trivially copyable, CUDA-safe (usable
    // from .cu and .cpp alike) — it carries no owning state, only a stream
    // handle, the breadcrumb sequence at record time, and a static tag.
    struct LFS_CORE_API CudaAwaitTicket {
        std::uintptr_t stream;
        std::uint64_t first_sequence;
        const char* operation_tag;
    };
    static_assert(std::is_trivially_copyable_v<CudaAwaitTicket>,
                  "CudaAwaitTicket crosses the nvcc/host boundary by value; "
                  "it must stay trivially copyable");

    enum class CudaFailureDisposition : uint8_t {
        Throw,
        LogOnly,
        LogOnlyNoLatch,
    };

    // Returns the assigned monotonic sequence number, so LFS_CUDA_LAUNCH_CHECK/
    // LFS_CUDA_AWAIT can name a breadcrumb range without a second call.
    LFS_CORE_API uint64_t record_cuda_breadcrumb(const char* tag,
                                                 const char* file,
                                                 uint32_t line,
                                                 cudaStream_t stream = nullptr) noexcept;
    LFS_CORE_API uint64_t record_cuda_breadcrumb(const char* tag,
                                                 const char* file,
                                                 uint32_t line,
                                                 cudaStream_t stream,
                                                 uint64_t a0,
                                                 uint64_t a1 = 0,
                                                 uint64_t a2 = 0) noexcept;
    LFS_CORE_API std::vector<CudaBreadcrumb> cuda_breadcrumbs_most_recent_first();
    LFS_CORE_API void clear_cuda_breadcrumbs_for_testing() noexcept;

    LFS_CORE_API void register_cuda_address_range(
        const void* base, std::size_t bytes, std::string label);
    LFS_CORE_API void unregister_cuda_address_range(const void* base);

    // The single runtime diagnostics control, generalized from the boolean
    // LFS_CUDA_SYNC_DEBUG into a comma-separated mode list. See
    // parse_diagnostic_modes() for the parsing contract.
    enum class DiagnosticMode : unsigned {
        CudaSync = 1u << 0,
        DeviceTrap = 1u << 1,
        VkFatal = 1u << 2,
    };

    struct LFS_CORE_API ParsedDiagnosticModes {
        unsigned modes = 0;
        bool unknown_tokens_present = false;
        std::string unknown_tokens;
        bool legacy_alias_present = false;
    };

    // Pure string -> bitmask parser: no getenv, no caching, no logging.
    // sync_debug_value/vk_validation_fatal_value are the raw LFS_CUDA_SYNC_DEBUG
    // and deprecated LFS_VK_VALIDATION_FATAL values (nullopt when unset).
    [[nodiscard]] LFS_CORE_API ParsedDiagnosticModes parse_diagnostic_modes(
        std::optional<std::string_view> sync_debug_value,
        std::optional<std::string_view> vk_validation_fatal_value) noexcept;

    // Reads and parses the environment exactly once into an immutable bitmask,
    // logging any deprecation/unknown-token warnings on first use.
    [[nodiscard]] LFS_CORE_API unsigned diagnostic_modes() noexcept;
    [[nodiscard]] LFS_CORE_API bool diagnostic_mode_enabled(DiagnosticMode mode) noexcept;

    LFS_CORE_API bool cuda_sync_debug_enabled() noexcept;
    LFS_CORE_API void initialize_cuda_diagnostics() noexcept;

    // Unavailable-family errors are terminal for CUDA use in this process.
    LFS_CORE_API bool is_cuda_unavailable_error(cudaError_t error) noexcept;
    LFS_CORE_API bool cuda_is_unavailable() noexcept;
    LFS_CORE_API bool latch_cuda_unavailable(cudaError_t error) noexcept;

    // Resets the unavailable latch, launch/await attribution state, address-range
    // registry, and failure-report deduplication state. Breadcrumb storage is reset separately.
    LFS_CORE_API void reset_cuda_diagnostics_for_testing() noexcept;

    // Current breadcrumb sequence counter without recording a new entry.
    // CUDA-safe, allocation-free, one relaxed atomic load.
    [[nodiscard]] LFS_CORE_API uint64_t current_cuda_breadcrumb_sequence() noexcept;

    // Most-recent-overwrite (not first-wins) process-wide latch of the last
    // native status observed by any LFS_CUDA_LAUNCH_CHECK/LFS_CUDA_AWAIT cold
    // path. Returns the value that was recorded before this call (cudaSuccess
    // if none since the last reset_cuda_diagnostics_for_testing()), then
    // stores `native`. Called only from the two cold paths below — never on
    // any fast/success path.
    LFS_CORE_API cudaError_t exchange_last_cuda_check_failure(cudaError_t native) noexcept;

    // Opens a submitted-range ticket for a later LFS_CUDA_AWAIT. Does not
    // wait, allocate, or synchronize. `operation_tag` must be a static string
    // literal.
    [[nodiscard]] LFS_CORE_API CudaAwaitTicket cuda_record_range(
        cudaStream_t stream, const char* operation_tag) noexcept;

    // Cold-path dispatch for LFS_CUDA_LAUNCH_CHECK. May return normally (the
    // sync-debug-only, no-real-failure case); calls the [[noreturn]] bridge
    // below otherwise. Declared here (CUDA-safe) so the macro can call it
    // from a .cu TU; defined in cuda_error.cpp (also CUDA-safe — it never
    // constructs lfs::Error, only a CudaFailureSeed).
    LFS_CORE_API void handle_cuda_launch_check_slow_path(
        cudaError_t status,
        cudaStream_t stream,
        const char* tag,
        SourceSite location,
        uint64_t last_sequence);

    // Cold-path dispatch for LFS_CUDA_AWAIT. Always builds and forwards a
    // seed (unlike the launch-check path, AWAIT's cold path is entered only
    // on a genuine non-success status — no sync-debug-only entry exists for
    // it, since it wraps a real expression that already ran).
    LFS_CORE_API void handle_cuda_await_failure(
        cudaError_t status,
        const CudaAwaitTicket& ticket,
        const char* tag,
        SourceSite location,
        uint64_t last_sequence);

    // The two cold bridges. CUDA-safe declarations (CudaFailureSeed is the
    // only host-visible type in the signature); their C++23 .cpp definitions
    // alone construct/throw Exception(Error) — see cuda_error_typed.cpp §1.2.
    // Never call these directly from a .cu file; only
    // handle_cuda_launch_check_slow_path/handle_cuda_await_failure do.
    [[noreturn]] LFS_CORE_API void report_cuda_launch_check_failure(CudaFailureSeed seed);
    [[noreturn]] LFS_CORE_API void report_cuda_await_failure(CudaFailureSeed seed);

    LFS_CORE_API CudaCheckState prepare_cuda_check(
        const char* expression,
        SourceSite location,
        cudaStream_t stream = nullptr) noexcept;

    // This sample must be taken before the guarded call for valid attribution.
    LFS_CORE_API CudaCheckState sample_cuda_pre_call_state(
        cudaStream_t stream = nullptr) noexcept;
    LFS_CORE_API CudaCheckCompletion complete_cuda_check(
        cudaError_t result,
        const CudaCheckState& state) noexcept;
    [[noreturn]] LFS_CORE_API void report_cuda_check_failure(
        const CudaCheckCompletion& completion,
        const CudaCheckState& state,
        const char* expression,
        std::string_view message,
        SourceSite location);
    LFS_CORE_API void finish_cuda_check(cudaError_t result,
                                        const CudaCheckState& state,
                                        const char* expression,
                                        std::string_view message,
                                        SourceSite location);
    LFS_CORE_API void ensure_cuda_success(
        cudaError_t result,
        const CudaCheckState& state,
        std::string_view expression,
        std::string_view message,
        SourceSite location,
        CudaFailureDisposition disposition = CudaFailureDisposition::Throw);
    LFS_CORE_API void ensure_cuda_success(
        cudaError_t result,
        std::string_view expression,
        std::string_view message,
        SourceSite location,
        CudaFailureDisposition disposition = CudaFailureDisposition::Throw);
    LFS_CORE_API void validate_cuda_device_pointer(
        const void* pointer,
        std::string_view name,
        SourceSite location);
    LFS_CORE_API void validate_cuda_device_pointer_optional(
        const void* pointer,
        std::string_view name,
        SourceSite location);

} // namespace lfs::core

#define LFS_CUDA_DETAIL_CHECK_IMPL(call, stream_value, arg0, arg1, arg2, message)     \
    do {                                                                              \
        const auto _lfs_cuda_site = LFS_SOURCE_SITE_CURRENT();                        \
        const cudaStream_t _lfs_cuda_stream = (stream_value);                         \
        const auto _lfs_cuda_state = ::lfs::core::prepare_cuda_check(                 \
            #call, _lfs_cuda_site, _lfs_cuda_stream);                                 \
        const auto _lfs_cuda_result = (call);                                         \
        static_assert(std::is_same_v<std::remove_cv_t<decltype(_lfs_cuda_result)>,    \
                                     cudaError_t>,                                    \
                      "LFS_CUDA_CHECK requires an expression returning cudaError_t"); \
        const uint64_t _lfs_cuda_arg0 = static_cast<uint64_t>(arg0);                  \
        const uint64_t _lfs_cuda_arg1 = static_cast<uint64_t>(arg1);                  \
        const uint64_t _lfs_cuda_arg2 = static_cast<uint64_t>(arg2);                  \
        ::lfs::core::record_cuda_breadcrumb(                                          \
            #call, __FILE__, __LINE__, _lfs_cuda_stream,                              \
            _lfs_cuda_arg0, _lfs_cuda_arg1, _lfs_cuda_arg2);                          \
        if (_lfs_cuda_result != cudaSuccess ||                                        \
            ::lfs::core::cuda_sync_debug_enabled()) [[unlikely]] {                    \
            const auto _lfs_cuda_completion = ::lfs::core::complete_cuda_check(       \
                _lfs_cuda_result, _lfs_cuda_state);                                   \
            if (_lfs_cuda_completion.effective_error != cudaSuccess) [[unlikely]] {   \
                ::lfs::core::report_cuda_check_failure(                               \
                    _lfs_cuda_completion, _lfs_cuda_state, #call, (message),          \
                    _lfs_cuda_site);                                                  \
            }                                                                         \
        }                                                                             \
    } while (false)

#define LFS_CUDA_CHECK(call) \
    LFS_CUDA_DETAIL_CHECK_IMPL(call, nullptr, 0, 0, 0, std::string_view{})

#define LFS_CUDA_CHECK_STREAM(call, stream_value) \
    LFS_CUDA_DETAIL_CHECK_IMPL(call, stream_value, 0, 0, 0, std::string_view{})

#define LFS_CUDA_CHECK_ARGS(call, arg0, arg1, arg2) \
    LFS_CUDA_DETAIL_CHECK_IMPL(call, nullptr, arg0, arg1, arg2, std::string_view{})

#define LFS_CUDA_CHECK_STREAM_ARGS(call, stream_value, arg0, arg1, arg2) \
    LFS_CUDA_DETAIL_CHECK_IMPL(call, stream_value, arg0, arg1, arg2, std::string_view{})

#define LFS_CUDA_CHECK_MSG(call, ...) \
    LFS_CUDA_DETAIL_CHECK_IMPL(       \
        call, nullptr, 0, 0, 0,       \
        ::lfs::core::detail::format_cuda_safe(__VA_ARGS__))

#define LFS_CUDA_CHECK_MSG_STREAM(call, stream_value, ...) \
    LFS_CUDA_DETAIL_CHECK_IMPL(                            \
        call, stream_value, 0, 0, 0,                       \
        ::lfs::core::detail::format_cuda_safe(__VA_ARGS__))

#define LFS_CUDA_CHECK_MSG_ARGS(call, arg0, arg1, arg2, ...) \
    LFS_CUDA_DETAIL_CHECK_IMPL(                              \
        call, nullptr, arg0, arg1, arg2,                     \
        ::lfs::core::detail::format_cuda_safe(__VA_ARGS__))

#define LFS_CUDA_CHECK_MSG_STREAM_ARGS(call, stream_value, arg0, arg1, arg2, ...) \
    LFS_CUDA_DETAIL_CHECK_IMPL(                                                   \
        call, stream_value, arg0, arg1, arg2,                                     \
        ::lfs::core::detail::format_cuda_safe(__VA_ARGS__))

#define LFS_ENSURE_CUDA_SUCCESS(result, expression)   \
    ::lfs::core::ensure_cuda_success(                 \
        (result), (expression), ::std::string_view{}, \
        LFS_SOURCE_SITE_CURRENT())

#define LFS_ENSURE_CUDA_SUCCESS_MSG(result, expression, message) \
    ::lfs::core::ensure_cuda_success(                            \
        (result), (expression), (message), LFS_SOURCE_SITE_CURRENT())

#define LFS_ENSURE_CUDA_SUCCESS_STATE(result, state, expression, message) \
    ::lfs::core::ensure_cuda_success(                                     \
        (result), (state), (expression), (message),                       \
        LFS_SOURCE_SITE_CURRENT())

#define LFS_VALIDATE_CUDA_DEVICE_POINTER(pointer, name) \
    ::lfs::core::validate_cuda_device_pointer(          \
        (pointer), (name), LFS_SOURCE_SITE_CURRENT())

#define LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(pointer, name) \
    ::lfs::core::validate_cuda_device_pointer_optional(          \
        (pointer), (name), LFS_SOURCE_SITE_CURRENT())

#define LFS_CUDA_BREADCRUMB(tag)                                                  \
    do {                                                                          \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,    \
                      "LFS_CUDA_BREADCRUMB tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb((tag), __FILE__, __LINE__);           \
    } while (false)

#define LFS_CUDA_BREADCRUMB_ARGS(tag, arg0, arg1, arg2)                                \
    do {                                                                               \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,         \
                      "LFS_CUDA_BREADCRUMB_ARGS tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb(                                           \
            (tag), __FILE__, __LINE__, nullptr,                                        \
            static_cast<uint64_t>(arg0),                                               \
            static_cast<uint64_t>(arg1),                                               \
            static_cast<uint64_t>(arg2));                                              \
    } while (false)

#define LFS_CUDA_BREADCRUMB_STREAM(tag, stream_value)                                    \
    do {                                                                                 \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,           \
                      "LFS_CUDA_BREADCRUMB_STREAM tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb((tag), __FILE__, __LINE__, (stream_value));  \
    } while (false)

#define LFS_CUDA_BREADCRUMB_STREAM_ARGS(tag, stream_value, arg0, arg1, arg2)                  \
    do {                                                                                      \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,                \
                      "LFS_CUDA_BREADCRUMB_STREAM_ARGS tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb(                                                  \
            (tag), __FILE__, __LINE__, (stream_value),                                        \
            static_cast<uint64_t>(arg0),                                                      \
            static_cast<uint64_t>(arg1),                                                      \
            static_cast<uint64_t>(arg2));                                                     \
    } while (false)

// Every public launcher or logical phase performs this immediately after its
// kernel<<<...>>>() launch, on the actual stream: ONE cudaPeekAtLastError,
// no sync, no allocation, no formatting on the success path (Section 5.12).
// `tag` must be a static string literal. Under LFS_CUDA_SYNC_DEBUG=cuda-sync
// the slow path additionally synchronizes to catch async faults early —
// production behavior does not change.
#define LFS_CUDA_LAUNCH_CHECK(stream, tag)                                          \
    do {                                                                            \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,      \
                      "LFS_CUDA_LAUNCH_CHECK tag must be a static string literal"); \
        const cudaStream_t _lfs_launch_stream = (stream);                           \
        const uint64_t _lfs_launch_seq = ::lfs::core::record_cuda_breadcrumb(       \
            (tag), __FILE__, __LINE__, _lfs_launch_stream);                         \
        const cudaError_t _lfs_launch_status = cudaPeekAtLastError();               \
        if (_lfs_launch_status != cudaSuccess ||                                    \
            ::lfs::core::cuda_sync_debug_enabled()) [[unlikely]] {                  \
            ::lfs::core::handle_cuda_launch_check_slow_path(                        \
                _lfs_launch_status, _lfs_launch_stream, (tag),                      \
                LFS_SOURCE_SITE_CURRENT(), _lfs_launch_seq);                        \
        }                                                                           \
    } while (false)

// `call` is the EXISTING wait/completion expression the algorithm already
// performs (cudaStreamWaitEvent/cudaEventSynchronize/cudaStreamSynchronize/
// etc., returning cudaError_t), evaluated exactly once — LFS_CUDA_AWAIT never
// adds a wait that was not already there. `ticket` came from
// cuda_record_range(); `tag` describes THIS await site (distinct from the
// ticket's own submission-site tag) and must be a static string literal.
#define LFS_CUDA_AWAIT(ticket, call, tag)                                             \
    do {                                                                              \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,        \
                      "LFS_CUDA_AWAIT tag must be a static string literal");          \
        const ::lfs::core::CudaAwaitTicket _lfs_await_ticket = (ticket);              \
        const cudaError_t _lfs_await_status = (call);                                 \
        static_assert(std::is_same_v<std::remove_cv_t<decltype(_lfs_await_status)>,   \
                                     cudaError_t>,                                    \
                      "LFS_CUDA_AWAIT requires an expression returning cudaError_t"); \
        const uint64_t _lfs_await_seq = ::lfs::core::record_cuda_breadcrumb(          \
            (tag), __FILE__, __LINE__,                                                \
            reinterpret_cast<cudaStream_t>(_lfs_await_ticket.stream));                \
        if (_lfs_await_status != cudaSuccess) [[unlikely]] {                          \
            ::lfs::core::handle_cuda_await_failure(                                   \
                _lfs_await_status, _lfs_await_ticket, (tag),                          \
                LFS_SOURCE_SITE_CURRENT(), _lfs_await_seq);                           \
        }                                                                             \
    } while (false)
