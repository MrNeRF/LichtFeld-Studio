/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <source_location>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

    enum class CudaFailureDisposition : uint8_t {
        Throw,
        LogOnly,
    };

    LFS_CORE_API void record_cuda_breadcrumb(const char* tag,
                                             const char* file,
                                             uint32_t line,
                                             cudaStream_t stream = nullptr) noexcept;
    LFS_CORE_API std::vector<CudaBreadcrumb> cuda_breadcrumbs_most_recent_first();
    LFS_CORE_API void clear_cuda_breadcrumbs_for_testing() noexcept;

    LFS_CORE_API bool cuda_sync_debug_enabled() noexcept;
    LFS_CORE_API void initialize_cuda_diagnostics() noexcept;

    // Unavailable-family errors are terminal for CUDA use in this process.
    LFS_CORE_API bool is_cuda_unavailable_error(cudaError_t error) noexcept;
    LFS_CORE_API bool cuda_is_unavailable() noexcept;
    LFS_CORE_API bool latch_cuda_unavailable(cudaError_t error) noexcept;

    // Resets only the unavailable latch and failure-report deduplication state.
    LFS_CORE_API void reset_cuda_diagnostics_for_testing() noexcept;
    LFS_CORE_API void reset_failure_report_dedup_for_testing() noexcept;
    LFS_CORE_API bool decide_failure_report_for_testing(
        std::string_view family,
        long long code,
        std::string_view site,
        uint64_t& out_count);
    LFS_CORE_API CudaCheckState prepare_cuda_check(
        const char* expression,
        const std::source_location& location,
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
        const std::source_location& location);
    LFS_CORE_API void finish_cuda_check(cudaError_t result,
                                        const CudaCheckState& state,
                                        const char* expression,
                                        std::string_view message,
                                        const std::source_location& location);
    LFS_CORE_API void ensure_cuda_success(
        cudaError_t result,
        const CudaCheckState& state,
        std::string_view expression,
        std::string_view message = {},
        const std::source_location& location = std::source_location::current(),
        CudaFailureDisposition disposition = CudaFailureDisposition::Throw);
    LFS_CORE_API void ensure_cuda_success(
        cudaError_t result,
        std::string_view expression,
        std::string_view message = {},
        const std::source_location& location = std::source_location::current(),
        CudaFailureDisposition disposition = CudaFailureDisposition::Throw);
    LFS_CORE_API void validate_cuda_device_pointer(
        const void* pointer,
        std::string_view name,
        const std::source_location& location = std::source_location::current());
    LFS_CORE_API void validate_cuda_device_pointer_optional(
        const void* pointer,
        std::string_view name,
        const std::source_location& location = std::source_location::current());

    LFS_CORE_API std::string capture_host_stacktrace(size_t skip_frames = 0);
    LFS_CORE_API std::string format_failure_report(
        std::string_view family,
        std::string_view contract,
        std::string_view expression,
        std::string_view message,
        const std::source_location& location,
        std::string_view stacktrace);
    LFS_CORE_API std::string format_contract_failure_report(
        std::string_view contract,
        std::string_view expression,
        std::string_view message,
        const std::source_location& location,
        std::string_view stacktrace);
    LFS_CORE_API void report_tensor_exception(
        std::string_view message,
        const std::source_location& location = std::source_location::current());

    namespace detail {

        [[noreturn]] LFS_CORE_API void assertion_failed(
            std::string_view contract,
            std::string_view expression,
            std::string_view message,
            std::source_location location);

        template <typename... Args>
        [[nodiscard]] std::string format_cuda_check_message(
            std::format_string<Args...> format,
            Args&&... args) {
            return std::format(format, std::forward<Args>(args)...);
        }

    } // namespace detail

} // namespace lfs::core

#define LFS_CUDA_DETAIL_CHECK_IMPL(call, message)                                     \
    do {                                                                              \
        ::lfs::core::record_cuda_breadcrumb(#call, __FILE__, __LINE__);               \
        const auto _lfs_cuda_state = ::lfs::core::prepare_cuda_check(                 \
            #call, std::source_location::current());                                  \
        const auto _lfs_cuda_result = (call);                                         \
        static_assert(std::is_same_v<std::remove_cv_t<decltype(_lfs_cuda_result)>,    \
                                     cudaError_t>,                                    \
                      "LFS_CUDA_CHECK requires an expression returning cudaError_t"); \
        if (_lfs_cuda_result != cudaSuccess ||                                        \
            ::lfs::core::cuda_sync_debug_enabled()) [[unlikely]] {                    \
            const auto _lfs_cuda_completion = ::lfs::core::complete_cuda_check(       \
                _lfs_cuda_result, _lfs_cuda_state);                                   \
            if (_lfs_cuda_completion.effective_error != cudaSuccess) [[unlikely]] {   \
                ::lfs::core::report_cuda_check_failure(                               \
                    _lfs_cuda_completion, _lfs_cuda_state, #call, (message),          \
                    std::source_location::current());                                 \
            }                                                                         \
        }                                                                             \
    } while (false)

#define LFS_CUDA_CHECK(call) LFS_CUDA_DETAIL_CHECK_IMPL(call, std::string_view{})

#define LFS_CUDA_CHECK_MSG(call, format, ...) \
    LFS_CUDA_DETAIL_CHECK_IMPL(               \
        call,                                 \
        ::lfs::core::detail::format_cuda_check_message((format)__VA_OPT__(, ) __VA_ARGS__))

#define LFS_CUDA_BREADCRUMB(tag)                                                  \
    do {                                                                          \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,    \
                      "LFS_CUDA_BREADCRUMB tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb((tag), __FILE__, __LINE__);           \
    } while (false)

#define LFS_CUDA_BREADCRUMB_STREAM(tag, stream_value)                                    \
    do {                                                                                 \
        static_assert(std::is_array_v<std::remove_reference_t<decltype(tag)>>,           \
                      "LFS_CUDA_BREADCRUMB_STREAM tag must be a static string literal"); \
        ::lfs::core::record_cuda_breadcrumb((tag), __FILE__, __LINE__, (stream_value));  \
    } while (false)
