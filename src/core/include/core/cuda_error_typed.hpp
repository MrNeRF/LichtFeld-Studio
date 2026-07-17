/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#if defined(__CUDACC__)
#error "core/cuda_error_typed.hpp is host-C++23-only and must never be parsed by nvcc. " \
       "It declares lfs::Error/lfs::Exception-returning CUDA adapters; CUDA translation " \
       "units use core/cuda_error.hpp's CudaCheckState/CudaCheckCompletion/ensure_cuda_success " \
       "or core/error_codes.hpp directly."
#endif

#include "core/cuda_error.hpp"
#include "core/error.hpp"
#include "core/export.hpp"
#include "core/source_site.hpp"

#include <cuda_runtime_api.h>
#include <string_view>

// lfs::core::{throw_cuda_error, log_cuda_teardown_failure} — Phase 6A typed direct-call/teardown
// adapters. See .codex_tmp/error-architecture-analysis.md Section 5.8 and
// .codex_tmp/phase-6a-cuda-spec.md Section 1 for the frozen contract. Summary:
//   * cuda_try samples pre-call CUDA state, executes the checked expression once,
//     and on failure THROWS lfs::Exception(Error{ErrorDomain::CUDA, ...}). It never
//     reports (propagate-without-logging, Section 5.1 rule 2); the owner boundary
//     that catches it reports exactly once.
//   * cuda_log_teardown is explicitly no-throw: on failure it builds the same typed
//     Error and reports it immediately through
//     ErrorReporter::get().report(error, ReportChannel::OwnerLog), because a
//     discarded teardown failure has no other owner who will ever see it. Its name
//     makes discarding a teardown failure intentional (Section 5.8).
//   * Both reuse cuda_error.hpp's existing CudaCheckState/CudaCheckCompletion
//     sticky-state sampling; a pre-existing sticky error is attached as a suppressed
//     Error on the frame (see cuda_error_typed.cpp), not silently overwritten.
namespace lfs::core {

    // Stable-taxonomy classification of a native cudaError_t. Reuses
    // is_cuda_unavailable_error's existing "unavailable" family; adds the
    // context-corrupting ("sticky", requires a fresh context/process) family as
    // DeviceLost; everything else with a recognizable invalid-argument shape as
    // InvalidArgument; default Internal. See cuda_error_typed.cpp for the full
    // switch and the rationale for each bucket.
    [[nodiscard]] LFS_CORE_API ErrorCode cuda_status_to_error_code(cudaError_t status) noexcept;

    // Builds (never throws, never reports) a fully-populated CUDA-domain Error from
    // an already-completed check. `state`/`completion` come from the same
    // prepare_cuda_check/complete_cuda_check pair LFS_CUDA_CHECK already uses.
    // If state.pre_call_sampled && (state.pre_call_error != cudaSuccess ||
    // state.pre_call_sync_error != cudaSuccess), the returned Error carries a
    // suppressed predecessor Error (via with_suppressed) describing the
    // pre-existing sticky status, and its top context frame's
    // SmallFields gains a "predecessor" field naming it — this is the "sample
    // pre-call state, attach as predecessor" rule (Section 5.8's sticky-state
    // rules; CudaFailureSeed's `predecessor` field is 6B's cross-language
    // equivalent of the same fact for launch checks, not reused here since this
    // path never crosses a .cu boundary).
    [[nodiscard]] LFS_CORE_API Error make_cuda_error(
        cudaError_t status,
        const CudaCheckState& state,
        const CudaCheckCompletion& completion,
        std::string_view expression,
        std::string_view message,
        SourceSite location);

    // TRY policy: throws lfs::Exception(make_cuda_error(...)) if status != cudaSuccess
    // or (sync-debug mode) a post-call/post-sync error is detected. Never reports.
    [[noreturn]] LFS_CORE_API void throw_cuda_error(
        cudaError_t status,
        const CudaCheckState& state,
        const CudaCheckCompletion& completion,
        std::string_view expression,
        std::string_view message,
        SourceSite location);

    // LOG_TEARDOWN policy: builds the Error and reports it through
    // ErrorReporter::get().report(error, ReportChannel::OwnerLog). Never throws.
    LFS_CORE_API void log_cuda_teardown_failure(
        cudaError_t status,
        const CudaCheckState& state,
        const CudaCheckCompletion& completion,
        std::string_view expression,
        std::string_view message,
        SourceSite location) noexcept;

} // namespace lfs::core

// Mirrors LFS_CUDA_DETAIL_CHECK_IMPL's capture shape (cuda_error.hpp:141-161):
// record a breadcrumb, sample pre-call state once, evaluate `call` exactly once,
// then dispatch on the frozen policy. `stream` is a cudaStream_t (may be nullptr
// for the legacy/default stream) used only for breadcrumb/state attribution — it
// is NOT re-synchronized or re-queried beyond what prepare_cuda_check/
// complete_cuda_check already do.
#define LFS_CUDA_DETAIL_TYPED_IMPL(call, stream, message, dispatch)                      \
    do {                                                                                 \
        const auto _lfs_cuda_typed_site = LFS_SOURCE_SITE_CURRENT();                     \
        const cudaStream_t _lfs_cuda_typed_stream = (stream);                            \
        ::lfs::core::record_cuda_breadcrumb(#call, __FILE__, __LINE__,                   \
                                            _lfs_cuda_typed_stream);                     \
        const auto _lfs_cuda_typed_state = ::lfs::core::prepare_cuda_check(              \
            #call, _lfs_cuda_typed_site, _lfs_cuda_typed_stream);                        \
        const auto _lfs_cuda_typed_result = (call);                                      \
        static_assert(std::is_same_v<std::remove_cv_t<decltype(_lfs_cuda_typed_result)>, \
                                     cudaError_t>,                                       \
                      "LFS_CUDA_TRY/LFS_CUDA_LOG_TEARDOWN require an expression "        \
                      "returning cudaError_t");                                          \
        const auto _lfs_cuda_typed_completion = ::lfs::core::complete_cuda_check(        \
            _lfs_cuda_typed_result, _lfs_cuda_typed_state);                              \
        if (_lfs_cuda_typed_completion.effective_error != cudaSuccess) [[unlikely]] {    \
            dispatch(_lfs_cuda_typed_result, _lfs_cuda_typed_state,                      \
                     _lfs_cuda_typed_completion, #call, (message),                       \
                     _lfs_cuda_typed_site);                                              \
        }                                                                                \
    } while (false)

#define LFS_CUDA_TRY_DISPATCH(result, state, completion, expr, message, site) \
    ::lfs::core::throw_cuda_error(result, state, completion, expr, message, site)

#define LFS_CUDA_LOG_TEARDOWN_DISPATCH(result, state, completion, expr, message, site) \
    ::lfs::core::log_cuda_teardown_failure(result, state, completion, expr, message, site)

// Direct-call TRY: `call` is a single cudaXxx(...) expression, `stream` is the
// cudaStream_t it was issued on (nullptr for legacy/default), `message` is a
// std::string_view (use lfs::core::detail::format_cuda_safe(...) for formatted
// context, matching LFS_CUDA_CHECK_MSG's existing convention).
#define LFS_CUDA_TRY(call, stream, message) \
    LFS_CUDA_DETAIL_TYPED_IMPL(call, stream, message, LFS_CUDA_TRY_DISPATCH)

// Teardown/reset-for-reuse LOG: same shape, never throws. Use at any destructor,
// shutdown(), cleanup()-for-reuse, or "log and fall back to a safe default"
// construction site (Section 5.1 rule 7's log-and-continue policy cell) — the
// macro name documents *why* the status is discarded, not merely that it is.
#define LFS_CUDA_LOG_TEARDOWN(call, stream, message) \
    LFS_CUDA_DETAIL_TYPED_IMPL(call, stream, message, LFS_CUDA_LOG_TEARDOWN_DISPATCH)
