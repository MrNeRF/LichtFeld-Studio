/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error_typed.hpp"

#include "core/error_reporter.hpp"

#include <format>

namespace lfs::core {
    namespace {

        [[nodiscard]] std::string cuda_typed_error_text(const cudaError_t error) {
            const char* name = cudaGetErrorName(error);
            const char* description = cudaGetErrorString(error);
            return std::format("{} ({}): {}", name ? name : "unknown CUDA error",
                               static_cast<int>(error),
                               description ? description : "description unavailable");
        }

        [[nodiscard]] SmallFields cuda_typed_fields(
            const cudaError_t status,
            const CudaCheckState& state,
            const CudaCheckCompletion& completion) {
            SmallFields fields;
            fields.add("stream", static_cast<std::int64_t>(state.stream));
            if (status != completion.effective_error) {
                fields.add("raw_status", cuda_typed_error_text(status));
            }
            if (completion.post_sync_error != cudaSuccess) {
                fields.add("post_sync_error", cuda_typed_error_text(completion.post_sync_error));
            }
            if (completion.post_peek_error != cudaSuccess) {
                fields.add("post_peek_error", cuda_typed_error_text(completion.post_peek_error));
            }
            return fields;
        }

        [[nodiscard]] Error make_cuda_failure_seed_error(const CudaFailureSeed& seed,
                                                         const std::string_view framing) {
            const bool has_predecessor = seed.predecessor != cudaSuccess;
            Error error = make_error(ErrorInit{
                .code = cuda_status_to_error_code(seed.native),
                .domain = ErrorDomain::CUDA,
                .detail = std::format("{}: {}", framing,
                                      seed.expression ? seed.expression : "<untagged>"),
                .detection = seed.source,
                .fields = SmallFields{}
                              .add("stream", static_cast<std::int64_t>(seed.stream))
                              .add("first_sequence", static_cast<std::int64_t>(seed.first_sequence))
                              .add("last_sequence", static_cast<std::int64_t>(seed.last_sequence)),
                .native = NativeError{
                    .domain = ErrorDomain::CUDA,
                    .code = static_cast<std::int64_t>(seed.native),
                    .name = cuda_typed_error_text(seed.native),
                },
            });
            if (has_predecessor) {
                Error predecessor = make_error(ErrorInit{
                    .code = cuda_status_to_error_code(seed.predecessor),
                    .domain = ErrorDomain::CUDA,
                    .detail = "a prior LFS_CUDA_LAUNCH_CHECK/LFS_CUDA_AWAIT already observed this "
                              "native status; this site may not be the true origin",
                    .detection = seed.source,
                    .native = NativeError{
                        .domain = ErrorDomain::CUDA,
                        .code = static_cast<std::int64_t>(seed.predecessor),
                        .name = cuda_typed_error_text(seed.predecessor),
                    },
                });
                error = std::move(error).with_suppressed(std::move(predecessor));
            }
            return error;
        }

    } // namespace

    ErrorCode cuda_status_to_error_code(const cudaError_t status) noexcept {
        if (status == cudaSuccess) {
            return ErrorCode::Internal; // callers only invoke this on a failure branch
        }
        if (is_cuda_unavailable_error(status)) {
            return ErrorCode::Unavailable;
        }
        switch (status) {
        case cudaErrorMemoryAllocation:
            return ErrorCode::ResourceExhausted;
        case cudaErrorIllegalAddress:
        case cudaErrorLaunchFailure:
        case cudaErrorLaunchTimeout:
        case cudaErrorECCUncorrectable:
        case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction:
        case cudaErrorMisalignedAddress:
        case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc:
        case cudaErrorAssert:
        case cudaErrorCudartUnloading:
            return ErrorCode::DeviceLost; // context-corrupting/sticky: subsystem must stop
        case cudaErrorInvalidValue:
        case cudaErrorInvalidDevice:
        case cudaErrorInvalidResourceHandle:
        case cudaErrorInvalidMemcpyDirection:
        case cudaErrorInvalidConfiguration:
            return ErrorCode::InvalidArgument;
        default:
            return ErrorCode::Internal;
        }
    }

    Error make_cuda_error(const cudaError_t status,
                          const CudaCheckState& state,
                          const CudaCheckCompletion& completion,
                          const std::string_view expression,
                          const std::string_view message,
                          const SourceSite location) {
        const cudaError_t effective = completion.effective_error;
        Error error = make_error(ErrorInit{
            .code = cuda_status_to_error_code(effective),
            .domain = ErrorDomain::CUDA,
            .detail = message.empty() ? std::string(expression)
                                      : std::format("{}: {}", expression, message),
            .detection = location,
            .fields = cuda_typed_fields(status, state, completion),
            .native = NativeError{
                .domain = ErrorDomain::CUDA,
                .code = static_cast<std::int64_t>(effective),
                .name = cuda_typed_error_text(effective),
            },
        });

        const bool has_predecessor = state.pre_call_sampled &&
                                     (state.pre_call_error != cudaSuccess ||
                                      state.pre_call_sync_error != cudaSuccess);
        if (has_predecessor) {
            const cudaError_t predecessor_status = state.pre_call_error != cudaSuccess
                                                       ? state.pre_call_error
                                                       : state.pre_call_sync_error;
            Error predecessor = make_error(ErrorInit{
                .code = cuda_status_to_error_code(predecessor_status),
                .domain = ErrorDomain::CUDA,
                .detail = "pre-existing CUDA error detected before this call; this "
                          "site is NOT the origin",
                .detection = location,
                .native = NativeError{
                    .domain = ErrorDomain::CUDA,
                    .code = static_cast<std::int64_t>(predecessor_status),
                    .name = cuda_typed_error_text(predecessor_status),
                },
            });
            error = std::move(error).with_suppressed(std::move(predecessor));
        }

        return error;
    }

    void throw_cuda_error(const cudaError_t status,
                          const CudaCheckState& state,
                          const CudaCheckCompletion& completion,
                          const std::string_view expression,
                          const std::string_view message,
                          const SourceSite location) {
        throw Exception(make_cuda_error(status, state, completion, expression, message, location));
    }

    void log_cuda_teardown_failure(const cudaError_t status,
                                   const CudaCheckState& state,
                                   const CudaCheckCompletion& completion,
                                   const std::string_view expression,
                                   const std::string_view message,
                                   const SourceSite location) noexcept {
        try {
            const Error error =
                make_cuda_error(status, state, completion, expression, message, location);
            ErrorReporter::get().report(error, ReportChannel::OwnerLog);
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): teardown-failure reporting must never
            // itself become a teardown failure; make_error/ErrorReporter::report
            // are already documented no-throw, this is defense-in-depth only.
        }
    }

    void report_cuda_launch_check_failure(const CudaFailureSeed seed) {
        throw Exception(make_cuda_failure_seed_error(seed, "kernel launch check failed"));
    }

    void report_cuda_await_failure(const CudaFailureSeed seed) {
        throw Exception(make_cuda_failure_seed_error(
            seed, std::format("observed at await (submitted range [{}, {}])",
                              seed.first_sequence, seed.last_sequence)));
    }

} // namespace lfs::core
