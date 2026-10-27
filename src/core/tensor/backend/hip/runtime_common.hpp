/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "architecture_runtime.hpp"
#include "expression_api.h"
#include "runtime_api.h"
#include "timing_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <vector>

struct LfsHipBuffer {
    void* data = nullptr;
    size_t bytes = 0;
};
struct LfsHipReadback {
    void* host = nullptr;
    size_t capacity = 0, bytes = 0, reserved_bytes = 0;
    hipEvent_t completion = nullptr;
    hipError_t copy_error = hipSuccess;
    bool pending = false, recorded = false;
    uint32_t* execution_error_device = nullptr;
    uint32_t* execution_error_host = nullptr;
    bool execution_error_initialized = false, execution_snapshot = false;
};

struct LfsHipTiming {
    std::vector<hipEvent_t> events;
    std::vector<uint64_t> sequence;
    uint64_t next_sequence = 0;
    bool retired = false;
};

struct LfsHipContext {
    int device = 0;
    uint32_t* index_error = nullptr;
    uint32_t* index_error_host = nullptr;
    void* scalar_readback_host = nullptr;
    bool index_work_pending = false, index_error_initialized = false;
    hipStream_t stream = nullptr;
    std::unordered_set<LfsHipBuffer*> buffers;
    std::unordered_set<LfsHipExpression*> expressions;
    std::unordered_set<LfsHipTiming*> timings;
    std::unordered_set<LfsHipReadback*> readbacks;
    float* reduction_workspace = nullptr;
    void* scan_workspace = nullptr;
    size_t scan_workspace_bytes = 0;
    void* sort_workspace = nullptr;
    size_t sort_workspace_bytes = 0;
};

hipError_t lfs_hip_release_expression_internal(LfsHipContext*, LfsHipExpression*);
hipError_t lfs_hip_release_timing_internal(LfsHipContext*, LfsHipTiming*);

hipError_t lfs_hip_fill_internal(LfsHipContext*, float*, size_t, float);

namespace {
    struct HipFailure : std::exception {
        hipError_t code;
        explicit HipFailure(hipError_t value) : code(value) {}
        const char* what() const noexcept override { return hipGetErrorString(code); }
    };

    struct InvalidIndex : std::invalid_argument {
        using std::invalid_argument::invalid_argument;
    };
    struct InvalidMaskCount : std::invalid_argument {
        using std::invalid_argument::invalid_argument;
    };

    void check(hipError_t status) {
        if (status != hipSuccess)
            throw HipFailure(status);
    }

    uint32_t* index_error_flag(LfsHipContext* context) {
        if (!context->index_error_host)
            check(hipHostMalloc(reinterpret_cast<void**>(&context->index_error_host), sizeof(uint32_t), hipHostMallocDefault));
        if (!context->index_error) {
            check(hipMalloc(&context->index_error, sizeof(uint32_t)));
        }
        if (!context->index_error_initialized) {
            try {
                check(hipMemsetAsync(context->index_error, 0, sizeof(uint32_t), context->stream));
                context->index_error_initialized = true;
            } catch (...) {
                // Retain ownership if cleanup itself fails; destruction can retry.
                if (hipFree(context->index_error) == hipSuccess)
                    context->index_error = nullptr;
                throw;
            }
        }
        context->index_work_pending = true;
        return context->index_error;
    }
    void check_execution_error(uint32_t invalid) {
        if (invalid & 2u)
            throw InvalidMaskCount("HIP masked scatter source count does not match mask");
        if (invalid & 1u)
            throw InvalidIndex("HIP index is negative or outside its extent");
    }
    void synchronize_context(LfsHipContext* context) {
        check(hipStreamSynchronize(context->stream));
        if (context->index_work_pending) {
            // Queue the read on the producer stream: PAL blocking hipMemcpy can
            // return the allocation's earlier contents after atomic writes.
            // Owned staging also survives a failed synchronization.
            check(hipMemcpyAsync(context->index_error_host, context->index_error, sizeof(uint32_t), hipMemcpyDeviceToHost, context->stream));
            check(hipStreamSynchronize(context->stream));
            const uint32_t invalid = *context->index_error_host;
            // Reset on the next checked submission, so a completed fence does
            // not enqueue fresh work or leave a flag write pending at teardown.
            context->index_error_initialized = false;
            context->index_work_pending = false;
            check_execution_error(invalid);
        }
    }
    __device__ bool checked_index(int64_t index, size_t extent, uint32_t* invalid) {
        if (index >= 0 && uint64_t(index) < extent)
            return true;
        atomicOr(invalid, 1u);
        return false;
    }

    // Call only after validating both ranges against their owning buffers.
    bool views_overlap(LfsHipView a, size_t an, LfsHipView b, size_t bn) {
        return an && bn && a.buffer == b.buffer && a.offset < b.offset + bn && b.offset < a.offset + an;
    }
    template <class F>
    int boundary(LfsHipError* error, F&& function) noexcept {
        if (error)
            *error = {};
        try {
            function();
            return 0;
        } catch (const HipFailure& failure) {
            if (error) {
                error->native_code = static_cast<int32_t>(failure.code);
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
            return 1;
        } catch (const InvalidMaskCount& failure) {
            if (error) {
                error->native_code = LFS_HIP_ERROR_MASK_COUNT;
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
            return 1;
        } catch (const InvalidIndex& failure) {
            if (error) {
                error->native_code = LFS_HIP_ERROR_INVALID_INDEX;
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
            return 1;
        } catch (const std::exception& failure) {
            // LFS-CENSUS-OK(empty-catch): translate exceptions to the C ABI error/status; never swallow success.
            if (error) {
                error->native_code = -1;
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
            return 1;
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): unknown exceptions become a nonzero C ABI result with optional diagnostics.
            if (error) {
                error->native_code = -1;
                std::snprintf(error->message, sizeof(error->message), "Unknown HIP module failure");
            }
            return 1;
        }
    }

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::invalid_argument(message);
    }

    void require_disjoint(LfsHipView a, size_t an, LfsHipView b, size_t bn, const char* message) {
        require(!views_overlap(a, an, b, bn), message);
    }
    void require_compatible_views(LfsHipView a, size_t an, LfsHipView b, size_t bn, bool allow_identical, const char* message) {
        require(!views_overlap(a, an, b, bn) || (allow_identical && a.offset == b.offset && an == bn), message);
    }
    void validate_device(int ordinal) {
        int count = 0;
        check(hipGetDeviceCount(&count));
        require(ordinal >= 0 && ordinal < count, "HIP device ordinal is out of range");
    }

    struct DeviceScope {
        int previous = 0;
        explicit DeviceScope(int selected) {
            check(hipGetDevice(&previous));
            check(hipSetDevice(selected));
        }
        ~DeviceScope() { (void)hipSetDevice(previous); }
    };

    void validate(LfsHipContext* context, const LfsHipBuffer* buffer, size_t offset, size_t bytes) {
        require(context && buffer, "Missing HIP context or buffer");
        // Membership is checked before dereferencing a foreign/released handle.
        require(context->buffers.contains(const_cast<LfsHipBuffer*>(buffer)), "Buffer does not belong to this HIP context");
        require(offset <= buffer->bytes && bytes <= buffer->bytes - offset, "HIP buffer range is out of bounds");
        require(bytes == 0 || buffer->data != nullptr, "HIP buffer mapping was already released");
    }

    // Clear ownership only after the driver accepts the release.
    void release_buffer_resources(LfsHipBuffer* buffer) {
        if (buffer->data) {
            check(hipFree(buffer->data));
            buffer->data = nullptr;
        }
    }

    size_t float_bytes(size_t count) {
        require(count <= SIZE_MAX / sizeof(float), "Float element count overflows");
        return count * sizeof(float);
    }

    unsigned blocks(size_t count) {
        return static_cast<unsigned>(std::min<size_t>((count - 1) / 256 + 1, 4096));
    }

} // namespace
