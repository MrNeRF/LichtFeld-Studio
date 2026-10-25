/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "architecture_runtime.hpp"
#include "expression_api.h"
#include "hip_lens.hpp"
#include "hip_pointwise_functors.hpp"
#include "runtime_api.h"
#include "timing_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <core/environment_math.hpp>
#include <cstdio>
#include <cstring>
#include <exception>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
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
};

struct LfsHipTiming {
    std::vector<hipEvent_t> events;
    std::vector<uint64_t> sequence;
    uint64_t next_sequence = 0;
    bool retired = false;
};

struct LfsHipContext {
    int device = 0;
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

namespace {
    void release_expression_native(LfsHipContext*, LfsHipExpression*);
    void release_timing_native(LfsHipContext*, LfsHipTiming*);
    struct HipFailure : std::exception {
        hipError_t code;
        explicit HipFailure(hipError_t value) : code(value) {}
        const char* what() const noexcept override { return hipGetErrorString(code); }
    };

    struct InvalidIndex : std::invalid_argument {
        using std::invalid_argument::invalid_argument;
    };

    void check(hipError_t status) {
        if (status != hipSuccess)
            throw HipFailure(status);
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
        } catch (const InvalidIndex& failure) {
            if (error) {
                error->native_code = LFS_HIP_ERROR_INVALID_INDEX;
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
        } catch (const std::exception& failure) {
            if (error) {
                error->native_code = -1;
                std::snprintf(error->message, sizeof(error->message), "%s", failure.what());
            }
        } catch (...) {
            if (error) {
                error->native_code = -1;
                std::snprintf(error->message, sizeof(error->message), "Unknown HIP module failure");
            }
        }
        return 1;
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

    __global__ void fill(float* output, size_t count, float value) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = value;
    }

    __global__ void affine(const float* input, float* output, size_t count, float scale, float bias) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = input[i] * scale + bias;
    }

    template <unsigned Mode, class In, class Out, class Op>
    __global__ void pointwise_kernel(const In* a, const In* b, Out* out, size_t count, float scalar, Op op) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float x = static_cast<float>(a[i]);
            float value;
            if constexpr (Mode == 0)
                value = op(x);
            else if constexpr (Mode == 1)
                value = op(x, scalar);
            else
                value = op(x, static_cast<float>(b[i]));
            if constexpr (std::is_same_v<Out, __half>)
                out[i] = __float2half_rn(value);
            else
                out[i] = static_cast<Out>(value);
        }
    }
    template <unsigned Mode, class Out, class In = float, class Op>
    void launch_pointwise(LfsHipContext* context, LfsHipView a, LfsHipView b, LfsHipView out, size_t count, float scalar, Op op) {
        require(count <= SIZE_MAX / sizeof(In) && count <= SIZE_MAX / sizeof(Out), "Pointwise element count overflows");
        const size_t input_bytes = count * sizeof(In);
        validate(context, a.buffer, a.offset, input_bytes);
        validate(context, out.buffer, out.offset, count * sizeof(Out));
        require(a.offset % alignof(In) == 0 && out.offset % alignof(Out) == 0, "Unaligned pointwise view");
        if constexpr (Mode == 2) {
            validate(context, b.buffer, b.offset, input_bytes);
            require(b.offset % alignof(In) == 0, "Unaligned pointwise rhs");
        }
        require_compatible_views(a, input_bytes, out, count * sizeof(Out), sizeof(Out) == sizeof(In), "Overlapping pointwise views require identical typed ranges");
        if constexpr (Mode == 2)
            require_compatible_views(b, input_bytes, out, count * sizeof(Out), sizeof(Out) == sizeof(In), "Overlapping pointwise views require identical typed ranges");
        if (!count)
            return;
        auto* lhs = reinterpret_cast<const In*>(static_cast<const char*>(a.buffer->data) + a.offset);
        const In* rhs = nullptr;
        if constexpr (Mode == 2)
            rhs = reinterpret_cast<const In*>(static_cast<const char*>(b.buffer->data) + b.offset);
        auto* output = reinterpret_cast<Out*>(static_cast<char*>(out.buffer->data) + out.offset);
        hipLaunchKernelGGL((pointwise_kernel<Mode, In, Out, Op>), dim3(blocks(count)), dim3(256), 0, context->stream, lhs, rhs, output, count, scalar, op);
        check(hipGetLastError());
    }
    __global__ void arange_kernel(float* out, size_t n, float start, float step) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
            out[i] = start + static_cast<float>(i) * step;
    }

} // namespace

extern "C" {
uint32_t lfs_hip_runtime_abi(void) { return LFS_HIP_RUNTIME_ABI_VERSION; }

int lfs_hip_device_count(int32_t* count, LfsHipError* error) {
    return boundary(error, [&] {
        require(count, "Missing device count output");
        *count = 0;
        int value = 0;
        check(hipGetDeviceCount(&value));
        *count = value;
    });
}

int lfs_hip_device_info(int32_t ordinal, LfsHipDevice* info, LfsHipError* error) {
    return boundary(error, [&] {
        require(info, "Missing device info output");
        *info = {};
        validate_device(ordinal);
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal));
        check(hipDeviceGetPCIBusId(info->pci_bus_id, sizeof(info->pci_bus_id), ordinal));
        std::snprintf(info->name, sizeof(info->name), "%s", properties.name);
        std::snprintf(info->architecture, sizeof(info->architecture), "%s", properties.gcnArchName);
        info->total_bytes = properties.totalGlobalMem;
        info->warp_size = properties.warpSize;
    });
}

int lfs_hip_context_create(int32_t ordinal, LfsHipContext** context, LfsHipError* error) {
    return boundary(error, [&] {
        require(context, "Missing context output");
        *context = nullptr;
        validate_device(ordinal);
        DeviceScope device(ordinal);
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal));
        const std::string_view architecture(properties.gcnArchName);
        require(lfs_hip_has_architecture(LFS_HIP_COMPILED_ARCH, architecture),
                "HIP Tensor module was compiled for a different GPU architecture");
        auto result = std::make_unique<LfsHipContext>();
        result->device = ordinal;
        check(hipStreamCreateWithFlags(&result->stream, hipStreamNonBlocking));
        *context = result.release();
    });
}

int lfs_hip_context_synchronize(LfsHipContext* context, LfsHipError* error) {
    return boundary(error, [&] {
        require(context, "Missing HIP context");
        DeviceScope device(context->device);
        check(hipStreamSynchronize(context->stream));
    });
}

int lfs_hip_buffer_allocate(LfsHipContext* context, size_t bytes, LfsHipBuffer** buffer, LfsHipError* error) {
    return boundary(error, [&] {
        require(buffer, "Missing buffer output");
        *buffer = nullptr;
        require(context && bytes, "Missing context or zero allocation size");
        DeviceScope device(context->device);
        auto result = std::make_unique<LfsHipBuffer>();
        result->bytes = bytes;
        // Complete fallible host bookkeeping before taking GPU ownership.
        context->buffers.insert(result.get());
        try {
            check(hipMalloc(&result->data, bytes));
        } catch (...) {
            context->buffers.erase(result.get());
            throw;
        }
        *buffer = result.release();
    });
}

int lfs_hip_buffer_release(LfsHipContext* context, LfsHipBuffer* buffer, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, buffer, 0, 0);
        DeviceScope device(context->device);
        check(hipStreamSynchronize(context->stream));
        release_buffer_resources(buffer);
        context->buffers.erase(buffer);
        delete buffer;
    });
}

int lfs_hip_context_destroy(LfsHipContext* context, LfsHipError* error) {
    return boundary(error, [&] {
        require(context, "Missing HIP context");
        require(context->readbacks.empty(), "Destroy HIP readback staging before its context");
        DeviceScope device(context->device);
        for (;;) {
            const auto retired = std::find_if(context->timings.begin(), context->timings.end(), [](const auto* pool) { return pool->retired; });
            if (retired == context->timings.end())
                break;
            release_timing_native(context, *retired);
        }
        require(context->timings.empty(), "Destroy HIP timing pools before their context");
        // On a runtime failure, retain the handle and unfreed allocations
        // for explicit recovery; never claim that queued work was drained.
        check(hipStreamSynchronize(context->stream));
        while (!context->expressions.empty())
            release_expression_native(context, *context->expressions.begin());
        while (!context->buffers.empty()) {
            auto* buffer = *context->buffers.begin();
            release_buffer_resources(buffer);
            context->buffers.erase(buffer);
            delete buffer;
        }
        if (context->reduction_workspace) {
            check(hipFree(context->reduction_workspace));
            context->reduction_workspace = nullptr;
        }
        if (context->scan_workspace) {
            check(hipFree(context->scan_workspace));
            context->scan_workspace = nullptr;
            context->scan_workspace_bytes = 0;
        }
        if (context->sort_workspace) {
            check(hipFree(context->sort_workspace));
            context->sort_workspace = nullptr;
            context->sort_workspace_bytes = 0;
        }
        check(hipStreamDestroy(context->stream));
        delete context;
    });
}

int lfs_hip_upload(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, const void* host, size_t bytes, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, buffer, offset, bytes);
        require(host || !bytes, "Missing upload source");
        DeviceScope device(context->device);
        // PAL Windows can dispatch a pageable copy ahead of earlier compute
        // after an event marker when pointers are embedded in kernel structs
        // (AMD rocm-systems #12571). This API is blocking already: complete the
        // producer before enqueueing the copy, not only after the copy.
        if (bytes)
            check(hipStreamSynchronize(context->stream));
        if (bytes)
            check(hipMemcpyAsync(static_cast<char*>(buffer->data) + offset, host, bytes, hipMemcpyHostToDevice, context->stream));
        check(hipStreamSynchronize(context->stream));
    });
}

int lfs_hip_download(LfsHipContext* context, const LfsHipBuffer* buffer, size_t offset, void* host, size_t bytes, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, buffer, offset, bytes);
        require(host || !bytes, "Missing download destination");
        DeviceScope device(context->device);
        // PAL Windows can dispatch a pageable copy ahead of earlier compute
        // after an event marker when pointers are embedded in kernel structs
        // (AMD rocm-systems #12571). This API is blocking already: complete the
        // producer before enqueueing the copy, not only after the copy.
        if (bytes)
            check(hipStreamSynchronize(context->stream));
        if (bytes)
            check(hipMemcpyAsync(host, static_cast<const char*>(buffer->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
    });
}

namespace {
    void validate_readback(LfsHipContext* context, LfsHipReadback* staging) {
        require(context && staging && context->readbacks.contains(staging), "Readback does not belong to this HIP context");
    }
    void ensure_staging_capacity(LfsHipReadback* staging, size_t bytes) {
        if (bytes <= staging->capacity)
            return;
        if (staging->host) {
            check(hipHostFree(staging->host));
            staging->host = nullptr;
            staging->capacity = 0;
            staging->bytes = 0;
            staging->reserved_bytes = 0;
        }
        void* replacement = nullptr;
        check(hipHostMalloc(&replacement, bytes, hipHostMallocDefault));
        staging->host = replacement;
        staging->capacity = bytes;
    }
} // namespace
int lfs_hip_readback_create(LfsHipContext* context, LfsHipReadback** output, LfsHipError* error) {
    return boundary(error, [&] {
        require(output, "Missing HIP readback output");
        *output = nullptr;
        require(context, "Missing HIP readback context");
        DeviceScope device(context->device);
        auto result = std::make_unique<LfsHipReadback>();
        context->readbacks.insert(result.get());
        try {
            check(hipEventCreateWithFlags(&result->completion, hipEventDisableTiming));
        } catch (...) {
            context->readbacks.erase(result.get());
            throw;
        }
        *output = result.release();
    });
}
int lfs_hip_readback_enqueue(LfsHipContext* context, LfsHipReadback* staging, const LfsHipBuffer* source, size_t offset, size_t bytes, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        require(!staging->pending, "HIP readback staging is already in use");
        if (bytes || source)
            validate(context, source, offset, bytes);
        else
            require(offset == 0, "Invalid empty HIP readback offset");
        DeviceScope device(context->device);
        ensure_staging_capacity(staging, bytes);
        staging->bytes = bytes;
        staging->reserved_bytes = 0;
        staging->recorded = false;
        staging->copy_error = hipSuccess;
        // Even an event-record failure must leave the copy owned until drained.
        staging->pending = true;
        if (bytes) {
            staging->copy_error = hipMemcpyAsync(staging->host, static_cast<const char*>(source->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream);
            check(staging->copy_error);
        }
        check(hipEventRecord(staging->completion, context->stream));
        staging->recorded = true;
    });
}
int lfs_hip_staged_upload(LfsHipContext* context, LfsHipReadback* staging, LfsHipBuffer* destination, size_t offset, const void* source, size_t bytes, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        require(!staging->pending, "HIP upload staging is already in use");
        require(source || !bytes, "Missing HIP upload source");
        if (bytes || destination)
            validate(context, destination, offset, bytes);
        else
            require(offset == 0, "Invalid empty HIP upload offset");
        DeviceScope device(context->device);
        ensure_staging_capacity(staging, bytes);
        if (bytes)
            std::memmove(staging->host, source, bytes);
        staging->bytes = bytes;
        staging->reserved_bytes = 0;
        staging->recorded = false;
        staging->copy_error = hipSuccess;
        staging->pending = true;
        if (bytes) {
            staging->copy_error = hipMemcpyAsync(static_cast<char*>(destination->data) + offset, staging->host, bytes, hipMemcpyHostToDevice, context->stream);
            check(staging->copy_error);
        }
        check(hipEventRecord(staging->completion, context->stream));
        staging->recorded = true;
    });
}
int lfs_hip_marker_record(LfsHipContext* context, LfsHipReadback* marker, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, marker);
        require(marker->capacity == 0 && marker->bytes == 0, "HIP ordering marker cannot reuse data staging");
        DeviceScope device(context->device);
        marker->pending = true;
        marker->recorded = false;
        check(hipEventRecord(marker->completion, context->stream));
        marker->recorded = true;
    });
}
int lfs_hip_readback_reserve(LfsHipContext* context, LfsHipReadback* staging, size_t bytes, uint64_t* host, LfsHipError* error) {
    return boundary(error, [&] {
        require(host, "Missing HIP packed readback pointer output");
        *host = 0;
        validate_readback(context, staging);
        require(!staging->pending && bytes, "Cannot resize pending or empty HIP packed readback");
        DeviceScope device(context->device);
        staging->reserved_bytes = 0;
        ensure_staging_capacity(staging, bytes);
        std::memset(staging->host, 0, bytes);
        staging->reserved_bytes = bytes;
        *host = reinterpret_cast<uintptr_t>(staging->host);
    });
}
int lfs_hip_readback_append(LfsHipContext* context, LfsHipReadback* staging, const LfsHipBuffer* source, size_t offset, size_t bytes, size_t destination_offset, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        require(staging->copy_error == hipSuccess, "Discard failed HIP readback before appending");
        validate(context, source, offset, bytes);
        require(staging->reserved_bytes && destination_offset <= staging->reserved_bytes && bytes <= staging->reserved_bytes - destination_offset,
                "HIP packed readback range exceeds reserved host slot");
        DeviceScope device(context->device);
        staging->bytes = staging->reserved_bytes;
        staging->pending = true;
        staging->recorded = false;
        if (bytes) {
            staging->copy_error = hipMemcpyAsync(static_cast<char*>(staging->host) + destination_offset, static_cast<const char*>(source->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream);
            check(staging->copy_error);
        }
        check(hipEventRecord(staging->completion, context->stream));
        staging->recorded = true;
    });
}
int lfs_hip_readback_wait(LfsHipContext* context, LfsHipReadback* staging, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        DeviceScope device(context->device);
        if (staging->pending) {
            check(staging->recorded ? hipEventSynchronize(staging->completion) : hipStreamSynchronize(context->stream));
            check(staging->copy_error);
        }
    });
}
int lfs_hip_readback_poll(LfsHipContext* context, LfsHipReadback* staging, void* destination, size_t destination_bytes, int32_t* ready, LfsHipError* error) {
    return boundary(error, [&] {
        require(ready, "Missing HIP readback readiness output");
        *ready = 0;
        validate_readback(context, staging);
        if (!staging->pending)
            return;
        require(!destination || destination_bytes >= staging->bytes, "HIP readback destination is too small");
        DeviceScope device(context->device);
        const auto status = staging->recorded ? hipEventQuery(staging->completion) : hipStreamQuery(context->stream);
        if (status == hipErrorNotReady)
            return;
        check(status);
        if (staging->copy_error != hipSuccess) {
            const auto copy_error = staging->copy_error;
            // Draining a failed copy must never expose stale or partial bytes.
            // A null poll discards the failed result, but still reports its error.
            if (!destination) {
                staging->pending = false;
                staging->copy_error = hipSuccess;
            }
            check(copy_error);
        }
        if (destination && staging->bytes)
            std::memmove(destination, staging->host, staging->bytes);
        staging->pending = false;
        *ready = 1;
    });
}
int lfs_hip_readback_destroy(LfsHipContext* context, LfsHipReadback* staging, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        DeviceScope device(context->device);
        if (staging->pending)
            check(hipStreamSynchronize(context->stream));
        if (staging->completion) {
            check(hipEventDestroy(staging->completion));
            staging->completion = nullptr;
        }
        if (staging->host) {
            check(hipHostFree(staging->host));
            staging->host = nullptr;
        }
        context->readbacks.erase(staging);
        delete staging;
    });
}

int lfs_hip_fill_f32(LfsHipContext* context, LfsHipBuffer* buffer, size_t offset, size_t count, float value, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, buffer, offset, float_bytes(count));
        require(offset % alignof(float) == 0, "Unaligned float offset");
        if (!count)
            return;
        DeviceScope device(context->device);
        auto* output = reinterpret_cast<float*>(static_cast<char*>(buffer->data) + offset);
        hipLaunchKernelGGL(fill, dim3(blocks(count)), dim3(256), 0, context->stream, output, count, value);
        check(hipGetLastError());
    });
}

int lfs_hip_affine_f32(LfsHipContext* context, const LfsHipBuffer* input, LfsHipBuffer* output, size_t count, float scale, float bias, LfsHipError* error) {
    return boundary(error, [&] {
        const size_t bytes = float_bytes(count);
        validate(context, input, 0, bytes);
        validate(context, output, 0, bytes);
        if (!count)
            return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL(affine, dim3(blocks(count)), dim3(256), 0, context->stream,
                           static_cast<const float*>(input->data), static_cast<float*>(output->data), count, scale, bias);
        check(hipGetLastError());
    });
}
int lfs_hip_copy(LfsHipContext* context, LfsHipView src, LfsHipView dst, size_t bytes, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, src.buffer, src.offset, bytes);
        validate(context, dst.buffer, dst.offset, bytes);
        require(src.buffer != dst.buffer || src.offset == dst.offset || src.offset + bytes <= dst.offset || dst.offset + bytes <= src.offset,
                "HIP device copy ranges overlap");
        DeviceScope device(context->device);
        if (bytes && (src.buffer != dst.buffer || src.offset != dst.offset))
            check(hipMemcpyAsync(static_cast<char*>(dst.buffer->data) + dst.offset, static_cast<const char*>(src.buffer->data) + src.offset, bytes, hipMemcpyDeviceToDevice, context->stream));
    });
}

int lfs_hip_memset(LfsHipContext* context, LfsHipView dst, size_t bytes, uint8_t value, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, dst.buffer, dst.offset, bytes);
        DeviceScope device(context->device);
        if (bytes)
            check(hipMemsetAsync(static_cast<char*>(dst.buffer->data) + dst.offset, value, bytes, context->stream));
    });
}

int lfs_hip_pointwise_f32(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && op < LFS_HIP_OP_COUNT, "Unknown HIP pointwise operation");
        DeviceScope device(context->device);
        using namespace lfs::core;
        switch (op) {
#include "hip_pointwise_f32.inc"
        default: throw std::invalid_argument("Unknown HIP pointwise operation");
        }
    });
}

int lfs_hip_pointwise_f16(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && op < LFS_HIP_OP_COUNT, "Unknown HIP pointwise operation");
        DeviceScope device(context->device);
        using namespace lfs::core;
        switch (op) {
#include "hip_pointwise_f16.inc"
        default: throw std::invalid_argument("Unknown HIP pointwise operation");
        }
    });
}

int lfs_hip_arange_f32(LfsHipContext* context, LfsHipView dst, size_t count, float start, float step, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, dst.buffer, dst.offset, float_bytes(count));
        require(dst.offset % 4 == 0, "Unaligned arange output");
        DeviceScope device(context->device);
        if (!count)
            return;
        hipLaunchKernelGGL(arange_kernel, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<float*>(static_cast<char*>(dst.buffer->data) + dst.offset), count, start, step);
        check(hipGetLastError());
    });
}

} // extern "C"

#include "tensor_ops_runtime.hpp"

namespace {
    __global__ void environment_composite_kernel(const float* rgb, const float* alpha,
                                                 const float* environment, uint8_t* output,
                                                 LfsHipEnvironment p, size_t pixels) {
        namespace em = lfs::core::envmath;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < pixels; i += size_t(gridDim.x) * blockDim.x) {
            const int x = int(i % size_t(p.band_width));
            const int y = int(i / size_t(p.band_width));
            auto direction = em::environmentWorldDirection(float(x), float(p.y_offset + y),
                                                           float(p.full_width), float(p.full_height), p.equirect_view != 0,
                                                           p.focal_x, p.focal_y, p.center_x, p.center_y, p.rotation);
            direction = em::normalized(em::rotateAroundY(direction, p.env_rotation_radians));
            const auto uv = em::equirectUvForDirection(direction);
            const auto tap = em::environmentBilinearTap(uv.u, uv.v, p.env_width, p.env_height);
            const auto fetch = [&](int px, int py) {
                const size_t offset = (size_t(py) * size_t(p.env_width) + size_t(px)) * 3;
                return em::Vec3{environment[offset], environment[offset + 1], environment[offset + 2]};
            };
            const auto top = em::mix(fetch(tap.x0, tap.y0), fetch(tap.x1, tap.y0), tap.tx);
            const auto bottom = em::mix(fetch(tap.x0, tap.y1), fetch(tap.x1, tap.y1), tap.tx);
            const auto background = em::shadeEnvironmentRadiance(em::mix(top, bottom, tap.ty), p.exposure_factor);
            const em::Vec3 color{rgb[i], rgb[pixels + i], rgb[2 * pixels + i]};
            const auto mixed = em::mix(background, color, alpha[i]);
            const auto to_byte = [](float value) {
                return static_cast<uint8_t>(fminf(fmaxf(value, 0.f), 1.f) * 255.f + 0.5f);
            };
            output[3 * i] = to_byte(mixed.x);
            output[3 * i + 1] = to_byte(mixed.y);
            output[3 * i + 2] = to_byte(mixed.z);
        }
    }
} // namespace
int lfs_hip_context_queue(LfsHipContext* context, uint64_t* queue, int32_t* device, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && queue && device, "Missing HIP queue descriptor");
        *queue = reinterpret_cast<uintptr_t>(context->stream);
        *device = context->device;
    });
}
int lfs_hip_context_ready(LfsHipContext* context, int32_t* ready, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && ready, "Missing HIP queue readiness output");
        *ready = 0;
        DeviceScope device(context->device);
        const auto status = hipStreamQuery(context->stream);
        if (status == hipErrorNotReady)
            return;
        check(status);
        *ready = 1;
    });
}
int lfs_hip_host_callback(LfsHipContext* context, void (*callback)(void*), void* user, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && callback, "Missing HIP queue callback");
        DeviceScope device(context->device);
        check(hipLaunchHostFunc(context->stream, callback, user));
    });
}
int lfs_hip_buffer_address(LfsHipContext* context, const LfsHipBuffer* buffer, uint64_t* address, LfsHipError* error) {
    return boundary(error, [&] {
        require(address, "Missing HIP address output");
        *address = 0;
        validate(context, buffer, 0, 0);
        *address = reinterpret_cast<uintptr_t>(buffer->data);
    });
}
int lfs_hip_environment_composite(LfsHipContext* context, LfsHipView rgb, LfsHipView alpha,
                                  LfsHipView environment, LfsHipView output,
                                  const LfsHipEnvironment* params, LfsHipError* error) {
    return boundary(error, [&] {
        require(params && params->band_width > 0 && params->band_height > 0 &&
                    params->full_width == params->band_width && params->full_height >= params->band_height &&
                    params->y_offset >= 0 && params->y_offset <= params->full_height - params->band_height &&
                    params->env_width > 0 && params->env_height > 0,
                "Invalid HIP environment composite dimensions");
        const size_t pixels = size_t(params->band_width) * size_t(params->band_height);
        const size_t env_pixels = size_t(params->env_width) * size_t(params->env_height);
        require(pixels <= INT32_MAX && env_pixels <= INT32_MAX && pixels <= SIZE_MAX / 3 && env_pixels <= SIZE_MAX / 3,
                "HIP environment composite extent overflow");
        validate(context, rgb.buffer, rgb.offset, float_bytes(pixels * 3));
        validate(context, alpha.buffer, alpha.offset, float_bytes(pixels));
        validate(context, environment.buffer, environment.offset, float_bytes(env_pixels * 3));
        validate(context, output.buffer, output.offset, pixels * 3);
        require(rgb.offset % 4 == 0 && alpha.offset % 4 == 0 && environment.offset % 4 == 0,
                "Unaligned HIP environment composite input");
        require(output.buffer != rgb.buffer && output.buffer != alpha.buffer &&
                    output.buffer != environment.buffer,
                "HIP environment output overlaps an input");
        DeviceScope device(context->device);
        hipLaunchKernelGGL(environment_composite_kernel, dim3(blocks(pixels)), dim3(256), 0, context->stream,
                           reinterpret_cast<const float*>(static_cast<const char*>(rgb.buffer->data) + rgb.offset),
                           reinterpret_cast<const float*>(static_cast<const char*>(alpha.buffer->data) + alpha.offset),
                           reinterpret_cast<const float*>(static_cast<const char*>(environment.buffer->data) + environment.offset),
                           static_cast<uint8_t*>(output.buffer->data) + output.offset, *params, pixels);
        check(hipGetLastError());
    });
}
int lfs_hip_inverse_lens(LfsHipContext* context, LfsHipView output, const LfsHipLens* lens, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && lens && lens->model_type == 0, "HIP lens inverse currently requires pinhole calibration");
        require(lens->src_width > 0 && lens->src_height > 0 && uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / 2,
                "HIP lens inverse extent exceeds signed indexing");
        require(std::isfinite(lens->src_fx) && std::isfinite(lens->src_fy) && lens->src_fx > 0 && lens->src_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP lens calibration");
        for (int i = 0; i < lens->num_distortion; ++i)
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP lens distortion");
        const size_t bytes = size_t(lens->src_width) * lens->src_height * sizeof(float2);
        validate(context, output.buffer, output.offset, bytes);
        require(output.offset % alignof(float2) == 0, "Misaligned HIP lens output");
        DeviceScope device(context->device);
        const dim3 block(16, 16), grid((lens->src_width + 15) / 16, (lens->src_height + 15) / 16);
        hipLaunchKernelGGL(hip_lens::inverse_distortion_sample_map_kernel, grid, block, 0, context->stream,
                           reinterpret_cast<float2*>(static_cast<char*>(output.buffer->data) + output.offset), *lens);
        check(hipGetLastError());
    });
}
namespace {
    template <typename T>
    __global__ void identity_rgb(const T* input, float* output, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = hip_lens::normalized_sample(input[i]);
    }
} // namespace
int lfs_hip_warp_rgb(LfsHipContext* context, LfsHipView input, uint32_t dtype,
                     LfsHipView output, const LfsHipLens* lens, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && lens && lens->model_type == 0, "HIP RGB warp requires pinhole calibration");
        require(dtype == LFS_HIP_F32 || dtype == LFS_HIP_U8, "HIP RGB warp input must be float32 or uint8");
        require(lens->src_width > 0 && lens->src_height > 0 && lens->dst_width > 0 && lens->dst_height > 0 &&
                    uint64_t(lens->src_width) * lens->src_height <= INT32_MAX / 3 &&
                    uint64_t(lens->dst_width) * lens->dst_height <= INT32_MAX / 3,
                "HIP RGB warp extent exceeds signed indexing");
        require(std::isfinite(lens->src_fx) && std::isfinite(lens->src_fy) && lens->src_fx > 0 && lens->src_fy > 0 &&
                    std::isfinite(lens->dst_fx) && std::isfinite(lens->dst_fy) && lens->dst_fx > 0 && lens->dst_fy > 0 &&
                    std::isfinite(lens->src_cx) && std::isfinite(lens->src_cy) && std::isfinite(lens->dst_cx) && std::isfinite(lens->dst_cy) &&
                    lens->num_distortion >= 0 && lens->num_distortion <= 12,
                "Invalid HIP RGB warp calibration");
        bool identity = lens->src_width == lens->dst_width && lens->src_height == lens->dst_height &&
                        lens->src_fx == lens->dst_fx && lens->src_fy == lens->dst_fy &&
                        lens->src_cx == lens->dst_cx && lens->src_cy == lens->dst_cy;
        for (int i = 0; i < lens->num_distortion; ++i) {
            require(std::isfinite(lens->distortion[i]), "Nonfinite HIP RGB warp distortion");
            identity = identity && lens->distortion[i] == 0;
        }
        const float footprint = std::max(lens->src_fx / lens->dst_fx, lens->src_fy / lens->dst_fy);
        require(std::isfinite(footprint) && footprint <= 64, "HIP RGB warp sampling footprint exceeds experimental limit 64");
        const int quadrature = std::max(8, int(std::ceil(footprint)));
        const uint64_t row_work = uint64_t(lens->dst_width) * quadrature * quadrature;
        require(identity || row_work <= 2 * 1024 * 1024, "HIP RGB warp row exceeds bounded sampling work");
        const size_t input_count = size_t(lens->src_width) * lens->src_height * 3;
        const size_t output_count = size_t(lens->dst_width) * lens->dst_height * 3;
        const size_t input_bytes = input_count * (dtype == LFS_HIP_U8 ? 1 : sizeof(float));
        const size_t output_bytes = output_count * sizeof(float);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(output.offset % 4 == 0 && (dtype == LFS_HIP_U8 || input.offset % 4 == 0), "Misaligned HIP RGB warp view");
        require(input.buffer != output.buffer || input.offset >= output.offset + output_bytes || output.offset >= input.offset + input_bytes,
                "HIP RGB warp input/output overlap");
        DeviceScope device(context->device);
        const auto* source = static_cast<const char*>(input.buffer->data) + input.offset;
        auto* destination = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        if (identity) {
            if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((identity_rgb<uint8_t>), dim3(blocks(output_count)), dim3(256), 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, output_count);
            else
                hipLaunchKernelGGL((identity_rgb<float>), dim3(blocks(output_count)), dim3(256), 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, output_count);
            check(hipGetLastError());
            return;
        }
        const int rows = std::max(1, std::min(32, int((2 * 1024 * 1024) / row_work)));
        for (int y = 0; y < lens->dst_height; y += rows) {
            const int end = std::min(lens->dst_height, y + rows);
            const dim3 block(16, 16), grid((lens->dst_width + 15) / 16, (end - y + 15) / 16);
            if (dtype == LFS_HIP_U8)
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<uint8_t, 3>), grid, block, 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, 3, y, end, quadrature, *lens);
            else
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<float, 3>), grid, block, 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, 3, y, end, quadrature, *lens);
            check(hipGetLastError());
        }
    });
}
#include "affine_runtime.hpp"
#include "bias_runtime.hpp"
#include "device_identity_runtime.hpp"
#include "export_runtime.hpp"
#include "expression_runtime.hpp"
#include "inference_runtime.hpp"
#include "inverse_warp_runtime.hpp"
#include "morton_runtime.hpp"
#include "multinomial_runtime.hpp"
#include "nn_attention_runtime.hpp"
#include "nn_conv_runtime.hpp"
#include "nn_linear_runtime.hpp"
#include "nn_runtime.hpp"
#include "point_filter_runtime.hpp"
#include "point_tree_runtime.hpp"
#include "pool_runtime.hpp"
#include "ppisp_runtime.hpp"
#include "sh_codec_runtime.hpp"
#include "shared_image_runtime.hpp"
#include "shared_resize_runtime.hpp"
#include "shared_warp_runtime.hpp"
#include "simplify_runtime.hpp"
#include "spatial_runtime.hpp"
#include "timing_runtime.hpp"
