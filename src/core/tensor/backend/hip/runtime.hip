/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "runtime_common.hpp"
namespace {
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
} // namespace
hipError_t lfs_hip_fill_internal(LfsHipContext* context, float* output, size_t count, float value) {
    if (!count)
        return hipSuccess;
    hipLaunchKernelGGL(fill, dim3(blocks(count)), dim3(256), 0, context->stream, output, count, value);
    return hipGetLastError();
}
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
        synchronize_context(context);
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
        synchronize_context(context);
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
        // On a runtime failure, retain the handle and unfreed allocations
        // for explicit recovery; never claim that queued work was drained.
        synchronize_context(context);
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
        if (context->index_error) {
            check(hipFree(context->index_error));
            context->index_error = nullptr;
        }
        if (context->index_error_host) {
            check(hipHostFree(context->index_error_host));
            context->index_error_host = nullptr;
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
            synchronize_context(context);
        if (bytes)
            check(hipMemcpyAsync(static_cast<char*>(buffer->data) + offset, host, bytes, hipMemcpyHostToDevice, context->stream));
        synchronize_context(context);
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
            synchronize_context(context);
        if (bytes)
            check(hipMemcpyAsync(host, static_cast<const char*>(buffer->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream));
        synchronize_context(context);
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
    __global__ void snapshot_execution_error(const uint32_t* source, uint32_t* destination) {
        *destination |= *source;
    }
    void capture_execution_error(LfsHipContext* context, LfsHipReadback* staging, bool reset, bool packed = false) {
        if (reset) {
            staging->execution_snapshot = false;
            staging->execution_error_initialized = false;
        }
        if (!context->index_work_pending)
            return;
        try {
            if (!staging->execution_error_host)
                check(hipHostMalloc(reinterpret_cast<void**>(&staging->execution_error_host), sizeof(uint32_t), hipHostMallocDefault));
            // This runs at the snapshot's position, never at wait/poll time.
            // Packed appends retain faults from all ranges, even if a context
            // synchronization consumes and resets the source flag in between.
            const auto* captured = context->index_error;
            if (packed) {
                if (!staging->execution_error_device)
                    check(hipMalloc(&staging->execution_error_device, sizeof(uint32_t)));
                if (!staging->execution_error_initialized) {
                    check(hipMemsetAsync(staging->execution_error_device, 0, sizeof(uint32_t), context->stream));
                    staging->execution_error_initialized = true;
                }
                hipLaunchKernelGGL(snapshot_execution_error, dim3(1), dim3(1), 0, context->stream,
                                   context->index_error, staging->execution_error_device);
                check(hipGetLastError());
                captured = staging->execution_error_device;
            }
            check(hipMemcpyAsync(staging->execution_error_host, captured,
                                 sizeof(uint32_t), hipMemcpyDeviceToHost, context->stream));
            staging->execution_snapshot = true;
        } catch (const HipFailure& failure) {
            // A failed error capture cannot turn a payload into a valid result.
            staging->copy_error = failure.code;
            throw;
        }
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
        capture_execution_error(context, staging, true);
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
        capture_execution_error(context, staging, true);
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
        marker->copy_error = hipSuccess;
        capture_execution_error(context, marker, true);
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
        const bool reset = !staging->pending;
        staging->pending = true;
        staging->recorded = false;
        if (bytes) {
            staging->copy_error = hipMemcpyAsync(static_cast<char*>(staging->host) + destination_offset, static_cast<const char*>(source->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream);
            check(staging->copy_error);
        }
        capture_execution_error(context, staging, reset, true);
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
            if (staging->execution_snapshot)
                check_execution_error(*staging->execution_error_host);
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
        if (staging->execution_snapshot && *staging->execution_error_host) {
            if (!destination)
                staging->pending = false; // Discard a drained failed snapshot.
            check_execution_error(*staging->execution_error_host);
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
            synchronize_context(context);
        if (staging->completion) {
            check(hipEventDestroy(staging->completion));
            staging->completion = nullptr;
        }
        if (staging->host) {
            check(hipHostFree(staging->host));
            staging->host = nullptr;
        }
        if (staging->execution_error_host) {
            check(hipHostFree(staging->execution_error_host));
            staging->execution_error_host = nullptr;
        }
        if (staging->execution_error_device) {
            check(hipFree(staging->execution_error_device));
            staging->execution_error_device = nullptr;
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
        check(lfs_hip_fill_internal(context, output, count, value));
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

} // extern "C"
