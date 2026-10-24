/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "architecture_runtime.hpp"
#include "runtime_api.h"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>
#include <unordered_set>

struct LfsHipBuffer {
    void* data = nullptr;
    size_t bytes = 0;
};
struct LfsHipContext {
    int device = 0;
    hipStream_t stream = nullptr;
    std::unordered_set<LfsHipBuffer*> buffers;
};

namespace {
    struct HipFailure : std::exception {
        hipError_t code;
        explicit HipFailure(hipError_t value) : code(value) {}
        const char* what() const noexcept override { return hipGetErrorString(code); }
    };

    void check(hipError_t status) {
        if (status != hipSuccess)
            throw HipFailure(status);
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

} // namespace

extern "C" {
uint32_t lfs_hip_runtime_abi(void) { return 1; }

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
        DeviceScope device(context->device);
        // On a runtime failure, retain the handle and unfreed allocations
        // for explicit recovery; never claim that queued work was drained.
        check(hipStreamSynchronize(context->stream));
        while (!context->buffers.empty()) {
            auto* buffer = *context->buffers.begin();
            release_buffer_resources(buffer);
            context->buffers.erase(buffer);
            delete buffer;
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
} // extern "C"
