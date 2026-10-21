/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_tensor_functors.hpp"
#include "runtime_api.h"
#include "timing_api.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <limits>
#include <type_traits>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "hip_lens.hpp"
#include <core/environment_math.hpp>

struct LfsHipBuffer {
    void* data = nullptr;
    size_t bytes = 0;
    hipExternalMemory_t external_memory = nullptr;
};

struct LfsHipReadback {
    void* host = nullptr;
    size_t capacity = 0, bytes = 0;
    hipEvent_t completion = nullptr;
    bool pending = false, recorded = false;
};

struct LfsHipTiming {
    std::vector<hipEvent_t> events;
    std::vector<uint64_t> sequence;
    uint64_t next_sequence = 0;
};

struct LfsHipContext {
    int device = 0;
    hipStream_t stream = nullptr;
    std::unordered_set<LfsHipBuffer*> buffers;
    std::unordered_set<LfsHipReadback*> readbacks;
    std::unordered_set<LfsHipTiming*> timings;
    float* reduction_workspace = nullptr;
    void* scan_workspace = nullptr;
    size_t scan_workspace_bytes = 0;
    void* sort_workspace = nullptr;
    size_t sort_workspace_bytes = 0;
};

namespace {
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

    // Keep a partially released import registered if the driver rejects cleanup.
    // A subsequent release must never free the same mapping twice.
    void release_buffer_resources(LfsHipBuffer* buffer) {
        if (buffer->data) {
            check(hipFree(buffer->data));
            buffer->data = nullptr;
        }
        if (buffer->external_memory) {
            check(hipDestroyExternalMemory(buffer->external_memory));
            buffer->external_memory = nullptr;
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
        auto overlap = [&](LfsHipView input, size_t bytes) {
            if (input.buffer != out.buffer || !bytes)
                return;
            const size_t end = input.offset + bytes, output_end = out.offset + count * sizeof(Out);
            require(input.offset >= output_end || out.offset >= end ||
                        (sizeof(Out) == sizeof(In) && input.offset == out.offset),
                    "Overlapping pointwise views require identical typed ranges");
        };
        overlap(a, input_bytes);
        if constexpr (Mode == 2)
            overlap(b, input_bytes);
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

    size_t typed_bytes(uint32_t dtype, size_t count) {
        require(dtype <= LFS_HIP_U32, "Unknown HIP conversion dtype");
        constexpr size_t sizes[] = {4, 2, 4, 8, 1, 1, 4};
        require(count <= SIZE_MAX / sizes[dtype], "HIP conversion byte count overflow");
        return count * sizes[dtype];
    }
    template <class To, class From>
    __device__ To convert_value(From value) {
        if constexpr (std::is_same_v<From, __half>)
            return convert_value<To>(__half2float(value));
        else if constexpr (std::is_same_v<To, __half>)
            return __float2half_rn(static_cast<float>(value));
        else if constexpr (std::is_same_v<To, uint8_t> && std::is_same_v<From, float>) {
            // Same truncate/modulo contract as detail::torch_uint8_cast.
            if (!isfinite(value))
                return 0;
            float wrapped = fmodf(truncf(value), 256.f);
            if (wrapped < 0)
                wrapped += 256.f;
            return static_cast<uint8_t>(wrapped);
        } else if constexpr (std::is_integral_v<To> && !std::is_same_v<To, bool> && std::is_same_v<From, float>) {
            // Guard before casting: out-of-range floating casts are undefined
            // in C++ and have different native instructions across vendors.
            if (isnan(value))
                return To{0};
            if (value <= static_cast<float>(std::numeric_limits<To>::lowest()))
                return std::numeric_limits<To>::lowest();
            if (value >= static_cast<float>(std::numeric_limits<To>::max()))
                return std::numeric_limits<To>::max();
            return static_cast<To>(truncf(value));
        } else
            return static_cast<To>(value);
    }
    template <class To, class From>
    __global__ void convert_kernel(const From* input, To* output, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = convert_value<To>(input[i]);
    }
    template <class To, class From>
    void launch_convert(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t count) {
        hipLaunchKernelGGL((convert_kernel<To, From>), dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const From*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<To*>(static_cast<char*>(output.buffer->data) + output.offset), count);
        check(hipGetLastError());
    }
    template <class From>
    void dispatch_convert(LfsHipContext* context, LfsHipView input, LfsHipView output, uint32_t dtype, size_t count) {
        switch (dtype) {
        case LFS_HIP_F32: launch_convert<float, From>(context, input, output, count); break;
        case LFS_HIP_F16: launch_convert<__half, From>(context, input, output, count); break;
        case LFS_HIP_I32: launch_convert<int32_t, From>(context, input, output, count); break;
        case LFS_HIP_I64: launch_convert<int64_t, From>(context, input, output, count); break;
        case LFS_HIP_U8: launch_convert<uint8_t, From>(context, input, output, count); break;
        case LFS_HIP_BOOL: launch_convert<bool, From>(context, input, output, count); break;
        case LFS_HIP_U32: launch_convert<uint32_t, From>(context, input, output, count); break;
        }
    }
    __global__ void validate_indices(const int64_t* input, size_t count, size_t extent, uint32_t* invalid) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            if (input[i] < 0 || uint64_t(input[i]) >= extent)
                atomicOr(invalid, 1u);
    }
    template <class T>
    __global__ void count_nonzero_kernel(const T* input, size_t count, unsigned long long* total) {
        __shared__ unsigned long long partial[256];
        unsigned long long local = 0;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            local += input[i] != T(0);
        partial[threadIdx.x] = local;
        __syncthreads();
        for (unsigned width = 128; width; width >>= 1) {
            if (threadIdx.x < width)
                partial[threadIdx.x] += partial[threadIdx.x + width];
            __syncthreads();
        }
        if (!threadIdx.x)
            atomicAdd(total, partial[0]);
    }
    template <class T>
    void launch_count_nonzero(LfsHipContext* context, LfsHipView input, size_t count, unsigned long long* total) {
        hipLaunchKernelGGL((count_nonzero_kernel<T>), dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const T*>(static_cast<const char*>(input.buffer->data) + input.offset), count, total);
        check(hipGetLastError());
    }
    // Same Philox4x32-10 round constants as the project's random.slang.
    // One counter block supplies four draws without persistent per-thread state.
    __device__ uint4 random_words(uint64_t counter, uint64_t seed) {
        uint4 words = make_uint4(uint32_t(counter), uint32_t(counter >> 32), 0, 0);
        uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
#pragma unroll
        for (unsigned round = 0; round < 10; ++round) {
            const uint32_t hi0 = __umulhi(0xD2511F53u, words.x), hi1 = __umulhi(0xCD9E8D57u, words.z);
            words = make_uint4(hi1 ^ words.y ^ k0, 0xCD9E8D57u * words.z, hi0 ^ words.w ^ k1, 0xD2511F53u * words.x);
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        return words;
    }
    __device__ float unit_draw(uint32_t word) { return float(word >> 8) * 0x1p-24f; }
    template <unsigned Kind>
    __global__ void random_kernel(void* output, LfsHipRandom program) {
        const size_t groups = (program.count + 3) / 4;
        for (size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x; group < groups; group += size_t(gridDim.x) * blockDim.x) {
            const auto words = random_words(group, program.seed);
            const uint32_t raw[] = {words.x, words.y, words.z, words.w};
            float values[4]{};
            if constexpr (Kind == LFS_HIP_RANDOM_NORMAL) {
#pragma unroll
                for (unsigned pair = 0; pair < 2; ++pair) {
                    const float u1 = float((raw[pair * 2] >> 8) + 1u) * 0x1p-24f, u2 = unit_draw(raw[pair * 2 + 1]);
                    const float radius = sqrtf(-2.f * logf(u1));
                    float sine, cosine;
                    sincosf(6.2831853071795864769f * u2, &sine, &cosine);
                    values[pair * 2] = fmaf(program.second, radius * cosine, program.first);
                    values[pair * 2 + 1] = fmaf(program.second, radius * sine, program.first);
                }
            }
            int integers[4]{};
#pragma unroll
            for (unsigned lane = 0; lane < 4; ++lane) {
                if constexpr (Kind == LFS_HIP_RANDOM_UNIFORM) {
                    const float u = unit_draw(raw[lane]), delta = program.second - program.first;
                    float v = isfinite(delta) ? fmaf(u, delta, program.first) : fmaf(u, program.second, (1.f - u) * program.first);
                    if (program.first == program.second)
                        v = program.first;
                    else
                        v = fmaxf(program.first, fminf(nextafterf(program.second, program.first), v));
                    values[lane] = v;
                } else if constexpr (Kind == LFS_HIP_RANDOM_BERNOULLI)
                    values[lane] = unit_draw(raw[lane]) < program.first ? 1.f : 0.f;
                else if constexpr (Kind == LFS_HIP_RANDOM_INTEGER) {
                    const uint64_t range = uint64_t(int64_t(program.high) - int64_t(program.low));
                    integers[lane] = int(int64_t(program.low) + int64_t((uint64_t(raw[lane]) * range) >> 32));
                }
            }
            const size_t begin = group * 4;
            if constexpr (Kind == LFS_HIP_RANDOM_INTEGER) {
                auto* target = static_cast<int32_t*>(output) + begin;
                if (program.count - begin >= 4 && reinterpret_cast<uintptr_t>(target) % 16 == 0)
                    *reinterpret_cast<int4*>(target) = make_int4(integers[0], integers[1], integers[2], integers[3]);
                else
                    for (unsigned lane = 0; lane < 4 && begin + lane < program.count; ++lane)
                        target[lane] = integers[lane];
            } else {
                auto* target = static_cast<float*>(output) + begin;
                if (program.count - begin >= 4 && reinterpret_cast<uintptr_t>(target) % 16 == 0)
                    *reinterpret_cast<float4*>(target) = make_float4(values[0], values[1], values[2], values[3]);
                else
                    for (unsigned lane = 0; lane < 4 && begin + lane < program.count; ++lane)
                        target[lane] = values[lane];
            }
        }
    }
    template <unsigned Kind>
    void launch_random(LfsHipContext* context, LfsHipView output, LfsHipRandom program) {
        hipLaunchKernelGGL((random_kernel<Kind>), dim3(blocks((program.count + 3) / 4)), dim3(256), 0, context->stream,
                           static_cast<char*>(output.buffer->data) + output.offset, program);
        check(hipGetLastError());
    }
    constexpr size_t reduction_workspace_floats = LFS_HIP_REDUCTION_WORKSPACE_FLOATS;
    template <unsigned Mode>
    __device__ float combine_reduce(float a, float b) {
        if constexpr (Mode == LFS_HIP_REDUCE_MIN)
            return lfs::core::ops::min_reduce_op{}(a, b);
        else if constexpr (Mode == LFS_HIP_REDUCE_MAX)
            return lfs::core::ops::max_reduce_op{}(a, b);
        else if constexpr (Mode == LFS_HIP_REDUCE_HAS_NAN || Mode == LFS_HIP_REDUCE_HAS_INF)
            return fmaxf(a, b);
        else
            return a + b;
    }
    template <unsigned Mode, bool Final>
    __global__ void reduction_kernel(const float* in, size_t count, float* out) {
        __shared__ float shared[256];
        float value = 0;
        if constexpr (Mode == LFS_HIP_REDUCE_MIN)
            value = INFINITY;
        if constexpr (Mode == LFS_HIP_REDUCE_MAX)
            value = -INFINITY;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            float input = in[i];
            if constexpr (!Final && Mode == LFS_HIP_REDUCE_HAS_NAN)
                input = isnan(input) ? 1.f : 0.f;
            if constexpr (!Final && Mode == LFS_HIP_REDUCE_HAS_INF)
                input = isinf(input) ? 1.f : 0.f;
            value = combine_reduce<Mode>(value, input);
        }
        shared[threadIdx.x] = value;
        __syncthreads();
        for (unsigned stride = 128; stride; stride >>= 1) {
            if (threadIdx.x < stride)
                shared[threadIdx.x] = combine_reduce<Mode>(shared[threadIdx.x], shared[threadIdx.x + stride]);
            __syncthreads();
        }
        if (!threadIdx.x)
            out[blockIdx.x] = shared[0];
    }
    template <unsigned Mode>
    void launch_reduce(LfsHipContext* context, LfsHipView input, size_t count) {
        const auto* data = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        const unsigned groups = blocks(count);
        hipLaunchKernelGGL((reduction_kernel<Mode, false>), dim3(groups), dim3(256), 0, context->stream, data, count, context->reduction_workspace);
        check(hipGetLastError());
        hipLaunchKernelGGL((reduction_kernel<Mode, true>), dim3(1), dim3(256), 0, context->stream,
                           context->reduction_workspace, groups, context->reduction_workspace + 4096);
        check(hipGetLastError());
    }

    constexpr unsigned pointwise_flags[] = {
#include "hip_tensor_flags.inc"
    };
    static_assert(sizeof(pointwise_flags) / sizeof(pointwise_flags[0]) == LFS_HIP_OP_COUNT);
    struct ChainOp {
        uint32_t op;
        float scalar;
        const float* rhs;
    };
    struct Chain {
        uint32_t count;
        ChainOp ops[16];
    };
    __device__ float evaluate_op(uint32_t op, float x, float y, float scalar) {
        using namespace lfs::core;
        switch (op) {
#include "hip_tensor_eval.inc"
        default: return NAN;
        }
    }
    __global__ void chain_kernel(const float* input, float* output, size_t count, Chain chain) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            float value = input[i];
            for (uint32_t stage = 0; stage < chain.count; ++stage) {
                const auto& op = chain.ops[stage];
                value = evaluate_op(op.op, value, op.rhs ? op.rhs[i] : 0, op.scalar);
            }
            output[i] = value;
        }
    }

    size_t layout_span(const LfsHipLayout& layout, unsigned element_bytes, bool writable) {
        require(layout.rank <= 8, "HIP Tensor rank exceeds eight dimensions");
        size_t count = 1, last = 0;
        bool empty = false;
        for (uint32_t i = 0; i < layout.rank; ++i)
            empty |= layout.dims[i] == 0;
        if (empty) {
            require(layout.elements == 0, "HIP empty layout has nonzero element count");
            return 0;
        }
        std::array<std::pair<uint64_t, uint64_t>, 8> sorted{};
        for (uint32_t i = 0; i < layout.rank; ++i) {
            require(layout.dims[i] <= SIZE_MAX / count, "HIP Tensor shape overflow");
            count *= layout.dims[i];
            require(!layout.strides[i] || layout.dims[i] - 1 <= (SIZE_MAX - last) / layout.strides[i], "HIP Tensor stride overflow");
            last += (layout.dims[i] - 1) * layout.strides[i];
            sorted[i] = {layout.strides[i], layout.dims[i]};
        }
        require(count == layout.elements && last < SIZE_MAX / element_bytes, "HIP Tensor layout count or byte span is invalid");
        if (writable) {
            std::sort(sorted.begin(), sorted.begin() + layout.rank);
            uint64_t covered = 1;
            for (uint32_t i = 0; i < layout.rank; ++i) {
                if (sorted[i].second <= 1)
                    continue;
                require(sorted[i].first >= covered, "HIP Tensor output strides overlap");
                covered += (sorted[i].second - 1) * sorted[i].first;
            }
        }
        return (last + 1) * element_bytes;
    }
    __device__ size_t layout_index(size_t index, const LfsHipLayout& layout) {
        size_t offset = 0;
        for (uint32_t r = layout.rank; r; --r) {
            offset += (index % layout.dims[r - 1]) * layout.strides[r - 1];
            index /= layout.dims[r - 1];
        }
        return offset;
    }
    template <class T>
    __global__ void axis_scan(T* data, LfsHipLayout layout, unsigned axis, size_t segments) {
        using Scan = hipcub::BlockScan<T, 256>;
        __shared__ typename Scan::TempStorage scratch;
        for (size_t segment = blockIdx.x; segment < segments; segment += gridDim.x) {
            size_t rest = segment, base = 0;
            for (unsigned r = layout.rank; r; --r) {
                if (r - 1 == axis)
                    continue;
                base += (rest % layout.dims[r - 1]) * layout.strides[r - 1];
                rest /= layout.dims[r - 1];
            }
            T carry = 0;
            for (size_t start = 0; start < layout.dims[axis]; start += 256) {
                const size_t index = start + threadIdx.x;
                T value = index < layout.dims[axis] ? data[base + index * layout.strides[axis]] : T(0);
                T prefix, aggregate;
                Scan(scratch).InclusiveSum(value, prefix, aggregate);
                if (index < layout.dims[axis])
                    data[base + index * layout.strides[axis]] = carry + prefix;
                carry += aggregate;
                __syncthreads();
            }
        }
    }
    template <class T>
    void launch_scan(LfsHipContext* context, LfsHipView view, const LfsHipLayout& layout, unsigned axis) {
        auto* data = reinterpret_cast<T*>(static_cast<char*>(view.buffer->data) + view.offset);
        const size_t segments = layout.elements / layout.dims[axis];
        if (segments == 1 && layout.strides[axis] == 1) {
            size_t required = 0;
            check(hipcub::DeviceScan::InclusiveSum(nullptr, required, data, data, static_cast<int>(layout.elements), context->stream));
            if (required > context->scan_workspace_bytes) {
                // Drain users before replacing the cached workspace. A failed
                // allocation leaves the previous workspace and data untouched.
                check(hipStreamSynchronize(context->stream));
                void* replacement = nullptr;
                check(hipMalloc(&replacement, required));
                if (context->scan_workspace) {
                    const auto status = hipFree(context->scan_workspace);
                    if (status != hipSuccess) {
                        (void)hipFree(replacement);
                        check(status);
                    }
                }
                context->scan_workspace = replacement;
                context->scan_workspace_bytes = required;
            }
            check(hipcub::DeviceScan::InclusiveSum(context->scan_workspace, required, data, data, static_cast<int>(layout.elements), context->stream));
        } else {
            hipLaunchKernelGGL((axis_scan<T>), dim3(static_cast<unsigned>(std::min<size_t>(segments, 4096))), dim3(256), 0, context->stream, data, layout, axis, segments);
            check(hipGetLastError());
        }
    }
    template <unsigned Mode>
    __device__ float reduce_identity() {
        if constexpr (Mode == LFS_HIP_REDUCE_MIN)
            return INFINITY;
        if constexpr (Mode == LFS_HIP_REDUCE_MAX)
            return -INFINITY;
        return 0;
    }
    // Short attribute reductions use one lane per output, avoiding a whole
    // workgroup for each three/four-component Gaussian attribute.
    template <unsigned Mode>
    __global__ void short_axis_kernel(const float* input, float* output, LfsHipLayout kept, LfsHipLayout reduced, float scale) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < kept.elements; i += size_t(gridDim.x) * blockDim.x) {
            float value = reduce_identity<Mode>();
            const size_t base = layout_index(i, kept);
            for (size_t r = 0; r < reduced.elements; ++r)
                value = combine_reduce<Mode>(value, input[base + layout_index(r, reduced)]);
            output[i] = value * scale;
        }
    }
    template <unsigned Mode>
    __global__ void axis_kernel(const float* input, float* output, LfsHipLayout kept, LfsHipLayout reduced, uint32_t parts, float scale) {
        __shared__ float values[256];
        for (size_t group = blockIdx.x; group < kept.elements * parts; group += gridDim.x) {
            const size_t item = group / parts, part = group % parts;
            const size_t base = layout_index(item, kept);
            float value = reduce_identity<Mode>();
            for (size_t r = part * 256 + threadIdx.x; r < reduced.elements; r += size_t(parts) * 256)
                value = combine_reduce<Mode>(value, input[base + layout_index(r, reduced)]);
            values[threadIdx.x] = value;
            __syncthreads();
            for (unsigned width = 128; width; width >>= 1) {
                if (threadIdx.x < width)
                    values[threadIdx.x] = combine_reduce<Mode>(values[threadIdx.x], values[threadIdx.x + width]);
                __syncthreads();
            }
            if (!threadIdx.x)
                output[group] = values[0] * scale;
            __syncthreads();
        }
    }
    template <unsigned Mode>
    void launch_axis(LfsHipContext* context, const float* input, float* output, LfsHipLayout kept, LfsHipLayout reduced, bool mean) {
        const float scale = mean ? (reduced.elements ? 1.f / float(reduced.elements) : NAN) : 1.f;
        if (reduced.elements <= 32) {
            hipLaunchKernelGGL((short_axis_kernel<Mode>), dim3(blocks(kept.elements)), dim3(256), 0, context->stream, input, output, kept, reduced, scale);
            check(hipGetLastError());
            return;
        }
        // Split long reductions with few outputs, e.g. N-by-3 column sums.
        // At most 4096 partial floats are needed regardless of model size.
        const uint32_t parts = kept.elements < 64 && reduced.elements > 4096
                                   ? static_cast<uint32_t>(std::min<size_t>(4096 / kept.elements, (reduced.elements - 1) / 1024 + 1))
                                   : 1;
        float* destination = parts > 1 ? context->reduction_workspace : output;
        const unsigned groups = static_cast<unsigned>(std::min<size_t>(4096, kept.elements * parts));
        hipLaunchKernelGGL((axis_kernel<Mode>), dim3(groups), dim3(256), 0, context->stream, input, destination, kept, reduced, parts, parts > 1 ? 1.f : scale);
        check(hipGetLastError());
        if (parts > 1) {
            LfsHipLayout partial_kept{}, partial_reduced{};
            partial_kept.rank = partial_reduced.rank = 1;
            partial_kept.dims[0] = partial_kept.elements = kept.elements;
            partial_kept.strides[0] = parts;
            partial_reduced.dims[0] = partial_reduced.elements = parts;
            partial_reduced.strides[0] = 1;
            hipLaunchKernelGGL((axis_kernel<Mode>), dim3(static_cast<unsigned>(kept.elements)), dim3(256), 0, context->stream,
                               destination, output, partial_kept, partial_reduced, 1, scale);
            check(hipGetLastError());
        }
    }
    template <class Index>
    __global__ void validate_strided_indices(const Index* indices, LfsHipLayout layout, size_t extent, uint32_t* invalid) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.elements; i += size_t(gridDim.x) * blockDim.x) {
            const int64_t index = indices[layout_index(i, layout)];
            if (index < 0 || uint64_t(index) >= extent)
                atomicOr(invalid, 1u);
        }
    }
    template <class Value, class Index>
    __global__ void gather_kernel(const Value* input, const Index* indices, Value* output, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < destination.elements; i += size_t(gridDim.x) * blockDim.x) {
            size_t coordinates = i, source_offset = 0, index_offset = 0;
            for (unsigned r = destination.rank; r; --r) {
                const size_t coordinate = coordinates % destination.dims[r - 1];
                coordinates /= destination.dims[r - 1];
                if (r - 1 == dim)
                    index_offset = coordinate;
                else
                    source_offset += coordinate * source.strides[r - 1];
            }
            if (select)
                index_offset *= index_layout.strides[0];
            else
                index_offset = layout_index(i, index_layout);
            int64_t index = indices[index_offset];
            const int64_t extent = static_cast<int64_t>(source.dims[dim]);
            if (boundary == 1)
                index = index < 0 ? 0 : index >= extent ? extent - 1
                                                        : index;
            else if (boundary == 2) {
                index %= extent;
                if (index < 0)
                    index += extent;
            }
            output[i] = input[source_offset + size_t(index) * source.strides[dim]];
        }
    }
    template <class Value, class Index>
    void launch_gather(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select) {
        hipLaunchKernelGGL((gather_kernel<Value, Index>), dim3(blocks(destination.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const Value*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset), source, index_layout, destination, dim, boundary, select);
        check(hipGetLastError());
    }
    template <class Index>
    void check_indices(LfsHipContext* context, LfsHipView indices, LfsHipLayout index_layout, size_t extent) {
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        auto* flag = reinterpret_cast<uint32_t*>(context->reduction_workspace + 4096);
        check(hipMemsetAsync(flag, 0, sizeof(uint32_t), context->stream));
        hipLaunchKernelGGL((validate_strided_indices<Index>), dim3(blocks(index_layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset), index_layout, extent, flag);
        check(hipGetLastError());
        uint32_t invalid = 0;
        check(hipMemcpyAsync(&invalid, flag, sizeof(invalid), hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
        if (invalid)
            throw InvalidIndex("HIP index is negative or outside its extent");
    }
    template <class Index>
    void dispatch_gather(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, unsigned width, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select) {
        if (boundary == 0)
            check_indices<Index>(context, indices, index_layout, source.dims[dim]);
        switch (width) {
        case 1: launch_gather<uint8_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 2: launch_gather<uint16_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 4: launch_gather<uint32_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 8: launch_gather<uint64_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        }
    }
    template <class Value, class Index, bool Add>
    __global__ void index_update_kernel(Value* output, const Index* indices, const Value* source, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < source_layout.elements; i += size_t(gridDim.x) * blockDim.x) {
            size_t remaining = i, offset = 0, selected = 0;
            for (unsigned r = source_layout.rank; r; --r) {
                const size_t coordinate = remaining % source_layout.dims[r - 1];
                remaining /= source_layout.dims[r - 1];
                if (r - 1 == dim)
                    selected = coordinate;
                else
                    offset += coordinate * destination.strides[r - 1];
            }
            offset += size_t(indices[selected]) * destination.strides[dim];
            if constexpr (Add)
                atomicAdd(output + offset, source[i]);
            else
                output[offset] = source[i];
        }
    }
    template <class Value, class Index>
    __global__ void index_fill_kernel(Value* output, const Index* indices, Value value, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < source_layout.elements; i += size_t(gridDim.x) * blockDim.x) {
            size_t remaining = i, offset = 0, selected = 0;
            for (unsigned r = source_layout.rank; r; --r) {
                const size_t coordinate = remaining % source_layout.dims[r - 1];
                remaining /= source_layout.dims[r - 1];
                if (r - 1 == dim)
                    selected = coordinate;
                else
                    offset += coordinate * destination.strides[r - 1];
            }
            output[offset + size_t(indices[selected]) * destination.strides[dim]] = value;
        }
    }
    template <class Value, class Index>
    void launch_index_fill(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, uint64_t value_bits) {
        hipLaunchKernelGGL((index_fill_kernel<Value, Index>), dim3(blocks(source_layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           static_cast<Value>(value_bits), destination, source_layout, dim);
        check(hipGetLastError());
    }
    template <class Index>
    void dispatch_index_fill(LfsHipContext* context, LfsHipView output, LfsHipView indices, unsigned width, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, size_t index_count, uint64_t value_bits) {
        LfsHipLayout index_layout{};
        index_layout.rank = 1;
        index_layout.dims[0] = index_layout.elements = index_count;
        index_layout.strides[0] = 1;
        check_indices<Index>(context, indices, index_layout, destination.dims[dim]);
        switch (width) {
        case 1: launch_index_fill<uint8_t, Index>(context, output, indices, destination, source_layout, dim, value_bits); break;
        case 4: launch_index_fill<uint32_t, Index>(context, output, indices, destination, source_layout, dim, value_bits); break;
        }
    }
    template <class Value, class Index, bool Add>
    void launch_index_update(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipView source, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim) {
        hipLaunchKernelGGL((index_update_kernel<Value, Index, Add>), dim3(blocks(source_layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           reinterpret_cast<const Value*>(static_cast<const char*>(source.buffer->data) + source.offset), destination, source_layout, dim);
        check(hipGetLastError());
    }
    template <class Index>
    void dispatch_index_update(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipView source, unsigned dtype, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, size_t index_count, bool add) {
        LfsHipLayout index_layout{};
        index_layout.rank = 1;
        index_layout.dims[0] = index_layout.elements = index_count;
        index_layout.strides[0] = 1;
        check_indices<Index>(context, indices, index_layout, destination.dims[dim]);
        if (add) {
            if (dtype == LFS_HIP_F32)
                launch_index_update<float, Index, true>(context, output, indices, source, destination, source_layout, dim);
            else
                launch_index_update<int32_t, Index, true>(context, output, indices, source, destination, source_layout, dim);
        } else
            switch (typed_bytes(dtype, 1)) {
            case 1: launch_index_update<uint8_t, Index, false>(context, output, indices, source, destination, source_layout, dim); break;
            case 2: launch_index_update<uint16_t, Index, false>(context, output, indices, source, destination, source_layout, dim); break;
            case 4: launch_index_update<uint32_t, Index, false>(context, output, indices, source, destination, source_layout, dim); break;
            case 8: launch_index_update<uint64_t, Index, false>(context, output, indices, source, destination, source_layout, dim); break;
            }
    }
    __device__ size_t broadcast_index(size_t index, const LfsHipLayout& input, const LfsHipLayout& output) {
        size_t offset = 0;
        for (uint32_t r = output.rank; r; --r) {
            const auto coordinate = index % output.dims[r - 1];
            index /= output.dims[r - 1];
            if (r > output.rank - input.rank) {
                const auto axis = r - 1 - (output.rank - input.rank);
                if (input.dims[axis] != 1)
                    offset += coordinate * input.strides[axis];
            }
        }
        return offset;
    }
    template <class Out, class Op>
    __global__ void broadcast_kernel(const float* lhs, const float* rhs, Out* output, LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, Op op) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.elements; i += size_t(gridDim.x) * blockDim.x)
            output[layout_index(i, out)] = op(lhs[broadcast_index(i, a, out)], rhs[broadcast_index(i, b, out)]);
    }
    template <class Out, class Op>
    void launch_broadcast(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output, LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, Op op) {
        if (!out.elements)
            return;
        hipLaunchKernelGGL((broadcast_kernel<Out, Op>), dim3(blocks(out.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const float*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset),
                           reinterpret_cast<const float*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset),
                           reinterpret_cast<Out*>(static_cast<char*>(output.buffer->data) + output.offset), a, b, out, op);
        check(hipGetLastError());
    }
    template <class T, bool Fill>
    __global__ void strided_kernel(const T* input, T* output, LfsHipLayout layout, T value) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.elements; i += size_t(gridDim.x) * blockDim.x) {
            if constexpr (Fill)
                output[layout_index(i, layout)] = value;
            else
                output[i] = input[layout_index(i, layout)];
        }
    }
    template <class Input, class Output>
    __global__ void scatter_strided_kernel(const Input* input, Output* output, LfsHipLayout layout) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.elements; i += size_t(gridDim.x) * blockDim.x)
            output[layout_index(i, layout)] = static_cast<Output>(input[i]);
    }
    template <class Input, class Output = Input>
    void launch_strided_scatter(LfsHipContext* context, LfsHipView input, LfsHipView output, LfsHipLayout layout) {
        hipLaunchKernelGGL((scatter_strided_kernel<Input, Output>), dim3(blocks(layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const Input*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<Output*>(static_cast<char*>(output.buffer->data) + output.offset), layout);
        check(hipGetLastError());
    }
    template <class T>
    void launch_strided_copy(LfsHipContext* context, LfsHipView input, LfsHipView output, LfsHipLayout layout) {
        hipLaunchKernelGGL((strided_kernel<T, false>), dim3(blocks(layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const T*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<T*>(static_cast<char*>(output.buffer->data) + output.offset), layout, T{});
        check(hipGetLastError());
    }
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

namespace {
    bool dense_layout(const LfsHipLayout& layout) {
        size_t stride = 1;
        for (uint32_t d = layout.rank; d; --d) {
            if (layout.dims[d - 1] > 1 && layout.strides[d - 1] != stride)
                return false;
            stride *= layout.dims[d - 1];
        }
        return true;
    }
    template <class T>
    __global__ void copy_layout_kernel(const T* input, T* output, LfsHipLayout source, LfsHipLayout destination) {
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < source.elements; index += size_t(gridDim.x) * blockDim.x)
            output[layout_index(index, destination)] = input[layout_index(index, source)];
    }
    template <class T>
    void launch_layout_copy(LfsHipContext* context, const LfsHipLayoutCopy& copy) {
        auto* input = reinterpret_cast<const T*>(static_cast<char*>(copy.input.buffer->data) + copy.input.offset);
        auto* output = reinterpret_cast<T*>(static_cast<char*>(copy.output.buffer->data) + copy.output.offset);
        hipLaunchKernelGGL((copy_layout_kernel<T>), dim3(blocks(copy.input_layout.elements)), dim3(256), 0, context->stream, input, output, copy.input_layout, copy.output_layout);
        check(hipGetLastError());
    }
    template <class T>
    struct NonzeroFlag {
        __host__ __device__ uint8_t operator()(T value) const { return value != T(0); }
    };
    template <class Input, class Flags, class T>
    void select_stable(LfsHipContext* context, Input input, Flags flags, T* output, size_t count, size_t capacity, uint64_t* selected) {
        size_t temporary_bytes = 0;
        check(hipcub::DeviceSelect::Flagged(nullptr, temporary_bytes, input, flags, output, static_cast<int*>(nullptr), static_cast<int>(count), context->stream));
        // Write directly only when every input item fits. Otherwise retain
        // results in scratch until the actual count proves capacity is safe.
        const bool direct = capacity >= count;
        const size_t result_bytes = direct ? 0 : ((count * sizeof(T) + 255) / 256) * 256;
        require(temporary_bytes <= SIZE_MAX - result_bytes - 256, "HIP selection workspace overflow");
        const size_t required = temporary_bytes + result_bytes + 256;
        if (required > context->sort_workspace_bytes) {
            check(hipStreamSynchronize(context->stream));
            void* replacement = nullptr;
            check(hipMalloc(&replacement, required));
            if (context->sort_workspace) {
                const auto status = hipFree(context->sort_workspace);
                if (status != hipSuccess) {
                    (void)hipFree(replacement);
                    check(status);
                }
            }
            context->sort_workspace = replacement;
            context->sort_workspace_bytes = required;
        }
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* total = reinterpret_cast<int*>(memory);
        auto* destination = direct ? output : reinterpret_cast<T*>(memory + 256);
        check(hipcub::DeviceSelect::Flagged(memory + 256 + result_bytes, temporary_bytes, input, flags, destination, total, static_cast<int>(count), context->stream));
        int actual = 0;
        check(hipMemcpyAsync(&actual, total, sizeof(actual), hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
        require(actual >= 0 && static_cast<size_t>(actual) <= capacity, "HIP selection output capacity is too small");
        if (!direct && actual)
            check(hipMemcpyAsync(output, destination, size_t(actual) * sizeof(T), hipMemcpyDeviceToDevice, context->stream));
        *selected = static_cast<uint64_t>(actual);
    }
    template <class T>
    void select_values(LfsHipContext* context, const void* input, const uint8_t* flags, void* output, size_t count, size_t capacity, uint64_t* selected) {
        select_stable(context, static_cast<const T*>(input), flags, static_cast<T*>(output), count, capacity, selected);
    }
    template <class T>
    void select_positions(LfsHipContext* context, const void* input, void* output, size_t count, size_t capacity, uint64_t* selected) {
        hipcub::CountingInputIterator<int64_t> positions(0);
        hipcub::TransformInputIterator<uint8_t, NonzeroFlag<T>, const T*> flags(static_cast<const T*>(input), NonzeroFlag<T>{});
        select_stable(context, positions, flags, static_cast<int64_t*>(output), count, capacity, selected);
    }
    template <class T>
    __global__ void masked_scatter_kernel(T* output, const uint8_t* mask, const T* source,
                                          const int* prefix, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            if (mask[i])
                output[i] = source[prefix[i]];
    }
    template <class T>
    void launch_masked_scatter(LfsHipContext* context, void* output, const uint8_t* mask,
                               const void* source, const int* prefix, size_t count) {
        hipLaunchKernelGGL((masked_scatter_kernel<T>), dim3(blocks(count)), dim3(256), 0,
                           context->stream, static_cast<T*>(output), mask,
                           static_cast<const T*>(source), prefix, count);
        check(hipGetLastError());
    }
} // namespace

namespace {
    __global__ void boolean_pointwise(uint32_t op, const uint8_t* lhs, const uint8_t* rhs, uint8_t* output, size_t count, float scalar) {
        const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
        if (i>=count) return;
        const auto a=lhs[i];
        const auto b=rhs ? rhs[i] : uint8_t(0);
        bool result=false;
        switch(op) {
        case LFS_HIP_OP_LogicalNot: result=!a; break;
        case LFS_HIP_OP_LogicalAndTensor: result=a && b; break;
        case LFS_HIP_OP_LogicalOrTensor: result=a || b; break;
        case LFS_HIP_OP_LogicalXorTensor: result=bool(a)!=bool(b); break;
        case LFS_HIP_OP_EqualTensor: result=a==b; break;
        case LFS_HIP_OP_NotEqualTensor: result=a!=b; break;
        case LFS_HIP_OP_EqualScalar: result=float(a)==scalar; break;
        case LFS_HIP_OP_NotEqualScalar: result=float(a)!=scalar; break;
        case LFS_HIP_OP_LessTensor: result=a<b; break;
        case LFS_HIP_OP_LessEqualTensor: result=a<=b; break;
        case LFS_HIP_OP_GreaterTensor: result=a>b; break;
        case LFS_HIP_OP_GreaterEqualTensor: result=a>=b; break;
        case LFS_HIP_OP_LessScalar: result=float(a)<scalar; break;
        case LFS_HIP_OP_LessEqualScalar: result=float(a)<=scalar; break;
        case LFS_HIP_OP_GreaterScalar: result=float(a)>scalar; break;
        case LFS_HIP_OP_GreaterEqualScalar: result=float(a)>=scalar; break;

        }
        output[i]=result ? 1 : 0;
    }
    __global__ void clamp_i32(int32_t* data, size_t count, int32_t minimum, int32_t maximum) {
        const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
        if(i>=count) return;
        auto value=data[i];
        value=value<minimum ? minimum : value;
        data[i]=value>maximum ? maximum : value;
    }
}

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
        require(architecture.substr(0, architecture.find(':')) == LFS_HIP_COMPILED_ARCH,
                "HIP Tensor module was compiled for a different GPU architecture");
        auto result = std::make_unique<LfsHipContext>();
        result->device = ordinal;
        check(hipStreamCreateWithFlags(&result->stream, hipStreamNonBlocking));
        *context = result.release();
    });
}

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
        check(hipMalloc(&result->data, bytes));
        try {
            context->buffers.insert(result.get());
        } catch (...) {
            (void)hipFree(result->data);
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
        require(context->timings.empty(), "Destroy HIP timing pools before their context");
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
        // Asynchronous pinned staging uses its separate event-backed API.
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
        // Asynchronous pinned staging uses its separate event-backed API.
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
}
int lfs_hip_readback_create(LfsHipContext* context, LfsHipReadback** output, LfsHipError* error) {
    return boundary(error, [&] {
        require(output, "Missing HIP readback output");
        *output = nullptr;
        require(context, "Missing HIP readback context");
        DeviceScope device(context->device);
        auto result = std::make_unique<LfsHipReadback>();
        check(hipEventCreateWithFlags(&result->completion, hipEventDisableTiming));
        try {
            context->readbacks.insert(result.get());
        } catch (...) {
            (void)hipEventDestroy(result->completion);
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
        if (bytes > staging->capacity) {
            void* replacement = nullptr;
            check(hipHostMalloc(&replacement, bytes, hipHostMallocDefault));
            if (staging->host) {
                const auto status = hipHostFree(staging->host);
                if (status != hipSuccess) {
                    (void)hipHostFree(replacement);
                    check(status);
                }
            }
            staging->host = replacement;
            staging->capacity = bytes;
        }
        staging->bytes = bytes;
        staging->recorded = false;
        // Even an event-record failure must leave the copy owned until drained.
        staging->pending = true;
        if (bytes)
            check(hipMemcpyAsync(staging->host, static_cast<const char*>(source->data) + offset, bytes, hipMemcpyDeviceToHost, context->stream));
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
        if (bytes > staging->capacity) {
            void* replacement = nullptr;
            check(hipHostMalloc(&replacement, bytes, hipHostMallocDefault));
            if (staging->host) {
                const auto status = hipHostFree(staging->host);
                if (status != hipSuccess) {
                    (void)hipHostFree(replacement);
                    check(status);
                }
            }
            staging->host = replacement;
            staging->capacity = bytes;
        }
        if (bytes)
            std::memcpy(staging->host, source, bytes);
        staging->bytes = bytes;
        staging->recorded = false;
        staging->pending = true;
        if (bytes)
            check(hipMemcpyAsync(static_cast<char*>(destination->data)+offset, staging->host, bytes, hipMemcpyHostToDevice, context->stream));
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
        if (bytes > staging->capacity) {
            void* replacement = nullptr;
            check(hipHostMalloc(&replacement, bytes, hipHostMallocDefault));
            if (staging->host) {
                const auto status = hipHostFree(staging->host);
                if (status != hipSuccess) {
                    (void)hipHostFree(replacement);
                    check(status);
                }
            }
            staging->host = replacement;
            staging->capacity = bytes;
        }
        std::memset(staging->host, 0, bytes);
        *host = reinterpret_cast<uintptr_t>(staging->host);
    });
}
int lfs_hip_readback_append(LfsHipContext* context, LfsHipReadback* staging, const LfsHipBuffer* source, size_t offset, size_t bytes, size_t destination_offset, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        validate(context, source, offset, bytes);
        require(destination_offset <= staging->capacity && bytes <= staging->capacity - destination_offset,
                "HIP packed readback range exceeds reserved host slot");
        DeviceScope device(context->device);
        staging->bytes = staging->capacity;
        staging->pending = true;
        staging->recorded = false;
        if (bytes)
            check(hipMemcpyAsync(static_cast<char*>(staging->host)+destination_offset, static_cast<const char*>(source->data)+offset, bytes, hipMemcpyDeviceToHost, context->stream));
        check(hipEventRecord(staging->completion, context->stream));
        staging->recorded = true;
    });
}
int lfs_hip_readback_wait(LfsHipContext* context, LfsHipReadback* staging, LfsHipError* error) {
    return boundary(error, [&] {
        validate_readback(context, staging);
        DeviceScope device(context->device);
        if (staging->pending)
            check(staging->recorded ? hipEventSynchronize(staging->completion) : hipStreamSynchronize(context->stream));
    });
}
int lfs_hip_readback_poll(LfsHipContext* context, LfsHipReadback* staging, void* destination, int32_t* ready, LfsHipError* error) {
    return boundary(error, [&] {
        require(ready, "Missing HIP readback readiness output");
        *ready = 0;
        validate_readback(context, staging);
        if (!staging->pending)
            return;
        DeviceScope device(context->device);
        const auto status = staging->recorded ? hipEventQuery(staging->completion) : hipStreamQuery(context->stream);
        if (status == hipErrorNotReady)
            return;
        check(status);
        if (destination && staging->bytes)
            std::memcpy(destination, staging->host, staging->bytes);
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

int lfs_hip_boolean_pointwise(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, size_t count, float scalar, LfsHipError* error) {
    return boundary(error,[&] {
        const bool binary=op>=LFS_HIP_OP_EqualTensor && op<=LFS_HIP_OP_LogicalXorTensor;
        require(binary || op==LFS_HIP_OP_LogicalNot || (op>=LFS_HIP_OP_EqualScalar && op<=LFS_HIP_OP_GreaterEqualScalar),"Unsupported HIP byte predicate");
        validate(context,lhs.buffer,lhs.offset,count); validate(context,output.buffer,output.offset,count);
        if(binary) validate(context,rhs.buffer,rhs.offset,count);
        if(!count) return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL(boolean_pointwise,dim3(blocks(count)),dim3(256),0,context->stream,op,
            static_cast<const uint8_t*>(lhs.buffer->data)+lhs.offset,
            binary ? static_cast<const uint8_t*>(rhs.buffer->data)+rhs.offset : nullptr,
            static_cast<uint8_t*>(output.buffer->data)+output.offset,count,scalar);
        check(hipGetLastError());
    });
}
int lfs_hip_clamp_i32(LfsHipContext* context, LfsHipView data, size_t count, int32_t minimum, int32_t maximum, LfsHipError* error) {
    return boundary(error,[&] {
        validate(context,data.buffer,data.offset,float_bytes(count));
        require(data.offset%alignof(int32_t)==0,"Unaligned HIP integer clamp");
        if(!count) return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL(clamp_i32,dim3(blocks(count)),dim3(256),0,context->stream,
            reinterpret_cast<int32_t*>(static_cast<char*>(data.buffer->data)+data.offset),count,minimum,maximum);
        check(hipGetLastError());
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
int lfs_hip_buffer_address(LfsHipContext* context, const LfsHipBuffer* buffer, uint64_t* address, LfsHipError* error) {
    return boundary(error, [&] {
        require(address, "Missing HIP address output");
        *address = 0;
        validate(context, buffer, 0, 0);
        *address = reinterpret_cast<uintptr_t>(buffer->data);
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
#include "hip_tensor_dispatch.inc"
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
#include "hip_tensor_half_dispatch.inc"
        default: throw std::invalid_argument("Unknown HIP pointwise operation");
        }
    });
}
int lfs_hip_broadcast_f32(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, const LfsHipLayout* lhs_layout, const LfsHipLayout* rhs_layout, const LfsHipLayout* output_layout, uint32_t output_element_bytes, LfsHipError* error) {
    return boundary(error, [&] {
        require(lhs_layout && rhs_layout && output_layout && op < LFS_HIP_OP_COUNT, "Invalid HIP broadcast descriptor");
        require((pointwise_flags[op] & 3) == 2, "HIP broadcast requires a binary Tensor operation");
        require(output_element_bytes == ((pointwise_flags[op] & 4) ? 1u : 4u), "HIP broadcast output dtype mismatch");
        const auto a = *lhs_layout, b = *rhs_layout, out = *output_layout;
        const size_t a_bytes = layout_span(a, 4, false), b_bytes = layout_span(b, 4, false);
        const size_t out_bytes = layout_span(out, output_element_bytes, true);
        require(a.rank <= out.rank && b.rank <= out.rank, "HIP broadcast input rank exceeds output");
        for (uint32_t r = 0; r < out.rank; ++r) {
            const auto ad = r < out.rank - a.rank ? 1 : a.dims[r - (out.rank - a.rank)];
            const auto bd = r < out.rank - b.rank ? 1 : b.dims[r - (out.rank - b.rank)];
            require(ad == bd || ad == 1 || bd == 1, "Incompatible HIP broadcast dimensions");
            const auto expected = ad == 1 ? bd : ad;
            require(out.dims[r] == expected, "HIP broadcast output dimensions mismatch");
        }
        validate(context, lhs.buffer, lhs.offset, a_bytes);
        validate(context, rhs.buffer, rhs.offset, b_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(lhs.offset % 4 == 0 && rhs.offset % 4 == 0 && output.offset % output_element_bytes == 0, "Unaligned HIP broadcast view");
        auto disjoint = [&](LfsHipView input, size_t bytes) {
            require(input.buffer != output.buffer || !bytes || !out_bytes || input.offset >= output.offset + out_bytes || output.offset >= input.offset + bytes,
                    "HIP broadcast output must not overlap an input");
        };
        disjoint(lhs, a_bytes);
        disjoint(rhs, b_bytes);
        DeviceScope device(context->device);
        using namespace lfs::core;
        switch (op) {
#include "hip_tensor_broadcast.inc"
        default: throw std::invalid_argument("Unsupported HIP broadcast operation");
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
int lfs_hip_reduce_f32(LfsHipContext* context, LfsHipView input, size_t count, uint32_t operation, float* output, LfsHipError* error) {
    return boundary(error, [&] {
        require(output && operation <= LFS_HIP_REDUCE_HAS_INF, "Invalid HIP reduction descriptor");
        validate(context, input.buffer, input.offset, float_bytes(count));
        require(input.offset % 4 == 0, "Unaligned reduction input");
        require(count || (operation != LFS_HIP_REDUCE_MIN && operation != LFS_HIP_REDUCE_MAX), "HIP extrema require nonempty input");
        if (!count) {
            *output = operation == LFS_HIP_REDUCE_MEAN ? NAN : 0;
            return;
        }
        DeviceScope device(context->device);
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        switch (operation) {
        case LFS_HIP_REDUCE_SUM:
        case LFS_HIP_REDUCE_MEAN: launch_reduce<LFS_HIP_REDUCE_SUM>(context, input, count); break;
        case LFS_HIP_REDUCE_MIN: launch_reduce<LFS_HIP_REDUCE_MIN>(context, input, count); break;
        case LFS_HIP_REDUCE_MAX: launch_reduce<LFS_HIP_REDUCE_MAX>(context, input, count); break;
        case LFS_HIP_REDUCE_HAS_NAN: launch_reduce<LFS_HIP_REDUCE_HAS_NAN>(context, input, count); break;
        case LFS_HIP_REDUCE_HAS_INF: launch_reduce<LFS_HIP_REDUCE_HAS_INF>(context, input, count); break;
        }
        check(hipMemcpyAsync(output, context->reduction_workspace + 4096, sizeof(float), hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
        if (operation == LFS_HIP_REDUCE_MEAN)
            *output /= static_cast<float>(count);
    });
}
namespace {
    struct SegmentOffset {
        int dimension;
        __host__ __device__ int operator()(int index) const { return index * dimension; }
    };
    __global__ void sort_pack(const float* input, float* keys, int64_t* indices, size_t count, size_t dimension, size_t inner) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const size_t segment = i / dimension, column = i % dimension;
            keys[i] = input[(segment / inner * dimension + column) * inner + segment % inner];
            indices[i] = column;
        }
    }
    __global__ void sort_unpack(const float* keys, const int64_t* indices, float* output, int64_t* order, size_t count, size_t dimension, size_t inner) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const size_t segment = i / dimension, column = i % dimension;
            const size_t destination = (segment / inner * dimension + column) * inner + segment % inner;
            output[destination] = keys[i];
            order[destination] = indices[i];
        }
    }
} // namespace
int lfs_hip_copy_layouts(LfsHipContext* context, const LfsHipLayoutCopy* copies, size_t count, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && (!count || copies) && count <= 4096, "Invalid HIP layout copy batch");
        for (size_t i = 0; i < count; ++i) {
            const auto& copy = copies[i];
            const auto width = copy.element_bytes;
            require(width == 1 || width == 2 || width == 4 || width == 8, "Invalid HIP layout copy width");
            const size_t source_bytes = layout_span(copy.input_layout, width, false);
            const size_t destination_bytes = layout_span(copy.output_layout, width, true);
            require(copy.input_layout.elements == copy.output_layout.elements, "HIP layout copy counts differ");
            validate(context, copy.input.buffer, copy.input.offset, source_bytes);
            validate(context, copy.output.buffer, copy.output.offset, destination_bytes);
            require(copy.input.offset % width == 0 && copy.output.offset % width == 0, "Unaligned HIP layout copy");
            require(!source_bytes || copy.input.buffer != copy.output.buffer || copy.input.offset + source_bytes <= copy.output.offset || copy.output.offset + destination_bytes <= copy.input.offset,
                    "HIP layout copy ranges overlap");
        }
        if (!count)
            return;
        DeviceScope device(context->device);
        for (size_t i = 0; i < count; ++i) {
            const auto& copy = copies[i];
            if (!copy.input_layout.elements)
                continue;
            if (dense_layout(copy.input_layout) && dense_layout(copy.output_layout)) {
                check(hipMemcpyAsync(static_cast<char*>(copy.output.buffer->data) + copy.output.offset,
                                     static_cast<char*>(copy.input.buffer->data) + copy.input.offset,
                                     copy.input_layout.elements * copy.element_bytes, hipMemcpyDeviceToDevice, context->stream));
                continue;
            }
            switch (copy.element_bytes) {
            case 1: launch_layout_copy<uint8_t>(context, copy); break;
            case 2: launch_layout_copy<uint16_t>(context, copy); break;
            case 4: launch_layout_copy<uint32_t>(context, copy); break;
            case 8: launch_layout_copy<uint64_t>(context, copy); break;
            }
        }
    });
}

int lfs_hip_select(LfsHipContext* context, LfsHipView input, LfsHipView mask, LfsHipView output, uint32_t dtype, uint32_t kind, size_t count, size_t capacity, uint64_t* selected, LfsHipError* error) {
    return boundary(error, [&] {
        require(selected, "Missing HIP selection count output");
        *selected = 0;
        require(kind <= 1 && count <= INT32_MAX && capacity <= count, "Invalid HIP selection descriptor");
        const size_t width = typed_bytes(dtype, 1), source_bytes = typed_bytes(dtype, count);
        require(!kind || dtype == LFS_HIP_F32 || dtype == LFS_HIP_U8 || dtype == LFS_HIP_BOOL, "Unsupported HIP nonzero type");
        const size_t output_width = kind ? 8 : width, output_bytes = capacity * output_width;
        validate(context, input.buffer, input.offset, source_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % width == 0 && output.offset % output_width == 0, "Unaligned HIP selection view");
        require(input.buffer != output.buffer || input.offset + source_bytes <= output.offset || output.offset + output_bytes <= input.offset,
                "HIP selection input and output overlap");
        if (!kind) {
            validate(context, mask.buffer, mask.offset, count);
            require(mask.buffer != output.buffer || mask.offset + count <= output.offset || output.offset + output_bytes <= mask.offset,
                    "HIP selection mask and output overlap");
        }
        if (!count)
            return;
        DeviceScope device(context->device);
        const void* source = static_cast<char*>(input.buffer->data) + input.offset;
        void* destination = static_cast<char*>(output.buffer->data) + output.offset;
        if (kind) {
            if (dtype == LFS_HIP_F32)
                select_positions<float>(context, source, destination, count, capacity, selected);
            else
                select_positions<uint8_t>(context, source, destination, count, capacity, selected);
        } else {
            const auto* flags = reinterpret_cast<const uint8_t*>(static_cast<char*>(mask.buffer->data) + mask.offset);
            switch (width) {
            case 1: select_values<uint8_t>(context, source, flags, destination, count, capacity, selected); break;
            case 2: select_values<uint16_t>(context, source, flags, destination, count, capacity, selected); break;
            case 4: select_values<uint32_t>(context, source, flags, destination, count, capacity, selected); break;
            case 8: select_values<uint64_t>(context, source, flags, destination, count, capacity, selected); break;
            }
        }
    });
}

int lfs_hip_masked_scatter(LfsHipContext* context, LfsHipView output, LfsHipView mask,
                           LfsHipView source, uint32_t dtype, size_t count,
                           size_t selected_count, LfsHipError* error) {
    return boundary(error, [&] {
        require(count <= INT32_MAX && selected_count <= count, "Invalid HIP masked scatter extent");
        const size_t width = typed_bytes(dtype, 1);
        const size_t output_bytes = typed_bytes(dtype, count);
        const size_t source_bytes = typed_bytes(dtype, selected_count);
        validate(context, output.buffer, output.offset, output_bytes);
        validate(context, mask.buffer, mask.offset, count);
        validate(context, source.buffer, source.offset, source_bytes);
        require(output.offset % width == 0 && source.offset % width == 0,
                "Unaligned HIP masked scatter view");
        const auto disjoint = [](LfsHipView a, size_t a_bytes, LfsHipView b, size_t b_bytes) {
            return a.buffer != b.buffer || !a_bytes || !b_bytes ||
                   a.offset + a_bytes <= b.offset || b.offset + b_bytes <= a.offset;
        };
        require(disjoint(output, output_bytes, source, source_bytes) &&
                    disjoint(output, output_bytes, mask, count) &&
                    disjoint(source, source_bytes, mask, count),
                "HIP masked scatter operands overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        const auto* flags = reinterpret_cast<const uint8_t*>(static_cast<const char*>(mask.buffer->data) + mask.offset);
        hipcub::TransformInputIterator<int, NonzeroFlag<uint8_t>, const uint8_t*> predicate(flags, NonzeroFlag<uint8_t>{});
        size_t temporary_bytes = 0;
        check(hipcub::DeviceScan::ExclusiveSum(nullptr, temporary_bytes, predicate,
                                               static_cast<int*>(nullptr), static_cast<int>(count), context->stream));
        const size_t prefix_bytes = (count * sizeof(int) + 255) / 256 * 256;
        require(temporary_bytes <= SIZE_MAX - prefix_bytes, "HIP masked scatter workspace overflow");
        const size_t required = prefix_bytes + temporary_bytes;
        if (required > context->sort_workspace_bytes) {
            check(hipStreamSynchronize(context->stream));
            void* replacement = nullptr;
            check(hipMalloc(&replacement, required));
            if (context->sort_workspace) {
                const auto status = hipFree(context->sort_workspace);
                if (status != hipSuccess) {
                    (void)hipFree(replacement);
                    check(status);
                }
            }
            context->sort_workspace = replacement;
            context->sort_workspace_bytes = required;
        }
        auto* prefix = static_cast<int*>(context->sort_workspace);
        auto* scratch = static_cast<char*>(context->sort_workspace) + prefix_bytes;
        check(hipcub::DeviceScan::ExclusiveSum(scratch, temporary_bytes, predicate, prefix,
                                               static_cast<int>(count), context->stream));
        check(hipStreamSynchronize(context->stream));
        int last_prefix = 0;
        uint8_t last_flag = 0;
        check(hipMemcpy(&last_prefix, prefix + count - 1, sizeof(last_prefix), hipMemcpyDeviceToHost));
        check(hipMemcpy(&last_flag, flags + count - 1, sizeof(last_flag), hipMemcpyDeviceToHost));
        require(size_t(last_prefix) + size_t(last_flag != 0) == selected_count,
                "HIP masked scatter source count does not match mask");
        if (!selected_count)
            return;
        auto* destination = static_cast<char*>(output.buffer->data) + output.offset;
        const auto* values = static_cast<const char*>(source.buffer->data) + source.offset;
        switch (width) {
        case 1: launch_masked_scatter<uint8_t>(context, destination, flags, values, prefix, count); break;
        case 2: launch_masked_scatter<uint16_t>(context, destination, flags, values, prefix, count); break;
        case 4: launch_masked_scatter<uint32_t>(context, destination, flags, values, prefix, count); break;
        case 8: launch_masked_scatter<uint64_t>(context, destination, flags, values, prefix, count); break;
        }
    });
}

int lfs_hip_sort_pairs(LfsHipContext* context, LfsHipView values, LfsHipView indices, size_t outer, size_t dimension, size_t inner, uint32_t descending, LfsHipError* error) {
    return boundary(error, [&] {
        require(descending <= 1 && dimension <= INT32_MAX && inner <= INT32_MAX && outer <= INT32_MAX, "Invalid HIP sort descriptor");
        require(!inner || outer <= size_t(INT32_MAX) / inner, "HIP sort segment count overflow");
        const size_t segments = outer * inner;
        require(!dimension || segments <= size_t(INT32_MAX) / dimension, "HIP sort item count overflow");
        const size_t count = segments * dimension;
        validate(context, values.buffer, values.offset, count * 4);
        validate(context, indices.buffer, indices.offset, count * 8);
        require(values.offset % 4 == 0 && indices.offset % 8 == 0, "Unaligned HIP sort view");
        require(values.buffer != indices.buffer || values.offset + count * 4 <= indices.offset || indices.offset + count * 8 <= values.offset,
                "HIP sort values and indices overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        auto* output = reinterpret_cast<float*>(static_cast<char*>(values.buffer->data) + values.offset);
        auto* order = reinterpret_cast<int64_t*>(static_cast<char*>(indices.buffer->data) + indices.offset);
        const size_t key_bytes = (count * 4 + 255) / 256 * 256;
        const size_t index_bytes = (count * 8 + 255) / 256 * 256;
        const size_t buffers = 2 * (key_bytes + index_bytes);
        hipcub::CountingInputIterator<int> counting(0);
        hipcub::TransformInputIterator<int, SegmentOffset, decltype(counting)> offsets(counting, SegmentOffset{static_cast<int>(dimension)});
        auto sort = [&](void* temporary, size_t& bytes, float* input_keys, float* output_keys, int64_t* input_indices, int64_t* output_indices) {
            if (segments == 1) {
                if (descending)
                    return hipcub::DeviceRadixSort::SortPairsDescending(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), 0, 32, context->stream);
                return hipcub::DeviceRadixSort::SortPairs(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), 0, 32, context->stream);
            }
            if (descending)
                return hipcub::DeviceSegmentedRadixSort::SortPairsDescending(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), static_cast<int>(segments), offsets, offsets + 1, 0, 32, context->stream);
            return hipcub::DeviceSegmentedRadixSort::SortPairs(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), static_cast<int>(segments), offsets, offsets + 1, 0, 32, context->stream);
        };
        size_t temporary_bytes = 0;
        check(sort(nullptr, temporary_bytes, output, output, order, order));
        require(temporary_bytes <= SIZE_MAX - buffers, "HIP sort workspace overflow");
        const size_t required = buffers + temporary_bytes;
        if (required > context->sort_workspace_bytes) {
            check(hipStreamSynchronize(context->stream));
            void* replacement = nullptr;
            check(hipMalloc(&replacement, required));
            if (context->sort_workspace) {
                const auto status = hipFree(context->sort_workspace);
                if (status != hipSuccess) {
                    (void)hipFree(replacement);
                    check(status);
                }
            }
            context->sort_workspace = replacement;
            context->sort_workspace_bytes = required;
        }
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* input_keys = reinterpret_cast<float*>(memory);
        auto* output_keys = reinterpret_cast<float*>(memory + key_bytes);
        auto* input_indices = reinterpret_cast<int64_t*>(memory + key_bytes * 2);
        auto* output_indices = reinterpret_cast<int64_t*>(memory + key_bytes * 2 + index_bytes);
        hipLaunchKernelGGL(sort_pack, dim3(blocks(count)), dim3(256), 0, context->stream, output, input_keys, input_indices, count, dimension, inner);
        check(hipGetLastError());
        check(sort(memory + buffers, temporary_bytes, input_keys, output_keys, input_indices, output_indices));
        hipLaunchKernelGGL(sort_unpack, dim3(blocks(count)), dim3(256), 0, context->stream, output_keys, output_indices, output, order, count, dimension, inner);
        check(hipGetLastError());
    });
}

int lfs_hip_cumsum(LfsHipContext* context, LfsHipView data, uint32_t dtype, const LfsHipLayout* layout, uint32_t axis, LfsHipError* error) {
    return boundary(error, [&] {
        require(layout && axis < layout->rank, "Invalid HIP scan axis");
        require(dtype == LFS_HIP_F32 || dtype == LFS_HIP_I32 || dtype == LFS_HIP_I64 || dtype == LFS_HIP_U32, "Unsupported HIP scan dtype");
        const unsigned width = dtype == LFS_HIP_I64 ? 8 : 4;
        const size_t bytes = layout_span(*layout, width, true);
        validate(context, data.buffer, data.offset, bytes);
        require(data.offset % width == 0 && layout->elements <= INT32_MAX, "HIP scan alignment or element limit exceeded");
        if (!layout->elements)
            return;
        DeviceScope device(context->device);
        switch (dtype) {
        case LFS_HIP_F32: launch_scan<float>(context, data, *layout, axis); break;
        // Unsigned accumulation defines modulo arithmetic for integer overflow.
        case LFS_HIP_I64: launch_scan<uint64_t>(context, data, *layout, axis); break;
        default: launch_scan<uint32_t>(context, data, *layout, axis); break;
        }
    });
}

int lfs_hip_trim_workspace(LfsHipContext* context, LfsHipError* error) {
    return boundary(error, [&] {
        require(context, "Missing HIP context");
        if (!context->scan_workspace && !context->sort_workspace)
            return;
        DeviceScope device(context->device);
        check(hipStreamSynchronize(context->stream));
        if (context->sort_workspace) {
            check(hipFree(context->sort_workspace));
            context->sort_workspace = nullptr;
            context->sort_workspace_bytes = 0;
        }
        if (context->scan_workspace) {
            check(hipFree(context->scan_workspace));
            context->scan_workspace = nullptr;
            context->scan_workspace_bytes = 0;
        }
    });
}

int lfs_hip_memory_info(LfsHipContext* context, LfsHipMemoryInfo* output, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && output, "Missing HIP memory information output");
        DeviceScope device(context->device);
        size_t free_bytes = 0, total = 0;
        check(hipMemGetInfo(&free_bytes, &total));
        *output = {};
        output->total_bytes = total;
        output->free_bytes = free_bytes;
        output->workspace_bytes = context->reduction_workspace ? reduction_workspace_floats * sizeof(float) : 0;
        output->workspace_bytes += context->scan_workspace_bytes;
        output->workspace_bytes += context->sort_workspace_bytes;
        output->owned_bytes = output->workspace_bytes;
        for (const auto* buffer : context->buffers)
            if (!buffer->external_memory) output->owned_bytes += buffer->bytes;
    });
}
int lfs_hip_chain_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t count, const LfsHipChain* source, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && source->count && source->count <= 16, "Invalid HIP Tensor chain length");
        const size_t bytes = float_bytes(count);
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP Tensor chain");
        auto validate_alias = [&](LfsHipView source) {
            require(source.buffer != output.buffer || source.offset == output.offset || source.offset + bytes <= output.offset || output.offset + bytes <= source.offset,
                    "Overlapping HIP Tensor chain views require identical ranges");
        };
        validate_alias(input);
        Chain chain{};
        chain.count = source->count;
        for (uint32_t i = 0; i < chain.count; ++i) {
            const auto& op = source->ops[i];
            require(op.op < LFS_HIP_OP_COUNT && !(pointwise_flags[op.op] & 4), "Unsupported HIP Tensor chain operation");
            chain.ops[i].op = op.op;
            chain.ops[i].scalar = op.scalar;
            if ((pointwise_flags[op.op] & 3) == 2) {
                validate(context, op.rhs.buffer, op.rhs.offset, bytes);
                require(op.rhs.offset % 4 == 0, "Unaligned HIP Tensor chain rhs");
                validate_alias(op.rhs);
                chain.ops[i].rhs = reinterpret_cast<const float*>(static_cast<const char*>(op.rhs.buffer->data) + op.rhs.offset);
            }
        }
        if (!count)
            return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL(chain_kernel, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset), count, chain);
        check(hipGetLastError());
    });
}

int lfs_hip_strided_scatter(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipLayout* source, uint32_t element_bytes, uint32_t int32_to_float32, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && (element_bytes == 1 || element_bytes == 2 || element_bytes == 4 || element_bytes == 8) && int32_to_float32 <= 1 && (!int32_to_float32 || element_bytes == 4), "Invalid HIP strided scatter descriptor");
        const size_t span = layout_span(*source, element_bytes, true);
        require(source->elements <= SIZE_MAX / element_bytes, "HIP strided scatter input overflow");
        const size_t bytes = source->elements * element_bytes;
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, span);
        require(input.offset % element_bytes == 0 && output.offset % element_bytes == 0, "Unaligned HIP strided scatter");
        require(input.buffer != output.buffer || !bytes || input.offset >= output.offset + span || output.offset >= input.offset + bytes, "HIP strided scatter input/output overlap");
        if (!source->elements)
            return;
        DeviceScope device(context->device);
        if (int32_to_float32)
            launch_strided_scatter<int32_t, float>(context, input, output, *source);
        else
            switch (element_bytes) {
            case 1: launch_strided_scatter<uint8_t>(context, input, output, *source); break;
            case 2: launch_strided_scatter<uint16_t>(context, input, output, *source); break;
            case 4: launch_strided_scatter<uint32_t>(context, input, output, *source); break;
            case 8: launch_strided_scatter<uint64_t>(context, input, output, *source); break;
            }
    });
}
int lfs_hip_strided_copy(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipLayout* source, uint32_t element_bytes, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && (element_bytes == 1 || element_bytes == 2 || element_bytes == 4 || element_bytes == 8), "Invalid HIP strided copy descriptor");
        const size_t span = layout_span(*source, element_bytes, false);
        require(source->elements <= SIZE_MAX / element_bytes, "HIP strided copy output overflow");
        const size_t bytes = source->elements * element_bytes;
        validate(context, input.buffer, input.offset, span);
        validate(context, output.buffer, output.offset, bytes);
        require(input.offset % element_bytes == 0 && output.offset % element_bytes == 0, "Unaligned HIP strided copy");
        require(input.buffer != output.buffer || input.offset + span <= output.offset || output.offset + bytes <= input.offset, "HIP strided copy input/output overlap");
        if (!source->elements)
            return;
        DeviceScope device(context->device);
        switch (element_bytes) {
        case 1: launch_strided_copy<uint8_t>(context, input, output, *source); break;
        case 2: launch_strided_copy<uint16_t>(context, input, output, *source); break;
        case 4: launch_strided_copy<uint32_t>(context, input, output, *source); break;
        case 8: launch_strided_copy<uint64_t>(context, input, output, *source); break;
        }
    });
}
int lfs_hip_random(LfsHipContext* context, LfsHipView output, const LfsHipRandom* program, LfsHipError* error) {
    return boundary(error, [&] {
        require(program && program->kind <= LFS_HIP_RANDOM_NORMAL, "Unknown HIP random distribution");
        const auto p = *program;
        if (p.kind == LFS_HIP_RANDOM_UNIFORM)
            require(isfinite(p.first) && isfinite(p.second) && p.first <= p.second, "HIP uniform bounds must be finite and ordered");
        if (p.kind == LFS_HIP_RANDOM_BERNOULLI)
            require(isfinite(p.first) && p.first >= 0 && p.first <= 1, "HIP Bernoulli probability must be in [0,1]");
        if (p.kind == LFS_HIP_RANDOM_INTEGER)
            require(p.low < p.high, "HIP randint requires low < high");
        if (p.kind == LFS_HIP_RANDOM_NORMAL)
            require(isfinite(p.first) && isfinite(p.second) && p.second >= 0, "HIP normal parameters must be finite with std >= 0");
        validate(context, output.buffer, output.offset, float_bytes(p.count));
        require(output.offset % 4 == 0, "Unaligned HIP random output");
        if (!p.count)
            return;
        DeviceScope device(context->device);
        if ((p.kind == LFS_HIP_RANDOM_UNIFORM && p.first == p.second) ||
            (p.kind == LFS_HIP_RANDOM_NORMAL && p.second == 0) ||
            (p.kind == LFS_HIP_RANDOM_BERNOULLI && (p.first == 0 || p.first == 1))) {
            hipLaunchKernelGGL(fill, dim3(blocks(p.count)), dim3(256), 0, context->stream,
                               reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset), p.count, p.first);
            check(hipGetLastError());
            return;
        }
        switch (p.kind) {
        case LFS_HIP_RANDOM_UNIFORM: launch_random<LFS_HIP_RANDOM_UNIFORM>(context, output, p); break;
        case LFS_HIP_RANDOM_BERNOULLI: launch_random<LFS_HIP_RANDOM_BERNOULLI>(context, output, p); break;
        case LFS_HIP_RANDOM_INTEGER: launch_random<LFS_HIP_RANDOM_INTEGER>(context, output, p); break;
        case LFS_HIP_RANDOM_NORMAL: launch_random<LFS_HIP_RANDOM_NORMAL>(context, output, p); break;
        }
    });
}
int lfs_hip_count_nonzero(LfsHipContext* context, LfsHipView input, uint32_t dtype, size_t count, uint64_t* result, LfsHipError* error) {
    return boundary(error, [&] {
        require(result && (dtype == LFS_HIP_F32 || dtype == LFS_HIP_U8 || dtype == LFS_HIP_BOOL), "HIP count_nonzero requires float32 or byte input and a result");
        *result = 0;
        const size_t bytes = typed_bytes(dtype, count), width = typed_bytes(dtype, 1);
        validate(context, input.buffer, input.offset, bytes);
        require(input.offset % width == 0, "Unaligned HIP count_nonzero input");
        if (!count)
            return;
        DeviceScope device(context->device);
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        auto* total = reinterpret_cast<unsigned long long*>(context->reduction_workspace);
        static_assert(sizeof(*total) == sizeof(*result));
        check(hipMemsetAsync(total, 0, sizeof(*total), context->stream));
        if (dtype == LFS_HIP_F32)
            launch_count_nonzero<float>(context, input, count, total);
        else
            launch_count_nonzero<uint8_t>(context, input, count, total);
        check(hipMemcpyAsync(result, total, sizeof(*result), hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
    });
}
int lfs_hip_convert(LfsHipContext* context, LfsHipView input, uint32_t input_dtype, LfsHipView output, uint32_t output_dtype, size_t count, LfsHipError* error) {
    return boundary(error, [&] {
        const size_t in_bytes = typed_bytes(input_dtype, count), out_bytes = typed_bytes(output_dtype, count);
        const size_t in_width = typed_bytes(input_dtype, 1), out_width = typed_bytes(output_dtype, 1);
        validate(context, input.buffer, input.offset, in_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(input.offset % in_width == 0 && output.offset % out_width == 0, "Unaligned HIP conversion");
        require(input.buffer != output.buffer || !count || input.offset >= output.offset + out_bytes || output.offset >= input.offset + in_bytes ||
                    (input_dtype == output_dtype && input.offset == output.offset),
                "HIP conversion ranges overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        if (input_dtype == output_dtype) {
            if (input.buffer != output.buffer || input.offset != output.offset)
                check(hipMemcpyAsync(static_cast<char*>(output.buffer->data) + output.offset, static_cast<const char*>(input.buffer->data) + input.offset, in_bytes, hipMemcpyDeviceToDevice, context->stream));
            return;
        }
        switch (input_dtype) {
        case LFS_HIP_F32: dispatch_convert<float>(context, input, output, output_dtype, count); break;
        case LFS_HIP_F16: dispatch_convert<__half>(context, input, output, output_dtype, count); break;
        case LFS_HIP_I32: dispatch_convert<int32_t>(context, input, output, output_dtype, count); break;
        case LFS_HIP_I64: dispatch_convert<int64_t>(context, input, output, output_dtype, count); break;
        case LFS_HIP_U8:
        case LFS_HIP_BOOL: dispatch_convert<uint8_t>(context, input, output, output_dtype, count); break;
        case LFS_HIP_U32: dispatch_convert<uint32_t>(context, input, output, output_dtype, count); break;
        }
    });
}
int lfs_hip_index_cast(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t count, size_t extent, LfsHipError* error) {
    return boundary(error, [&] {
        const size_t in_bytes = typed_bytes(LFS_HIP_I64, count), out_bytes = typed_bytes(LFS_HIP_I32, count);
        require(extent <= size_t(INT32_MAX) + 1, "HIP index extent exceeds int32 addressing");
        validate(context, input.buffer, input.offset, in_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(input.offset % 8 == 0 && output.offset % 4 == 0, "Unaligned HIP index cast");
        require(input.buffer != output.buffer || !count || input.offset >= output.offset + out_bytes || output.offset >= input.offset + in_bytes, "HIP index cast ranges overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        auto* flag = reinterpret_cast<uint32_t*>(context->reduction_workspace + 4096);
        check(hipMemsetAsync(flag, 0, sizeof(uint32_t), context->stream));
        hipLaunchKernelGGL(validate_indices, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const int64_t*>(static_cast<const char*>(input.buffer->data) + input.offset), count, extent, flag);
        check(hipGetLastError());
        uint32_t invalid = 0;
        check(hipMemcpyAsync(&invalid, flag, sizeof(invalid), hipMemcpyDeviceToHost, context->stream));
        check(hipStreamSynchronize(context->stream));
        if (invalid)
            throw InvalidIndex("HIP index is negative or outside its extent");
        launch_convert<int32_t, int64_t>(context, input, output, count);
    });
}
int lfs_hip_index_update(LfsHipContext* context, LfsHipView output, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, LfsHipView source, const LfsHipLayout* output_layout, uint32_t dim, size_t index_count, uint32_t add, LfsHipError* error) {
    return boundary(error, [&] {
        require(output_layout && dim < output_layout->rank && add <= 1, "Invalid HIP indexed update descriptor");
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP update indices must be int32 or int64");
        require(!add || dtype == LFS_HIP_F32 || dtype == LFS_HIP_I32, "HIP index_add supports float32 and int32");
        const auto destination = *output_layout;
        const size_t width = typed_bytes(dtype, 1), iw = typed_bytes(index_dtype, 1);
        const size_t output_span = layout_span(destination, static_cast<unsigned>(width), true);
        require(destination.dims[dim] <= uint64_t(INT64_MAX), "HIP update extent exceeds signed index range");
        auto source_layout = destination;
        source_layout.dims[dim] = index_count;
        source_layout.elements = 1;
        for (unsigned r = 0; r < source_layout.rank; ++r)
            if (!source_layout.dims[r])
                source_layout.elements = 0;
        if (source_layout.elements)
            for (unsigned r = 0; r < source_layout.rank; ++r) {
                require(source_layout.dims[r] <= SIZE_MAX / source_layout.elements, "HIP indexed update source shape overflow");
                source_layout.elements *= source_layout.dims[r];
            }
        const size_t source_bytes = typed_bytes(dtype, source_layout.elements), index_bytes = typed_bytes(index_dtype, index_count);
        validate(context, output.buffer, output.offset, output_span);
        validate(context, source.buffer, source.offset, source_bytes);
        validate(context, indices.buffer, indices.offset, index_bytes);
        require(output.offset % width == 0 && source.offset % width == 0 && indices.offset % iw == 0, "Unaligned HIP indexed update");
        auto disjoint = [&](LfsHipView input, size_t bytes) { require(input.buffer != output.buffer || !bytes || !output_span || input.offset >= output.offset + output_span || output.offset >= input.offset + bytes, "HIP update output overlaps an input"); };
        disjoint(source, source_bytes);
        disjoint(indices, index_bytes);
        if (!source_layout.elements)
            return;
        require(destination.dims[dim] > 0, "HIP indexed update cannot write an empty axis");
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_index_update<int32_t>(context, output, indices, source, dtype, destination, source_layout, dim, index_count, add != 0);
        else
            dispatch_index_update<int64_t>(context, output, indices, source, dtype, destination, source_layout, dim, index_count, add != 0);
    });
}
int lfs_hip_index_fill(LfsHipContext* context, LfsHipView output, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, const LfsHipLayout* output_layout, uint32_t dim, size_t index_count, uint64_t value_bits, LfsHipError* error) {
    return boundary(error, [&] {
        require(output_layout && dim < output_layout->rank, "Invalid HIP index fill descriptor");
        require(dtype == LFS_HIP_F32 || dtype == LFS_HIP_I32 || dtype == LFS_HIP_U8 || dtype == LFS_HIP_BOOL,
                "HIP index fill supports Float32, Int32 and byte data");
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP fill indices must be int32 or int64");
        const auto destination = *output_layout;
        const size_t width = typed_bytes(dtype, 1), iw = typed_bytes(index_dtype, 1);
        const size_t output_span = layout_span(destination, static_cast<unsigned>(width), true);
        auto source_layout = destination;
        source_layout.dims[dim] = index_count;
        source_layout.elements = index_count;
        for (unsigned r = 0; r < source_layout.rank; ++r)
            if (r != dim) {
                require(source_layout.dims[r] == 0 || source_layout.elements <= SIZE_MAX / source_layout.dims[r], "HIP index fill shape overflow");
                source_layout.elements *= source_layout.dims[r];
            }
        validate(context, output.buffer, output.offset, output_span);
        validate(context, indices.buffer, indices.offset, typed_bytes(index_dtype, index_count));
        require(output.offset % width == 0 && indices.offset % iw == 0, "Unaligned HIP index fill");
        require(output.buffer != indices.buffer || !output_span || !index_count ||
                    indices.offset >= output.offset + output_span || output.offset >= indices.offset + typed_bytes(index_dtype, index_count),
                "HIP fill output overlaps indices");
        if (!source_layout.elements)
            return;
        require(destination.dims[dim] > 0, "HIP index fill cannot write an empty axis");
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_index_fill<int32_t>(context, output, indices, width, destination, source_layout, dim, index_count, value_bits);
        else
            dispatch_index_fill<int64_t>(context, output, indices, width, destination, source_layout, dim, index_count, value_bits);
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
                    output.buffer != environment.buffer, "HIP environment output overlaps an input");
        DeviceScope device(context->device);
        hipLaunchKernelGGL(environment_composite_kernel, dim3(blocks(pixels)), dim3(256), 0, context->stream,
            reinterpret_cast<const float*>(static_cast<const char*>(rgb.buffer->data) + rgb.offset),
            reinterpret_cast<const float*>(static_cast<const char*>(alpha.buffer->data) + alpha.offset),
            reinterpret_cast<const float*>(static_cast<const char*>(environment.buffer->data) + environment.offset),
            static_cast<uint8_t*>(output.buffer->data) + output.offset, *params, pixels);
        check(hipGetLastError());
    });
}
int lfs_hip_gather(LfsHipContext* context, LfsHipView input, uint32_t dtype, LfsHipView indices, uint32_t index_dtype, LfsHipView output, const LfsHipLayout* input_layout, const LfsHipLayout* index_layout, uint32_t dim, uint32_t boundary_mode, uint32_t index_select, LfsHipError* error) {
    return boundary(error, [&] {
        require(input_layout && index_layout && dim < input_layout->rank && boundary_mode <= 2 && index_select <= 1, "Invalid HIP gather descriptor");
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP indices must be int32 or int64");
        const auto source = *input_layout, idx = *index_layout;
        const size_t width = typed_bytes(dtype, 1), index_width = typed_bytes(index_dtype, 1);
        const size_t input_span = layout_span(source, static_cast<unsigned>(width), false), index_span = layout_span(idx, static_cast<unsigned>(index_width), false);
        LfsHipLayout destination = source;
        if (index_select) {
            require(idx.rank == 1, "HIP index_select requires a vector of indices");
            destination.dims[dim] = idx.elements;
        } else {
            require(idx.rank == source.rank, "HIP gather indices and input must have equal rank");
            destination = idx;
            for (unsigned r = 0; r < source.rank; ++r)
                if (r != dim)
                    require(idx.dims[r] <= source.dims[r], "HIP gather non-index axis exceeds input");
        }
        size_t count = 1;
        for (unsigned r = 0; r < destination.rank; ++r)
            if (!destination.dims[r])
                count = 0;
        if (count)
            for (unsigned r = 0; r < destination.rank; ++r) {
                require(destination.dims[r] <= SIZE_MAX / count, "HIP gather output shape overflow");
                count *= destination.dims[r];
            }
        destination.elements = count;
        require(source.dims[dim] <= uint64_t(INT64_MAX), "HIP gather extent exceeds signed index range");
        require(!count || source.dims[dim] > 0, "HIP gather cannot index an empty axis");
        const size_t output_bytes = typed_bytes(dtype, count);
        validate(context, input.buffer, input.offset, input_span);
        validate(context, indices.buffer, indices.offset, index_span);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % width == 0 && indices.offset % index_width == 0 && output.offset % width == 0, "Unaligned HIP gather views");
        auto disjoint = [&](LfsHipView value, size_t bytes) { require(value.buffer != output.buffer || !bytes || !output_bytes || value.offset >= output.offset + output_bytes || output.offset >= value.offset + bytes, "HIP gather output overlaps an input"); };
        disjoint(input, input_span);
        disjoint(indices, index_span);
        if (!count)
            return;
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_gather<int32_t>(context, input, indices, output, static_cast<unsigned>(width), source, idx, destination, dim, boundary_mode, index_select != 0);
        else
            dispatch_gather<int64_t>(context, input, indices, output, static_cast<unsigned>(width), source, idx, destination, dim, boundary_mode, index_select != 0);
    });
}
int lfs_hip_reduce_layout_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, const LfsHipLayout* source, uint32_t axes_mask, uint32_t operation, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && operation <= LFS_HIP_REDUCE_MAX, "Unsupported HIP axis reduction");
        const size_t input_bytes = layout_span(*source, 4, false);
        require((axes_mask >> source->rank) == 0, "HIP reduction axis exceeds rank");
        if (!axes_mask)
            axes_mask = (1u << source->rank) - 1;
        LfsHipLayout kept{}, reduced{};
        kept.elements = reduced.elements = 1;
        auto append = [&](LfsHipLayout& layout, unsigned axis) {
            layout.dims[layout.rank] = source->dims[axis];
            layout.strides[layout.rank++] = source->strides[axis];
        };
        for (unsigned axis = 0; axis < source->rank; ++axis)
            append(axes_mask & (1u << axis) ? reduced : kept, axis);
        auto product = [](LfsHipLayout& layout) {
            for (unsigned i = 0; i < layout.rank; ++i) {
                if (!layout.dims[i]) {
                    layout.elements = 0;
                    return;
                }
            }
            for (unsigned i = 0; i < layout.rank; ++i) {
                require(layout.dims[i] <= SIZE_MAX / layout.elements, "HIP reduction shape overflow");
                layout.elements *= layout.dims[i];
            }
        };
        product(kept);
        product(reduced);
        require(!kept.elements || reduced.elements || (operation != LFS_HIP_REDUCE_MIN && operation != LFS_HIP_REDUCE_MAX), "HIP extrema reduction requires nonempty axis");
        const size_t output_bytes = float_bytes(kept.elements);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP reduction view");
        require(input.buffer != output.buffer || !input_bytes || !output_bytes || input.offset >= output.offset + output_bytes || output.offset >= input.offset + input_bytes, "HIP reduction input/output overlap");
        if (!kept.elements)
            return;
        DeviceScope device(context->device);
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
        const auto* in = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        auto* out = reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset);
        switch (operation) {
        case LFS_HIP_REDUCE_SUM: launch_axis<LFS_HIP_REDUCE_SUM>(context, in, out, kept, reduced, false); break;
        case LFS_HIP_REDUCE_MEAN: launch_axis<LFS_HIP_REDUCE_SUM>(context, in, out, kept, reduced, true); break;
        case LFS_HIP_REDUCE_MIN: launch_axis<LFS_HIP_REDUCE_MIN>(context, in, out, kept, reduced, false); break;
        case LFS_HIP_REDUCE_MAX: launch_axis<LFS_HIP_REDUCE_MAX>(context, in, out, kept, reduced, false); break;
        }
    });
}
int lfs_hip_fill_layout_bits(LfsHipContext* context, LfsHipView output, const LfsHipLayout* layout, uint32_t width, uint64_t bits, LfsHipError* error) {
    return boundary(error,[&] {
        require(layout && (width==1 || width==2 || width==4 || width==8),"Invalid HIP typed fill descriptor");
        const auto span=layout_span(*layout,width,true);
        validate(context,output.buffer,output.offset,span);
        require(output.offset%width==0,"Unaligned HIP typed fill");
        if(!layout->elements) return;
        DeviceScope device(context->device);
        auto launch=[&]<class T>() {
            hipLaunchKernelGGL((strided_kernel<T,true>),dim3(blocks(layout->elements)),dim3(256),0,context->stream,
                static_cast<const T*>(nullptr),reinterpret_cast<T*>(static_cast<char*>(output.buffer->data)+output.offset),*layout,static_cast<T>(bits));
        };
        switch(width) {
        case 1: launch.template operator()<uint8_t>(); break;
        case 2: launch.template operator()<uint16_t>(); break;
        case 4: launch.template operator()<uint32_t>(); break;
        case 8: launch.template operator()<uint64_t>(); break;
        }
        check(hipGetLastError());
    });
}
int lfs_hip_fill_strided_f32(LfsHipContext* context, LfsHipView output, const LfsHipLayout* source, float value, LfsHipError* error) {
    return boundary(error, [&] {
        require(source, "Missing HIP strided fill layout");
        const size_t span = layout_span(*source, 4, true);
        validate(context, output.buffer, output.offset, span);
        require(output.offset % 4 == 0, "Unaligned HIP strided fill");
        if (!source->elements)
            return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL((strided_kernel<float, true>), dim3(blocks(source->elements)), dim3(256), 0, context->stream,
                           nullptr, reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset), *source, value);
        check(hipGetLastError());
    });
}
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
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<uint8_t>), grid, block, 0, context->stream,
                                   reinterpret_cast<const uint8_t*>(source), destination, 3, y, end, quadrature, *lens);
            else
                hipLaunchKernelGGL((hip_lens::undistort_image_kernel<float>), grid, block, 0, context->stream,
                                   reinterpret_cast<const float*>(source), destination, 3, y, end, quadrature, *lens);
            check(hipGetLastError());
        }
    });
}

#include "shared_image_runtime.hpp"

#include "shared_resize_runtime.hpp"

#include "shared_warp_runtime.hpp"
#include "inverse_warp_runtime.hpp"
#include "inference_runtime.hpp"
#include "gemm_runtime.hpp"
#include "integer_reduce_runtime.hpp"
#include "bias_runtime.hpp"

#include "sh_codec_runtime.hpp"
#include "chain_reduce_runtime.hpp"
#include "device_identity_runtime.hpp"
#include "pool_runtime.hpp"
#include "where_runtime.hpp"
#include "external_memory_runtime.hpp"
#include "point_filter_runtime.hpp"
#include "predicate_runtime.hpp"
#include "timing_runtime.hpp"
