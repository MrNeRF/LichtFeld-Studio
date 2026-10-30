/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "predicate_api.h"
#include <hipcub/hipcub.hpp>

namespace {
    // Queue scalar copies into owned staging: failed synchronization must not
    // leave a write targeting a caller's stack or expose an incomplete result.
    void read_device_scalar(LfsHipContext* context, void* output, const void* input, size_t bytes,
                            void* second_output = nullptr, const void* second_input = nullptr, size_t second_bytes = 0) {
        require(output && input && bytes && bytes <= 16 && second_bytes <= 16 - bytes &&
                    (!second_bytes || (second_output && second_input)),
                "Invalid HIP scalar readback size");
        if (!context->scalar_readback_host)
            check(hipHostMalloc(&context->scalar_readback_host, 16, hipHostMallocDefault));
        check(hipMemcpyAsync(context->scalar_readback_host, input, bytes, hipMemcpyDeviceToHost, context->stream));
        if (second_bytes)
            check(hipMemcpyAsync(static_cast<char*>(context->scalar_readback_host) + bytes, second_input, second_bytes, hipMemcpyDeviceToHost, context->stream));
        synchronize_context(context);
        std::memcpy(output, context->scalar_readback_host, bytes);
        if (second_bytes)
            std::memcpy(second_output, static_cast<const char*>(context->scalar_readback_host) + bytes, second_bytes);
    }

    // Workspace data are disposable once prior users have drained. Release
    // before growing so even a failed free cannot orphan a new allocation.
    void ensure_workspace(LfsHipContext* context, void*& memory, size_t& capacity, size_t required) {
        if (required <= capacity)
            return;
        synchronize_context(context);
        if (memory) {
            check(hipFree(memory));
            memory = nullptr;
            capacity = 0;
        }
        void* replacement = nullptr;
        check(hipMalloc(&replacement, required));
        memory = replacement;
        capacity = required;
    }
    struct SegmentOffset {
        int dimension;
        __host__ __device__ int operator()(int index) const { return index * dimension; }
    };
    __global__ void sort_pack(const float* input, float* keys, float* originals, int64_t* indices, size_t count, size_t dimension, size_t inner, bool descending) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const size_t segment = i / dimension, column = i % dimension;
            const float value = input[(segment / inner * dimension + column) * inner + segment % inner];
            originals[i] = value;
            // The public comparator treats all NaNs, and both signed zeros,
            // as equivalent keys. Keep their original bits in separate storage.
            keys[i] = isnan(value) ? __uint_as_float(0x7fc00000u) : value == 0.f ? 0.f
                                                                                 : value;
            if (descending)
                keys[i] = -keys[i];
            indices[i] = static_cast<int64_t>(column);
        }
    }
    __global__ void sort_unpack(const float* originals, const int64_t* indices, float* output, int64_t* order, size_t count, size_t dimension, size_t inner) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const size_t segment = i / dimension, column = i % dimension;
            const size_t destination = (segment / inner * dimension + column) * inner + segment % inner;
            output[destination] = originals[segment * dimension + static_cast<size_t>(indices[i])];
            order[destination] = indices[i];
        }
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
    template <class To, class From>
    void convert_supported(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t count) {
        constexpr bool floating_target = std::is_same_v<To, float> || std::is_same_v<To, __half>;
        constexpr bool integer_target = std::is_same_v<To, int32_t> || std::is_same_v<To, int64_t> || std::is_same_v<To, uint8_t>;
        constexpr bool supported =
            (std::is_same_v<From, float> && (integer_target || std::is_same_v<To, __half> || std::is_same_v<To, uint32_t>)) ||
            ((std::is_same_v<From, __half> || std::is_same_v<From, int32_t> || std::is_same_v<From, int64_t>) && (floating_target || integer_target) && !std::is_same_v<From, To>) ||
            (std::is_same_v<From, uint8_t> && (floating_target || std::is_same_v<To, int32_t> || std::is_same_v<To, int64_t> || std::is_same_v<To, bool>)) ||
            (std::is_same_v<From, uint32_t> && (std::is_same_v<To, float> || std::is_same_v<To, int64_t>));
        if constexpr (supported)
            launch_convert<To, From>(context, input, output, count);
        else
            throw std::invalid_argument("Unsupported HIP Tensor conversion pair");
    }
    template <class From>
    void dispatch_convert(LfsHipContext* context, LfsHipView input, LfsHipView output, uint32_t dtype, size_t count) {
        switch (dtype) {
        case LFS_HIP_F32: convert_supported<float, From>(context, input, output, count); break;
        case LFS_HIP_F16: convert_supported<__half, From>(context, input, output, count); break;
        case LFS_HIP_I32: convert_supported<int32_t, From>(context, input, output, count); break;
        case LFS_HIP_I64: convert_supported<int64_t, From>(context, input, output, count); break;
        case LFS_HIP_U8: convert_supported<uint8_t, From>(context, input, output, count); break;
        case LFS_HIP_BOOL: convert_supported<bool, From>(context, input, output, count); break;
        case LFS_HIP_U32: convert_supported<uint32_t, From>(context, input, output, count); break;
        }
    }
    __global__ void index_cast_kernel(const int64_t* input, int32_t* output, size_t count, size_t extent, uint32_t* invalid) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            if (checked_index(input[i], extent, invalid))
                output[i] = static_cast<int32_t>(input[i]);
    }
    template <class Input, class Counter>
    __global__ void count_nonzero_kernel(Input input, size_t count, Counter* total) {
        __shared__ Counter partial[256];
        Counter local = 0;
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            local += static_cast<double>(input[static_cast<std::conditional_t<std::is_pointer_v<Input>, size_t, int64_t>>(i)]) != 0;
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
        hipLaunchKernelGGL((count_nonzero_kernel<const T*, unsigned long long>), dim3(blocks(count)), dim3(256), 0, context->stream,
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
    size_t shape_elements(const LfsHipLayout& layout) {
        require(layout.rank <= 8, "HIP Tensor rank exceeds eight dimensions");
        for (uint32_t i = 0; i < layout.rank; ++i)
            if (!layout.dims[i])
                return 0;
        size_t count = 1;
        for (uint32_t i = 0; i < layout.rank; ++i) {
            require(layout.dims[i] <= SIZE_MAX / count, "HIP Tensor shape overflow");
            count *= layout.dims[i];
        }
        return count;
    }
    LfsHipLayout contiguous_layout(LfsHipLayout layout) {
        size_t stride = 1;
        for (uint32_t i = layout.rank; i; --i) {
            layout.strides[i - 1] = stride;
            if (layout.elements)
                stride *= layout.dims[i - 1];
        }
        return layout;
    }
    void split_reduction_layout(const LfsHipLayout& source, uint32_t axes, LfsHipLayout& kept, LfsHipLayout& reduced) {
        kept = {};
        reduced = {};
        kept.elements = reduced.elements = 1;
        for (unsigned axis = 0; axis < source.rank; ++axis) {
            auto& layout = (axes & (1u << axis)) ? reduced : kept;
            layout.dims[layout.rank] = source.dims[axis];
            layout.strides[layout.rank++] = source.strides[axis];
        }
        kept.elements = shape_elements(kept);
        reduced.elements = shape_elements(reduced);
    }
    constexpr size_t reduction_workspace_floats = 4097;
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
    template <unsigned Mode>
    struct FloatReduction {
        using Accumulator = float;
        __device__ float identity() const {
            if constexpr (Mode == LFS_HIP_REDUCE_MIN)
                return INFINITY;
            if constexpr (Mode == LFS_HIP_REDUCE_MAX)
                return -INFINITY;
            return 0;
        }
        __device__ float operator()(float a, float b) const { return combine_reduce<Mode>(a, b); }
    };
    template <class Out, bool Scaled = true>
    struct ReductionStore {
        Out* output;
        float scale = 1;
        template <class T>
        __device__ void operator()(size_t index, T value) const {
            if constexpr (Scaled) {
                if constexpr (std::is_same_v<Out, __half>) {
                    volatile float rounded = value * scale;
                    output[index] = __float2half_rn(rounded);
                } else
                    output[index] = value * scale;
            } else
                output[index] = static_cast<Out>(value);
        }
    };
    template <class Reader, class Policy, class Store>
    __global__ void block_reduction(Reader input, Store output, size_t rows, size_t size, unsigned parts) {
        using Acc = typename Policy::Accumulator;
        __shared__ Acc scratch[256];
        const Policy policy{};
        for (size_t group = blockIdx.x; group < rows * parts; group += gridDim.x) {
            const size_t row = group / parts, part = group % parts;
            Acc value = policy.identity();
            for (size_t i = part * 256 + threadIdx.x; i < size; i += size_t(parts) * 256)
                value = policy(value, input(row, i));
            scratch[threadIdx.x] = value;
            __syncthreads();
            for (unsigned offset = 128; offset; offset >>= 1) {
                if (threadIdx.x < offset)
                    scratch[threadIdx.x] = policy(scratch[threadIdx.x], scratch[threadIdx.x + offset]);
                __syncthreads();
            }
            if (!threadIdx.x)
                output(group, scratch[0]);
            __syncthreads();
        }
    }
    template <class Reader, class Policy, class Store>
    __global__ void serial_reduction(Reader input, Store output, size_t rows, size_t size) {
        const Policy policy{};
        for (size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows; row += size_t(gridDim.x) * blockDim.x) {
            auto value = policy.identity();
            for (size_t i = 0; i < size; ++i)
                value = policy(value, input(row, i));
            output(row, value);
        }
    }
    struct ContiguousReduction {
        const float* input;
        size_t stride;
        __device__ float operator()(size_t row, size_t index) const { return input[row * stride + index]; }
    };
    template <unsigned Mode, bool Final>
    struct ScalarReduction {
        const float* input;
        __device__ float operator()(size_t, size_t index) const {
            const float value = input[index];
            if constexpr (!Final && Mode == LFS_HIP_REDUCE_HAS_NAN)
                return isnan(value) ? 1.f : 0.f;
            if constexpr (!Final && Mode == LFS_HIP_REDUCE_HAS_INF)
                return isinf(value) ? 1.f : 0.f;
            return value;
        }
    };
    void ensure_reduction_workspace(LfsHipContext* context) {
        if (!context->reduction_workspace)
            check(hipMalloc(&context->reduction_workspace, reduction_workspace_floats * sizeof(float)));
    }
    template <unsigned Mode>
    void launch_reduce(LfsHipContext* context, LfsHipView input, size_t count) {
        const auto* data = reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset);
        const unsigned groups = blocks(count);
        hipLaunchKernelGGL((block_reduction<ScalarReduction<Mode, false>, FloatReduction<Mode>, ReductionStore<float, false>>),
                           dim3(groups), dim3(256), 0, context->stream, (ScalarReduction<Mode, false>{data}),
                           (ReductionStore<float, false>{context->reduction_workspace}), size_t(1), count, groups);
        check(hipGetLastError());
        hipLaunchKernelGGL((block_reduction<ScalarReduction<Mode, true>, FloatReduction<Mode>, ReductionStore<float, false>>),
                           dim3(1), dim3(256), 0, context->stream, (ScalarReduction<Mode, true>{context->reduction_workspace}),
                           (ReductionStore<float, false>{context->reduction_workspace + 4096}), size_t(1), size_t(groups), 1u);
        check(hipGetLastError());
    }

    using namespace lfs::core;
    using namespace lfs::core::internal;
    constexpr unsigned pointwise_flags[] = {
#define LFS_POINTWISE_OP(Id, Functor, Name) (hip_pointwise::Lowering<Functor>::mode | (hip_pointwise::Lowering<Functor>::predicate ? 4 : 0)),
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
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
    template <class F>
    __device__ float evaluate_pointwise(float x, float y, float scalar) {
        using L = hip_pointwise::Lowering<F>;
        const auto op = L::make(scalar);
        if constexpr (L::predicate)
            return NAN;
        else if constexpr (L::mode == 2)
            return op(x, y);
        else if constexpr (L::mode == 1)
            return op(x, scalar);
        else
            return op(x);
    }
    __device__ float evaluate_op(uint32_t op, float x, float y, float scalar) {
        using namespace lfs::core;
        using namespace lfs::core::internal;
        switch (op) {
#define LFS_POINTWISE_OP(Id, Functor, Name) \
    case LFS_HIP_OP_##Id: return evaluate_pointwise<Functor>(x, y, scalar);
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
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
            ensure_workspace(context, context->scan_workspace, context->scan_workspace_bytes, required);
            check(hipcub::DeviceScan::InclusiveSum(context->scan_workspace, required, data, data, static_cast<int>(layout.elements), context->stream));
        } else {
            hipLaunchKernelGGL((axis_scan<T>), dim3(static_cast<unsigned>(std::min<size_t>(segments, 4096))), dim3(256), 0, context->stream, data, layout, axis, segments);
            check(hipGetLastError());
        }
    }
    template <class In>
    struct LayoutReduction {
        const In* input;
        LfsHipLayout kept, reduced;
        __device__ float operator()(size_t row, size_t index) const {
            return float(input[layout_index(row, kept) + layout_index(index, reduced)]);
        }
    };
    unsigned reduction_parts(size_t rows, size_t size) {
        return rows < 64 && size > 4096
                   ? static_cast<unsigned>(std::min<size_t>(4096 / rows, (size - 1) / 1024 + 1))
                   : 1;
    }
    template <unsigned Mode, class Out, bool BlockFinal = true, class Reader>
    void launch_float_reduction(LfsHipContext* context, Reader input, Out* output, size_t rows, size_t size, bool mean) {
        const float scale = mean ? (size ? 1.f / float(size) : NAN) : 1.f;
        using Policy = FloatReduction<Mode>;
        using Store = ReductionStore<Out>;
        if (size <= 32) {
            hipLaunchKernelGGL((serial_reduction<Reader, Policy, Store>), dim3(blocks(rows)), dim3(256), 0, context->stream,
                               input, (Store{output, scale}), rows, size);
            check(hipGetLastError());
            return;
        }
        const unsigned parts = reduction_parts(rows, size);
        const unsigned groups = static_cast<unsigned>(std::min<size_t>(4096, rows * parts));
        if (parts > 1) {
            ensure_reduction_workspace(context);
            hipLaunchKernelGGL((block_reduction<Reader, Policy, ReductionStore<float>>), dim3(groups), dim3(256), 0, context->stream,
                               input, (ReductionStore<float>{context->reduction_workspace}), rows, size, parts);
        } else {
            hipLaunchKernelGGL((block_reduction<Reader, Policy, Store>), dim3(groups), dim3(256), 0, context->stream,
                               input, (Store{output, scale}), rows, size, parts);
        }
        check(hipGetLastError());
        if (parts > 1) {
            const ContiguousReduction partial{context->reduction_workspace, parts};
            // Keep the original final-pass order: axis reductions use a block,
            // while contiguous fused chains sum the bounded partials serially.
            if constexpr (BlockFinal)
                hipLaunchKernelGGL((block_reduction<ContiguousReduction, Policy, Store>), dim3(static_cast<unsigned>(rows)), dim3(256), 0, context->stream,
                                   partial, (Store{output, scale}), rows, size_t(parts), 1u);
            else
                hipLaunchKernelGGL((serial_reduction<ContiguousReduction, Policy, Store>), dim3(blocks(rows)), dim3(256), 0, context->stream,
                                   partial, (Store{output, scale}), rows, size_t(parts));
            check(hipGetLastError());
        }
    }
    template <unsigned Mode, class In = float, class Out = float>
    void launch_axis(LfsHipContext* context, const In* input, Out* output, LfsHipLayout kept, LfsHipLayout reduced, bool mean) {
        launch_float_reduction<Mode>(context, LayoutReduction<In>{input, kept, reduced}, output, kept.elements, reduced.elements, mean);
    }
    template <class Index, unsigned Operation, bool Put>
    __global__ void flat_index_kernel(const float* input, const Index* indices, float* output, size_t extent, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            int64_t selected = indices[i];
            if (selected < 0)
                selected += static_cast<int64_t>(extent);
            const size_t position = selected < 0 ? 0 : size_t(selected) >= extent ? extent - 1
                                                                                  : size_t(selected);
            if constexpr (Put) {
                output[position] = input[i];
            } else {
                const float value = input[position];
                if constexpr (Operation == LFS_HIP_OP_Abs)
                    output[i] = fabsf(value);
                else if constexpr (Operation == LFS_HIP_OP_Sqrt)
                    output[i] = sqrtf(value);
                else
                    output[i] = -value;
            }
        }
    }
    template <class Index, unsigned Operation, bool Put>
    void launch_flat_index(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, size_t extent, size_t count) {
        hipLaunchKernelGGL((flat_index_kernel<Index, Operation, Put>), dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const float*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset), extent, count);
        check(hipGetLastError());
    }
    template <class Index, bool Put>
    void dispatch_flat_index(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output,
                             size_t extent, size_t count, unsigned operation) {
        if constexpr (Put) {
            launch_flat_index<Index, 0, true>(context, input, indices, output, extent, count);
        } else {
            switch (operation) {
            case LFS_HIP_OP_Abs: launch_flat_index<Index, LFS_HIP_OP_Abs, false>(context, input, indices, output, extent, count); break;
            case LFS_HIP_OP_Sqrt: launch_flat_index<Index, LFS_HIP_OP_Sqrt, false>(context, input, indices, output, extent, count); break;
            case LFS_HIP_OP_Neg: launch_flat_index<Index, LFS_HIP_OP_Neg, false>(context, input, indices, output, extent, count); break;
            }
        }
    }
    template <bool Put>
    void flat_index(LfsHipContext* context, LfsHipView input, LfsHipView indices, uint32_t index_dtype,
                    LfsHipView output, size_t extent, size_t count, unsigned operation) {
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP flat indices must be Int32 or Int64");
        require(Put || operation == LFS_HIP_OP_Abs || operation == LFS_HIP_OP_Sqrt || operation == LFS_HIP_OP_Neg,
                "Unsupported HIP fused gather unary operation");
        require(extent <= uint64_t(INT64_MAX) && (!count || extent), "Invalid HIP flat index extent");
        const size_t input_bytes = typed_bytes(LFS_HIP_F32, Put ? count : extent);
        const size_t output_bytes = typed_bytes(LFS_HIP_F32, Put ? extent : count);
        const size_t index_width = typed_bytes(index_dtype, 1), index_bytes = typed_bytes(index_dtype, count);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, indices.buffer, indices.offset, index_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % sizeof(float) == 0 && output.offset % sizeof(float) == 0 && indices.offset % index_width == 0,
                "Unaligned HIP flat index views");
        require_disjoint(input, input_bytes, output, output_bytes, "HIP flat index output overlaps an input");
        require_disjoint(indices, index_bytes, output, output_bytes, "HIP flat index output overlaps an input");
        if (!count)
            return;
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_flat_index<int32_t, Put>(context, input, indices, output, extent, count, operation);
        else
            dispatch_flat_index<int64_t, Put>(context, input, indices, output, extent, count, operation);
    }
    template <class Value, class Index>
    __global__ void gather_kernel(const Value* input, const Index* indices, Value* output, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select, uint32_t* invalid) {
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
            if (boundary == 0 && !checked_index(index, static_cast<size_t>(extent), invalid))
                continue;
            output[i] = input[source_offset + size_t(index) * source.strides[dim]];
        }
    }
    template <class Value, class Index>
    void launch_gather(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select) {
        auto* invalid = index_error_flag(context);
        hipLaunchKernelGGL((gather_kernel<Value, Index>), dim3(blocks(destination.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const Value*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset), source, index_layout, destination, dim, boundary, select, invalid);
        check(hipGetLastError());
    }
    template <class Index>
    void dispatch_gather(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, unsigned width, LfsHipLayout source, LfsHipLayout index_layout, LfsHipLayout destination, unsigned dim, unsigned boundary, bool select) {
        switch (width) {
        case 1: launch_gather<uint8_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 2: launch_gather<uint16_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 4: launch_gather<uint32_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        case 8: launch_gather<uint64_t, Index>(context, input, indices, output, source, index_layout, destination, dim, boundary, select); break;
        }
    }
    template <class Value, class Index, bool Add, bool Fill = false>
    __global__ void index_update_kernel(Value* output, const Index* indices, const Value* source, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, uint32_t* invalid, Value constant) {
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
            const int64_t index = indices[selected];
            if (!checked_index(index, destination.dims[dim], invalid))
                continue;
            offset += size_t(index) * destination.strides[dim];
            if constexpr (Add)
                atomicAdd(output + offset, source[i]);
            else
                output[offset] = Fill ? constant : source[i];
        }
    }
    template <class Value, class Index>
    void launch_index_fill(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, uint64_t value_bits) {
        auto* invalid = index_error_flag(context);
        hipLaunchKernelGGL((index_update_kernel<Value, Index, false, true>), dim3(blocks(source_layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           static_cast<const Value*>(nullptr), destination, source_layout, dim, invalid, static_cast<Value>(value_bits));
        check(hipGetLastError());
    }
    template <class Index>
    void dispatch_index_fill(LfsHipContext* context, LfsHipView output, LfsHipView indices, unsigned width, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, uint64_t value_bits) {
        switch (width) {
        case 1: launch_index_fill<uint8_t, Index>(context, output, indices, destination, source_layout, dim, value_bits); break;
        case 4: launch_index_fill<uint32_t, Index>(context, output, indices, destination, source_layout, dim, value_bits); break;
        }
    }
    template <class Value, class Index, bool Add>
    void launch_index_update(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipView source, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim) {
        auto* invalid = index_error_flag(context);
        hipLaunchKernelGGL((index_update_kernel<Value, Index, Add>), dim3(blocks(source_layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<Value*>(static_cast<char*>(output.buffer->data) + output.offset),
                           reinterpret_cast<const Index*>(static_cast<const char*>(indices.buffer->data) + indices.offset),
                           reinterpret_cast<const Value*>(static_cast<const char*>(source.buffer->data) + source.offset), destination, source_layout, dim, invalid, Value{});
        check(hipGetLastError());
    }
    template <class Index>
    void dispatch_index_update(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipView source, unsigned dtype, LfsHipLayout destination, LfsHipLayout source_layout, unsigned dim, bool add) {
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
    template <class Out, class In, class Op>
    __global__ void broadcast_kernel(const In* lhs, const In* rhs, Out* output, LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, Op op) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < out.elements; i += size_t(gridDim.x) * blockDim.x)
            output[layout_index(i, out)] = op(float(lhs[broadcast_index(i, a, out)]), float(rhs[broadcast_index(i, b, out)]));
    }
    template <class Out, class In = float, class Op>
    void launch_broadcast(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output, LfsHipLayout a, LfsHipLayout b, LfsHipLayout out, Op op) {
        if (!out.elements)
            return;
        hipLaunchKernelGGL((broadcast_kernel<Out, In, Op>), dim3(blocks(out.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const In*>(static_cast<const char*>(lhs.buffer->data) + lhs.offset),
                           reinterpret_cast<const In*>(static_cast<const char*>(rhs.buffer->data) + rhs.offset),
                           reinterpret_cast<Out*>(static_cast<char*>(output.buffer->data) + output.offset), a, b, out, op);
        check(hipGetLastError());
    }
    template <class T>
    __global__ void fill_layout_kernel(T* output, LfsHipLayout layout, T value) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < layout.elements; i += size_t(gridDim.x) * blockDim.x)
            output[layout_index(i, layout)] = value;
    }
    bool dense_layout(const LfsHipLayout& layout) {
        size_t stride = 1;
        for (uint32_t d = layout.rank; d; --d) {
            if (layout.dims[d - 1] > 1 && layout.strides[d - 1] != stride)
                return false;
            stride *= layout.dims[d - 1];
        }
        return true;
    }
    template <class T, class Out = T>
    __global__ void copy_layout_kernel(const T* input, Out* output, LfsHipLayout source, LfsHipLayout destination) {
        for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < source.elements; index += size_t(gridDim.x) * blockDim.x)
            output[layout_index(index, destination)] = static_cast<Out>(input[layout_index(index, source)]);
    }
    template <class T>
    void launch_layout_copy(LfsHipContext* context, const LfsHipLayoutCopy& copy) {
        auto* input = reinterpret_cast<const T*>(static_cast<char*>(copy.input.buffer->data) + copy.input.offset);
        auto* output = reinterpret_cast<T*>(static_cast<char*>(copy.output.buffer->data) + copy.output.offset);
        hipLaunchKernelGGL((copy_layout_kernel<T>), dim3(blocks(copy.input_layout.elements)), dim3(256), 0, context->stream, input, output, copy.input_layout, copy.output_layout);
        check(hipGetLastError());
    }
    template <class Input, class Output = Input>
    void launch_strided_scatter(LfsHipContext* context, LfsHipView input, LfsHipView output, LfsHipLayout layout) {
        hipLaunchKernelGGL((copy_layout_kernel<Input, Output>), dim3(blocks(layout.elements)), dim3(256), 0, context->stream,
                           reinterpret_cast<const Input*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<Output*>(static_cast<char*>(output.buffer->data) + output.offset), contiguous_layout(layout), layout);
        check(hipGetLastError());
    }
    template <class T>
    void launch_strided_copy(LfsHipContext* context, LfsHipView input, LfsHipView output, LfsHipLayout layout) {
        launch_layout_copy<T>(context, LfsHipLayoutCopy{input, output, layout, contiguous_layout(layout), sizeof(T)});
    }
    template <class T>
    struct NonzeroFlag {
        __host__ __device__ uint8_t operator()(T value) const { return value != T(0); }
    };
    struct SelectionFlag {
        const void* data;
        uint32_t dtype;
        __host__ __device__ uint8_t operator()(int64_t index) const {
            switch (dtype) {
            case LFS_HIP_F32: return static_cast<const float*>(data)[index] != 0;
            case LFS_HIP_F16: return static_cast<const __half*>(data)[index] != __half(0);
            case LFS_HIP_I32: return static_cast<const int32_t*>(data)[index] != 0;
            case LFS_HIP_I64: return static_cast<const int64_t*>(data)[index] != 0;
            case LFS_HIP_U32: return static_cast<const uint32_t*>(data)[index] != 0;
            default: return static_cast<const uint8_t*>(data)[index] != 0;
            }
        }
    };
    template <class Input, class Flags, class T>
    void select_stable(LfsHipContext* context, Input input, Flags flags, T* output, size_t count, size_t capacity, uint64_t* selected, size_t known_count) {
        size_t temporary_bytes = 0;
        check(hipcub::DeviceSelect::Flagged(nullptr, temporary_bytes, input, flags, output, static_cast<int*>(nullptr), static_cast<int>(count), context->stream));
        require(temporary_bytes <= SIZE_MAX - 256, "HIP selection workspace overflow");
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, temporary_bytes + 256);
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* total = reinterpret_cast<int*>(memory);
        // Validate an exact-sized output before allowing DeviceSelect to write.
        // The extra pass uses a scalar counter instead of a full-sized result copy.
        if (known_count == SIZE_MAX && capacity < count) {
            check(hipMemsetAsync(total, 0, sizeof(*total), context->stream));
            hipLaunchKernelGGL((count_nonzero_kernel<Flags, int>), dim3(blocks(count)), dim3(256), 0, context->stream, flags, count, total);
            check(hipGetLastError());
            int actual = 0;
            read_device_scalar(context, &actual, total, sizeof(actual));
            require(actual >= 0 && static_cast<size_t>(actual) <= capacity, "HIP selection output capacity is too small");
            if (!actual) {
                *selected = 0;
                return;
            }
        }
        check(hipcub::DeviceSelect::Flagged(memory + 256, temporary_bytes, input, flags, output, total, static_cast<int>(count), context->stream));
        if (known_count != SIZE_MAX) {
            *selected = known_count;
            return;
        }
        int actual = 0;
        read_device_scalar(context, &actual, total, sizeof(actual));
        require(actual >= 0 && static_cast<size_t>(actual) <= capacity, "HIP selection output capacity is too small");
        *selected = static_cast<uint64_t>(actual);
    }
    template <class T>
    void select_values(LfsHipContext* context, const void* input, const uint8_t* flags, void* output, size_t count, size_t capacity, uint64_t* selected, size_t known_count) {
        select_stable(context, static_cast<const T*>(input), flags, static_cast<T*>(output), count, capacity, selected, known_count);
    }
    template <class T>
    void select_positions(LfsHipContext* context, const void* input, void* output, size_t count, size_t capacity, uint64_t* selected, size_t known_count) {
        constexpr uint32_t dtype = std::is_same_v<T, float> ? LFS_HIP_F32 : std::is_same_v<T, __half> ? LFS_HIP_F16
                                                                        : std::is_same_v<T, int32_t>  ? LFS_HIP_I32
                                                                        : std::is_same_v<T, int64_t>  ? LFS_HIP_I64
                                                                        : std::is_same_v<T, uint32_t> ? LFS_HIP_U32
                                                                                                      : LFS_HIP_U8;
        hipcub::CountingInputIterator<int64_t> counting(0);
        hipcub::TransformInputIterator<uint8_t, SelectionFlag, decltype(counting)> flags(counting, SelectionFlag{input, dtype});
        select_stable(context, counting, flags, static_cast<int64_t*>(output), count, capacity, selected, known_count);
    }
    template <class T>
    __global__ void masked_scatter_kernel(T* output, const uint8_t* mask, const T* source,
                                          const int* prefix, size_t count, size_t selected_count, uint32_t* invalid) {
        if (size_t(prefix[count - 1]) + size_t(mask[count - 1] != 0) != selected_count) {
            atomicOr(invalid, 2u);
            return;
        }
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             i < count; i += size_t(gridDim.x) * blockDim.x)
            if (mask[i])
                output[i] = source[prefix[i]];
    }
    template <class T>
    void launch_masked_scatter(LfsHipContext* context, void* output, const uint8_t* mask,
                               const void* source, const int* prefix, size_t count, size_t selected_count) {
        auto* invalid = index_error_flag(context);
        hipLaunchKernelGGL((masked_scatter_kernel<T>), dim3(blocks(count)), dim3(256), 0,
                           context->stream, static_cast<T*>(output), mask,
                           static_cast<const T*>(source), prefix, count, selected_count, invalid);
        check(hipGetLastError());
    }
    __global__ void clamp_i32(int32_t* data, size_t count, int32_t minimum, int32_t maximum) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            data[i] = lfs::core::ops::clamp_range_op<int32_t>{minimum, maximum}(data[i]);
        }
    }

    template <class F, class Storage>
    void dispatch_broadcast(LfsHipContext* context, LfsHipView lhs, LfsHipView rhs, LfsHipView output,
                            LfsHipLayout a, LfsHipLayout b, LfsHipLayout out) {
        using L = hip_pointwise::Lowering<F>;
        using Out = std::conditional_t<L::predicate, uint8_t, Storage>;
        if constexpr (L::mode == 2)
            launch_broadcast<Out, Storage>(context, lhs, rhs, output, a, b, out, F{});
        else
            throw std::invalid_argument("Unsupported HIP broadcast operation");
    }

    void launch_integral_broadcast(LfsHipContext*, uint32_t, LfsHipView, LfsHipView, LfsHipView, LfsHipLayout, LfsHipLayout, LfsHipLayout, uint32_t);

} // namespace

int lfs_hip_clamp_i32(LfsHipContext* context, LfsHipView data, size_t count, int32_t minimum, int32_t maximum, LfsHipError* error) {
    return boundary(error, [&] {
        validate(context, data.buffer, data.offset, float_bytes(count));
        require(data.offset % alignof(int32_t) == 0, "Unaligned HIP integer clamp");
        if (!count)
            return;
        DeviceScope device(context->device);
        hipLaunchKernelGGL(clamp_i32, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<int32_t*>(static_cast<char*>(data.buffer->data) + data.offset), count, minimum, maximum);
        check(hipGetLastError());
    });
}

int lfs_hip_scalar_left_typed(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView output, size_t count, float scalar, uint32_t dtype, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && op < LFS_HIP_OP_COUNT && (pointwise_flags[op] & 3) == 1 &&
                    dtype == LFS_HIP_F32,
                "Invalid HIP scalar-left descriptor");
        DeviceScope device(context->device);
        using namespace lfs::core;
        using namespace lfs::core::internal;
        const auto dispatch = [&]<class Storage>() {
            switch (op) {
#define LFS_SCALAR_LEFT(Id, Functor)                                                             \
    case LFS_HIP_OP_##Id:                                                                        \
        launch_pointwise<0, Storage, Storage>(context, lhs, {}, output, count, scalar,           \
                                              ops::scalar_left_op<ops::Functor, float>{scalar}); \
        break;
                LFS_SCALAR_LEFT(AddScalar, add_op)
                LFS_SCALAR_LEFT(SubScalar, sub_op)
                LFS_SCALAR_LEFT(MulScalar, mul_op)
                LFS_SCALAR_LEFT(DivScalar, div_op)
#undef LFS_SCALAR_LEFT
            default: throw std::invalid_argument("Unsupported HIP scalar-left operation");
            }
        };
        dispatch.template operator()<float>();
    });
}

int lfs_hip_broadcast_typed(LfsHipContext* context, uint32_t op, LfsHipView lhs, LfsHipView rhs, LfsHipView output, const LfsHipLayout* lhs_layout, const LfsHipLayout* rhs_layout, const LfsHipLayout* output_layout, uint32_t input_dtype, uint32_t output_element_bytes, LfsHipError* error) {
    return boundary(error, [&] {
        require(input_dtype <= LFS_HIP_U32, "Unsupported HIP broadcast input dtype");
        const uint32_t width = input_dtype == LFS_HIP_F16 ? 2u : input_dtype == LFS_HIP_I64                             ? 8u
                                                             : input_dtype == LFS_HIP_U8 || input_dtype == LFS_HIP_BOOL ? 1u
                                                                                                                        : 4u;
        require(lhs_layout && rhs_layout && output_layout && op < LFS_HIP_OP_COUNT, "Invalid HIP broadcast descriptor");
        require((pointwise_flags[op] & 3) == 2, "HIP broadcast requires a binary Tensor operation");
        require(output_element_bytes == ((pointwise_flags[op] & 4) ? 1u : width), "HIP broadcast output dtype mismatch");
        const auto a = *lhs_layout, b = *rhs_layout, out = *output_layout;
        const size_t a_bytes = layout_span(a, width, false), b_bytes = layout_span(b, width, false);
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
        require(lhs.offset % width == 0 && rhs.offset % width == 0 && output.offset % output_element_bytes == 0, "Unaligned HIP broadcast view");
        require_disjoint(lhs, a_bytes, output, out_bytes, "HIP broadcast output must not overlap an input");
        require_disjoint(rhs, b_bytes, output, out_bytes, "HIP broadcast output must not overlap an input");
        DeviceScope device(context->device);
        using namespace lfs::core;
        using namespace lfs::core::internal;
        if (input_dtype != LFS_HIP_F32 && input_dtype != LFS_HIP_F16) {
            launch_integral_broadcast(context, op, lhs, rhs, output, a, b, out, input_dtype);
        } else if (input_dtype == LFS_HIP_F16) {
            switch (op) {
#define LFS_POINTWISE_OP(Id, Functor, Name) \
    case LFS_HIP_OP_##Id: dispatch_broadcast<Functor, __half>(context, lhs, rhs, output, a, b, out); break;
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
            default: throw std::invalid_argument("Unsupported HIP half broadcast operation");
            }
        } else {
            switch (op) {
#define LFS_POINTWISE_OP(Id, Functor, Name) \
    case LFS_HIP_OP_##Id: dispatch_broadcast<Functor, float>(context, lhs, rhs, output, a, b, out); break;
#include "pointwise_ops.def"
#undef LFS_POINTWISE_OP
            default: throw std::invalid_argument("Unsupported HIP broadcast operation");
            }
        }
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
        ensure_reduction_workspace(context);
        switch (operation) {
        case LFS_HIP_REDUCE_SUM:
        case LFS_HIP_REDUCE_MEAN: launch_reduce<LFS_HIP_REDUCE_SUM>(context, input, count); break;
        case LFS_HIP_REDUCE_MIN: launch_reduce<LFS_HIP_REDUCE_MIN>(context, input, count); break;
        case LFS_HIP_REDUCE_MAX: launch_reduce<LFS_HIP_REDUCE_MAX>(context, input, count); break;
        case LFS_HIP_REDUCE_HAS_NAN: launch_reduce<LFS_HIP_REDUCE_HAS_NAN>(context, input, count); break;
        case LFS_HIP_REDUCE_HAS_INF: launch_reduce<LFS_HIP_REDUCE_HAS_INF>(context, input, count); break;
        }
        read_device_scalar(context, output, context->reduction_workspace + 4096, sizeof(float));
        if (operation == LFS_HIP_REDUCE_MEAN)
            *output /= static_cast<float>(count);
    });
}

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
            require_disjoint(copy.input, source_bytes, copy.output, destination_bytes, "HIP layout copy ranges overlap");
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

int lfs_hip_select(LfsHipContext* context, LfsHipView input, LfsHipView mask, LfsHipView output, const LfsHipSelection* p, uint64_t* selected, LfsHipError* error) {
    return boundary(error, [&] {
        require(p && p->struct_bytes == sizeof(*p) && p->count_known <= 1 && p->count <= SIZE_MAX && p->capacity <= SIZE_MAX,
                "Invalid HIP selection descriptor");
        const auto dtype = p->dtype, kind = p->kind;
        const size_t count = size_t(p->count), capacity = size_t(p->capacity);
        require(!p->count_known || p->selected_count <= capacity, "HIP selection known count exceeds capacity");
        const size_t known_count = p->count_known ? size_t(p->selected_count) : SIZE_MAX;
        require(selected, "Missing HIP selection count output");
        *selected = 0;
        require(kind <= 1 && count <= INT32_MAX && capacity <= count, "Invalid HIP selection descriptor");
        const size_t width = typed_bytes(dtype, 1), source_bytes = typed_bytes(dtype, count);
        const size_t output_width = kind ? 8 : width, output_bytes = capacity * output_width;
        validate(context, input.buffer, input.offset, source_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % width == 0 && output.offset % output_width == 0, "Unaligned HIP selection view");
        require_disjoint(input, source_bytes, output, output_bytes, "HIP selection input and output overlap");
        if (!kind) {
            validate(context, mask.buffer, mask.offset, count);
            require_disjoint(mask, count, output, output_bytes, "HIP selection mask and output overlap");
        }
        if (!count)
            return;
        DeviceScope device(context->device);
        const void* source = static_cast<char*>(input.buffer->data) + input.offset;
        void* destination = static_cast<char*>(output.buffer->data) + output.offset;
        if (kind) {
            switch (dtype) {
            case LFS_HIP_F32: select_positions<float>(context, source, destination, count, capacity, selected, known_count); break;
            case LFS_HIP_F16: select_positions<__half>(context, source, destination, count, capacity, selected, known_count); break;
            case LFS_HIP_I32: select_positions<int32_t>(context, source, destination, count, capacity, selected, known_count); break;
            case LFS_HIP_I64: select_positions<int64_t>(context, source, destination, count, capacity, selected, known_count); break;
            case LFS_HIP_U32: select_positions<uint32_t>(context, source, destination, count, capacity, selected, known_count); break;
            default: select_positions<uint8_t>(context, source, destination, count, capacity, selected, known_count); break;
            }
        } else {
            const auto* flags = reinterpret_cast<const uint8_t*>(static_cast<char*>(mask.buffer->data) + mask.offset);
            switch (width) {
            case 1: select_values<uint8_t>(context, source, flags, destination, count, capacity, selected, known_count); break;
            case 2: select_values<uint16_t>(context, source, flags, destination, count, capacity, selected, known_count); break;
            case 4: select_values<uint32_t>(context, source, flags, destination, count, capacity, selected, known_count); break;
            case 8: select_values<uint64_t>(context, source, flags, destination, count, capacity, selected, known_count); break;
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
        require_disjoint(output, output_bytes, source, source_bytes, "HIP masked scatter operands overlap");
        require_disjoint(output, output_bytes, mask, count, "HIP masked scatter operands overlap");
        require_disjoint(source, source_bytes, mask, count, "HIP masked scatter operands overlap");
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
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        auto* prefix = static_cast<int*>(context->sort_workspace);
        auto* scratch = static_cast<char*>(context->sort_workspace) + prefix_bytes;
        check(hipcub::DeviceScan::ExclusiveSum(scratch, temporary_bytes, predicate, prefix,
                                               static_cast<int>(count), context->stream));
        auto* destination = static_cast<char*>(output.buffer->data) + output.offset;
        const auto* values = static_cast<const char*>(source.buffer->data) + source.offset;
        switch (width) {
        case 1: launch_masked_scatter<uint8_t>(context, destination, flags, values, prefix, count, selected_count); break;
        case 2: launch_masked_scatter<uint16_t>(context, destination, flags, values, prefix, count, selected_count); break;
        case 4: launch_masked_scatter<uint32_t>(context, destination, flags, values, prefix, count, selected_count); break;
        case 8: launch_masked_scatter<uint64_t>(context, destination, flags, values, prefix, count, selected_count); break;
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
        require_disjoint(values, count * 4, indices, count * 8, "HIP sort values and indices overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        auto* output = reinterpret_cast<float*>(static_cast<char*>(values.buffer->data) + values.offset);
        auto* order = reinterpret_cast<int64_t*>(static_cast<char*>(indices.buffer->data) + indices.offset);
        const size_t key_bytes = (count * 4 + 255) / 256 * 256;
        const size_t index_bytes = (count * 8 + 255) / 256 * 256;
        const size_t buffers = 3 * key_bytes + 2 * index_bytes;
        hipcub::CountingInputIterator<int> counting(0);
        hipcub::TransformInputIterator<int, SegmentOffset, decltype(counting)> offsets(counting, SegmentOffset{static_cast<int>(dimension)});
        auto sort = [&](void* temporary, size_t& bytes, float* input_keys, float* output_keys, int64_t* input_indices, int64_t* output_indices) {
            if (segments == 1) {
                return hipcub::DeviceRadixSort::SortPairs(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), 0, 32, context->stream);
            }
            return hipcub::DeviceSegmentedRadixSort::SortPairs(temporary, bytes, input_keys, output_keys, input_indices, output_indices, static_cast<int>(count), static_cast<int>(segments), offsets, offsets + 1, 0, 32, context->stream);
        };
        size_t temporary_bytes = 0;
        check(sort(nullptr, temporary_bytes, output, output, order, order));
        require(temporary_bytes <= SIZE_MAX - buffers, "HIP sort workspace overflow");
        const size_t required = buffers + temporary_bytes;
        ensure_workspace(context, context->sort_workspace, context->sort_workspace_bytes, required);
        auto* memory = static_cast<char*>(context->sort_workspace);
        auto* input_keys = reinterpret_cast<float*>(memory);
        auto* output_keys = reinterpret_cast<float*>(memory + key_bytes);
        auto* originals = reinterpret_cast<float*>(memory + key_bytes * 2);
        auto* input_indices = reinterpret_cast<int64_t*>(memory + key_bytes * 3);
        auto* output_indices = reinterpret_cast<int64_t*>(memory + key_bytes * 3 + index_bytes);
        hipLaunchKernelGGL(sort_pack, dim3(blocks(count)), dim3(256), 0, context->stream, output, input_keys, originals, input_indices, count, dimension, inner, descending);
        check(hipGetLastError());
        check(sort(memory + buffers, temporary_bytes, input_keys, output_keys, input_indices, output_indices));
        hipLaunchKernelGGL(sort_unpack, dim3(blocks(count)), dim3(256), 0, context->stream, originals, output_indices, output, order, count, dimension, inner);
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
        if (!context->scan_workspace && !context->sort_workspace && !context->reduction_workspace)
            return;
        DeviceScope device(context->device);
        synchronize_context(context);
        if (context->reduction_workspace) {
            check(hipFree(context->reduction_workspace));
            context->reduction_workspace = nullptr;
        }
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
        *output = {};
        DeviceScope device(context->device);
        size_t free_bytes = 0, total = 0;
        check(hipMemGetInfo(&free_bytes, &total));
        *output = {};
        output->total_bytes = total;
        output->free_bytes = free_bytes;
        output->workspace_bytes = context->reduction_workspace ? reduction_workspace_floats * sizeof(float) : 0;
        output->workspace_bytes += context->scan_workspace_bytes;
        output->workspace_bytes += context->sort_workspace_bytes;
        output->owned_bytes = output->workspace_bytes + (context->index_error ? sizeof(uint32_t) : 0);
        for (const auto* staging : context->readbacks)
            if (staging->execution_error_device)
                output->owned_bytes += sizeof(uint32_t);
        for (const auto* buffer : context->buffers)
            output->owned_bytes += buffer->bytes;
    });
}

int lfs_hip_chain_f32(LfsHipContext* context, LfsHipView input, LfsHipView output, size_t count, const LfsHipChain* source, LfsHipError* error) {
    return boundary(error, [&] {
        require(source && source->count && source->count <= 16, "Invalid HIP Tensor chain length");
        const size_t bytes = float_bytes(count);
        validate(context, input.buffer, input.offset, bytes);
        validate(context, output.buffer, output.offset, bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP Tensor chain");
        require_compatible_views(input, bytes, output, bytes, true, "Overlapping HIP Tensor chain views require identical ranges");
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
                require_compatible_views(op.rhs, bytes, output, bytes, true, "Overlapping HIP Tensor chain views require identical ranges");
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
        require_disjoint(input, bytes, output, span, "HIP strided scatter input/output overlap");
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
        require_disjoint(input, span, output, bytes, "HIP strided copy input/output overlap");
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
            check(lfs_hip_fill_internal(context, reinterpret_cast<float*>(static_cast<char*>(output.buffer->data) + output.offset), p.count, p.first));
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
        require(result && dtype <= LFS_HIP_U32, "HIP count_nonzero requires a supported dtype and a result");
        *result = 0;
        const size_t bytes = typed_bytes(dtype, count), width = typed_bytes(dtype, 1);
        validate(context, input.buffer, input.offset, bytes);
        require(input.offset % width == 0, "Unaligned HIP count_nonzero input");
        if (!count)
            return;
        DeviceScope device(context->device);
        ensure_reduction_workspace(context);
        auto* total = reinterpret_cast<unsigned long long*>(context->reduction_workspace);
        static_assert(sizeof(*total) == sizeof(*result));
        check(hipMemsetAsync(total, 0, sizeof(*total), context->stream));
        switch (dtype) {
        case LFS_HIP_F32: launch_count_nonzero<float>(context, input, count, total); break;
        case LFS_HIP_F16: launch_count_nonzero<__half>(context, input, count, total); break;
        case LFS_HIP_I32: launch_count_nonzero<int32_t>(context, input, count, total); break;
        case LFS_HIP_I64: launch_count_nonzero<int64_t>(context, input, count, total); break;
        case LFS_HIP_U32: launch_count_nonzero<uint32_t>(context, input, count, total); break;
        default: launch_count_nonzero<uint8_t>(context, input, count, total); break;
        }
        read_device_scalar(context, result, total, sizeof(*result));
    });
}

int lfs_hip_convert(LfsHipContext* context, LfsHipView input, uint32_t input_dtype, LfsHipView output, uint32_t output_dtype, size_t count, LfsHipError* error) {
    return boundary(error, [&] {
        const size_t in_bytes = typed_bytes(input_dtype, count), out_bytes = typed_bytes(output_dtype, count);
        const size_t in_width = typed_bytes(input_dtype, 1), out_width = typed_bytes(output_dtype, 1);
        validate(context, input.buffer, input.offset, in_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(input.offset % in_width == 0 && output.offset % out_width == 0, "Unaligned HIP conversion");
        require_compatible_views(input, in_bytes, output, out_bytes, input_dtype == output_dtype, "HIP conversion ranges overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        if (input_dtype == output_dtype || (input_dtype == LFS_HIP_BOOL && output_dtype == LFS_HIP_U8)) {
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
        require_disjoint(input, in_bytes, output, out_bytes, "HIP index cast ranges overlap");
        if (!count)
            return;
        DeviceScope device(context->device);
        auto* invalid = index_error_flag(context);
        hipLaunchKernelGGL(index_cast_kernel, dim3(blocks(count)), dim3(256), 0, context->stream,
                           reinterpret_cast<const int64_t*>(static_cast<const char*>(input.buffer->data) + input.offset),
                           reinterpret_cast<int32_t*>(static_cast<char*>(output.buffer->data) + output.offset), count, extent, invalid);
        check(hipGetLastError());
    });
}

int lfs_hip_index_update(LfsHipContext* context, LfsHipView output, LfsHipView indices, LfsHipView source, const LfsHipIndexUpdate* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(p && p->struct_bytes == sizeof(*p), "Invalid HIP index_update descriptor size");
        const auto dtype = p->dtype, index_dtype = p->index_dtype, dim = p->dim, add = p->add;
        require(p->index_count <= SIZE_MAX, "HIP update index count overflow");
        const size_t index_count = size_t(p->index_count);
        const auto* output_layout = &p->output_layout;

        require(output_layout && dim < output_layout->rank && add <= 1, "Invalid HIP indexed update descriptor");
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP update indices must be int32 or int64");
        require(!add || dtype == LFS_HIP_F32 || dtype == LFS_HIP_I32, "HIP index_add supports float32 and int32");
        const auto destination = *output_layout;
        const size_t width = typed_bytes(dtype, 1), iw = typed_bytes(index_dtype, 1);
        const size_t output_span = layout_span(destination, static_cast<unsigned>(width), true);
        require(destination.dims[dim] <= uint64_t(INT64_MAX), "HIP update extent exceeds signed index range");
        auto source_layout = destination;
        source_layout.dims[dim] = index_count;
        source_layout.elements = shape_elements(source_layout);
        const size_t source_bytes = typed_bytes(dtype, source_layout.elements), index_bytes = typed_bytes(index_dtype, index_count);
        validate(context, output.buffer, output.offset, output_span);
        validate(context, source.buffer, source.offset, source_bytes);
        validate(context, indices.buffer, indices.offset, index_bytes);
        require(output.offset % width == 0 && source.offset % width == 0 && indices.offset % iw == 0, "Unaligned HIP indexed update");
        require_disjoint(source, source_bytes, output, output_span, "HIP update output overlaps an input");
        require_disjoint(indices, index_bytes, output, output_span, "HIP update output overlaps an input");
        if (!source_layout.elements)
            return;
        require(destination.dims[dim] > 0, "HIP indexed update cannot write an empty axis");
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_index_update<int32_t>(context, output, indices, source, dtype, destination, source_layout, dim, add != 0);
        else
            dispatch_index_update<int64_t>(context, output, indices, source, dtype, destination, source_layout, dim, add != 0);
    });
}

int lfs_hip_index_fill(LfsHipContext* context, LfsHipView output, LfsHipView indices, const LfsHipIndexFill* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(p && p->struct_bytes == sizeof(*p), "Invalid HIP index_fill descriptor size");
        const auto dtype = p->dtype, index_dtype = p->index_dtype, dim = p->dim;
        require(p->index_count <= SIZE_MAX, "HIP fill index count overflow");
        const size_t index_count = size_t(p->index_count);
        const auto value_bits = p->value_bits;
        const auto* output_layout = &p->output_layout;

        require(output_layout && dim < output_layout->rank, "Invalid HIP index fill descriptor");
        require(dtype == LFS_HIP_F32 || dtype == LFS_HIP_I32 || dtype == LFS_HIP_U8 || dtype == LFS_HIP_BOOL,
                "HIP index fill supports Float32, Int32 and byte data");
        require(index_dtype == LFS_HIP_I32 || index_dtype == LFS_HIP_I64, "HIP fill indices must be int32 or int64");
        const auto destination = *output_layout;
        const size_t width = typed_bytes(dtype, 1), iw = typed_bytes(index_dtype, 1);
        const size_t output_span = layout_span(destination, static_cast<unsigned>(width), true);
        auto source_layout = destination;
        source_layout.dims[dim] = index_count;
        source_layout.elements = shape_elements(source_layout);
        validate(context, output.buffer, output.offset, output_span);
        validate(context, indices.buffer, indices.offset, typed_bytes(index_dtype, index_count));
        require(output.offset % width == 0 && indices.offset % iw == 0, "Unaligned HIP index fill");
        require_disjoint(output, output_span, indices, typed_bytes(index_dtype, index_count), "HIP fill output overlaps indices");
        if (!source_layout.elements)
            return;
        require(destination.dims[dim] > 0, "HIP index fill cannot write an empty axis");
        DeviceScope device(context->device);
        if (index_dtype == LFS_HIP_I32)
            dispatch_index_fill<int32_t>(context, output, indices, static_cast<unsigned>(width), destination, source_layout, dim, value_bits);
        else
            dispatch_index_fill<int64_t>(context, output, indices, static_cast<unsigned>(width), destination, source_layout, dim, value_bits);
    });
}

int lfs_hip_gather_unary_f32(LfsHipContext* context, LfsHipView input, LfsHipView indices, uint32_t index_dtype,
                             LfsHipView output, size_t input_size, size_t index_count, uint32_t operation, LfsHipError* error) {
    return boundary(error, [&] { flat_index<false>(context, input, indices, index_dtype, output, input_size, index_count, operation); });
}

int lfs_hip_index_put_f32(LfsHipContext* context, LfsHipView output, LfsHipView indices, uint32_t index_dtype,
                          LfsHipView values, size_t output_size, size_t index_count, LfsHipError* error) {
    return boundary(error, [&] { flat_index<true>(context, values, indices, index_dtype, output, output_size, index_count, 0); });
}

int lfs_hip_gather(LfsHipContext* context, LfsHipView input, LfsHipView indices, LfsHipView output, const LfsHipGather* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(p && p->struct_bytes == sizeof(*p), "Invalid HIP gather descriptor size");
        const auto dtype = p->dtype, index_dtype = p->index_dtype, dim = p->dim;
        const auto boundary_mode = p->boundary_mode, index_select = p->index_select;
        const auto* input_layout = &p->input_layout;
        const auto* index_layout = &p->index_layout;

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
        const size_t count = shape_elements(destination);
        destination.elements = count;
        require(source.dims[dim] <= uint64_t(INT64_MAX), "HIP gather extent exceeds signed index range");
        require(!count || source.dims[dim] > 0, "HIP gather cannot index an empty axis");
        const size_t output_bytes = typed_bytes(dtype, count);
        validate(context, input.buffer, input.offset, input_span);
        validate(context, indices.buffer, indices.offset, index_span);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % width == 0 && indices.offset % index_width == 0 && output.offset % width == 0, "Unaligned HIP gather views");
        require_disjoint(input, input_span, output, output_bytes, "HIP gather output overlaps an input");
        require_disjoint(indices, index_span, output, output_bytes, "HIP gather output overlaps an input");
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
        split_reduction_layout(*source, axes_mask, kept, reduced);
        require(!kept.elements || reduced.elements || (operation != LFS_HIP_REDUCE_MIN && operation != LFS_HIP_REDUCE_MAX), "HIP extrema reduction requires nonempty axis");
        const size_t output_bytes = float_bytes(kept.elements);
        validate(context, input.buffer, input.offset, input_bytes);
        validate(context, output.buffer, output.offset, output_bytes);
        require(input.offset % 4 == 0 && output.offset % 4 == 0, "Unaligned HIP reduction view");
        require_disjoint(input, input_bytes, output, output_bytes, "HIP reduction input/output overlap");
        if (!kept.elements)
            return;
        DeviceScope device(context->device);
        ensure_reduction_workspace(context);
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
    return boundary(error, [&] {
        require(layout && (width == 1 || width == 2 || width == 4 || width == 8), "Invalid HIP typed fill descriptor");
        const auto span = layout_span(*layout, width, true);
        validate(context, output.buffer, output.offset, span);
        require(output.offset % width == 0, "Unaligned HIP typed fill");
        if (!layout->elements)
            return;
        DeviceScope device(context->device);
        auto launch = [&]<class T>() {
            hipLaunchKernelGGL((fill_layout_kernel<T>), dim3(blocks(layout->elements)), dim3(256), 0, context->stream,
                               reinterpret_cast<T*>(static_cast<char*>(output.buffer->data) + output.offset), *layout, static_cast<T>(bits));
        };
        switch (width) {
        case 1: launch.template operator()<uint8_t>(); break;
        case 2: launch.template operator()<uint16_t>(); break;
        case 4: launch.template operator()<uint32_t>(); break;
        case 8: launch.template operator()<uint64_t>(); break;
        }
        check(hipGetLastError());
    });
}

#include "arg_extreme_runtime.hpp"
#include "cdist_runtime.hpp"
#include "chain_reduce_runtime.hpp"
#include "gemm_runtime.hpp"
#include "integer_pointwise_runtime.hpp"
#include "integer_reduce_runtime.hpp"
#include "integral_broadcast_runtime.hpp"
#include "predicate_runtime.hpp"
#include "where_runtime.hpp"
