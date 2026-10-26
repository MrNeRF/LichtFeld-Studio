/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "where_api.h"
namespace {
    void require_layout_alias(LfsHipView input, size_t bytes, const LfsHipLayout& layout,
                              LfsHipView output, size_t out_bytes, const LfsHipLayout& out) {
        if (!views_overlap(input, bytes, output, out_bytes))
            return;
        bool identical = input.offset == output.offset && layout.rank == out.rank;
        for (uint32_t r = 0; r < out.rank && identical; ++r)
            identical = layout.dims[r] == out.dims[r] && layout.strides[r] == out.strides[r];
        require(identical, "HIP where output partially overlaps or broadcasts an aliased input");
    }

    template <class T>
    __global__ void viewer_where(const uint8_t*, const T*, T*, size_t, T);
} // namespace
extern "C" int lfs_hip_masked_fill(LfsHipContext* context, LfsHipView mask, LfsHipView output, uint64_t count, uint32_t dtype, float value, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && count <= SIZE_MAX, "Invalid HIP masked fill descriptor");
        const size_t width = typed_bytes(dtype, 1), bytes = typed_bytes(dtype, count);
        if (!count)
            return;
        validate(context, mask.buffer, mask.offset, count);
        validate(context, output.buffer, output.offset, bytes);
        require(output.offset % width == 0, "Misaligned HIP masked fill output");
        require(mask.buffer != output.buffer || mask.offset >= output.offset + bytes || output.offset >= mask.offset + count, "HIP masked fill aliases its predicate");
        DeviceScope selected(context->device);
        auto* m = static_cast<const uint8_t*>(mask.buffer->data) + mask.offset;
        auto* raw = static_cast<char*>(output.buffer->data) + output.offset;
        auto launch = [&]<class T>(T scalar) { hipLaunchKernelGGL((viewer_where<T>), dim3(blocks(count)), dim3(256), 0, context->stream, m, reinterpret_cast<T*>(raw), reinterpret_cast<T*>(raw), count, scalar); };
        switch (dtype) {
        case LFS_HIP_F32: launch(value); break;
        case LFS_HIP_F16: launch(__float2half_rn(value)); break;
        case LFS_HIP_I32:
            require(std::isfinite(value) && double(value) >= INT32_MIN && double(value) <= INT32_MAX, "HIP masked fill Int32 scalar is out of range");
            launch(static_cast<int32_t>(value));
            break;
        case LFS_HIP_I64:
            require(std::isfinite(value) && double(value) >= -9223372036854775808. && double(value) < 9223372036854775808., "HIP masked fill Int64 scalar is out of range");
            launch(static_cast<int64_t>(value));
            break;
        case LFS_HIP_U8:
            require(std::isfinite(value) && value >= 0 && value <= 255, "HIP masked fill UInt8 scalar is out of range");
            launch(static_cast<uint8_t>(value));
            break;
        case LFS_HIP_U32:
            require(std::isfinite(value) && value >= 0 && double(value) <= UINT32_MAX, "HIP masked fill UInt32 scalar is out of range");
            launch(static_cast<uint32_t>(value));
            break;
        case LFS_HIP_BOOL: launch(static_cast<uint8_t>(value != 0)); break;
        default: require(false, "Unsupported HIP masked fill dtype");
        }
        check(hipGetLastError());
    });
}
namespace {
    template <class T>
    __global__ void viewer_where(const uint8_t* mask, const T* source, T* output, size_t count, T value) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            output[i] = mask[i] ? value : source[i];
    }
} // namespace
namespace {
    template <class T>
    __global__ void tensor_where_layout(const uint8_t* condition, const T* x, const T* y, T* output, LfsHipWhere p) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p.output.elements; i += size_t(gridDim.x) * blockDim.x)
            output[layout_index(i, p.output)] = condition[broadcast_index(i, p.condition, p.output)] ? x[broadcast_index(i, p.x, p.output)] : y[broadcast_index(i, p.y, p.output)];
    }
} // namespace
extern "C" int lfs_hip_where_layout(LfsHipContext* context, LfsHipView condition, LfsHipView x, LfsHipView y, LfsHipView output, const LfsHipWhere* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p), "Invalid HIP where descriptor");
        const auto width = unsigned(typed_bytes(p->dtype, 1));
        const size_t mask_bytes = layout_span(p->condition, 1, false), x_bytes = layout_span(p->x, width, false),
                     y_bytes = layout_span(p->y, width, false), out_bytes = layout_span(p->output, width, true);
        const auto& out = p->output;
        for (const auto* input : {&p->condition, &p->x, &p->y})
            require(input->rank <= out.rank, "HIP where input rank exceeds output");
        for (uint32_t r = 0; r < out.rank; ++r) {
            uint64_t expected = 1;
            for (const auto* input : {&p->condition, &p->x, &p->y}) {
                const auto dimension = r < out.rank - input->rank ? uint64_t{1} : input->dims[r - (out.rank - input->rank)];
                require(dimension == 1 || expected == 1 || dimension == expected, "Incompatible HIP where broadcast dimensions");
                if (expected == 1)
                    expected = dimension;
            }
            require(out.dims[r] == expected, "HIP where output shape mismatch");
        }
        if (!out.elements)
            return;
        validate(context, condition.buffer, condition.offset, mask_bytes);
        validate(context, x.buffer, x.offset, x_bytes);
        validate(context, y.buffer, y.offset, y_bytes);
        validate(context, output.buffer, output.offset, out_bytes);
        require(x.offset % width == 0 && y.offset % width == 0 && output.offset % width == 0, "Misaligned HIP where view");
        require_disjoint(condition, mask_bytes, output, out_bytes, "HIP where output aliases its condition");
        require_layout_alias(x, x_bytes, p->x, output, out_bytes, out);
        require_layout_alias(y, y_bytes, p->y, output, out_bytes, out);
        DeviceScope selected(context->device);
        const auto address = [](LfsHipView view) { return static_cast<char*>(view.buffer->data) + view.offset; };
        auto launch = [&]<class T>() {
            hipLaunchKernelGGL((tensor_where_layout<T>), dim3(blocks(out.elements)), dim3(256), 0, context->stream,
                               reinterpret_cast<const uint8_t*>(address(condition)), reinterpret_cast<const T*>(address(x)),
                               reinterpret_cast<const T*>(address(y)), reinterpret_cast<T*>(address(output)), *p);
        };
        // Select raw storage bits: preserve Float16/Float32 NaN payloads and
        // Int64 values beyond the exact range of floating-point conversion.
        switch (width) {
        case 1: launch.template operator()<uint8_t>(); break;
        case 2: launch.template operator()<uint16_t>(); break;
        case 4: launch.template operator()<uint32_t>(); break;
        case 8: launch.template operator()<uint64_t>(); break;
        }
        check(hipGetLastError());
    });
}
extern "C" int lfs_hip_where_scalar(LfsHipContext* context, LfsHipView mask, LfsHipView source, LfsHipView output, size_t count, uint32_t dtype, float value, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && (dtype == LFS_HIP_F32 || dtype == LFS_HIP_F16), "HIP renderer selection requires Float32 or Float16");
        const size_t width = dtype == LFS_HIP_F32 ? 4 : 2;
        require(count <= SIZE_MAX / width, "HIP renderer selection extent overflow");
        if (!count)
            return;
        validate(context, mask.buffer, mask.offset, count);
        validate(context, source.buffer, source.offset, count * width);
        validate(context, output.buffer, output.offset, count * width);
        require(source.offset % width == 0 && output.offset % width == 0, "Misaligned HIP renderer selection");
        require(!views_overlap(mask, count, output, count * width), "HIP renderer selection output aliases its mask");
        require(!views_overlap(source, count * width, output, count * width) || source.offset == output.offset, "HIP renderer selection source partially overlaps output");
        DeviceScope selected(context->device);
        const auto address = [](LfsHipView view) { return static_cast<char*>(view.buffer->data) + view.offset; };
        const auto* m = reinterpret_cast<const uint8_t*>(address(mask));
        if (dtype == LFS_HIP_F32) {
            hipLaunchKernelGGL((viewer_where<float>), dim3(blocks(count)), dim3(256), 0, context->stream, m, reinterpret_cast<const float*>(address(source)), reinterpret_cast<float*>(address(output)), count, value);
        } else {
            hipLaunchKernelGGL((viewer_where<__half>), dim3(blocks(count)), dim3(256), 0, context->stream, m, reinterpret_cast<const __half*>(address(source)), reinterpret_cast<__half*>(address(output)), count, __float2half_rn(value));
        }
        check(hipGetLastError());
    });
}
