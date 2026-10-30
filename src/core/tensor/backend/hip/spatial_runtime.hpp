/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_point_filter.hpp"
#include "hip_spatial_components.hpp"
namespace {
    namespace hs = lfs::core::internal;
    __global__ void radius_build_native(const float* points, const uint8_t* references,
                                        int32_t* heads, int32_t* next, size_t count, uint32_t mask, float radius) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            const float* p = points + 3 * i;
            next[i] = -1;
            if (references[i] && hs::finite_point(p)) {
                const auto bucket = hs::hash_cell(hs::cell(p[0], radius), hs::cell(p[1], radius), hs::cell(p[2], radius), mask);
                next[i] = atomicExch(heads + bucket, int32_t(i));
            }
        }
    }
    template <unsigned Mode>
    __global__ void radius_query_native(const float* points, const uint8_t* references, const int32_t* heads,
                                        const int32_t* next, void* output, const uint8_t* queries, const void* values, const float* radii,
                                        size_t count, uint32_t mask, float radius, int32_t maximum, bool exclude_self) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            if constexpr (Mode == 0)
                static_cast<uint8_t*>(output)[i] = (!queries || queries[i]) && hs::pointHasNeighbor(points, references, heads, next, i, mask, radius, exclude_self);
            else if constexpr (Mode == 1)
                static_cast<int32_t*>(output)[i] = (!queries || queries[i]) ? hs::pointNeighborCount(points, heads, next, i, mask, radius, maximum) : 0;
            else if constexpr (Mode == 2)
                static_cast<float*>(output)[i] = hs::pointNeighborSpacing(points, heads, next, i, mask, radius);
            else if constexpr (Mode == 3)
                static_cast<int32_t*>(output)[i] = hs::pointNeighborMin(points, static_cast<const int32_t*>(values), heads, next, i, mask, radius, radii);
            else
                static_cast<float*>(output)[i] = hs::pointNeighborMin(points, static_cast<const float*>(values), heads, next, i, mask, radius, radii);
        }
    }
} // namespace
int lfs_hip_radius_query(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output,
                         size_t count, size_t buckets, float radius, uint32_t mode, int32_t maximum, uint32_t exclude_self, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && mode <= 5 && count <= INT32_MAX && buckets &&
                    !(buckets & (buckets - 1)) && buckets <= uint64_t(UINT32_MAX) + 1 &&
                    std::isfinite(radius) && radius > 0 && (mode != 1 || maximum > 0) && exclude_self <= 1,
                "Invalid HIP radius query descriptor");
        if (!count)
            return;
        require(count <= SIZE_MAX / 12 && buckets <= SIZE_MAX / 4, "HIP spatial byte range overflow");
        const size_t sizes[]{count * 12, count, buckets * 4, count * 4, count, count * 4, count * 4};
        const size_t out_bytes = mode == 0 ? count : count * 4;
        validate(context, output.buffer, output.offset, out_bytes);
        require(mode == 0 || output.offset % 4 == 0, "Misaligned HIP spatial output");
        const auto overlaps = [](LfsHipView a, size_t na, LfsHipView b, size_t nb) {
            return a.buffer == b.buffer && a.offset < b.offset + nb && b.offset < a.offset + na;
        };
        for (unsigned i = 0; i < 7; ++i) {
            const bool required = i < 4 || (i == 5 && (mode == 3 || mode == 4));
            require(!required || inputs[i].buffer, "Missing HIP spatial input");
            if (!inputs[i].buffer)
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(i == 1 || i == 4 || inputs[i].offset % 4 == 0, "Misaligned HIP spatial input");
            require(!overlaps(inputs[i], sizes[i], output, out_bytes), "HIP spatial output aliases input");
            for (unsigned j : {2u, 3u})
                if (i != j)
                    require(!overlaps(inputs[i], sizes[i], inputs[j], sizes[j]), "HIP spatial workspace aliases input");
        }
        const auto pointer = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        auto* heads = static_cast<int32_t*>(pointer(inputs[2]));
        auto* next = static_cast<int32_t*>(pointer(inputs[3]));
        const auto* points = static_cast<const float*>(pointer(inputs[0]));
        const auto* refs = static_cast<const uint8_t*>(pointer(inputs[1]));
        DeviceScope device(context->device);
        check(hipMemsetAsync(heads, 0xff, buckets * 4, context->stream));
        hipLaunchKernelGGL(radius_build_native, dim3(blocks(count)), dim3(256), 0, context->stream,
                           points, refs, heads, next, count, uint32_t(buckets - 1), radius);
        check(hipGetLastError());
        const auto launch = [&]<unsigned Mode>() {
            hipLaunchKernelGGL((radius_query_native<Mode>), dim3(blocks(count)), dim3(256), 0, context->stream,
                               points, refs, heads, next, pointer(output), static_cast<const uint8_t*>(pointer(inputs[4])), pointer(inputs[5]),
                               static_cast<const float*>(pointer(inputs[6])), count, uint32_t(buckets - 1), radius, maximum, bool(exclude_self));
        };
        switch (mode) {
        case 0: launch.template operator()<0>(); break;
        case 1: launch.template operator()<1>(); break;
        case 2: launch.template operator()<2>(); break;
        case 3: launch.template operator()<3>(); break;
        case 4: launch.template operator()<4>(); break;
        case 5:
            hipLaunchKernelGGL(hip_spatial_components::union_components, dim3(unsigned((count + 255) / 256)), dim3(256), 0, context->stream,
                               points, refs, heads, next, static_cast<int32_t*>(pointer(output)), count, uint32_t(buckets - 1), radius);
            check(hipGetLastError());
            hipLaunchKernelGGL(hip_spatial_components::flatten_components, dim3(unsigned((count + 255) / 256)), dim3(256), 0, context->stream,
                               static_cast<int32_t*>(pointer(output)), count);
            break;
        }
        check(hipGetLastError());
    });
}
namespace {
    __global__ void proximity_build_native(const float* targets, int32_t* heads, int32_t* next, size_t count, uint32_t mask, float width) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            next[i] = -1;
            const float* p = targets + 3 * i;
            if (hs::finite_point(p))
                next[i] = atomicExch(heads + hs::hash_cell(hs::cell(p[0], width), hs::cell(p[1], width), hs::cell(p[2], width), mask), int32_t(i));
        }
    }
    template <bool Coverage>
    __global__ void proximity_query_native(const float* points, const float* targets, const int32_t* heads, const int32_t* next,
                                           int32_t* output, size_t count, int target_count, uint32_t mask, float parameter) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
            if constexpr (Coverage)
                output[i] = hs::cameraFrustumCount(points + 3 * i, targets, target_count, parameter);
            else
                output[i] = target_count ? hs::nearestPoint(points + 3 * i, targets, heads, next, target_count, mask, parameter) : -1;
        }
    }
} // namespace
int lfs_hip_proximity(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output, size_t queries,
                      size_t targets, size_t buckets, float parameter, uint32_t coverage, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && coverage <= 1 && queries <= INT32_MAX && targets <= INT32_MAX &&
                    buckets && !(buckets & (buckets - 1)) && buckets <= uint64_t(UINT32_MAX) + 1 && std::isfinite(parameter) &&
                    (coverage || parameter > 0),
                "Invalid HIP proximity descriptor");
        if (!queries)
            return;
        require(queries <= SIZE_MAX / 12 && targets <= SIZE_MAX / 64 && buckets <= SIZE_MAX / 4, "HIP proximity byte range overflow");
        const size_t sizes[]{queries * 12, targets * (coverage ? 64 : 12), coverage ? 0 : buckets * 4, coverage ? 0 : targets * 4};
        validate(context, output.buffer, output.offset, queries * 4);
        require(output.offset % 4 == 0, "Misaligned HIP proximity output");
        const auto overlap = [](LfsHipView a, size_t na, LfsHipView b, size_t nb) {
            return na && nb && a.buffer == b.buffer && a.offset < b.offset + nb && b.offset < a.offset + na;
        };
        for (unsigned i = 0; i < 4; ++i) {
            if (!sizes[i])
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(inputs[i].offset % 4 == 0, "Misaligned HIP proximity input");
            require(!overlap(inputs[i], sizes[i], output, queries * 4), "HIP proximity output aliases input");
            if (!coverage)
                for (unsigned j : {2u, 3u})
                    if (i != j)
                        require(!overlap(inputs[i], sizes[i], inputs[j], sizes[j]), "HIP proximity workspace aliases input");
        }
        const auto pointer = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        DeviceScope device(context->device);
        auto* heads = static_cast<int32_t*>(pointer(inputs[2]));
        auto* next = static_cast<int32_t*>(pointer(inputs[3]));
        auto* p = static_cast<const float*>(pointer(inputs[0]));
        auto* t = static_cast<const float*>(pointer(inputs[1]));
        auto* out = static_cast<int32_t*>(pointer(output));
        if (coverage) {
            hipLaunchKernelGGL((proximity_query_native<true>), dim3(blocks(queries)), dim3(256), 0, context->stream,
                               p, t, heads, next, out, queries, int(targets), 0, parameter);
        } else {
            check(hipMemsetAsync(heads, 0xff, buckets * 4, context->stream));
            if (targets) {
                hipLaunchKernelGGL(proximity_build_native, dim3(blocks(targets)), dim3(256), 0, context->stream,
                                   t, heads, next, targets, uint32_t(buckets - 1), parameter);
                check(hipGetLastError());
            }
            hipLaunchKernelGGL((proximity_query_native<false>), dim3(blocks(queries)), dim3(256), 0, context->stream,
                               p, t, heads, next, out, queries, int(targets), uint32_t(buckets - 1), parameter);
        }
        check(hipGetLastError());
    });
}
