/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_point_filter.hpp"
#include "point_filter_api.h"
namespace {
    template <bool NodeOnly>
    __global__ void filter_points_native(lfs::core::internal::PointFilterArgs p) {
        if constexpr (NodeOnly)
            p.flags = LFS_FILTER_NODES;
        constexpr uint32_t elements = NodeOnly ? 4 : 1;
        const uint64_t start = uint64_t(blockIdx.x) * blockDim.x * elements + threadIdx.x;
#pragma unroll
        for (uint32_t element = 0; element < elements; ++element) {
            const uint64_t i = start + element * blockDim.x;
            if (i < p.count && p.mask[i] && !lfs::core::internal::pointPassesFilter(p, uint32_t(i)))
                p.mask[i] = 0;
        }
    }
} // namespace
extern "C" int lfs_hip_filter_points(LfsHipContext* context, LfsHipView mask,
                                     const LfsHipPointFilter* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p), "Invalid HIP point filter descriptor");
        require(!(p->flags & ~uint32_t(63)), "Invalid HIP point filter flags");
        require(p->model == 0 || p->model == 1 || p->model == 3, "Invalid HIP point filter projection");
        if (!p->count)
            return;
        validate(context, mask.buffer, mask.offset, p->count);
        using Args = lfs::core::internal::PointFilterArgs;
        Args a{};
        a.mask = static_cast<uint8_t*>(mask.buffer->data) + mask.offset;
        a.count = p->count;
        a.transform_count = p->transform_count;
        a.allowed_count = p->allowed_count;
        a.flags = p->flags;
        const auto input = [&]<class T>(size_t index, size_t count, bool required) -> const T* {
            const auto view = p->inputs[index];
            require(!required || count == 0 || view.buffer, "Missing HIP point filter input");
            if (!view.buffer)
                return nullptr;
            require(count <= SIZE_MAX / sizeof(T), "HIP point filter range overflows");
            validate(context, view.buffer, view.offset, count * sizeof(T));
            require(view.offset % alignof(T) == 0, "Misaligned HIP point filter input");
            require(view.buffer != mask.buffer || view.offset >= mask.offset + p->count ||
                        mask.offset >= view.offset + count * sizeof(T),
                    "HIP point filter aliases output");
            return reinterpret_cast<const T*>(static_cast<const char*>(view.buffer->data) + view.offset);
        };
        const bool geometry = p->flags & LFS_FILTER_GEOMETRY;
        a.points = input.template operator()<float>(0, size_t(p->count) * 3, geometry);
        a.transforms = input.template operator()<float>(1, size_t(p->transform_count) * 16, false);
        require(!a.transforms || p->transform_count, "Empty HIP point filter transform table");
        a.indices = input.template operator()<int32_t>(2, p->count, p->flags & LFS_FILTER_NODES);
        a.allowed = input.template operator()<uint8_t>(3, p->allowed_count, p->flags & LFS_FILTER_NODES);
        a.box_transform = input.template operator()<float>(4, 16, p->flags & LFS_FILTER_BOX);
        a.box_min = input.template operator()<float>(5, 3, p->flags & LFS_FILTER_BOX);
        a.box_max = input.template operator()<float>(6, 3, p->flags & LFS_FILTER_BOX);
        a.ellipsoid_transform = input.template operator()<float>(7, 16, p->flags & LFS_FILTER_ELLIPSOID);
        a.ellipsoid_radii = input.template operator()<float>(8, 3, p->flags & LFS_FILTER_ELLIPSOID);
        std::copy_n(p->window, 9, a.rotation);
        std::copy_n(p->window + 9, 3, a.translation);
        a.focal_x = p->window[12];
        a.focal_y = p->window[13];
        a.center_x = p->window[14];
        a.center_y = p->window[15];
        a.ortho_scale = p->window[16];
        a.width = p->window[17];
        a.height = p->window[18];
        a.near_depth = p->window[19];
        a.far_depth = p->window[20];
        a.scale_x = p->window[21];
        a.scale_y = p->window[22];
        a.offset_x = p->window[23];
        a.offset_y = p->window[24];
        a.model = static_cast<lfs::core::internal::PointProjectionModel>(p->model);
        if (p->flags & LFS_FILTER_WINDOW)
            require(a.width > 0 && a.height > 0, "Invalid HIP point filter window");
        DeviceScope selected(context->device);
        if (geometry)
            hipLaunchKernelGGL((filter_points_native<false>), dim3((uint64_t(p->count) + 255) / 256), dim3(256), 0, context->stream, a);
        else if (p->flags & LFS_FILTER_NODES)
            hipLaunchKernelGGL((filter_points_native<true>), dim3((uint64_t(p->count) + 1023) / 1024), dim3(256), 0, context->stream, a);
        check(hipGetLastError());
    });
}
namespace {
    template <uint32_t Phase>
    __global__ void update_labels_native(lfs::core::internal::LabelUpdateParams p) {
        const auto row = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (row < p.count)
            lfs::core::internal::updateLabel<Phase>(p, uint32_t(row));
    }
    __global__ void and_live_native(uint8_t* mask, const uint8_t* live, size_t count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            if (!live[i])
                mask[i] = 0;
    }
    __global__ void histogram_labels_native(const uint8_t* values, int32_t* counts, uint32_t count) {
        __shared__ int32_t local[256];
        local[threadIdx.x] = 0;
        __syncthreads();
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
            if (values[i])
                atomicAdd(&local[values[i]], 1);
        __syncthreads();
        if (local[threadIdx.x])
            atomicAdd(&counts[threadIdx.x], local[threadIdx.x]);
    }
} // namespace
extern "C" int lfs_hip_update_labels(LfsHipContext* context, LfsHipView output, LfsHipView selected,
                                     const LfsHipLabelUpdate* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p), "Invalid HIP label descriptor");
        require(p->count <= INT32_MAX && p->output_count <= INT32_MAX && p->allowed_count <= INT32_MAX && p->label <= 255 && p->mode <= 2,
                "Invalid HIP label range or mode");
        require(p->indices.buffer || p->count == p->output_count, "HIP label counts differ without indices");
        if (!p->count || !p->output_count)
            return;
        validate(context, output.buffer, output.offset, p->output_count);
        const auto input = [&]<class T>(LfsHipView view, size_t count, bool required) -> const T* {
            require(!required || !count || view.buffer, "Missing HIP label input");
            if (!view.buffer)
                return nullptr;
            require(count <= SIZE_MAX / sizeof(T), "HIP label range overflows");
            validate(context, view.buffer, view.offset, count * sizeof(T));
            require(view.offset % alignof(T) == 0, "Misaligned HIP label input");
            require(view.buffer != output.buffer || view.offset >= output.offset + p->output_count ||
                        output.offset >= view.offset + count * sizeof(T),
                    "HIP label input aliases output");
            return reinterpret_cast<const T*>(static_cast<const char*>(view.buffer->data) + view.offset);
        };
        lfs::core::internal::LabelUpdateParams a{};
        a.output = static_cast<uint8_t*>(output.buffer->data) + output.offset;
        a.selected = input.template operator()<uint8_t>(selected, p->count, true);
        a.existing = input.template operator()<uint8_t>(p->existing, p->output_count, false);
        a.locked = input.template operator()<uint8_t>(p->locked, 256, false);
        a.indices = input.template operator()<int32_t>(p->indices, p->count, false);
        a.categories = input.template operator()<int32_t>(p->categories, p->count, false);
        a.allowed = input.template operator()<uint8_t>(p->allowed, p->allowed_count, bool(a.categories));
        a.count = p->count;
        a.output_count = p->output_count;
        a.allowed_count = p->allowed_count;
        a.label = p->label;
        a.mode = p->mode;
        DeviceScope device(context->device);
        const dim3 grid((uint64_t(p->count) + 255) / 256);
        if (a.indices) {
            if (a.existing)
                check(hipMemcpyAsync(a.output, a.existing, p->output_count, hipMemcpyDeviceToDevice, context->stream));
            else
                check(hipMemsetAsync(a.output, 0, p->output_count, context->stream));
            if (p->mode == 2) {
                hipLaunchKernelGGL((update_labels_native<1>), grid, dim3(256), 0, context->stream, a);
                check(hipGetLastError());
            }
            hipLaunchKernelGGL((update_labels_native<2>), grid, dim3(256), 0, context->stream, a);
        } else
            hipLaunchKernelGGL((update_labels_native<0>), grid, dim3(256), 0, context->stream, a);
        check(hipGetLastError());
    });
}
extern "C" int lfs_hip_and_live(LfsHipContext* context, LfsHipView mask, LfsHipView live, uint64_t count, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && count <= SIZE_MAX, "Invalid HIP live mask count");
        if (!count)
            return;
        validate(context, mask.buffer, mask.offset, count);
        validate(context, live.buffer, live.offset, count);
        require(mask.buffer != live.buffer || mask.offset == live.offset || mask.offset >= live.offset + count || live.offset >= mask.offset + count,
                "HIP live masks partially overlap");
        DeviceScope device(context->device);
        hipLaunchKernelGGL(and_live_native, dim3(blocks(count)), dim3(256), 0, context->stream,
                           static_cast<uint8_t*>(mask.buffer->data) + mask.offset, static_cast<const uint8_t*>(live.buffer->data) + live.offset, size_t(count));
        check(hipGetLastError());
    });
}
extern "C" int lfs_hip_histogram_u8(LfsHipContext* context, LfsHipView values, LfsHipView counts, uint64_t count, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && count <= INT32_MAX, "Invalid HIP label histogram count");
        validate(context, counts.buffer, counts.offset, 256 * sizeof(int32_t));
        require(counts.offset % alignof(int32_t) == 0, "Misaligned HIP label histogram output");
        if (!count)
            return;
        validate(context, values.buffer, values.offset, count);
        require(values.buffer != counts.buffer || values.offset >= counts.offset + 256 * sizeof(int32_t) || counts.offset >= values.offset + count,
                "HIP label histogram aliases its input");
        DeviceScope device(context->device);
        hipLaunchKernelGGL(histogram_labels_native, dim3(blocks(count)), dim3(256), 0, context->stream,
                           static_cast<const uint8_t*>(values.buffer->data) + values.offset,
                           reinterpret_cast<int32_t*>(static_cast<char*>(counts.buffer->data) + counts.offset), uint32_t(count));
        check(hipGetLastError());
    });
}
namespace {
    template <class T>
    T* point_data(LfsHipContext* context, LfsHipView view, size_t count) {
        require(count <= SIZE_MAX / sizeof(T), "HIP point operation range overflows");
        validate(context, view.buffer, view.offset, count * sizeof(T));
        require(view.offset % alignof(T) == 0, "Misaligned HIP point operation input");
        return reinterpret_cast<T*>(static_cast<char*>(view.buffer->data) + view.offset);
    }
    void separate_point_ranges(LfsHipView input, size_t bytes, LfsHipView output, size_t output_bytes) {
        require(input.buffer != output.buffer || input.offset >= output.offset + output_bytes || output.offset >= input.offset + bytes,
                "HIP point operation output aliases an input");
    }
    __global__ void project_points_native(const float* points, float* output, size_t count,
                                          lfs::core::internal::PointProjection projection, const float* transforms, size_t transform_count,
                                          const int32_t* indices, const uint8_t* visibility, size_t visibility_count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            lfs::core::internal::projectPoint(points, output, i, projection, transforms, transform_count, indices, visibility, visibility_count);
    }
    template <lfs::core::internal::PointRegion2DKind Kind>
    __global__ void mark_points_native(uint8_t* mask, const float* points, size_t count,
                                       lfs::core::internal::PointRegion2D region, const float* geometry, size_t geometry_count) {
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x)
            if (lfs::core::internal::pointInRegion<Kind>(points[2 * i], points[2 * i + 1], region, geometry, geometry_count))
                mask[i] = 1;
    }
} // namespace
extern "C" int lfs_hip_project_points(LfsHipContext* context, LfsHipView points, LfsHipView output,
                                      const LfsHipPointProject* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && !p->reserved && p->count <= INT32_MAX && p->transform_count <= INT32_MAX &&
                    p->visibility_count <= INT32_MAX && p->width && p->height && p->width <= INT32_MAX && p->height <= INT32_MAX,
                "Invalid HIP point projection descriptor");
        require(p->model == 0 || p->model == 1 || p->model == 3, "Invalid HIP point projection model");
        require(!p->visibility.buffer || p->indices.buffer, "HIP point visibility requires indices");
        if (!p->count)
            return;
        auto* xy = point_data<float>(context, output, size_t(p->count) * 2);
        const auto* xyz = point_data<float>(context, points, size_t(p->count) * 3);
        separate_point_ranges(points, size_t(p->count) * 3 * sizeof(float), output, size_t(p->count) * 2 * sizeof(float));
        const float* transforms = nullptr;
        const int32_t* indices = nullptr;
        const uint8_t* visibility = nullptr;
        require(!p->transform_count || p->transforms.buffer, "Missing HIP point projection transforms");
        require(!p->visibility_count || p->visibility.buffer, "Missing HIP point projection visibility table");
        if (p->transforms.buffer) {
            transforms = point_data<float>(context, p->transforms, size_t(p->transform_count) * 16);
            separate_point_ranges(p->transforms, size_t(p->transform_count) * 16 * sizeof(float), output, size_t(p->count) * 2 * sizeof(float));
        }
        if (p->indices.buffer) {
            indices = point_data<int32_t>(context, p->indices, p->count);
            separate_point_ranges(p->indices, size_t(p->count) * sizeof(int32_t), output, size_t(p->count) * 2 * sizeof(float));
        }
        if (p->visibility.buffer) {
            visibility = point_data<uint8_t>(context, p->visibility, p->visibility_count);
            separate_point_ranges(p->visibility, p->visibility_count, output, size_t(p->count) * 2 * sizeof(float));
        }
        lfs::core::internal::PointProjection projection{};
        std::copy_n(p->frame, 9, projection.rotation);
        std::copy_n(p->frame + 9, 3, projection.translation);
        projection.focal_x = p->frame[12];
        projection.focal_y = p->frame[13];
        projection.center_x = p->frame[14];
        projection.center_y = p->frame[15];
        projection.ortho_scale = p->frame[16];
        projection.invalid_value = p->frame[17];
        projection.width = int32_t(p->width);
        projection.height = int32_t(p->height);
        projection.model = static_cast<lfs::core::internal::PointProjectionModel>(p->model);
        DeviceScope device(context->device);
        hipLaunchKernelGGL(project_points_native, dim3(blocks(p->count)), dim3(256), 0, context->stream, xyz, xy, size_t(p->count),
                           projection, transforms, size_t(p->transform_count), indices, visibility, size_t(p->visibility_count));
        check(hipGetLastError());
    });
}
extern "C" int lfs_hip_mark_points_2d(LfsHipContext* context, LfsHipView mask, LfsHipView points, LfsHipView geometry,
                                      const LfsHipPointRegion* p, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && p && p->struct_bytes == sizeof(*p) && p->count <= INT32_MAX && p->geometry_count <= INT32_MAX && p->kind <= 3,
                "Invalid HIP point region descriptor");
        if (!p->count)
            return;
        auto* selected = point_data<uint8_t>(context, mask, p->count);
        const auto* xy = point_data<float>(context, points, size_t(p->count) * 2);
        separate_point_ranges(points, size_t(p->count) * 2 * sizeof(float), mask, p->count);
        const bool uses_geometry = p->kind == 2 || p->kind == 3;
        const float* vertices = nullptr;
        if (uses_geometry) {
            if (!p->geometry_count)
                return;
            vertices = point_data<float>(context, geometry, size_t(p->geometry_count) * 2);
            separate_point_ranges(geometry, size_t(p->geometry_count) * 2 * sizeof(float), mask, p->count);
        }
        using Kind = lfs::core::internal::PointRegion2DKind;
        const lfs::core::internal::PointRegion2D region{static_cast<Kind>(p->kind), p->x0, p->y0, p->x1, p->y1, p->radius, p->minimum_coordinate};
        DeviceScope device(context->device);
        const auto launch = [&]<Kind K>() { hipLaunchKernelGGL((mark_points_native<K>), dim3(blocks(p->count)), dim3(256), 0, context->stream,
                                                               selected, xy, size_t(p->count), region, vertices, size_t(p->geometry_count)); };
        switch (static_cast<Kind>(p->kind)) {
        case Kind::Disk: launch.template operator()<Kind::Disk>(); break;
        case Kind::Rectangle: launch.template operator()<Kind::Rectangle>(); break;
        case Kind::Polygon: launch.template operator()<Kind::Polygon>(); break;
        case Kind::Disks: launch.template operator()<Kind::Disks>(); break;
        }
        check(hipGetLastError());
    });
}
