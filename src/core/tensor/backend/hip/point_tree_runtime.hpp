/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "point_tree_api.h"
#include "spatial_runtime.hpp"
namespace {
    size_t point_tree_boxes_native(const LfsHipPointTree& p, bool triangles = false) {
        require(p.points <= INT32_MAX && p.references <= INT32_MAX && (triangles || p.references <= p.points) && p.levels <= 8, "Invalid HIP point-tree parameters");
        size_t boxes_count = 0;
        uint32_t children = p.references, levels = 0;
        while (children) {
            require(levels < p.levels, "Missing HIP point-tree level");
            const uint32_t count = (children - 1) / 32 + 1;
            require(p.level_offset[levels] == boxes_count && p.level_count[levels] == count, "Invalid HIP point-tree level layout");
            boxes_count += count;
            ++levels;
            if (count <= 32)
                break;
            children = count;
        }
        require(levels == p.levels, "Invalid HIP point-tree depth");
        return boxes_count;
    }
    template <unsigned Mode>
    __global__ void point_tree_query_native(const float* points, const float* sorted, const float* boxes, const int32_t* visit,
                                            const float* radii, const uint8_t* queries, void* output, hs::PointTreeProgram program) {
        for (size_t rank = size_t(blockIdx.x) * blockDim.x + threadIdx.x; rank < program.points; rank += size_t(gridDim.x) * blockDim.x) {
            const int32_t index = visit[rank];
            if (index < 0 || uint32_t(index) >= program.points)
                continue;
            const int64_t reference_rank = rank < program.references ? int64_t(rank) : -1;
            if constexpr (Mode == 2)
                static_cast<int32_t*>(output)[index] = hs::triangleTreeParity(sorted, boxes, program, points + size_t(index) * 3);
            else if constexpr (Mode == 1)
                static_cast<float*>(output)[index] = hs::pointTreeSpacing(sorted, boxes, program, points + size_t(index) * 3, reference_rank, program.radius);
            else
                static_cast<int32_t*>(output)[index] = (!queries || queries[index]) ? hs::pointTreeCount(sorted, boxes, program, points + size_t(index) * 3, reference_rank, radii[index], program.max_count) : 0;
        }
    }
} // namespace
int lfs_hip_point_tree_query(LfsHipContext* context, const LfsHipView* inputs, LfsHipView output,
                             const LfsHipPointTree* descriptor, uint32_t spacing, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && descriptor && spacing <= 2, "Invalid HIP point-tree descriptor");
        const auto& p = *descriptor;
        require(spacing == 2 || (spacing ? std::isnormal(p.radius) && p.radius > 0 && (!p.points || p.references > 0) : p.max_count > 0),
                "Invalid HIP point-tree query parameters");
        const size_t boxes_count = point_tree_boxes_native(p, spacing == 2);
        if (!p.points)
            return;
        const auto checked_bytes = [](size_t n, size_t width) {
            require(n <= SIZE_MAX / width, "HIP point-tree byte range overflow");
            return n * width;
        };
        const size_t sizes[]{checked_bytes(p.points, 12), checked_bytes(p.references, spacing == 2 ? 36 : 12), checked_bytes(boxes_count, 24),
                             checked_bytes(p.points, 4), spacing ? 0 : checked_bytes(p.points, 4), spacing ? 0 : size_t(p.points)};
        const auto bytes = checked_bytes(p.points, 4);
        validate(context, output.buffer, output.offset, bytes);
        require(output.offset % 4 == 0, "Misaligned HIP point-tree output");
        for (unsigned i = 0; i < 6; ++i) {
            if (!sizes[i] || (i == 5 && !inputs[i].buffer))
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(i == 5 || inputs[i].offset % 4 == 0, "Misaligned HIP point-tree input");
            require(inputs[i].buffer != output.buffer || inputs[i].offset >= output.offset + bytes ||
                        output.offset >= inputs[i].offset + sizes[i],
                    "HIP point-tree output aliases input");
        }
        const auto pointer = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        hs::PointTreeProgram program{};
        program.points = p.points;
        program.references = p.references;
        program.levels = p.levels;
        std::copy_n(p.level_offset, 8, program.level_offset);
        std::copy_n(p.level_count, 8, program.level_count);
        program.radius = p.radius;
        program.max_count = p.max_count;
        DeviceScope device(context->device);
        const auto launch = [&]<unsigned Mode>() {
            hipLaunchKernelGGL((point_tree_query_native<Mode>), dim3(blocks(p.points)), dim3(256), 0, context->stream,
                               static_cast<const float*>(pointer(inputs[0])), static_cast<const float*>(pointer(inputs[1])),
                               static_cast<const float*>(pointer(inputs[2])), static_cast<const int32_t*>(pointer(inputs[3])),
                               static_cast<const float*>(pointer(inputs[4])), static_cast<const uint8_t*>(pointer(inputs[5])), pointer(output), program);
        };
        if (spacing == 2)
            launch.template operator()<2>();
        else if (spacing)
            launch.template operator()<1>();
        else
            launch.template operator()<0>();
        check(hipGetLastError());
    });
}
int lfs_hip_point_tree_components(LfsHipContext* context, const LfsHipView* inputs, LfsHipView labels,
                                  const LfsHipPointTree* descriptor, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && descriptor, "Invalid HIP point-tree components descriptor");
        const auto& p = *descriptor;
        const size_t boxes_count = point_tree_boxes_native(p);
        if (!p.points)
            return;
        const auto bytes = [](size_t n, size_t width) {require(n<=SIZE_MAX/width,"HIP point-tree components range overflow");return n*width; };
        const size_t sizes[]{bytes(p.points, 12), bytes(p.references, 12), bytes(boxes_count, 24), bytes(boxes_count, 4),
                             bytes(p.references, 4), bytes(p.references, 4), bytes(p.points, 4)};
        const size_t out_bytes = bytes(p.points, 4);
        validate(context, labels.buffer, labels.offset, out_bytes);
        require(labels.offset % 4 == 0, "Misaligned HIP point-tree components output");
        for (unsigned i = 0; i < 7; ++i)
            if (sizes[i]) {
                validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
                require(inputs[i].offset % 4 == 0, "Misaligned HIP point-tree components input");
                require(inputs[i].buffer != labels.buffer || inputs[i].offset >= labels.offset + out_bytes ||
                            labels.offset >= inputs[i].offset + sizes[i],
                        "HIP point-tree components output aliases input");
            }
        const auto pointer = [](LfsHipView v) -> void* { return v.buffer ? static_cast<char*>(v.buffer->data) + v.offset : nullptr; };
        hs::PointTreeProgram tree{};
        tree.points = p.points;
        tree.references = p.references;
        tree.levels = p.levels;
        std::copy_n(p.level_offset, 8, tree.level_offset);
        std::copy_n(p.level_count, 8, tree.level_count);
        tree.radius = p.radius;
        tree.max_count = p.max_count;
        DeviceScope device(context->device);
        auto* parent = static_cast<int32_t*>(pointer(labels));
        if (p.references) {
            hipLaunchKernelGGL(hip_spatial_components::point_tree_union, dim3(unsigned((p.references + 255) / 256)), dim3(256), 0, context->stream,
                               static_cast<const float*>(pointer(inputs[0])), static_cast<const float*>(pointer(inputs[1])), static_cast<const float*>(pointer(inputs[2])),
                               static_cast<const float*>(pointer(inputs[3])), static_cast<const int32_t*>(pointer(inputs[4])),
                               static_cast<const float*>(pointer(inputs[5])), static_cast<const float*>(pointer(inputs[6])), parent, tree);
            check(hipGetLastError());
        }
        hipLaunchKernelGGL(hip_spatial_components::flatten_components, dim3(unsigned((p.points + 255) / 256)), dim3(256), 0, context->stream, parent, size_t(p.points));
        check(hipGetLastError());
    });
}
