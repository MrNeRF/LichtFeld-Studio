/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "hip_point_filter.hpp"
#include "point_tree_api.h"
int lfs_hip_affine(LfsHipContext* context, const LfsHipView* inputs, const LfsHipView* outputs,
                   const float* matrix, size_t count, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && inputs && outputs && matrix && count <= INT32_MAX, "Invalid HIP affine descriptor");
        if (!count)
            return;
        const size_t sizes[]{count * 12, count * 16, count * 36};
        for (unsigned i = 0; i < 3; ++i) {
            if (i == 2 && !inputs[i].buffer)
                continue;
            validate(context, inputs[i].buffer, inputs[i].offset, sizes[i]);
            require(inputs[i].offset % 4 == 0, "Misaligned HIP affine input");
        }
        for (unsigned o = 0; o < 2; ++o) {
            validate(context, outputs[o].buffer, outputs[o].offset, sizes[o]);
            require(outputs[o].offset % 4 == 0, "Misaligned HIP affine output");
            for (unsigned i = 0; i < 3; ++i) {
                if (!inputs[i].buffer)
                    continue;
                const bool disjoint = inputs[i].buffer != outputs[o].buffer || inputs[i].offset >= outputs[o].offset + sizes[o] ||
                                      outputs[o].offset >= inputs[i].offset + sizes[i];
                require(disjoint || (i == o && inputs[i].offset == outputs[o].offset), "Overlapping HIP affine input/output");
            }
        }
        require(outputs[0].buffer != outputs[1].buffer || outputs[0].offset >= outputs[1].offset + sizes[1] ||
                    outputs[1].offset >= outputs[0].offset + sizes[0],
                "Overlapping HIP affine outputs");
        const auto pointer = [](LfsHipView v) -> float* { return v.buffer ? reinterpret_cast<float*>(static_cast<char*>(v.buffer->data) + v.offset) : nullptr; };
        hip_affine::LinearTransform native{};
        std::copy_n(matrix, 9, native.rows);
        DeviceScope device(context->device);
        hip_affine::transform_geometry<<<(count + 255) / 256, 256, 0, context->stream>>>(native, pointer(inputs[0]), pointer(inputs[1]),
                                                                                         pointer(outputs[0]), pointer(outputs[1]), count, pointer(inputs[2]));
        check(hipGetLastError());
    });
}
