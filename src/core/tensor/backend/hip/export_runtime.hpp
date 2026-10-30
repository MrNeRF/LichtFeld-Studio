/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "export_kernels.hpp"
#include "point_tree_api.h"
int lfs_hip_export_dispatch(LfsHipContext* context, uint32_t module, uint32_t phase, const void* parameters, size_t bytes, size_t work, LfsHipError* error) {
    return boundary(error, [&] {
        require(context && parameters && work && work <= INT32_MAX && module < 9 && phase <= 20, "Invalid HIP export dispatch");
        DeviceScope device(context->device);
        const auto groups = module == 8 ? (work + 255) / 256 : std::min<size_t>((work + 255) / 256, 65535);
#define HIP_EXPORT_CASE(ID, NAME)                                                 \
    case ID: {                                                                    \
        hip_##NAME::Parameters_0 p{};                                             \
        require(bytes == sizeof(p), "Invalid HIP export parameters size");        \
        std::memcpy(&p, parameters, sizeof(p));                                   \
        hip_##NAME::NAME##_kernel<<<groups, 256, 0, context->stream>>>(p, phase); \
        check(hipGetLastError());                                                 \
        break;                                                                    \
    }
        switch (module) {
            HIP_EXPORT_CASE(0, export_kmeans);
            HIP_EXPORT_CASE(1, export_kmeans_accumulate);
            HIP_EXPORT_CASE(2, export_kmeans_group);
            HIP_EXPORT_CASE(3, export_radix);
            HIP_EXPORT_CASE(4, export_morton);
            HIP_EXPORT_CASE(5, export_decimate);
            HIP_EXPORT_CASE(6, export_decimate_cost);
            HIP_EXPORT_CASE(7, export_decimate_merge);
            HIP_EXPORT_CASE(8, point_raster);
        }
#undef HIP_EXPORT_CASE
        check(hipGetLastError());
    });
}
