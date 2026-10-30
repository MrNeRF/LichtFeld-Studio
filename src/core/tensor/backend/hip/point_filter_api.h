/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LfsHipPointFilter {
    uint32_t struct_bytes, count, transform_count, allowed_count, flags, model;
    /* rotation[9], translation[3], focal[2], center[2], ortho scale,
     * width/height, near/far depth, scale[2], offset[2]. */
    float window[25];
    LfsHipView inputs[9];
} LfsHipPointFilter;
LFS_HIP_API int lfs_hip_filter_points(LfsHipContext*, LfsHipView mask,
                                      const LfsHipPointFilter*, LfsHipError*);
typedef struct LfsHipLabelUpdate {
    uint32_t struct_bytes, count, output_count, allowed_count, label, mode;
    LfsHipView existing, locked, indices, categories, allowed;
} LfsHipLabelUpdate;
LFS_HIP_API int lfs_hip_update_labels(LfsHipContext*, LfsHipView output,
                                      LfsHipView selected, const LfsHipLabelUpdate*, LfsHipError*);
LFS_HIP_API int lfs_hip_and_live(LfsHipContext*, LfsHipView mask, LfsHipView live,
                                 uint64_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_histogram_u8(LfsHipContext*, LfsHipView values, LfsHipView counts,
                                     uint64_t count, LfsHipError*);
typedef struct LfsHipPointProject {
    uint32_t struct_bytes, count, model, width, height, transform_count, visibility_count, reserved;
    /* rotation[9], translation[3], focal[2], center[2], ortho scale, invalid value. */
    float frame[18];
    LfsHipView transforms, indices, visibility;
} LfsHipPointProject;
typedef struct LfsHipPointRegion {
    uint32_t struct_bytes, count, kind, geometry_count;
    float x0, y0, x1, y1, radius, minimum_coordinate;
} LfsHipPointRegion;
LFS_HIP_API int lfs_hip_project_points(LfsHipContext*, LfsHipView points, LfsHipView output,
                                       const LfsHipPointProject*, LfsHipError*);
LFS_HIP_API int lfs_hip_mark_points_2d(LfsHipContext*, LfsHipView mask, LfsHipView points,
                                       LfsHipView geometry, const LfsHipPointRegion*, LfsHipError*);
#ifdef __cplusplus
}
#endif
