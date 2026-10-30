/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipPointTree {
    uint32_t points, references, levels;
    uint32_t level_offset[8], level_count[8];
    float radius;
    int32_t max_count;
} LfsHipPointTree;
typedef struct LfsHipPpisp {
    float exposure_factor, vignetting[15], color_matrix[9], crf[15];
    int32_t y_offset, full_height, x_offset, full_width;
} LfsHipPpisp;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_simplify_merge(LfsHipContext*, const LfsHipView* inputs, const LfsHipView* outputs, uint32_t groups, uint32_t appearance, LfsHipError*);
LFS_HIP_API int lfs_hip_export_dispatch(LfsHipContext*, uint32_t module, uint32_t phase, const void* parameters, size_t bytes, size_t work, LfsHipError*);
LFS_HIP_API int lfs_hip_multinomial(LfsHipContext*, LfsHipView weights, LfsHipView output, size_t categories,
                                    size_t samples, uint64_t seed, uint32_t replacement, LfsHipError*);
LFS_HIP_API int lfs_hip_morton(LfsHipContext*, LfsHipView positions, LfsHipView keys, LfsHipView indices,
                               size_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_affine(LfsHipContext*, const LfsHipView* inputs, const LfsHipView* outputs,
                               const float* matrix, size_t count, LfsHipError*);
LFS_HIP_API int lfs_hip_ppisp(LfsHipContext*, LfsHipView input, LfsHipView output,
                              int32_t width, int32_t height, const LfsHipPpisp*, LfsHipError*);
/* inputs: positions, sorted positions, boxes, visit, radii, optional query mask. */
LFS_HIP_API int lfs_hip_point_tree_components(LfsHipContext*, const LfsHipView* inputs, LfsHipView labels,
                                              const LfsHipPointTree*, LfsHipError*);
LFS_HIP_API int lfs_hip_point_tree_query(LfsHipContext*, const LfsHipView* inputs, LfsHipView output,
                                         const LfsHipPointTree*, uint32_t spacing, LfsHipError*);
#ifdef __cplusplus
}
#endif
