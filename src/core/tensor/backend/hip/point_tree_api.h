/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipPointTree {
    uint32_t points,references,levels;
    uint32_t level_offset[8],level_count[8];
    float radius;
    int32_t max_count;
} LfsHipPointTree;
#ifdef __cplusplus
extern "C" {
#endif
/* inputs: positions, sorted positions, boxes, visit, radii, optional query mask. */
LFS_HIP_API int lfs_hip_point_tree_components(LfsHipContext*,const LfsHipView* inputs,LfsHipView labels,
                                            const LfsHipPointTree*,LfsHipError*);
LFS_HIP_API int lfs_hip_point_tree_query(LfsHipContext*,const LfsHipView* inputs,LfsHipView output,
                                       const LfsHipPointTree*,uint32_t spacing,LfsHipError*);
#ifdef __cplusplus
}
#endif
