/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipColorCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint64_t queue,bindings[24];
    int32_t dimensions[12];
    float values[8];
    uint32_t flags[4];
} LfsHipColorCall;
typedef struct LfsHipAnchorCall {
    uint32_t struct_bytes,capacity;
    int32_t device,width,height,model;
    uint64_t queue,points,point_count,view,prior;
    float intrinsics[4],near_plane,aabb_lo[3],aabb_hi[3];
    float radial[6],tangential[2],thin_prism[4];
} LfsHipAnchorCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_bilateral(const LfsHipColorCall*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_ppisp(const LfsHipColorCall*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_geometry(const LfsHipColorCall*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_anchor(const LfsHipAnchorCall*,float*,uint32_t*,LfsHipError*);
#ifdef __cplusplus
}
#endif
