/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_color_api.h"
typedef struct LfsHipMeshCamera {
    float rows[12];float fx,fy,cx,cy;int32_t width,height;
    float guard_min_x,guard_max_x,guard_min_y,guard_max_y;
    int32_t distorted,model,num_distortion;float distortion[12];float src_fx,src_fy,src_cx,src_cy,dst_fx,dst_fy;
} LfsHipMeshCamera;
typedef struct LfsHipMeshCall {
    uint32_t struct_bytes,operation;int32_t device,vertex_count,face_count,radius;float z_near;
    uint64_t queue,vertices,indices,samples,large_faces,counters,mask;
    LfsHipMeshCamera camera;
} LfsHipMeshCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_masks(const LfsHipColorCall*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_mesh(const LfsHipMeshCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
