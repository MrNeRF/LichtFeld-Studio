/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipApplicationGsplat LfsHipApplicationGsplat;
typedef struct LfsHipGsplatForward {
    uint32_t struct_bytes;int32_t count,width,height,channels,camera_model,active_degree,layout_degree;
    uint64_t means,quats,scales,opacities,sh0,shN,view,K,radial,tangential,thin_prism;
    uint64_t background,image_background,image,alpha,rays;
} LfsHipGsplatForward;
typedef struct LfsHipGsplatBackward {
    uint32_t struct_bytes,expected_depth;float flatten_weight;
    uint64_t image,alpha,densification,error_map,edge_map,edge_scores,screen_share;
    uint64_t grad_means,grad_quats,grad_scales,grad_opacity,grad_sh0,grad_shN;
} LfsHipGsplatBackward;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_create(int32_t,uint64_t,uint64_t,LfsHipApplicationGsplat**,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_forward(LfsHipApplicationGsplat*,const LfsHipGsplatForward*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_backward(LfsHipApplicationGsplat*,const LfsHipGsplatBackward*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_release(LfsHipApplicationGsplat*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_info(LfsHipApplicationGsplat*,uint64_t*,uint64_t*,LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_gsplat_destroy(LfsHipApplicationGsplat*,LfsHipError*);
#ifdef __cplusplus
}
#endif
