/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
enum { LFS_HIP_IMAGE_HEATMAP,LFS_HIP_IMAGE_ROI,LFS_HIP_IMAGE_RESIZE,LFS_HIP_IMAGE_RANDOM,
       LFS_HIP_IMAGE_CANNY,LFS_HIP_IMAGE_NORMALIZE,LFS_HIP_IMAGE_QUANTIZE };
typedef struct LfsHipImageCall {
    uint32_t struct_bytes,operation,byte_input,inverse;
    int32_t device,slot,h,w,c,source_h,source_w;
    uint64_t queue,bindings[6],count,seed;
    float value,fx,fy,cx,cy,affine[12],minimum[3],maximum[3];
} LfsHipImageCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_image(const LfsHipImageCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
