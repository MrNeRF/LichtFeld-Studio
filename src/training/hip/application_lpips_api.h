/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipLpipsPush {
    uint64_t a,b,c,d,e,f,g;
    uint32_t count,channels,height,width,out_channels,out_height,out_width;
    uint32_t stride,padding,dilation,flags;
    float shift0,shift1,shift2,scale0,scale1,scale2,inverse_count;
} LfsHipLpipsPush;
typedef struct LfsHipLpipsCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint64_t queue,dispatch_count;
    LfsHipLpipsPush parameters;
} LfsHipLpipsCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_lpips(const LfsHipLpipsCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
