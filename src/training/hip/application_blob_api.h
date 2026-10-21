/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipBlobParameters {
    uint64_t a,b,c,d,e,f;
    int32_t width,height,source_width,source_height;
    uint32_t count,capacity,factor,byte_input,radius,horizontal,x_max,y_max;
    int32_t view,neighbors;
    float weights[7];
} LfsHipBlobParameters;
typedef struct LfsHipBlobCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint32_t dispatch_count;
    uint64_t queue;
    LfsHipBlobParameters parameters;
} LfsHipBlobCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_blob(const LfsHipBlobCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
