/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipStructureParameters {
    uint64_t a,b,c,d,e,f,g;
    int32_t width,height,first_row,rows;
    uint32_t stride,blocks,image_byte,mask_byte,radius,first_scale,valid_padding,reserved;
    float gain,sigma_sq;
    float filter_g[21],filter_d1[21],filter_d2[21];
} LfsHipStructureParameters;
typedef struct LfsHipStructureCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint32_t dispatch_count;
    uint64_t queue;
    LfsHipStructureParameters parameters;
} LfsHipStructureCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_structure(const LfsHipStructureCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
