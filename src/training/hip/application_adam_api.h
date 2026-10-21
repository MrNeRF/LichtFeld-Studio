/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "application_fast_api.h"
typedef struct LfsHipAdamEntry {
    uint64_t parameter,packed,bounds,gradient;
    int32_t primitives,attributes,mean_step,screen_share;
    float lr,bc1_rcp,bc2_sqrt_rcp;
} LfsHipAdamEntry;
typedef struct LfsHipAdamCall {
    uint32_t struct_bytes,operation;
    int32_t device,entries;
    uint64_t queue,masks[5];
    int32_t lengths[5];
    float beta1,beta2,eps,frozen_lr,crop_lr,median_extent,screen_limit,screen_penalty;
    LfsHipAdamEntry batch[6];
    LfsHipFastAdamGroup sh;
    uint64_t gradient,indices,flags,touched;
    int32_t count,primitives,attributes,bits,layout,active_bases,layout_slots;
} LfsHipAdamCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_adam(const LfsHipAdamCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
