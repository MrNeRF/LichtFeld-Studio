/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipMrnfCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint32_t count,mask_count;
    uint64_t queue,bindings[6],seed;
    float value[5];
} LfsHipMrnfCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_mrnf(const LfsHipMrnfCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
