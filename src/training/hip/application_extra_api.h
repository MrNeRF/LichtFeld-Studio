/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipExtraCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint64_t queue,bindings[5],count;
    float weight;
} LfsHipExtraCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_extra(const LfsHipExtraCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
