/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
enum { LFS_HIP_REFINE_SPLIT, LFS_HIP_REFINE_FILL, LFS_HIP_REFINE_COUNTS,
       LFS_HIP_REFINE_NORMALIZE, LFS_HIP_REFINE_CLIP, LFS_HIP_REFINE_SCORES,
       LFS_HIP_REFINE_DEAD, LFS_HIP_REFINE_ROTATION };
typedef struct LfsHipRefineCall {
    uint32_t struct_bytes, operation;
    int32_t device;
    uint32_t reserved;
    uint64_t queue, bindings[12], count, rows, lengths[4];
    float value;
} LfsHipRefineCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_refine(const LfsHipRefineCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
