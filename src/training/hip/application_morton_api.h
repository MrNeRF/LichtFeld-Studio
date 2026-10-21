/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
/* operation: 0 codes, 1 row moments, 2 SH moments, 3 grouped SH moments,
 * 4 grouped Q16 SH values. Workspaces always belong to Core Tensor. */
typedef struct LfsHipMortonCall {
    uint32_t struct_bytes,operation,bits,width;
    int32_t device;
    uint64_t queue,bindings[6],count,scratch_bytes;
} LfsHipMortonCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_morton(const LfsHipMortonCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
