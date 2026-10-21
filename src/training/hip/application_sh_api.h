/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
enum { LFS_HIP_SH_ENCODE,LFS_HIP_SH_DECODE,LFS_HIP_SH_BLOCK_IDS,LFS_HIP_SH_BLOCK_RUNS,
       LFS_HIP_SH_TOUCHED,LFS_HIP_SH_RANGE,LFS_HIP_SH_ZERO,LFS_HIP_SH_GATHER,
       LFS_HIP_SH_GATHER_CANONICAL,LFS_HIP_SH_APPEND,LFS_HIP_SH_SCATTER,LFS_HIP_SH_FILL };
typedef struct LfsHipShCall {
    uint32_t struct_bytes,operation,query,index64,byte_elements,storage;
    int32_t device;
    uint32_t source_rest,destination_rest;
    uint64_t queue,bindings[10],count,primitives,source_rows,destination_offset;
    uint64_t float_offset,float_count,code_offset,bounds_offset,workspace_bytes;
} LfsHipShCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_sh(LfsHipShCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
