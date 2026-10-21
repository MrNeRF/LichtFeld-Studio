/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#ifdef __cplusplus
extern "C" {
#endif

enum {
    LFS_HIP_POPSPA_KEEP = 0,
    LFS_HIP_POPSPA_PROJECT = 1,
    LFS_HIP_POPSPA_PENALTY = 2,
    LFS_HIP_POPSPA_EFFECTIVE_RANK = 3
};

/* Borrowed native allocations, retained by the Tensor adapter. Scores are
 * Float64 rows stored in UInt8 [N,8]. Scratch comes from the Core pool; the
 * native operation never creates a CUDA context or a second allocator. */
typedef struct LfsHipPopspaCall {
    uint32_t struct_bytes, operation;
    int32_t device;
    uint32_t update_dual;
    uint64_t queue, count, keep_count, active_count;
    uint64_t scores, raw, z, u, active, frozen, gradient, loss, keep;
    uint64_t workspace, workspace_bytes;
    float rho, rank_weight, thin_weight, epsilon;
} LfsHipPopspaCall;

LFS_HIP_TRAIN_API int lfs_hip_application_popspa_workspace(
    const LfsHipPopspaCall*, uint64_t* bytes, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_popspa(
    const LfsHipPopspaCall*, LfsHipError*);

#ifdef __cplusplus
}
#endif
