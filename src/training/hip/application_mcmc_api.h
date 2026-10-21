/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#ifdef __cplusplus
extern "C" {
#endif
enum LfsHipMcmcOperation {
    LFS_HIP_MCMC_RELOCATE=0, LFS_HIP_MCMC_NOISE=1, LFS_HIP_MCMC_COPY=2,
    LFS_HIP_MCMC_UPDATE=3, LFS_HIP_MCMC_SAMPLE_ALIVE=4, LFS_HIP_MCMC_SAMPLE_ALL=5,
    LFS_HIP_MCMC_FOLD_ERROR=6
};
/* Tensor ownership, sizes/dtypes, index guards and scan are checked by the
 * application adapter. Native kernels borrow the Tensor runtime's HIP queue. */
typedef struct LfsHipMcmcCall {
    uint32_t struct_bytes, operation;
    int32_t device;
    uint64_t queue;
    uint64_t bindings[8];
    uint64_t count, model_count, domain_count, mask_count, seed;
    float value;
} LfsHipMcmcCall;
LFS_HIP_TRAIN_API int lfs_hip_application_mcmc_initialize(int32_t device, uint64_t queue, int32_t n_max, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_mcmc(const LfsHipMcmcCall*, LfsHipError*);
#ifdef __cplusplus
}
#endif
