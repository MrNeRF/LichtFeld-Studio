/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipNorm {uint64_t rows,cols;float eps;uint32_t dtype;} LfsHipNorm;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_nn_norm(LfsHipContext*,LfsHipView input,LfsHipView weight,LfsHipView bias,LfsHipView output,const LfsHipNorm*,LfsHipError*);
#ifdef __cplusplus
}
#endif
