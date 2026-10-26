/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipGemm {
    uint32_t struct_bytes, transpose_b, relu;
    uint64_t batch, m, n, k;
} LfsHipGemm;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_gemm_f32(LfsHipContext*, LfsHipView a, LfsHipView b, LfsHipView bias, LfsHipView output, const LfsHipGemm*, LfsHipError*);
#ifdef __cplusplus
}
#endif
