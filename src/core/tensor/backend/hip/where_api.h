/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LfsHipWhere {
    uint32_t struct_bytes, dtype;
    LfsHipLayout condition, x, y, output;
} LfsHipWhere;
LFS_HIP_API int lfs_hip_where_layout(LfsHipContext*, LfsHipView condition, LfsHipView x, LfsHipView y, LfsHipView output, const LfsHipWhere*, LfsHipError*);
LFS_HIP_API int lfs_hip_masked_fill(LfsHipContext*, LfsHipView mask, LfsHipView output, uint64_t count, uint32_t dtype, float value, LfsHipError*);
LFS_HIP_API int lfs_hip_where_scalar(LfsHipContext*, LfsHipView mask, LfsHipView source, LfsHipView output, size_t count, uint32_t dtype, float value, LfsHipError*);
#ifdef __cplusplus
}
#endif
