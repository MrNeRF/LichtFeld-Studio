/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* operation is a LfsHipReduceOp for all input/output dtypes. */
LFS_HIP_API int lfs_hip_reduce_typed_layout(LfsHipContext*, LfsHipView, LfsHipView, const LfsHipLayout*, uint32_t axes, uint32_t operation, uint32_t input_dtype, uint32_t output_dtype, LfsHipError*);
#ifdef __cplusplus
}
#endif
