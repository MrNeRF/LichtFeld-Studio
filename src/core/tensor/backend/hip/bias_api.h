/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_bias_f32(LfsHipContext*, LfsHipView, LfsHipView, LfsHipView, uint32_t count, uint32_t channels, uint32_t spatial, uint32_t relu, LfsHipError*);
#ifdef __cplusplus
}
#endif
