/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_inverse_warp(LfsHipContext*, LfsHipView input, LfsHipView output,
                                     LfsHipView validity, uint32_t channels, uint32_t mode,
                                     const LfsHipLens*, LfsHipError*);
#ifdef __cplusplus
}
#endif
