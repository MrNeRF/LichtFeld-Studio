/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
typedef struct LfsHipInferencePush {
    uint64_t input, output;
    uint32_t total, step;
    int32_t channels, height, width, out_height, out_width, kernel_h, kernel_w;
    int32_t stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, offset, columns, mode, coord, include_pad;
    float u0, u1, v0, v1;
} LfsHipInferencePush;
typedef struct LfsHipInference {
    uint32_t struct_bytes, operation;
    LfsHipInferencePush parameters;
} LfsHipInference;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_API int lfs_hip_inference_typed(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipInference*, uint32_t dtype, LfsHipError*);
LFS_HIP_API int lfs_hip_inference_f32(LfsHipContext*, LfsHipView input, LfsHipView output, const LfsHipInference*, LfsHipError*);
#ifdef __cplusplus
}
#endif
