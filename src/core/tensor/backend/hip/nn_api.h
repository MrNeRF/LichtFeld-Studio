/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "inference_api.h"
#include "runtime_api.h"
typedef struct LfsHipNorm {
    uint64_t rows, cols;
    float eps;
    uint32_t dtype;
} LfsHipNorm;
typedef struct LfsHipLinear {
    uint64_t batch, m, n, k;
    uint32_t dtype, transpose_b, batched_b, activation;
} LfsHipLinear;
typedef struct LfsHipAttention {
    uint64_t groups, heads, queries, keys, dim;
    int64_t strides[4];
    float scale;
    uint32_t dtype;
} LfsHipAttention;
typedef struct LfsHipConv {
    uint64_t batch, out_channels, groups;
    uint32_t dtype, transpose, activation, reserved;
    LfsHipInferencePush geometry;
} LfsHipConv;
#ifdef __cplusplus
extern "C" {
#endif
/* inputs: activations, weights, optional bias, scale and residual. */
LFS_HIP_API int lfs_hip_nn_conv(LfsHipContext*, const LfsHipView* inputs, LfsHipView output, const LfsHipConv*, LfsHipError*);
LFS_HIP_API int lfs_hip_nn_attention(LfsHipContext*, const LfsHipView* inputs, LfsHipView output, const LfsHipAttention*, LfsHipError*);
LFS_HIP_API int lfs_hip_nn_linear(LfsHipContext*, const LfsHipView* inputs, LfsHipView output, const LfsHipLinear*, LfsHipError*);
LFS_HIP_API int lfs_hip_nn_norm(LfsHipContext*, LfsHipView input, LfsHipView weight, LfsHipView bias, LfsHipView output, const LfsHipNorm*, LfsHipError*);
#ifdef __cplusplus
}
#endif
