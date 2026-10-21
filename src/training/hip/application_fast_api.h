/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Application-facing FastGS seam. Addresses are native HIP device addresses
 * validated and retained by the Tensor adapter. The borrowed native HIP queue
 * and its context/module owner outlive this handle. No CUDA stream is admitted. */
typedef struct LfsHipApplicationFast LfsHipApplicationFast;
enum { LFS_HIP_FAST_INSTANCE_OVERFLOW = 6 };
typedef struct LfsHipFastForward {
    uint32_t struct_bytes;
    uint64_t means, scales, rotations, opacities, sh0, shN, sh_bounds;
    uint64_t view, camera, background_color, background_image;
    uint64_t image, alpha, depth, normal, screen_share;
    int32_t count, active_bases, layout_bases, width, height;
    uint32_t sh_bits, sh_cells, mip_filter;
    float fx, fy, cx, cy, dilation;
} LfsHipFastForward;
typedef struct LfsHipFastAdamGroup {
    uint64_t parameter, packed, bounds, value_bounds, frozen, crop, screen_share, gradient_out;
    int32_t joint_bits, value_bits, value_cells, primitives, elements, attributes;
    int32_t frozen_count, crop_count, screen_count;
    uint32_t enabled;
    float step_size, bc2_sqrt_rcp, frozen_lr, crop_lr, screen_limit, screen_penalty;
} LfsHipFastAdamGroup;
typedef struct LfsHipFastBackward {
    uint32_t struct_bytes;
    uint64_t grad_image, grad_alpha, grad_depth, grad_normal;
    uint64_t densification, error_map, edge_map, edge_scores;
    uint64_t scale_loss, opacity_loss, erank_loss, dc_loss, sh_loss, rendered_count;
    uint64_t impact_sum, impact_peak, mean_far;
    int32_t mean_far_count, densification_type;
    uint32_t scale_reg_log, per_splat_mean_step;
    float beta1, beta2, eps, scale_reg_weight, scale_reg_normalizer;
    float opacity_reg_weight, flatten_reg_weight, erank_reg_weight, dc_reg_weight, sh_reg_weight;
    float median_extent;
    LfsHipFastAdamGroup groups[6]; /* means, scales, rotations, opacity, SH0, SHN */
} LfsHipFastBackward;
typedef struct LfsHipFastInfo {
    uint32_t struct_bytes;
    uint64_t live_bytes, peak_bytes, budget_bytes;
    int32_t instances, visible;
} LfsHipFastInfo;
LFS_HIP_TRAIN_API int lfs_hip_application_fast_create(int32_t device, uint64_t queue, uint64_t workspace_budget, LfsHipApplicationFast**, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_fast_forward(LfsHipApplicationFast*, const LfsHipFastForward*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_fast_backward(LfsHipApplicationFast*, const LfsHipFastBackward*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_fast_release(LfsHipApplicationFast*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_fast_info(LfsHipApplicationFast*, LfsHipFastInfo*, LfsHipError*);
LFS_HIP_TRAIN_API int lfs_hip_application_fast_destroy(LfsHipApplicationFast*, LfsHipError*);
#ifdef __cplusplus
}
#endif
