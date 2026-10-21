/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipPhotoCall {
    uint32_t struct_bytes, path, backward, byte_target, byte_mask, valid_padding;
    int32_t device,n,c,h,w;
    uint64_t queue,bindings[13];
    float weight,normalization,denominator;
} LfsHipPhotoCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_photo(const LfsHipPhotoCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
