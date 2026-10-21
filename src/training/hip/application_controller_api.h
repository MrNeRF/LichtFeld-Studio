/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
typedef struct LfsHipControllerCall {
    uint32_t struct_bytes,operation;
    int32_t device;
    uint32_t m,n,write_prior;
    uint64_t queue,features,input,grad,activation,weight,weight_gradient,bias_gradient,grad_input;
    float prior;
} LfsHipControllerCall;
#ifdef __cplusplus
extern "C" {
#endif
LFS_HIP_TRAIN_API int lfs_hip_application_controller(const LfsHipControllerCall*,LfsHipError*);
#ifdef __cplusplus
}
#endif
