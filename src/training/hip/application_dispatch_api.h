/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Hooks bracket host submission, including failure cleanup. They do not wait
 * on a GPU timeline and must not throw across this private C boundary. */
typedef struct LfsHipApplicationDispatch {
    uint32_t struct_bytes;
    void* user;
    int (*begin)(void*,LfsHipError*);
    void (*end)(void*);
} LfsHipApplicationDispatch;
LFS_HIP_TRAIN_API int lfs_hip_application_dispatch_install(const LfsHipApplicationDispatch*,LfsHipError*);
#ifdef __cplusplus
}
#endif
