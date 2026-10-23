/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "training_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* begin/end bracket host submission, including failure cleanup, without GPU
 * waits. Scratch is Core-owned; its producer must finish before release.
 * No callback may throw across this private C boundary. */
typedef struct LfsHipApplicationDispatch {
    uint32_t struct_bytes;
    void* user;
    int (*begin)(void*,LfsHipError*);
    void (*end)(void*);
    int (*allocate_scratch)(void*, int device, uint64_t producer_queue, uint64_t bytes, uint64_t* address, void** owner, LfsHipError*);
    void (*release_scratch)(void*, void* owner);
} LfsHipApplicationDispatch;
LFS_HIP_TRAIN_API int lfs_hip_application_dispatch_install(const LfsHipApplicationDispatch*,LfsHipError*);
#ifdef __cplusplus
}
#endif
