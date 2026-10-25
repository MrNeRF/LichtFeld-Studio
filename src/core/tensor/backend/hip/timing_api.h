/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct LfsHipTiming LfsHipTiming;
/* Optional extension of the private runtime ABI. Events belong to one context;
 * mark only enqueues. elapsed never waits and reports ready=0 for unset,
 * unfinished, or chronologically reversed marks. Destroy before the context. */
LFS_HIP_API int lfs_hip_timing_create(LfsHipContext*, size_t count, LfsHipTiming**, LfsHipError*);
LFS_HIP_API int lfs_hip_timing_mark(LfsHipContext*, LfsHipTiming*, size_t index, LfsHipError*);
LFS_HIP_API int lfs_hip_timing_wait(LfsHipContext*, LfsHipTiming*, size_t index, LfsHipError*);
LFS_HIP_API int lfs_hip_timing_elapsed(LfsHipContext*, LfsHipTiming*, size_t begin, size_t end,
                                       float* milliseconds, int32_t* ready, LfsHipError*);
LFS_HIP_API int lfs_hip_timing_destroy(LfsHipContext*, LfsHipTiming*, LfsHipError*);
#ifdef __cplusplus
}
#endif
