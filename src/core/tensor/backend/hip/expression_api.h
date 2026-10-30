/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct LfsHipExpression LfsHipExpression;
LFS_HIP_API int lfs_hip_expression_identity_for_device(int32_t device, char* output, size_t capacity, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_compile_for_device(int32_t device, const char* source, char* artifact, size_t capacity,
                                                      size_t* bytes, char* diagnostics, size_t diagnostic_capacity, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_identity(char* output, size_t capacity, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_compile(const char* source, char* artifact, size_t capacity,
                                           size_t* bytes, char* diagnostics, size_t diagnostic_capacity, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_load(LfsHipContext*, const void* artifact, size_t bytes, LfsHipExpression**, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_launch(LfsHipContext*, LfsHipExpression*, const uint32_t* words, size_t count,
                                          size_t work, LfsHipError*);
LFS_HIP_API int lfs_hip_expression_destroy(LfsHipContext*, LfsHipExpression*, LfsHipError*);
#ifdef __cplusplus
}
#endif
