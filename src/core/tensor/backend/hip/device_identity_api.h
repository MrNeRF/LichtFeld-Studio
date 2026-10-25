/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Separate sized extension: preserve the original LfsHipDevice ABI. */
typedef struct LfsHipDeviceIdentity {
    uint32_t struct_bytes;
    uint8_t uuid[16], luid[8];
    uint32_t device_node_mask, integrated;
    char compiled_architecture[256];
} LfsHipDeviceIdentity;
LFS_HIP_API int lfs_hip_device_identity(int32_t ordinal, LfsHipDeviceIdentity*, LfsHipError*);
#ifdef __cplusplus
}
#endif
