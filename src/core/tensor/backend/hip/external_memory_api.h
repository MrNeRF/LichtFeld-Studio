/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "runtime_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Private extension. No foreign C++ owner or SDK type crosses this boundary.
 * The exporting allocation must outlive the returned buffer. The caller
 * synchronizes both APIs before transferring access. Both Win32 NT handles
 * and FDs stay caller-owned; the importer duplicates FDs before passing to HIP.
 * buffer_release drains the HIP queue, unmaps and destroys the import.
 * On a cleanup failure, a non-null output accompanies the error: retain the
 * exporter and retry buffer_release before releasing its memory.
 */
enum { LFS_HIP_EXTERNAL_WIN32 = 1, LFS_HIP_EXTERNAL_FD = 2 };
typedef struct LfsHipExternalMemory {
    uint32_t struct_bytes, handle_type;
    uint64_t handle, allocation_bytes, mapped_bytes;
    uint32_t dedicated, reserved;
} LfsHipExternalMemory;
LFS_HIP_API int lfs_hip_buffer_import_external(LfsHipContext*, const LfsHipExternalMemory*, LfsHipBuffer**, LfsHipError*);
#ifdef __cplusplus
}
#endif
