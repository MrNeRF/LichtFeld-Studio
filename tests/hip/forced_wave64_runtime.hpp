/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
// SDK-only test shim, never included by a normal application build. A forced
// wave64 kernel on RDNA still reports the device's native wave32 preference to
// host code. rocPRIM must dispatch the wave64 instantiation compiled for this
// test, otherwise it requests symbols absent from the device image.
#include <hip/hip_runtime.h>
inline hipError_t lfs_hip_test_wave64_device_attribute(int* value, hipDeviceAttribute_t attribute, int device) {
    const auto status = ::hipDeviceGetAttribute(value, attribute, device);
    if (status == hipSuccess && attribute == hipDeviceAttributeWarpSize)
        *value = 64;
    return status;
}
#define hipDeviceGetAttribute lfs_hip_test_wave64_device_attribute
