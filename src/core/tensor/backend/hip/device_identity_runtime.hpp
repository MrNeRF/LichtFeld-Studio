/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "device_identity_api.h"
extern "C" int lfs_hip_device_identity(int32_t ordinal, LfsHipDeviceIdentity* output, LfsHipError* error) {
    return boundary(error, [&] {
        require(output && output->struct_bytes == sizeof(*output), "Invalid HIP device identity descriptor");
        validate_device(ordinal);
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, ordinal));
        LfsHipDeviceIdentity result{};
        result.struct_bytes = sizeof(result);
        static_assert(sizeof(properties.uuid.bytes) == sizeof(result.uuid));
        std::memcpy(result.uuid, properties.uuid.bytes, sizeof(result.uuid));
#ifdef _WIN32
        static_assert(sizeof(properties.luid) == sizeof(result.luid));
        std::memcpy(result.luid, properties.luid, sizeof(result.luid));
        result.device_node_mask = properties.luidDeviceNodeMask;
#endif
        result.integrated = properties.integrated ? 1u : 0u;
        const auto architecture = lfs_hip_base_architecture(properties.gcnArchName);
        const auto supported = lfs_hip_has_architecture(LFS_HIP_COMPILED_ARCH, architecture);
        std::snprintf(result.compiled_architecture, sizeof(result.compiled_architecture), "%.*s",
                      int(supported ? architecture.size() : sizeof("unsupported") - 1), supported ? architecture.data() : "unsupported");
        *output = result;
    });
}
