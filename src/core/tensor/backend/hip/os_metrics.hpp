/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "device_identity_api.h"
#include <core/gpu_device_info.hpp>
namespace lfs::core::internal {
    LFS_CORE_API void hip_process_budget(const LfsHipDeviceIdentity&, GpuDeviceInfo&);
}
