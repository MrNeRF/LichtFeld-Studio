// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/gpu_device_info.hpp"
#include <algorithm>

namespace lfs::core {
    bool gpu_device_matches_uuid(const GpuDeviceInfo& device, std::span<const uint8_t, 16> uuid) {
        return std::ranges::equal(device.uuid, uuid);
    }
} // namespace lfs::core
