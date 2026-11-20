/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace lfs::rendering {

    // Records the Vulkan physical-device UUID expected to back any subsequent
    // CUDA/Vulkan interop. Call once at startup, after Vulkan device selection.
    // The first interop init() then verifies the current CUDA device matches;
    // mismatch is a hard failure (returns false from init() with a clear error).
    LFS_CORE_API void setExpectedVulkanDeviceUuid(const std::array<std::uint8_t, 16>& uuid);

    // Lazily verifies the CUDA current device's UUID against the value passed
    // to setExpectedVulkanDeviceUuid(). Returns std::nullopt on success, or an
    // error message on mismatch / missing setup. Result is cached.
    [[nodiscard]] LFS_CORE_API std::optional<std::string> verifyCudaMatchesVulkanDevice();

} // namespace lfs::rendering
