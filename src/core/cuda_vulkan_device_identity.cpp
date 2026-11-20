/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_vulkan_device_identity.hpp"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <mutex>

namespace lfs::rendering {
    namespace {
        std::mutex g_device_uuid_mutex;
        std::optional<std::array<std::uint8_t, 16>> g_expected_vk_uuid;
        bool g_device_match_resolved = false;
        bool g_device_match_ok = false;
        std::string g_device_match_error;

        std::string formatUuid(const std::array<std::uint8_t, 16>& uuid) {
            std::string s;
            s.reserve(36);
            for (std::size_t i = 0; i < uuid.size(); ++i) {
                if (i == 4 || i == 6 || i == 8 || i == 10) {
                    s.push_back('-');
                }
                std::array<char, 3> buf{};
                std::snprintf(buf.data(), buf.size(), "%02x", uuid[i]);
                s.append(buf.data(), 2);
            }
            return s;
        }

    } // namespace

    void setExpectedVulkanDeviceUuid(const std::array<std::uint8_t, 16>& uuid) {
        std::lock_guard lk(g_device_uuid_mutex);
        g_expected_vk_uuid = uuid;
        g_device_match_resolved = false;
        g_device_match_ok = false;
        g_device_match_error.clear();
    }

    std::optional<std::string> verifyCudaMatchesVulkanDevice() {
        std::lock_guard lk(g_device_uuid_mutex);
        if (g_device_match_resolved) {
            if (g_device_match_ok) {
                return std::nullopt;
            }
            return g_device_match_error;
        }
        g_device_match_resolved = true;
        if (!g_expected_vk_uuid) {
            g_device_match_error =
                "Vulkan device UUID was not registered before CUDA/Vulkan interop init; "
                "call setExpectedVulkanDeviceUuid() once at startup";
            return g_device_match_error;
        }
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDevice failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }
        cudaDeviceProp props{};
        status = cudaGetDeviceProperties(&props, cuda_device);
        if (status != cudaSuccess) {
            g_device_match_error = std::format("cudaGetDeviceProperties failed: {} ({})",
                                               cudaGetErrorName(status), cudaGetErrorString(status));
            return g_device_match_error;
        }
        std::array<std::uint8_t, 16> cuda_uuid_bytes{};
        std::memcpy(cuda_uuid_bytes.data(), props.uuid.bytes, 16);
        if (cuda_uuid_bytes != *g_expected_vk_uuid) {
            g_device_match_error = std::format(
                "CUDA device {} (UUID {}) does not match the selected Vulkan physical device (UUID {}). "
                "Set CUDA_VISIBLE_DEVICES to expose the same GPU to both APIs.",
                cuda_device,
                formatUuid(cuda_uuid_bytes),
                formatUuid(*g_expected_vk_uuid));
            return g_device_match_error;
        }
        g_device_match_ok = true;
        return std::nullopt;
    }

} // namespace lfs::rendering
