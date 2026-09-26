/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace lfs::core {
    using VulkanDeviceUuid = std::array<std::uint8_t, 16>;

    inline bool matches_vulkan_device_preference(std::string_view requested,
                                                 std::size_t index,
                                                 const VulkanDeviceUuid& uuid) {
        if (requested.empty())
            return true;
        std::size_t requested_index = 0;
        const auto [end, error] = std::from_chars(requested.data(), requested.data() + requested.size(), requested_index);
        if (error == std::errc{} && end == requested.data() + requested.size())
            return index == requested_index;
        std::string normalized;
        for (const unsigned char c : requested) {
            if (c != '-' && c != '{' && c != '}')
                normalized += static_cast<char>(std::tolower(c));
        }
        constexpr char hex[] = "0123456789abcdef";
        std::string device_uuid;
        for (const auto byte : uuid) {
            device_uuid += hex[byte >> 4];
            device_uuid += hex[byte & 15];
        }
        return normalized == device_uuid;
    }

    struct VulkanDeviceCandidate {
        VulkanDeviceUuid uuid{};
        bool discrete = false;
        bool required_features = false;
        bool required_extensions = false;
    };

    inline std::optional<std::size_t> select_headless_vulkan_device(
        std::span<const VulkanDeviceCandidate> candidates,
        std::string_view requested,
        const std::optional<VulkanDeviceUuid>& cuda_uuid) {
        std::optional<std::size_t> selected;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto& candidate = candidates[index];
            if (!candidate.required_features || !candidate.required_extensions ||
                !matches_vulkan_device_preference(requested, index, candidate.uuid) ||
                (cuda_uuid && candidate.uuid != *cuda_uuid))
                continue;
            if (!selected || (!candidates[*selected].discrete && candidate.discrete))
                selected = index;
        }
        return selected;
    }
} // namespace lfs::core
