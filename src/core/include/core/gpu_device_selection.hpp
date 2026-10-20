/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace lfs::core {
    inline std::string gpu_identity_string(std::span<const uint8_t> uuid) {
        constexpr char hex[] = "0123456789abcdef";
        std::string identity;
        for (const auto byte : uuid) {
            identity += hex[byte >> 4];
            identity += hex[byte & 15];
        }
        return identity;
    }
    inline std::string normalize_gpu_identity(std::string_view identity) {
        if (identity.starts_with("GPU-") || identity.starts_with("gpu-"))
            identity.remove_prefix(4);
        std::string normalized;
        for (const unsigned char c : identity)
            if (c != '-' && c != '{' && c != '}' && !std::isspace(c))
                normalized += static_cast<char>(std::tolower(c));
        return normalized;
    }

    template <class Device>
    std::optional<size_t> automatic_gpu_device(std::span<const Device> devices, bool rendering, bool prefer_discrete) {
        std::optional<size_t> selected;
        for (size_t i = 0; i < devices.size(); ++i) {
            if (!(rendering ? devices[i].rendering_unavailable_reason : devices[i].tensor_unavailable_reason).empty())
                continue;
            if (!selected || (prefer_discrete && !devices[*selected].discrete && devices[i].discrete))
                selected = i;
        }
        return selected;
    }

    template <class Device>
    std::optional<size_t> find_gpu_device(std::span<const Device> devices, std::string_view requested) {
        if (requested.empty())
            return std::nullopt;
        const auto identity = normalize_gpu_identity(requested);
        for (size_t i = 0; i < devices.size(); ++i)
            if (!devices[i].uuid.empty() && identity == normalize_gpu_identity(devices[i].uuid))
                return i;
        unsigned index = 0;
        const auto [end, ec] = std::from_chars(requested.data(), requested.data() + requested.size(), index);
        if (ec == std::errc{} && end == requested.data() + requested.size())
            for (size_t i = 0; i < devices.size(); ++i)
                if (devices[i].index == index)
                    return i;
        return std::nullopt;
    }
} // namespace lfs::core
