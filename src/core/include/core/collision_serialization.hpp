/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <bit>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>

namespace lfs::core {
    inline constexpr size_t collision_payload_header_bytes = 17 * sizeof(uint32_t);
    // COLV v1: little-endian metadata followed by unexpanded octree words.
    inline std::vector<std::byte> encodeCollision(const CollisionVolume& volume,
                                                  size_t byte_budget = std::numeric_limits<size_t>::max()) {
        if (byte_budget < collision_payload_header_bytes ||
            volume.words.size() > CollisionVolume::max_words ||
            volume.words.size() > (std::vector<std::byte>().max_size() - collision_payload_header_bytes) / sizeof(uint32_t) ||
            volume.words.size() > (byte_budget - collision_payload_header_bytes) / sizeof(uint32_t))
            throw std::length_error("Collision payload exceeds its addressable representation or requested budget");
        std::vector<std::byte> result;
        result.reserve(collision_payload_header_bytes + volume.words.size() * sizeof(uint32_t));
        const auto put = [&](uint32_t word) {
            for (int shift = 0; shift < 32; shift += 8)
                result.push_back(static_cast<std::byte>(word >> shift));
        };
        put(0x564c4f43); // COLV
        put(1);
        for (int i = 0; i < 3; ++i)
            put(std::bit_cast<uint32_t>(volume.origin[i]));
        for (int i = 0; i < 3; ++i)
            put(static_cast<uint32_t>(volume.dimensions[i]));
        put(volume.edge);
        put(static_cast<uint32_t>(volume.occupied_count));
        put(static_cast<uint32_t>(volume.occupied_count >> 32));
        put(std::bit_cast<uint32_t>(volume.settings.voxel_size));
        put(std::bit_cast<uint32_t>(volume.settings.opacity));
        put(std::bit_cast<uint32_t>(volume.settings.max_gaussian_area));
        put(volume.settings.auto_resolution);
        put(volume.settings.cleanup);
        put(static_cast<uint32_t>(volume.words.size()));
        for (const auto word : volume.words)
            put(word);
        return result;
    }

    inline std::shared_ptr<const CollisionVolume> decodeCollision(
        std::span<const std::byte> bytes, size_t byte_budget = std::numeric_limits<size_t>::max()) {
        if (bytes.size() > byte_budget)
            throw std::length_error("Collision payload exceeds the project memory budget");
        if (bytes.size() < collision_payload_header_bytes + sizeof(uint32_t) ||
            bytes.size() % sizeof(uint32_t))
            throw std::invalid_argument("Invalid collision payload length");
        size_t offset = 0;
        const auto get = [&]() {
            uint32_t word = 0;
            for (int shift = 0; shift < 32; shift += 8)
                word |= uint32_t(std::to_integer<uint8_t>(bytes[offset++])) << shift;
            return word;
        };
        if (get() != 0x564c4f43)
            throw std::invalid_argument("Invalid collision payload magic");
        if (get() != 1)
            throw std::invalid_argument("Unsupported collision payload version");
        auto volume = std::make_shared<CollisionVolume>();
        for (int i = 0; i < 3; ++i)
            volume->origin[i] = std::bit_cast<float>(get());
        for (int i = 0; i < 3; ++i)
            volume->dimensions[i] = std::bit_cast<int32_t>(get());
        volume->edge = std::bit_cast<int32_t>(get());
        const uint64_t low = get(), high = get();
        volume->occupied_count = low | (high << 32);
        volume->settings.voxel_size = std::bit_cast<float>(get());
        volume->settings.opacity = std::bit_cast<float>(get());
        volume->settings.max_gaussian_area = std::bit_cast<float>(get());
        const auto automatic = get(), cleanup = get();
        if (automatic > 1 || cleanup > 1)
            throw std::invalid_argument("Invalid collision settings flags");
        volume->settings.auto_resolution = automatic;
        volume->settings.cleanup = cleanup;
        const auto count = get();
        if (count > CollisionVolume::max_words || count != (bytes.size() - offset) / sizeof(uint32_t))
            throw std::invalid_argument("Invalid collision word count");
        volume->words.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            volume->words.push_back(get());
        volume->validate();
        return volume;
    }
} // namespace lfs::core
