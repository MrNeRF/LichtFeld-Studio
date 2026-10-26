/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "chapter_binary_utils.hpp"
#include "core/collision.hpp"
#include <memory>
#include <stdexcept>

namespace lfs::io::project {
    inline constexpr size_t collision_payload_header_bytes = 17 * sizeof(uint32_t);
    // COLV v1: little-endian metadata followed by unexpanded octree words.
    inline std::vector<std::byte> encodeCollision(const core::CollisionVolume& volume) {
        using namespace chapter_binary;
        if (volume.words.size() > core::CollisionVolume::max_words ||
            volume.words.size() > (std::vector<std::byte>().max_size() - collision_payload_header_bytes) / sizeof(uint32_t))
            throw std::length_error("Collision payload exceeds its addressable representation");
        std::vector<std::byte> result(collision_payload_header_bytes + volume.words.size() * sizeof(uint32_t));
        write_u32(result, 0, 0x564c4f43); // COLV
        write_u32(result, 4, 1);
        for (int i = 0; i < 3; ++i) {
            write_f32(result, 8 + i * 4, volume.origin[i]);
            write_i32(result, 20 + i * 4, volume.dimensions[i]);
        }
        write_i32(result, 32, volume.edge);
        write_u64(result, 36, volume.occupied_count);
        write_f32(result, 44, volume.settings.voxel_size);
        write_f32(result, 48, volume.settings.opacity);
        write_f32(result, 52, volume.settings.max_gaussian_area);
        write_u32(result, 56, volume.settings.auto_resolution);
        write_u32(result, 60, volume.settings.cleanup);
        write_u32(result, 64, static_cast<uint32_t>(volume.words.size()));
        for (size_t i = 0; i < volume.words.size(); ++i)
            write_u32(result, collision_payload_header_bytes + i * sizeof(uint32_t), volume.words[i]);
        return result;
    }

    inline std::shared_ptr<const core::CollisionVolume> decodeCollision(std::span<const std::byte> bytes) {
        using namespace chapter_binary;
        if (bytes.size() < collision_payload_header_bytes + sizeof(uint32_t) ||
            bytes.size() % sizeof(uint32_t))
            throw std::invalid_argument("Invalid collision payload length");
        if (read_u32(bytes, 0) != 0x564c4f43)
            throw std::invalid_argument("Invalid collision payload magic");
        if (read_u32(bytes, 4) != 1)
            throw std::invalid_argument("Unsupported collision payload version");
        const auto automatic = read_u32(bytes, 56), cleanup = read_u32(bytes, 60);
        if (automatic > 1 || cleanup > 1)
            throw std::invalid_argument("Invalid collision settings flags");
        const auto count = read_u32(bytes, 64);
        if (count > core::CollisionVolume::max_words || count != (bytes.size() - collision_payload_header_bytes) / sizeof(uint32_t))
            throw std::invalid_argument("Invalid collision word count");
        auto volume = std::make_shared<core::CollisionVolume>();
        volume->origin = {read_f32(bytes, 8), read_f32(bytes, 12), read_f32(bytes, 16)};
        volume->dimensions = {read_i32(bytes, 20), read_i32(bytes, 24), read_i32(bytes, 28)};
        volume->edge = read_i32(bytes, 32);
        volume->occupied_count = read_u64(bytes, 36);
        volume->settings = {read_f32(bytes, 44), read_f32(bytes, 48), read_f32(bytes, 52), automatic != 0, cleanup != 0};
        volume->words.resize(count);
        for (uint32_t i = 0; i < count; ++i)
            volume->words[i] = read_u32(bytes, collision_payload_header_bytes + i * sizeof(uint32_t));
        volume->validate();
        return volume;
    }
} // namespace lfs::io::project
