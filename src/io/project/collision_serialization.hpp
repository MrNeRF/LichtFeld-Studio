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
            throw core::collisionError("collision.output_limit", "Collision payload exceeds its addressable representation", lfs::ErrorCode::ResourceExhausted);
        std::vector<std::byte> result(collision_payload_header_bytes + volume.words.size() * sizeof(uint32_t));
        const uint32_t header[]{
            0x564c4f43, 1, // COLV, version
            std::bit_cast<uint32_t>(volume.origin.x), std::bit_cast<uint32_t>(volume.origin.y), std::bit_cast<uint32_t>(volume.origin.z),
            uint32_t(volume.dimensions.x), uint32_t(volume.dimensions.y), uint32_t(volume.dimensions.z), uint32_t(volume.edge),
            uint32_t(volume.occupied_count), uint32_t(volume.occupied_count >> 32),
            std::bit_cast<uint32_t>(volume.settings.voxel_size), std::bit_cast<uint32_t>(volume.settings.opacity), std::bit_cast<uint32_t>(volume.settings.max_gaussian_area),
            volume.settings.auto_resolution, volume.settings.cleanup, uint32_t(volume.words.size())};
        static_assert(sizeof(header) == collision_payload_header_bytes);
        for (size_t i = 0; i < std::size(header); ++i)
            write_u32(result, i * sizeof(uint32_t), header[i]);
        for (size_t i = 0; i < volume.words.size(); ++i)
            write_u32(result, collision_payload_header_bytes + i * sizeof(uint32_t), volume.words[i]);
        return result;
    }

    inline std::shared_ptr<const core::CollisionVolume> decodeCollision(std::span<const std::byte> bytes) {
        using namespace chapter_binary;
        if (bytes.size() < collision_payload_header_bytes + sizeof(uint32_t) ||
            bytes.size() % sizeof(uint32_t))
            throw core::collisionError("collision.data_invalid", "Invalid collision payload length", lfs::ErrorCode::DataLoss);
        if (read_u32(bytes, 0) != 0x564c4f43)
            throw core::collisionError("collision.data_invalid", "Invalid collision payload magic", lfs::ErrorCode::DataLoss);
        if (read_u32(bytes, 4) != 1)
            throw core::collisionError("collision.data_invalid", "Unsupported collision payload version", lfs::ErrorCode::DataLoss);
        const auto automatic = read_u32(bytes, 56), cleanup = read_u32(bytes, 60);
        if (automatic > 1 || cleanup > 1)
            throw core::collisionError("collision.data_invalid", "Invalid collision settings flags", lfs::ErrorCode::DataLoss);
        const auto count = read_u32(bytes, 64);
        if (count > core::CollisionVolume::max_words || count != (bytes.size() - collision_payload_header_bytes) / sizeof(uint32_t))
            throw core::collisionError("collision.data_invalid", "Invalid collision word count", lfs::ErrorCode::DataLoss);
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
