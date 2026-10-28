/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
// Tensor-free collision volume types; the .licht reader and format tests build them without the tensor library.
namespace lfs::core {
    struct LFS_CORE_API CollisionSettings {
        float voxel_size = .05f, opacity = .1f, max_gaussian_area = 0;
        bool auto_resolution = false, cleanup = true;
        void validate() const;
    };
    struct LFS_CORE_API CollisionVolume {
        static constexpr uint32_t brick_tag = 0x80000000u;
        static constexpr size_t max_words = brick_tag; // The high pointer bit identifies a brick.
        CollisionSettings settings;
        glm::vec3 origin{0};
        glm::ivec3 dimensions{0};
        int edge = 1;
        uint64_t occupied_count = 0;
        // 0/1 uniform; branch = child base + 2; tagged leaf = two mask words.
        std::vector<uint32_t> words;
        bool occupied(glm::ivec3 cell) const;
        void validate() const;
    };
} // namespace lfs::core
