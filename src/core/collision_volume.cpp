/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include <algorithm>
#include <cmath>
#include <format>

namespace lfs::core {
    void CollisionSettings::validate() const {
        if (!std::isfinite(voxel_size) || voxel_size <= 0 || !std::isfinite(opacity) || opacity < 0 || opacity >= 1 ||
            !std::isfinite(max_gaussian_area) || max_gaussian_area < 0 || (!cleanup && max_gaussian_area > 0))
            throw collisionError("collision.settings_invalid",
                                 std::format("Invalid collision settings: voxel_size={} must be finite and positive, opacity={} must be finite in [0,1), max_gaussian_area={} must be finite and nonnegative and requires cleanup={} when positive",
                                             voxel_size, opacity, max_gaussian_area, cleanup),
                                 ErrorCode::InvalidArgument,
                                 std::format("{}; {}; {}; {}", voxel_size, opacity, max_gaussian_area, int(cleanup)));
    }
    bool CollisionVolume::occupied(glm::ivec3 p) const {
        if (glm::any(glm::lessThan(p, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(p, dimensions)))
            return false;
        uint32_t value = words[0];
        for (int size = edge / 2; value > 1 && !(value & brick_tag); size /= 2)
            value = words[value - 2 + bool(p.x & size) + 2 * bool(p.y & size) + 4 * bool(p.z & size)];
        if (value <= 1)
            return value;
        const auto base = value & ~brick_tag;
        const auto mask = uint64_t(words[base]) | (uint64_t(words[base + 1]) << 32);
        return (mask >> ((p.x & 3) + 4 * (p.y & 3) + 16 * (p.z & 3))) & 1;
    }

    void CollisionVolume::validate() const {
        settings.validate();
        if (words.empty() || words.size() > max_words || edge < 4 || edge > (1 << 21) || (edge & (edge - 1)) ||
            glm::any(glm::lessThanEqual(dimensions, glm::ivec3(0))) || glm::any(glm::greaterThan(dimensions, glm::ivec3(edge))) ||
            !std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z))
            throw collisionError("collision.data_invalid", "Invalid collision volume bounds or storage", ErrorCode::InvalidArgument);
        std::vector<bool> seen(words.size());
        uint64_t count = 0;
        auto mark = [&](size_t i) {
            if (i >= words.size() || seen[i])
                throw collisionError("collision.data_invalid", "Invalid collision pointer or shared subtree", ErrorCode::InvalidArgument);
            seen[i] = true;
        };
        auto visit = [&](auto&& self, uint32_t i, glm::ivec3 lo, int size) -> void {
            mark(i);
            const auto v = words[i];
            if (v == 1) {
                const auto n = glm::max(glm::min(lo + size, dimensions) - lo, glm::ivec3(0));
                count += uint64_t(n.x) * n.y * n.z;
            } else if (v & brick_tag) {
                const auto base = v & ~brick_tag;
                if (size != 4 || base <= i)
                    throw collisionError("collision.data_invalid", "Invalid collision brick", ErrorCode::InvalidArgument);
                mark(base);
                mark(base + 1);
                const auto mask = uint64_t(words[base]) | (uint64_t(words[base + 1]) << 32);
                for (int b = 0; b < 64; ++b)
                    if (mask & (uint64_t(1) << b)) {
                        const auto p = lo + glm::ivec3(b & 3, (b >> 2) & 3, b >> 4);
                        if (glm::any(glm::greaterThanEqual(p, dimensions)))
                            throw collisionError("collision.data_invalid", "Collision brick contains occupied padding", ErrorCode::InvalidArgument);
                        ++count;
                    }
            } else if (v > 1) {
                if (size <= 4 || v - 2 <= i || uint64_t(v) + 6 > words.size())
                    throw collisionError("collision.data_invalid", "Invalid collision branch", ErrorCode::InvalidArgument);
                for (int c = 0; c < 8; ++c)
                    self(self, v - 2 + c, lo + glm::ivec3(c & 1, (c >> 1) & 1, c >> 2) * (size / 2), size / 2);
            }
        };
        visit(visit, 0, {}, edge);
        if (count != occupied_count || std::find(seen.begin(), seen.end(), false) != seen.end())
            throw collisionError("collision.data_invalid", "Collision occupancy count or topology is invalid", ErrorCode::InvalidArgument);
    }
    glm::mat4 collisionInverse(const glm::mat4& transform) {
        const double determinant = glm::determinant(glm::dmat3(transform));
        const auto inverse = glm::inverse(transform);
        if (!std::isfinite(determinant) || determinant == 0 || transform[0][3] != 0 ||
            transform[1][3] != 0 || transform[2][3] != 0 || transform[3][3] != 1)
            throw collisionError("collision.source_transform_invalid", "Collision transform must be invertible and affine", ErrorCode::InvalidArgument);
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 4; ++row)
                if (!std::isfinite(transform[column][row]) || !std::isfinite(inverse[column][row]))
                    throw collisionError("collision.source_transform_invalid", "Collision transform or inverse is not finite", ErrorCode::InvalidArgument,
                                         std::format("[{}][{}]: {} / {}", column, row, transform[column][row], inverse[column][row]));
        return inverse;
    }
} // namespace lfs::core
