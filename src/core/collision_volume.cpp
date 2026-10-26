/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include <algorithm>
#include <cmath>

namespace lfs::core {
    void CollisionSettings::validate() const {
        if (!std::isfinite(voxel_size) || voxel_size <= 0 || !std::isfinite(opacity) || opacity < 0 || opacity >= 1 ||
            !std::isfinite(max_gaussian_area) || max_gaussian_area < 0)
            throw std::invalid_argument("Cell size must be positive, opacity must be in [0,1), and maximum area must be nonnegative");
        if (!cleanup && max_gaussian_area > 0)
            throw std::invalid_argument("Maximum Gaussian area requires cleanup filtering");
    }
    bool CollisionVolume::occupied(glm::ivec3 p) const {
        if (glm::any(glm::lessThan(p, glm::ivec3(0))) || glm::any(glm::greaterThanEqual(p, dimensions)))
            return false;
        uint32_t index = 0;
        for (int size = edge;; size /= 2) {
            const auto value = words[index];
            if (value <= 1)
                return value;
            if (value & brick_tag) {
                const auto base = value & ~brick_tag;
                const auto mask = uint64_t(words[base]) | (uint64_t(words[base + 1]) << 32);
                return (mask >> ((p.x & 3) + 4 * (p.y & 3) + 16 * (p.z & 3))) & 1;
            }
            const auto b = glm::ivec3(glm::greaterThanEqual(p, glm::ivec3(size / 2)));
            index = value - 2 + b.x + 2 * b.y + 4 * b.z;
            p -= b * (size / 2);
        }
    }
    void CollisionVolume::validate() const {
        settings.validate();
        if (words.empty() || words.size() > max_words || edge < 4 || edge > (1 << 21) || (edge & (edge - 1)) ||
            glm::any(glm::lessThanEqual(dimensions, glm::ivec3(0))) || glm::any(glm::greaterThan(dimensions, glm::ivec3(edge))) ||
            !std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z))
            throw std::invalid_argument("Invalid collision volume bounds or storage");
        std::vector<bool> seen(words.size());
        uint64_t count = 0;
        auto mark = [&](size_t i) {
            if (i >= words.size() || seen[i])
                throw std::invalid_argument("Invalid collision pointer or shared subtree");
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
                    throw std::invalid_argument("Invalid collision brick");
                mark(base);
                mark(base + 1);
                const auto mask = uint64_t(words[base]) | (uint64_t(words[base + 1]) << 32);
                for (int b = 0; b < 64; ++b)
                    if (mask & (uint64_t(1) << b)) {
                        const auto p = lo + glm::ivec3(b & 3, (b >> 2) & 3, b >> 4);
                        if (glm::any(glm::greaterThanEqual(p, dimensions)))
                            throw std::invalid_argument("Collision brick contains occupied padding");
                        ++count;
                    }
            } else if (v > 1) {
                if (size <= 4 || v - 2 <= i || uint64_t(v) + 6 > words.size())
                    throw std::invalid_argument("Invalid collision branch");
                for (int c = 0; c < 8; ++c)
                    self(self, v - 2 + c, lo + glm::ivec3(c & 1, (c >> 1) & 1, c >> 2) * (size / 2), size / 2);
            }
        };
        visit(visit, 0, {}, edge);
        if (count != occupied_count || std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::invalid_argument("Collision occupancy count or topology is invalid");
    }
} // namespace lfs::core
