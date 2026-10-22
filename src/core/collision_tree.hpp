/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/collision.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>
#include <span>
#include <stdexcept>
#include <stop_token>
#include <unordered_set>
#include <vector>

namespace lfs::core::collision_detail {
    struct Leaf {
        glm::ivec3 min;
        int size;
        uint32_t index;
        bool solid;
        glm::ivec3 max() const { return min + size; }
    };

    // 0/1 are uniform empty/solid; other words point to eight contiguous children.
    class Tree {
        std::vector<uint32_t> free_blocks_;
        uint64_t work_ = 0;
        std::stop_token stop_;

        void tick() {
            if ((++work_ & 4095) == 0 && stop_.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
            if (work_ > 4'000'000'000ULL)
                throw std::runtime_error("Sparse collision operation work budget exceeded");
        }
        static bool overlap(glm::ivec3 lo, glm::ivec3 hi, glm::ivec3 min, int n) {
            return glm::all(glm::lessThan(lo, min + n)) && glm::all(glm::greaterThan(hi, min));
        }
        uint32_t split(uint32_t index) {
            const auto value = nodes[index];
            if (value > 1)
                return value - 2;
            uint32_t base;
            if (free_blocks_.empty()) {
                if (nodes.size() + 8 > max_nodes)
                    throw std::runtime_error("Sparse collision node memory budget exceeded");
                base = uint32_t(nodes.size());
                nodes.resize(nodes.size() + 8);
            } else {
                base = free_blocks_.back();
                free_blocks_.pop_back();
            }
            std::fill_n(nodes.begin() + base, 8, value);
            nodes[index] = base + 2;
            return base;
        }
        void release(uint32_t index) {
            const auto value = nodes[index];
            if (value <= 1)
                return;
            const auto base = value - 2;
            for (int i = 0; i < 8; ++i)
                release(base + i);
            free_blocks_.push_back(base);
        }
        void set(uint32_t index, glm::ivec3 min, int n, glm::ivec3 lo, glm::ivec3 hi, bool value) {
            tick();
            if (!overlap(lo, hi, min, n) || nodes[index] == uint32_t(value))
                return;
            if (glm::all(glm::lessThanEqual(lo, min)) && glm::all(glm::greaterThanEqual(hi, min + n))) {
                release(index);
                nodes[index] = value;
                return;
            }
            const auto base = split(index);
            const int half = n / 2;
            for (int i = 0; i < 8; ++i)
                set(base + i, min + glm::ivec3(i & 1, (i >> 1) & 1, i >> 2) * half,
                    half, lo, hi, value);
            const auto first = nodes[base];
            if (first <= 1 && std::all_of(nodes.begin() + base, nodes.begin() + base + 8,
                                          [first](auto v) { return v == first; })) {
                nodes[index] = first;
                free_blocks_.push_back(base);
            }
        }
        template <typename Fn>
        void walk(uint32_t index, glm::ivec3 min, int n, glm::ivec3 lo, glm::ivec3 hi, Fn& fn) const {
            if (!overlap(lo, hi, min, n))
                return;
            auto value = nodes[index];
            if (value <= 1) {
                fn(Leaf{min, n, index, value != 0});
            } else {
                const int half = n / 2;
                for (int i = 0; i < 8; ++i)
                    walk(value - 2 + i, min + glm::ivec3(i & 1, (i >> 1) & 1, i >> 2) * half,
                         half, lo, hi, fn);
            }
        }

    public:
        static constexpr size_t max_nodes = CollisionVolume::max_tree_words;
        int size = 1;
        glm::ivec3 dimensions{1};
        std::vector<uint32_t> nodes{0};
        explicit Tree(glm::ivec3 dims, std::stop_token stop = {})
            : stop_(stop), dimensions(dims) {
            for (int axis = 0; axis < 3; ++axis)
                if (dims[axis] <= 0 || dims[axis] > (1 << 21))
                    throw std::invalid_argument("Sparse collision axis exceeds supported integer precision");
            while (size < std::max({dims.x, dims.y, dims.z}))
                size *= 2;
        }
        Leaf leaf(glm::ivec3 p) const {
            uint32_t index = 0;
            glm::ivec3 min(0);
            int n = size;
            while (nodes[index] > 1) {
                n /= 2;
                auto bit = glm::ivec3(glm::greaterThanEqual(p, min + n));
                index = nodes[index] - 2 + bit.x + 2 * bit.y + 4 * bit.z;
                min += bit * n;
            }
            return {min, n, index, nodes[index] != 0};
        }
        bool occupied(glm::ivec3 p) const {
            return glm::all(glm::greaterThanEqual(p, glm::ivec3(0))) &&
                   glm::all(glm::lessThan(p, dimensions)) && leaf(p).solid;
        }
        int uniform(glm::ivec3 lo, glm::ivec3 hi) const {
            const bool outside = glm::any(glm::lessThan(lo, glm::ivec3(0))) ||
                                 glm::any(glm::greaterThan(hi, dimensions));
            lo = glm::max(lo, glm::ivec3(0));
            hi = glm::min(hi, dimensions);
            if (glm::any(glm::greaterThanEqual(lo, hi)))
                return 0;
            const auto value = outside ? false : occupied(lo);
            std::function<bool(uint32_t, glm::ivec3, int)> matches = [&](uint32_t i, glm::ivec3 min, int n) {
                if (!overlap(lo, hi, min, n))
                    return true;
                if (nodes[i] <= 1)
                    return bool(nodes[i]) == value;
                for (int child = 0; child < 8; ++child)
                    if (!matches(nodes[i] - 2 + child,
                                 min + glm::ivec3(child & 1, (child >> 1) & 1, child >> 2) * (n / 2), n / 2))
                        return false;
                return true;
            };
            return matches(0, {}, size) ? int(value) : -1;
        }
        void setBox(glm::ivec3 lo, glm::ivec3 hi, bool value = true) {
            lo = glm::max(lo, glm::ivec3(0));
            hi = glm::min(hi, dimensions);
            if (glm::all(glm::lessThan(lo, hi)))
                set(0, {}, size, lo, hi, value);
        }
        template <typename Fn>
        void visit(glm::ivec3 lo, glm::ivec3 hi, Fn fn) const {
            lo = glm::max(lo, glm::ivec3(0));
            hi = glm::min(hi, dimensions);
            if (glm::all(glm::lessThan(lo, hi)))
                walk(0, {}, size, lo, hi, fn);
        }
        template <typename Fn>
        void leaves(Fn fn) const { visit({}, dimensions, std::move(fn)); }
        void unite(const Tree& other) {
            other.leaves([&](const Leaf& l) {
                if (l.solid)
                    setBox(l.min, l.max());
            });
        }
        void complement() {
            std::function<void(uint32_t)> flip = [&](uint32_t i) {
                if (nodes[i] <= 1)
                    nodes[i] = !nodes[i];
                else {
                    const auto base = nodes[i] - 2;
                    for (int a = 0; a < 8; ++a)
                        flip(base + a);
                }
            };
            flip(0);
        }
        void compact() {
            auto old = std::move(nodes);
            nodes.assign(1, 0);
            std::function<void(uint32_t, uint32_t)> copy = [&](uint32_t src, uint32_t dst) {
                if (old[src] <= 1) {
                    nodes[dst] = old[src];
                    return;
                }
                auto base = uint32_t(nodes.size());
                nodes.resize(nodes.size() + 8);
                nodes[dst] = base + 2;
                for (int i = 0; i < 8; ++i)
                    copy(old[src] - 2 + i, base + i);
            };
            copy(0, 0);
            free_blocks_.clear();
        }
        Tree flood(std::span<const glm::ivec3> seeds, bool value, bool boundary = false) const {
            Tree result(dimensions, stop_);
            std::unordered_set<uint32_t> visited;
            std::vector<Leaf> queue;
            auto add = [&](const Leaf& l) {
                if (l.solid == value && visited.insert(l.index).second) {
                    if (queue.size() >= 8'000'000)
                        throw std::runtime_error("Sparse collision flood memory budget exceeded");
                    queue.push_back(l);
                }
            };
            for (auto seed : seeds)
                if (glm::all(glm::greaterThanEqual(seed, glm::ivec3(0))) && glm::all(glm::lessThan(seed, dimensions)))
                    add(leaf(seed));
            if (boundary)
                leaves([&](const Leaf& l) {
                    if (glm::any(glm::equal(l.min, glm::ivec3(0))) ||
                        glm::any(glm::greaterThanEqual(l.max(), dimensions)))
                        add(l);
                });
            for (size_t head = 0; head < queue.size(); ++head) {
                if ((head & 4095) == 0 && stop_.stop_requested())
                    throw std::runtime_error("Collision generation cancelled");
                const auto l = queue[head];
                result.setBox(l.min, l.max());
                for (int axis = 0; axis < 3; ++axis)
                    for (int sign : {-1, 1}) {
                        auto lo = l.min, hi = l.max();
                        lo[axis] = sign < 0 ? lo[axis] - 1 : hi[axis];
                        hi[axis] = lo[axis] + 1;
                        visit(lo, hi, add);
                    }
            }
            result.compact();
            return result;
        }
    };
} // namespace lfs::core::collision_detail
