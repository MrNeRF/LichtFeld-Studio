/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "cuda/collision_voxelization.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>

namespace lfs::core::collision_detail {
    // Balanced median splits end at <=8 sources. Compute exact preorder offsets
    // without atomics, over-allocation or an auxiliary counting traversal.
    inline size_t collisionBvhNodeCount(size_t count) {
        if (count <= 8)
            return 1;
        const auto level_width = size_t(1) << (std::bit_width(count / 8) - 1);
        const auto leaves = count / level_width > 8 ? 2 * level_width : level_width + count - 8 * level_width;
        return 2 * leaves - 1;
    }
    struct CollisionBvh {
        std::unique_ptr<cuda::CollisionBvhNode[]> nodes;
        size_t size;
        std::span<const cuda::CollisionBvhNode> view() const { return {nodes.get(), size}; }
    };
    template <class Gaussian>
    CollisionBvh buildCollisionBvh(std::span<Gaussian> gaussians, std::stop_token stop = {}, unsigned parallelism = 8) {
        auto check = [&] {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        };
        check();
        if (gaussians.empty())
            throw std::invalid_argument("No prepared collision Gaussians");
        const auto node_count = collisionBvhNodeCount(gaussians.size());
        if (node_count > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Collision acceleration structure budget exceeded");
        // Uninitialized storage is completely assigned before upload. Each child
        // owns disjoint Gaussian and node spans, preserving serial ordering.
        CollisionBvh result{std::make_unique_for_overwrite<cuda::CollisionBvhNode[]>(node_count), node_count};
        const unsigned workers = std::clamp(parallelism, 1u, 8u);
        auto build = [&](auto&& self, uint32_t index, size_t begin, size_t end, unsigned depth) -> void {
            check();
            auto& node = result.nodes[index];
            for (int axis = 0; axis < 3; ++axis) {
                node.min[axis] = std::numeric_limits<int>::max();
                node.max[axis] = std::numeric_limits<int>::min();
            }
            for (size_t i = begin; i < end; ++i)
                for (int axis = 0; axis < 3; ++axis) {
                    node.min[axis] = std::min(node.min[axis], gaussians[i].min[axis]);
                    node.max[axis] = std::max(node.max[axis], gaussians[i].max[axis]);
                }
            node.left = node.right = 0;
            node.begin = uint32_t(begin);
            node.count = uint32_t(end - begin);
            if (end - begin <= 8) {
                float support = 0;
                for (size_t i = begin; i < end; ++i)
                    support = std::max(support, std::min({gaussians[i].extent[0], gaussians[i].extent[1], gaussians[i].extent[2]}));
                node.left = std::bit_cast<uint32_t>(support);
                return;
            }
            int axis = 0;
            for (int a = 1; a < 3; ++a)
                if (int64_t(node.max[a]) - node.min[a] > int64_t(node.max[axis]) - node.min[axis])
                    axis = a;
            const auto middle = begin + (end - begin) / 2;
            std::nth_element(gaussians.begin() + begin, gaussians.begin() + middle, gaussians.begin() + end,
                             [&](const auto& a, const auto& b) {
                                 return int64_t(a.min[axis]) + a.max[axis] < int64_t(b.min[axis]) + b.max[axis];
                             });
            node.left = index + 1;
            node.right = node.left + uint32_t(collisionBvhNodeCount(middle - begin));
            node.count = 0;
            if (workers > 1 && end - begin >= 65536 && depth < 3) {
                tbb::parallel_invoke([&] { self(self, node.left, begin, middle, depth + 1); },
                                     [&] { self(self, node.right, middle, end, depth + 1); });
            } else {
                self(self, node.left, begin, middle, depth + 1);
                self(self, node.right, middle, end, depth + 1);
            }
            auto support = [&](uint32_t child) {
                const auto& n = result.nodes[child];
                return std::bit_cast<float>(n.count ? n.left : n.begin);
            };
            node.begin = std::bit_cast<uint32_t>(std::max(support(node.left), support(node.right)));
        };
        tbb::task_arena arena(workers);
        arena.execute([&] { build(build, 0, 0, gaussians.size(), 0); });
        return result;
    }
} // namespace lfs::core::collision_detail
