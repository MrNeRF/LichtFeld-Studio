/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "collision_bvh.hpp"
#include "collision_sparse.hpp"
#include "cuda/collision_voxelization.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>

namespace lfs::core {
    CollisionVolume voxelizeCollisionGpu(std::span<const CollisionGaussian> source, CollisionSettings settings,
                                         std::stop_token stop, CollisionProgress progress) {
        settings.validate();
        if (source.empty() || source.size() > CollisionGaussian::max_count)
            throw std::invalid_argument("Invalid collision source count");
        auto check = [&] {
            if (stop.stop_requested())
                throw std::runtime_error("Collision generation cancelled");
        };
        auto report = [&](float value) {
            check();
            if (progress)
                progress(std::format("Voxelizing GPU: {}/100", int(value * 100)));
        };
        report(0);
        auto input = collision_detail::prepareCollision(source, settings, stop);
        if (progress)
            progress("Voxelizing acceleration structure (CPU)");
        auto& gs = input.gaussians;
        auto bvh = collision_detail::buildCollisionBvh(std::span(gs), stop);
        report(.15f);
        auto volume = std::move(input.volume);
        collision_detail::Tree tree(volume.dimensions, stop);
        static_assert(sizeof(collision_detail::PreparedGaussian) == sizeof(cuda::CollisionGaussianGpu));
        static_assert(offsetof(collision_detail::PreparedGaussian, inverse) == offsetof(cuda::CollisionGaussianGpu, inverse));
        static_assert(offsetof(collision_detail::PreparedGaussian, min) == offsetof(cuda::CollisionGaussianGpu, min));
        auto bricks = cuda::voxelizeCollisionBricks(
            {reinterpret_cast<const cuda::CollisionGaussianGpu*>(gs.data()), gs.size()}, bvh.view(), tree.size,
            {volume.dimensions.x, volume.dimensions.y, volume.dimensions.z}, {volume.origin.x, volume.origin.y, volume.origin.z},
            settings.voxel_size, -std::log1p(-double(settings.opacity)), stop, report);
        if (bricks.empty())
            throw std::runtime_error("No occupied voxels; lower opacity or voxel size");
        report(.97f);
        if (progress)
            progress("Voxelizing sorting occupied bricks (CPU)");
        // Only compact occupied 4^3 masks return to the host. Sorted prefixes
        // build the canonical Walk/archive tree directly, without per-voxel inserts.
        std::sort(bricks.begin(), bricks.end(), [](auto a, auto b) { return (a.morton & cuda::CollisionBrick::key_mask) < (b.morton & cuda::CollisionBrick::key_mask); });
        if (progress)
            progress("Voxelizing assembling occupancy (CPU)");
        auto allocate = [&] {
            if (tree.nodes.size() + 8 > CollisionVolume::max_tree_words)
                throw std::runtime_error("Collision CPU tree budget exceeded");
            auto base = uint32_t(tree.nodes.size());
            tree.nodes.resize(tree.nodes.size() + 8);
            return base;
        };
        auto merge = [&](uint32_t index, uint32_t base) {
            auto first = tree.nodes[base];
            if (first <= 1 && std::all_of(tree.nodes.begin() + base, tree.nodes.begin() + base + 8, [&](auto v) { return v == first; })) {
                tree.nodes[index] = first;
                tree.nodes.resize(base);
            }
        };
        auto maskTree = [&](auto&& self, uint32_t index, glm::ivec3 p, int n, uint64_t mask) -> void {
            uint64_t region = 0;
            for (int z = p.z; z < p.z + n; ++z)
                for (int y = p.y; y < p.y + n; ++y)
                    for (int x = p.x; x < p.x + n; ++x)
                        region |= uint64_t(1) << (x + 4 * y + 16 * z);
            if (!(mask & region))
                return;
            if ((mask & region) == region) {
                tree.nodes[index] = 1;
                return;
            }
            const auto base = allocate();
            tree.nodes[index] = base + 2;
            for (int c = 0; c < 8; ++c)
                self(self, base + c, p + glm::ivec3(c & 1, (c >> 1) & 1, c >> 2) * (n / 2), n / 2, mask);
        };
        auto assemble = [&](auto&& self, uint32_t index, int n, size_t begin, size_t end) -> void {
            if (begin == end)
                return;
            check();
            if ((bricks[begin].morton >> 57) && (4 << (bricks[begin].morton >> 57)) == uint64_t(n)) {
                tree.nodes[index] = 1;
                return;
            }
            if (n <= 4) {
                maskTree(maskTree, index, {}, n, bricks[begin].mask);
                return;
            }
            const auto base = allocate();
            tree.nodes[index] = base + 2;
            const int shift = 3 * (std::countr_zero(unsigned(n)) - 3);
            for (int c = 0; c < 8; ++c) {
                auto next = begin;
                while (next < end && ((bricks[next].morton >> shift) & 7) == uint64_t(c))
                    ++next;
                self(self, base + c, n / 2, begin, next);
                begin = next;
            }
            merge(index, base);
        };
        assemble(assemble, 0, tree.size, 0, bricks.size());
        volume.tree = std::move(tree.nodes);
        volume.tree_size = tree.size;
        if (progress)
            progress("Voxelizing validating occupancy (CPU)");
        volume.validate();
        report(1);
        return volume;
    }
} // namespace lfs::core
