/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#if LFS_HAS_CUDA
#include "cuda/collision_voxelization.hpp"
#endif
#include <algorithm>
#include <bit>

namespace lfs::core {
#if LFS_HAS_CUDA
    size_t cuda::collisionSortWorkspaceBytes(size_t count) {
        (void)collisionOutputBytes(count);
        return queryCollisionSortWorkspaceBytes(count);
    }
#endif
    CollisionSource captureCollisionSource(const SplatData& data, const glm::mat4& transform) {
        TensorCompletion fence;
        for (const auto* tensor : {&data.means_raw(), &data.scaling_raw(), &data.rotation_raw(),
                                   &data.opacity_raw(), &data.deleted()})
            if (const auto backend = gpu_backend_of(*tensor))
                fence.include(*backend);
        CollisionSource source;
        if (data.lod_tree && data.lod_tree->has_tree()) {
            if (data.size() != data.lod_tree->total_nodes())
                throw std::runtime_error("Collision requires resident original LOD leaves; load the complete model first");
            auto leaves = make_lod_leaf_view(data);
            source = captureCollisionSource(*leaves, transform);
        } else
            source = {data.means_raw().to(DataType::Float32), data.get_scaling().to(DataType::Float32),
                      data.get_rotation().to(DataType::Float32), data.get_opacity().to(DataType::Float32),
                      data.deleted().is_valid() ? data.deleted().clone() : Tensor{}, transform};
        // Materialize lazy activations while the caller still excludes source writers.
        for (const auto* tensor : {&source.means, &source.scales, &source.rotations, &source.opacity})
            (void)tensor->data_ptr();
        for (const auto* tensor : {&source.means, &source.scales, &source.rotations, &source.opacity, &source.deleted})
            fence.include(*tensor);
        fence.wait();
        return source;
    }
    CollisionVolume generateCollision(const CollisionSource& source, CollisionSettings settings,
                                      std::stop_token stop, ProgressCallback progress, std::function<void(float)> resolution, std::function<void(uint64_t)> mixed_blocks) {
        settings.validate();
#if LFS_HAS_CUDA
        if (default_gpu_backend() != GpuBackend::CUDA)
            throw std::runtime_error("Collision generation requires the CUDA tensor backend");
        auto output = cuda::voxelizeCollision(source, settings, stop, progress, resolution, mixed_blocks);
        auto& v = output.volume;
        auto& bricks = output.bricks;
        v.words.assign(1, 0);
        auto build = [&](auto&& self, uint32_t i, int size, size_t begin, size_t end) -> void {
            if (begin == end)
                return;
            reportProgress(stop, {}, 0, "");
            if ((bricks[begin].morton >> 57) && (4 << (bricks[begin].morton >> 57)) == uint64_t(size)) {
                v.words[i] = 1;
                v.occupied_count += uint64_t(size) * size * size;
            } else if (size == 4) {
                const auto mask = bricks[begin].mask;
                v.occupied_count += std::popcount(mask);
                if (mask == ~uint64_t(0)) {
                    v.words[i] = 1;
                } else {
                    if (v.words.size() > CollisionVolume::max_words - 2)
                        throw std::length_error("Collision brick pointer exceeds the packed address range");
                    v.words[i] = CollisionVolume::brick_tag | uint32_t(v.words.size());
                    v.words.push_back(uint32_t(mask));
                    v.words.push_back(uint32_t(mask >> 32));
                }
            } else {
                if (v.words.size() > CollisionVolume::max_words - 8)
                    throw std::length_error("Collision branch pointer exceeds the packed address range");
                const auto base = uint32_t(v.words.size());
                v.words.resize(base + 8);
                v.words[i] = base + 2;
                const int shift = 3 * (std::countr_zero(unsigned(size)) - 3);
                for (int c = 0; c < 8; ++c) {
                    auto next = begin;
                    while (next < end && ((bricks[next].morton >> shift) & 7) == uint64_t(c))
                        ++next;
                    self(self, base + c, size / 2, begin, next);
                    begin = next;
                }
                if (std::all_of(v.words.begin() + base, v.words.begin() + base + 8, [](auto n) { return n == 1; })) {
                    v.words[i] = 1;
                    v.words.resize(base);
                }
            }
        };
        reportProgress(stop, progress, .96f, "Assembling occupancy");
        build(build, 0, v.edge, 0, bricks.size());
        if (!v.occupied_count)
            throw std::runtime_error("No occupied cells; lower opacity or check the source");
        v.validate();
        reportProgress(stop, progress, 1, "Collision ready");
        return std::move(v);
#else
        throw std::runtime_error("Collision generation requires a build with CUDA support");
#endif
    }
} // namespace lfs::core
