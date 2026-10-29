/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_completion.hpp"
#include <algorithm>
#include <cstdint>
#include <vector>

namespace lfs::core {
    CollisionSource captureCollisionSource(const SplatData& data, const glm::mat4& transform, size_t begin, size_t count) {
        // Clones use the allocation/current queue; activations and deferred inputs
        // may use their home queues. Fence all of them, including partial LOD work.
        TensorCompletion fence;
        for (const auto* tensor : {&data.means_raw(), &data.scaling_raw(), &data.rotation_raw(),
                                   &data.opacity_raw(), &data.deleted()})
            if (const auto backend = gpu_backend_of(*tensor))
                fence.include(*backend);
        if (begin > data.size() || (count != SIZE_MAX && count > data.size() - begin))
            throw collisionError("collision.source_unavailable", "Collision source range is unavailable");
        count = std::min(count, data.size() - begin);
        const auto slice = [&](const Tensor& tensor) { return tensor.slice(0, begin, begin + count); };
        SplatData view;
        view.means_raw() = slice(data.means_raw());
        view.scaling_raw() = slice(data.scaling_raw());
        view.rotation_raw() = slice(data.rotation_raw());
        view.opacity_raw() = slice(data.opacity_raw());
        if (data.deleted().is_valid())
            view.deleted() = slice(data.deleted()).to(DataType::Bool);
        if (data.lod_tree && data.lod_tree->has_tree()) {
            if (data.size() != data.lod_tree->total_nodes())
                throw collisionError("collision.source_unavailable", "Collision requires resident original LOD leaves; load the complete model first");
            std::vector<bool> interior(count);
            for (size_t i = 0; i < count; ++i)
                interior[i] = data.lod_tree->child_count_at(begin + i) != 0;
            auto hidden = Tensor::from_vector(interior, {count}, view.means_raw().device());
            view.deleted() = view.deleted().is_valid() ? hidden.logical_or(view.deleted()) : hidden;
            if (data.lod_tree->lod_opacity_encoded)
                view.opacity_raw() = view.opacity_raw().logit();
        }
        CollisionSource source{view.means_raw().to(DataType::Float32), view.get_scaling().to(DataType::Float32),
                               view.get_rotation().to(DataType::Float32), view.get_opacity().to(DataType::Float32),
                               view.deleted(), transform};
        // Materialize lazy activations while the caller still excludes source writers.
        for (const auto* tensor : {&source.means, &source.scales, &source.rotations, &source.opacity, &source.deleted})
            if (tensor->is_valid()) {
                (void)tensor->data_ptr();
                fence.include(*tensor);
            }
        fence.wait();
        return source;
    }
#if !LFS_HAS_CUDA
    // Voxelization runs CUDA kernels; collision_voxelization.cpp implements it in CUDA builds.
    CollisionVolume generateCollision(const CollisionSource&, CollisionSettings settings, std::stop_token,
                                      ProgressCallback, std::function<void(float)>, std::function<void(uint64_t)>) {
        settings.validate();
        throw collisionError("collision.cuda_required", "Collision generation requires a build with CUDA support", ErrorCode::Unsupported);
    }
#endif
} // namespace lfs::core
