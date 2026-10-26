/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/collision_error.hpp"
#include "core/splat_data.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <format>
#include <set>
namespace lfs::core {
    void CollisionSettings::validate() const {
        if (!std::isfinite(voxel_size) || voxel_size <= 0 || !std::isfinite(opacity) || opacity < 0 || opacity >= 1 ||
            !std::isfinite(max_gaussian_area) || max_gaussian_area < 0 || (!cleanup && max_gaussian_area > 0))
            throw collisionError("collision.settings_invalid",
                                 std::format("Invalid collision settings: voxel_size={} must be finite and positive, opacity={} must be finite in [0,1), max_gaussian_area={} must be finite and nonnegative and requires cleanup={} when positive",
                                             voxel_size, opacity, max_gaussian_area, cleanup),
                                 ErrorCode::InvalidArgument,
                                 std::format("voxel_size={}; opacity={}; max_gaussian_area={}; cleanup={}", voxel_size, opacity, max_gaussian_area, cleanup));
    }
    CollisionSource captureCollisionSource(const SplatData& data, const glm::mat4& transform, size_t begin, size_t count) {
        struct ReadFence {
            std::set<cudaStream_t> streams;
            cudaError_t finish() noexcept {
                auto first_error = cudaSuccess;
                for (const auto stream : streams) {
                    const auto status = cudaStreamSynchronize(stream);
                    if (first_error == cudaSuccess)
                        first_error = status;
                    ensure_cuda_success(status, "cudaStreamSynchronize(collision capture)",
                                        "Finishing source reads before releasing the caller's writer exclusion",
                                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
                streams.clear();
                return first_error;
            }
            ~ReadFence() { (void)finish(); }
        } fence;
        for (const auto* tensor : {&data.means_raw(), &data.scaling_raw(), &data.rotation_raw(),
                                   &data.opacity_raw(), &data.deleted()})
            if (tensor->is_valid() && tensor->device() == Device::CUDA)
                fence.streams.insert(tensor->stream());
        if (!fence.streams.empty())
            fence.streams.insert(getCurrentCUDAStream());
        // Clones use the allocation/current stream; activations and deferred inputs
        // may use their home streams. Fence all of them, including partial LOD work.
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
            if (tensor->is_valid())
                (void)tensor->data_ptr();
        LFS_ENSURE_CUDA_SUCCESS(fence.finish(), "Collision capture failed");
        return source;
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
