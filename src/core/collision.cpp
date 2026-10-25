/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/collision.hpp"
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "cuda/collision_voxelization.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cuda_runtime.h>
namespace lfs::core {
    size_t cuda::collisionSortWorkspaceBytes(size_t count) {
        (void)collisionOutputBytes(count);
        return queryCollisionSortWorkspaceBytes(count);
    }
    void CollisionSettings::validate() const {
        if (!std::isfinite(voxel_size) || voxel_size <= 0 || !std::isfinite(opacity) || opacity < 0 || opacity >= 1 ||
            !std::isfinite(max_gaussian_area) || max_gaussian_area < 0)
            throw std::invalid_argument("Cell size must be positive, opacity must be in [0,1), and maximum area must be nonnegative");
        if (!cleanup && max_gaussian_area > 0)
            throw std::invalid_argument("Maximum Gaussian area requires cleanup filtering");
    }
    CollisionSource captureCollisionSource(const SplatData& data, const glm::mat4& transform) {
        struct ReadFence {
            std::vector<cudaStream_t> streams;
            cudaError_t finish() noexcept {
                auto first_error = cudaSuccess;
                for (const auto stream : streams) {
                    const auto status = cudaStreamSynchronize(stream);
                    if (status != cudaSuccess) {
                        if (first_error == cudaSuccess)
                            first_error = status;
                        ensure_cuda_success(status, "cudaStreamSynchronize(collision capture)",
                                            "Finishing source reads before releasing the caller's writer exclusion",
                                            LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                    }
                }
                streams.clear();
                return first_error;
            }
            ~ReadFence() { (void)finish(); }
        } fence;
        fence.streams.reserve(6);
        for (const auto* tensor : {&data.means_raw(), &data.scaling_raw(), &data.rotation_raw(),
                                   &data.opacity_raw(), &data.deleted()})
            if (tensor->is_valid() && tensor->device() == Device::CUDA &&
                std::find(fence.streams.begin(), fence.streams.end(), tensor->stream()) == fence.streams.end())
                fence.streams.push_back(tensor->stream());
        if (!fence.streams.empty() &&
            std::find(fence.streams.begin(), fence.streams.end(), getCurrentCUDAStream()) == fence.streams.end())
            fence.streams.push_back(getCurrentCUDAStream());
        // Clones use the allocation/current stream; activations and deferred inputs
        // may use their home streams. Fence all of them, including partial LOD work.
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
        const auto status = fence.finish();
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("Collision capture failed: ") + cudaGetErrorString(status));
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
    CollisionVolume generateCollision(const CollisionSource& source, CollisionSettings settings,
                                      std::stop_token stop, ProgressCallback progress, std::function<void(float)> resolution, std::function<void(uint64_t)> mixed_blocks) {
        settings.validate();
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
    }
} // namespace lfs::core
