/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "cuda/collision_voxelization.hpp"
#include "core/cuda/morton.cuh"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/cuda_allocation.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <bit>
#include <climits>
#include <format>
#include <numbers>
namespace lfs::core::cuda {
    namespace {
        template <class T>
        struct Buffer {
            UniqueCudaAllocation<StreamOrderedCudaAllocator> storage;
            T* data = nullptr;
            size_t capacity = 0;
            void reserve(size_t n) {
                if (n <= capacity)
                    return;
                storage = UniqueCudaAllocation<StreamOrderedCudaAllocator>(checked_product(n, sizeof(T), "Collision allocation"), getCurrentCUDAStream(), "collision");
                data = storage.template as<T>();
                capacity = n;
            }
        };
    } // namespace
    CollisionBricks voxelizeCollision(const CollisionSource& s, CollisionSettings settings, std::stop_token stop, ProgressCallback progress, const std::function<void(float)>& resolution, const std::function<void(uint64_t)>& mixed_blocks) {
        reportProgress(stop, progress, 0, "Preparing collision geometry");
        if (!s.means.is_valid() || s.means.ndim() != 2 || s.means.shape()[1] != 3)
            throw std::invalid_argument("Collision means must have shape [N,3]");
        const auto total = s.means.shape()[0];
        if (!total || total > size_t(INT_MAX))
            throw std::invalid_argument("Collision source count exceeds the signed tensor index range or is empty");
        const auto stream = getCurrentCUDAStream();
        if (!MemoryPressureCoordinator::instance().preflight({.operation = "Collision geometry",
                                                              .temporary_device_bytes = checked_product(total, 2 * sizeof(CollisionGaussianGpu) + 72, "Collision geometry")})
                 .ok)
            throw std::runtime_error("Insufficient GPU memory for collision geometry; free memory and retry");
        const auto blocks = [](size_t n) { return unsigned((n + 255) / 256); };
        Buffer<CollisionGaussianGpu> gs;
        gs.reserve(total);
        auto metrics = Tensor::empty({total, 3}, Device::GPU);
        auto bounds = Tensor::empty({6, total}, Device::GPU);
        auto errors = Tensor::empty({1}, Device::GPU, DataType::UInt32);
        LFS_CUDA_CHECK(cudaMemsetAsync(errors.ptr<uint32_t>(), 0, sizeof(uint32_t), stream));
        auto host_check = [&](const Tensor& t, size_t count, DataType type) {
            if (!t.is_valid() || t.numel() != count || t.dtype() != type)
                throw std::invalid_argument("Invalid collision geometry tensor");
        };
        host_check(s.means, total * 3, DataType::Float32);
        host_check(s.scales, total * 3, DataType::Float32);
        host_check(s.rotations, total * 4, DataType::Float32);
        host_check(s.opacity, total, DataType::Float32);
        if (s.deleted.is_valid())
            host_check(s.deleted, total, DataType::Bool);
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                if (!std::isfinite(s.transform[c][r]))
                    throw std::invalid_argument("Non-finite collision source transform");
        if (s.transform[0][3] != 0 || s.transform[1][3] != 0 || s.transform[2][3] != 0 || s.transform[3][3] != 1 ||
            glm::determinant(glm::dmat3(s.transform)) == 0)
            throw std::invalid_argument("Collision source transform must be invertible and affine");
        {
            auto means = s.means.to(Device::GPU, stream).contiguous(), scales = s.scales.to(Device::GPU, stream).contiguous();
            auto rotation = s.rotations.to(Device::GPU, stream).contiguous(), opacity = s.opacity.to(Device::GPU, stream).contiguous();
            auto deleted = s.deleted.is_valid() ? s.deleted.to(Device::GPU, stream).contiguous() : Tensor{};
            prepare_inputs_for_stream({&means, &scales, &rotation, &opacity}, stream);
            if (deleted.is_valid())
                prepare_inputs_for_stream({&deleted}, stream);
            launch_prepare(blocks(total), 256, stream, means.ptr<float>(), scales.ptr<float>(), rotation.ptr<float>(), opacity.ptr<float>(), deleted.is_valid() ? deleted.ptr<bool>() : nullptr, total, 0, s.transform, gs.data, metrics.ptr<float>(), errors.ptr<uint32_t>());
        }
        if (errors.to(Device::CPU).ptr<uint32_t>()[0])
            throw std::invalid_argument("Non-finite Gaussian geometry");
        reportProgress(stop, progress, .04f, "Filtering collision sources");
        std::array<float, 3> fences{INFINITY, INFINITY, INFINITY};
        float thin = -INFINITY;
        for (int a = 0; a < 3; ++a) {
            auto column = metrics.slice(1, a, a + 1).contiguous().reshape({int(total)});
            auto samples = column.masked_select(column.isfinite());
            if (!settings.cleanup || samples.numel() < size_t(a ? 128 : 16))
                continue;
            const auto sorted = samples.sort(0).first;
            const auto q1 = sorted.slice(0, samples.numel() / 4, samples.numel() / 4 + 1).item<float>();
            const auto q3 = sorted.slice(0, 3 * samples.numel() / 4, 3 * samples.numel() / 4 + 1).item<float>();
            fences[a] = q3 + std::max(3 * (q3 - q1), std::log(a ? 8.f : 64.f));
            if (a == 1)
                thin = q1 - std::log(4.f);
        }
        if (settings.max_gaussian_area)
            fences[0] = std::log(settings.max_gaussian_area / float(9 * std::numbers::pi));
        launch_filter(blocks(total), 256, stream, gs.data, metrics.ptr<float>(), bounds.ptr<float>(), total, fences[0], fences[1], fences[2], thin);

        auto area = metrics.slice(1, 0, 1).contiguous().reshape({int(total)});
        auto retained = area.masked_select(area.isfinite());
        if (!retained.numel())
            throw std::runtime_error("Source cleanup excluded all splats; disable cleanup or increase maximum area");
        if (retained.numel() * 10 < total)
            LOG_WARN("Collision source filtering retained {} of {} Gaussians; reduce filtering if coverage is incomplete", retained.numel(), total);
        if (settings.auto_resolution) {
            const auto sorted = retained.sort(0).first;
            const auto median = sorted.slice(0, retained.numel() / 2, retained.numel() / 2 + 1).item<float>();
            settings.voxel_size = std::exp(median * .5f) * std::max(.05f, float(std::sqrt(retained.numel() * 4 * std::numbers::pi / 32'000'000)));
            settings.validate();
        }
        if (resolution)
            resolution(settings.voxel_size);
        auto bound = [&](bool high) {
            auto column = bounds.slice(0, high ? 3 : 0, high ? 6 : 3);
            auto reduced = (high ? column.max(1) : column.min(1)).to(Device::CPU).contiguous();
            return glm::vec3(reduced.ptr<float>()[0], reduced.ptr<float>()[1], reduced.ptr<float>()[2]);
        };
        auto low = bound(false), high = bound(true);
        Buffer<uint64_t> key_in, key_out;
        Buffer<uint32_t> order_in, order_out;
        key_in.reserve(total);
        key_out.reserve(total);
        order_in.reserve(total);
        order_out.reserve(total);
        launch_keys(blocks(total), 256, stream, gs.data, key_in.data, order_in.data, total, low,
                    {morton_multiplier(high.x - low.x), morton_multiplier(high.y - low.y), morton_multiplier(high.z - low.z)});

        sortCollisionSources(key_in.data, key_out.data, order_in.data, order_out.data, total, stream);
        Buffer<CollisionGaussianGpu> sorted;
        sorted.reserve(total);
        launch_gather(blocks(total), 256, stream, gs.data, sorted.data, order_out.data, total);

        gs = std::move(sorted);
        key_in = {};
        key_out = {};
        order_in = {};
        order_out = {};
        metrics = {};
        area = {};
        retained = {};
        const auto leaves = std::bit_ceil(uint32_t((total + 7) / 8));
        Buffer<CollisionBvhNode> bvh;
        bvh.reserve(2 * leaves - 1);
        auto build_bvh = [&] {
            launch_leaf_bounds(blocks(leaves), 256, stream, gs.data, bvh.data, total, leaves, settings.voxel_size);

            for (auto first = leaves / 2; first; first /= 2) {
                launch_parent_bounds(blocks(first), 256, stream, bvh.data, first);
            }
        };
        build_bvh();
        auto keep = Tensor::empty({total}, Device::GPU, DataType::Bool);
        const auto cutoff = -std::log1p(-double(settings.opacity));
        launch_eligible(blocks(total), 256, stream, gs.data, bvh.data, keep.ptr<bool>(), total, cutoff, settings.voxel_size);

        Grid grid{{0, 0, 0}, settings.voxel_size, {1, 1, 1}, cutoff};
        launch_grid_indices(blocks(total), 256, stream, gs.data, bounds.ptr<float>(), keep.ptr<bool>(), total, grid);

        low = bound(false);
        high = bound(true);
        if (!std::isfinite(low.x + low.y + low.z + high.x + high.y + high.z))
            throw std::runtime_error("No occupied support; lower opacity or check the source");
        CollisionBricks result;
        auto& v = result.volume;
        v.settings = settings;
        v.origin = glm::floor((low - settings.voxel_size * .5f) / settings.voxel_size) * settings.voxel_size - 2 * settings.voxel_size;
        const auto dimensions = glm::ceil((glm::dvec3(high) + settings.voxel_size * .5 - glm::dvec3(v.origin)) / double(settings.voxel_size)) + 3.;
        if (glm::any(glm::lessThan(dimensions, glm::dvec3(1))) || glm::any(glm::greaterThan(dimensions, glm::dvec3(1 << 21))) || !std::isfinite(dimensions.x + dimensions.y + dimensions.z))
            throw std::runtime_error("Collision bounds exceed coordinate limits; increase cell size or remove remote outliers");
        v.dimensions = glm::ivec3(dimensions);
        LOG_INFO("Collision grid {} x {} x {} at {} scene units; output limit {} sparse records", v.dimensions.x, v.dimensions.y, v.dimensions.z, settings.voxel_size, max_collision_records);
        v.edge = std::max(4u, std::bit_ceil(unsigned(std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z}))));
        for (int a = 0; a < 3; ++a) {
            grid.origin[a] = v.origin[a];
            grid.dimensions[a] = v.dimensions[a];
        }
        launch_grid_indices(blocks(total), 256, stream, gs.data, bounds.ptr<float>(), nullptr, total, grid);

        build_bvh();
        bounds = {};
        keep = {};
        errors = {};
        struct Frame {
            Buffer<Region> regions;
            uint32_t count = 0, offset = 0;
            int edge = 0;
        };
        std::vector<Frame> frames;
        for (int n = v.edge;; n /= 2) {
            frames.emplace_back();
            frames.back().edge = n;
            if (n == 4)
                break;
        }
        frames[0].regions.reserve(1);
        frames[0].count = 1;
        LFS_CUDA_CHECK(cudaMemsetAsync(frames[0].regions.data, 0, sizeof(Region), stream));
        Buffer<uint32_t> count;
        count.reserve(2);
        Buffer<CollisionBrick> output;
        std::vector<Buffer<CollisionBrick>> chunks;
        size_t result_size = 0;
        uint64_t mixed_total = 0;
        auto preflight = [&](const char* phase, size_t additional, size_t records, size_t peak_bytes) {
            const auto memory = MemoryPressureCoordinator::instance().preflight({.operation = phase, .temporary_device_bytes = additional});
            if (!memory.ok)
                throw std::runtime_error(std::format("{} needs {} additional bytes ({} available including reusable pools); {} records at cell size {}, output peak {} bytes. Free GPU memory or explicitly choose a larger cell size", phase, additional, memory.effective_free_bytes, records, settings.voxel_size, peak_bytes));
        };
        auto append = [&](uint32_t n) {
            if (n > output.capacity)
                throw std::runtime_error("Collision dispatch exceeded its output buffer");
            const auto bytes = collisionOutputBytes(result_size, n);
            if (!n)
                return;
            const auto chunk_bytes = size_t(n) * sizeof(CollisionBrick);
            preflight("Collision output growth", chunk_bytes, bytes / sizeof(CollisionBrick), bytes);
            chunks.emplace_back();
            try {
                chunks.back().reserve(n);
            } catch (const MemoryAllocationError& error) {
                throw std::runtime_error(std::format("Collision output growth failed at {} records ({} bytes, limit {}): {}. Free GPU memory or choose a larger cell size", bytes / sizeof(CollisionBrick), bytes, max_collision_records, error.what()));
            }
            LFS_CUDA_CHECK(cudaMemcpyAsync(chunks.back().data, output.data, size_t(n) * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
            result_size = bytes / sizeof(CollisionBrick);
        };
        const double volume = double(v.edge) * v.edge * v.edge;
        double completed = 0;
        int depth = 0;
        while (depth >= 0) {
            reportProgress(stop, progress, .2f + .72f * float(std::clamp(completed / volume, 0., 1.)), "Voxelizing collision");
            auto& frame = frames[depth];
            if (frame.offset == frame.count) {
                --depth;
                continue;
            }
            uint32_t occupied = 0;
            if (frame.edge > 4) {
                const auto batch = std::min(frame.count - frame.offset, max_regions / 8);
                auto& child = frames[depth + 1];
                child.regions.reserve(size_t(batch) * 8);
                output.reserve(size_t(batch) * 8);
                LFS_CUDA_CHECK(cudaMemsetAsync(count.data, 0, 2 * sizeof(uint32_t), stream));
                launch_split_regions((uint64_t(batch) * 8 + 127) / 128, 128, stream, frame.regions.data, frame.offset, batch, frame.edge, grid, gs.data, bvh.data, child.regions.data, count.data, output.data, count.data + 1);

                LFS_CUDA_CHECK(cudaMemcpyAsync(&child.count, count.data, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(&occupied, count.data + 1, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                append(occupied);
                const double half = frame.edge / 2;
                completed += (uint64_t(batch) * 8 - child.count) * half * half * half;
                frame.offset += batch;
                child.offset = 0;
                if (child.count)
                    ++depth;
            } else {
                const auto batch = std::min(frame.count - frame.offset, 65536u);
                output.reserve(batch);
                LFS_CUDA_CHECK(cudaMemsetAsync(count.data, 0, 2 * sizeof(uint32_t), stream));
                launch_evaluate_bricks(batch, 64, stream, frame.regions.data, frame.offset, grid, gs.data, bvh.data, output.data, count.data);

                uint32_t mixed = 0;
                LFS_CUDA_CHECK(cudaMemcpyAsync(&occupied, count.data, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(&mixed, count.data + 1, 4, cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                mixed_total += mixed;
                if (mixed_blocks)
                    mixed_blocks(mixed_total);
                append(occupied);
                completed += batch * double(frame.edge) * frame.edge * frame.edge;
                frame.offset += batch;
            }
        }
        reportProgress(stop, progress, .93f, "Sorting collision records");
        frames.clear();
        gs = {};
        bvh = {};
        output = {};
        Buffer<CollisionBrick> compact;
        LOG_INFO("Collision generated {} sparse records", result_size);
        const auto sort_bytes = queryCollisionSortWorkspaceBytes(result_size);
        const auto compact_bytes = collisionOutputBytes(result_size);
        const auto peak_bytes = saturating_add(compact_bytes, std::max(compact_bytes, sort_bytes));
        LOG_INFO("Collision compaction: {} records, {} bytes chunks + compact; sort peak {} bytes ({} workspace)", result_size, saturating_add(compact_bytes, compact_bytes), saturating_add(compact_bytes, sort_bytes), sort_bytes);
        preflight("Collision compaction", compact_bytes, result_size, peak_bytes);
        try {
            compact.reserve(result_size);
            size_t offset = 0;
            for (auto& chunk : chunks) {
                LFS_CUDA_CHECK(cudaMemcpyAsync(compact.data + offset, chunk.data, chunk.capacity * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
                offset += chunk.capacity;
                chunk = {};
            }
            // Complete copies and releases before accounting the next phase's pool capacity.
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            preflight("Collision sorting", sort_bytes, result_size, peak_bytes);
            if (result_size)
                sortCollisionBricks(compact.data, result_size, stream);
        } catch (const MemoryAllocationError& error) {
            throw std::runtime_error(std::format("Collision output allocation failed for {} records at cell size {} (compact {} bytes, sort workspace {} bytes). Free GPU memory or explicitly choose a larger cell size: {}", result_size, settings.voxel_size, compact_bytes, sort_bytes, error.what()));
        }
        result.bricks.resize(result_size);
        if (result_size)
            LFS_CUDA_CHECK(cudaMemcpyAsync(result.bricks.data(), compact.data, result_size * sizeof(CollisionBrick), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
} // namespace lfs::core::cuda
