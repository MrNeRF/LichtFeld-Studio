/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/checked_arithmetic.hpp"
#include "core/collision.hpp"
#include "core/cuda/morton.cuh"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cub_workspace.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "cuda/collision_voxelization_kernels.hpp"
#include <bit>
#include <climits>
#include <format>
#include <numbers>
#include <set>
namespace lfs::core::cuda {
    namespace {
        template <size_t N, class... Args>
        void launch(const char (&tag)[N], void (*kernel)(Args...), dim3 blocks, dim3 threads,
                    cudaStream_t stream, std::type_identity_t<Args>... args) {
            void* arguments[]{&args...};
            LFS_CUDA_CHECK(cudaLaunchKernel(kernel, blocks, threads, arguments, 0, stream));
            LFS_CUDA_LAUNCH_CHECK(stream, tag);
        }
        template <class T>
        struct Buffer {
            tensor_ops::ScopedDeviceBuffer storage;
            explicit Buffer(size_t n = 0) { reserve(n); }
            T* data() const { return storage.template as<T>(); }
            size_t capacity() const { return storage.size() / sizeof(T); }
            void reserve(size_t n) {
                if (n > capacity())
                    storage = tensor_ops::ScopedDeviceBuffer(checked_product(n, sizeof(T), "Collision allocation"), getCurrentCUDAStream(), "collision");
            }
        };
    } // namespace
    size_t collisionOutputBytes(size_t count, size_t additional) {
        if (count > max_collision_records || additional > max_collision_records - count)
            throw collisionError("collision.output_limit",
                                 std::format("Collision output record limit exceeded: {} + {} records, limit {}. Choose a larger cell size or Automatic resolution; no output was coarsened or cropped", count, additional, max_collision_records),
                                 ErrorCode::ResourceExhausted, std::format("{} + {} > {}", count, additional, max_collision_records));
        return (count + additional) * sizeof(CollisionBrick);
    }
} // namespace lfs::core::cuda
namespace lfs::core {
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
    CollisionVolume generateCollision(const CollisionSource& s, CollisionSettings settings, std::stop_token stop, ProgressCallback progress, std::function<void(float)> resolution, std::function<void(uint64_t)> mixed_blocks) {
        using namespace cuda;
        settings.validate();
        reportProgress(stop, progress, 0, collisionStage("gallery.transfer.phase.preparing", "Preparing collision geometry"));
        if (!s.means.is_valid() || s.means.ndim() != 2 || s.means.shape()[1] != 3)
            throw std::invalid_argument("Collision means must have shape [N,3]");
        const auto total = s.means.shape()[0];
        if (!total || total > size_t(INT_MAX))
            throw std::invalid_argument("Collision source count exceeds the signed tensor index range or is empty");
        const auto stream = getCurrentCUDAStream();
        if (!MemoryPressureCoordinator::instance().preflight({.operation = "Collision geometry",
                                                              .temporary_device_bytes = checked_product(total, 2 * sizeof(CollisionGaussianGpu) + 72, "Collision geometry")})
                 .ok)
            throw collisionError("collision.memory", "Insufficient GPU memory for collision geometry; free memory and retry", ErrorCode::ResourceExhausted);
        const auto blocks = [](size_t n) { return unsigned((n + 255) / 256); };
        Buffer<CollisionGaussianGpu> gs(total);
        auto metrics = Tensor::empty({total, 3}, Device::CUDA);
        auto bounds = Tensor::empty({6, total}, Device::CUDA);
        auto errors = Tensor::empty({1}, Device::CUDA, DataType::UInt32);
        LFS_CUDA_CHECK(cudaMemsetAsync(errors.ptr<uint32_t>(), 0, sizeof(uint32_t), stream));
        (void)collisionInverse(s.transform);
        {
            auto input = [&](const Tensor& tensor, size_t width, DataType type = DataType::Float32) {
                if (!tensor.is_valid() || tensor.numel() != total * width || tensor.dtype() != type)
                    throw std::invalid_argument("Invalid collision geometry tensor");
                auto result = tensor.to(Device::CUDA, stream).contiguous();
                prepare_inputs_for_stream({&result}, stream);
                return result;
            };
            auto means = input(s.means, 3), scales = input(s.scales, 3);
            auto rotation = input(s.rotations, 4), opacity = input(s.opacity, 1);
            auto deleted = s.deleted.is_valid() ? input(s.deleted, 1, DataType::Bool) : Tensor{};
            launch("collision.prepare", prepare, blocks(total), 256, stream, means.ptr<float>(), scales.ptr<float>(), rotation.ptr<float>(), opacity.ptr<float>(), deleted.is_valid() ? deleted.ptr<bool>() : nullptr, total, s.transform, gs.data(), metrics.ptr<float>(), errors.ptr<uint32_t>());
        }
        if (errors.to(Device::CPU).ptr<uint32_t>()[0])
            throw collisionError("collision.geometry_nonfinite", "Non-finite Gaussian geometry: source or transformed positions, scales, rotations or opacity are not representable", ErrorCode::InvalidArgument);
        reportProgress(stop, progress, .04f, collisionStage("collision.cleanup", "Filtering collision sources"));
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
        launch("collision.filter", filter, blocks(total), 256, stream, gs.data(), metrics.ptr<float>(), bounds.ptr<float>(), total, fences[0], fences[1], fences[2], thin);

        auto area = metrics.slice(1, 0, 1).contiguous().reshape({int(total)});
        auto retained = area.masked_select(area.isfinite());
        if (!retained.numel())
            throw collisionError("collision.cleanup_empty", "Source cleanup excluded all splats; disable cleanup or increase maximum area");
        if (retained.numel() * 10 < total)
            LOG_WARN("Collision source filtering retained {} of {} Gaussians; reduce filtering if coverage is incomplete", retained.numel(), total);
        if (settings.auto_resolution) {
            const auto sorted = retained.sort(0).first;
            const auto median = sorted.slice(0, retained.numel() / 2, retained.numel() / 2 + 1).item<float>();
            // Reserve fourfold surface-sample headroom; all terms scale with source geometry.
            settings.voxel_size = std::exp(median * .5f) * std::max(.1f, float(std::sqrt(retained.numel() * 4 * std::numbers::pi / 8'000'000)));
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
        {
            Buffer<uint64_t> key_in(total), key_out(total);
            Buffer<uint32_t> order_in(total), order_out(total);
            launch("collision.keys", keys, blocks(total), 256, stream, gs.data(), key_in.data(), order_in.data(), total, low,
                   {morton_multiplier(high.x - low.x), morton_multiplier(high.y - low.y), morton_multiplier(high.z - low.z)});

            sortCollisionSources(key_in.data(), key_out.data(), order_in.data(), order_out.data(), total, stream);
            Buffer<CollisionGaussianGpu> sorted(total);
            launch("collision.gather", gather, blocks(total), 256, stream, gs.data(), sorted.data(), order_out.data(), total);

            gs = std::move(sorted);
        }
        metrics = {};
        area = {};
        retained = {};
        const auto leaves = std::bit_ceil(uint32_t((total + 7) / 8));
        Buffer<CollisionBvhNode> bvh(2 * leaves - 1);
        auto build_bvh = [&] {
            launch("collision.leaf_bounds", leaf_bounds, blocks(leaves), 256, stream, gs.data(), bvh.data(), total, leaves, settings.voxel_size);

            for (auto first = leaves / 2; first; first /= 2) {
                launch("collision.parent_bounds", parent_bounds, blocks(first), 256, stream, bvh.data(), first);
            }
        };
        build_bvh();
        auto keep = Tensor::empty({total}, Device::CUDA, DataType::Bool);
        const auto cutoff = -std::log1p(-double(settings.opacity));
        launch("collision.eligible", eligible, blocks(total), 256, stream, gs.data(), bvh.data(), keep.ptr<bool>(), total, cutoff, settings.voxel_size);

        Grid grid{{0, 0, 0}, settings.voxel_size, {1, 1, 1}, cutoff};
        launch("collision.grid_indices", grid_indices, blocks(total), 256, stream, gs.data(), bounds.ptr<float>(), keep.ptr<bool>(), total, grid);

        low = bound(false);
        high = bound(true);
        if (!std::isfinite(low.x + low.y + low.z + high.x + high.y + high.z))
            throw collisionError("collision.empty", "No occupied support; lower opacity or check the source");
        CollisionVolume v{.settings = settings};
        v.origin = glm::floor((low - settings.voxel_size * .5f) / settings.voxel_size) * settings.voxel_size - 2 * settings.voxel_size;
        const auto dimensions = glm::ceil((glm::dvec3(high) + settings.voxel_size * .5 - glm::dvec3(v.origin)) / double(settings.voxel_size)) + 3.;
        if (glm::any(glm::lessThan(dimensions, glm::dvec3(1))) || glm::any(glm::greaterThan(dimensions, glm::dvec3(1 << 21))) || !std::isfinite(dimensions.x + dimensions.y + dimensions.z))
            throw collisionError("collision.output_limit", "Collision bounds exceed coordinate limits; increase cell size or remove remote outliers", ErrorCode::BoundsViolation, std::format("{} x {} x {} cells", dimensions.x, dimensions.y, dimensions.z));
        v.dimensions = glm::ivec3(dimensions);
        LOG_INFO("Collision grid {} x {} x {} at {} scene units; output limit {} sparse records", v.dimensions.x, v.dimensions.y, v.dimensions.z, settings.voxel_size, max_collision_records);
        v.edge = std::max(4u, std::bit_ceil(unsigned(std::max({v.dimensions.x, v.dimensions.y, v.dimensions.z}))));
        grid = {{v.origin.x, v.origin.y, v.origin.z}, settings.voxel_size, {v.dimensions.x, v.dimensions.y, v.dimensions.z}, cutoff};
        launch("collision.grid_indices", grid_indices, blocks(total), 256, stream, gs.data(), bounds.ptr<float>(), nullptr, total, grid);

        build_bvh();
        bounds = {};
        keep = {};
        errors = {};
        std::vector<Buffer<Region>> regions(std::countr_zero(unsigned(v.edge)) - 1);
        regions[0].reserve(1);
        LFS_CUDA_CHECK(cudaMemsetAsync(regions[0].data(), 0, sizeof(Region), stream));
        Buffer<uint32_t> count(2);
        Buffer<CollisionBrick> output;
        std::vector<Buffer<CollisionBrick>> chunks;
        size_t result_size = 0;
        uint64_t mixed_total = 0;
        auto preflight = [&](const char* phase, size_t additional, size_t records, size_t peak_bytes) {
            const auto memory = MemoryPressureCoordinator::instance().preflight({.operation = phase, .temporary_device_bytes = additional});
            if (!memory.ok)
                throw collisionError("collision.memory", std::format("{} needs {} additional bytes ({} available including reusable pools); {} records at cell size {}, output peak {} bytes. Free GPU memory or explicitly choose a larger cell size", phase, additional, memory.effective_free_bytes, records, settings.voxel_size, peak_bytes),
                                     ErrorCode::ResourceExhausted, std::format("{} B / {} B", additional, memory.effective_free_bytes));
        };
        auto append = [&](uint32_t n) {
            if (n > output.capacity())
                throw std::runtime_error("Collision dispatch exceeded its output buffer");
            const auto bytes = collisionOutputBytes(result_size, n);
            if (!n)
                return;
            const auto chunk_bytes = size_t(n) * sizeof(CollisionBrick);
            preflight("Collision output growth", chunk_bytes, bytes / sizeof(CollisionBrick), bytes);
            try {
                chunks.emplace_back(n);
            } catch (const MemoryAllocationError& error) {
                throw collisionError("collision.memory", std::format("Collision output growth failed at {} records ({} bytes, limit {}): {}. Free GPU memory or choose a larger cell size", bytes / sizeof(CollisionBrick), bytes, max_collision_records, error.what()),
                                     ErrorCode::ResourceExhausted, std::format("{} B", bytes));
            }
            LFS_CUDA_CHECK(cudaMemcpyAsync(chunks.back().data(), output.data(), size_t(n) * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
            result_size = bytes / sizeof(CollisionBrick);
        };
        const double volume = double(v.edge) * v.edge * v.edge;
        double completed = 0;
        auto visit = [&](auto&& self, int depth, int edge, uint32_t region_count) -> void {
            const bool split = edge > 4;
            for (uint32_t offset = 0; offset < region_count;) {
                reportProgress(stop, progress, .2f + .72f * float(std::clamp(completed / volume, 0., 1.)), collisionStage("new_project.create", "Voxelizing collision"));
                const auto batch = std::min(region_count - offset, split ? max_regions / 8 : 65536u);
                output.reserve(split ? size_t(batch) * 8 : batch);
                LFS_CUDA_CHECK(cudaMemsetAsync(count.data(), 0, 2 * sizeof(uint32_t), stream));
                if (split) {
                    regions[depth + 1].reserve(size_t(batch) * 8);
                    launch("collision.split_regions", split_regions, (uint64_t(batch) * 8 + 127) / 128, 128, stream, regions[depth].data(), offset, batch, edge, grid, gs.data(), bvh.data(), regions[depth + 1].data(), count.data(), output.data(), count.data() + 1);
                } else {
                    launch("collision.evaluate_bricks", evaluate_bricks, batch, 64, stream, regions[depth].data(), offset, grid, gs.data(), bvh.data(), output.data(), count.data());
                }
                std::array<uint32_t, 2> counts{};
                LFS_CUDA_CHECK(cudaMemcpyAsync(counts.data(), count.data(), sizeof(counts), cudaMemcpyDeviceToHost, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                offset += batch;
                if (split) {
                    const double half = edge / 2;
                    completed += (uint64_t(batch) * 8 - counts[0]) * half * half * half;
                } else {
                    mixed_total += counts[1];
                    if (mixed_blocks)
                        mixed_blocks(mixed_total);
                    completed += batch * double(edge) * edge * edge;
                }
                append(counts[split ? 1 : 0]);
                if (split && counts[0])
                    self(self, depth + 1, edge / 2, counts[0]);
            }
        };
        visit(visit, 0, v.edge, 1);
        reportProgress(stop, progress, .93f, collisionStage("gallery.transfer.phase.processing", "Sorting collision records"));
        regions.clear();
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
                LFS_CUDA_CHECK(cudaMemcpyAsync(compact.data() + offset, chunk.data(), chunk.capacity() * sizeof(CollisionBrick), cudaMemcpyDeviceToDevice, stream));
                offset += chunk.capacity();
                chunk = {};
            }
            // Complete copies and releases before accounting the next phase's pool capacity.
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            preflight("Collision sorting", sort_bytes, result_size, peak_bytes);
            if (result_size)
                sortCollisionBricks(compact.data(), result_size, stream);
        } catch (const MemoryAllocationError& error) {
            throw collisionError("collision.memory", std::format("Collision output allocation failed for {} records at cell size {} (compact {} bytes, sort workspace {} bytes). Free GPU memory or explicitly choose a larger cell size: {}", result_size, settings.voxel_size, compact_bytes, sort_bytes, error.what()),
                                 ErrorCode::ResourceExhausted, std::format("{} B + {} B", compact_bytes, sort_bytes));
        }
        std::vector<CollisionBrick> bricks(result_size);
        if (result_size)
            LFS_CUDA_CHECK(cudaMemcpyAsync(bricks.data(), compact.data(), result_size * sizeof(CollisionBrick), cudaMemcpyDeviceToHost, stream));
        LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
        compact = {};
        v.words.assign(1, 0);
        auto build = [&](auto&& self, uint32_t i, int size, size_t begin, size_t end) -> void {
            if (begin == end)
                return;
            reportProgress(stop, {}, 0, "");
            if (((bricks[begin].morton >> 57) && (4 << (bricks[begin].morton >> 57)) == uint64_t(size)) ||
                (size == 4 && bricks[begin].mask == ~uint64_t(0))) {
                v.words[i] = 1;
                v.occupied_count += uint64_t(size) * size * size;
            } else if (size == 4) {
                const auto mask = bricks[begin].mask;
                v.occupied_count += std::popcount(mask);
                if (v.words.size() > CollisionVolume::max_words - 2)
                    throw std::length_error("Collision brick pointer exceeds the packed address range");
                v.words[i] = CollisionVolume::brick_tag | uint32_t(v.words.size());
                v.words.insert(v.words.end(), {uint32_t(mask), uint32_t(mask >> 32)});
            } else {
                if (v.words.size() > CollisionVolume::max_words - 8)
                    throw std::length_error("Collision branch pointer exceeds the packed address range");
                const auto base = uint32_t(v.words.size());
                v.words.resize(base + 8);
                v.words[i] = base + 2;
                const int shift = 3 * (std::countr_zero(unsigned(size)) - 3);
                for (int c = 0; c < 8; ++c) {
                    const auto next = std::partition_point(bricks.begin() + begin, bricks.begin() + end,
                                                           [&](const auto& brick) { return ((brick.morton >> shift) & 7) == uint64_t(c); }) -
                                      bricks.begin();
                    self(self, base + c, size / 2, begin, next);
                    begin = next;
                }
                if (std::all_of(v.words.begin() + base, v.words.begin() + base + 8, [](auto n) { return n == 1; })) {
                    v.words[i] = 1;
                    v.words.resize(base);
                }
            }
        };
        reportProgress(stop, progress, .96f, collisionStage("gallery.transfer.phase.processing", "Assembling occupancy"));
        build(build, 0, v.edge, 0, bricks.size());
        if (!v.occupied_count)
            throw collisionError("collision.empty", "No occupied cells; lower opacity or check the source");
        v.validate();
        reportProgress(stop, progress, 1, collisionStage("status.ready", "Collision ready"));
        return v;
    }
} // namespace lfs::core
