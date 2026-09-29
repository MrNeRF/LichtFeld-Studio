/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/memory_pressure.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "lfs/training/ops/pair_sort_vulkan.hpp"
#include "vulkan/fast_state.hpp"
#include <bit>
#include <climits>
#include <cmath>

namespace lfs::training::vulkan {
    using namespace gpu_ops;
    using core::DataType;
    using core::Device;
    std::vector<StorageRef> fast_storage(const FastState& s) {
        std::vector<StorageRef> result;
        for (const auto& t : s.inputs)
            if (t.is_valid() && t.numel())
                result.push_back(ref(t));
        for (const auto* t : {&s.projected, &s.visibility, &s.offsets, &s.original_to_work, &s.work_to_original,
                              &s.counts, &s.keys_a, &s.keys_b, &s.values_a, &s.values_b, &s.ranges, &s.transmittance, &s.last,
                              &s.image, &s.alpha, &s.depth, &s.normal})
            if (t->is_valid() && t->numel())
                result.push_back(ref(*t));
        return result;
    }
    namespace {
        Tensor inclusive_counts(FastState& s, In input, size_t level = 0) {
            auto output = s.temporary(12 + level * 2, input.numel(), DataType::Int64);
            const size_t blocks = (input.numel() + 255) / 256;
            auto totals = s.temporary(13 + level * 2, blocks, DataType::Int64);
            FastPush p{};
            p.count = input.numel();
            p.stage = 6;
            p.counts = address(input);
            p.offsets = address(output);
            p.visibility = address(totals);
            const std::array reads{ref(input)};
            const std::array writes{ref(output), ref(totals)};
            dispatch("fast_forward", p, reads, writes, groups(input.numel()), p.stage);
            if (blocks > 1) {
                auto prefix = inclusive_counts(s, totals, level + 1);
                p.stage = 7;
                p.visibility = address(prefix);
                const std::array offset_reads{ref(output), ref(prefix)};
                const std::array offset_writes{ref(output)};
                dispatch("fast_forward", p, offset_reads, offset_writes, groups(input.numel()), p.stage);
            }
            return output;
        }
        void check(In t, size_t elements) {
            LFS_ASSERT_MSG(t.is_valid() && t.is_contiguous() && t.dtype() == DataType::Float32 && t.numel() == elements,
                           "Fast input must be contiguous float with the expected size");
            if (elements)
                (void)ref(t);
        }
        void launch(FastState& s, uint32_t stage, size_t work) {
            s.push.stage = stage;
            std::vector<StorageRef> reads, writes;
            const auto append = [](auto& refs, std::initializer_list<const Tensor*> tensors) {
                for (const auto* tensor : tensors)
                    if (tensor->is_valid() && tensor->numel())
                        refs.push_back(ref(*tensor));
            };
            if (stage == 2) {
                for (size_t i = 0; i < 9; ++i)
                    append(reads, {&s.inputs[i]});
                append(reads, {&s.inputs[11]});
                append(writes, {&s.projected, &s.counts, &s.original_to_work,
                                &s.work_to_original, &s.inputs[11]});
            } else if (stage == 3) {
                append(reads, {&s.projected, &s.offsets});
                append(writes, {&s.keys_a, &s.values_a});
            } else if (stage == 4) {
                append(reads, {&s.keys_a, &s.keys_b, &s.values_a, &s.values_b});
                append(writes, {&s.ranges});
            } else {
                LFS_ASSERT_MSG(stage == 5, "Unexpected forward raster stage");
                append(reads, {&s.projected, &s.values_a, &s.values_b, &s.ranges,
                               &s.inputs[9], &s.inputs[10]});
                append(writes, {&s.image, &s.alpha, &s.depth, &s.normal,
                                &s.transmittance, &s.last});
            }
            if (s.push.padding != 0 && stage >= 3 && stage <= 5) {
                const size_t word = stage == 3 ? 4 : stage == 4 ? 12
                                                                : 20;
                dispatch("fast_forward", s.push, reads, writes, 0, specialization(s.push, stage), ref(s.visibility), word * 4);
            } else
                dispatch("fast_forward", s.push, reads, writes, groups(work), specialization(s.push, stage));
        }
        RasterResult failure(FastState& s, RasterResult::Code code, std::string message) {
            s.message = std::move(message);
            s.live = false;
            return {code, false, s.message};
        }
    } // namespace
    RasterResult fast_forward(FastSaved& saved, const SplatInputs& inputs,
                              In view, In camera, In background, In background_image, const FastParams& params,
                              const RenderOutputs& outputs, Out screen_share) {
        LFS_ASSERT_MSG(saved.backend != nullptr, "Fast forward requires a created state");
        auto& s = static_cast<FastState&>(*saved.backend);
        s.live = false;
        s.message.clear();
        const int width = params.tile_w > 0 ? params.tile_w : params.full_image.w;
        const int height = params.tile_h > 0 ? params.tile_h : params.full_image.h;
        LFS_ASSERT_MSG(width > 0 && height > 0 && uint64_t(width) * height <= INT_MAX, "Fast image exceeds signed indexing");
        LFS_ASSERT_MSG(params.intrinsics.fx > 0 && params.intrinsics.fy > 0 && std::isfinite(params.intrinsics.fx) && std::isfinite(params.intrinsics.fy), "Fast requires positive finite focal lengths");
        const size_t count = inputs.means.shape()[0], pixels = size_t(width) * height;
        if (count == 0)
            return failure(s, RasterResult::Code::Failed, "Fast n_primitives is 0");
        LFS_ASSERT_MSG(count <= INT_MAX, "Fast primitive count exceeds signed indexing");
        check(inputs.means, count * 3);
        check(inputs.raw_scales, count * 3);
        check(inputs.raw_rotations, count * 4);
        check(inputs.raw_opacities, count);
        check(inputs.sh0, count * 3);
        check(view, 16);
        check(camera, 3);
        if (background_image.is_valid() && background_image.numel())
            check(background_image, 3 * pixels);
        else if (background.is_valid())
            check(background, 3);
        if (screen_share.is_valid() && screen_share.numel())
            check(screen_share, count);
        LFS_ASSERT_MSG(params.sh.active_bases == 1 || params.sh.active_bases == 4 || params.sh.active_bases == 9 || params.sh.active_bases == 16, "Fast SH degree must be 0..3");
        LFS_ASSERT_MSG(params.sh.layout_bases >= params.sh.active_bases && params.sh.layout_bases <= 16, "Fast SH layout is smaller than active degree");
        s.inputs = {inputs.means, inputs.raw_scales, inputs.raw_rotations, inputs.raw_opacities, inputs.sh0, inputs.shN,
                    inputs.sh_value_bounds, view, camera, background, background_image, screen_share};
        // Drop the previous step's workspace handles so a stage this step skips
        // does not leave them in fast_storage(). The status buffer persists.
        for (auto* tensor : {&s.projected, &s.visibility, &s.offsets, &s.original_to_work,
                             &s.work_to_original, &s.counts, &s.keys_a, &s.keys_b,
                             &s.values_a, &s.values_b, &s.ranges, &s.transmittance,
                             &s.last})
            *tensor = Tensor{};
        auto& p = s.push;
        p = {};
        p.count = count;
        p.width = width;
        p.height = height;
        p.grid_width = (width + 15) / 16;
        p.grid_height = (height + 15) / 16;
        const uint64_t tiles = uint64_t(p.grid_width) * p.grid_height;
        LFS_ASSERT_MSG(p.grid_width <= 65535 && p.grid_height <= 65535 && tiles <= INT_MAX / 256, "Fast tile grid exceeds indexing capacity");
        const uint32_t tile_bits = std::bit_width(static_cast<uint32_t>(tiles - 1));
        p.depth_bits = std::min(23u, 32u - tile_bits);
        p.active_bases = params.sh.active_bases;
        p.rest = params.sh.layout_bases - 1;
        p.mip = params.mip_filter;
        p.render_normal = params.render_normal;
        p.render_depth = params.render_depth;
        p.fx = params.intrinsics.fx;
        p.fy = params.intrinsics.fy;
        p.cx = params.intrinsics.cx - params.tile_x;
        p.cy = params.intrinsics.cy - params.tile_y;
        p.clip_left = (-0.15f * width - p.cx) / p.fx;
        p.clip_right = (1.15f * width - p.cx) / p.fx;
        p.clip_top = (-0.15f * height - p.cy) / p.fy;
        p.clip_bottom = (1.15f * height - p.cy) / p.fy;
        p.means = address(inputs.means);
        p.scales = address(inputs.raw_scales);
        p.rotations = address(inputs.raw_rotations);
        p.opacities = address(inputs.raw_opacities);
        p.sh0 = address(inputs.sh0);
        p.view = address(view);
        p.camera = address(camera);
        p.background = address(background);
        p.background_image = address(background_image);
        p.screen_share = address(screen_share);
        try {
            s.mark(0);
            if (p.rest) {
                const bool q16 = params.sh.storage == ShStorage::Q16;
                const auto dtype = params.sh.storage == ShStorage::Float32 ? DataType::Float32 : DataType::Float16;
                const size_t width = q16 ? p.rest * 3 : ((p.rest * 3 + 3) / 4) * 4;
                LFS_ASSERT_MSG(inputs.shN.is_contiguous() && inputs.shN.dtype() == dtype &&
                                   inputs.shN.numel() >= ((count + 31) / 32) * 32 * width,
                               "Fast SH storage is smaller than its swizzled layout");
                if (q16)
                    LFS_ASSERT_MSG(inputs.sh_value_bounds.is_contiguous() && inputs.sh_value_bounds.dtype() == DataType::Float32 &&
                                       inputs.sh_value_bounds.numel() >= ((count + 255) / 256) * 2,
                                   "Fast SH bounds are smaller than the live block count");
                p.sh = address(inputs.shN);
                p.sh_bounds = q16 ? address(inputs.sh_value_bounds) : 0;
                p.sh_format = params.sh.storage == ShStorage::Float32 ? 1 : q16 ? 3
                                                                                : 2;
            }
            s.image = Tensor::empty({3, size_t(height), size_t(width)}, Device::GPU);
            s.alpha = Tensor::empty({1, size_t(height), size_t(width)}, Device::GPU);
            s.depth = params.render_depth ? Tensor::empty({1, size_t(height), size_t(width)}, Device::GPU) : Tensor{};
            s.normal = params.render_normal ? Tensor::empty({3, size_t(height), size_t(width)}, Device::GPU) : Tensor{};
            p.image = address(s.image);
            p.alpha = address(s.alpha);
            p.depth = address(s.depth);
            p.normal = address(s.normal);
            s.ranges = s.temporary(0, tiles * 2, DataType::Int32, true);
            p.ranges = address(s.ranges);
            s.transmittance = s.temporary(1, tiles * 256, DataType::Float32);
            s.last = s.temporary(2, tiles * 256, DataType::Int32, true);
            p.transmittance = address(s.transmittance);
            p.last = address(s.last);
            // Tile validation uses the deferred device-fault channel, which is
            // checked by the next queue/readback synchronization.
            p.status = core::internal::acquire_vulkan_context()->fault_address();
            // Dense work ids retain original splat order for stable depth ties.
            p.visible = count;
            s.original_to_work = s.temporary(4, count, DataType::Int32);
            s.work_to_original = s.temporary(5, count, DataType::Int32);
            p.original_to_work = address(s.original_to_work);
            p.work_to_original = address(s.work_to_original);
            s.projected = s.temporary(6, count * sizeof(Projected) / 4, DataType::Float32);
            p.projected = address(s.projected);
            s.counts = s.temporary(7, count, DataType::Int64);
            p.counts = address(s.counts);
            launch(s, 2, count);
            s.mark(1);
            s.mark(2);
            if (p.visible) {
                s.mark(3);
                s.mark(4);
                s.offsets = inclusive_counts(s, s.counts);
                p.offsets = address(s.offsets);
                s.mark(5);
                if (s.indirect) {
                    s.visibility = s.temporary(3, 32, DataType::UInt32);
                    p.visibility = address(s.visibility);
                    p.padding = core::internal::acquire_vulkan_context()->caps().max_workgroup_count[0];
                    size_t capacity = s.scratch[8].is_valid() ? s.scratch[8].numel() : std::min(size_t{1} << 20, count * 8);
                    capacity = std::max(size_t{256}, capacity);
                    for (;;) {
                        // Retained buffers already include bounded growth headroom.
                        for (size_t slot = 8; slot < 12; ++slot) {
                            if (!s.scratch[slot].is_valid() || s.scratch[slot].numel() < capacity)
                                s.scratch[slot] = Tensor::empty({capacity}, Device::GPU, DataType::UInt32);
                        }
                        s.keys_a = Tensor(s.scratch[8]);
                        s.keys_b = Tensor(s.scratch[9]);
                        s.values_a = Tensor(s.scratch[10]);
                        s.values_b = Tensor(s.scratch[11]);
                        p.instances = capacity;
                        p.keys = address(s.keys_a);
                        p.values = address(s.values_a);
                        p.stage = 8;
                        const std::array count_reads{ref(s.offsets)};
                        const std::array count_writes{ref(s.visibility)};
                        dispatch("fast_forward", p, count_reads, count_writes, 1, specialization(p, 8));
                        s.scalar_readback.enqueue_range(s.visibility, 4, 8);
                        s.mark(6);
                        launch(s, 3, p.visible);
                        const bool in_a = vulkan_pair_sort({&s.keys_a, &s.keys_b, &s.values_a, &s.values_b}, p.instances, 0, tile_bits + p.depth_bits, false, nullptr, &s.visibility);
                        p.keys = address(in_a ? s.keys_a : s.keys_b);
                        p.values = address(in_a ? s.values_a : s.values_b);
                        launch(s, 4, p.instances);
                        s.mark(7);
                        s.mark(8);
                        launch(s, 5, tiles * 256);
                        s.mark(9);
                        // Keep the already-recorded indirect forward work running
                        // while the host validates the earlier status copy.
                        if (s.submit_before_status)
                            core::internal::acquire_vulkan_context()->recorders().flush_current();
                        std::array<uint32_t, 2> status{};
                        // Validate before exposing output or scheduling any optimizer writes.
                        const bool ready = s.scalar_readback.poll(std::as_writable_bytes(std::span(status)));
                        if (!ready)
                            s.scalar_readback.wait(std::as_writable_bytes(std::span(status)));
                        if (status[0] == 0) {
                            outputs.image = Tensor(s.image);
                            outputs.alpha = Tensor(s.alpha);
                            outputs.depth = Tensor(s.depth);
                            outputs.normal = Tensor(s.normal);
                            s.live = true;
                            return {RasterResult::Code::Success, status[1] != 0, {}};
                        }
                        if (status[0] == 2)
                            return failure(s, RasterResult::Code::InstanceOverflow, "Fast instance count exceeds signed indexing");
                        uint64_t required = 0;
                        s.scalar_readback.enqueue_range(s.visibility, 24 * 4, sizeof(required));
                        s.scalar_readback.wait(std::as_writable_bytes(std::span(&required, 1)));
                        // Limit speculative slack to 4 MiB per pair buffer.
                        capacity = std::min(uint64_t(INT_MAX), required + std::min((required + 3) / 4, uint64_t{1} << 20));
                        capacity = std::min(size_t(INT_MAX), (capacity + 255) & ~size_t{255});
                    }
                } else {
                    int64_t instances = 0;
                    s.scalar_readback.enqueue_range(s.offsets, (p.visible - 1) * sizeof(instances), sizeof(instances));
                    s.scalar_readback.wait(std::as_writable_bytes(std::span(&instances, 1)));
                    if (instances < 0 || instances > INT_MAX)
                        return failure(s, RasterResult::Code::InstanceOverflow, "Fast instance count exceeds signed indexing");
                    p.instances = static_cast<uint32_t>(instances);
                    s.mark(6);
                    s.keys_a = s.temporary(8, p.instances, DataType::UInt32);
                    s.keys_b = s.temporary(9, p.instances, DataType::UInt32);
                    s.values_a = s.temporary(10, p.instances, DataType::UInt32);
                    s.values_b = s.temporary(11, p.instances, DataType::UInt32);
                    p.keys = address(s.keys_a);
                    p.values = address(s.values_a);
                    launch(s, 3, p.visible);
                    bool in_a = vulkan_pair_sort({&s.keys_a, &s.keys_b, &s.values_a, &s.values_b}, p.instances, 0, tile_bits + p.depth_bits, false);
                    p.keys = address(in_a ? s.keys_a : s.keys_b);
                    p.values = address(in_a ? s.values_a : s.values_b);
                    launch(s, 4, p.instances);
                }
                s.mark(7);
            }
            if (!p.visible) {
                for (size_t event = 3; event <= 7; ++event)
                    s.mark(event);
            }
            s.mark(8);
            launch(s, 5, tiles * 256);
            s.mark(9);
            outputs.image = Tensor(s.image);
            outputs.alpha = Tensor(s.alpha);
            outputs.depth = Tensor(s.depth);
            outputs.normal = Tensor(s.normal);
            s.live = true;
            return {RasterResult::Code::Success, p.instances != 0, {}};
        } catch (const core::MemoryAllocationError& error) {
            return failure(s, RasterResult::Code::ResourceExhausted, error.what());
        } catch (const std::bad_alloc& error) {
            return failure(s, RasterResult::Code::ResourceExhausted, error.what());
        }
    }
} // namespace lfs::training::vulkan
