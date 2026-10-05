/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_rasterizer.hpp"

#include "core/gpu_device_runtime.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "splat_blend_discs.hpp"
#include "splat_blend_gs32.hpp"
#include "splat_blend_gs64.hpp"
#include "splat_blend_gs_batch.hpp"
#include "splat_blend_gs_fast.hpp"
#include "splat_blend_gs_prefix.hpp"
#include "splat_blend_gut32.hpp"
#include "splat_blend_gut64.hpp"
#include "splat_blend_points.hpp"
#include "scratch_arena.hpp"
#include "splat_present.hpp"
#include "splat_tile_binner.hpp"

#include <deque>
#include <format>
#include <map>
#include <span>
#include <tuple>
#include <utility>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;
        constexpr uint32_t kSingleSimd = 128, kSourceSorted = 256, kDepthBatches = 512, kDepthPrefix = 1024;
        constexpr uint32_t kOpaqueBackground = 4096, kOverlay = 1;
        // Below this many sources the extra source sort is not worth its passes.
        constexpr uint32_t kSourceSortMinimum = 4096;
        // Depth batches split tiles longer than this into chunks blended in
        // parallel; a frame enables them once a completed frame had such a tile.
        constexpr uint32_t kDepthChunkSize = 576, kParallelMinTileInstances = 32768;
        // Features whose blend the depth batches do not cover: overlays,
        // portal edges, logical IDs, Spark, GUT legacy color, macro half, panorama.
        constexpr uint32_t kSerialOnlyFlags = 1 | 4 | 8 | 16 | 32 | 64 | 8192;

        struct BlendParameters {
            uint64_t pointers[19] = {};
            uint32_t logical_count = 0, dispatch_flags = 0;
        };
        struct PresentPointers {
            uint64_t pointers[8] = {};
        };
        static_assert(sizeof(BlendParameters) == 160 && sizeof(PresentPointers) == 64);

        lfs::Result<void> failure(std::string detail) {
            return lfs::Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                     .domain = ErrorDomain::Rendering,
                                                     .detail = std::move(detail),
                                                     .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    struct SplatRasterizer::Impl {
        core::GpuBackend backend;
        SplatTileBinner binner;
        std::map<std::pair<uint32_t, bool>, std::unique_ptr<M>> blends;
        std::unique_ptr<M> fast_blend, batch_blend, prefix_blend, present;
        Tensor raster, present_parameters, color, depth, pick, rgba, linear_depth, selection_colors;
        bool presented = false; // rgba and linear_depth hold an image of this extent
        // Replaced scratch returns to the tensor cache, which keeps blocks for
        // reuse; the viewer rarely regrows, so release them a frame later,
        // once their last reads have completed.
        bool trim = false;
        std::function<void(const char*)> marker;
        void mark(const char* stage) {
            if (marker)
                marker(stage);
        }
        uint32_t reserved_splats = 0, reserved_capacity = 0;
        // Depth-batch scratch, sized from the instances of a completed frame.
        Tensor depth_jobs, partial_color, partial_depth, partial_pick;
        uint32_t parallel_instances = 0;
        uint32_t width = 0, height = 0;
        std::deque<core::TensorUpload> uploads;
        // The last completed RasterStatus, read without waiting. It picks
        // source sorting and depth batches for frames of the same source count.
        core::TensorReadback status_readback;
        uint32_t readback_count = 0, previous_count = 0;
        struct {
            uint64_t required = 0;
            uint32_t error = 1, blend_threads = 0, maximum_tile_instances = 0, padding = 0;
        } previous;
        static_assert(sizeof(previous) == SplatTileBinner::kRasterStatusBytes);

        // Whether a completed frame of this source count fit its capacity.
        bool previous_fits(const uint32_t count, const uint32_t capacity) {
            if (status_readback.pending() && status_readback.poll(std::as_writable_bytes(std::span(&previous, 1))))
                previous_count = readback_count;
            return count >= kSourceSortMinimum && count <= capacity && previous_count == count && previous.error == 0;
        }
        // Reserves depth-batch scratch for the previous frame's instances plus
        // headroom; false keeps the serial blend.
        bool reserve_depth_batches(const SplatRasterParameters& frame) {
            const auto needed = uint32_t(std::min<uint64_t>(frame.capacity, previous.required + previous.required / 8));
            if (needed > parallel_instances) {
                const size_t slots = (size_t(needed) + kDepthChunkSize - 1) / kDepthChunkSize + frame.tiles;
                // The job table is filled with -1 every frame; Int32 keeps fill_ available.
                depth_jobs = Tensor::empty({slots * 2}, Device::GPU, DataType::Int32);
                std::tie(partial_color, partial_depth, partial_pick) =
                    std::tuple_cat(carve_arena<3>({slots * 256 * 16, slots * 256 * 16, slots * 256 * 4}));
                parallel_instances = needed;
                trim = true;
            }
            return parallel_instances != 0;
        }
        void read_status(const uint32_t count) {
            if (status_readback.pending())
                return;
            status_readback.enqueue(binner.status());
            readback_count = count;
        }

        explicit Impl(const core::GpuBackend b) : backend(b), binner(b) {}

        // Per-frame parameters ride the open batch, never waiting on the GPU.
        void upload_bytes(Tensor& destination, const std::span<const std::byte> bytes) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid() || destination.bytes() != bytes.size())
                destination = Tensor::empty({bytes.size()}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, bytes);
        }
        template <class T>
        void upload(Tensor& destination, const T& value) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid())
                destination = Tensor::empty({sizeof(T)}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, std::as_bytes(std::span(&value, 1)));
        }

        lfs::Result<M*> blend(const SplatRasterMode mode, const uint32_t flags) {
            const auto load = [&](std::span<const M::Entry> entries, std::unique_ptr<M>& slot) -> lfs::Result<M*> {
                if (!slot) {
                    auto loaded = M::load(entries, backend);
                    if (!loaded)
                        return lfs::Result<M*>(std::move(loaded).error());
                    slot = std::move(*loaded);
                }
                return slot.get();
            };
            // The opaque-background bit changes no Slang blend arithmetic.
            if (mode == SplatRasterMode::Gaussian) {
                switch (flags & ~(kSourceSorted | kOpaqueBackground)) {
                case kSingleSimd: return load(splat_blend_gs_fast_entries(), fast_blend);
                case kSingleSimd | kDepthBatches: return load(splat_blend_gs_batch_entries(), batch_blend);
                case kSingleSimd | kDepthBatches | kDepthPrefix: return load(splat_blend_gs_prefix_entries(), prefix_blend);
                default: break;
                }
            }
            const bool single = (flags & kSingleSimd) != 0;
            auto& slot = blends[{uint32_t(mode), single}];
            switch (mode) {
            case SplatRasterMode::Gaussian: return load(single ? splat_blend_gs32_entries() : splat_blend_gs64_entries(), slot);
            case SplatRasterMode::Gut: return load(single ? splat_blend_gut32_entries() : splat_blend_gut64_entries(), slot);
            case SplatRasterMode::Points: return load(splat_blend_points_entries(), slot);
            case SplatRasterMode::Discs: return load(splat_blend_discs_entries(), slot);
            }
            return lfs::Result<M*>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Rendering,
                                          .detail = std::format("Unknown splat raster mode {}", uint32_t(mode)),
                                          .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    };

    SplatRasterizer::SplatRasterizer(const core::GpuBackend backend) : impl_(std::make_unique<Impl>(backend)) {}
    SplatRasterizer::~SplatRasterizer() = default;

    lfs::Result<void> SplatRasterizer::reserve(const uint32_t splats, const uint32_t width, const uint32_t height, const uint32_t capacity) {
        auto& s = *impl_;
        const uint32_t tiles = ((width + 15) / 16) * ((height + 15) / 16);
        if (auto r = s.binner.reserve(std::max(splats, 1u), tiles, capacity); !r)
            return r;
        if (splats > s.reserved_splats || capacity > s.reserved_capacity) {
            s.reserved_splats = std::max(s.reserved_splats, splats);
            s.reserved_capacity = std::max(s.reserved_capacity, capacity);
            s.trim = true;
        }
        if (width != s.width || height != s.height) {
            const core::GpuBackendScope scope(s.backend);
            const size_t pixels = size_t(width) * height;
            std::tie(s.color, s.depth, s.pick, s.rgba, s.linear_depth) =
                std::tuple_cat(carve_arena<5>({pixels * 8, pixels * 16, pixels * 4, pixels * 4, pixels * 4}));
            s.width = width;
            s.height = height;
            s.presented = false;
            s.trim = true;
        }
        return {};
    }

    lfs::Result<void> SplatRasterizer::rasterize(const Tensor& projected, const Tensor* gut, const uint32_t count,
                                            const SplatRasterMode mode, const SplatRasterParameters& parameters,
                                            const SplatRasterOverlay* overlay) {
        auto& s = *impl_;
        if (parameters.width != s.width || parameters.height != s.height || parameters.count != count ||
            parameters.mode != uint32_t(mode) || (mode == SplatRasterMode::Gut && !gut))
            return failure(std::format("Splat raster parameters disagree with the frame (extent={}x{} vs {}x{}, count={} vs {}, mode={} vs {}, gut={})",
                                       parameters.width, parameters.height, s.width, s.height, parameters.count, count,
                                       parameters.mode, uint32_t(mode), gut != nullptr));
        // The rasterizer owns source sorting and depth batches.
        if (((parameters.flags & kOverlay) != 0) != (overlay != nullptr) ||
            (overlay && (!overlay->parameters || !overlay->flags || (!overlay->selection && parameters.mask_limits[0]) ||
                         (!overlay->preview && parameters.mask_limits[1]))))
            return failure(std::format("Splat overlay inputs disagree with raster flags {:#x} (overlay={}, selection_count={}, preview_count={})",
                                       parameters.flags, overlay != nullptr, parameters.mask_limits[0], parameters.mask_limits[1]));
        if (parameters.flags & (kSourceSorted | kDepthBatches | kDepthPrefix))
            return failure(std::format("Splat raster flags {:#x} are chosen by the rasterizer", parameters.flags));
        const core::GpuBackendScope scope(s.backend);
        if (std::exchange(s.trim, false))
            core::gpu_trim_cached_memory(s.backend);
        auto frame = parameters;
        bool batches = false;
        if (s.previous_fits(count, frame.capacity)) {
            if (s.previous.required > count / 4)
                frame.flags |= kSourceSorted;
            batches = mode == SplatRasterMode::Gaussian && (frame.flags & kSingleSimd) && !(frame.flags & kSerialOnlyFlags) &&
                      s.previous.maximum_tile_instances > kParallelMinTileInstances && s.reserve_depth_batches(frame);
            if (batches) {
                frame.flags |= kDepthBatches;
                frame.mask_limits[2] = s.parallel_instances;
            }
            // Dense 3DGUT frames blend one subgroup per 8x4 pixels.
            if (mode == SplatRasterMode::Gut && s.previous.required > uint64_t(frame.tiles) * 512)
                frame.flags |= kSingleSimd;
        }
        s.upload(s.raster, frame);
        if (overlay && !overlay->selection_colors.empty())
            s.upload_bytes(s.selection_colors, overlay->selection_colors);
        if (auto r = s.binner.bin(projected, s.raster, count, frame.tiles, (frame.flags & kSourceSorted) != 0); !r)
            return r;
        s.mark("ranges");
        auto program = s.blend(mode, frame.flags);
        if (!program)
            return lfs::Result<void>::failure(std::move(program).error());
        auto prefix_program = batches ? s.blend(mode, frame.flags | kDepthPrefix) : program;
        if (!prefix_program)
            return lfs::Result<void>::failure(std::move(prefix_program).error());
        const auto& status = s.binner.status();
        const auto optional = [batches](const Tensor& tensor) { return batches ? &tensor : nullptr; };
        const std::array bindings{
            M::Binding{0, &projected}, M::Binding{8, &s.binner.indices()}, M::Binding{16, &s.binner.ranges()},
            M::Binding{24, &status, RW}, M::Binding{32, &s.raster}, M::Binding{40, overlay ? overlay->parameters : nullptr},
            M::Binding{48, overlay ? overlay->flags : nullptr}, M::Binding{56, overlay ? overlay->selection : nullptr},
            M::Binding{64, overlay ? overlay->preview : nullptr},
            M::Binding{72, overlay && !overlay->selection_colors.empty() ? &s.selection_colors : nullptr}, M::Binding{80, gut},
            M::Binding{88, nullptr}, M::Binding{96, optional(s.depth_jobs)}, M::Binding{104, optional(s.partial_color), RW},
            M::Binding{112, optional(s.partial_depth), RW}, M::Binding{120, optional(s.partial_pick), RW},
            M::Binding{128, &s.color, RW}, M::Binding{136, &s.depth, RW}, M::Binding{144, &s.pick, RW}};
        const auto dispatch = [&](const char* function, const uint32_t dispatch_flags, const M::Dispatch& shape) {
            const BlendParameters blend{.logical_count = count, .dispatch_flags = dispatch_flags};
            auto call = shape;
            call.function = function;
            call.arguments = {std::as_bytes(std::span(&blend, 1)), bindings};
            return (dispatch_flags & kDepthPrefix ? *prefix_program : *program)->dispatch(call);
        };
        if (batches) {
            // Prefix: short tiles in full and the first chunk of each long one;
            // then the remaining chunks in parallel; then their composition.
            if (auto r = s.binner.depth_batches(s.raster, frame.tiles, s.depth_jobs); !r)
                return r;
            s.mark("jobs");
            if (auto r = dispatch("tile_blend", kDepthPrefix, {.groups = {frame.tiles * 8, 1, 1}, .group = {32, 1, 1}}); !r)
                return r;
            s.mark("prefix");
            if (auto r = dispatch("tile_blend", 0, {.group = {32, 1, 1}, .indirect = &s.binner.dispatch_args(), .indirect_offset = 18}); !r)
                return r;
            s.mark("chunks");
            if (auto r = dispatch("tile_depth_compose", 0, {.groups = {frame.tiles * 8, 1, 1}, .group = {32, 1, 1}}); !r)
                return r;
        } else {
            const bool single = (frame.flags & kSingleSimd) != 0;
            if (auto r = dispatch("tile_blend", 0, {.groups = {frame.tiles * (single ? 8u : 4u), 1, 1}, .group = {single ? 32u : 64u, 1, 1}}); !r)
                return r;
        }
        s.mark(batches ? "compose" : "blend");
        s.read_status(count);
        return {};
    }

    lfs::Result<void> SplatRasterizer::present(const SplatPresentParameters& parameters) {
        auto& s = *impl_;
        if (s.width == 0)
            return failure("Splat present before any reservation");
        const core::GpuBackendScope scope(s.backend);
        if (!s.present) {
            auto loaded = M::load(splat_present_entries(), s.backend);
            if (!loaded)
                return lfs::Result<void>::failure(std::move(loaded).error());
            s.present = std::move(*loaded);
        }
        // An overflowing frame keeps the last image: present then copies the
        // outputs onto themselves.
        auto frame = parameters;
        frame.has_previous = s.presented ? 1u : 0u;
        frame.extent = {s.width, s.height, s.width, s.height};
        s.upload(s.present_parameters, frame);
        const PresentPointers pointers{};
        const std::array bindings{
            M::Binding{0, &s.color}, M::Binding{8, &s.depth}, M::Binding{16, &s.binner.status()},
            M::Binding{24, &s.present_parameters}, M::Binding{32, s.presented ? &s.rgba : nullptr},
            M::Binding{40, s.presented ? &s.linear_depth : nullptr}, M::Binding{48, &s.rgba, RW},
            M::Binding{56, &s.linear_depth, RW}};
        if (auto r = s.present->dispatch({.function = "present_viewer",
                                          .arguments = {std::as_bytes(std::span(&pointers, 1)), bindings},
                                          .groups = {(s.width + 15) / 16, (s.height + 15) / 16, 1},
                                          .group = {16, 16, 1}});
            !r)
            return r;
        s.presented = true;
        s.mark("present");
        return {};
    }

    void SplatRasterizer::set_stage_marker(std::function<void(const char*)> marker) {
        impl_->marker = marker;
        impl_->binner.set_stage_marker(std::move(marker));
    }

    const Tensor& SplatRasterizer::status() const { return impl_->binner.status(); }
    const Tensor& SplatRasterizer::color() const { return impl_->color; }
    const Tensor& SplatRasterizer::depth() const { return impl_->depth; }
    const Tensor& SplatRasterizer::pick() const { return impl_->pick; }
    const Tensor& SplatRasterizer::rgba() const { return impl_->rgba; }
    const Tensor& SplatRasterizer::linear_depth() const { return impl_->linear_depth; }
} // namespace lfs::rendering

