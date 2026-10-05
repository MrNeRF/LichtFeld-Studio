/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_rasterizer.hpp"

#include "core/gpu_kernel_module.hpp"
#include "core/tensor_readback.hpp"
#include "core/tensor_upload.hpp"
#include "splat_blend_discs.hpp"
#include "splat_blend_gs32.hpp"
#include "splat_blend_gs64.hpp"
#include "splat_blend_gs_fast.hpp"
#include "splat_blend_gut32.hpp"
#include "splat_blend_gut64.hpp"
#include "splat_blend_points.hpp"
#include "splat_present.hpp"
#include "splat_tile_binner.hpp"

#include <deque>
#include <format>
#include <map>
#include <span>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;
        constexpr uint32_t kSingleSimd = 128, kSourceSorted = 256;
        // Below this many sources the extra source sort is not worth its passes.
        constexpr uint32_t kSourceSortMinimum = 4096;

        struct BlendParameters {
            uint64_t pointers[19] = {};
            uint32_t logical_count = 0, padding = 0;
        };
        struct PresentPointers {
            uint64_t pointers[8] = {};
        };
        static_assert(sizeof(BlendParameters) == 160 && sizeof(PresentPointers) == 64);

        Result<void> failure(std::string detail) {
            return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                     .domain = ErrorDomain::Rendering,
                                                     .detail = std::move(detail),
                                                     .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    } // namespace

    struct SplatRasterizer::Impl {
        core::GpuBackend backend;
        SplatTileBinner binner;
        std::map<std::pair<uint32_t, bool>, std::unique_ptr<M>> blends;
        std::unique_ptr<M> fast_blend, present;
        Tensor raster, present_parameters, color, depth, pick;
        uint32_t width = 0, height = 0;
        std::deque<core::TensorUpload> uploads;
        // The last completed RasterStatus, read without waiting. It picks
        // source sorting for frames of the same source count.
        core::TensorReadback status_readback;
        uint32_t readback_count = 0, previous_count = 0;
        struct {
            uint64_t required = 0;
            uint32_t error = 1, rest[3] = {};
        } previous;
        static_assert(sizeof(previous) == SplatTileBinner::kRasterStatusBytes);

        bool source_sort(const uint32_t count, const uint32_t capacity) {
            if (status_readback.pending() && status_readback.poll(std::as_writable_bytes(std::span(&previous, 1))))
                previous_count = readback_count;
            return count >= kSourceSortMinimum && count <= capacity && previous_count == count && previous.error == 0 &&
                   previous.required > count / 4;
        }
        void read_status(const uint32_t count) {
            if (status_readback.pending())
                return;
            status_readback.enqueue(binner.status());
            readback_count = count;
        }

        explicit Impl(const core::GpuBackend b) : backend(b), binner(b) {}

        // Per-frame parameters ride the open batch, never waiting on the GPU.
        template <class T>
        void upload(Tensor& destination, const T& value) {
            std::erase_if(uploads, [](core::TensorUpload& slot) { return slot.poll(); });
            if (!destination.is_valid())
                destination = Tensor::empty({sizeof(T)}, Device::GPU, DataType::UInt8);
            uploads.emplace_back().enqueue_in_batch(destination, std::as_bytes(std::span(&value, 1)));
        }

        Result<M*> blend(const SplatRasterMode mode, const uint32_t flags) {
            const auto load = [&](std::span<const M::Entry> entries, std::unique_ptr<M>& slot) -> Result<M*> {
                if (!slot) {
                    auto loaded = M::load(entries, backend);
                    if (!loaded)
                        return Result<M*>(std::move(loaded).error());
                    slot = std::move(*loaded);
                }
                return slot.get();
            };
            if (mode == SplatRasterMode::Gaussian && (flags & ~kSourceSorted) == kSingleSimd)
                return load(splat_blend_gs_fast_entries(), fast_blend);
            const bool single = (flags & kSingleSimd) != 0;
            auto& slot = blends[{uint32_t(mode), single}];
            switch (mode) {
            case SplatRasterMode::Gaussian: return load(single ? splat_blend_gs32_entries() : splat_blend_gs64_entries(), slot);
            case SplatRasterMode::Gut: return load(single ? splat_blend_gut32_entries() : splat_blend_gut64_entries(), slot);
            case SplatRasterMode::Points: return load(splat_blend_points_entries(), slot);
            case SplatRasterMode::Discs: return load(splat_blend_discs_entries(), slot);
            }
            return Result<M*>(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Rendering,
                                          .detail = std::format("Unknown splat raster mode {}", uint32_t(mode)),
                                          .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    };

    SplatRasterizer::SplatRasterizer(const core::GpuBackend backend) : impl_(std::make_unique<Impl>(backend)) {}
    SplatRasterizer::~SplatRasterizer() = default;

    Result<void> SplatRasterizer::reserve(const uint32_t splats, const uint32_t width, const uint32_t height, const uint32_t capacity) {
        auto& s = *impl_;
        const uint32_t tiles = ((width + 15) / 16) * ((height + 15) / 16);
        if (auto r = s.binner.reserve(std::max(splats, 1u), tiles, capacity); !r)
            return r;
        if (width != s.width || height != s.height) {
            const core::GpuBackendScope scope(s.backend);
            const size_t pixels = size_t(width) * height;
            s.color = Tensor::empty({height, width, 4}, Device::GPU, DataType::Float16);
            s.depth = Tensor::empty({height, width, 4}, Device::GPU, DataType::Float32);
            s.pick = Tensor::empty({std::max<size_t>(pixels, 1)}, Device::GPU, DataType::UInt32);
            s.width = width;
            s.height = height;
        }
        return {};
    }

    Result<void> SplatRasterizer::rasterize(const Tensor& projected, const Tensor* gut, const uint32_t count,
                                            const SplatRasterMode mode, const SplatRasterParameters& parameters) {
        auto& s = *impl_;
        if (parameters.width != s.width || parameters.height != s.height || parameters.count != count ||
            parameters.mode != uint32_t(mode) || (mode == SplatRasterMode::Gut && !gut))
            return failure(std::format("Splat raster parameters disagree with the frame (extent={}x{} vs {}x{}, count={} vs {}, mode={} vs {}, gut={})",
                                       parameters.width, parameters.height, s.width, s.height, parameters.count, count,
                                       parameters.mode, uint32_t(mode), gut != nullptr));
        // The rasterizer owns source sorting; depth batches are not ported yet.
        if (parameters.flags & (kSourceSorted | 512u | 1024u))
            return failure(std::format("Unsupported splat blend flags {:#x}", parameters.flags));
        const core::GpuBackendScope scope(s.backend);
        auto frame = parameters;
        const bool source_sorted = s.source_sort(count, frame.capacity);
        if (source_sorted)
            frame.flags |= kSourceSorted;
        s.upload(s.raster, frame);
        if (auto r = s.binner.bin(projected, s.raster, count, frame.tiles, source_sorted); !r)
            return r;
        auto program = s.blend(mode, frame.flags);
        if (!program)
            return Result<void>::failure(std::move(program).error());
        const BlendParameters blend{.logical_count = count};
        const auto& status = s.binner.status();
        const std::array bindings{
            M::Binding{0, &projected}, M::Binding{8, &s.binner.indices()}, M::Binding{16, &s.binner.ranges()},
            M::Binding{24, &status, RW}, M::Binding{32, &s.raster}, M::Binding{40, nullptr}, M::Binding{48, nullptr},
            M::Binding{56, nullptr}, M::Binding{64, nullptr}, M::Binding{72, nullptr}, M::Binding{80, gut},
            M::Binding{88, nullptr}, M::Binding{96, nullptr}, M::Binding{104, nullptr}, M::Binding{112, nullptr},
            M::Binding{120, nullptr}, M::Binding{128, &s.color, RW}, M::Binding{136, &s.depth, RW}, M::Binding{144, &s.pick, RW}};
        const bool single = (frame.flags & kSingleSimd) != 0;
        if (auto r = (*program)->dispatch({.function = "tile_blend",
                                           .arguments = {std::as_bytes(std::span(&blend, 1)), bindings},
                                           .groups = {frame.tiles * (single ? 8u : 4u), 1, 1},
                                           .group = {single ? 32u : 64u, 1, 1}});
            !r)
            return r;
        s.read_status(count);
        return {};
    }

    Result<void> SplatRasterizer::present(const SplatPresentParameters& parameters, Tensor& rgba, Tensor& linear_depth,
                                          const Tensor* previous_rgba, const Tensor* previous_depth) {
        auto& s = *impl_;
        if (parameters.extent[0] != s.width || parameters.extent[1] != s.height ||
            rgba.numel() != size_t(s.width) * s.height * 4 || linear_depth.numel() != size_t(s.width) * s.height ||
            (parameters.has_previous && (!previous_rgba || !previous_depth)))
            return failure(std::format("Splat present outputs disagree with the frame (extent={}x{} vs {}x{}, rgba={}, depth={}, previous={})",
                                       parameters.extent[0], parameters.extent[1], s.width, s.height, rgba.numel(), linear_depth.numel(),
                                       parameters.has_previous));
        const core::GpuBackendScope scope(s.backend);
        if (!s.present) {
            auto loaded = M::load(splat_present_entries(), s.backend);
            if (!loaded)
                return Result<void>::failure(std::move(loaded).error());
            s.present = std::move(*loaded);
        }
        s.upload(s.present_parameters, parameters);
        const PresentPointers pointers{};
        const std::array bindings{
            M::Binding{0, &s.color}, M::Binding{8, &s.depth}, M::Binding{16, &s.binner.status()},
            M::Binding{24, &s.present_parameters}, M::Binding{32, parameters.has_previous ? previous_rgba : nullptr},
            M::Binding{40, parameters.has_previous ? previous_depth : nullptr}, M::Binding{48, &rgba, RW},
            M::Binding{56, &linear_depth, RW}};
        return s.present->dispatch({.function = "present_viewer",
                                    .arguments = {std::as_bytes(std::span(&pointers, 1)), bindings},
                                    .groups = {(s.width + 15) / 16, (s.height + 15) / 16, 1},
                                    .group = {16, 16, 1}});
    }

    const Tensor& SplatRasterizer::status() const { return impl_->binner.status(); }
    const Tensor& SplatRasterizer::color() const { return impl_->color; }
    const Tensor& SplatRasterizer::depth() const { return impl_->depth; }
    const Tensor& SplatRasterizer::pick() const { return impl_->pick; }
} // namespace lfs::rendering
