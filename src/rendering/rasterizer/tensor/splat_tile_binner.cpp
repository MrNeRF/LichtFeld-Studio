/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_tile_binner.hpp"

#include "splat_scan.hpp"
#include "splat_sort.hpp"
#include "splat_tiles.hpp"

#include <array>
#include <bit>
#include <format>
#include <span>
#include <utility>

namespace lfs::rendering {
    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using M = core::GpuKernelModule;
        constexpr auto RW = M::Access::ReadWrite;

        // Parameter blocks, in the field order of the Slang modules.
        struct ScanParameters {
            uint64_t input = 0, output = 0, sums = 0, offsets = 0;
            uint32_t count = 0, primitive_counts = 0;
        };
        struct SortParameters {
            uint64_t keys_in = 0, values_in = 0, keys_out = 0, values_out = 0;
            uint64_t histogram = 0, histogram_offsets = 0, digit_offsets = 0, status = 0;
            uint32_t shift = 0, padding = 0;
        };
        struct TileParameters {
            uint64_t splats = 0, counts = 0, offsets = 0, status = 0, dispatch_args = 0, keys = 0;
            uint64_t indices = 0, ranges = 0, source_order = 0, source_counts = 0, raster = 0;
        };
        static_assert(sizeof(ScanParameters) == 40 && sizeof(SortParameters) == 72 && sizeof(TileParameters) == 88);

        constexpr uint32_t kSortArgs = 0, kRangeArgs = 3, kDispatchArgs = 21;

        uint32_t ceil_div(const uint64_t value, const uint32_t divisor) {
            return static_cast<uint32_t>((value + divisor - 1) / divisor);
        }

        template <class P>
        std::span<const std::byte> bytes(const P& parameters) { return std::as_bytes(std::span(&parameters, 1)); }

        std::unique_ptr<M> load(std::span<const M::Entry> entries, const core::GpuBackend backend) {
            auto loaded = M::load(entries, backend);
            if (!loaded)
                throw std::runtime_error(format_for_developer(loaded.error()));
            return std::move(*loaded);
        }
    } // namespace

    struct SplatTileBinner::Impl {
        core::GpuBackend backend;
        std::unique_ptr<M> scan, sort, tiles;
        uint32_t max_splats = 0, max_tiles = 0, capacity = 0;
        Tensor counts, offsets, status, dispatch_args, ranges;
        std::array<Tensor, 2> keys, indices;
        Tensor histogram, histogram_offsets, digit_offsets;
        struct ScanLevel {
            Tensor sums, offsets;
        };
        std::vector<ScanLevel> scan_levels;
        uint32_t sorted = 0;

        Result<void> run(M& module, const M::Dispatch& dispatch) { return module.dispatch(dispatch); }

        // Exclusive scan of `count` uint64 values, one level of block sums per call.
        Result<void> scan_counts(const Tensor& input, const Tensor& output, const uint32_t count, const size_t level) {
            if (count == 0)
                return {};
            auto& storage = scan_levels.at(level);
            const uint32_t groups = ceil_div(count, 256);
            const ScanParameters blocks{.count = count, .primitive_counts = level == 0};
            const std::array block_bindings{M::Binding{0, &input}, M::Binding{8, &output, RW},
                                            M::Binding{16, &storage.sums, RW}, M::Binding{24, nullptr}};
            if (auto r = run(*scan, {.function = "scan_blocks", .arguments = {bytes(blocks), block_bindings},
                                     .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (groups <= 1)
                return {};
            if (auto r = scan_counts(storage.sums, storage.offsets, groups, level + 1); !r)
                return r;
            const ScanParameters add{.count = count};
            const std::array add_bindings{M::Binding{0, nullptr}, M::Binding{8, &output, RW},
                                          M::Binding{16, nullptr}, M::Binding{24, &storage.offsets}};
            return run(*scan, {.function = "scan_add", .arguments = {bytes(add), add_bindings},
                               .groups = {groups, 1, 1}, .group = {256, 1, 1}});
        }

        Result<void> sort_pass(const uint32_t pass) {
            const uint32_t src = pass % 2, dst = 1 - src;
            const SortParameters parameters{.shift = pass * 8};
            const std::array bindings{M::Binding{0, &keys[src]}, M::Binding{8, &indices[src]},
                                      M::Binding{16, &keys[dst], RW}, M::Binding{24, &indices[dst], RW},
                                      M::Binding{32, &histogram, RW}, M::Binding{40, &histogram_offsets, RW},
                                      M::Binding{48, &digit_offsets, RW}, M::Binding{56, &status}};
            const M::Arguments arguments{bytes(parameters), bindings};
            if (auto r = run(*sort, {.function = "sort_histogram", .arguments = arguments, .group = {256, 1, 1},
                                     .indirect = &dispatch_args, .indirect_offset = kSortArgs});
                !r)
                return r;
            if (auto r = run(*sort, {.function = "sort_digit_scan", .arguments = arguments,
                                     .groups = {256, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (auto r = run(*sort, {.function = "sort_digit_offsets", .arguments = arguments, .group = {256, 1, 1}}); !r)
                return r;
            return run(*sort, {.function = "sort_scatter", .arguments = arguments, .group = {256, 1, 1},
                               .indirect = &dispatch_args, .indirect_offset = kSortArgs});
        }
    };

    SplatTileBinner::SplatTileBinner(const core::GpuBackend backend) : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        impl_->scan = load(splat_scan_entries(), backend);
        impl_->sort = load(splat_sort_entries(), backend);
        impl_->tiles = load(splat_tiles_entries(), backend);
        const core::GpuBackendScope scope(backend);
        impl_->status = Tensor::zeros({kRasterStatusBytes / 4}, Device::GPU, DataType::UInt32);
        impl_->dispatch_args = Tensor::zeros({kDispatchArgs}, Device::GPU, DataType::UInt32);
        impl_->digit_offsets = Tensor::zeros({257}, Device::GPU, DataType::UInt32);
    }

    SplatTileBinner::~SplatTileBinner() = default;

    Result<void> SplatTileBinner::reserve(const uint32_t splats, const uint32_t tiles, const uint32_t capacity) {
        auto& s = *impl_;
        const core::GpuBackendScope scope(s.backend);
        if (splats > s.max_splats) {
            s.counts = Tensor::empty({splats}, Device::GPU, DataType::Int64);
            s.offsets = Tensor::empty({splats}, Device::GPU, DataType::Int64);
            s.scan_levels.clear();
            for (uint64_t n = splats; n > 1;) {
                const uint32_t groups = ceil_div(n, 256);
                s.scan_levels.push_back({Tensor::empty({groups}, Device::GPU, DataType::Int64),
                                         Tensor::empty({groups}, Device::GPU, DataType::Int64)});
                n = groups;
            }
            if (s.scan_levels.empty())
                s.scan_levels.push_back({Tensor::empty({1}, Device::GPU, DataType::Int64),
                                         Tensor::empty({1}, Device::GPU, DataType::Int64)});
            s.max_splats = splats;
        }
        if (tiles > s.max_tiles) {
            s.ranges = Tensor::empty({size_t(tiles) * 2}, Device::GPU, DataType::UInt32);
            s.max_tiles = tiles;
        }
        if (capacity > s.capacity) {
            for (int n = 0; n < 2; ++n) {
                s.keys[n] = Tensor::empty({capacity}, Device::GPU, DataType::Int64);
                s.indices[n] = Tensor::empty({capacity}, Device::GPU, DataType::UInt32);
            }
            const size_t histogram = size_t(std::max(1u, ceil_div(capacity, 2048))) * 256;
            s.histogram = Tensor::empty({histogram}, Device::GPU, DataType::UInt32);
            s.histogram_offsets = Tensor::empty({histogram}, Device::GPU, DataType::UInt32);
            s.capacity = capacity;
        }
        return {};
    }

    Result<void> SplatTileBinner::bin(const Tensor& splats, const Tensor& raster, const uint32_t count, const uint32_t tiles) {
        auto& s = *impl_;
        if (count > s.max_splats || tiles > s.max_tiles || s.capacity == 0 || tiles == 0)
            return Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                     .domain = ErrorDomain::Rendering,
                                                     .detail = std::format("Tile binning exceeds its reservation (count={}/{}, tiles={}/{}, capacity={})",
                                                                           count, s.max_splats, tiles, s.max_tiles, s.capacity),
                                                     .detection = LFS_SOURCE_SITE_CURRENT()}));
        const core::GpuBackendScope scope(s.backend);
        const TileParameters parameters{};
        const std::array bindings{M::Binding{0, &splats}, M::Binding{8, &s.counts, RW}, M::Binding{16, &s.offsets, RW},
                                  M::Binding{24, &s.status, RW}, M::Binding{32, &s.dispatch_args, RW},
                                  M::Binding{40, &s.keys[0], RW}, M::Binding{48, &s.indices[0], RW},
                                  M::Binding{56, &s.ranges, RW}, M::Binding{64, nullptr}, M::Binding{72, nullptr},
                                  M::Binding{80, &raster}};
        const M::Arguments arguments{bytes(parameters), bindings};
        const uint32_t groups = ceil_div(count, 256);
        if (count != 0) {
            if (auto r = s.run(*s.tiles, {.function = "tile_counts", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (auto r = s.scan_counts(s.counts, s.offsets, count, 0); !r)
                return r;
        }
        if (auto r = s.run(*s.tiles, {.function = "tile_status", .arguments = arguments, .group = {1, 1, 1}}); !r)
            return r;
        // Depth bits first (4 passes), then the bits of the tile index.
        const uint32_t passes = 4 + (std::bit_width(tiles - 1) + 7) / 8;
        if (count != 0) {
            if (auto r = s.run(*s.tiles, {.function = "tile_instances", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            for (uint32_t pass = 0; pass < passes; ++pass)
                if (auto r = s.sort_pass(pass); !r)
                    return r;
        }
        s.sorted = passes % 2;
        s.ranges.zero_();
        if (count != 0) {
            auto sorted_bindings = bindings;
            sorted_bindings[5].tensor = &s.keys[s.sorted];
            sorted_bindings[6].tensor = &s.indices[s.sorted];
            return s.run(*s.tiles, {.function = "tile_ranges", .arguments = {bytes(parameters), sorted_bindings},
                                    .group = {256, 1, 1}, .indirect = &s.dispatch_args, .indirect_offset = kRangeArgs});
        }
        return {};
    }

    const Tensor& SplatTileBinner::status() const { return impl_->status; }
    const Tensor& SplatTileBinner::keys() const { return impl_->keys[impl_->sorted]; }
    const Tensor& SplatTileBinner::indices() const { return impl_->indices[impl_->sorted]; }
    const Tensor& SplatTileBinner::ranges() const { return impl_->ranges; }
} // namespace lfs::rendering
