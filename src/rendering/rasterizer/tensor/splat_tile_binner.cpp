/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_tile_binner.hpp"

#include "splat_scan.hpp"
#include "splat_sort.hpp"
#include "splat_sort32.hpp"
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
            uint64_t source_keys = 0, source_ids = 0, group_offsets = 0, depth_jobs = 0;
        };
        static_assert(sizeof(ScanParameters) == 40 && sizeof(SortParameters) == 72 && sizeof(TileParameters) == 120);

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
        std::unique_ptr<M> scan, sort, sort32, tiles;
        uint32_t max_splats = 0, max_tiles = 0, capacity = 0;
        Tensor counts, offsets, group_offsets, status, dispatch_args, ranges;
        std::array<Tensor, 2> keys, indices;
        Tensor histogram, histogram_offsets, digit_offsets;
        struct ScanLevel {
            Tensor sums, offsets;
        };
        std::vector<ScanLevel> scan_levels;
        uint32_t sorted = 0;

        lfs::Result<void> run(M& module, const M::Dispatch& dispatch) { return module.dispatch(dispatch); }

        // Exclusive scan of `count` uint64 values, one level of block sums per call.
        lfs::Result<void> scan_counts(const Tensor& input, const Tensor& output, const uint32_t count, const size_t level) {
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

        lfs::Result<void> sort_pass(M& module, const uint32_t pass) {
            const uint32_t src = pass % 2, dst = 1 - src;
            const SortParameters parameters{.shift = pass * 8};
            const std::array bindings{M::Binding{0, &keys[src]}, M::Binding{8, &indices[src]},
                                      M::Binding{16, &keys[dst], RW}, M::Binding{24, &indices[dst], RW},
                                      M::Binding{32, &histogram, RW}, M::Binding{40, &histogram_offsets, RW},
                                      M::Binding{48, &digit_offsets, RW}, M::Binding{56, &status}};
            const M::Arguments arguments{bytes(parameters), bindings};
            if (auto r = run(module, {.function = "sort_histogram", .arguments = arguments, .group = {256, 1, 1},
                                     .indirect = &dispatch_args, .indirect_offset = kSortArgs});
                !r)
                return r;
            if (auto r = run(module, {.function = "sort_digit_scan", .arguments = arguments,
                                     .groups = {256, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (auto r = run(module, {.function = "sort_digit_offsets", .arguments = arguments, .group = {256, 1, 1}}); !r)
                return r;
            return run(module, {.function = "sort_scatter", .arguments = arguments, .group = {256, 1, 1},
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

    lfs::Result<void> SplatTileBinner::reserve(const uint32_t splats, const uint32_t tiles, const uint32_t capacity) {
        auto& s = *impl_;
        const core::GpuBackendScope scope(s.backend);
        if (splats > s.max_splats) {
            s.counts = Tensor::empty({splats}, Device::GPU, DataType::Int64);
            s.offsets = Tensor::empty({splats}, Device::GPU, DataType::Int64);
            s.group_offsets = Tensor::empty({ceil_div(splats, 256)}, Device::GPU, DataType::Int64);
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

    lfs::Result<void> SplatTileBinner::bin(const Tensor& splats, const Tensor& raster, const uint32_t count, const uint32_t tiles,
                                      const bool source_sorted) {
        auto& s = *impl_;
        if (count > s.max_splats || tiles > s.max_tiles || s.capacity == 0 || tiles == 0 || (source_sorted && count > s.capacity))
            return lfs::Result<void>::failure(make_error({.code = ErrorCode::InvalidArgument,
                                                     .domain = ErrorDomain::Rendering,
                                                     .detail = std::format("Tile binning exceeds its reservation (count={}/{}, tiles={}/{}, capacity={}, source_sorted={})",
                                                                           count, s.max_splats, tiles, s.max_tiles, s.capacity, source_sorted),
                                                     .detection = LFS_SOURCE_SITE_CURRENT()}));
        const core::GpuBackendScope scope(s.backend);
        if (source_sorted && !s.sort32)
            s.sort32 = load(splat_sort32_entries(), s.backend);
        M& tile_sort = source_sorted ? *s.sort32 : *s.sort;
        const TileParameters parameters{};
        const uint32_t groups = ceil_div(count, 256);
        // Source-sorted frames swap the count and offset buffers: the source
        // pass leaves per-source counts in `counts`, read by tile_counts.
        const Tensor& counts = source_sorted ? s.offsets : s.counts;
        const Tensor& offsets = source_sorted ? s.counts : s.offsets;
        if (source_sorted && count != 0) {
            // Depth keys of visible sources in keys[1], packed into keys[0],
            // sorted, then inverted to the source order in keys[1].
            const std::array bindings{M::Binding{0, &splats}, M::Binding{8, nullptr}, M::Binding{16, &s.offsets, RW},
                                      M::Binding{24, &s.status, RW}, M::Binding{32, &s.dispatch_args, RW},
                                      M::Binding{40, &s.keys[0], RW}, M::Binding{48, &s.indices[0], RW},
                                      M::Binding{56, nullptr}, M::Binding{64, &s.keys[1], RW}, M::Binding{72, &s.counts, RW},
                                      M::Binding{80, &raster}, M::Binding{88, &s.keys[1], RW},
                                      M::Binding{96, &s.indices[1], RW}, M::Binding{104, &s.group_offsets, RW},
                                      M::Binding{112, nullptr}};
            const M::Arguments arguments{bytes(parameters), bindings};
            if (auto r = s.run(*s.tiles, {.function = "source_key_groups", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (auto r = s.scan_counts(s.offsets, s.group_offsets, groups, 0); !r)
                return r;
            if (auto r = s.run(*s.tiles, {.function = "source_compact", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            for (uint32_t pass = 0; pass < 4; ++pass)
                if (auto r = s.sort_pass(tile_sort, pass); !r)
                    return r;
            if (auto r = s.run(*s.tiles, {.function = "source_permutation", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
        }
        const std::array bindings{M::Binding{0, &splats}, M::Binding{8, &counts, RW}, M::Binding{16, &offsets, RW},
                                  M::Binding{24, &s.status, RW}, M::Binding{32, &s.dispatch_args, RW},
                                  M::Binding{40, &s.keys[0], RW}, M::Binding{48, &s.indices[0], RW},
                                  M::Binding{56, &s.ranges, RW}, M::Binding{64, source_sorted ? &s.keys[1] : nullptr, RW},
                                  M::Binding{72, source_sorted ? &s.counts : nullptr, RW}, M::Binding{80, &raster},
                                  M::Binding{88, nullptr}, M::Binding{96, nullptr}, M::Binding{104, nullptr},
                                  M::Binding{112, nullptr}};
        const M::Arguments arguments{bytes(parameters), bindings};
        if (count != 0) {
            if (auto r = s.run(*s.tiles, {.function = "tile_counts", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            if (auto r = s.scan_counts(counts, offsets, count, 0); !r)
                return r;
        }
        if (auto r = s.run(*s.tiles, {.function = "tile_status", .arguments = arguments, .group = {1, 1, 1}}); !r)
            return r;
        // Depth bits first (4 passes, unless the sources are already in depth
        // order), then the bits of the tile index. The parity of the pass
        // count, and so the sorted buffer, is the same either way.
        const uint32_t tile_passes = (std::bit_width(tiles - 1) + 7) / 8;
        const uint32_t passes = (source_sorted ? 0 : 4) + tile_passes;
        if (count != 0) {
            if (auto r = s.run(*s.tiles, {.function = "tile_instances", .arguments = arguments,
                                          .groups = {groups, 1, 1}, .group = {256, 1, 1}});
                !r)
                return r;
            for (uint32_t pass = 0; pass < passes; ++pass)
                if (auto r = s.sort_pass(tile_sort, pass); !r)
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

    lfs::Result<void> SplatTileBinner::depth_batches(const Tensor& raster, const uint32_t tiles, Tensor& jobs) {
        auto& s = *impl_;
        const core::GpuBackendScope scope(s.backend);
        jobs.fill_(-1.f); // every unused slot reads as tile ~0
        const TileParameters parameters{};
        const std::array bindings{M::Binding{0, nullptr}, M::Binding{8, nullptr}, M::Binding{16, nullptr},
                                  M::Binding{24, &s.status}, M::Binding{32, nullptr}, M::Binding{40, nullptr},
                                  M::Binding{48, nullptr}, M::Binding{56, &s.ranges}, M::Binding{64, nullptr},
                                  M::Binding{72, nullptr}, M::Binding{80, &raster}, M::Binding{88, nullptr},
                                  M::Binding{96, nullptr}, M::Binding{104, nullptr}, M::Binding{112, &jobs, RW}};
        return s.run(*s.tiles, {.function = "tile_depth_batches", .arguments = {bytes(parameters), bindings},
                                .groups = {ceil_div(tiles, 256), 1, 1}, .group = {256, 1, 1}});
    }

    const Tensor& SplatTileBinner::dispatch_args() const { return impl_->dispatch_args; }
    const Tensor& SplatTileBinner::status() const { return impl_->status; }
    const Tensor& SplatTileBinner::keys() const { return impl_->keys[impl_->sorted]; }
    const Tensor& SplatTileBinner::indices() const { return impl_->indices[impl_->sorted]; }
    const Tensor& SplatTileBinner::ranges() const { return impl_->ranges; }
} // namespace lfs::rendering
