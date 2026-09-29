/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal FastRasterOps; rasterization/fast_rasterizer_cuda.cpp and the fastgs
// kernels are the reference behaviour. Kernels: kernels/metal/fast_raster.metal
// (forward, binning, sort) and fast_backward.metal (blend backward, fused Adam).

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include "core/memory_pressure.hpp"
#include "core/tensor_upload.hpp"
#include "lfs/training/vram_ledger.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <format>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using lfs::gpu_ops::Tensor;
        using Code = lfs::gpu_ops::RasterResult::Code;
        namespace mk = metal;

        constexpr uint32_t kTile = 16;
        constexpr uint32_t kTilePixels = kTile * kTile;
        constexpr uint32_t kBlendThreads = 128;
        // kFastShParts and kFastShSlotsPerThread in fast_backward.metal:
        // fast_backward_sh spreads a primitive's SH rest slots over kShParts threads.
        constexpr uint32_t kShParts = 4;
        constexpr uint32_t kShSlotsPerThread = 3;
        // kFastBwdThreads in fast_backward.metal.
        constexpr uint32_t kBackwardThreads = 64;
        constexpr uint32_t kGradStride = 12;
        constexpr uint32_t kScanBlock = 2048;
        constexpr uint32_t kSortBlock = 2048;
        constexpr float kNearPlane = 0.01f;
        constexpr float kFarPlane = 1e10f;

        // Function constant indices shared with the kernels.
        constexpr uint32_t kShBasesConstant = 60;
        constexpr uint32_t kRenderDepthConstant = 61;
        constexpr uint32_t kRenderNormalConstant = 62;
        constexpr uint32_t kDensificationConstant = 63;
        constexpr uint32_t kNormalChannelConstant = 64;
        constexpr uint32_t kMipFilterConstant = 65;
        constexpr uint32_t kShLayoutRestConstant = 66;
        constexpr uint32_t kShStorageConstant = 67;
        constexpr uint32_t kDepthGradConstant = 68;
        constexpr uint32_t kEdgeWeightConstant = 69;

        // Tensors captured by forward for the backward of the same frame.
        struct Frame {
            bool live = false;
            Tensor means, scales, rotations, opacities, sh0, shN, sh_bounds, view, camera, bg_color, bg_image;
            uint32_t n = 0, width = 0, height = 0, grid_w = 0, grid_h = 0;
            uint32_t n_instances = 0, sorted = 0;
            uint32_t sh_bases = 1, sh_storage = 0, sh_layout_rest = 0;
            float fx = 0.f, fy = 0.f, cx = 0.f, cy = 0.f;
            bool mip_filter = false, normals = false;
        };

        // Buffers are reused across frames and grow with headroom.
        struct MetalFastState : lfs::gpu_ops::BackendState {
            Tensor image, alpha, depth, normal;
            int width = -1, height = -1;
            Tensor mean_box, conic_opacity, color_depth, tile_info, normals, n_touched, offsets;
            Tensor grads, normal_grads;
            Tensor ranges, final_transmittance, n_contrib;
            Tensor block_sums, counts, histogram;
            std::array<Tensor, 2> keys, values;
            // The host's copy of the instance count, and the instance buffers'
            // size, which lets a frame encode its raster tail before the count arrives.
            Tensor count_copy;
            uint32_t instance_capacity = 0;
            Frame frame;
            std::string message;
        };

        MetalFastState& state_of(lfs::gpu_ops::FastSaved& saved) {
            if (!saved.backend)
                throw std::logic_error("Metal fast raster op called without a created state");
            return static_cast<MetalFastState&>(*saved.backend);
        }

        const MetalFastState* state_of(const lfs::gpu_ops::FastSaved& saved) {
            return static_cast<const MetalFastState*>(saved.backend.get());
        }

        void reserve(Tensor& buffer, const size_t count, const DataType dtype) {
            if (buffer.is_valid() && buffer.dtype() == dtype && buffer.numel() >= count)
                return;
            buffer = Tensor();
            buffer = Tensor::empty({std::max<size_t>(count + count / 4, 1024)}, Device::GPU, dtype);
        }

        uint32_t div_up(const uint64_t value, const uint64_t divisor) {
            return static_cast<uint32_t>((value + divisor - 1) / divisor);
        }

        bool present(const Tensor& tensor) { return tensor.is_valid() && tensor.numel() > 0; }

        uint64_t address_if(const Tensor& tensor) { return present(tensor) ? mk::address(tensor) : 0; }

        template <class Params>
        void launch(const std::string_view function, const Params& params, const std::span<const Tensor* const> uses,
                    const uint32_t groups_x, const uint32_t groups_y, const uint32_t width,
                    const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
            static_assert(sizeof(Params) <= 1024, "Metal parameter blocks are limited to 1 KiB");
            mk::kernels().launch({.function = function,
                                  .params = std::as_bytes(std::span(&params, 1)),
                                  .uses = uses,
                                  .groups = {groups_x, groups_y, 1},
                                  .group = {width, 1, 1},
                                  .constants = std::span(constants.begin(), constants.size())});
        }

        template <class Params>
        void launch(const std::string_view function, const Params& params,
                    const std::initializer_list<const Tensor*> uses, const uint32_t groups_x, const uint32_t groups_y,
                    const uint32_t width, const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
            launch(function, params, std::span(uses.begin(), uses.size()), groups_x, groups_y, width, constants);
        }

        void require_contiguous(const Tensor& tensor, const char* const name) {
            if (present(tensor) && !tensor.is_contiguous())
                throw std::invalid_argument(std::format("Metal fast raster needs a contiguous {}", name));
        }

        uint32_t sh_bases_class(const uint32_t bases) { return bases <= 1 ? 1 : bases <= 4 ? 4
                                                                            : bases <= 9   ? 9
                                                                                           : 16; }

        // Tile bits plus depth bits of the packed instance key (packed_instance_depth_bits).
        uint32_t tile_bits(const uint32_t n_tiles) { return n_tiles <= 1 ? 0 : std::bit_width(n_tiles - 1); }
        uint32_t depth_bits(const uint32_t n_tiles) {
            return static_cast<uint32_t>(std::clamp(32 - static_cast<int>(tile_bits(n_tiles)), 0, 23));
        }

        void fill(Tensor& buffer, const size_t count, const uint32_t value) {
            struct {
                uint64_t data;
                uint32_t count, value;
            } params{mk::address(buffer), static_cast<uint32_t>(count), value};
            launch("fast_fill", params, {&buffer}, core::GpuKernelModule::groups_for(count, 256), 1, 256);
        }

        struct ScanParams {
            uint64_t input, output, block_sums, total;
            uint32_t n, n_blocks;
        };

        // Exclusive scan of n uint32 values; `total` receives the 64-bit sum as two words.
        void exclusive_scan(MetalFastState& s, const Tensor& input, Tensor& output, const uint32_t n,
                            Tensor* const total) {
            const uint32_t blocks = div_up(n, kScanBlock);
            reserve(s.block_sums, blocks, DataType::UInt32);
            const ScanParams params{mk::address(input), mk::address(output), mk::address(s.block_sums),
                                    total != nullptr ? mk::address(*total) : 0, n, blocks};
            launch("fast_scan_reduce", params, {&input, &s.block_sums}, blocks, 1, 256);
            launch("fast_scan_top", params, {&s.block_sums, total}, 1, 1, 1024);
            launch("fast_scan_down", params, {&input, &output, &s.block_sums}, blocks, 1, 256);
        }

        struct SortParams {
            uint64_t keys_in, values_in, keys_out, values_out, histogram, counts;
            uint32_t n, n_blocks, shift, unused;
        };

        // Stable sort of the key/value pairs on bits [0, end_bit): as many as
        // s.counts holds, dispatched for `capacity`. Returns the buffer index
        // holding the result.
        uint32_t sort_instances(MetalFastState& s, const uint32_t capacity, const uint32_t end_bit) {
            const uint32_t blocks = div_up(capacity, kSortBlock);
            reserve(s.histogram, size_t{256} * blocks, DataType::UInt32);
            uint32_t current = 0;
            for (uint32_t shift = 0; shift < end_bit; shift += 8) {
                auto& keys_in = s.keys[current];
                auto& values_in = s.values[current];
                auto& keys_out = s.keys[current ^ 1];
                auto& values_out = s.values[current ^ 1];
                const SortParams params{mk::address(keys_in), mk::address(values_in), mk::address(keys_out),
                                        mk::address(values_out), mk::address(s.histogram), mk::address(s.counts),
                                        capacity, blocks, shift, 0};
                launch("fast_sort_histogram", params, {&keys_in, &s.histogram, &s.counts}, blocks, 1, 256);
                exclusive_scan(s, s.histogram, s.histogram, 256 * blocks, nullptr);
                launch("fast_sort_scatter", params,
                       {&keys_in, &values_in, &keys_out, &values_out, &s.histogram, &s.counts}, blocks, 1, 256);
                current ^= 1;
            }
            return current;
        }

        struct PreprocessParams {
            uint64_t means, scales, rotations, opacities, sh0, shN, sh_bounds, view, camera;
            uint64_t mean_box, conic_opacity, color_depth, tile_info, n_touched, normals, max_screen_share;
            uint32_t n, grid_w, grid_h, depth_bits;
            uint32_t mip_filter, unused0, unused1, unused2;
            float fx, fy, cx, cy;
            float clip_left, clip_right, clip_top, clip_bottom;
            float near_plane, far_plane;
        };

        struct InstanceParams {
            uint64_t mean_box, conic_opacity, tile_info, n_touched, offsets, keys, values;
            uint32_t n, grid_w, depth_bits, capacity;
        };

        struct RangeParams {
            uint64_t keys, ranges, counts;
            uint32_t capacity, n_tiles, depth_bits, unused;
        };

        struct BlendParams {
            uint64_t ranges, values, mean_box, conic_opacity, color_depth, normals;
            uint64_t image, alpha, depth, normal, n_contrib, final_transmittance, bg_color, bg_image;
            uint32_t width, height, grid_w, unused;
        };

        struct ClipBounds {
            float left, right, top, bottom;
        };

        // ewa_clip_bounds: 15% guard band around the viewport in normalized coordinates.
        ClipBounds clip_bounds(const Frame& f) {
            const float w = static_cast<float>(f.width);
            const float h = static_cast<float>(f.height);
            return {(-0.15f * w - f.cx) / f.fx, (1.15f * w - f.cx) / f.fx, (-0.15f * h - f.cy) / f.fy,
                    (1.15f * h - f.cy) / f.fy};
        }

        lfs::gpu_ops::RasterResult fail(MetalFastState& s, const Code code, std::string message) {
            s.message = std::move(message);
            return {.code = code, .has_work = false, .message = s.message};
        }

        void ensure_output(Tensor& tensor, const size_t channels, const int height, const int width) {
            const core::TensorShape shape({channels, static_cast<size_t>(height), static_cast<size_t>(width)});
            if (!tensor.is_valid() || tensor.shape() != shape)
                tensor = Tensor::empty(shape, Device::GPU, DataType::Float32);
        }

        lfs::gpu_ops::RasterResult forward(lfs::gpu_ops::FastSaved& saved, const lfs::gpu_ops::SplatInputs& splats,
                                           const Tensor& view, const Tensor& camera_position, const Tensor& bg_color,
                                           const Tensor& background_image, const lfs::gpu_ops::FastParams& params,
                                           const lfs::gpu_ops::RenderOutputs& outputs, Tensor& max_screen_share) {
            auto& s = state_of(saved);
            s.frame = {};
            if (!splats.means.is_valid() || splats.means.ndim() != 2 || splats.means.shape()[0] == 0)
                return fail(s, Code::Failed, "n_primitives is 0 - model has no gaussians");
            if (splats.means.shape()[0] > static_cast<size_t>(std::numeric_limits<int>::max()))
                return fail(s, Code::Failed,
                            std::format("n_primitives {} exceeds int range", splats.means.shape()[0]));
            const auto n = static_cast<uint32_t>(splats.means.shape()[0]);
            const int width = params.tile_w > 0 ? params.tile_w : params.full_image.w;
            const int height = params.tile_h > 0 ? params.tile_h : params.full_image.h;
            const uint32_t active_bases = params.sh.active_bases;
            const uint32_t layout_bases = params.sh.layout_bases;
            if (width <= 0 || height <= 0 || active_bases == 0 || active_bases > 16 || layout_bases == 0 ||
                layout_bases > 16 || layout_bases < active_bases)
                return fail(s, Code::Failed,
                            std::format("FastGS forward preflight: invalid dimensions (n_primitives={}, "
                                        "active_sh_bases={}, sh_layout_bases={}, width={}, height={})",
                                        n, active_bases, layout_bases, width, height));
            const bool q16 = params.sh.storage == lfs::gpu_ops::ShStorage::Q16;
            if (q16 && active_bases > 1 && !present(splats.sh_value_bounds))
                throw std::invalid_argument("Metal fast raster: Q16 SH storage needs value bounds");
            for (const auto& [tensor, name] :
                 {std::pair{&splats.means, "means"}, {&splats.raw_scales, "scales"}, {&splats.raw_rotations, "rotations"}, {&splats.raw_opacities, "opacities"}, {&splats.sh0, "sh0"}, {&splats.shN, "shN"}, {&view, "view"}, {&camera_position, "camera position"}})
                require_contiguous(*tensor, name);
            // Kernels index these by primitive; a short tensor would be read out of bounds.
            const size_t padded = (size_t{n} + 31) / 32 * 32;
            const size_t rest_slots = std::min<size_t>(12, ((layout_bases - 1) * 3 + 3) / 4);
            const size_t shN_bytes = active_bases <= 1 ? 0
                                     : q16             ? padded * (layout_bases - 1) * 3 * 2
                                                       : padded * rest_slots * 4 * (params.sh.storage == lfs::gpu_ops::ShStorage::Float32 ? 4 : 2);
            for (const auto& [tensor, bytes, name] :
                 {std::tuple{&splats.raw_scales, size_t{n} * 12, "scales"},
                  {&splats.raw_rotations, size_t{n} * 16, "rotations"},
                  {&splats.raw_opacities, size_t{n} * 4, "opacities"},
                  {&splats.sh0, size_t{n} * 12, "sh0"},
                  {&splats.shN, shN_bytes, "shN"},
                  {&view, size_t{64}, "view"},
                  {&camera_position, size_t{12}, "camera position"}})
                if (bytes > 0 && (!tensor->is_valid() || tensor->bytes() < bytes))
                    throw std::invalid_argument(std::format("Metal fast raster: {} holds {} bytes, {} primitives need {}",
                                                            name, tensor->is_valid() ? tensor->bytes() : 0, n, bytes));

            Frame f;
            f.n = n;
            f.width = static_cast<uint32_t>(width);
            f.height = static_cast<uint32_t>(height);
            f.grid_w = div_up(f.width, kTile);
            f.grid_h = div_up(f.height, kTile);
            f.fx = params.intrinsics.fx;
            f.fy = params.intrinsics.fy;
            f.cx = params.intrinsics.cx - static_cast<float>(params.tile_x);
            f.cy = params.intrinsics.cy - static_cast<float>(params.tile_y);
            f.sh_bases = sh_bases_class(active_bases);
            f.sh_storage = static_cast<uint32_t>(params.sh.storage);
            f.sh_layout_rest = layout_bases - 1;
            f.mip_filter = params.mip_filter;
            f.normals = params.render_normal;
            const uint64_t n_tiles_64 = uint64_t{f.grid_w} * f.grid_h;
            if (n_tiles_64 > std::numeric_limits<int>::max())
                return fail(s, Code::Failed, std::format("n_tiles {} exceeds int range", n_tiles_64));
            const auto n_tiles = static_cast<uint32_t>(n_tiles_64);
            const size_t pixels = size_t{f.width} * f.height;
            const uint32_t key_depth_bits = depth_bits(n_tiles);
            const bool has_bg_image = present(background_image);
            if (has_bg_image && background_image.numel() != 3 * pixels)
                throw std::invalid_argument(std::format("background image must be [3, {}, {}], got {}", height, width,
                                                        background_image.shape().str()));
            const Tensor bg_image = has_bg_image ? background_image.contiguous() : Tensor();

            try {
                if (s.width != width || s.height != height) {
                    s.image = Tensor();
                    s.alpha = Tensor();
                    s.depth = Tensor();
                    s.normal = Tensor();
                    s.width = width;
                    s.height = height;
                }
                ensure_output(s.image, 3, height, width);
                ensure_output(s.alpha, 1, height, width);
                if (params.render_depth)
                    ensure_output(s.depth, 1, height, width);
                else
                    s.depth = Tensor();
                if (params.render_normal)
                    ensure_output(s.normal, 3, height, width);

                reserve(s.mean_box, size_t{n} * 4, DataType::Float32);
                reserve(s.conic_opacity, size_t{n} * 4, DataType::Float32);
                reserve(s.color_depth, size_t{n} * 4, DataType::Float32);
                reserve(s.tile_info, size_t{n} * 4, DataType::UInt32);
                reserve(s.n_touched, n, DataType::UInt32);
                reserve(s.offsets, n, DataType::UInt32);
                if (params.render_normal)
                    reserve(s.normals, size_t{n} * 4, DataType::Float32);
                reserve(s.ranges, size_t{n_tiles} * 2, DataType::UInt32);
                reserve(s.final_transmittance, size_t{n_tiles} * kTilePixels, DataType::Float32);
                reserve(s.n_contrib, pixels, DataType::UInt32);
                reserve(s.counts, 2, DataType::UInt32);

                const bool share = max_screen_share.is_valid() && max_screen_share.ndim() == 1 &&
                                   max_screen_share.numel() >= n;
                const Tensor none;
                const Tensor& share_use = share ? max_screen_share : none;
                const ClipBounds clip = clip_bounds(f);
                const PreprocessParams pre{
                    .means = mk::address(splats.means),
                    .scales = mk::address(splats.raw_scales),
                    .rotations = mk::address(splats.raw_rotations),
                    .opacities = mk::address(splats.raw_opacities),
                    .sh0 = mk::address(splats.sh0),
                    .shN = active_bases > 1 ? address_if(splats.shN) : 0,
                    .sh_bounds = q16 ? address_if(splats.sh_value_bounds) : 0,
                    .view = mk::address(view),
                    .camera = mk::address(camera_position),
                    .mean_box = mk::address(s.mean_box),
                    .conic_opacity = mk::address(s.conic_opacity),
                    .color_depth = mk::address(s.color_depth),
                    .tile_info = mk::address(s.tile_info),
                    .n_touched = mk::address(s.n_touched),
                    .normals = params.render_normal ? mk::address(s.normals) : 0,
                    .max_screen_share = share ? mk::address(max_screen_share) : 0,
                    .n = n,
                    .grid_w = f.grid_w,
                    .grid_h = f.grid_h,
                    .depth_bits = key_depth_bits,
                    .mip_filter = f.mip_filter ? 1u : 0u,
                    .unused0 = 0,
                    .unused1 = 0,
                    .unused2 = 0,
                    .fx = f.fx,
                    .fy = f.fy,
                    .cx = f.cx,
                    .cy = f.cy,
                    .clip_left = clip.left,
                    .clip_right = clip.right,
                    .clip_top = clip.top,
                    .clip_bottom = clip.bottom,
                    .near_plane = kNearPlane,
                    .far_plane = kFarPlane,
                };
                launch("fast_preprocess", pre,
                       {&splats.means, &splats.raw_scales, &splats.raw_rotations, &splats.raw_opacities, &splats.sh0,
                        &splats.shN, &splats.sh_value_bounds, &view, &camera_position, &s.mean_box, &s.conic_opacity,
                        &s.color_depth, &s.tile_info, &s.n_touched, &s.normals, &share_use},
                       div_up(n, 256), 1, 256,
                       {{kShBasesConstant, f.sh_bases},
                        {kShLayoutRestConstant, f.sh_layout_rest},
                        {kShStorageConstant, f.sh_storage}});
                exclusive_scan(s, s.n_touched, s.offsets, n, &s.counts);

                const uint32_t sort_bits = tile_bits(n_tiles) + key_depth_bits;
                const bool solid_bg = !has_bg_image && bg_color.is_valid() && bg_color.numel() >= 3;
                const Tensor& bg_use = has_bg_image ? bg_image : (solid_bg ? bg_color : none);
                const Tensor& normals_use = params.render_normal ? s.normals : none;
                // Instances, their sort, the tile ranges and the blend for buffers
                // of `capacity` instances; the kernels take the count from s.counts.
                const auto encode_raster = [&](const uint32_t capacity) {
                    fill(s.ranges, size_t{n_tiles} * 2, 0);
                    for (uint32_t i = 0; i < 2; ++i) {
                        reserve(s.keys[i], capacity, DataType::UInt32);
                        reserve(s.values[i], capacity, DataType::UInt32);
                    }
                    const InstanceParams inst{mk::address(s.mean_box), mk::address(s.conic_opacity),
                                              mk::address(s.tile_info), mk::address(s.n_touched),
                                              mk::address(s.offsets), mk::address(s.keys[0]),
                                              mk::address(s.values[0]), n, f.grid_w, key_depth_bits, capacity};
                    launch("fast_create_instances", inst,
                           {&s.mean_box, &s.conic_opacity, &s.tile_info, &s.n_touched, &s.offsets, &s.keys[0],
                            &s.values[0]},
                           div_up(n, 256), 1, 256);
                    f.sorted = sort_instances(s, capacity, sort_bits);
                    const RangeParams range{mk::address(s.keys[f.sorted]), mk::address(s.ranges),
                                            mk::address(s.counts), capacity, n_tiles, key_depth_bits, 0};
                    launch("fast_tile_ranges", range, {&s.keys[f.sorted], &s.ranges, &s.counts},
                           div_up(capacity, 256), 1, 256);
                    const BlendParams blend{
                        .ranges = mk::address(s.ranges),
                        .values = mk::address(s.values[f.sorted]),
                        .mean_box = mk::address(s.mean_box),
                        .conic_opacity = mk::address(s.conic_opacity),
                        .color_depth = mk::address(s.color_depth),
                        .normals = params.render_normal ? mk::address(s.normals) : 0,
                        .image = mk::address(s.image),
                        .alpha = mk::address(s.alpha),
                        .depth = params.render_depth ? mk::address(s.depth) : 0,
                        .normal = params.render_normal ? mk::address(s.normal) : 0,
                        .n_contrib = mk::address(s.n_contrib),
                        .final_transmittance = mk::address(s.final_transmittance),
                        .bg_color = solid_bg ? mk::address(bg_color) : 0,
                        .bg_image = has_bg_image ? mk::address(bg_image) : 0,
                        .width = f.width,
                        .height = f.height,
                        .grid_w = f.grid_w,
                        .unused = 0,
                    };
                    launch("fast_blend_forward", blend,
                           {&s.ranges, &s.values[f.sorted], &s.mean_box, &s.conic_opacity, &s.color_depth,
                            &normals_use, &s.image, &s.alpha, &s.depth, &s.normal, &s.n_contrib,
                            &s.final_transmittance, &bg_use},
                           f.grid_w, f.grid_h, kBlendThreads,
                           {{kRenderDepthConstant, params.render_depth ? 1u : 0u},
                            {kRenderNormalConstant, params.render_normal ? 1u : 0u}});
                };

                // The frame's one host round trip is the instance count. The host
                // reads a copy that the raster tail does not touch, submitted with
                // the scan, while the GPU runs the tail sized for the capacity of
                // earlier frames. A frame needing more encodes the tail again.
                reserve(s.count_copy, 2, DataType::UInt32);
                s.count_copy.slice(0, 0, 2).copy_(s.counts.slice(0, 0, 2));
                core::TensorFence counted(core::GpuBackend::Metal);
                counted.record(core::TensorExecutionTarget::current());
                const uint32_t speculated = s.instance_capacity;
                if (speculated > 0)
                    encode_raster(speculated);
                const Tensor counts = s.count_copy.slice(0, 0, 2).to(Device::CPU);
                const uint32_t* words = counts.ptr<uint32_t>();
                const uint64_t n_instances = uint64_t{words[0]} | (uint64_t{words[1]} << 32);
                if (n_instances > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    return fail(s, Code::InstanceOverflow,
                                std::format("FastGS instance count exceeds 32-bit range: {} instances from {} "
                                            "primitives across {} tiles",
                                            n_instances, n, n_tiles));
                f.n_instances = static_cast<uint32_t>(n_instances);
                if (speculated == 0 || f.n_instances > speculated) {
                    s.instance_capacity = std::max<uint32_t>(1024, f.n_instances + f.n_instances / 4);
                    encode_raster(s.instance_capacity);
                }
            } catch (const core::MemoryAllocationError& e) {
                return fail(s, Code::ResourceExhausted, std::format("OUT_OF_MEMORY: {}", e.what()));
            }

            f.means = splats.means;
            f.scales = splats.raw_scales;
            f.rotations = splats.raw_rotations;
            f.opacities = splats.raw_opacities;
            f.sh0 = splats.sh0;
            f.shN = splats.shN;
            f.sh_bounds = q16 ? splats.sh_value_bounds : Tensor();
            f.view = view;
            f.camera = camera_position;
            f.bg_color = bg_color;
            f.bg_image = bg_image;
            f.live = true;
            outputs.image = s.image;
            outputs.alpha = s.alpha;
            outputs.depth = params.render_depth ? s.depth : Tensor();
            outputs.normal = params.render_normal ? s.normal : Tensor();
            const bool has_work = f.n_instances > 0;
            s.frame = std::move(f);
            return {.code = Code::Success, .has_work = has_work, .message = {}};
        }

        // FastAdamGroup of fast_backward.metal; the fields of FusedAdamParam.
        struct AdamGroupParams {
            uint64_t param, packed, bounds, value_bounds, frozen, crop, screen_share;
            int32_t frozen_n, crop_n, screen_share_n;
            int32_t joint_bits, value_bits, value_cells;
            int32_t primitives, elements, attributes;
            float step_size, bc2_sqrt_rcp, frozen_lr_scale, cropbox_lr_scale;
            float screen_share_limit, screen_share_penalty;
            uint32_t enabled;
        };
        static_assert(sizeof(AdamGroupParams) == 120);

        // fast_adam_group: disabled groups stay zero and are never touched.
        AdamGroupParams adam_group(const lfs::gpu_ops::BackwardAdamParam& src, std::vector<const Tensor*>& uses) {
            AdamGroupParams dst{};
            if (!src.enabled)
                return dst;
            dst.param = mk::address(src.parameter);
            uses.push_back(&src.parameter);
            if (src.value_bits == 16) {
                dst.value_bits = 16;
                if (present(src.sh_value_bounds)) {
                    dst.value_bounds = mk::address(src.sh_value_bounds);
                    dst.value_cells = src.value_cells;
                    uses.push_back(&src.sh_value_bounds);
                }
            }
            dst.packed = address_if(src.packed_moments);
            dst.bounds = address_if(src.joint_bounds);
            uses.push_back(&src.packed_moments);
            uses.push_back(&src.joint_bounds);
            dst.joint_bits = src.joint_bits;
            dst.primitives = src.primitives;
            if (present(src.frozen_mask)) {
                dst.frozen = mk::address(src.frozen_mask);
                dst.frozen_n = static_cast<int32_t>(src.frozen_mask.numel());
                uses.push_back(&src.frozen_mask);
            }
            dst.frozen_lr_scale = src.frozen_lr_scale;
            if (present(src.crop_damping_mask)) {
                dst.crop = mk::address(src.crop_damping_mask);
                dst.crop_n = static_cast<int32_t>(src.crop_damping_mask.numel());
                uses.push_back(&src.crop_damping_mask);
            }
            dst.cropbox_lr_scale = src.cropbox_lr_scale;
            dst.elements = src.elements;
            dst.attributes = src.attributes;
            dst.step_size = src.step_size;
            dst.bc2_sqrt_rcp = src.bc2_sqrt_rcp;
            dst.enabled = 1;
            if (present(src.screen_share) && src.screen_share_limit > 0.f && src.screen_share_limit < 1.f) {
                dst.screen_share = mk::address(src.screen_share);
                dst.screen_share_n = static_cast<int32_t>(src.screen_share.numel());
                dst.screen_share_limit = src.screen_share_limit;
                dst.screen_share_penalty = src.screen_share_penalty;
                uses.push_back(&src.screen_share);
            }
            return dst;
        }

        struct BlendBackwardParams {
            uint64_t ranges, values, mean_box, conic_opacity, color_depth, normals;
            uint64_t grad_image, grad_alpha, grad_depth, grad_normal, bg_color, bg_image, n_contrib,
                final_transmittance;
            uint64_t grads, normal_grads, densification, error_map, edge_weight, edge_score;
            uint32_t n_instances, n_primitives, width, height;
            uint32_t grid_w, unused0, unused1, unused2;
        };

        struct BackwardShParams {
            uint64_t means, camera, shN, sh_bounds, n_touched, grads;
            AdamGroupParams sh0, shN_adam;
            float beta1, beta2, eps;
            uint32_t n;
        };

        struct BackwardGeometryParams {
            uint64_t means, scales, rotations, opacities, view, camera, n_touched, grads, normal_grads;
            uint64_t densification, far_mask, scale_loss, opacity_loss, sparsity_sigmoid, sparsity_z, sparsity_u;
            AdamGroupParams means_adam, rotation_adam, scaling_adam, opacity_adam;
            float beta1, beta2, eps;
            float scale_reg_weight, flatten_reg_weight, opacity_reg_weight, sparsity_rho, sparsity_grad_loss;
            float median_extent, r_min, r_max;
            float width, height, fx, fy;
            float clip_left, clip_right, clip_top, clip_bottom;
            uint32_t n, far_mask_n, sparsity_n, per_splat_mean_step;
        };

        // A read-only per-pixel map given as [H, W] or [1, H, W], made contiguous.
        Tensor pixel_map(const Tensor& map, const Frame& f, const char* const name) {
            if (!present(map))
                return {};
            const bool planar = map.ndim() == 2 || (map.ndim() == 3 && map.shape()[0] == 1);
            if (!planar || map.shape()[map.ndim() - 2] != f.height || map.shape()[map.ndim() - 1] != f.width ||
                map.dtype() != DataType::Float32)
                throw std::invalid_argument(std::format("{} must be a float [H, W] or [1, H, W] map of {}x{}, got {} {}",
                                                        name, f.height, f.width, map.shape().str(),
                                                        static_cast<int>(map.dtype())));
            return map.contiguous();
        }

        // A read-only [3, H, W] float map, made contiguous.
        Tensor channel_map(const Tensor& map, const Frame& f, const char* const name) {
            if (!present(map))
                return {};
            if (map.ndim() != 3 || map.shape()[0] != 3 || map.shape()[1] != f.height || map.shape()[2] != f.width ||
                map.dtype() != DataType::Float32)
                throw std::invalid_argument(std::format("{} must be a float [3, {}, {}] map, got {} {}", name, f.height,
                                                        f.width, map.shape().str(), static_cast<int>(map.dtype())));
            return map.contiguous();
        }

        void backward(lfs::gpu_ops::FastSaved& saved, const lfs::gpu_ops::RenderGradients& gradients,
                      Tensor& densification, const Tensor& error_map, const Tensor& edge_map, Tensor& edge_scores,
                      const lfs::gpu_ops::BackwardAdam& adam, const DensificationType densification_type) {
            auto& s = state_of(saved);
            const Frame f = std::exchange(s.frame, {});
            if (!f.live)
                throw std::logic_error("Metal fast raster backward needs the forward of the same frame");
            const Tensor grad_image = channel_map(gradients.image, f, "FastGS backward image gradient");
            if (!grad_image.is_valid())
                throw std::invalid_argument("FastGS backward expects a [3, H, W] image gradient, got none");
            const Tensor grad_alpha = pixel_map(gradients.alpha, f, "grad_alpha");
            const Tensor grad_depth = pixel_map(gradients.depth, f, "grad_depth");
            const Tensor grad_normal = channel_map(gradients.normal, f, "grad_normal");
            const bool normal_channel = grad_normal.is_valid();
            if (normal_channel && !f.normals)
                throw std::invalid_argument("grad_normal needs a forward that rendered normals");
            if (present(edge_map) != present(edge_scores))
                throw std::invalid_argument(std::format("Invalid edge-score inputs: edge map {}, edge scores {}",
                                                        present(edge_map) ? "present" : "absent",
                                                        present(edge_scores) ? "present" : "absent"));
            const Tensor edge_weight = pixel_map(edge_map, f, "edge_map");
            if (present(edge_scores) && (edge_scores.numel() < f.n || !edge_scores.is_contiguous()))
                throw std::invalid_argument(std::format("edge scores need {} contiguous values, got {}", f.n,
                                                        edge_scores.numel()));

            const bool update_densification = densification.is_valid() && densification.ndim() == 2 &&
                                              densification.shape()[1] >= f.n;
            if (update_densification && !densification.is_contiguous())
                throw std::invalid_argument("densification statistics must be contiguous");
            const bool pixel_error = update_densification && present(error_map);
            const Tensor error = pixel_error ? pixel_map(error_map, f, "error_map") : Tensor();
            // blend_backward template choice; the None mode counts visibility in the geometry pass.
            const uint32_t blend_densification = densification_type == DensificationType::MRNF && update_densification
                                                     ? 2u
                                                 : pixel_error ? 1u
                                                               : 0u;
            const bool count_visible = update_densification && !pixel_error &&
                                       densification_type == DensificationType::None;

            using lfs::gpu_ops::AdamSlot;
            const auto& group = [&](const AdamSlot slot) -> const lfs::gpu_ops::BackwardAdamParam& {
                return adam.groups[static_cast<size_t>(slot)];
            };
            if (!std::ranges::any_of(adam.groups, [](const auto& g) { return g.enabled; }))
                throw std::runtime_error("FastGS fused Adam state is not available");

            reserve(s.grads, size_t{f.n} * kGradStride, DataType::Float32);
            fill(s.grads, size_t{f.n} * kGradStride, 0);
            if (normal_channel) {
                reserve(s.normal_grads, size_t{f.n} * 4, DataType::Float32);
                fill(s.normal_grads, size_t{f.n} * 4, 0);
            }
            const Tensor none;
            const Tensor& normal_grads_use = normal_channel ? s.normal_grads : none;

            if (f.n_instances > 0) {
                const bool has_bg_image = present(f.bg_image);
                const bool solid_bg = !has_bg_image && f.bg_color.is_valid() && f.bg_color.numel() >= 3;
                const Tensor& bg_use = has_bg_image ? f.bg_image : (solid_bg ? f.bg_color : none);
                const Tensor& densification_use = blend_densification != 0 ? densification : none;
                const BlendBackwardParams params{
                    .ranges = mk::address(s.ranges),
                    .values = mk::address(s.values[f.sorted]),
                    .mean_box = mk::address(s.mean_box),
                    .conic_opacity = mk::address(s.conic_opacity),
                    .color_depth = mk::address(s.color_depth),
                    .normals = normal_channel ? mk::address(s.normals) : 0,
                    .grad_image = mk::address(grad_image),
                    .grad_alpha = address_if(grad_alpha),
                    .grad_depth = address_if(grad_depth),
                    .grad_normal = normal_channel ? mk::address(grad_normal) : 0,
                    .bg_color = solid_bg ? mk::address(f.bg_color) : 0,
                    .bg_image = has_bg_image ? mk::address(f.bg_image) : 0,
                    .n_contrib = mk::address(s.n_contrib),
                    .final_transmittance = mk::address(s.final_transmittance),
                    .grads = mk::address(s.grads),
                    .normal_grads = normal_channel ? mk::address(s.normal_grads) : 0,
                    .densification = blend_densification != 0 ? mk::address(densification) : 0,
                    .error_map = pixel_error ? mk::address(error) : 0,
                    .edge_weight = address_if(edge_weight),
                    .edge_score = address_if(edge_scores),
                    .n_instances = f.n_instances,
                    .n_primitives = f.n,
                    .width = f.width,
                    .height = f.height,
                    .grid_w = f.grid_w,
                    .unused0 = 0,
                    .unused1 = 0,
                    .unused2 = 0,
                };
                const Tensor& normals_use = normal_channel ? s.normals : none;
                launch("fast_blend_backward", params,
                       {&s.ranges, &s.values[f.sorted], &s.mean_box, &s.conic_opacity, &s.color_depth, &normals_use,
                        &grad_image, &grad_alpha, &grad_depth, &grad_normal, &bg_use, &s.n_contrib,
                        &s.final_transmittance, &s.grads, &normal_grads_use, &densification_use, &error, &edge_weight,
                        &edge_scores},
                       f.grid_w * f.grid_h, 1, kBackwardThreads,
                       {{kDensificationConstant, blend_densification},
                        {kNormalChannelConstant, normal_channel ? 1u : 0u},
                        {kDepthGradConstant, present(grad_depth) ? 1u : 0u},
                        {kEdgeWeightConstant, present(edge_weight) && present(edge_scores) ? 1u : 0u}});
            }

            const uint32_t blocks = div_up(f.n, 256);
            std::vector<const Tensor*> uses{&f.means, &f.camera, &f.shN, &f.sh_bounds, &s.n_touched, &s.grads};
            const BackwardShParams sh{
                .means = mk::address(f.means),
                .camera = mk::address(f.camera),
                .shN = f.sh_bases > 1 ? address_if(f.shN) : 0,
                .sh_bounds = address_if(f.sh_bounds),
                .n_touched = mk::address(s.n_touched),
                .grads = mk::address(s.grads),
                .sh0 = adam_group(group(AdamSlot::Sh0), uses),
                .shN_adam = adam_group(group(AdamSlot::ShN), uses),
                .beta1 = adam.beta1,
                .beta2 = adam.beta2,
                .eps = adam.eps,
                .n = f.n,
            };
            // The kernels clamp to degree 3: 15 rest coefficients in 12 float4 slots.
            LFS_ASSERT_MSG(f.sh_layout_rest <= 15 && (f.sh_layout_rest * 3 + 3) / 4 <= kShParts * kShSlotsPerThread,
                           std::format("fast_backward_sh covers 15 SH rest coefficients in {} slots, the layout has {}",
                                       kShParts * kShSlotsPerThread, f.sh_layout_rest));
            launch("fast_backward_sh", sh, std::span<const Tensor* const>(uses), blocks, 1, 256 * kShParts,
                   {{kShBasesConstant, f.sh_bases},
                    {kShLayoutRestConstant, f.sh_layout_rest},
                    {kShStorageConstant, f.sh_storage}});

            const bool scale_loss = adam.scale_reg_weight > 0.f && present(adam.scale_reg_loss);
            const bool opacity_loss = adam.opacity_reg_weight > 0.f && present(adam.opacity_reg_loss);
            const bool sparsity = present(adam.sparsity_sigmoid);
            const auto& means_group = group(AdamSlot::Means);
            const bool far_mask = present(adam.far_mask) && means_group.enabled && means_group.primitives > 0;
            const ClipBounds clip = clip_bounds(f);
            uses = {&f.means, &f.scales, &f.rotations, &f.opacities, &f.view, &f.camera, &s.n_touched, &s.grads,
                    &normal_grads_use};
            if (count_visible)
                uses.push_back(&densification);
            if (far_mask)
                uses.push_back(&adam.far_mask);
            if (scale_loss)
                uses.push_back(&adam.scale_reg_loss);
            if (opacity_loss)
                uses.push_back(&adam.opacity_reg_loss);
            if (sparsity)
                uses.insert(uses.end(), {&adam.sparsity_sigmoid, &adam.sparsity_z, &adam.sparsity_u});
            const BackwardGeometryParams geometry{
                .means = mk::address(f.means),
                .scales = mk::address(f.scales),
                .rotations = mk::address(f.rotations),
                .opacities = mk::address(f.opacities),
                .view = mk::address(f.view),
                .camera = mk::address(f.camera),
                .n_touched = mk::address(s.n_touched),
                .grads = mk::address(s.grads),
                .normal_grads = normal_channel ? mk::address(s.normal_grads) : 0,
                .densification = count_visible ? mk::address(densification) : 0,
                .far_mask = far_mask ? mk::address(adam.far_mask) : 0,
                .scale_loss = scale_loss ? mk::address(adam.scale_reg_loss) : 0,
                .opacity_loss = opacity_loss ? mk::address(adam.opacity_reg_loss) : 0,
                .sparsity_sigmoid = sparsity ? mk::address(adam.sparsity_sigmoid) : 0,
                .sparsity_z = sparsity ? address_if(adam.sparsity_z) : 0,
                .sparsity_u = sparsity ? address_if(adam.sparsity_u) : 0,
                .means_adam = adam_group(means_group, uses),
                .rotation_adam = adam_group(group(AdamSlot::Rotation), uses),
                .scaling_adam = adam_group(group(AdamSlot::Scaling), uses),
                .opacity_adam = adam_group(group(AdamSlot::Opacity), uses),
                .beta1 = adam.beta1,
                .beta2 = adam.beta2,
                .eps = adam.eps,
                .scale_reg_weight = adam.scale_reg_weight,
                .flatten_reg_weight = adam.flatten_reg_weight,
                .opacity_reg_weight = adam.opacity_reg_weight,
                .sparsity_rho = adam.sparsity_rho,
                .sparsity_grad_loss = adam.sparsity_grad_loss,
                .median_extent = adam.median_extent,
                .r_min = adam.r_min,
                .r_max = adam.r_max,
                .width = static_cast<float>(f.width),
                .height = static_cast<float>(f.height),
                .fx = f.fx,
                .fy = f.fy,
                .clip_left = clip.left,
                .clip_right = clip.right,
                .clip_top = clip.top,
                .clip_bottom = clip.bottom,
                .n = f.n,
                .far_mask_n = far_mask ? static_cast<uint32_t>(std::min<size_t>(adam.far_mask.numel(),
                                                                                static_cast<size_t>(means_group.primitives)))
                                       : 0,
                .sparsity_n = sparsity ? mk::count32(adam.sparsity_sigmoid.numel(), "sparsity") : 0,
                .per_splat_mean_step = adam.per_splat_mean_step ? 1u : 0u,
            };
            launch("fast_backward_geometry", geometry, std::span<const Tensor* const>(uses), blocks, 1, 256,
                   {{kMipFilterConstant, f.mip_filter ? 1u : 0u}});
        }

        void release(lfs::gpu_ops::FastSaved& saved) noexcept {
            if (saved.backend)
                static_cast<MetalFastState&>(*saved.backend).frame = {};
        }

        void release_caches(lfs::gpu_ops::FastSaved& saved) noexcept {
            if (!saved.backend)
                return;
            auto& s = static_cast<MetalFastState&>(*saved.backend);
            s.image = Tensor();
            s.alpha = Tensor();
            s.depth = Tensor();
            s.normal = Tensor();
            s.width = -1;
            s.height = -1;
        }

        void record_vram(const lfs::gpu_ops::FastSaved& saved, const Tensor& image, const Tensor& alpha,
                         const bool run_gaussian_backward, const size_t num_primitives) {
            const auto* s = state_of(saved);
            if (s == nullptr)
                return;
            constexpr std::string_view scope = "rasterizer.fastgs";
            size_t per_primitive = 0;
            for (const Tensor* tensor : {&s->mean_box, &s->conic_opacity, &s->color_depth, &s->tile_info, &s->normals,
                                         &s->n_touched, &s->offsets})
                per_primitive += tensor_reserved_bytes(*tensor);
            size_t per_tile = 0;
            for (const Tensor* tensor : {&s->ranges, &s->final_transmittance, &s->n_contrib})
                per_tile += tensor_reserved_bytes(*tensor);
            size_t sort = 0;
            for (const Tensor* tensor :
                 {&s->keys[0], &s->keys[1], &s->values[0], &s->values[1], &s->histogram, &s->block_sums})
                sort += tensor_reserved_bytes(*tensor);
            record_vram_current(scope, "forward.per_primitive_buffers", per_primitive);
            record_vram_current(scope, "forward.per_tile_buffers", per_tile);
            record_vram_current(scope, "forward.sort_buffers", sort);
            record_vram_current(scope, "backward.gradient_helpers",
                                run_gaussian_backward ? num_primitives * (kGradStride + (s->frame.normals ? 4 : 0)) *
                                                            sizeof(float)
                                                      : 0,
                                true);
            record_vram_tensor(scope, "output.image", image);
            record_vram_tensor(scope, "output.alpha", alpha);
            record_vram_tensor(scope, "saved.bg_color", s->frame.bg_color);
        }

        // Pipelines compile at their first launch: one zero-step forward and
        // backward on a tiny scene.
        void warmup() {
            lfs::gpu_ops::FastSaved saved{.backend = std::make_unique<MetalFastState>()};
            constexpr size_t count = 4;
            const std::vector<float> identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 1};
            const std::vector<float> rotation{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
            auto upload = [](const std::vector<float>& values, core::TensorShape shape) {
                return Tensor::from_blob(const_cast<float*>(values.data()), shape, Device::CPU, DataType::Float32)
                    .to(Device::GPU);
            };
            const Tensor means = Tensor::zeros({count, 3}, Device::GPU);
            const Tensor scales = Tensor::full({count, 3}, -2.f, Device::GPU);
            const Tensor rotations = upload(rotation, {count, 4});
            const Tensor opacities = Tensor::zeros({count, 1}, Device::GPU);
            const Tensor sh0 = Tensor::zeros({count, 1, 3}, Device::GPU);
            const Tensor view = upload(identity, {4, 4});
            const Tensor camera = Tensor::zeros({3}, Device::GPU);
            Tensor none;
            Tensor image, alpha, depth, normal, share;
            const lfs::gpu_ops::FastParams params{.full_image = {32, 32}, .intrinsics = {32.f, 32.f, 16.f, 16.f}};
            const auto result =
                forward(saved, {.means = means, .raw_scales = scales, .raw_rotations = rotations, .raw_opacities = opacities, .sh0 = sh0, .shN = none, .sh_value_bounds = none},
                        view, camera, none, none, params,
                        {.image = image, .alpha = alpha, .depth = depth, .normal = normal}, share);
            if (result.code != Code::Success)
                throw std::runtime_error(std::format("Metal fast raster warmup forward failed: {}", result.message));
            Tensor parameter = opacities.clone();
            Tensor moments = Tensor::zeros({count, 4}, Device::GPU, DataType::UInt8);
            Tensor bounds = Tensor::zeros({1, 4}, Device::GPU);
            const lfs::gpu_ops::BackwardAdamParam off{.parameter = none,
                                                      .packed_moments = none,
                                                      .joint_bounds = none,
                                                      .sh_value_bounds = none,
                                                      .frozen_mask = none,
                                                      .crop_damping_mask = none,
                                                      .screen_share = none};
            const lfs::gpu_ops::BackwardAdamParam opacity{.parameter = parameter,
                                                          .packed_moments = moments,
                                                          .joint_bounds = bounds,
                                                          .sh_value_bounds = none,
                                                          .frozen_mask = none,
                                                          .crop_damping_mask = none,
                                                          .screen_share = none,
                                                          .joint_bits = 16,
                                                          .primitives = static_cast<int>(count),
                                                          .elements = static_cast<int>(count),
                                                          .attributes = 1,
                                                          .enabled = true};
            const lfs::gpu_ops::BackwardAdam adam{.groups = {off, off, off, opacity, off, off},
                                                  .scale_reg_loss = none,
                                                  .opacity_reg_loss = none,
                                                  .sparsity_sigmoid = none,
                                                  .sparsity_z = none,
                                                  .sparsity_u = none,
                                                  .far_mask = none};
            const Tensor grad = Tensor::zeros_like(image);
            backward(saved, {.image = grad, .alpha = none, .depth = none, .normal = none}, none, none, none, none,
                     adam, DensificationType::None);
        }

        lfs::gpu_ops::State create() { return std::make_unique<MetalFastState>(); }
    } // namespace

    const lfs::gpu_ops::FastRasterOps& metal_fast_ops() {
        static const lfs::gpu_ops::FastRasterOps ops{
            .create = create,
            .forward = forward,
            .backward = backward,
            .release = release,
            .warmup = warmup,
            .record_vram = record_vram,
            .release_caches = release_caches,
        };
        return ops;
    }
} // namespace lfs::training
