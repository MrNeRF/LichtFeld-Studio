/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fastgs_tile_culling_boundary_cuda.hpp"
#include "kernel_utils.cuh"

namespace fastgs_tile_culling_boundary_test {
    namespace {

        using fast_lfs::rasterization::kernels::ceil_tile_clamped;
        using fast_lfs::rasterization::kernels::compute_exact_n_touched_tiles;
        using fast_lfs::rasterization::kernels::ellipse_box_overlap_test;
        using fast_lfs::rasterization::kernels::ellipse_range_bound;
        using fast_lfs::rasterization::kernels::floor_tile_clamped;
        using fast_lfs::rasterization::kernels::splat_overlaps_subtile_ellipse;
        namespace cfg = fast_lfs::rasterization::config;

        static_assert(cfg::tile_width == 16 && cfg::tile_height == 16,
                      "boundary probes assume 16×16 tiles");
        static_assert(cfg::warp_subtile_width == 8 && cfg::warp_subtile_height == 4,
                      "boundary probes assume 8×4 warp sub-tiles");

        __device__ uint collect_instance_tiles(
            const float2 mean2d,
            const float3 conic,
            const uint4 screen_bounds,
            const float power_threshold,
            const uint grid_width,
            uint* tiles,
            const uint max_tiles) {
            const float2 mean2d_shifted = mean2d - 0.5f;
            const float radius_sq = 2.0f * power_threshold;
            uint n = 0;

            const uint screen_bounds_width = screen_bounds.y - screen_bounds.x;
            const uint screen_bounds_height = screen_bounds.w - screen_bounds.z;

            if (screen_bounds_height <= screen_bounds_width) {
                for (uint tile_y = screen_bounds.z; tile_y < screen_bounds.w; tile_y++) {
                    const float y0 = static_cast<float>(tile_y * cfg::tile_height) - mean2d_shifted.y;
                    const float y1 = y0 + static_cast<float>(cfg::tile_height);
                    const float2 bound = ellipse_range_bound(conic, radius_sq, y0, y1);
                    const uint min_x = floor_tile_clamped(
                        bound.x + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, cfg::tile_width);
                    const uint max_x = ceil_tile_clamped(
                        bound.y + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, cfg::tile_width);
                    for (uint tile_x = min_x; tile_x < max_x && n < max_tiles; tile_x++) {
                        tiles[n++] = tile_y * grid_width + tile_x;
                    }
                }
            } else {
                const float3 conic_transposed = make_float3(conic.z, conic.y, conic.x);
                for (uint tile_x = screen_bounds.x; tile_x < screen_bounds.y; tile_x++) {
                    const float x0 = static_cast<float>(tile_x * cfg::tile_width) - mean2d_shifted.x;
                    const float x1 = x0 + static_cast<float>(cfg::tile_width);
                    const float2 bound = ellipse_range_bound(conic_transposed, radius_sq, x0, x1);
                    const uint min_y = floor_tile_clamped(
                        bound.x + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, cfg::tile_height);
                    const uint max_y = ceil_tile_clamped(
                        bound.y + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, cfg::tile_height);
                    for (uint tile_y = min_y; tile_y < max_y && n < max_tiles; tile_y++) {
                        tiles[n++] = tile_y * grid_width + tile_x;
                    }
                }
            }
            return n;
        }

        __device__ bool pixel_contributes(
            const float2 mean2d,
            const float3 conic,
            const float power_threshold,
            const float opacity,
            const uint x,
            const uint y) {
            if (!(opacity >= cfg::min_alpha_threshold)) {
                return false;
            }
            const double dx = static_cast<double>(mean2d.x) - (static_cast<double>(x) + 0.5);
            const double dy = static_cast<double>(mean2d.y) - (static_cast<double>(y) + 0.5);
            const double sigma =
                0.5 * (static_cast<double>(conic.x) * dx * dx + static_cast<double>(conic.z) * dy * dy) +
                static_cast<double>(conic.y) * dx * dy;
            return sigma >= 0.0 && sigma <= static_cast<double>(power_threshold);
        }

        __device__ bool tile_in_list(const uint* tiles, uint n, uint tile) {
            for (uint i = 0; i < n; i++) {
                if (tiles[i] == tile) {
                    return true;
                }
            }
            return false;
        }

        // Walk domain matching preprocess: closed interval on mean±extent.
        // Inclusive start is ceil(t)-1; exclusive end uses ceil_tile_clamped so
        // master's IEEE-ceil helper still drops an exact tile-boundary hit.
        __device__ uint4 probe_screen_bounds(
            const float2 mean2d,
            const float extent_x,
            const float extent_y,
            const uint grid_width,
            const uint grid_height) {
            const float tw = static_cast<float>(cfg::tile_width);
            const float th = static_cast<float>(cfg::tile_height);
            const int x_min = max(0, __float2int_ru((mean2d.x - extent_x) / tw) - 1);
            const int y_min = max(0, __float2int_ru((mean2d.y - extent_y) / th) - 1);
            const uint x_max = ceil_tile_clamped(mean2d.x + extent_x, 0, grid_width, cfg::tile_width);
            const uint y_max = ceil_tile_clamped(mean2d.y + extent_y, 0, grid_height, cfg::tile_height);
            return make_uint4(
                min(grid_width, static_cast<uint>(x_min)),
                min(grid_width, x_max),
                min(grid_height, static_cast<uint>(y_min)),
                min(grid_height, y_max));
        }

        __global__ void floor_ceil_kernel(FloorCeilSample* samples, int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            FloorCeilSample s = samples[i];
            s.floor_tile = floor_tile_clamped(s.coord, s.min_tile, s.max_tile, s.tile_size);
            s.ceil_tile = ceil_tile_clamped(s.coord, s.min_tile, s.max_tile, s.tile_size);
            samples[i] = s;
        }

        __global__ void overlap_kernel(OverlapSample* samples, int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            OverlapSample s = samples[i];
            s.overlaps = ellipse_box_overlap_test(
                             make_float3(s.inv_a, s.inv_b, s.inv_c), s.x0, s.x1, s.y0, s.y1)
                             ? 1
                             : 0;
            samples[i] = s;
        }

        __global__ void splat_subtile_kernel(SplatSubtileSample* samples, int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            SplatSubtileSample s = samples[i];
            s.power_threshold = logf(s.opacity * cfg::min_alpha_threshold_rcp);
            s.overlaps = splat_overlaps_subtile_ellipse(
                             make_float2(s.mean_x, s.mean_y),
                             make_float3(s.conic_x, s.conic_y, s.conic_z),
                             s.opacity,
                             s.sub_x0, s.sub_y0, s.sub_w, s.sub_h)
                             ? 1
                             : 0;
            samples[i] = s;
        }

        __global__ void agreement_kernel(AgreementSample* samples, int n) {
            const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= n) {
                return;
            }
            AgreementSample s = samples[i];
            const float2 mean2d = make_float2(s.mean_x, s.mean_y);
            const float3 conic = make_float3(s.conic_x, s.conic_y, s.conic_z);
            const uint grid_width = (s.width + static_cast<uint>(cfg::tile_width) - 1u) /
                                    static_cast<uint>(cfg::tile_width);
            const uint grid_height = (s.height + static_cast<uint>(cfg::tile_height) - 1u) /
                                     static_cast<uint>(cfg::tile_height);
            const float power_threshold = logf(s.opacity * cfg::min_alpha_threshold_rcp);
            s.power_threshold = power_threshold;
            const float det = fmaxf(conic.x * conic.z - conic.y * conic.y, 1e-20f);
            const float extent_x = fmaxf(sqrtf(2.0f * fmaxf(power_threshold, 0.0f) * (conic.z / det)) - 0.5f, 0.0f);
            const float extent_y = fmaxf(sqrtf(2.0f * fmaxf(power_threshold, 0.0f) * (conic.x / det)) - 0.5f, 0.0f);
            const uint4 screen_bounds = probe_screen_bounds(
                mean2d, extent_x, extent_y, grid_width, grid_height);

            s.count = compute_exact_n_touched_tiles(
                mean2d, conic, screen_bounds, power_threshold, true);

            uint instance_tiles[kMaxTiles];
            uint n_instances = 0;
            if (s.count > 0) {
                const uint walk_cap = s.count < static_cast<uint>(kMaxTiles)
                                          ? s.count
                                          : static_cast<uint>(kMaxTiles);
                n_instances = collect_instance_tiles(
                    mean2d, conic, screen_bounds, power_threshold, grid_width,
                    instance_tiles, walk_cap);
            }
            s.n_instances = n_instances;
            for (uint t = 0; t < static_cast<uint>(kMaxTiles); t++) {
                s.instance_tiles[t] = t < n_instances ? instance_tiles[t] : 0u;
            }

            uint blend_tiles[kMaxTiles];
            uint n_blend_tiles = 0;
            uint n_blend_pixels = 0;
            for (uint y = 0; y < s.height; y++) {
                for (uint x = 0; x < s.width; x++) {
                    if (!pixel_contributes(mean2d, conic, power_threshold, s.opacity, x, y)) {
                        continue;
                    }
                    n_blend_pixels++;
                    const uint tile = (y / static_cast<uint>(cfg::tile_height)) * grid_width +
                                      (x / static_cast<uint>(cfg::tile_width));
                    if (!tile_in_list(blend_tiles, n_blend_tiles, tile) &&
                        n_blend_tiles < static_cast<uint>(kMaxTiles)) {
                        blend_tiles[n_blend_tiles++] = tile;
                    }
                }
            }
            s.n_blend_tiles = n_blend_tiles;
            s.n_blend_pixels = n_blend_pixels;
            for (uint t = 0; t < static_cast<uint>(kMaxTiles); t++) {
                s.blend_tiles[t] = t < n_blend_tiles ? blend_tiles[t] : 0u;
            }

            uint n_missing = 0;
            for (uint t = 0; t < n_blend_tiles; t++) {
                if (!tile_in_list(instance_tiles, n_instances, blend_tiles[t])) {
                    n_missing++;
                }
            }
            s.n_missing_blend_tiles = n_missing;

            uint n_bwd_drops = 0;
            const float sub_w = static_cast<float>(cfg::warp_subtile_width);
            const float sub_h = static_cast<float>(cfg::warp_subtile_height);
            for (uint tile_y = 0; tile_y < grid_height; tile_y++) {
                for (uint tile_x = 0; tile_x < grid_width; tile_x++) {
                    const uint origin_x = tile_x * static_cast<uint>(cfg::tile_width);
                    const uint origin_y = tile_y * static_cast<uint>(cfg::tile_height);
                    for (uint st = 0; st < 8u; st++) {
                        const uint ox = (st & 1u) * static_cast<uint>(cfg::warp_subtile_width);
                        const uint oy = (st >> 1) * static_cast<uint>(cfg::warp_subtile_height);
                        bool any_pixel = false;
                        for (uint ly = 0; ly < static_cast<uint>(cfg::warp_subtile_height); ly++) {
                            for (uint lx = 0; lx < static_cast<uint>(cfg::warp_subtile_width); lx++) {
                                const uint x = origin_x + ox + lx;
                                const uint y = origin_y + oy + ly;
                                if (x < s.width && y < s.height &&
                                    pixel_contributes(mean2d, conic, power_threshold, s.opacity, x, y)) {
                                    any_pixel = true;
                                }
                            }
                        }
                        if (!any_pixel) {
                            continue;
                        }
                        const bool overlap = splat_overlaps_subtile_ellipse(
                            mean2d, conic, s.opacity,
                            static_cast<float>(origin_x + ox),
                            static_cast<float>(origin_y + oy),
                            sub_w, sub_h);
                        if (!overlap) {
                            n_bwd_drops++;
                        }
                    }
                }
            }
            s.n_bwd_drops = n_bwd_drops;
            samples[i] = s;
        }

        template <typename T, typename Kernel>
        cudaError_t launch_inplace(T* host, int n, Kernel kernel) {
            if (n <= 0) {
                return cudaSuccess;
            }
            T* device = nullptr;
            const size_t bytes = static_cast<size_t>(n) * sizeof(T);
            cudaError_t err = cudaMalloc(&device, bytes);
            if (err != cudaSuccess) {
                return err;
            }
            err = cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(device);
                return err;
            }
            const int threads = 32;
            const int blocks = (n + threads - 1) / threads;
            kernel<<<blocks, threads>>>(device, n);
            err = cudaGetLastError();
            if (err == cudaSuccess) {
                err = cudaDeviceSynchronize();
            }
            if (err == cudaSuccess) {
                err = cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);
            }
            cudaFree(device);
            return err;
        }

    } // namespace

    cudaError_t eval_floor_ceil(FloorCeilSample* samples, int n) {
        return launch_inplace(samples, n, floor_ceil_kernel);
    }

    cudaError_t eval_ellipse_overlap(OverlapSample* samples, int n) {
        return launch_inplace(samples, n, overlap_kernel);
    }

    cudaError_t eval_splat_subtile(SplatSubtileSample* samples, int n) {
        return launch_inplace(samples, n, splat_subtile_kernel);
    }

    cudaError_t eval_agreement(AgreementSample* samples, int n) {
        return launch_inplace(samples, n, agreement_kernel);
    }

} // namespace fastgs_tile_culling_boundary_test
