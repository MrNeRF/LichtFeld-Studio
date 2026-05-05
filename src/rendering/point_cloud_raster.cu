/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "point_cloud_raster.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::rendering::pcraster {

    namespace {

        constexpr std::uint64_t kEmptyPacked = ~0ULL;

        __device__ inline float matMulRow(const float* M, int row, float x, float y, float z, float w) {
            return M[row + 0] * x + M[row + 4] * y + M[row + 8] * z + M[row + 12] * w;
        }

        __device__ inline std::uint64_t packDepthColor(float depth, float r, float g, float b) {
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);
            const std::uint32_t depth_u = __float_as_uint(fmaxf(depth, 0.0f));
            const std::uint32_t r8 = static_cast<std::uint32_t>(r * 255.0f + 0.5f);
            const std::uint32_t g8 = static_cast<std::uint32_t>(g * 255.0f + 0.5f);
            const std::uint32_t b8 = static_cast<std::uint32_t>(b * 255.0f + 0.5f);
            const std::uint32_t color = r8 | (g8 << 8) | (b8 << 16);
            return (static_cast<std::uint64_t>(depth_u) << 32) | static_cast<std::uint64_t>(color);
        }

        __global__ void clearPackedBuffer(std::uint64_t* buffer, int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            buffer[idx] = kEmptyPacked;
        }

        __global__ void rasterizePointsKernel(LaunchParams params, std::uint64_t* packed_buffer) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= static_cast<int>(params.n_points)) {
                return;
            }

            int transform_index = 0;
            if (params.transform_indices) {
                int t = params.transform_indices[idx];
                if (t < 0) {
                    t = 0;
                }
                if (t >= params.n_transforms) {
                    t = params.n_transforms > 0 ? params.n_transforms - 1 : 0;
                }
                transform_index = t;
            }
            if (params.visibility_mask && transform_index < params.n_visibility &&
                !params.visibility_mask[transform_index]) {
                return;
            }

            float x = params.positions[idx * 3 + 0];
            float y = params.positions[idx * 3 + 1];
            float z = params.positions[idx * 3 + 2];

            if (params.transforms && params.n_transforms > 0) {
                const float* M = &params.transforms[transform_index * 16];
                const float nx = matMulRow(M, 0, x, y, z, 1.0f);
                const float ny = matMulRow(M, 1, x, y, z, 1.0f);
                const float nz = matMulRow(M, 2, x, y, z, 1.0f);
                const float nw = matMulRow(M, 3, x, y, z, 1.0f);
                if (fabsf(nw) > 1e-6f) {
                    x = nx / nw;
                    y = ny / nw;
                    z = nz / nw;
                } else {
                    x = nx;
                    y = ny;
                    z = nz;
                }
            }

            bool desaturate = false;
            if (params.has_crop) {
                const float* T = params.crop.to_local;
                const float lx = matMulRow(T, 0, x, y, z, 1.0f);
                const float ly = matMulRow(T, 1, x, y, z, 1.0f);
                const float lz = matMulRow(T, 2, x, y, z, 1.0f);
                const bool inside = (lx >= params.crop.min[0] && lx <= params.crop.max[0] &&
                                     ly >= params.crop.min[1] && ly <= params.crop.max[1] &&
                                     lz >= params.crop.min[2] && lz <= params.crop.max[2]);
                const bool visible = params.crop.inverse ? !inside : inside;
                if (!visible) {
                    if (!params.crop.desaturate) {
                        return;
                    }
                    desaturate = true;
                }
            }

            const float* V = params.view;
            const float view_x = matMulRow(V, 0, x, y, z, 1.0f);
            const float view_y = matMulRow(V, 1, x, y, z, 1.0f);
            const float view_z = matMulRow(V, 2, x, y, z, 1.0f);

            float pixel_x = 0.0f;
            float pixel_y = 0.0f;
            float depth = 0.0f;

            if (params.equirectangular) {
                const float len = sqrtf(view_x * view_x + view_y * view_y + view_z * view_z);
                if (len <= 1e-6f) {
                    return;
                }
                const float dx = view_x / len;
                const float dy = view_y / len;
                const float dz = view_z / len;
                const float pi = 3.14159265358979323846f;
                const float u = 0.5f + atan2f(dx, -dz) / (2.0f * pi);
                const float v = 0.5f + asinf(fminf(fmaxf(dy, -1.0f), 1.0f)) / pi;
                pixel_x = u * static_cast<float>(params.width - 1);
                pixel_y = v * static_cast<float>(params.height - 1);
                if (!isfinite(pixel_x) || !isfinite(pixel_y) ||
                    pixel_x < 0.0f || pixel_x >= static_cast<float>(params.width) ||
                    pixel_y < 0.0f || pixel_y >= static_cast<float>(params.height)) {
                    return;
                }
                depth = len;
            } else {
                const float* P = params.view_proj;
                const float cx = matMulRow(P, 0, x, y, z, 1.0f);
                const float cy = matMulRow(P, 1, x, y, z, 1.0f);
                const float cz = matMulRow(P, 2, x, y, z, 1.0f);
                const float cw = matMulRow(P, 3, x, y, z, 1.0f);
                if (fabsf(cw) <= 1e-6f) {
                    return;
                }
                const float ndc_x = cx / cw;
                const float ndc_y = cy / cw;
                const float ndc_z = cz / cw;
                if (!isfinite(ndc_x) || !isfinite(ndc_y) || !isfinite(ndc_z) ||
                    ndc_x < -1.0f || ndc_x > 1.0f ||
                    ndc_y < -1.0f || ndc_y > 1.0f ||
                    ndc_z < -1.0f || ndc_z > 1.0f) {
                    return;
                }
                pixel_x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(params.width - 1);
                pixel_y = (ndc_y * 0.5f + 0.5f) * static_cast<float>(params.height - 1);
                depth = params.orthographic ? -view_z : fmaxf(-view_z, 0.0f);
                if (depth <= 0.0f && !params.orthographic) {
                    return;
                }
            }

            const float voxel = fmaxf(params.voxel_size * params.scaling_modifier, 1e-5f);
            int radius = 1;
            if (params.orthographic) {
                const float pixels_per_world =
                    static_cast<float>(params.height) / fmaxf(params.ortho_scale, 1e-5f);
                radius = max(1, static_cast<int>(ceilf(voxel * pixels_per_world * 0.5f)));
            } else {
                radius = max(1, static_cast<int>(ceilf(voxel * params.focal_y / fmaxf(depth, 1e-4f))));
            }

            float r = params.colors[idx * 3 + 0];
            float g = params.colors[idx * 3 + 1];
            float b = params.colors[idx * 3 + 2];
            r = fminf(fmaxf(r, 0.0f), 1.0f);
            g = fminf(fmaxf(g, 0.0f), 1.0f);
            b = fminf(fmaxf(b, 0.0f), 1.0f);
            if (desaturate) {
                const float gray = 0.299f * r + 0.587f * g + 0.114f * b;
                r = r + (gray - r) * 0.75f;
                g = g + (gray - g) * 0.75f;
                b = b + (gray - b) * 0.75f;
            }

            const std::uint64_t packed = packDepthColor(depth, r, g, b);
            const int center_x = static_cast<int>(rintf(pixel_x));
            const int center_y = static_cast<int>(rintf(pixel_y));
            const int r_sq = radius * radius;

            for (int dy = -radius; dy <= radius; ++dy) {
                const int yy = center_y + dy;
                if (yy < 0 || yy >= params.height) {
                    continue;
                }
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = center_x + dx;
                    if (xx < 0 || xx >= params.width) {
                        continue;
                    }
                    if (dx * dx + dy * dy > r_sq) {
                        continue;
                    }
                    atomicMin(reinterpret_cast<unsigned long long*>(&packed_buffer[yy * params.width + xx]),
                              static_cast<unsigned long long>(packed));
                }
            }
        }

        __global__ void unpackKernel(const std::uint64_t* packed_buffer,
                                     int width, int height, int channels,
                                     float bg_r, float bg_g, float bg_b, float bg_a,
                                     bool transparent, float far_plane,
                                     float* image, float* depth) {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) {
                return;
            }
            const int idx = y * width + x;
            const int pixel_count = width * height;

            const std::uint64_t packed = packed_buffer[idx];
            if (packed == kEmptyPacked) {
                image[0 * pixel_count + idx] = bg_r;
                image[1 * pixel_count + idx] = bg_g;
                image[2 * pixel_count + idx] = bg_b;
                if (channels == 4) {
                    image[3 * pixel_count + idx] = transparent ? 0.0f : bg_a;
                }
                depth[idx] = far_plane;
                return;
            }

            const std::uint32_t color_u = static_cast<std::uint32_t>(packed & 0xFFFFFFFFu);
            const std::uint32_t depth_u = static_cast<std::uint32_t>(packed >> 32);
            const float depth_v = __uint_as_float(depth_u);

            image[0 * pixel_count + idx] = static_cast<float>(color_u & 0xFFu) / 255.0f;
            image[1 * pixel_count + idx] = static_cast<float>((color_u >> 8) & 0xFFu) / 255.0f;
            image[2 * pixel_count + idx] = static_cast<float>((color_u >> 16) & 0xFFu) / 255.0f;
            if (channels == 4) {
                image[3 * pixel_count + idx] = 1.0f;
            }
            depth[idx] = depth_v;
        }

    } // namespace

    cudaError_t launchPointCloudRaster(const LaunchParams& params) {
        if (params.width <= 0 || params.height <= 0 || params.n_points == 0 ||
            (params.channels != 3 && params.channels != 4) ||
            !params.image || !params.depth) {
            return cudaErrorInvalidValue;
        }

        const std::size_t pixels = static_cast<std::size_t>(params.width) *
                                   static_cast<std::size_t>(params.height);
        std::uint64_t* packed = nullptr;
        cudaError_t status = cudaMallocAsync(reinterpret_cast<void**>(&packed),
                                             pixels * sizeof(std::uint64_t), params.stream);
        if (status != cudaSuccess) {
            return status;
        }

        const int clear_threads = 256;
        const int clear_blocks = static_cast<int>((pixels + clear_threads - 1) / clear_threads);
        clearPackedBuffer<<<clear_blocks, clear_threads, 0, params.stream>>>(
            packed, static_cast<int>(pixels));
        if ((status = cudaGetLastError()) != cudaSuccess) {
            cudaFreeAsync(packed, params.stream);
            return status;
        }

        const int raster_threads = 256;
        const int raster_blocks =
            static_cast<int>((params.n_points + raster_threads - 1) / raster_threads);
        rasterizePointsKernel<<<raster_blocks, raster_threads, 0, params.stream>>>(params, packed);
        if ((status = cudaGetLastError()) != cudaSuccess) {
            cudaFreeAsync(packed, params.stream);
            return status;
        }

        const dim3 unpack_threads(16, 16);
        const dim3 unpack_blocks((params.width + unpack_threads.x - 1) / unpack_threads.x,
                                 (params.height + unpack_threads.y - 1) / unpack_threads.y);
        unpackKernel<<<unpack_blocks, unpack_threads, 0, params.stream>>>(
            packed,
            params.width, params.height, params.channels,
            params.bg_r, params.bg_g, params.bg_b, params.bg_a,
            params.transparent_background, params.far_plane,
            params.image, params.depth);
        status = cudaGetLastError();
        cudaFreeAsync(packed, params.stream);
        return status;
    }

} // namespace lfs::rendering::pcraster
