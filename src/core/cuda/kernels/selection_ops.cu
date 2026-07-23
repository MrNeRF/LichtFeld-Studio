/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/selection_ops.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include <cassert>
#include <cfloat>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace lfs::core::cuda {

    namespace {

        constexpr int BLOCK_SIZE = 256;

        __device__ inline int3 pos_to_cell(float3 pos, float3 grid_min, float inv_cell_size) {
            return make_int3(
                static_cast<int>(floorf((pos.x - grid_min.x) * inv_cell_size)),
                static_cast<int>(floorf((pos.y - grid_min.y) * inv_cell_size)),
                static_cast<int>(floorf((pos.z - grid_min.z) * inv_cell_size)));
        }

        __device__ inline int cell_to_hash(int3 cell, int3 dims) {
            if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
                cell.x >= dims.x || cell.y >= dims.y || cell.z >= dims.z)
                return -1;
            return cell.x + cell.y * dims.x + cell.z * dims.x * dims.y;
        }

        __global__ void compute_cell_ids(
            const float* __restrict__ positions,
            int* __restrict__ cell_ids,
            float3 grid_min, float inv_cell_size, int3 grid_dims,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            float3 p = make_float3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
            int3 cell = pos_to_cell(p, grid_min, inv_cell_size);
            cell_ids[idx] = cell_to_hash(cell, grid_dims);
        }

        __global__ void find_cell_starts(
            const int* __restrict__ sorted_cell_ids,
            int* __restrict__ cell_start,
            int* __restrict__ cell_end,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            const int cell = sorted_cell_ids[idx];
            if (cell < 0)
                return;

            if (idx == 0 || sorted_cell_ids[idx - 1] != cell)
                cell_start[cell] = idx;
            if (idx == N - 1 || sorted_cell_ids[idx + 1] != cell)
                cell_end[cell] = idx + 1;
        }

        __global__ void grow_kernel(
            const float* __restrict__ positions,
            const uint8_t* __restrict__ mask,
            uint8_t* __restrict__ out_mask,
            const int* __restrict__ sorted_indices,
            const int* __restrict__ cell_start,
            const int* __restrict__ cell_end,
            float3 grid_min, float inv_cell_size, int3 grid_dims,
            float radius_sq, uint8_t group_id,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            // Already selected — keep
            if (mask[idx] > 0) {
                out_mask[idx] = mask[idx];
                return;
            }

            float3 p = make_float3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
            int3 center_cell = pos_to_cell(p, grid_min, inv_cell_size);

            // Check 27 neighbor cells
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int3 nc = make_int3(center_cell.x + dx, center_cell.y + dy, center_cell.z + dz);
                        int hash = cell_to_hash(nc, grid_dims);
                        if (hash < 0)
                            continue;

                        int start = cell_start[hash];
                        int end = cell_end[hash];
                        if (start < 0)
                            continue;

                        for (int j = start; j < end; ++j) {
                            int other = sorted_indices[j];
                            if (mask[other] == 0)
                                continue;

                            float dist_x = positions[other * 3] - p.x;
                            float dist_y = positions[other * 3 + 1] - p.y;
                            float dist_z = positions[other * 3 + 2] - p.z;
                            if (dist_x * dist_x + dist_y * dist_y + dist_z * dist_z <= radius_sq) {
                                out_mask[idx] = group_id;
                                return;
                            }
                        }
                    }
                }
            }
        }

        __global__ void shrink_kernel(
            const float* __restrict__ positions,
            const uint8_t* __restrict__ mask,
            uint8_t* __restrict__ out_mask,
            const int* __restrict__ sorted_indices,
            const int* __restrict__ cell_start,
            const int* __restrict__ cell_end,
            float3 grid_min, float inv_cell_size, int3 grid_dims,
            float radius_sq,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            // Unselected — stays unselected
            if (mask[idx] == 0)
                return;

            float3 p = make_float3(positions[idx * 3], positions[idx * 3 + 1], positions[idx * 3 + 2]);
            int3 center_cell = pos_to_cell(p, grid_min, inv_cell_size);

            // If any unselected neighbor within radius → deselect (boundary erosion)
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int3 nc = make_int3(center_cell.x + dx, center_cell.y + dy, center_cell.z + dz);
                        int hash = cell_to_hash(nc, grid_dims);
                        if (hash < 0)
                            continue;

                        int start = cell_start[hash];
                        int end = cell_end[hash];
                        if (start < 0)
                            continue;

                        for (int j = start; j < end; ++j) {
                            int other = sorted_indices[j];
                            if (mask[other] > 0)
                                continue;

                            float dist_x = positions[other * 3] - p.x;
                            float dist_y = positions[other * 3 + 1] - p.y;
                            float dist_z = positions[other * 3 + 2] - p.z;
                            if (dist_x * dist_x + dist_y * dist_y + dist_z * dist_z <= radius_sq) {
                                out_mask[idx] = 0;
                                return;
                            }
                        }
                    }
                }
            }

            out_mask[idx] = mask[idx];
        }

        __global__ void opacity_threshold_kernel(
            const float* __restrict__ opacity_raw,
            uint8_t* __restrict__ out_mask,
            float min_opacity, float max_opacity, uint8_t group_id,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            const float activated = 1.0f / (1.0f + expf(-opacity_raw[idx]));
            out_mask[idx] = (activated >= min_opacity && activated <= max_opacity) ? group_id : 0;
        }

        __global__ void scale_threshold_kernel(
            const float* __restrict__ scale_raw,
            uint8_t* __restrict__ out_mask,
            float max_scale, uint8_t group_id,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            const float sx = expf(scale_raw[idx * 3]);
            const float sy = expf(scale_raw[idx * 3 + 1]);
            const float sz = expf(scale_raw[idx * 3 + 2]);
            const float max_s = fmaxf(sx, fmaxf(sy, sz));
            out_mask[idx] = (max_s <= max_scale) ? group_id : 0;
        }

        // SH_C0 constant for decoding DC spherical harmonic component to RGB
        constexpr float SH_C0 = 0.28209479177387814f;

        __device__ inline float decode_sh_color(float sh_val) {
            return fminf(1.0f, fmaxf(0.0f, 0.5f + sh_val * SH_C0));
        }

        __global__ void color_threshold_kernel(
            const float* __restrict__ sh0,
            uint8_t* __restrict__ out_mask,
            float ref_r, float ref_g, float ref_b,
            float threshold, uint8_t group_id,
            int N) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N)
                return;

            // SH0 is stored as [N, 1, 3] or [N, 3] — stride is always 3 floats per gaussian
            const float r = decode_sh_color(sh0[idx * 3]);
            const float g = decode_sh_color(sh0[idx * 3 + 1]);
            const float b = decode_sh_color(sh0[idx * 3 + 2]);

            const bool match = fabsf(r - ref_r) <= threshold &&
                               fabsf(g - ref_g) <= threshold &&
                               fabsf(b - ref_b) <= threshold;
            out_mask[idx] = match ? group_id : 0;
        }

        __device__ inline void atomicMinFloat(float* addr, float val) {
            int* addr_as_int = reinterpret_cast<int*>(addr);
            int old = *addr_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(addr_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
            } while (assumed != old);
        }

        __device__ inline void atomicMaxFloat(float* addr, float val) {
            int* addr_as_int = reinterpret_cast<int*>(addr);
            int old = *addr_as_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
            } while (assumed != old);
        }

        __global__ void compute_aabb_kernel(
            const float* __restrict__ positions,
            float* __restrict__ aabb,
            int N) {
            __shared__ float s_min[3][BLOCK_SIZE];
            __shared__ float s_max[3][BLOCK_SIZE];

            const int tid = threadIdx.x;
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;

            for (int c = 0; c < 3; ++c) {
                s_min[c][tid] = FLT_MAX;
                s_max[c][tid] = -FLT_MAX;
            }

            if (idx < N) {
                for (int c = 0; c < 3; ++c) {
                    float v = positions[idx * 3 + c];
                    s_min[c][tid] = v;
                    s_max[c][tid] = v;
                }
            }
            __syncthreads();

            for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    for (int c = 0; c < 3; ++c) {
                        s_min[c][tid] = fminf(s_min[c][tid], s_min[c][tid + s]);
                        s_max[c][tid] = fmaxf(s_max[c][tid], s_max[c][tid + s]);
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                for (int c = 0; c < 3; ++c) {
                    atomicMinFloat(&aabb[c], s_min[c][0]);
                    atomicMaxFloat(&aabb[3 + c], s_max[c][0]);
                }
            }
        }

        struct SpatialGrid {
            Tensor sorted_indices;
            Tensor cell_start;
            Tensor cell_end;
            float3 grid_min;
            float inv_cell_size;
            int3 grid_dims;
            int num_cells;
        };

        SpatialGrid build_grid(const Tensor& means, float cell_size, cudaStream_t stream) {
            const int N = static_cast<int>(means.size(0));
            const float* pos_ptr = means.ptr<float>();

            // Compute AABB via parallel reduction
            // aabb[0..2] = min_xyz, aabb[3..5] = max_xyz
            auto aabb_buf = Tensor::empty({6}, Device::CUDA, DataType::Float32);
            {
                float init[6] = {FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpyAsync(aabb_buf.ptr<float>(), init, 6 * sizeof(float), cudaMemcpyHostToDevice, stream),
                    "build_grid: AABB init upload");
            }

            int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const auto aabb_ticket = ::lfs::core::cuda_record_range(stream, "core.selection.build_grid.aabb_pipeline");
            compute_aabb_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(pos_ptr, aabb_buf.ptr<float>(), N);
            LFS_CUDA_LAUNCH_CHECK(stream, "core.selection.build_grid.compute_aabb");

            float aabb_host[6];
            LFS_CUDA_CHECK_MSG(
                cudaMemcpyAsync(aabb_host, aabb_buf.ptr<float>(), 6 * sizeof(float), cudaMemcpyDeviceToHost, stream),
                "build_grid: AABB readback");
            LFS_CUDA_AWAIT(aabb_ticket, cudaStreamSynchronize(stream), "core.selection.build_grid.aabb_readback");

            float3 grid_min = make_float3(aabb_host[0] - cell_size, aabb_host[1] - cell_size, aabb_host[2] - cell_size);
            float3 grid_max = make_float3(aabb_host[3] + cell_size, aabb_host[4] + cell_size, aabb_host[5] + cell_size);
            float inv_cell_size = 1.0f / cell_size;

            int3 grid_dims = make_int3(
                static_cast<int>(ceilf((grid_max.x - grid_min.x) * inv_cell_size)),
                static_cast<int>(ceilf((grid_max.y - grid_min.y) * inv_cell_size)),
                static_cast<int>(ceilf((grid_max.z - grid_min.z) * inv_cell_size)));

            // Clamp grid to reasonable size
            constexpr int MAX_DIM = 1024;
            grid_dims.x = std::min(grid_dims.x, MAX_DIM);
            grid_dims.y = std::min(grid_dims.y, MAX_DIM);
            grid_dims.z = std::min(grid_dims.z, MAX_DIM);

            const int64_t total = static_cast<int64_t>(grid_dims.x) * grid_dims.y * grid_dims.z;
            assert(total <= INT_MAX && "Grid cell count overflows int32");
            int num_cells = static_cast<int>(total);

            // Compute cell IDs
            auto cell_ids = Tensor::empty({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);
            auto sorted_indices = Tensor::empty({static_cast<size_t>(N)}, Device::CUDA, DataType::Int32);

            compute_cell_ids<<<blocks, BLOCK_SIZE, 0, stream>>>(
                pos_ptr, cell_ids.ptr<int>(),
                grid_min, inv_cell_size, grid_dims, N);
            LFS_CUDA_LAUNCH_CHECK(stream, "core.selection.build_grid.compute_cell_ids");

            // Initialize sorted indices
            thrust::device_ptr<int> si_ptr(sorted_indices.ptr<int>());
            thrust::sequence(thrust::cuda::par.on(stream), si_ptr, si_ptr + N);

            // Sort by cell ID
            thrust::device_ptr<int> ci_ptr(cell_ids.ptr<int>());
            thrust::sort_by_key(thrust::cuda::par.on(stream), ci_ptr, ci_ptr + N, si_ptr);

            // Build cell start/end
            auto cell_start = Tensor::empty({static_cast<size_t>(num_cells)}, Device::CUDA, DataType::Int32);
            auto cell_end = Tensor::empty({static_cast<size_t>(num_cells)}, Device::CUDA, DataType::Int32);
            thrust::fill(thrust::cuda::par.on(stream),
                         thrust::device_ptr<int>(cell_start.ptr<int>()),
                         thrust::device_ptr<int>(cell_start.ptr<int>()) + num_cells, -1);
            thrust::fill(thrust::cuda::par.on(stream),
                         thrust::device_ptr<int>(cell_end.ptr<int>()),
                         thrust::device_ptr<int>(cell_end.ptr<int>()) + num_cells, -1);

            find_cell_starts<<<blocks, BLOCK_SIZE, 0, stream>>>(
                cell_ids.ptr<int>(), cell_start.ptr<int>(), cell_end.ptr<int>(), N);
            LFS_CUDA_LAUNCH_CHECK(stream, "core.selection.build_grid.find_cell_starts");

            return SpatialGrid{
                std::move(sorted_indices),
                std::move(cell_start),
                std::move(cell_end),
                grid_min,
                inv_cell_size,
                grid_dims,
                num_cells};
        }

    } // namespace

    Tensor selection_grow(const Tensor& mask, const Tensor& means, float radius, uint8_t group_id) {
        assert(mask.device() == Device::CUDA);
        assert(means.device() == Device::CUDA);
        assert(mask.dtype() == DataType::UInt8);
        assert(means.dtype() == DataType::Float32);
        assert(mask.numel() == means.size(0));
        assert(means.ndim() == 2 && means.size(1) == 3);
        assert(radius > 0.0f);

        nvtxRangePush("selection_grow");

        const int N = static_cast<int>(mask.numel());
        if (N == 0) {
            nvtxRangePop();
            return mask.clone();
        }

        cudaStream_t stream = mask.stream();
        auto grid = build_grid(means, radius, stream);

        auto out_mask = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::UInt8);

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grow_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            means.ptr<float>(), mask.ptr<uint8_t>(), out_mask.ptr<uint8_t>(),
            grid.sorted_indices.ptr<int>(), grid.cell_start.ptr<int>(), grid.cell_end.ptr<int>(),
            grid.grid_min, grid.inv_cell_size, grid.grid_dims,
            radius * radius, group_id, N);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.selection.grow");

        nvtxRangePop();
        return out_mask;
    }

    Tensor selection_shrink(const Tensor& mask, const Tensor& means, float radius) {
        assert(mask.device() == Device::CUDA);
        assert(means.device() == Device::CUDA);
        assert(mask.dtype() == DataType::UInt8);
        assert(means.dtype() == DataType::Float32);
        assert(mask.numel() == means.size(0));
        assert(means.ndim() == 2 && means.size(1) == 3);
        assert(radius > 0.0f);

        nvtxRangePush("selection_shrink");

        const int N = static_cast<int>(mask.numel());
        if (N == 0) {
            nvtxRangePop();
            return mask.clone();
        }

        cudaStream_t stream = mask.stream();
        auto grid = build_grid(means, radius, stream);

        auto out_mask = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA, DataType::UInt8);

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        shrink_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            means.ptr<float>(), mask.ptr<uint8_t>(), out_mask.ptr<uint8_t>(),
            grid.sorted_indices.ptr<int>(), grid.cell_start.ptr<int>(), grid.cell_end.ptr<int>(),
            grid.grid_min, grid.inv_cell_size, grid.grid_dims,
            radius * radius, N);
        LFS_CUDA_LAUNCH_CHECK(stream, "core.selection.shrink");

        nvtxRangePop();
        return out_mask;
    }

    Tensor select_by_opacity(const Tensor& opacity_raw, float min_opacity, float max_opacity, uint8_t group_id) {
        assert(opacity_raw.device() == Device::CUDA);
        assert(opacity_raw.dtype() == DataType::Float32);

        nvtxRangePush("select_by_opacity");

        const int N = static_cast<int>(opacity_raw.numel());
        if (N == 0) {
            nvtxRangePop();
            return Tensor::empty({0}, Device::CUDA, DataType::UInt8);
        }

        auto out_mask = Tensor::empty({static_cast<size_t>(N)}, Device::CUDA, DataType::UInt8);

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        opacity_threshold_kernel<<<blocks, BLOCK_SIZE, 0, opacity_raw.stream()>>>(
            opacity_raw.ptr<float>(), out_mask.ptr<uint8_t>(),
            min_opacity, max_opacity, group_id, N);
        LFS_CUDA_LAUNCH_CHECK(opacity_raw.stream(), "core.selection.opacity_threshold");

        nvtxRangePop();
        return out_mask;
    }

    Tensor select_by_scale(const Tensor& scale_raw, float max_scale, uint8_t group_id) {
        assert(scale_raw.device() == Device::CUDA);
        assert(scale_raw.dtype() == DataType::Float32);
        assert(scale_raw.ndim() == 2 && scale_raw.size(1) == 3);

        nvtxRangePush("select_by_scale");

        const int N = static_cast<int>(scale_raw.size(0));
        if (N == 0) {
            nvtxRangePop();
            return Tensor::empty({0}, Device::CUDA, DataType::UInt8);
        }

        auto out_mask = Tensor::empty({static_cast<size_t>(N)}, Device::CUDA, DataType::UInt8);

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_threshold_kernel<<<blocks, BLOCK_SIZE, 0, scale_raw.stream()>>>(
            scale_raw.ptr<float>(), out_mask.ptr<uint8_t>(),
            max_scale, group_id, N);
        LFS_CUDA_LAUNCH_CHECK(scale_raw.stream(), "core.selection.scale_threshold");

        nvtxRangePop();
        return out_mask;
    }

    Tensor select_by_color(const Tensor& sh0,
                           float ref_r, float ref_g, float ref_b,
                           float threshold, uint8_t group_id) {
        assert(sh0.device() == Device::CUDA);
        assert(sh0.dtype() == DataType::Float32);

        nvtxRangePush("select_by_color");

        // Support both [N, 1, 3] and [N, 3] layouts
        const int N = static_cast<int>(sh0.size(0));
        if (N == 0) {
            nvtxRangePop();
            return Tensor::empty({0}, Device::CUDA, DataType::UInt8);
        }

        auto out_mask = Tensor::empty({static_cast<size_t>(N)}, Device::CUDA, DataType::UInt8);

        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        color_threshold_kernel<<<blocks, BLOCK_SIZE, 0, sh0.stream()>>>(
            sh0.ptr<float>(), out_mask.ptr<uint8_t>(),
            ref_r, ref_g, ref_b, threshold, group_id, N);
        LFS_CUDA_LAUNCH_CHECK(sh0.stream(), "core.selection.color_threshold");

        nvtxRangePop();
        return out_mask;
    }

} // namespace lfs::core::cuda
