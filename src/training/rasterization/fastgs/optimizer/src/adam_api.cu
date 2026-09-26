/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_api.h"
#include "adam_kernels.cuh"
#include "core/cuda_error.hpp"
#include "optimizer_config.h"
#include "utils.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace fast_lfs::optimizer {

    void adam_step_joint_contiguous_batched(
        const JointContiguousBatchEntry* host_entries,
        const int n_entries,
        const bool* frozen_mask,
        const int frozen_mask_size,
        const float frozen_lr_scale,
        const bool* crop_damping_mask,
        const int crop_damping_mask_size,
        const float cropbox_lr_scale,
        const float beta1,
        const float beta2,
        const float eps,
        cudaStream_t stream,
        const float* mean_step_scale_raw,
        const int mean_step_scale_n,
        const float mean_step_median_extent,
        const float mean_step_r_min,
        const float mean_step_r_max,
        const bool* mean_step_far_mask,
        const int mean_step_far_mask_n,
        const float* screen_share_max,
        const int screen_share_n,
        const float screen_share_limit,
        const float screen_share_penalty) {
        if (n_entries <= 0) {
            return;
        }
        if (n_entries > kMaxContiguousBatch) {
            throw std::runtime_error("adam_step_joint_contiguous_batched: too many entries");
        }
        int max_prims = 0;
        for (int i = 0; i < n_entries; ++i) {
            if (!host_entries[i].param || !host_entries[i].packed ||
                !host_entries[i].bounds || !host_entries[i].grad) {
                throw std::runtime_error("adam_step_joint_contiguous_batched: null entry");
            }
            if (host_entries[i].n_prims <= 0 || host_entries[i].n_attr <= 0) {
                throw std::runtime_error("adam_step_joint_contiguous_batched: n_prims/n_attr");
            }
            max_prims = std::max(max_prims, host_entries[i].n_prims);
        }
        if (mean_step_scale_raw != nullptr) {
            LFS_VALIDATE_CUDA_DEVICE_POINTER(mean_step_scale_raw, "mean_step_scale_raw");
        }
        if (mean_step_far_mask != nullptr) {
            LFS_VALIDATE_CUDA_DEVICE_POINTER(mean_step_far_mask, "mean_step_far_mask");
        }
        if (screen_share_max != nullptr) {
            LFS_VALIDATE_CUDA_DEVICE_POINTER(screen_share_max, "screen_share_max");
        }
        kernels::adam::JointContiguousBatch batch{};
        std::copy_n(host_entries, n_entries, batch.entries);
        constexpr int kBS = 256;
        const int n_blocks = (max_prims + kBS - 1) / kBS;
        const dim3 grid(n_blocks, n_entries);
        kernels::adam::adam_step_joint_contiguous_batched_cu<16><<<grid, kBS, 0, stream>>>(
            batch,
            frozen_mask, frozen_mask_size, frozen_lr_scale,
            crop_damping_mask, crop_damping_mask_size, cropbox_lr_scale,
            beta1, beta2, eps,
            mean_step_scale_raw, mean_step_scale_n, mean_step_median_extent,
            mean_step_r_min, mean_step_r_max, mean_step_far_mask, mean_step_far_mask_n,
            screen_share_max, screen_share_n, screen_share_limit, screen_share_penalty);
        LFS_CUDA_LAUNCH_CHECK(stream, "adam_step_joint_contiguous_batched");
    }

    void joint_encode_zero_rows_at_indices(
        std::uint8_t* packed,
        float* bounds,
        const int64_t* indices_device,
        const int n_indices,
        const int n_attr,
        const int bits,
        const int n_prims,
        cudaStream_t stream) {
        LFS_VALIDATE_CUDA_DEVICE_POINTER(packed, "packed");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(bounds, "bounds");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(indices_device, "indices_device");
        if (n_indices <= 0)
            return;
        if (n_attr <= 0)
            throw std::runtime_error("n_attr must be positive");
        if (n_prims <= 0)
            throw std::runtime_error("joint_encode_zero_rows: n_prims must be positive");
        if (bits != 8 && bits != 16)
            throw std::runtime_error("joint_encode_zero_rows: bits must be 8 or 16");

        constexpr int kBS = lfs::training::joint_adam::kBlockSizeDevice;
        const int n_blocks = (n_prims + kBS - 1) / kBS;
        const std::size_t touched_bytes =
            (static_cast<std::size_t>(n_blocks) + 3u) & ~static_cast<std::size_t>(3u);
        std::uint8_t* flags = nullptr;
        std::uint8_t* block_touched = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&flags, static_cast<std::size_t>(n_prims), stream));
        LFS_CUDA_CHECK(cudaMallocAsync(&block_touched, touched_bytes, stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(flags, 0, static_cast<std::size_t>(n_prims), stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(block_touched, 0, touched_bytes, stream));

        const dim3 mark_grid(div_round_up(n_indices, config::block_size_adam_step));
        const dim3 mark_block(config::block_size_adam_step);
        kernels::adam::joint_encode_zero_mark_cu<<<mark_grid, mark_block, 0, stream>>>(
            flags, block_touched, indices_device, n_indices, n_prims);
        LFS_CUDA_LAUNCH_CHECK(stream, "joint_encode_zero_mark");

        if (bits == 16) {
            kernels::adam::joint_encode_zero_rows_cu<16><<<n_blocks, kBS, 0, stream>>>(
                packed, bounds, flags, block_touched, n_attr, n_prims);
        } else {
            kernels::adam::joint_encode_zero_rows_cu<8><<<n_blocks, kBS, 0, stream>>>(
                packed, bounds, flags, block_touched, n_attr, n_prims);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "joint_encode_zero_rows_at_indices");
        LFS_CUDA_CHECK(cudaFreeAsync(flags, stream));
        LFS_CUDA_CHECK(cudaFreeAsync(block_touched, stream));
    }

    void joint_encode_zero_shN_at_indices(
        std::uint8_t* packed,
        float* bounds,
        const int64_t* indices_device,
        const int n_indices,
        const int slots_per_primitive,
        const int bits,
        const int n_prims,
        cudaStream_t stream) {
        LFS_VALIDATE_CUDA_DEVICE_POINTER(packed, "packed");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(bounds, "bounds");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(indices_device, "indices_device");
        if (n_indices <= 0)
            return;
        if (slots_per_primitive <= 0)
            return;
        if (n_prims <= 0)
            throw std::runtime_error("joint_encode_zero_shN: n_prims must be positive");
        if (bits != 8 && bits != 16)
            throw std::runtime_error("joint_encode_zero_shN: bits must be 8 or 16");

        constexpr int kBS = lfs::training::joint_adam::kBlockSizeDevice;
        const int n_blocks = (n_prims + kBS - 1) / kBS;
        const std::size_t touched_bytes =
            (static_cast<std::size_t>(n_blocks) + 3u) & ~static_cast<std::size_t>(3u);
        std::uint8_t* flags = nullptr;
        std::uint8_t* block_touched = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&flags, static_cast<std::size_t>(n_prims), stream));
        LFS_CUDA_CHECK(cudaMallocAsync(&block_touched, touched_bytes, stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(flags, 0, static_cast<std::size_t>(n_prims), stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(block_touched, 0, touched_bytes, stream));

        const dim3 mark_grid(div_round_up(n_indices, config::block_size_adam_step));
        const dim3 mark_block(config::block_size_adam_step);
        kernels::adam::joint_encode_zero_mark_cu<<<mark_grid, mark_block, 0, stream>>>(
            flags, block_touched, indices_device, n_indices, n_prims);
        LFS_CUDA_LAUNCH_CHECK(stream, "joint_encode_zero_mark");

        if (bits == 16) {
            kernels::adam::joint_encode_zero_shN_cu<16><<<n_blocks, kBS, 0, stream>>>(
                packed, bounds, flags, block_touched, slots_per_primitive, n_prims);
        } else {
            kernels::adam::joint_encode_zero_shN_cu<8><<<n_blocks, kBS, 0, stream>>>(
                packed, bounds, flags, block_touched, slots_per_primitive, n_prims);
        }
        LFS_CUDA_LAUNCH_CHECK(stream, "joint_encode_zero_shN_at_indices");
        LFS_CUDA_CHECK(cudaFreeAsync(flags, stream));
        LFS_CUDA_CHECK(cudaFreeAsync(block_touched, stream));
    }

} // namespace fast_lfs::optimizer
