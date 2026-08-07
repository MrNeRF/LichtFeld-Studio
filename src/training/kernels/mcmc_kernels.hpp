/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::training::mcmc {

    /**
     * Initialize relocation coefficients in __constant__ memory.
     * Must be called once before first use of launch_relocation_kernel.
     * Pre-computes C(n,k) * (-1)^k * rsqrt(k+1) for all n,k < n_max.
     *
     * @param n_max - Maximum ratio value (must be <= 51)
     */
    void init_relocation_coefficients(int n_max);

    /**
     * Relocation kernel - Equation (9) from "3D Gaussian Splatting as Markov Chain Monte Carlo"
     *
     * Computes new opacities and scales for relocated Gaussians based on their sampling ratios.
     * Requires init_relocation_coefficients() to have been called first.
     *
     * @param opacities [N] - Original opacity values
     * @param scales [N, 3] - Original scale values
     * @param ratios [N] - Number of times each Gaussian was sampled (int32)
     * @param new_opacities [N] - Output: relocated opacity values
     * @param new_scales [N, 3] - Output: relocated scale values
     * @param N - Number of Gaussians
     * @param stream - CUDA stream for async execution
     */
    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        float min_opacity,
        float* new_opacities,
        float* new_scales,
        size_t N,
        void* stream = nullptr);

    /**
     * Add noise kernel - Injects position noise scaled by covariance
     *
     * Adds Gaussian noise to mean positions, scaled by the Gaussian covariance
     * and learning rate. Used for MCMC exploration.
     *
     * @param raw_opacities [N] - Raw (pre-sigmoid) opacity values
     * @param raw_scales [N, 3] - Raw (pre-exp) scale values
     * @param raw_quats [N, 4] - Raw quaternion rotation values
     * @param noise [N, 3] - Random noise from N(0,1)
     * @param means [N, 3] - Mean positions (modified in-place)
     * @param frozen_mask [N] - Optional mask of rows that must not be modified
     * @param frozen_mask_size - Number of entries in frozen_mask
     * @param current_lr - Current learning rate for noise scaling
     * @param N - Number of Gaussians
     * @param stream - CUDA stream for async execution
     */
    void launch_add_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        const float* noise,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N,
        void* stream = nullptr);

    /**
     * Fused scatter kernel - Copy multiple parameters from src to dst indices
     *
     * Copies parameters from sampled indices to dead indices in a single kernel.
     * Replaces 12 separate kernel launches (6 index_select + 6 index_put_).
     *
     * @param src_indices [n_copy] - Source indices to read from (int64)
     * @param dst_indices [n_copy] - Destination indices to write to (int64)
     * @param means [N, 3] - Mean positions (modified in-place)
     * @param sh0 [N, 1, 3] - SH0 coefficients (modified in-place)
     * @param shN [N, sh_rest, 3] - SH rest coefficients (modified in-place)
     * @param scales [N, 3] - Scales (modified in-place)
     * @param rotations [N, 4] - Rotations (modified in-place)
     * @param opacities [N, 1] or [N] - Opacities (modified in-place)
     * @param n_copy - Number of copies to perform
     * @param sh_rest - Number of SH rest coefficients
     * @param opacity_dim - Opacity dimension (1 for [N,1], 0 for [N])
     * @param stream - CUDA stream
     */
    void launch_copy_gaussian_params(
        const int64_t* src_indices,
        const int64_t* dst_indices,
        float* means,
        float* sh0,
        float* shN,
        float* scales,
        float* rotations,
        float* opacities,
        size_t n_copy,
        size_t sh_rest,
        int opacity_dim,
        size_t N, // Add N parameter for bounds checking
        void* stream = nullptr);

    /**
     * Update scaling and opacity at specific indices (preserves tensor capacity)
     *
     * Updates scaling and opacity values at specified indices without reallocating tensors.
     * Replaces index_put_() which creates new tensors and loses pre-allocated capacity.
     *
     * @param indices [n_indices] - Indices to update (int64)
     * @param new_scaling [n_indices, 3] - New scaling values
     * @param new_opacity_raw [n_indices] or [n_indices, 1] - New opacity values
     * @param scaling_raw [N, 3] - Scaling tensor (modified in-place)
     * @param opacity_raw [N] or [N, 1] - Opacity tensor (modified in-place)
     * @param n_indices - Number of indices to update
     * @param opacity_dim - Opacity dimension (1 for [N,1], 0 for [N])
     * @param N - Total number of Gaussians (for bounds checking)
     * @param stream - CUDA stream
     */
    void launch_update_scaling_opacity(
        const int64_t* indices,
        const float* new_scaling,
        const float* new_opacity_raw,
        float* scaling_raw,
        float* opacity_raw,
        size_t n_indices,
        int opacity_dim,
        size_t N,
        void* stream = nullptr);

    /**
     * Fused multinomial sampling + gather kernel (no intermediate allocations)
     *
     * Performs multinomial sampling from sampling_weights[alive_indices] and directly gathers
     * the sampled opacities and scales in a single kernel launch.
     *
     * Algorithm:
     * 1. Thread block cooperatively computes cumulative sum of probabilities
     * 2. Each thread generates random samples and performs binary search
     * 3. Maps local indices → global indices (alive_indices[local_idx])
     * 4. Directly gathers opacities and scales at sampled indices
     *
     * @param sampling_weights [N] - Sampling probabilities (full array)
     * @param opacities [N] - Source opacities used for relocation math
     * @param scaling_raw [N, 3] - Source raw scales (exp() applied inline)
     * @param alive_indices [n_alive] - Indices of alive Gaussians
     * @param n_alive - Number of alive Gaussians
     * @param n_samples - Number of samples to draw
     * @param seed - Random seed for sampling
     * @param sampled_global_indices [n_samples] - Output: sampled global indices
     * @param sampled_opacities [n_samples] - Output: gathered opacities
     * @param sampled_scales [n_samples, 3] - Output: gathered scales
     * @param N - Total number of Gaussians (for bounds checking)
     * @param stream - CUDA stream
     */
    void launch_multinomial_sample_and_gather(
        const float* sampling_weights,
        const float* opacities,
        const float* scaling_raw,
        const int64_t* alive_indices,
        size_t n_alive,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_global_indices,
        float* sampled_opacities,
        float* sampled_scales,
        size_t N,
        void* stream = nullptr);

    /**
     * Fused multinomial sampling from all sampling weights (no intermediate allocations)
     *
     * Variant that samples from all N weights (not just a subset).
     * Used in add_new_gs() where we sample from the full population.
     *
     * @param sampling_weights [N] - Sampling probabilities (full array)
     * @param opacities [N] - Source opacities used for relocation math
     * @param scaling_raw [N, 3] - Source raw scales (exp() applied inline)
     * @param N - Number of Gaussians
     * @param n_samples - Number of samples to draw
     * @param seed - Random seed for sampling
     * @param sampled_indices [n_samples] - Output: sampled indices
     * @param sampled_opacities [n_samples] - Output: gathered opacities
     * @param sampled_scales [n_samples, 3] - Output: gathered scales
     * @param stream - CUDA stream
     */
    void launch_multinomial_sample_all(
        const float* sampling_weights,
        const float* opacities,
        const float* scaling_raw,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_indices,
        float* sampled_opacities,
        float* sampled_scales,
        void* stream = nullptr);

    /**
     * In-place element-wise maximum: a[i] = max(a[i], b[i])
     *
     * Replaces allocating maximum() call that creates a new tensor every iteration.
     *
     * @param a [N] - First tensor, modified in-place
     * @param b [N] - Second tensor
     * @param N - Number of elements
     * @param stream - CUDA stream
     */
    void launch_elementwise_max_inplace(
        float* a,
        const float* b,
        size_t N,
        void* stream = nullptr);

} // namespace lfs::training::mcmc
