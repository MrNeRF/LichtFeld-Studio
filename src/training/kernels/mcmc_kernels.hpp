/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cstdint>

namespace lfs::training::mcmc {

    /**
     * Relocation kernel - Equation (9) from "3D Gaussian Splatting as Markov Chain Monte Carlo"
     *
     * Computes new opacities and scales for relocated Gaussians based on their sampling ratios.
     *
     * @param opacities [N] - Original opacity values
     * @param scales [N, 3] - Original scale values
     * @param ratios [N] - Number of times each Gaussian was sampled (int32)
     * @param binoms [n_max, n_max] - Precomputed binomial coefficients
     * @param n_max - Maximum ratio value (size of binomial table)
     * @param new_opacities [N] - Output: relocated opacity values
     * @param new_scales [N, 3] - Output: relocated scale values
     * @param N - Number of Gaussians
     * @param stream - CUDA stream for async execution
     */
    void launch_relocation_kernel(
        const float* opacities,
        const float* scales,
        const int32_t* ratios,
        const float* binoms,
        int n_max,
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
        float current_lr,
        size_t N,
        void* stream = nullptr);

    /**
     * Fused gather kernel - Collect multiple parameters at specified indices
     *
     * Gathers parameters for multiple Gaussians in a single kernel launch.
     * Replaces multiple separate index_select operations.
     *
     * @param indices [n_samples] - Indices to gather from (int64)
     * @param src_means [N, 3] - Source mean positions
     * @param src_sh0 [N, 1, 3] - Source SH0 coefficients
     * @param src_shN [N, sh_rest, 3] - Source SH rest coefficients
     * @param src_scales [N, 3] - Source scales
     * @param src_rotations [N, 4] - Source rotations
     * @param src_opacities [N, 1] or [N] - Source opacities
     * @param dst_means [n_samples, 3] - Output means
     * @param dst_sh0 [n_samples, 1, 3] - Output SH0
     * @param dst_shN [n_samples, sh_rest, 3] - Output SH rest
     * @param dst_scales [n_samples, 3] - Output scales
     * @param dst_rotations [n_samples, 4] - Output rotations
     * @param dst_opacities [n_samples, 1] or [n_samples] - Output opacities
     * @param n_samples - Number of samples to gather
     * @param sh_rest - Number of SH rest coefficients
     * @param opacity_dim - Opacity dimension (1 for [N,1], 0 for [N])
     * @param stream - CUDA stream
     */
    void launch_gather_gaussian_params(
        const int64_t* indices,
        const float* src_means,
        const float* src_sh0,
        const float* src_shN,
        const float* src_scales,
        const float* src_rotations,
        const float* src_opacities,
        float* dst_means,
        float* dst_sh0,
        float* dst_shN,
        float* dst_scales,
        float* dst_rotations,
        float* dst_opacities,
        size_t n_samples,
        size_t sh_rest,
        int opacity_dim,
        size_t N, // Add N parameter for bounds checking
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
     * Histogram kernel - Count occurrences of indices
     *
     * Computes a histogram of index occurrences using atomic operations.
     * Used in relocate_count_occurrences and add_new_count_occurrences to replace
     * the index_add_ operation which is slow for scattered indices.
     *
     * @param indices [n_samples] - Input indices (int64)
     * @param counts [N] - Output counts (int32), should be zero-initialized
     * @param n_samples - Number of indices
     * @param N - Maximum index value + 1 (for bounds checking)
     * @param stream - CUDA stream
     */
    void launch_histogram(
        const int64_t* indices,
        int32_t* counts,
        size_t n_samples,
        size_t N,
        void* stream = nullptr);

    /**
     * Fast histogram using sort + run-length encoding
     *
     * Much faster than allocating N-sized array when n_samples << N.
     * Uses O(n) memory instead of O(N).
     *
     * Algorithm:
     * 1. Create (index, position) pairs
     * 2. Sort by index
     * 3. Count runs of identical indices
     * 4. Scatter counts back to original positions
     *
     * @param indices [n_samples] - Input indices (int64)
     * @param output_counts [n_samples] - Output: count for each input index
     * @param n_samples - Number of samples
     * @param stream - CUDA stream
     */
    void launch_histogram_sort(
        const int64_t* indices,
        int32_t* output_counts,
        size_t n_samples,
        void* stream = nullptr);

    /**
     * Fused gather kernel for 2 tensors - Replaces two index_select calls
     *
     * Gathers values from two source tensors at specified indices in a single kernel.
     * Used in relocate_get_sampled_params to replace separate index_select calls
     * for opacities and scales.
     *
     * @param indices [n_samples] - Indices to gather from (int64)
     * @param src_a [N, dim_a] - First source tensor
     * @param src_b [N, dim_b] - Second source tensor
     * @param dst_a [n_samples, dim_a] - First output tensor
     * @param dst_b [n_samples, dim_b] - Second output tensor
     * @param n_samples - Number of samples to gather
     * @param dim_a - Dimension of first tensor
     * @param dim_b - Dimension of second tensor
     * @param N - Source tensor size (for bounds checking)
     * @param stream - CUDA stream
     */
    void launch_gather_2tensors(
        const int64_t* indices,
        const float* src_a,
        const float* src_b,
        float* dst_a,
        float* dst_b,
        size_t n_samples,
        size_t dim_a,
        size_t dim_b,
        size_t N,
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
     * Fused kernel: Compute raw opacity and scaling values (ZERO intermediate allocations)
     *
     * Performs: clamp(opacity) -> inverse_sigmoid -> log -> optional unsqueeze
     *           log(scales)
     *
     * Replaces multiple tensor operations with a single fused kernel.
     *
     * @param opacities [n] - Input opacity values (post-relocation)
     * @param scales [n, 3] - Input scale values (post-relocation)
     * @param opacity_raw [n] or [n, 1] - Output raw opacity (inverse sigmoid + log)
     * @param scaling_raw [n, 3] - Output raw scaling (log)
     * @param n - Number of values
     * @param min_opacity - Minimum opacity for clamping
     * @param opacity_dim - Output opacity dimension (1 for [n,1], 0 for [n])
     * @param stream - CUDA stream
     */
    void launch_compute_raw_values(
        const float* opacities,
        const float* scales,
        float* opacity_raw,
        float* scaling_raw,
        size_t n,
        float min_opacity,
        int opacity_dim,
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
     * Compute rotation magnitude squared (ZERO intermediate allocations)
     *
     * Computes ||q||^2 = q[0]^2 + q[1]^2 + q[2]^2 + q[3]^2 for each quaternion.
     * Replaces (rotation * rotation).sum(-1) which creates intermediate [N, 4] tensor.
     *
     * @param rotations [N, 4] - Input quaternion rotations
     * @param mag_sq [N] - Output: magnitude squared for each rotation
     * @param N - Number of Gaussians
     * @param stream - CUDA stream
     */
    void launch_compute_rotation_mag_sq(
        const float* rotations,
        float* mag_sq,
        size_t N,
        void* stream = nullptr);

    /**
     * Fused dead mask computation (ZERO intermediate allocations)
     *
     * Directly computes boolean dead mask from opacities and rotations in a single pass.
     * Replaces the two-step process of:
     *   1. Computing rot_mag_sq = (rotation * rotation).sum(-1)  [creates [N] intermediate]
     *   2. Computing dead_mask = (opacity <= min_opacity).logical_or(rot_mag_sq < 1e-8f)
     *
     * Dead Gaussians are those with:
     *   - opacity <= min_opacity OR
     *   - ||rotation||^2 < 1e-8 (near-zero rotation magnitude)
     *
     * @param opacities [N] - Opacity values
     * @param rotations [N, 4] - Quaternion rotations
     * @param dead_mask [N] - Output: boolean mask (uint8_t)
     * @param N - Number of Gaussians
     * @param min_opacity - Minimum valid opacity threshold
     * @param stream - CUDA stream
     */
    void launch_compute_dead_mask(
        const float* opacities,
        const float* rotations,
        uint8_t* dead_mask,
        size_t N,
        float min_opacity,
        void* stream = nullptr);

} // namespace lfs::training::mcmc
