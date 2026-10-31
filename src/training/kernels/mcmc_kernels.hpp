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
     * Relocation - Equation (9) from "3D Gaussian Splatting as Markov Chain Monte Carlo"
     *
     * Each copy row holds a copy of its source row. Recomputes opacity and
     * scale for the number of times the source was sampled and writes them,
     * raw, to both the copy and the source. Requires init_relocation_coefficients().
     *
     * @param src_indices [n] - Source row of each copy
     * @param copy_indices [n] - Copy rows; nullptr means first_copy + i
     * @param ratios [n] - Times each copy's source was sampled (int32)
     * @param scaling_raw [N, 3] - Raw scales (modified in-place)
     * @param opacity_raw [N] or [N, 1] - Raw opacities (modified in-place)
     */
    void launch_relocate_copies(
        const int64_t* src_indices,
        const int64_t* copy_indices,
        int64_t first_copy,
        const int32_t* ratios,
        float min_opacity,
        float* scaling_raw,
        float* opacity_raw,
        size_t n,
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
     * Fused RNG, covariance transform, and means update.
     * Honors frozen_mask. The seed controls the curand sequence.
     */
    void launch_inject_noise_kernel(
        const float* raw_opacities,
        const float* raw_scales,
        const float* raw_quats,
        float* means,
        const bool* frozen_mask,
        size_t frozen_mask_size,
        float current_lr,
        size_t N,
        uint64_t seed,
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
     * Fused multinomial sampling from all sampling weights (no intermediate allocations)
     *
     * Samples from all N weights. Relocation passes its dead rows as
     * `excluded` instead of compacting the alive rows into an index list.
     *
     * @param sampling_weights [N] - Sampling probabilities (full array), finite and >= 0
     * @param min_weight - Floor applied to each weight as it is read, like
     *                     Tensor::clamp_min (-inf: weights are used as given)
     * @param excluded [N] - Optional; nonzero rows are never sampled (nullptr: none)
     * @param N - Number of Gaussians
     * @param n_samples - Number of samples to draw
     * @param seed - Random seed for sampling
     * @param sampled_indices [n_samples] - Output: sampled indices
     * @param stream - CUDA stream
     * @return false, with nothing sampled, when no row has a positive weight
     */
    [[nodiscard]] bool launch_multinomial_sample_all(
        const float* sampling_weights,
        float min_weight,
        const uint8_t* excluded,
        size_t N,
        size_t n_samples,
        uint64_t seed,
        int64_t* sampled_indices,
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

    /**
     * fold densification error row into running max and zero the
     * densification rows in one pass (replaces max + zero_). Only the error
     * row is maxed; the weight row, when kept, is only cleared.
     */
    void launch_max_error_and_zero_densification(
        float* error_max,
        float* error_row,
        float* weight_row,
        size_t N,
        void* stream = nullptr);

} // namespace lfs::training::mcmc
