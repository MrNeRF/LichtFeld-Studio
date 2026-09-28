/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal McmcOps; see ops/mcmc_cuda.cpp for the reference behaviour.

#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <cmath>
#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::McmcRows;
        using lfs::gpu_ops::SampleDomain;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;

        constexpr int kRelocationMax = 51;

        uint32_t count32(const size_t count, const char* what) {
            if (count > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument(std::format("{} count {} exceeds uint32", what, count));
            return static_cast<uint32_t>(count);
        }

        // The CUDA table lives in __constant__ memory; this one is never freed,
        // so it outlives the tensor backend at exit.
        Tensor& relocation_coefficients() {
            static Tensor* const table = new Tensor();
            return *table;
        }

        void initialize(const int n_max) {
            if (n_max < 0 || n_max > kRelocationMax)
                throw std::invalid_argument(
                    std::format("MCMC relocation n_max must be within [0, {}], got {}", kRelocationMax, n_max));
            std::vector<float> coefficients(kRelocationMax * kRelocationMax, 0.0f);
            for (int n = 0; n < n_max; ++n) {
                float binom = 1.0f;
                for (int k = 0; k <= n; ++k) {
                    const float sign = k % 2 == 0 ? 1.0f : -1.0f;
                    // CUDA's host rsqrtf rounds 1 / sqrt in double.
                    const auto rsqrt = static_cast<float>(1.0 / std::sqrt(static_cast<double>(k + 1)));
                    coefficients[n * kRelocationMax + k] = binom * sign * rsqrt;
                    if (k < n)
                        binom *= static_cast<float>(n - k) / static_cast<float>(k + 1);
                }
            }
            relocation_coefficients() =
                Tensor::from_vector(coefficients, {coefficients.size()}, core::Device::GPU);
        }

        struct RelocateParams {
            uint64_t opacity, scales, ratios, coefficients, new_opacity, new_scales;
            uint32_t count;
            float min_opacity;
        };

        void relocate(const Tensor& opacity, const Tensor& scales, const Tensor& ratios, Tensor& new_opacity,
                      Tensor& new_scales, const float min_opacity) {
            const size_t n = opacity.numel();
            if (n == 0)
                return;
            const Tensor& coefficients = relocation_coefficients();
            if (!coefficients.is_valid())
                throw std::logic_error("MCMC relocate needs initialize() to upload the relocation coefficients");
            const RelocateParams params{mk::address(opacity), mk::address(scales),
                                        mk::address(ratios), mk::address(coefficients),
                                        mk::address(new_opacity), mk::address(new_scales),
                                        count32(n, "MCMC relocate"), min_opacity};
            mk::launch_items("mcmc_relocate", params,
                             {&opacity, &scales, &ratios, &coefficients, &new_opacity, &new_scales}, n);
        }

        struct NoiseParams {
            uint64_t raw_opacity, raw_scales, raw_quats, frozen, means;
            uint64_t seed;
            uint32_t frozen_count, count;
            float lr;
        };

        void noise(const Tensor& raw_opacity, const Tensor& raw_scales, const Tensor& raw_quats, const Tensor& frozen,
                   Tensor& means, const uint64_t seed, const float lr) {
            const size_t n = means.shape()[0];
            if (n == 0)
                return;
            const NoiseParams params{mk::address(raw_opacity),
                                     mk::address(raw_scales),
                                     mk::address(raw_quats),
                                     mk::address(frozen),
                                     mk::address(means),
                                     seed,
                                     frozen.is_valid() ? count32(frozen.numel(), "MCMC frozen mask") : 0u,
                                     count32(n, "MCMC noise"),
                                     lr};
            mk::launch_items("mcmc_noise", params, {&raw_opacity, &raw_scales, &raw_quats, &frozen, &means}, n);
        }

        struct RowsParams {
            uint64_t source, destination, means, sh0, scales, quats, opacity, new_scales, new_opacity;
            uint32_t count, rows;
        };

        void copy_rows(const Tensor& source, const Tensor& destination, const McmcRows& rows) {
            const size_t n = destination.numel();
            if (n == 0)
                return;
            const RowsParams params{mk::address(source),
                                    mk::address(destination),
                                    mk::address(rows.means),
                                    mk::address(rows.sh0),
                                    mk::address(rows.raw_scales),
                                    mk::address(rows.raw_quats),
                                    mk::address(rows.raw_opacity),
                                    0,
                                    0,
                                    count32(n, "MCMC copy"),
                                    count32(rows.means.shape()[0], "MCMC rows")};
            mk::launch_items("mcmc_copy_rows", params,
                             {&source, &destination, &rows.means, &rows.sh0, &rows.raw_scales, &rows.raw_quats,
                              &rows.raw_opacity},
                             n);
        }

        void update_rows(const Tensor& indices, const Tensor& raw_scales, const Tensor& raw_opacity,
                         Tensor& raw_scales_out, Tensor& raw_opacity_out) {
            const size_t n = indices.numel();
            if (n == 0)
                return;
            const RowsParams params{0,
                                    mk::address(indices),
                                    0,
                                    0,
                                    mk::address(raw_scales_out),
                                    0,
                                    mk::address(raw_opacity_out),
                                    mk::address(raw_scales),
                                    mk::address(raw_opacity),
                                    count32(n, "MCMC update"),
                                    count32(raw_scales_out.shape()[0], "MCMC rows")};
            mk::launch_items("mcmc_update_rows", params,
                             {&indices, &raw_scales, &raw_opacity, &raw_scales_out, &raw_opacity_out}, n);
        }

        struct SampleParams {
            uint64_t cumsum, alive, opacity, raw_scales, indices, sampled_opacity, sampled_scales;
            uint64_t seed;
            uint32_t categories, samples;
        };

        void sample(const Tensor& weights, const Tensor& opacity, const Tensor& raw_scales, const Tensor& alive,
                    Tensor& indices, Tensor& sampled_opacity, Tensor& sampled_scales, const SampleDomain domain,
                    const uint64_t seed) {
            const size_t samples = indices.numel();
            const bool by_alive = domain == SampleDomain::AliveIndices;
            const size_t categories = by_alive ? alive.numel() : opacity.numel();
            if (samples == 0 || categories == 0)
                return;
            if (categories > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument(std::format("MCMC multinomial input {} exceeds INT_MAX", categories));
            const Tensor flat = weights.reshape({-1});
            if (!by_alive && flat.numel() < categories)
                throw std::invalid_argument(
                    std::format("MCMC sampling needs a weight per row, got {} for {} rows", flat.numel(), categories));
            const Tensor probabilities = by_alive ? flat.index_select(0, alive) : flat.slice(0, 0, categories);
            const Tensor cumsum = probabilities.cumsum(0);
            const SampleParams params{mk::address(cumsum),
                                      by_alive ? mk::address(alive) : 0,
                                      mk::address(opacity),
                                      mk::address(raw_scales),
                                      mk::address(indices),
                                      mk::address(sampled_opacity),
                                      mk::address(sampled_scales),
                                      seed,
                                      static_cast<uint32_t>(categories),
                                      count32(samples, "MCMC samples")};
            mk::launch_items("mcmc_sample", params,
                             {&cumsum, &alive, &opacity, &raw_scales, &indices, &sampled_opacity, &sampled_scales},
                             samples);
        }

        struct FoldParams {
            uint64_t error_max, densification;
            uint32_t count;
        };

        void fold_error(Tensor& error_max, Tensor& densification) {
            const size_t n = error_max.numel();
            if (n == 0)
                return;
            const FoldParams params{mk::address(error_max), mk::address(densification), count32(n, "MCMC fold")};
            mk::launch_items("mcmc_fold_error", params, {&error_max, &densification}, n);
        }
    } // namespace

    const lfs::gpu_ops::McmcOps& metal_mcmc_ops() {
        static const lfs::gpu_ops::McmcOps ops{
            .initialize = initialize,
            .relocate = relocate,
            .noise = noise,
            .copy_rows = copy_rows,
            .update_rows = update_rows,
            .sample = sample,
            .fold_error = fold_error,
        };
        return ops;
    }
} // namespace lfs::training
