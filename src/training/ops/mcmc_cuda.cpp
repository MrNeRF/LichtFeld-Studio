/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mcmc_cuda.hpp"

#include "core/tensor_cuda_interop.hpp"
#include "kernels/mcmc_kernels.hpp"

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::McmcRows;
        using lfs::gpu_ops::SampleDomain;
        using lfs::gpu_ops::Tensor;

        void relocate(const Tensor& opacity, const Tensor& scales, const Tensor& ratios,
                      Tensor& new_opacity, Tensor& new_scales, float min_opacity) {
            mcmc::launch_relocation_kernel(opacity.ptr<float>(), scales.ptr<float>(), ratios.ptr<int32_t>(),
                                           min_opacity, new_opacity.ptr<float>(), new_scales.ptr<float>(), opacity.numel(),
                                           core::getCurrentCUDAStream());
        }
        void noise(const Tensor& opacity, const Tensor& scales, const Tensor& quats, const Tensor& frozen,
                   Tensor& means, uint64_t seed, float lr) {
            mcmc::launch_inject_noise_kernel(opacity.ptr<float>(), scales.ptr<float>(), quats.ptr<float>(),
                                             means.ptr<float>(), frozen.is_valid() ? frozen.ptr<bool>() : nullptr,
                                             frozen.is_valid() ? frozen.numel() : 0, lr, means.shape()[0], seed, core::getCurrentCUDAStream());
        }
        void copy_rows(const Tensor& source, const Tensor& destination, const McmcRows& rows) {
            mcmc::launch_copy_gaussian_params(source.ptr<int64_t>(), destination.ptr<int64_t>(),
                                              rows.means.ptr<float>(), rows.sh0.ptr<float>(), nullptr, rows.raw_scales.ptr<float>(),
                                              rows.raw_quats.ptr<float>(), rows.raw_opacity.ptr<float>(), destination.numel(), 0,
                                              rows.raw_opacity.ndim() == 2 ? 1 : 0, rows.means.shape()[0], core::getCurrentCUDAStream());
        }
        void update_rows(const Tensor& indices, const Tensor& scales, const Tensor& opacity,
                         Tensor& scales_out, Tensor& opacity_out) {
            mcmc::launch_update_scaling_opacity(indices.ptr<int64_t>(), scales.ptr<float>(), opacity.ptr<float>(),
                                                scales_out.ptr<float>(), opacity_out.ptr<float>(), indices.numel(),
                                                opacity_out.ndim() == 2 ? 1 : 0, scales_out.shape()[0], core::getCurrentCUDAStream());
        }
        void sample(const Tensor& weights, const Tensor& opacity, const Tensor& scales, const Tensor& alive,
                    Tensor& indices, Tensor& sampled_opacity, Tensor& sampled_scales, SampleDomain domain, uint64_t seed) {
            if (domain == SampleDomain::AliveIndices) {
                mcmc::launch_multinomial_sample_and_gather(weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(),
                                                           alive.ptr<int64_t>(), alive.numel(), indices.numel(), seed, indices.ptr<int64_t>(),
                                                           sampled_opacity.ptr<float>(), sampled_scales.ptr<float>(), opacity.numel(), core::getCurrentCUDAStream());
            } else {
                mcmc::launch_multinomial_sample_all(weights.ptr<float>(), opacity.ptr<float>(), scales.ptr<float>(),
                                                    opacity.numel(), indices.numel(), seed, indices.ptr<int64_t>(), sampled_opacity.ptr<float>(),
                                                    sampled_scales.ptr<float>(), core::getCurrentCUDAStream());
            }
        }
        void fold_error(Tensor& error_max, Tensor& densification) {
            mcmc::launch_max_error_and_zero_densification(error_max.ptr<float>(), densification.ptr<float>(),
                                                          error_max.numel(), core::getCurrentCUDAStream());
        }
        const lfs::gpu_ops::McmcOps kCudaMcmcOps{
            .initialize = mcmc::init_relocation_coefficients,
            .relocate = relocate,
            .noise = noise,
            .copy_rows = copy_rows,
            .update_rows = update_rows,
            .sample = sample,
            .fold_error = fold_error,
        };
    } // namespace

    const lfs::gpu_ops::McmcOps& cuda_mcmc_ops() {
        return kCudaMcmcOps;
    }
} // namespace lfs::training
