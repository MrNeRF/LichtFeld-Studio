/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/bilateral_cuda.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "training/kernels/bilateral_grid.cuh"

namespace lfs::training {
    namespace {
        using lfs::gpu_ops::AdamUpdateParams;
        using lfs::gpu_ops::BilateralOps;
        using lfs::gpu_ops::GridSliceParams;
        using lfs::gpu_ops::GridTransform;
        using lfs::gpu_ops::In;
        using lfs::gpu_ops::Layout;
        using lfs::gpu_ops::Out;

        // Preserve the legacy null-stream launches used by BilateralGrid.
        void slice_forward(In grid, In rgb, In shared_offset, Out output, const GridSliceParams& params) {
            const auto& shape = grid.shape();
            const auto& image = rgb.shape();
            const bool chw = params.layout == Layout::CHW;
            const int h = static_cast<int>(image[chw ? 1 : 0]);
            const int w = static_cast<int>(image[chw ? 2 : 1]);
            const auto launch = params.transform == GridTransform::ExposureChroma
                                    ? (chw ? kernels::launch_bilateral_grid_slice_forward_exposure_chroma_chw : kernels::launch_bilateral_grid_slice_forward_exposure_chroma)
                                    : (chw ? kernels::launch_bilateral_grid_slice_forward_chw : kernels::launch_bilateral_grid_slice_forward);
            launch(grid.ptr<float>(), rgb.ptr<float>(), output.ptr<float>(),
                   static_cast<int>(shape[2]), static_cast<int>(shape[3]), static_cast<int>(shape[4]), h, w,
                   shared_offset.ptr<float>(), nullptr);
        }
        void slice_backward(In grid, In rgb, In grad_output, In shared_offset, Out grad_grid, Out grad_rgb, const GridSliceParams& params) {
            const auto& shape = grid.shape();
            const auto& image = rgb.shape();
            const bool chw = params.layout == Layout::CHW;
            const int h = static_cast<int>(image[chw ? 1 : 0]);
            const int w = static_cast<int>(image[chw ? 2 : 1]);
            const auto launch = params.transform == GridTransform::ExposureChroma
                                    ? (chw ? kernels::launch_bilateral_grid_slice_backward_exposure_chroma_chw : kernels::launch_bilateral_grid_slice_backward_exposure_chroma)
                                    : (chw ? kernels::launch_bilateral_grid_slice_backward_chw : kernels::launch_bilateral_grid_slice_backward);
            launch(grid.ptr<float>(), rgb.ptr<float>(), grad_output.ptr<float>(), grad_grid.ptr<float>(), grad_rgb.ptr<float>(),
                   static_cast<int>(shape[2]), static_cast<int>(shape[3]), static_cast<int>(shape[4]), h, w,
                   shared_offset.ptr<float>(), nullptr, params.warp_aggregate);
        }
        void tv_forward(In grids, Out loss, Out reduction_temp, int norm_n) {
            const auto& shape = grids.shape();
            kernels::launch_bilateral_grid_tv_forward(grids.ptr<float>(), loss.ptr<float>(), reduction_temp.ptr<float>(),
                                                      static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3]), static_cast<int>(shape[4]), norm_n, nullptr);
        }
        void tv_backward(In grids, Out gradients, float grad_loss, int norm_n) {
            const auto& shape = grids.shape();
            kernels::launch_bilateral_grid_tv_backward(grids.ptr<float>(), grad_loss, gradients.ptr<float>(),
                                                       static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3]), static_cast<int>(shape[4]), norm_n, nullptr);
        }
        void project_mean(Out grids, In mean, In identity, int per_image) {
            const auto& shape = grids.shape();
            kernels::launch_bilateral_grid_project_mean(grids.ptr<float>(), mean.ptr<float>(), identity.ptr<float>(),
                                                        static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3]), static_cast<int>(shape[4]), per_image, nullptr);
        }
        void update_offset(Out channel_sum, Out shared_offset, In identity, In old_mean, In new_mean, float spatial, float inv_n_spatial) {
            kernels::launch_bilateral_grid_update_shared_offset(channel_sum.ptr<float>(), shared_offset.ptr<float>(),
                                                                identity.ptr<float>(), old_mean.ptr<float>(), new_mean.ptr<float>(), static_cast<int>(channel_sum.numel()), spatial, inv_n_spatial, nullptr);
        }
        void adam(Out grid, Out moment1, Out moment2, In gradient, const AdamUpdateParams& params) {
            kernels::launch_bilateral_grid_adam_update(grid.ptr<float>(), moment1.ptr<float>(), moment2.ptr<float>(), gradient.ptr<float>(),
                                                       static_cast<int>(grid.numel()), params.lr, params.beta1, params.beta2, params.bc1_rcp, params.bc2_sqrt_rcp, params.eps, nullptr);
        }
        void scale_moments(Out moment1, Out moment2, float scale1, float scale2) {
            kernels::launch_bilateral_grid_scale_moments(moment1.ptr<float>(), moment2.ptr<float>(), static_cast<int>(moment1.numel()), scale1, scale2, nullptr);
        }
        void upload_slice(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(device.ptr<float>() + device_offset, host.ptr<float>() + host_offset,
                                           elements * sizeof(float), cudaMemcpyHostToDevice, core::getCurrentCUDAStream()));
        }
        void download_slice(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(host.ptr<float>() + host_offset, device.ptr<float>() + device_offset,
                                           elements * sizeof(float), cudaMemcpyDeviceToHost, core::getCurrentCUDAStream()));
        }
        const BilateralOps kCudaBilateralOps{
            .slice_forward = slice_forward,
            .slice_backward = slice_backward,
            .tv_forward = tv_forward,
            .tv_backward = tv_backward,
            .project_mean = project_mean,
            .update_offset = update_offset,
            .adam = adam,
            .scale_moments = scale_moments,
            .upload_slice = upload_slice,
            .download_slice = download_slice,
        };
    } // namespace
    const lfs::gpu_ops::BilateralOps& cuda_bilateral_ops() { return kCudaBilateralOps; }
} // namespace lfs::training
