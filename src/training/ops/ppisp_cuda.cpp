/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/ppisp_cuda.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "training/kernels/ppisp.cuh"
#include "training/kernels/ppisp_controller.cuh"

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;

        float* optional_ptr(Out tensor) {
            return tensor.is_valid() ? tensor.ptr<float>() : nullptr;
        }

        // Preserve launcher-side resolution of the current stream.
        void forward(const PPISPInputs& p, In rgb, Out corrected, const PPISPRegion& r) {
            kernels::launch_ppisp_forward_chw_region(
                p.exposure.ptr<float>(), p.vignetting.ptr<float>(), p.color.ptr<float>(), p.crf.ptr<float>(),
                rgb.ptr<float>(), corrected.ptr<float>(), static_cast<int>(rgb.shape()[1]),
                static_cast<int>(rgb.shape()[2]), r.x_offset, r.y_offset,
                r.full_width > 0 ? r.full_width : static_cast<int>(rgb.shape()[2]), r.full_height,
                r.cameras, r.frames, r.camera_index, r.frame_index, nullptr);
        }

        void backward(const PPISPInputs& p, In rgb, In grad_output, const PPISPOutputs& g, Out grad_rgb,
                      int cameras, int frames, int camera_index, int frame_index) {
            kernels::launch_ppisp_backward_chw(
                p.exposure.ptr<float>(), p.vignetting.ptr<float>(), p.color.ptr<float>(), p.crf.ptr<float>(),
                rgb.ptr<float>(), grad_output.ptr<float>(), g.exposure.ptr<float>(), g.vignetting.ptr<float>(),
                g.color.ptr<float>(), g.crf.ptr<float>(), grad_rgb.ptr<float>(),
                static_cast<int>(rgb.shape()[1]), static_cast<int>(rgb.shape()[2]),
                cameras, frames, camera_index, frame_index, nullptr);
        }

        void adam(const PPISPAdamGroup& g, const PPISPAdamUpdateParams& p) {
            kernels::launch_ppisp_adam_update(g.parameter.ptr<float>(), g.moment1.ptr<float>(),
                                              g.moment2.ptr<float>(), g.gradient.ptr<float>(), static_cast<int>(g.parameter.numel()),
                                              p.lr, p.beta1, p.beta2, p.bc1_rcp, p.bc2_sqrt_rcp, p.eps, nullptr);
        }

        void adam_batch(const std::array<PPISPAdamGroup, 4>& groups, const PPISPAdamUpdateParams& p) {
            const auto bind = [](const PPISPAdamGroup& g) {
                return g.parameter.is_valid()
                           ? kernels::PPISPAdamGroup{g.parameter.ptr<float>(), g.moment1.ptr<float>(),
                                                     g.moment2.ptr<float>(), g.gradient.ptr<float>(),
                                                     static_cast<int>(g.parameter.numel())}
                           : kernels::PPISPAdamGroup{};
            };
            // Resolve CRF before the other launcher arguments, as in the component.
            const auto crf = bind(groups[3]);
            kernels::launch_ppisp_adam_update_batched(bind(groups[0]), bind(groups[1]), bind(groups[2]), crf,
                                                      p.lr, p.beta1, p.beta2, p.bc1_rcp, p.bc2_sqrt_rcp, p.eps, nullptr);
        }

        void vignetting_regularization(In parameters, Out gradient, Out loss,
                                       float center, float channel, float non_positive) {
            kernels::launch_ppisp_vignetting_reg(parameters.ptr<float>(), optional_ptr(gradient), optional_ptr(loss),
                                                 static_cast<int>(parameters.numel() / 15), center, channel, non_positive, nullptr);
        }

        void project_mean(Out exposure, Out color) {
            kernels::launch_ppisp_project_mean(exposure.ptr<float>(), color.ptr<float>(),
                                               static_cast<int>(exposure.numel()), nullptr);
        }

        void initialize(const PPISPOutputs& p) {
            kernels::launch_ppisp_init_identity(p.exposure.ptr<float>(), p.vignetting.ptr<float>(),
                                                p.color.ptr<float>(), p.crf.ptr<float>(), static_cast<int>(p.vignetting.numel() / 15),
                                                static_cast<int>(p.exposure.numel()), nullptr);
        }

        void prepare_input(In features, Out fc_input, float exposure_prior) {
            constexpr int CNN_FLAT_DIM = 1600;
            const auto stream = core::getCurrentCUDAStream();
            if (!features.is_valid()) {
                LFS_CUDA_CHECK(cudaMemcpyAsync(fc_input.ptr<float>() + CNN_FLAT_DIM,
                                               &exposure_prior, sizeof(float), cudaMemcpyHostToDevice, stream));
                LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                return;
            }
            cudaMemcpyAsync(fc_input.ptr<float>(), features.ptr<float>(), CNN_FLAT_DIM * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            if (exposure_prior != 1.0f) {
                cudaMemcpyAsync(fc_input.ptr<float>() + CNN_FLAT_DIM, &exposure_prior, sizeof(float),
                                cudaMemcpyHostToDevice, stream);
            }
        }

        void backward_layer(In grad_output, In activation, In weight,
                            Out weight_gradient, Out bias_gradient, Out grad_input) {
            const auto stream = core::getCurrentCUDAStream();
            const int m = static_cast<int>(grad_output.numel());
            const int n = static_cast<int>(activation.numel());
            kernels::launch_outer_product_accumulate(grad_output.ptr<float>(), activation.ptr<float>(),
                                                     weight_gradient.ptr<float>(), m, n, 1.0f, stream);
            kernels::launch_bias_grad_accumulate(grad_output.ptr<float>(), bias_gradient.ptr<float>(), m, stream);
            if (grad_input.is_valid()) {
                core::tensor_ops::launch_sgemm(grad_output.ptr<float>(), weight.ptr<float>(),
                                               grad_input.ptr<float>(), 1, n, m, stream);
                kernels::launch_relu_backward(grad_input.ptr<float>(), activation.ptr<float>(),
                                              grad_input.ptr<float>(), n, stream);
            }
        }

        const PPISPOps kCudaPPISPOps{
            .forward = forward,
            .backward = backward,
            .adam = adam,
            .adam_batch = adam_batch,
            .vignetting_regularization = vignetting_regularization,
            .project_mean = project_mean,
            .initialize = initialize,
        };
        const ControllerOps kCudaControllerOps{
            .prepare_input = prepare_input,
            .backward_layer = backward_layer,
        };
    } // namespace

    const gpu_ops::PPISPOps& cuda_ppisp_ops() { return kCudaPPISPOps; }
    const gpu_ops::ControllerOps& cuda_controller_ops() { return kCudaControllerOps; }
} // namespace lfs::training
