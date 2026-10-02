/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/extra_loss_cuda.hpp"

#include "components/sparsity_optimizer_kernels.hpp"
#include "training/kernels/regularization.cuh"

namespace lfs::training {
    namespace {
        using namespace lfs::gpu_ops;

        void regularize(In raw, Out gradient, Out loss, Out reduction_temp, Regularizer kind, float weight) {
            float* grad = gradient.is_valid() ? gradient.ptr<float>() : nullptr;
            if (kind == Regularizer::Scale) {
                kernels::launch_fused_scale_regularization(raw.ptr<float>(), grad, loss.ptr<float>(),
                                                           reduction_temp.ptr<float>(), raw.numel(), weight, nullptr);
            } else {
                kernels::launch_fused_opacity_regularization(raw.ptr<float>(), grad, loss.ptr<float>(),
                                                             reduction_temp.ptr<float>(), raw.numel(), weight, nullptr);
            }
        }

        void admm(In sigmoid, In z, In u, Out gradient, float rho, float grad_loss, bool accumulate) {
            launch_admm_backward_fused(gradient.ptr<float>(), sigmoid.ptr<float>(), z.ptr<float>(), u.ptr<float>(),
                                       rho, grad_loss, sigmoid.numel(), accumulate);
        }

        const ExtraLossOps kCudaExtraLossOps{
            .regularize = regularize,
            .admm = admm,
        };
    } // namespace

    const lfs::gpu_ops::ExtraLossOps& cuda_extra_loss_ops() { return kCudaExtraLossOps; }
} // namespace lfs::training
