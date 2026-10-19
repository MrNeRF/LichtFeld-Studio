/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/popspa_cuda.hpp"

#include "kernels/effective_rank_regularization.cuh"
#include "kernels/kernel_stream.hpp"
#include "kernels/popspa.cuh"

#include <initializer_list>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using gpu_ops::Tensor;

        void sync(std::initializer_list<const Tensor*> tensors, cudaStream_t stream) {
            for (const auto* tensor : tensors)
                if (tensor->is_valid())
                    tensor->sync_to_stream(stream);
        }

        void keep_indices(const Tensor& scores, const Tensor& active, const Tensor& frozen, size_t k, Tensor& keep) {
            const size_t n = scores.shape()[0];
            auto scratch = Tensor::empty({n}, Device::GPU, DataType::Int64);
            keep = Tensor::empty({k}, Device::GPU, DataType::Int64);
            const auto stream = resolve_stream(scores.stream());
            sync({&scores, &active, &frozen, &scratch, &keep}, stream);
            kernels::launch_popspa_keep_indices(reinterpret_cast<const double*>(scores.data_ptr()), active.ptr<bool>(),
                                                frozen.ptr<bool>(), n, k, scratch.ptr<int64_t>(), keep.ptr<int64_t>(), stream);
            scratch.set_stream(stream);
            keep.set_stream(stream);
        }

        void project(const Tensor& raw, Tensor& z, Tensor& u, const Tensor& active, const Tensor& frozen, size_t k,
                     bool update_dual) {
            const size_t n = raw.numel();
            auto values = Tensor::empty({n, sizeof(double)}, Device::GPU, DataType::UInt8);
            auto indices = Tensor::empty({n}, Device::GPU, DataType::Int64);
            auto keep = Tensor::empty({k}, Device::GPU, DataType::Int64);
            const auto stream = resolve_stream(raw.stream());
            sync({&raw, &z, &u, &active, &frozen, &values, &indices, &keep}, stream);
            kernels::launch_popspa_project(raw.ptr<float>(), z.ptr<float>(), u.ptr<float>(), active.ptr<bool>(), frozen.ptr<bool>(),
                                           n, k, reinterpret_cast<double*>(values.data_ptr()), indices.ptr<int64_t>(),
                                           keep.ptr<int64_t>(), update_dual, stream);
            for (const auto* tensor : {&active, &frozen})
                tensor->record_stream(stream);
            for (auto* tensor : {&z, &u, &values, &indices, &keep})
                tensor->set_stream(stream);
        }

        void penalty(const Tensor& raw, const Tensor& z, const Tensor& u, const Tensor& active, const Tensor& frozen,
                     Tensor& grad, Tensor& loss, float rho) {
            const auto stream = resolve_stream(raw.stream());
            sync({&raw, &z, &u, &active, &frozen, &grad, &loss}, stream);
            kernels::launch_popspa_penalty(raw.ptr<float>(), z.ptr<float>(), u.ptr<float>(), active.ptr<bool>(), frozen.ptr<bool>(),
                                           grad.ptr<float>(), loss.ptr<float>(), raw.numel(), rho, stream);
            for (const auto* tensor : {&z, &u, &active, &frozen})
                tensor->record_stream(stream);
            grad.set_stream(stream);
            loss.set_stream(stream);
        }

        void effective_rank(const Tensor& log_scales, Tensor& grad, const Tensor& active, const Tensor& frozen, Tensor& loss,
                            size_t active_count, float rank_weight, float thin_weight, float epsilon) {
            const auto stream = resolve_stream(log_scales.stream());
            sync({&log_scales, &grad, &active, &frozen, &loss}, stream);
            kernels::launch_effective_rank_regularization(log_scales.ptr<float>(), grad.ptr<float>(),
                                                          active.is_valid() ? active.ptr<bool>() : nullptr,
                                                          frozen.is_valid() ? frozen.ptr<bool>() : nullptr, loss.ptr<float>(),
                                                          log_scales.shape()[0], active_count, rank_weight, thin_weight,
                                                          epsilon, stream);
            loss.set_stream(stream);
            grad.set_stream(stream);
        }

        const gpu_ops::PopspaOps kCudaPopspaOps{
            .score_encoding = gpu_ops::PopScoreEncoding::Float64,
            .keep_indices = keep_indices,
            .project = project,
            .penalty = penalty,
            .effective_rank = effective_rank,
        };
    } // namespace

    const gpu_ops::PopspaOps& cuda_popspa_ops() {
        return kCudaPopspaOps;
    }
} // namespace lfs::training
