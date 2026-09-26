/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/photometric_cuda.hpp"

#include "lfs/kernels/l1_loss.cuh"
#include "lfs/kernels/loss_tensor_contract.hpp"
#include "lfs/kernels/ssim.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lfs::training {
    namespace {

        struct CudaPhotoState : lfs::gpu_ops::BackendState {
            kernels::LossWorkspaceArena arena;
            lfs::core::Tensor grad_buffer;
            lfs::core::Tensor loss_scalar;
            lfs::core::Tensor l1_reduction;
            size_t l1_blocks = 0;
            kernels::SSIMMapWorkspace error_maps;
            // Keeps the last allocating metric's inputs alive until the next call.
            std::vector<lfs::core::Tensor> metric_keep;
            lfs::core::RankedDims l1_dims;
            bool l1_ready = false;

            void ensure_l1(const lfs::core::TensorShape& shape, const size_t num_blocks) {
                const lfs::core::RankedDims& dims = shape.dims();
                if (l1_ready && l1_dims == dims && l1_blocks == num_blocks) {
                    return;
                }
                grad_buffer = lfs::core::Tensor::empty(shape, lfs::core::Device::GPU);
                loss_scalar = lfs::core::Tensor::zeros({1}, lfs::core::Device::GPU);
                if (num_blocks > 0) {
                    l1_reduction = lfs::core::Tensor::empty({num_blocks}, lfs::core::Device::GPU);
                }
                l1_dims = dims;
                l1_blocks = num_blocks;
                l1_ready = true;
            }
        };

        // operator= deep-copies when both sides are same-shaped views, and a
        // steady publish aliases the workspace, so that assignment cloned the map.
        void bind_handle(lfs::core::Tensor& dst, const lfs::core::Tensor& src) {
            if (dst.is_valid() && src.is_valid() && dst.storage_ptr() == src.storage_ptr() &&
                dst.shape() == src.shape() && dst.dtype() == src.dtype()) {
                return;
            }
            if (dst.is_view() && dst.is_valid() && src.is_valid() && dst.shape() == src.shape() &&
                dst.dtype() == src.dtype()) {
                dst = lfs::core::Tensor{};
            }
            dst = src;
        }

        template <typename Fn>
        void dispatch_target_ptr(const lfs::core::Tensor& target, Fn&& fn) {
            if (target.dtype() == lfs::core::DataType::UInt8) {
                fn(target.ptr<uint8_t>());
            } else {
                fn(target.ptr<float>());
            }
        }

        [[nodiscard]] CudaPhotoState& state_of(lfs::gpu_ops::PhotoSaved& saved) {
            if (!saved.backend) {
                throw std::logic_error("photometric op called without a created state");
            }
            return static_cast<CudaPhotoState&>(*saved.backend);
        }

        [[nodiscard]] const CudaPhotoState* state_of(const lfs::gpu_ops::PhotoSaved& saved) {
            return static_cast<const CudaPhotoState*>(saved.backend.get());
        }

        void publish_maps(lfs::gpu_ops::PhotoSaved& saved,
                          const lfs::core::Tensor& ssim_map,
                          const lfs::core::Tensor& cs_map) {
            bind_handle(saved.ssim_map, ssim_map);
            bind_handle(saved.cs_map, cs_map);
        }

        void squeeze_batch(lfs::core::Tensor& gradient, const lfs::core::Tensor& image) {
            if (image.ndim() == 3 && gradient.is_valid()) {
                gradient = gradient.squeeze(0);
            }
        }

        [[nodiscard]] size_t reserved_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return 0;
            }
            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }
            size_t row_elems = 1;
            if (tensor.ndim() > 1) {
                for (size_t dim = 1; dim < tensor.ndim(); ++dim) {
                    row_elems *= tensor.shape()[dim];
                }
            }
            return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
        }

        void keep_metric(CudaPhotoState& state, std::vector<lfs::core::Tensor> tensors) {
            state.metric_keep = std::move(tensors);
        }

        lfs::gpu_ops::State photo_create() {
            return std::make_unique<CudaPhotoState>();
        }

        void photo_evaluate(
            lfs::gpu_ops::PhotoSaved& saved,
            const lfs::gpu_ops::Tensor& corrected,
            const lfs::gpu_ops::Tensor& raw,
            const lfs::gpu_ops::Tensor& target,
            const lfs::gpu_ops::Tensor& mask,
            const lfs::gpu_ops::PhotoParams& params,
            lfs::gpu_ops::Tensor& loss,
            lfs::gpu_ops::Tensor& grad_corrected,
            lfs::gpu_ops::Tensor& grad_raw) {
            auto& state = state_of(saved);
            kernels::validate_loss_weight(params.ssim_weight);
            const bool writes_raw = params.path == lfs::gpu_ops::PhotoPath::Decoupled ||
                                    params.path == lfs::gpu_ops::PhotoPath::MaskedDecoupled;
            if (!writes_raw && grad_raw.is_valid()) {
                grad_raw = {};
            }

            switch (params.path) {
            case lfs::gpu_ops::PhotoPath::L1: {
                auto [rendered_4d, gt_4d] = kernels::prepare_loss_images(corrected, target);
                const size_t count = rendered_4d.numel();
                const size_t num_blocks = std::min((count + 255) / 256, size_t{1024});
                state.ensure_l1(rendered_4d.shape(), num_blocks);
                lfs::core::pin_operands({&rendered_4d, &gt_4d});
                dispatch_target_ptr(gt_4d, [&](auto* gt_ptr) {
                    kernels::launch_fused_l1_loss(
                        rendered_4d.ptr<float>(),
                        gt_ptr,
                        state.grad_buffer.ptr<float>(),
                        state.loss_scalar.ptr<float>(),
                        state.l1_reduction.ptr<float>(),
                        count,
                        nullptr);
                });
                bind_handle(grad_corrected, state.grad_buffer);
                bind_handle(loss, state.loss_scalar);
                squeeze_batch(grad_corrected, corrected);
                break;
            }
            case lfs::gpu_ops::PhotoPath::SSIM: {
                // ssim_forward prepares the images and sizes the workspace.
                auto& workspace = state.arena.pure_ssim();
                auto [ssim_value, ssim_ctx] = kernels::ssim_forward(
                    corrected, target, workspace, params.valid_padding);
                auto loss_tensor =
                    lfs::core::Tensor::full({1}, 1.0f, lfs::core::Device::GPU) - ssim_value;
                bind_handle(loss, loss_tensor);
                bind_handle(grad_corrected, kernels::ssim_backward(ssim_ctx, workspace, -1.0f));
                squeeze_batch(grad_corrected, corrected);
                publish_maps(saved, workspace.ssim_map, workspace.cs_map);
                break;
            }
            case lfs::gpu_ops::PhotoPath::Fused: {
                // fused_l1_ssim_* prepares the images and sizes the workspace.
                auto& workspace = state.arena.fused();
                auto [loss_tensor, fused_ctx] = kernels::fused_l1_ssim_forward(
                    corrected, target, params.ssim_weight, workspace, params.valid_padding);
                bind_handle(grad_corrected, kernels::fused_l1_ssim_backward(fused_ctx, workspace));
                bind_handle(loss, loss_tensor);
                squeeze_batch(grad_corrected, corrected);
                publish_maps(saved, workspace.ssim_map, workspace.cs_map);
                break;
            }
            case lfs::gpu_ops::PhotoPath::Decoupled: {
                auto& workspace = state.arena.decoupled();
                auto [loss_tensor, ctx] = kernels::decoupled_fused_l1_ssim_forward(
                    corrected, raw, target, params.ssim_weight, workspace, params.valid_padding);
                auto grads = kernels::decoupled_fused_l1_ssim_backward(ctx, workspace);
                bind_handle(loss, loss_tensor);
                bind_handle(grad_corrected, grads.grad_corrected);
                bind_handle(grad_raw, grads.grad_raw);
                if (corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                    grad_raw = grad_raw.squeeze(0);
                }
                publish_maps(saved, workspace.ssim_map, workspace.cs_map);
                break;
            }
            case lfs::gpu_ops::PhotoPath::MaskedFused: {
                auto& workspace = state.arena.masked_fused();
                auto [loss_tensor, ctx] = kernels::masked_fused_l1_ssim_forward(
                    corrected, target, mask, params.ssim_weight, workspace);
                bind_handle(grad_corrected, kernels::masked_fused_l1_ssim_backward(ctx, workspace));
                bind_handle(loss, loss_tensor);
                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
                publish_maps(saved, workspace.ssim_map, workspace.cs_map);
                break;
            }
            case lfs::gpu_ops::PhotoPath::MaskedDecoupled: {
                auto& workspace = state.arena.masked_decoupled();
                auto [loss_tensor, ctx] = kernels::masked_decoupled_fused_l1_ssim_forward(
                    corrected, raw, target, mask, params.ssim_weight, workspace);
                auto grads = kernels::masked_decoupled_fused_l1_ssim_backward(ctx, workspace);
                bind_handle(loss, loss_tensor);
                bind_handle(grad_corrected, grads.grad_corrected);
                bind_handle(grad_raw, grads.grad_raw);
                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
                if (grad_raw.ndim() == 4 && corrected.ndim() == 3) {
                    grad_raw = grad_raw.squeeze(0);
                }
                publish_maps(saved, workspace.ssim_map, workspace.cs_map);
                break;
            }
            }
        }

        lfs::gpu_ops::Tensor photo_metric(
            lfs::gpu_ops::PhotoSaved& saved,
            const lfs::gpu_ops::Tensor& predicted,
            const lfs::gpu_ops::Tensor& target,
            const bool maps,
            const bool valid_padding) {
            auto& state = state_of(saved);
            if (maps) {
                auto result = kernels::ssim_forward_map(predicted, target, valid_padding);
                publish_maps(saved, result.ssim_map, result.cs_map);
                auto value = result.ssim_value;
                keep_metric(state, {
                                       result.ssim_map,
                                       result.cs_map,
                                       result.ssim_value,
                                       result.ctx.img1,
                                       result.ctx.img2,
                                       result.ctx.dm_dmu1,
                                       result.ctx.dm_dsigma1_sq,
                                       result.ctx.dm_dsigma12,
                                   });
                return value;
            }

            auto [value, ctx] = kernels::ssim_forward(predicted, target, valid_padding);
            keep_metric(state, {
                                   value,
                                   ctx.img1,
                                   ctx.img2,
                                   ctx.dm_dmu1,
                                   ctx.dm_dsigma1_sq,
                                   ctx.dm_dsigma12,
                               });
            return value;
        }

        void photo_error_map(
            lfs::gpu_ops::PhotoSaved& saved,
            const lfs::gpu_ops::Tensor& predicted,
            const lfs::gpu_ops::Tensor& target,
            lfs::gpu_ops::Tensor& error,
            const bool contrast_structure_only) {
            kernels::ssim_error_map_forward(
                predicted, target, state_of(saved).error_maps, error, contrast_structure_only);
        }

        void photo_map_to_error(const lfs::gpu_ops::Tensor& map, lfs::gpu_ops::Tensor& error) {
            kernels::launch_ssim_to_error_map(map, error);
        }

        const lfs::gpu_ops::PhotometricOps kCudaPhotometricOps{
            .create = photo_create,
            .evaluate = photo_evaluate,
            .metric = photo_metric,
            .error_map = photo_error_map,
            .map_to_error = photo_map_to_error,
        };

    } // namespace

    const lfs::gpu_ops::PhotometricOps& cuda_photometric_ops() {
        return kCudaPhotometricOps;
    }

    PhotoWorkspaceBytes photo_workspace_bytes(const lfs::gpu_ops::PhotoSaved& saved) {
        const CudaPhotoState* state = state_of(saved);
        if (!state) {
            return {};
        }
        return PhotoWorkspaceBytes{
            .required = state->arena.required_bytes(),
            .allocated = state->arena.allocated_bytes(),
            .error_map = reserved_bytes(state->error_maps.ssim_map),
        };
    }

    void photo_shrink_to_required(lfs::gpu_ops::PhotoSaved& saved) {
        if (saved.backend) {
            state_of(saved).arena.shrink_to_required();
        }
    }

    void photo_reset(lfs::gpu_ops::PhotoSaved& saved) {
        if (saved.backend) {
            state_of(saved).arena.reset();
        }
    }

} // namespace lfs::training
