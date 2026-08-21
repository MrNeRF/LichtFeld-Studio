/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/morton_reorder.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "kernels/morton_reorder_kernels.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lfs::training::morton {
    namespace {
        using core::BoundaryMode;
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::TensorShape;

        [[nodiscard]] std::size_t prim_capacity(const core::SplatData& splat) {
            const auto n = static_cast<std::size_t>(splat.size());
            if (!splat.means().is_valid()) {
                return n;
            }
            const auto cap = splat.means().capacity();
            return std::max(cap > 0 ? cap : n, n);
        }

        void permute_dim0_prefix(Tensor& tensor, const Tensor& perm) {
            const std::size_t n = perm.numel();
            if (!tensor.is_valid() || tensor.numel() == 0 || n == 0) {
                return;
            }
            if (tensor.ndim() == 0 || tensor.size(0) < n) {
                return;
            }
            const std::size_t old0 = tensor.size(0);
            const std::size_t cap = std::max(tensor.capacity() > 0 ? tensor.capacity() : old0, old0);
            auto dims = tensor.shape().dims();
            dims[0] = n;
            Tensor gathered = Tensor::zeros_direct(
                TensorShape(dims), std::max(cap, n), tensor.device(), tensor.dtype());
            gathered.set_stream(tensor.stream());
            if (old0 == n) {
                tensor.index_select_into(gathered, 0, perm, BoundaryMode::Assert);
                tensor = std::move(gathered);
                return;
            }
            Tensor prefix = tensor.slice(0, 0, n);
            if (!prefix.is_contiguous()) {
                prefix = prefix.contiguous();
            }
            prefix.index_select_into(gathered, 0, perm, BoundaryMode::Assert);
            dims[0] = old0;
            Tensor dest = Tensor::zeros_direct(
                TensorShape(dims), cap, tensor.device(), tensor.dtype());
            dest.set_stream(tensor.stream());
            dest.slice(0, 0, n).copy_from(gathered);
            dest.slice(0, n, old0).copy_from(tensor.slice(0, n, old0));
            tensor = std::move(dest);
        }

        void permute_named_param(
            core::SplatData& splat,
            Tensor& tensor,
            const Tensor& perm,
            std::string_view name) {
            const std::size_t n = perm.numel();
            if (!tensor.is_valid() || tensor.numel() == 0 || n == 0) {
                return;
            }
            if (tensor.ndim() == 0 || tensor.size(0) != n) {
                permute_dim0_prefix(tensor, perm);
                return;
            }
            const std::size_t cap = std::max(tensor.capacity() > 0 ? tensor.capacity() : n, n);
            Tensor dest = splat.allocate_named_param(tensor.shape(), cap, tensor.dtype(), name);
            dest.set_stream(tensor.stream());
            tensor.index_select_into(dest, 0, perm, BoundaryMode::Assert);
            tensor = std::move(dest);
        }

        void permute_shN(core::SplatData& splat, const Tensor& perm, cudaStream_t stream) {
            auto& shN = splat.shN();
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            const std::size_t n = static_cast<std::size_t>(splat.size());
            if (!shN.is_valid() || shN.numel() == 0 || rest == 0 || n == 0) {
                return;
            }

            const bool expanded = sh_value::ensure_shN_fp32_for_mutation(splat);
            auto& live = splat.shN();
            if (!live.is_valid() || live.dtype() != DataType::Float32) {
                if (expanded) {
                    (void)sh_value::commit_shN_after_mutation(splat);
                }
                return;
            }

            const std::size_t cap = prim_capacity(splat);
            const std::size_t logical = core::sh_swizzled_float_count(n, rest);
            const std::size_t cap_floats = core::sh_swizzled_float_count(cap, rest);
            Tensor dest = splat.allocate_named_param(
                TensorShape({logical}),
                std::max(logical, cap_floats),
                DataType::Float32,
                "SplatData.shN");
            dest.set_name("splat.shN");
            dest.set_stream(stream);
            if (live.stream() != stream) {
                live.set_stream(stream);
            }
            core::shN_swizzled_gather_self_i64(
                live.ptr<float>(),
                dest.ptr<float>(),
                perm.ptr<std::int64_t>(),
                n,
                0,
                rest,
                stream);
            live = std::move(dest);
            if (expanded) {
                (void)sh_value::commit_shN_after_mutation(splat);
            }
        }

        void permute_optimizer(AdamOptimizer& optimizer, const Tensor& perm, cudaStream_t stream) {
            const std::size_t n = perm.numel();
            if (n == 0) {
                return;
            }

            if (optimizer.frozen_mask().is_valid() && optimizer.frozen_mask().numel() >= n) {
                Tensor frozen = optimizer.frozen_mask();
                permute_dim0_prefix(frozen, perm);
                optimizer.set_frozen_mask(std::move(frozen));
            }
            if (optimizer.crop_damping_mask().is_valid() &&
                optimizer.crop_damping_mask().numel() >= n) {
                Tensor crop = optimizer.crop_damping_mask();
                permute_dim0_prefix(crop, perm);
                optimizer.set_crop_damping_mask(std::move(crop));
            }

            for (const auto type : AdamOptimizer::all_param_types()) {
                if (type == ParamType::ShN) {
                    continue;
                }
                auto* state = optimizer.get_state_mutable(type);
                if (state == nullptr) {
                    continue;
                }
                if (state->grad.is_valid() && state->grad.numel() > 0 &&
                    state->grad.ndim() > 0 && state->grad.size(0) == n) {
                    permute_dim0_prefix(state->grad, perm);
                }
                if (!state->is_joint() || !state->exp_avg.is_valid() ||
                    !state->joint_bounds.is_valid()) {
                    continue;
                }
                lfs::core::waitForCUDAStream(stream, state->exp_avg.stream());
                lfs::core::waitForCUDAStream(stream, state->joint_bounds.stream());
                lfs::core::waitForCUDAStream(stream, perm.stream());

                const int bpc = joint_adam::bytes_per_cell(state->joint_bits);
                if (bpc <= 0 || state->exp_avg.ndim() != 2) {
                    continue;
                }
                const int n_attr = static_cast<int>(state->exp_avg.size(1)) / bpc;
                if (n_attr <= 0 || state->exp_avg.size(0) != n) {
                    continue;
                }
                const std::size_t packed_cap =
                    std::max(state->exp_avg.capacity() > 0 ? state->exp_avg.capacity() : n, n);
                Tensor dest_packed = Tensor::zeros_direct(
                    state->exp_avg.shape(), packed_cap, Device::CUDA, DataType::UInt8);
                dest_packed.set_stream(stream);
                const std::size_t nb = joint_adam::n_bounds_for_prims(n);
                const std::size_t nb_cap = std::max(
                    state->joint_bounds.capacity() > 0 ? state->joint_bounds.capacity() : nb, nb);
                Tensor dest_bounds = Tensor::zeros_direct(
                    TensorShape({nb, std::size_t{4}}), nb_cap, Device::CUDA, DataType::Float32);
                dest_bounds.set_stream(stream);
                kernels::launch_joint_permute_contiguous(
                    state->exp_avg.ptr<std::uint8_t>(),
                    state->joint_bounds.ptr<float>(),
                    dest_packed.ptr<std::uint8_t>(),
                    dest_bounds.ptr<float>(),
                    perm.ptr<std::int64_t>(),
                    static_cast<int>(n),
                    n_attr,
                    state->joint_bits,
                    stream);
                state->exp_avg = std::move(dest_packed);
                state->joint_bounds = std::move(dest_bounds);
            }
        }

        void permute_optimizer_shN(
            AdamOptimizer& optimizer,
            const core::SplatData& splat,
            const Tensor& perm,
            cudaStream_t stream) {
            auto* state = optimizer.get_state_mutable(ParamType::ShN);
            if (state == nullptr || !state->is_joint() || !state->exp_avg.is_valid() ||
                !state->joint_bounds.is_valid()) {
                return;
            }
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            const int slots = static_cast<int>(core::sh_float4_slots_for_rest(rest));
            const std::size_t n = perm.numel();
            if (slots <= 0 || n == 0) {
                return;
            }
            if (state->grad.is_valid() && state->grad.dtype() == DataType::Float32 &&
                state->grad.numel() > 0) {
                const std::size_t logical = core::sh_swizzled_float_count(n, rest);
                const std::size_t cap = std::max(
                    state->grad.capacity() > 0 ? state->grad.capacity() : logical, logical);
                Tensor dest = Tensor::zeros_direct(
                    TensorShape({logical}), cap, Device::CUDA, DataType::Float32);
                dest.set_stream(stream);
                core::shN_swizzled_gather_self_i64(
                    state->grad.ptr<float>(),
                    dest.ptr<float>(),
                    perm.ptr<std::int64_t>(),
                    n,
                    0,
                    rest,
                    stream);
                state->grad = std::move(dest);
            }

            lfs::core::waitForCUDAStream(stream, state->exp_avg.stream());
            lfs::core::waitForCUDAStream(stream, state->joint_bounds.stream());
            const std::size_t packed_n = state->exp_avg.size(0);
            const std::size_t packed_cap = std::max(
                state->exp_avg.capacity() > 0 ? state->exp_avg.capacity() : packed_n, packed_n);
            Tensor dest_packed = Tensor::zeros_direct(
                state->exp_avg.shape(), packed_cap, Device::CUDA, DataType::UInt8);
            dest_packed.set_stream(stream);
            const std::size_t nb = joint_adam::n_bounds_for_prims(n);
            const std::size_t nb_cap = std::max(
                state->joint_bounds.capacity() > 0 ? state->joint_bounds.capacity() : nb, nb);
            Tensor dest_bounds = Tensor::zeros_direct(
                TensorShape({nb, std::size_t{4}}), nb_cap, Device::CUDA, DataType::Float32);
            dest_bounds.set_stream(stream);
            kernels::launch_joint_permute_shN(
                state->exp_avg.ptr<std::uint8_t>(),
                state->joint_bounds.ptr<float>(),
                dest_packed.ptr<std::uint8_t>(),
                dest_bounds.ptr<float>(),
                perm.ptr<std::int64_t>(),
                static_cast<int>(n),
                slots,
                state->joint_bits,
                stream);
            state->exp_avg = std::move(dest_packed);
            state->joint_bounds = std::move(dest_bounds);
        }
    } // namespace

    void permute_row_tensor(Tensor& tensor, const Tensor& perm) {
        if (!tensor.is_valid() || tensor.numel() == 0 || !perm.is_valid() || perm.numel() == 0) {
            return;
        }
        const std::size_t n = perm.numel();
        if (tensor.ndim() == 2 && tensor.size(1) == n && tensor.size(0) != n) {
            Tensor dest = Tensor::zeros(tensor.shape(), tensor.device(), tensor.dtype());
            dest.set_stream(tensor.stream());
            tensor.index_select_into(dest, 1, perm, BoundaryMode::Assert);
            tensor = std::move(dest);
            return;
        }
        permute_dim0_prefix(tensor, perm);
    }

    ReorderResult apply_morton_reorder(
        core::SplatData& splat,
        AdamOptimizer* optimizer,
        cudaStream_t stream) {
        ReorderResult result;
        if (splat.has_frozen_ranges()) {
            LOG_DEBUG("Skipping Morton reorder: {} frozen-range span(s) are active",
                      splat.frozen_ranges().size());
            return result;
        }
        const auto n = static_cast<std::size_t>(splat.size());
        if (n == 0 || !splat.means().is_valid()) {
            return result;
        }

        LiveModelMutationGuard mutation_guard("morton_reorder");
        if (stream == nullptr) {
            stream = core::getCurrentCUDAStream();
        }
        if (splat.means().stream() != stream) {
            splat.means().set_stream(stream);
        }

        result.permutation = kernels::launch_morton_permutation(splat.means(), stream);
        if (!result.permutation.is_valid() || result.permutation.numel() != n) {
            LOG_ERROR("Morton reorder failed to produce a permutation of length {}", n);
            return result;
        }

        permute_named_param(splat, splat.means(), result.permutation, "SplatData.means");
        permute_named_param(splat, splat.sh0(), result.permutation, "SplatData.sh0");
        permute_named_param(splat, splat.scaling_raw(), result.permutation, "SplatData.scaling");
        permute_named_param(splat, splat.rotation_raw(), result.permutation, "SplatData.rotation");
        permute_named_param(splat, splat.opacity_raw(), result.permutation, "SplatData.opacity");
        permute_shN(splat, result.permutation, stream);

        if (splat._densification_info.is_valid() && splat._densification_info.numel() > 0) {
            permute_row_tensor(splat._densification_info, result.permutation);
        }
        if (splat.has_deleted_mask()) {
            permute_row_tensor(splat.deleted(), result.permutation);
            splat.notify_deleted_mask_changed();
        }

        if (optimizer != nullptr) {
            permute_optimizer(*optimizer, result.permutation, stream);
            permute_optimizer_shN(*optimizer, splat, result.permutation, stream);
        }

        splat.note_param_layout_changed();
        LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(), "morton reorder device barrier");
        result.applied = true;
        LOG_INFO("Morton reordered {} Gaussians", n);
        return result;
    }

} // namespace lfs::training::morton
