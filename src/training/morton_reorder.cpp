/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/morton_reorder.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "kernels/morton_reorder_kernels.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"
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

        // Gather through a private scratch buffer and copy back into the live
        // tensor. The exportable allocator returns views at fixed region
        // offsets per name, so a fresh "allocation" would alias the source.
        void permute_named_param(Tensor& tensor, const Tensor& perm) {
            const std::size_t n = perm.numel();
            if (!tensor.is_valid() || tensor.numel() == 0 || n == 0) {
                return;
            }
            if (tensor.ndim() == 0 || tensor.size(0) != n) {
                permute_dim0_prefix(tensor, perm);
                return;
            }
            Tensor scratch = Tensor::zeros_direct(tensor.shape(), n, tensor.device(), tensor.dtype());
            scratch.set_stream(tensor.stream());
            tensor.index_select_into(scratch, 0, perm, BoundaryMode::Assert);
            tensor.copy_from(scratch);
        }

        void permute_shN_q16(core::SplatData& splat, const Tensor& perm, cudaStream_t stream) {
            auto& live = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            const std::size_t n = static_cast<std::size_t>(splat.size());
            const std::size_t n_cells = core::sh_value_quant::sh_value_u16_count(n, rest);
            const std::size_t n_bound_floats = core::sh_value_quant::n_bounds_for_prims(n) * 2;
            if (live.numel() < n_cells) {
                throw std::runtime_error("Morton reorder: q16 shN storage smaller than its logical size");
            }
            if (!bounds.is_valid() || bounds.numel() < n_bound_floats) {
                throw std::runtime_error(
                    "Morton reorder: shN_value_bounds short/missing — refusing silent SH wipe");
            }

            if (live.stream() != stream) {
                live.set_stream(stream);
            }
            if (bounds.stream() != stream) {
                bounds.set_stream(stream);
            }
            lfs::core::waitForCUDAStream(stream, live.stream());
            lfs::core::waitForCUDAStream(stream, bounds.stream());
            lfs::core::waitForCUDAStream(stream, perm.stream());

            const auto* src_u16 = reinterpret_cast<const std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(live));
            const auto* src_bounds = static_cast<const float*>(
                lfs::core::resolve_exportable_device_ptr(bounds));
            const auto* perm_ptr = perm.ptr<std::int64_t>();

            Tensor dest_u16 = Tensor::zeros_direct(
                TensorShape({n_cells}), n_cells, Device::CUDA, DataType::Float16);
            dest_u16.set_stream(stream);
            Tensor dest_bounds = Tensor::zeros(
                TensorShape({n_bound_floats}), Device::CUDA, DataType::Float32);
            dest_bounds.set_stream(stream);

            constexpr std::size_t kChunk = static_cast<std::size_t>(sh_value::kBlockSize);
            const std::size_t chunk_floats = core::sh_swizzled_float_count(kChunk, rest);
            Tensor fp32_chunk = Tensor::zeros(
                TensorShape({chunk_floats}), Device::CUDA, DataType::Float32);
            fp32_chunk.set_stream(stream);

            auto* dest_codes = reinterpret_cast<std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(dest_u16));
            auto* dest_mm = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(dest_bounds));

            for (std::size_t offset = 0; offset < n; offset += kChunk) {
                const std::size_t chunk = std::min(kChunk, n - offset);
                core::sh_value_quant::decode_shN_u16_gathered_to_float4(
                    src_u16,
                    src_bounds,
                    perm_ptr,
                    fp32_chunk.ptr<float>(),
                    offset,
                    chunk,
                    n,
                    rest,
                    stream);
                core::sh_value_quant::encode_shN_float4_to_u16(
                    fp32_chunk.ptr<float>(),
                    dest_codes + core::sh_value_quant::sh_value_u16_count(offset, rest),
                    dest_mm + core::sh_value_quant::n_bounds_for_prims(offset) * 2,
                    chunk,
                    rest,
                    stream);
            }

            LFS_CUDA_CHECK(cudaMemcpyAsync(
                lfs::core::resolve_exportable_device_ptr(live),
                dest_codes,
                n_cells * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,
                stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(
                lfs::core::resolve_exportable_device_ptr(bounds),
                dest_mm,
                n_bound_floats * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream));
            LFS_CUDA_CHECK_MSG(
                cudaStreamSynchronize(stream), "q16 morton permute copy-back");
        }

        void permute_shN_fp32(core::SplatData& splat, const Tensor& perm, cudaStream_t stream) {
            const bool expanded = sh_value::ensure_shN_fp32_for_mutation(splat);
            auto& live = splat.shN();
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            const std::size_t n = static_cast<std::size_t>(splat.size());
            if (!live.is_valid() || live.dtype() != DataType::Float32) {
                if (expanded) {
                    (void)sh_value::commit_shN_after_mutation(splat);
                }
                return;
            }

            const std::size_t logical = core::sh_swizzled_float_count(n, rest);
            if (live.numel() < logical) {
                throw std::runtime_error("Morton reorder: shN storage smaller than its logical size");
            }
            Tensor scratch = Tensor::zeros_direct(
                TensorShape({logical}), logical, Device::CUDA, DataType::Float32);
            scratch.set_stream(stream);
            if (live.stream() != stream) {
                live.set_stream(stream);
            }
            core::shN_swizzled_gather_self_i64(
                live.ptr<float>(),
                scratch.ptr<float>(),
                perm.ptr<std::int64_t>(),
                n,
                0,
                rest,
                stream);
            if (live.numel() == logical) {
                live.copy_from(scratch);
            } else {
                live.slice(0, 0, logical).copy_from(scratch);
            }
            if (expanded) {
                (void)sh_value::commit_shN_after_mutation(splat);
            }
        }
    } // namespace

    void permute_shN(core::SplatData& splat, const lfs::core::Tensor& perm, cudaStream_t stream) {
        auto& shN = splat.shN();
        const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
        const std::size_t n = static_cast<std::size_t>(splat.size());
        if (!shN.is_valid() || shN.numel() == 0 || rest == 0 || n == 0 ||
            !perm.is_valid() || perm.numel() != n) {
            return;
        }
        if (stream == nullptr) {
            stream = core::getCurrentCUDAStream();
        }
        LiveModelMutationGuard mutation_guard("permute_shN");
        if (splat.shN_value_quantized() && shN.dtype() == lfs::core::DataType::Float16) {
            permute_shN_q16(splat, perm, stream);
            return;
        }
        permute_shN_fp32(splat, perm, stream);
    }

    namespace {

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

        permute_named_param(splat.means(), result.permutation);
        permute_named_param(splat.sh0(), result.permutation);
        permute_named_param(splat.scaling_raw(), result.permutation);
        permute_named_param(splat.rotation_raw(), result.permutation);
        permute_named_param(splat.opacity_raw(), result.permutation);
        permute_shN(splat, result.permutation, stream);

        if (splat._densification_info.is_valid() && splat._densification_info.numel() > 0) {
            permute_row_tensor(splat._densification_info, result.permutation);
        }
        if (splat._max_screen_share.is_valid() && splat._max_screen_share.numel() > 0) {
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(),
                               "wait fused adam before screen-share morton permute");
            permute_row_tensor(splat._max_screen_share, result.permutation);
            if (optimizer != nullptr) {
                optimizer->refresh_screen_share_buffer();
            }
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
        lfs::core::Tensor::trim_memory_pool();
        result.applied = true;
        LOG_INFO("Morton reordered {} Gaussians", n);
        return result;
    }

} // namespace lfs::training::morton
