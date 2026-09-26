/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/morton_reorder.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor_completion.hpp"
#include "kernels/morton_reorder_kernels.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
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

        // Morton runs between training frames, so its SH destinations borrow the
        // idle rasterizer arena: whole when they fit, otherwise in groups sized to
        // what is committed. The borrow ends after the reorder's device barrier.
        constexpr std::size_t MIN_MORTON_ARENA_BYTES = 16ull << 20;

        // Buffer for a grouped permutation whose unit is one cell (or slot)
        // across every tile: the arena block when it holds at least one unit,
        // otherwise a private allocation bounded by GROUP_BUDGET_BYTES.
        struct GroupScratch {
            void* ptr = nullptr;
            std::size_t bytes = 0;
            Tensor owner;
        };

        [[nodiscard]] GroupScratch group_scratch(const IdleArenaScratch& scratch,
                                                 const std::size_t unit_bytes,
                                                 const std::size_t units,
                                                 cudaStream_t stream) {
            constexpr std::size_t GROUP_BUDGET_BYTES = 64ull << 20;
            if (scratch.capacity() >= unit_bytes) {
                return {scratch.data(), scratch.capacity(), {}};
            }
            const std::size_t budget_units = std::max<std::size_t>(1, GROUP_BUDGET_BYTES / unit_bytes);
            const std::size_t bytes = std::min(units, budget_units) * unit_bytes;
            Tensor owner = Tensor::empty_exact({bytes}, DataType::UInt8);
            owner.set_stream(stream);
            return {owner.data_ptr(), bytes, std::move(owner)};
        }

        void copy_back(Tensor& live, const void* source, const std::size_t bytes, cudaStream_t stream) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(live.data_ptr(), source, bytes, cudaMemcpyDeviceToDevice, stream));
            if (live.stream() != stream) {
                lfs::core::waitForCUDAStream(live.stream(), stream);
            }
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
            gathered.set_stream(core::getCurrentCUDAStream());
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
            dest.set_stream(core::getCurrentCUDAStream());
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
            scratch.set_stream(core::getCurrentCUDAStream());
            tensor.index_select_into(scratch, 0, perm, BoundaryMode::Assert);
            tensor.copy_from(scratch);
        }

        void permute_shN_q16(core::SplatData& splat, const Tensor& perm, cudaStream_t stream,
                             const IdleArenaScratch& scratch) {
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

            Tensor dest_bounds = Tensor::empty_exact({n_bound_floats}, DataType::Float32);
            dest_bounds.set_stream(stream);
            dest_bounds.zero_();
            auto* dest_mm = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(dest_bounds));
            auto* live_codes = reinterpret_cast<std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(live));

            if (auto* dest_codes = static_cast<std::uint16_t*>(scratch.zeroed(n_cells * sizeof(std::uint16_t)))) {
                core::sh_value_quant::encode_shN_u16_gathered(
                    src_u16, src_bounds, perm_ptr, dest_codes, dest_mm, n, n, rest, stream);
                LFS_CUDA_CHECK(cudaMemcpyAsync(
                    live_codes, dest_codes, n_cells * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice, stream));
            } else {
                constexpr std::size_t R = core::kShReorderSize;
                const std::uint32_t cells_per_prim = core::sh_value_quant::n_value_cells_per_prim(rest);
                const std::size_t tiles = core::sh_swizzled_padded_n(n) / R;
                const std::size_t cell_bytes = tiles * R * sizeof(std::uint16_t);
                const auto group = group_scratch(scratch, cell_bytes, cells_per_prim, stream);
                const auto cells_per_group = static_cast<std::uint32_t>(
                    std::min<std::size_t>(group.bytes / cell_bytes, cells_per_prim));
                core::sh_value_quant::gathered_shN_u16_block_bounds(
                    src_u16, src_bounds, perm_ptr, dest_mm, n, n, rest, stream);
                for (std::uint32_t first = 0; first < cells_per_prim; first += cells_per_group) {
                    const std::uint32_t last = std::min(first + cells_per_group, cells_per_prim);
                    const std::size_t width = static_cast<std::size_t>(last - first) * R * sizeof(std::uint16_t);
                    LFS_CUDA_CHECK(cudaMemsetAsync(group.ptr, 0, width * tiles, stream));
                    core::sh_value_quant::encode_shN_u16_gathered_cells(
                        src_u16, src_bounds, perm_ptr, dest_mm, static_cast<std::uint16_t*>(group.ptr),
                        n, n, rest, first, last, stream);
                    LFS_CUDA_CHECK(cudaMemcpy2DAsync(
                        live_codes + static_cast<std::size_t>(first) * R,
                        static_cast<std::size_t>(cells_per_prim) * R * sizeof(std::uint16_t),
                        group.ptr, width, width, tiles, cudaMemcpyDeviceToDevice, stream));
                }
            }
            bounds.flatten().slice(0, 0, n_bound_floats).copy_(dest_bounds);
            lfs::core::TensorCompletion completion;
            completion.include(live);
            completion.include(bounds);
            completion.wait();
        }

        void permute_shN_fp32(core::SplatData& splat, const Tensor& perm, cudaStream_t stream,
                              const IdleArenaScratch& scratch) {
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
            if (live.stream() != stream) {
                live.set_stream(stream);
            }
            Tensor fallback;
            auto* gathered = static_cast<float*>(scratch.zeroed(logical * sizeof(float)));
            if (gathered == nullptr) {
                fallback = Tensor::zeros_direct(
                    TensorShape({logical}), logical, Device::CUDA, DataType::Float32);
                fallback.set_stream(stream);
                gathered = fallback.ptr<float>();
            }
            core::shN_swizzled_gather_self_i64(
                live.ptr<float>(),
                gathered,
                perm.ptr<std::int64_t>(),
                n,
                0,
                rest,
                stream);
            if (!fallback.is_valid()) {
                copy_back(live, gathered, logical * sizeof(float), stream);
            } else if (live.numel() == logical) {
                live.copy_from(fallback);
            } else {
                live.slice(0, 0, logical).copy_from(fallback);
            }
            if (expanded) {
                (void)sh_value::commit_shN_after_mutation(splat);
            }
        }

        void permute_shN_impl(core::SplatData& splat, const Tensor& perm, cudaStream_t stream,
                              const IdleArenaScratch& scratch) {
            auto& shN = splat.shN();
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            const std::size_t n = static_cast<std::size_t>(splat.size());
            if (!shN.is_valid() || shN.numel() == 0 || rest == 0 || n == 0 ||
                !perm.is_valid() || perm.numel() != n) {
                return;
            }
            LiveModelMutationGuard mutation_guard("permute_shN");
            if (splat.shN_value_quantized() && shN.dtype() == lfs::core::DataType::Float16) {
                permute_shN_q16(splat, perm, stream, scratch);
                return;
            }
            permute_shN_fp32(splat, perm, stream, scratch);
        }
    } // namespace

    void permute_shN(core::SplatData& splat, const lfs::core::Tensor& perm, cudaStream_t stream) {
        if (stream == nullptr) {
            stream = core::getCurrentCUDAStream();
        }
        permute_shN_impl(splat, perm, stream, IdleArenaScratch(0, 0, stream));
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
                Tensor dest_packed = Tensor::empty_exact(state->exp_avg.shape(), DataType::UInt8);
                dest_packed.set_stream(stream);
                dest_packed.zero_();
                const std::size_t nb = joint_adam::n_bounds_for_prims(n);
                Tensor dest_bounds = Tensor::empty_exact({nb, std::size_t{4}}, DataType::Float32);
                dest_bounds.set_stream(stream);
                dest_bounds.zero_();
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
                state->exp_avg.copy_from(dest_packed);
                state->joint_bounds.copy_from(dest_bounds);
            }
        }

        void permute_optimizer_shN(
            AdamOptimizer& optimizer,
            const core::SplatData& splat,
            const Tensor& perm,
            cudaStream_t stream,
            const IdleArenaScratch& scratch) {
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
                    TensorShape({logical}), cap, Device::GPU, DataType::Float32);
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
            const std::size_t packed_bytes = state->exp_avg.bytes();
            const std::size_t nb = joint_adam::n_bounds_for_prims(n);
            Tensor dest_bounds = Tensor::empty_exact({nb, std::size_t{4}}, DataType::Float32);
            dest_bounds.set_stream(stream);
            dest_bounds.zero_();
            if (auto* packed = static_cast<std::uint8_t*>(scratch.zeroed(packed_bytes))) {
                kernels::launch_joint_permute_shN(
                    state->exp_avg.ptr<std::uint8_t>(),
                    state->joint_bounds.ptr<float>(),
                    packed,
                    dest_bounds.ptr<float>(),
                    perm.ptr<std::int64_t>(),
                    static_cast<int>(n),
                    slots,
                    state->joint_bits,
                    stream);
                copy_back(state->exp_avg, packed, packed_bytes, stream);
            } else {
                constexpr std::size_t R = core::kShReorderSize;
                const std::size_t tiles = (n + R - 1) / R;
                const std::size_t slot_bytes =
                    tiles * R * 4 * static_cast<std::size_t>(joint_adam::bytes_per_cell(state->joint_bits));
                const auto group = group_scratch(scratch, slot_bytes, static_cast<std::size_t>(slots), stream);
                kernels::launch_joint_permute_shN_grouped(
                    state->exp_avg.ptr<std::uint8_t>(),
                    state->joint_bounds.ptr<float>(),
                    dest_bounds.ptr<float>(),
                    perm.ptr<std::int64_t>(),
                    static_cast<int>(n),
                    slots,
                    state->joint_bits,
                    static_cast<std::uint8_t*>(group.ptr),
                    group.bytes,
                    stream);
                if (state->exp_avg.stream() != stream) {
                    lfs::core::waitForCUDAStream(state->exp_avg.stream(), stream);
                }
            }
            state->joint_bounds.copy_from(dest_bounds);
        }
    } // namespace

    namespace {
        [[nodiscard]] std::size_t morton_scratch_bytes(const core::SplatData& splat,
                                                       const AdamOptimizer* optimizer) {
            const auto n = static_cast<std::size_t>(splat.size());
            const auto rest = static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
            std::size_t bytes = 0;
            if (rest > 0 && splat.shN().is_valid()) {
                bytes = splat.shN_value_quantized()
                            ? core::sh_value_quant::sh_value_u16_count(n, rest) * sizeof(std::uint16_t)
                            : core::sh_swizzled_float_count(n, rest) * sizeof(float);
            }
            if (optimizer != nullptr) {
                const auto* state = optimizer->get_state(ParamType::ShN);
                if (state != nullptr && state->is_joint() && state->exp_avg.is_valid()) {
                    bytes = std::max(bytes, state->exp_avg.bytes());
                }
            }
            return bytes;
        }
    } // namespace

    void permute_row_tensor(Tensor& tensor, const Tensor& perm) {
        if (!tensor.is_valid() || tensor.numel() == 0 || !perm.is_valid() || perm.numel() == 0) {
            return;
        }
        const std::size_t n = perm.numel();
        if (tensor.ndim() == 2 && tensor.size(1) == n && tensor.size(0) != n) {
            Tensor dest = Tensor::zeros(tensor.shape(), tensor.device(), tensor.dtype());
            dest.set_stream(core::getCurrentCUDAStream());
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

        const std::size_t scratch_bytes = morton_scratch_bytes(splat, optimizer);
        const IdleArenaScratch scratch(scratch_bytes, std::min(scratch_bytes, MIN_MORTON_ARENA_BYTES), stream);

        permute_named_param(splat.means(), result.permutation);
        permute_named_param(splat.sh0(), result.permutation);
        permute_named_param(splat.scaling_raw(), result.permutation);
        permute_named_param(splat.rotation_raw(), result.permutation);
        permute_named_param(splat.opacity_raw(), result.permutation);
        permute_shN_impl(splat, result.permutation, stream, scratch);

        if (splat._densification_info.is_valid() && splat._densification_info.numel() > 0) {
            permute_row_tensor(splat._densification_info, result.permutation);
        }
        if (splat._max_screen_share.is_valid() && splat._max_screen_share.numel() > 0) {
            core::gpu_device_barrier(core::GpuBackend::CUDA);
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
            permute_optimizer_shN(*optimizer, splat, result.permutation, stream, scratch);
        }

        splat.note_param_layout_changed();
        core::gpu_device_barrier(core::GpuBackend::CUDA);
        // Morton's temporary permutation buffers normally return to the CUDA
        // pool immediately. Keep the VRAM hygiene trim when that leaves a
        // substantial reserved/unused tail, but avoid paying a device-wide
        // sync for small tails on every reorder.
        constexpr std::size_t MORTON_TRIM_UNUSED_THRESHOLD = 64ULL << 20;
        lfs::core::Tensor::trim_memory_pool_if_reserved_unused_exceeds(
            MORTON_TRIM_UNUSED_THRESHOLD);
        result.applied = true;
        LOG_INFO("Morton reordered {} Gaussians", n);
        return result;
    }

} // namespace lfs::training::morton
