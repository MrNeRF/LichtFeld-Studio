/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/morton_cuda.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_data.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor_completion.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "kernels/morton_reorder_kernels.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/sh_value_storage.hpp"

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using core::TensorShape;
        using lfs::gpu_ops::JointCodecParams;
        using lfs::gpu_ops::JointLayout;
        using lfs::gpu_ops::Tensor;

        Tensor permutation(const Tensor& means) {
            return kernels::launch_morton_permutation(means, core::getCurrentCUDAStream());
        }

        void permute_joint(const Tensor& packed, const Tensor& bounds, const Tensor& permutation,
                           Tensor& new_packed, Tensor& new_bounds, const JointCodecParams& p) {
            const auto launch = p.layout == JointLayout::Rows
                                    ? kernels::launch_joint_permute_contiguous
                                    : kernels::launch_joint_permute_shN;
            launch(packed.ptr<std::uint8_t>(), bounds.ptr<float>(),
                   new_packed.ptr<std::uint8_t>(), new_bounds.ptr<float>(), permutation.ptr<std::int64_t>(),
                   p.primitives, p.attributes_or_slots, p.bits, core::getCurrentCUDAStream());
        }

        void permute_joint_grouped(Tensor& packed, const Tensor& bounds, const Tensor& permutation,
                                   Tensor& new_bounds, Tensor& scratch, const JointCodecParams& p) {
            kernels::launch_joint_permute_shN_grouped(
                packed.ptr<std::uint8_t>(), bounds.ptr<float>(), new_bounds.ptr<float>(),
                permutation.ptr<std::int64_t>(), p.primitives, p.attributes_or_slots, p.bits,
                scratch.ptr<std::uint8_t>(), scratch.bytes(), core::getCurrentCUDAStream());
        }

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

        void copy_back_native(Tensor& live, const void* source, const std::size_t bytes, cudaStream_t stream) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(live.data_ptr(), source, bytes, cudaMemcpyDeviceToDevice, stream));
            if (live.stream() != stream) {
                lfs::core::waitForCUDAStream(live.stream(), stream);
            }
        }

        void permute_shN_q16(core::SplatData& splat, const Tensor& perm, core::TensorExecutionTarget target,
                             const IdleArenaScratch& scratch) {
            const auto stream = static_cast<cudaStream_t>(target.native_handle());
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

        void permute_shN_fp32(core::SplatData& splat, const Tensor& perm, core::TensorExecutionTarget target,
                              const IdleArenaScratch& scratch) {
            const auto stream = static_cast<cudaStream_t>(target.native_handle());
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
                copy_back_native(live, gathered, logical * sizeof(float), stream);
            } else if (live.numel() == logical) {
                live.copy_from(fallback);
            } else {
                live.slice(0, 0, logical).copy_from(fallback);
            }
            if (expanded) {
                (void)sh_value::commit_shN_after_mutation(splat);
            }
        }

        void copy_back(Tensor& live, const void* source, size_t bytes, core::TensorExecutionTarget target) {
            copy_back_native(live, source, bytes, static_cast<cudaStream_t>(target.native_handle()));
        }
        void gather_gradient(const Tensor& source, const Tensor& perm, Tensor& destination,
                             uint32_t rest, core::TensorExecutionTarget target) {
            core::shN_swizzled_gather_self_i64(source.ptr<float>(), destination.ptr<float>(), perm.ptr<int64_t>(),
                                               perm.numel(), 0, rest, static_cast<cudaStream_t>(target.native_handle()));
        }
        const lfs::gpu_ops::MortonOps kCudaMortonOps{
            .permutation = permutation,
            .permute_joint = permute_joint,
            .permute_joint_grouped = permute_joint_grouped,
            .permute_sh_q16 = permute_shN_q16,
            .permute_sh_fp32 = permute_shN_fp32,
            .copy_back = copy_back,
            .gather_gradient = gather_gradient,
        };
    } // namespace

    const lfs::gpu_ops::MortonOps& cuda_morton_ops() {
        return kCudaMortonOps;
    }
} // namespace lfs::training
