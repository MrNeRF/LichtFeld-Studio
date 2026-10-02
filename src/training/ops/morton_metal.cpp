/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal MortonOps; see ops/morton_cuda.cpp for the reference behaviour.

#include "core/logger.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor_completion.hpp"
#include "lfs/training/idle_arena_scratch.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "metal_families.hpp"
#include "metal_kernels.hpp"
#include "sh_metal.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace lfs::training {
    namespace {
        using core::DataType;
        using core::Device;
        using core::TensorShape;
        using lfs::gpu_ops::JointCodecParams;
        using lfs::gpu_ops::JointLayout;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;
        namespace quant = core::sh_value_quant;

        constexpr uint32_t kBlock = 256;
        constexpr size_t kReorder = core::kShReorderSize;

        uint32_t blocks(const int primitives) {
            return std::max(1u, (static_cast<uint32_t>(primitives) + kBlock - 1) / kBlock);
        }

        struct CodeParams {
            uint64_t means, lo, hi, codes;
            uint32_t count;
        };

        Tensor permutation(const Tensor& means) {
            if (!means.is_valid() || means.ndim() != 2 || means.size(1) != 3 || means.dtype() != DataType::Float32 ||
                means.device() != Device::GPU) {
                LOG_ERROR("morton permutation: means must be GPU Float32 [N, 3]");
                return {};
            }
            const size_t n = means.size(0);
            if (n == 0)
                return {};
            const Tensor dense = means.contiguous();
            const Tensor lo = dense.min(0);
            const Tensor hi = dense.max(0);
            Tensor keys = Tensor::empty({n}, Device::GPU, DataType::Float32);
            const CodeParams params{mk::address(dense), mk::address(lo), mk::address(hi), mk::address(keys),
                                    static_cast<uint32_t>(n)};
            mk::launch_items("morton_codes", params, {&dense, &lo, &hi, &keys}, n);
            // Codes are below 2^30, so their bits read as finite non-negative
            // floats that order like the integers. The float sort is stable, as
            // thrust's radix sort by key is.
            return keys.sort(0, false).second;
        }

        struct JointParams {
            uint64_t source, source_bounds, destination, destination_bounds, permutation;
            int32_t primitives;
            uint32_t width, swizzled;
            int32_t bits;
            uint32_t slot_begin, slot_end;
        };

        JointParams joint_params(const Tensor& packed, const Tensor& bounds, const Tensor& perm,
                                 const Tensor& destination, const Tensor& destination_bounds,
                                 const JointCodecParams& p) {
            if (p.bits != 8 && p.bits != 16)
                throw std::runtime_error(std::format("joint permute bits must be 8 or 16, got {}", p.bits));
            const auto width = static_cast<uint32_t>(p.attributes_or_slots);
            return {mk::address(packed),
                    mk::address(bounds),
                    mk::address(destination),
                    mk::address(destination_bounds),
                    mk::address(perm),
                    p.primitives,
                    width,
                    p.layout == JointLayout::SwizzledSH ? 1u : 0u,
                    p.bits,
                    0u,
                    width};
        }

        void permute_joint(const Tensor& packed, const Tensor& bounds, const Tensor& perm, Tensor& new_packed,
                           Tensor& new_bounds, const JointCodecParams& p) {
            if (p.primitives <= 0 || p.attributes_or_slots <= 0 || !packed.is_valid() || !new_packed.is_valid() ||
                !bounds.is_valid() || !new_bounds.is_valid() || !perm.is_valid())
                return;
            const JointParams params = joint_params(packed, bounds, perm, new_packed, new_bounds, p);
            mk::launch("morton_joint_bounds", params, {&packed, &bounds, &perm, &new_bounds}, blocks(p.primitives));
            mk::launch_items("morton_joint_encode", params, {&packed, &bounds, &perm, &new_packed, &new_bounds},
                             static_cast<size_t>(p.primitives));
        }

        // Reencodes slot groups that fit `scratch` and copies each back into
        // `packed`; a group reads only its own source slots, so in place is safe.
        void permute_joint_grouped(Tensor& packed, const Tensor& bounds, const Tensor& perm, Tensor& new_bounds,
                                   Tensor& scratch, const JointCodecParams& p) {
            if (p.primitives <= 0 || p.attributes_or_slots <= 0 || !packed.is_valid() || !bounds.is_valid() ||
                !new_bounds.is_valid() || !perm.is_valid() || !scratch.is_valid())
                return;
            JointParams params = joint_params(packed, bounds, perm, scratch, new_bounds, p);
            const auto slots = static_cast<size_t>(p.attributes_or_slots);
            const size_t tiles = (static_cast<size_t>(p.primitives) + kReorder - 1) / kReorder;
            const size_t slot_tile_bytes = kReorder * 4 * static_cast<size_t>(joint_adam::bytes_per_cell(p.bits));
            if (scratch.bytes() < tiles * slot_tile_bytes)
                throw std::invalid_argument(std::format("joint permute scratch holds {} bytes, one slot needs {}",
                                                        scratch.bytes(), tiles * slot_tile_bytes));
            const size_t slots_per_group = std::clamp<size_t>(scratch.bytes() / (tiles * slot_tile_bytes), 1, slots);
            mk::launch("morton_joint_bounds", params, {&packed, &bounds, &perm, &new_bounds}, blocks(p.primitives));
            for (size_t first = 0; first < slots; first += slots_per_group) {
                const size_t last = std::min(first + slots_per_group, slots);
                const size_t width = (last - first) * slot_tile_bytes;
                metal_sh_ops().fill_bytes(scratch, width * tiles, 0);
                params.slot_begin = static_cast<uint32_t>(first);
                params.slot_end = static_cast<uint32_t>(last);
                mk::launch_items("morton_joint_encode", params, {&packed, &bounds, &perm, &scratch, &new_bounds},
                                 static_cast<size_t>(p.primitives));
                mk::copy_rows(mk::address(scratch), mk::address(packed) + first * slot_tile_bytes, width, tiles, width,
                              slots * slot_tile_bytes, {&scratch, &packed});
            }
        }

        struct Q16Params {
            uint64_t source, source_bounds, permutation, destination_bounds, destination;
            uint32_t count, cells, cell_begin, cell_end;
        };

        // The Metal session lends no idle arena, so the destination is always
        // encoded in cell groups through an owned buffer of at most 64 MiB, as
        // CUDA does without an arena.
        void permute_sh_q16(core::SplatData& splat, const Tensor& perm, core::TensorExecutionTarget,
                            const IdleArenaScratch&) {
            auto& live = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
            const auto n = static_cast<size_t>(splat.size());
            const size_t n_cells = quant::sh_value_u16_count(n, rest);
            const size_t n_bound_floats = quant::n_bounds_for_prims(n) * 2;
            if (live.numel() < n_cells)
                throw std::runtime_error(std::format(
                    "Morton reorder: q16 shN storage holds {} codes, its logical size is {}", live.numel(), n_cells));
            if (!bounds.is_valid() || bounds.numel() < n_bound_floats)
                throw std::runtime_error(std::format(
                    "Morton reorder: shN_value_bounds hold {} floats, {} needed; refusing a silent SH wipe",
                    bounds.is_valid() ? bounds.numel() : 0, n_bound_floats));
            if (n == 0 || rest == 0)
                return;

            constexpr size_t kGroupBudget = 64ull << 20;
            const uint32_t cells = quant::n_value_cells_per_prim(rest);
            const size_t tiles = core::sh_swizzled_padded_n(n) / kReorder;
            const size_t cell_bytes = tiles * kReorder * sizeof(uint16_t);
            const size_t cells_per_group =
                std::min<size_t>(std::max<size_t>(1, kGroupBudget / cell_bytes), cells);
            Tensor dest_bounds = Tensor::zeros({n_bound_floats}, Device::GPU, DataType::Float32);
            Tensor group = Tensor::empty({cells_per_group * cell_bytes}, Device::GPU, DataType::UInt8);
            Q16Params params{mk::live_address(live), mk::live_address(bounds), mk::address(perm),
                             mk::address(dest_bounds), mk::address(group), static_cast<uint32_t>(n),
                             cells, 0, 0};
            mk::launch("morton_q16_bounds", params, {&live, &bounds, &perm, &dest_bounds},
                       static_cast<uint32_t>(quant::n_bounds_for_prims(n)));
            const uint64_t live_codes = mk::live_address(live);
            for (uint32_t first = 0; first < cells; first += static_cast<uint32_t>(cells_per_group)) {
                const uint32_t last = std::min(first + static_cast<uint32_t>(cells_per_group), cells);
                const size_t width = static_cast<size_t>(last - first) * kReorder * sizeof(uint16_t);
                group.zero_();
                params.cell_begin = first;
                params.cell_end = last;
                mk::launch_items("morton_q16_encode", params, {&live, &bounds, &perm, &dest_bounds, &group}, n);
                mk::copy_rows(mk::address(group), live_codes + first * kReorder * sizeof(uint16_t), width, tiles,
                              width, cells * kReorder * sizeof(uint16_t), {&group, &live});
            }
            bounds.flatten().slice(0, 0, n_bound_floats).copy_(dest_bounds);
            core::TensorCompletion completion;
            completion.include(live);
            completion.include(bounds);
            completion.wait();
        }

        void permute_sh_fp32(core::SplatData& splat, const Tensor& perm, core::TensorExecutionTarget,
                             const IdleArenaScratch&) {
            const bool expanded = sh_value::ensure_shN_fp32_for_mutation(splat);
            auto& live = splat.shN();
            const auto rest = static_cast<uint32_t>(splat.max_sh_coeffs_rest());
            const auto n = static_cast<size_t>(splat.size());
            if (!live.is_valid() || live.dtype() != DataType::Float32) {
                if (expanded)
                    (void)sh_value::commit_shN_after_mutation(splat);
                return;
            }
            const size_t logical = core::sh_swizzled_float_count(n, rest);
            if (live.numel() < logical)
                throw std::runtime_error(std::format(
                    "Morton reorder: shN storage holds {} floats, its logical size is {}", live.numel(), logical));
            Tensor gathered = Tensor::zeros_direct(TensorShape({logical}), logical, Device::GPU, DataType::Float32);
            metal_sh_ops().gather_swizzled(live, perm, gathered,
                                           {.source_rows = n, .count = n, .source_rest = rest, .destination_rest = rest});
            if (live.numel() == logical)
                live.copy_from(gathered);
            else
                live.slice(0, 0, logical).copy_from(gathered);
            if (expanded)
                (void)sh_value::commit_shN_after_mutation(splat);
        }

        void copy_back(Tensor& live, const void* source, const size_t bytes, core::TensorExecutionTarget) {
            mk::copy_rows(reinterpret_cast<uint64_t>(source), mk::address(live), bytes, 1, 0, 0, {&live});
        }

        void gather_gradient(const Tensor& source, const Tensor& perm, Tensor& destination, const uint32_t rest,
                             core::TensorExecutionTarget) {
            metal_sh_ops().gather_swizzled(source, perm, destination,
                                           {.source_rows = perm.numel(), .count = perm.numel(), .source_rest = rest});
        }
    } // namespace

    const lfs::gpu_ops::MortonOps& metal_morton_ops() {
        static const lfs::gpu_ops::MortonOps ops{
            .permutation = permutation,
            .permute_joint = permute_joint,
            .permute_joint_grouped = permute_joint_grouped,
            .permute_sh_q16 = permute_sh_q16,
            .permute_sh_fp32 = permute_sh_fp32,
            .copy_back = copy_back,
            .gather_gradient = gather_gradient,
        };
        return ops;
    }
} // namespace lfs::training
