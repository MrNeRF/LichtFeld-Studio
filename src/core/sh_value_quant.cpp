/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/sh_value_quant.hpp"

#include "core/assert.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <format>
#include <utility>
#include <vector>

namespace lfs::core::sh_value_quant {
    namespace {
        std::atomic<int>& override_flag() {
            static std::atomic<int> g{-1};
            return g;
        }

        constexpr std::uint32_t kMaxEncodeCells = 48;
        constexpr float kQMax = 65535.0f;
        constexpr float kEps = 1e-20f;
        constexpr float kInactiveLo = 1e30f;
        constexpr float kInactiveHi = -1e30f;

        Tensor pad_dim0(const Tensor& src, const size_t extra) {
            if (extra == 0) {
                return src;
            }
            MovementArgs args;
            args.args = std::vector<std::pair<int, int>>{
                {0, static_cast<int>(extra)}};
            return src.movement(MovementOp::Pad, args);
        }

        Tensor pad_last_dim(const Tensor& src, const size_t extra) {
            if (extra == 0) {
                return src;
            }
            std::vector<std::pair<int, int>> padding(src.ndim(), {0, 0});
            padding.back() = {0, static_cast<int>(extra)};
            MovementArgs args;
            args.args = std::move(padding);
            return src.movement(MovementOp::Pad, args);
        }

        Tensor gather_block_cells(const Tensor& src_in,
                                  const std::size_t n_primitives,
                                  const std::uint32_t slots_per_prim,
                                  const std::uint32_t n_encode,
                                  const std::size_t n_quant_blocks) {
            const size_t n_needed_32 = n_quant_blocks * (static_cast<size_t>(kBlockSize) /
                                                         static_cast<size_t>(kShReorderSize));
            if (slots_per_prim == 0) {
                return internal::allocate_zeros_like(
                    src_in,
                    TensorShape({n_quant_blocks, static_cast<size_t>(kBlockSize),
                                 static_cast<size_t>(n_encode)}),
                    DataType::Float32);
            }

            Tensor src = src_in;
            if (src.dtype() == DataType::Float16) {
                src = src.to(DataType::Float32);
            }
            LFS_ASSERT_MSG(src.dtype() == DataType::Float32,
                           "SH q16 tensor encode requires Float32 or Float16 source");
            src = src.contiguous().reshape(TensorShape({src.numel()}));

            const size_t floats_per_32 =
                static_cast<size_t>(slots_per_prim) * kShReorderSize * 4u;
            const size_t n_32_src = sh_swizzled_block_count(n_primitives);
            const size_t need_floats = n_32_src * floats_per_32;
            LFS_ASSERT_MSG(src.numel() >= need_floats,
                           std::format("SH q16 tensor encode source has {} floats, need {}",
                                       src.numel(), need_floats));
            if (src.numel() > need_floats) {
                src = src.slice(0, 0, need_floats);
            }

            src = src.reshape(TensorShape({n_32_src,
                                           static_cast<size_t>(slots_per_prim),
                                           static_cast<size_t>(kShReorderSize),
                                           4}));
            if (n_32_src < n_needed_32) {
                src = pad_dim0(src, n_needed_32 - n_32_src);
            } else if (n_32_src > n_needed_32) {
                src = src.slice(0, 0, n_needed_32);
            }

            src = src.reshape(TensorShape({n_quant_blocks,
                                           n_needed_32 / n_quant_blocks,
                                           static_cast<size_t>(slots_per_prim),
                                           static_cast<size_t>(kShReorderSize),
                                           4}));
            src = src.permute({0, 1, 3, 2, 4}).contiguous();
            const size_t packed_cells = static_cast<size_t>(slots_per_prim) * 4u;
            src = src.reshape(TensorShape({n_quant_blocks,
                                           static_cast<size_t>(kBlockSize),
                                           packed_cells}));
            if (packed_cells > n_encode) {
                return src.slice(2, 0, n_encode).contiguous();
            }
            if (packed_cells < n_encode) {
                return pad_last_dim(src, static_cast<size_t>(n_encode) - packed_cells);
            }
            return src;
        }
    } // namespace

    void set_enabled_for_testing(const std::optional<bool> enabled) {
        if (!enabled.has_value()) {
            override_flag().store(-1, std::memory_order_relaxed);
            return;
        }
        override_flag().store(*enabled ? 1 : 0, std::memory_order_relaxed);
    }

    bool enabled() {
        const int o = override_flag().load(std::memory_order_relaxed);
        if (o >= 0)
            return o != 0;
        return true; // production default: always ON
    }

    void encode_shN_float4_to_u16_tensor(
        const Tensor& src_float4_swizzled,
        const std::size_t n_primitives,
        const std::uint32_t slots_per_prim,
        const std::uint32_t n_cells_per_prim,
        Tensor& codes_out,
        Tensor& bounds_out) {
        LFS_ASSERT_MSG(src_float4_swizzled.is_valid(),
                       "SH q16 tensor encode requires a valid source tensor");

        const GpuBackend backend =
            gpu_backend_of(src_float4_swizzled).value_or(GpuBackend::CUDA);
        GpuBackendScope scope(backend);

        if (n_primitives == 0 || n_cells_per_prim == 0) {
            codes_out = internal::allocate_like(
                src_float4_swizzled, TensorShape({0}), DataType::Float16);
            bounds_out = internal::allocate_like(
                src_float4_swizzled, TensorShape({0, 2}), DataType::Float32);
            codes_out.set_stream(src_float4_swizzled.stream());
            bounds_out.set_stream(src_float4_swizzled.stream());
            return;
        }

        const auto n_encode =
            n_cells_per_prim > kMaxEncodeCells ? kMaxEncodeCells : n_cells_per_prim;
        const size_t n_quant_blocks = n_bounds_for_prims(n_primitives);
        const size_t n_32_out = sh_swizzled_block_count(n_primitives);

        Tensor cells = gather_block_cells(
            src_float4_swizzled, n_primitives, slots_per_prim, n_encode, n_quant_blocks);
        cells.set_stream(src_float4_swizzled.stream());

        Tensor prim_idx = Tensor::arange(
            static_cast<float>(n_quant_blocks * static_cast<size_t>(kBlockSize)));
        prim_idx.set_stream(src_float4_swizzled.stream());
        prim_idx = prim_idx.reshape(TensorShape(
            {n_quant_blocks, static_cast<size_t>(kBlockSize)}));
        const Tensor valid = prim_idx.lt(static_cast<float>(n_primitives));
        const Tensor valid_3d =
            valid.unsqueeze(-1).expand(cells.shape()).contiguous();

        const Tensor lo_fill = internal::allocate_like(
            cells, TensorShape({1}), DataType::Float32, kInactiveLo);
        const Tensor hi_fill = internal::allocate_like(
            cells, TensorShape({1}), DataType::Float32, kInactiveHi);
        const Tensor zero = internal::allocate_like(
            cells, TensorShape({1}), DataType::Float32, 0.0f);

        Tensor lo = Tensor::where(valid_3d, cells, lo_fill).min(2, false).min(1, false);
        Tensor hi = Tensor::where(valid_3d, cells, hi_fill).max(2, false).max(1, false);
        const Tensor empty_block = lo.gt(hi);
        lo = Tensor::where(empty_block, zero, lo);
        hi = Tensor::where(empty_block, zero, hi);

        const Tensor lo_b = lo.unsqueeze(-1).unsqueeze(-1);
        const Tensor hi_b = hi.unsqueeze(-1).unsqueeze(-1);
        const Tensor range = hi_b.sub(lo_b).clamp_min(kEps);
        const Tensor scale = cells.sub(lo_b).mul(kQMax).div(range);
        const Tensor scale_floor = scale.floor();
        const Tensor frac = scale.sub(scale_floor);
        Tensor qf = Tensor::where(frac.lt(0.5f), scale_floor, scale_floor.add(1.0f))
                        .clamp(0.0f, kQMax);
        qf = Tensor::where(valid_3d, qf, zero);
        if (n_encode < n_cells_per_prim) {
            qf = pad_last_dim(qf, static_cast<size_t>(n_cells_per_prim - n_encode));
        }

        const size_t groups = static_cast<size_t>(kBlockSize) / kShReorderSize;
        qf = qf.reshape(TensorShape({n_quant_blocks, groups,
                                     static_cast<size_t>(kShReorderSize),
                                     static_cast<size_t>(n_cells_per_prim)}));
        qf = qf.permute({0, 1, 3, 2}).contiguous();
        qf = qf.reshape(TensorShape({n_quant_blocks * groups,
                                     static_cast<size_t>(n_cells_per_prim),
                                     static_cast<size_t>(kShReorderSize)}));
        if (qf.size(0) > n_32_out) {
            qf = qf.slice(0, 0, n_32_out);
        }
        qf = qf.contiguous().reshape(TensorShape({qf.numel()}));

        const size_t n_code_cells = qf.numel();
        const Tensor high_f = qf.mul(1.0f / 256.0f).floor();
        const Tensor low_f = qf.sub(high_f.mul(256.0f));
        Tensor packed = Tensor::cat(
                            {low_f.reshape(TensorShape({1, n_code_cells})),
                             high_f.reshape(TensorShape({1, n_code_cells}))},
                            0)
                            .t()
                            .contiguous()
                            .to(DataType::UInt8);
        codes_out = packed.view_as(DataType::Float16);
        bounds_out = Tensor::cat(
                         {lo.reshape(TensorShape({1, n_quant_blocks})),
                          hi.reshape(TensorShape({1, n_quant_blocks}))},
                         0)
                         .t()
                         .contiguous();
        codes_out.set_stream(src_float4_swizzled.stream());
        bounds_out.set_stream(src_float4_swizzled.stream());
    }
} // namespace lfs::core::sh_value_quant
