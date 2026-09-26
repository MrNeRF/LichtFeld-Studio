/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/assert.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/sh_value_quant_kernels.hpp"
namespace lfs::core::internal {
    namespace {
        template <class T>
        T* data(StorageRef s) { return reinterpret_cast<T*>(static_cast<char*>(s.data) + s.byte_offset); }
    } // namespace
    void CudaBackendOps::sh_codec(StorageRef source, StorageRef destination, const ShCodecProgram& program, ExecContext context) {
        LFS_FACADE_TRACE(sh_codec);
        const auto& p = program.codec;
        const auto stream = context.cuda_stream;
        auto* out = data<float>(destination);
        const auto* src = data<float>(source);
        const auto* ids = program.indices ? data<int64_t>(*program.indices) : nullptr;
        if (p.destination_format == ShFormat::Q16) {
            if (p.source_format == ShFormat::Q16)
                sh_value_quant::encode_shN_u16_gathered(data<uint16_t>(source), data<float>(*program.source_bounds), ids,
                                                        data<uint16_t>(destination), data<float>(*program.destination_bounds), p.count, p.source_rows, p.source_rest, stream);
            else
                sh_value_quant::encode_shN_float4_to_u16(src, data<uint16_t>(destination), data<float>(*program.destination_bounds), p.count, p.source_rest, stream);
        } else if (p.source_format == ShFormat::Q16) {
            const auto* bounds = data<float>(*program.source_bounds);
            if (p.destination_format == ShFormat::Canonical) {
                if (ids)
                    sh_value_quant::decode_shN_u16_gathered_to_canonical(data<uint16_t>(source), bounds, ids, out, p.count, p.source_rows, p.source_rest, stream);
                else
                    sh_value_quant::decode_shN_u16_range_to_canonical(data<uint16_t>(source), bounds, out, p.source_offset * p.destination_rest * 3, p.count * p.destination_rest * 3, p.source_rows, p.destination_rest, p.source_rest, stream);
            } else if (ids)
                sh_value_quant::decode_shN_u16_gathered_to_float4(data<uint16_t>(source), bounds, ids, out, 0, p.count, p.source_rows, p.source_rest, stream);
            else
                sh_value_quant::decode_shN_u16_range_to_float4(data<uint16_t>(source), bounds, out, p.source_offset, p.count, p.source_rows, p.source_rest, stream);
        } else if (p.source_format == ShFormat::Float16) {
            sh_value_quant::decode_shN_f16_range_to_canonical(data<uint16_t>(source), out + p.destination_offset * p.destination_rest * 3,
                                                              p.source_offset * p.destination_rest * 3, p.count * p.destination_rest * 3, p.source_rows, p.destination_rest, p.source_rest, stream);
        } else if (p.source_format == ShFormat::Canonical) {
            src += p.source_offset * p.source_rest * 3;
            if (p.scatter)
                shN_swizzled_scatter_linear(out, data<int>(*program.indices), src, p.count, p.source_rest, p.destination_rest, stream);
            else if (p.destination_offset == 0 && p.count == p.destination_rows)
                reorder_sh_to_swizzled(src, out, p.count, p.source_rest, p.destination_rest, stream);
            else
                shN_swizzled_gather_from_linear(out, p.destination_offset, src, p.count, p.source_rest, p.destination_rest, stream);
        } else if (p.destination_format == ShFormat::Canonical) {
            out += p.destination_offset * p.destination_rest * 3;
            if (program.indices) {
                if (program.indices->dtype == DataType::Int64)
                    shN_swizzled_gather_to_linear_i64(src, ids, out, p.count, p.destination_rest, p.source_rest, stream);
                else
                    shN_swizzled_gather_to_linear(src, data<int>(*program.indices), out, p.count, p.destination_rest, p.source_rest, stream);
            } else if (p.source_offset == 0)
                undo_reorder_sh_from_swizzled(src, out, p.count, p.destination_rest, p.source_rest, stream);
            else
                undo_reorder_sh_range_from_swizzled(src, out, p.source_offset * p.destination_rest * 3, p.count * p.destination_rest * 3, p.source_rows, p.destination_rest, p.source_rest, stream);
        } else if (program.indices) {
            LFS_ASSERT_MSG(p.source_rest == p.destination_rest && !p.scatter, "CUDA swizzle gather requires equal layouts");
            if (program.indices->dtype == DataType::Int64)
                shN_swizzled_gather_self_i64(src, out, ids, p.count, p.destination_offset, p.source_rest, stream);
            else
                shN_swizzled_gather_self(src, out, data<int>(*program.indices), p.count, p.destination_offset, p.source_rest, stream);
        } else
            shN_swizzled_copy_range(src, out, p.source_offset, p.count, p.destination_offset, p.source_rest, p.destination_rest, stream);
    }
} // namespace lfs::core::internal
