/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/sh_cuda.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "lfs/cuda_scratch.hpp"

#include <cstdint>
#include <stdexcept>

namespace lfs::training {
    namespace {

        using lfs::gpu_ops::BackendState;
        using lfs::gpu_ops::In;
        using lfs::gpu_ops::Out;
        using lfs::gpu_ops::Q16TouchParams;
        using lfs::gpu_ops::ShRangeParams;
        using lfs::gpu_ops::ShRowsParams;
        using lfs::gpu_ops::ShStorage;
        using lfs::gpu_ops::State;
        using lfs::gpu_ops::Tensor;

        struct Q16RunState final : BackendState {
            cuda_scratch::Q16BlockRunWorkspace workspace;
        };

        [[nodiscard]] void* device_ptr(const Tensor& tensor) {
            return lfs::core::resolve_exportable_device_ptr(tensor);
        }

        State create_run_scratch() {
            return std::make_unique<Q16RunState>();
        }

        void encode_q16(
            const Tensor& swizzled, Tensor& codes, Tensor& bounds,
            const size_t primitives, const uint32_t rest,
            const size_t code_offset, const size_t bounds_offset) {
            auto* dst_codes = static_cast<std::uint16_t*>(device_ptr(codes));
            auto* dst_bounds = static_cast<float*>(device_ptr(bounds));
            lfs::core::sh_value_quant::encode_shN_float4_to_u16(
                swizzled.ptr<float>(),
                dst_codes + code_offset,
                dst_bounds + bounds_offset,
                primitives,
                rest,
                lfs::core::getCurrentCUDAStream());
        }

        void decode_q16(
            const Tensor& codes, const Tensor& bounds, Tensor& swizzled,
            const size_t primitives, const uint32_t rest) {
            if (codes.has_exportable_provenance() && bounds.has_exportable_provenance() &&
                bounds.exportable_bound_generation() != codes.exportable_bound_generation()) {
                LOG_ERROR(
                    "q16 codes/bounds generation pair mismatch: codes_gen={} bounds_gen={}",
                    codes.exportable_bound_generation(),
                    bounds.exportable_bound_generation());
            }
            lfs::core::sh_value_quant::decode_shN_u16_to_float4(
                static_cast<const std::uint16_t*>(device_ptr(codes)),
                static_cast<const float*>(device_ptr(bounds)),
                static_cast<float*>(device_ptr(swizzled)),
                primitives,
                rest,
                lfs::core::getCurrentCUDAStream());
        }

        void block_ids(const Tensor& destination_indices, Tensor& block_ids_out) {
            lfs::core::sh_value_quant::fill_quant_block_ids_f32(
                destination_indices.ptr<std::int64_t>(),
                block_ids_out.ptr<float>(),
                destination_indices.numel(),
                lfs::core::getCurrentCUDAStream());
        }

        void block_runs(
            BackendState& state, const Tensor& sorted_block_ids,
            Tensor& unique_blocks, Tensor& offsets, Tensor& run_count) {
            auto& workspace = static_cast<Q16RunState&>(state).workspace;
            const auto stream = lfs::core::getCurrentCUDAStream();
            const size_t n = sorted_block_ids.numel();
            workspace.ensure(
                n,
                lfs::core::sh_value_quant::sorted_block_runs_scan_workspace_bytes(n, stream),
                stream);
            const lfs::core::sh_value_quant::SortedBlockRunScratch scratch{
                .flags = workspace.flags.ptr<std::int32_t>(),
                .compact = workspace.compact.ptr<std::int32_t>(),
                .scan = workspace.scan.data_ptr(),
                .scan_bytes = workspace.scan_bytes,
            };
            lfs::core::sh_value_quant::build_sorted_block_runs(
                sorted_block_ids.ptr<float>(),
                unique_blocks.ptr<int>(),
                offsets.ptr<int>(),
                run_count.ptr<int>(),
                n,
                stream,
                &scratch);
        }

        void reencode_touched(
            Tensor& codes, Tensor& bounds, const Tensor& canonical, const Tensor& destinations,
            const Tensor& unique_blocks, const Tensor& offsets, const Tensor& run_count,
            const Tensor& canonical_order, const Q16TouchParams& params) {
            lfs::core::sh_value_quant::reencode_touched_q16_blocks(
                static_cast<std::uint16_t*>(device_ptr(codes)),
                static_cast<float*>(device_ptr(bounds)),
                canonical.ptr<float>(),
                destinations.ptr<std::int64_t>(),
                unique_blocks.ptr<int>(),
                offsets.ptr<int>(),
                run_count.ptr<int>(),
                params.sorted_count,
                params.primitives,
                params.decode_source_rows,
                params.rest,
                lfs::core::getCurrentCUDAStream(),
                canonical_order.is_valid() ? canonical_order.ptr<std::int64_t>() : nullptr);
        }

        void decode_range(
            const Tensor& values, const Tensor& bounds, Tensor& canonical,
            const ShRangeParams& params) {
            const auto stream = lfs::core::getCurrentCUDAStream();
            auto* dst = static_cast<float*>(device_ptr(canonical));
            if (params.storage == ShStorage::Q16) {
                lfs::core::sh_value_quant::decode_shN_u16_range_to_canonical(
                    static_cast<const std::uint16_t*>(device_ptr(values)),
                    static_cast<const float*>(device_ptr(bounds)),
                    dst,
                    params.canonical_float_offset,
                    params.float_count,
                    params.primitives,
                    params.destination_rest,
                    params.layout_rest,
                    stream);
                return;
            }
            if (params.storage == ShStorage::IeeeFloat16) {
                lfs::core::sh_value_quant::decode_shN_f16_range_to_canonical(
                    static_cast<const std::uint16_t*>(device_ptr(values)),
                    dst,
                    params.canonical_float_offset,
                    params.float_count,
                    params.primitives,
                    params.destination_rest,
                    params.layout_rest,
                    stream);
                return;
            }
            lfs::core::undo_reorder_sh_range_from_swizzled(
                static_cast<const float*>(device_ptr(values)),
                dst,
                params.canonical_float_offset,
                params.float_count,
                params.primitives,
                params.destination_rest,
                params.layout_rest,
                stream);
        }

        void zero_rows(Tensor& values, const Tensor& indices, const uint32_t rest) {
            lfs::core::shN_swizzled_zero_at_indices(
                static_cast<float*>(device_ptr(values)),
                indices.ptr<int>(),
                indices.numel(),
                rest,
                lfs::core::getCurrentCUDAStream());
        }

        void gather_swizzled(
            const Tensor& source, const Tensor& indices, Tensor& destination,
            const ShRowsParams& params) {
            const auto stream = lfs::core::getCurrentCUDAStream();
            const bool i64 = indices.dtype() == lfs::core::DataType::Int64;
            if (source.dtype() == lfs::core::DataType::Float32) {
                auto* src = static_cast<const float*>(device_ptr(source));
                auto* dst = static_cast<float*>(device_ptr(destination));
                if (i64) {
                    lfs::core::shN_swizzled_gather_self_i64(
                        src, dst, indices.ptr<std::int64_t>(),
                        params.count, params.destination_offset, params.source_rest, stream);
                    return;
                }
                lfs::core::shN_swizzled_gather_self(
                    src, dst, indices.ptr<int>(),
                    params.count, params.destination_offset, params.source_rest, stream);
                return;
            }
            if (i64) {
                throw std::runtime_error("gather_swizzled: byte rows require int32 indices");
            }
            lfs::core::shN_swizzled_gather_self_u8(
                static_cast<const std::uint8_t*>(device_ptr(source)),
                static_cast<std::uint8_t*>(device_ptr(destination)),
                indices.ptr<int>(),
                params.count,
                params.destination_offset,
                params.source_rest,
                stream);
        }

        void gather_canonical(
            const Tensor& source, const Tensor& indices, Tensor& canonical,
            const ShRowsParams& params) {
            lfs::core::shN_swizzled_gather_to_linear_i64(
                static_cast<const float*>(device_ptr(source)),
                indices.ptr<std::int64_t>(),
                canonical.ptr<float>(),
                params.count,
                params.destination_rest,
                params.source_rest,
                lfs::core::getCurrentCUDAStream());
        }

        void append_canonical(
            const Tensor& canonical, Tensor& destination, const ShRowsParams& params) {
            lfs::core::shN_swizzled_gather_from_linear(
                static_cast<float*>(device_ptr(destination)),
                params.destination_offset,
                canonical.ptr<float>(),
                params.count,
                params.source_rest,
                params.destination_rest,
                lfs::core::getCurrentCUDAStream());
        }

        void scatter_canonical(
            const Tensor& canonical, const Tensor& indices, Tensor& destination,
            const ShRowsParams& params) {
            lfs::core::shN_swizzled_scatter_linear(
                static_cast<float*>(device_ptr(destination)),
                indices.ptr<int>(),
                canonical.ptr<float>(),
                params.count,
                params.source_rest,
                params.destination_rest,
                lfs::core::getCurrentCUDAStream());
        }

        void fill_bytes(Tensor& storage, const size_t byte_count, const uint8_t value) {
            const cudaError_t err = cudaMemsetAsync(
                storage.data_ptr(), value, byte_count, lfs::core::getCurrentCUDAStream());
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("sh fill_bytes: ") + cudaGetErrorString(err));
            }
        }

        const lfs::gpu_ops::ShOps kCudaShOps{
            .create_run_scratch = create_run_scratch,
            .encode_q16 = encode_q16,
            .decode_q16 = decode_q16,
            .block_ids = block_ids,
            .block_runs = block_runs,
            .reencode_touched = reencode_touched,
            .decode_range = decode_range,
            .zero_rows = zero_rows,
            .gather_swizzled = gather_swizzled,
            .gather_canonical = gather_canonical,
            .append_canonical = append_canonical,
            .scatter_canonical = scatter_canonical,
            .fill_bytes = fill_bytes,
        };

    } // namespace

    const lfs::gpu_ops::ShOps& cuda_sh_ops() {
        return kCudaShOps;
    }

} // namespace lfs::training
