/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal ShOps; see ops/sh_cuda.cpp for the reference behaviour.

#include "sh_metal.hpp"

#include "core/logger.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_exportable_storage.hpp"
#include "metal_families.hpp"
#include "metal_kernels.hpp"

#include <format>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::training {
    namespace metal {

        uint64_t live_address(const core::Tensor& tensor) {
            if (tensor.is_valid() && tensor.has_exportable_provenance())
                return reinterpret_cast<uint64_t>(core::resolve_exportable_device_ptr(tensor));
            return address(tensor);
        }

        struct CopyParams {
            uint64_t source, destination, width, rows, source_pitch, destination_pitch;
        };

        void copy_rows(const uint64_t source, const uint64_t destination, const size_t width, const size_t rows,
                       const size_t source_pitch, const size_t destination_pitch,
                       const std::initializer_list<const core::Tensor*> uses) {
            if (width == 0 || rows == 0)
                return;
            const CopyParams params{source, destination, width, rows, source_pitch, destination_pitch};
            launch_items("sh_copy_rows", params, uses, width * rows);
        }

    } // namespace metal

    namespace {
        using core::DataType;
        using lfs::gpu_ops::BackendState;
        using lfs::gpu_ops::Q16TouchParams;
        using lfs::gpu_ops::ShRangeParams;
        using lfs::gpu_ops::ShRowsParams;
        using lfs::gpu_ops::ShStorage;
        using lfs::gpu_ops::State;
        using lfs::gpu_ops::Tensor;
        namespace mk = metal;
        namespace quant = core::sh_value_quant;

        void require_indices(const Tensor& indices, const DataType dtype, const char* operation) {
            if (indices.dtype() != dtype)
                throw std::invalid_argument(std::format("{} indices must be {}, got {}", operation,
                                                        core::dtype_name(dtype), core::dtype_name(indices.dtype())));
        }

        // Block runs need no persistent workspace on Metal.
        struct RunScratch final : BackendState {};

        State create_run_scratch() { return std::make_unique<RunScratch>(); }

        struct EncodeParams {
            uint64_t source, codes, bounds;
            uint32_t primitives, slots, cells;
        };

        void encode_q16(const Tensor& swizzled, Tensor& codes, Tensor& bounds, const size_t primitives,
                        const uint32_t rest, const size_t code_offset, const size_t bounds_offset) {
            if (primitives == 0 || rest == 0)
                return;
            const EncodeParams params{mk::address(swizzled), mk::live_address(codes) + code_offset * sizeof(uint16_t),
                                      mk::live_address(bounds) + bounds_offset * sizeof(float),
                                      static_cast<uint32_t>(primitives), core::sh_float4_slots_for_rest(rest),
                                      quant::n_value_cells_per_prim(rest)};
            mk::launch("sh_encode_q16", params, {&swizzled, &codes, &bounds},
                       static_cast<uint32_t>(quant::n_bounds_for_prims(primitives)));
        }

        struct DecodeParams {
            uint64_t codes, bounds, destination;
            uint32_t primitives, slots, cells;
        };

        void decode_q16(const Tensor& codes, const Tensor& bounds, Tensor& swizzled, const size_t primitives,
                        const uint32_t rest) {
            if (codes.has_exportable_provenance() && bounds.has_exportable_provenance() &&
                bounds.exportable_bound_generation() != codes.exportable_bound_generation()) {
                LOG_ERROR("q16 codes/bounds generation pair mismatch: codes_gen={} bounds_gen={}",
                          codes.exportable_bound_generation(), bounds.exportable_bound_generation());
            }
            if (primitives == 0 || rest == 0)
                return;
            const DecodeParams params{mk::live_address(codes), mk::live_address(bounds), mk::live_address(swizzled),
                                      static_cast<uint32_t>(primitives), core::sh_float4_slots_for_rest(rest),
                                      quant::n_value_cells_per_prim(rest)};
            mk::launch_items("sh_decode_q16", params, {&codes, &bounds, &swizzled}, primitives);
        }

        struct BlockIdParams {
            uint64_t indices, ids;
            uint32_t count;
        };

        void block_ids(const Tensor& destination_indices, Tensor& ids) {
            const size_t n = destination_indices.numel();
            if (n == 0)
                return;
            require_indices(destination_indices, DataType::Int64, "q16 block_ids");
            const BlockIdParams params{mk::address(destination_indices), mk::address(ids), static_cast<uint32_t>(n)};
            mk::launch_items("sh_block_ids", params, {&destination_indices, &ids}, n);
        }

        struct RunParams {
            uint64_t keys, starts, inclusive, unique, offsets, run_count;
            uint32_t count;
        };

        // Run starts, their inclusive scan (the tensor cumsum, where CUDA runs a
        // CUB exclusive scan) and compaction.
        void block_runs(BackendState&, const Tensor& sorted_block_ids, Tensor& unique_blocks, Tensor& offsets,
                        Tensor& run_count) {
            const size_t n = sorted_block_ids.numel();
            if (n == 0)
                return;
            if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::out_of_range(std::format("q16 block runs take at most {} items, got {}",
                                                    std::numeric_limits<int>::max(), n));
            Tensor starts = Tensor::empty({n}, core::Device::GPU, DataType::Int32);
            RunParams params{mk::address(sorted_block_ids), mk::address(starts), 0, mk::address(unique_blocks),
                             mk::address(offsets), mk::address(run_count), static_cast<uint32_t>(n)};
            mk::launch_items("sh_run_starts", params, {&sorted_block_ids, &starts}, n);
            const Tensor inclusive = starts.cumsum(0);
            params.inclusive = mk::address(inclusive);
            mk::launch_items("sh_run_compact", params,
                             {&sorted_block_ids, &starts, &inclusive, &unique_blocks, &offsets, &run_count}, n);
        }

        struct ReencodeParams {
            uint64_t codes, bounds, canonical, canonical_rows, destinations, unique, offsets, run_count;
            uint32_t sorted_count, primitives, decode_rows, cells;
        };

        void reencode_touched(Tensor& codes, Tensor& bounds, const Tensor& canonical, const Tensor& destinations,
                              const Tensor& unique_blocks, const Tensor& offsets, const Tensor& run_count,
                              const Tensor& canonical_order, const Q16TouchParams& p) {
            if (p.sorted_count == 0 || p.rest == 0 || p.primitives == 0)
                return;
            require_indices(destinations, DataType::Int64, "q16 reencode_touched");
            constexpr size_t kMax = std::numeric_limits<uint32_t>::max();
            const ReencodeParams params{
                mk::live_address(codes),
                mk::live_address(bounds),
                mk::address(canonical),
                mk::address(canonical_order),
                mk::address(destinations),
                mk::address(unique_blocks),
                mk::address(offsets),
                mk::address(run_count),
                static_cast<uint32_t>(p.sorted_count),
                static_cast<uint32_t>(p.primitives),
                static_cast<uint32_t>(std::min(p.decode_source_rows, kMax)),
                quant::n_value_cells_per_prim(p.rest),
            };
            // One group per sorted row: the run count stays on the device.
            mk::launch("sh_reencode_touched", params,
                       {&codes, &bounds, &canonical, &canonical_order, &destinations, &unique_blocks, &offsets,
                        &run_count},
                       static_cast<uint32_t>(p.sorted_count));
        }

        void validate_range(const ShRangeParams& p) {
            if (p.primitives == 0 || p.destination_rest == 0)
                throw std::invalid_argument(std::format("SH range decode needs primitives and rest, got {} and {}",
                                                        p.primitives, p.destination_rest));
            const uint64_t per_primitive = uint64_t{p.destination_rest} * core::kShChannels;
            if (p.primitives > std::numeric_limits<uint64_t>::max() / per_primitive ||
                p.canonical_float_offset > p.primitives * per_primitive ||
                p.float_count > p.primitives * per_primitive - p.canonical_float_offset)
                throw std::out_of_range(std::format(
                    "SH range [{}, +{}) exceeds the canonical {} x {} floats", p.canonical_float_offset,
                    p.float_count, p.primitives, per_primitive));
        }

        struct RangeParams {
            uint64_t values, bounds, destination, count;
            uint32_t primitive_offset, cell_offset, floats_per_primitive, width, storage;
        };

        void decode_range(const Tensor& values, const Tensor& bounds, Tensor& canonical, const ShRangeParams& p) {
            if (p.float_count == 0)
                return;
            if (!values.is_valid() || !canonical.is_valid() || (p.storage == ShStorage::Q16 && !bounds.is_valid()))
                throw std::invalid_argument("SH range decode needs values, a destination and Q16 bounds");
            if (p.storage != ShStorage::Float32 && p.layout_rest < p.destination_rest)
                throw std::invalid_argument(std::format("SH range decode layout rest {} is below destination rest {}",
                                                        p.layout_rest, p.destination_rest));
            validate_range(p);
            const uint32_t slots = core::sh_float4_slots_for_rest(p.layout_rest);
            if (p.storage != ShStorage::Q16 && slots == 0)
                throw std::invalid_argument("SH range decode has no source slots");
            // Divide the absolute 64-bit offset once on the host. Each shader
            // lane only needs a 32-bit local quotient and a bounded carry.
            const uint32_t floats_per_primitive = p.destination_rest * core::kShChannels;
            const RangeParams params{
                .values = mk::live_address(values),
                .bounds = p.storage == ShStorage::Q16 ? mk::live_address(bounds) : 0,
                .destination = mk::live_address(canonical),
                .count = p.float_count,
                .primitive_offset = static_cast<uint32_t>(p.canonical_float_offset / floats_per_primitive),
                .cell_offset = static_cast<uint32_t>(p.canonical_float_offset % floats_per_primitive),
                .floats_per_primitive = floats_per_primitive,
                .width = p.storage == ShStorage::Q16 ? quant::n_value_cells_per_prim(p.layout_rest) : slots,
                .storage = p.storage == ShStorage::Q16 ? 2u : p.storage == ShStorage::IeeeFloat16 ? 1u
                                                                                                  : 0u,
            };
            mk::launch_items("sh_decode_range", params, {&values, &bounds, &canonical}, p.float_count);
        }

        struct ZeroParams {
            uint64_t values, indices;
            uint32_t count, slots;
        };

        void zero_rows(Tensor& values, const Tensor& indices, const uint32_t rest) {
            const uint32_t slots = core::sh_float4_slots_for_rest(rest);
            if (indices.numel() == 0 || slots == 0)
                return;
            require_indices(indices, DataType::Int32, "sh zero_rows");
            const ZeroParams params{mk::live_address(values), mk::address(indices),
                                    mk::count32(indices.numel(), "SH indices"), slots};
            mk::launch_items("sh_zero_rows", params, {&values, &indices}, indices.numel());
        }

        struct GatherParams {
            uint64_t source, destination, indices;
            uint32_t count, destination_offset, slots, wide_indices, slot_bytes;
        };

        void gather_swizzled(const Tensor& source, const Tensor& indices, Tensor& destination, const ShRowsParams& p) {
            const bool wide = indices.dtype() == DataType::Int64;
            const bool floats = source.dtype() == DataType::Float32;
            if (!floats && wide)
                throw std::runtime_error("gather_swizzled: byte rows require int32 indices");
            const uint32_t slots = core::sh_float4_slots_for_rest(p.source_rest);
            if (p.count == 0 || slots == 0)
                return;
            const GatherParams params{mk::live_address(source),
                                      mk::live_address(destination),
                                      mk::address(indices),
                                      static_cast<uint32_t>(p.count),
                                      static_cast<uint32_t>(p.destination_offset),
                                      slots,
                                      wide ? 1u : 0u,
                                      floats ? 16u : 4u};
            mk::launch_items("sh_gather_swizzled", params, {&source, &indices, &destination}, p.count);
        }

        struct CanonicalParams {
            uint64_t swizzled, canonical, indices;
            uint32_t count, destination_offset, active_floats, slots, wide_indices;
        };

        void gather_canonical(const Tensor& source, const Tensor& indices, Tensor& canonical, const ShRowsParams& p) {
            const uint32_t slots = core::sh_float4_slots_for_rest(p.source_rest);
            if (p.count == 0 || p.destination_rest == 0 || slots == 0)
                return;
            require_indices(indices, DataType::Int64, "sh gather_canonical");
            const CanonicalParams params{mk::live_address(source),
                                         mk::address(canonical),
                                         mk::address(indices),
                                         static_cast<uint32_t>(p.count),
                                         0,
                                         p.destination_rest * core::kShChannels,
                                         slots,
                                         1u};
            mk::launch_items("sh_gather_canonical", params, {&source, &indices, &canonical}, p.count);
        }

        void write_canonical(const Tensor& canonical, const Tensor* indices, Tensor& destination,
                             const ShRowsParams& p) {
            const uint32_t slots = core::sh_float4_slots_for_rest(p.destination_rest);
            if (p.count == 0 || slots == 0)
                return;
            if (indices != nullptr)
                require_indices(*indices, DataType::Int32, "sh scatter_canonical");
            const CanonicalParams params{mk::live_address(destination),
                                         mk::address(canonical),
                                         indices != nullptr ? mk::address(*indices) : 0,
                                         static_cast<uint32_t>(p.count),
                                         static_cast<uint32_t>(p.destination_offset),
                                         p.source_rest * core::kShChannels,
                                         slots,
                                         0u};
            mk::launch_items("sh_write_canonical", params, {&canonical, indices, &destination}, p.count);
        }

        void append_canonical(const Tensor& canonical, Tensor& destination, const ShRowsParams& p) {
            write_canonical(canonical, nullptr, destination, p);
        }

        void scatter_canonical(const Tensor& canonical, const Tensor& indices, Tensor& destination,
                               const ShRowsParams& p) {
            write_canonical(canonical, &indices, destination, p);
        }

        struct FillParams {
            uint64_t destination, count;
            uint32_t value;
        };

        void fill_bytes(Tensor& storage, const size_t byte_count, const uint8_t value) {
            if (byte_count == 0)
                return;
            const FillParams params{mk::address(storage), byte_count, value};
            mk::launch_items("sh_fill_bytes", params, {&storage}, byte_count);
        }

        // Metal tensors cannot view raw device memory, so the result is an owned
        // concatenation whose bytes are also written to `data`.
        Tensor concatenate_into_arena(const std::span<const Tensor> parts, char* data, const core::TensorShape shape,
                                      const core::DataType dtype, core::TensorExecutionTarget) {
            for (const Tensor& part : parts) {
                if (!part.is_contiguous() || part.device() != core::Device::GPU || part.dtype() != dtype)
                    throw std::invalid_argument(std::format(
                        "arena parts must be contiguous GPU {} tensors, got {} {}", core::dtype_name(dtype),
                        part.is_contiguous() ? "contiguous" : "strided", core::dtype_name(part.dtype())));
            }
            Tensor result = Tensor::cat(std::vector<Tensor>(parts.begin(), parts.end()), 0);
            if (result.numel() != shape.elements())
                throw std::invalid_argument(std::format("arena parts hold {} elements, the arena shape {}",
                                                        result.numel(), shape.elements()));
            result = result.reshape(shape);
            mk::copy_rows(mk::address(result), reinterpret_cast<uint64_t>(data), result.bytes(), 1, 0, 0, {&result});
            return result;
        }
    } // namespace

    const lfs::gpu_ops::ShOps& metal_sh_ops() {
        static const lfs::gpu_ops::ShOps ops{
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
            .concatenate_into_arena = concatenate_into_arena,
        };
        return ops;
    }
} // namespace lfs::training
