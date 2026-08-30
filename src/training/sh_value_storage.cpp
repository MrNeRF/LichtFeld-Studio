/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/sh_value_storage.hpp"

#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lfs::training::sh_value {

    namespace {
        using core::DataType;
        using core::Device;
        using core::Tensor;
        using core::TensorShape;

        [[nodiscard]] std::uint32_t layout_rest(const core::SplatData& splat) {
            return static_cast<std::uint32_t>(splat.max_sh_coeffs_rest());
        }

        /// Primitive capacity for quant buffers: means capacity (max_cap), never exact-N only.
        [[nodiscard]] std::size_t prim_capacity(const core::SplatData& splat) {
            const auto n = static_cast<std::size_t>(splat.size());
            if (!splat.means().is_valid())
                return n;
            const auto cap = splat.means().capacity();
            return std::max(cap > 0 ? cap : n, n);
        }

        /// Full device barrier after encode/decode. Stream-only sync is not enough:
        /// densify runs on the strategy stream while the next forward may launch on
        // the training stream without an intervening wait (multi-stream UAF).
        void sync_codec_stream(cudaStream_t /*stream*/) {
            LFS_CUDA_CHECK_MSG(cudaDeviceSynchronize(), "sh_value quant codec device barrier");
        }

        [[nodiscard]] Tensor as_i64_indices(const Tensor& indices, cudaStream_t stream) {
            Tensor out = indices;
            if (out.dtype() != DataType::Int64) {
                out = out.to(DataType::Int64);
            }
            if (out.device() != Device::CUDA) {
                out = out.cuda();
            }
            if (!out.is_contiguous()) {
                out = out.contiguous();
            }
            if (out.stream() != stream) {
                out.set_stream(stream);
            }
            return out;
        }

        [[nodiscard]] Tensor make_range_i64(std::size_t start, std::size_t count, cudaStream_t stream) {
            Tensor out = Tensor::empty(TensorShape({count}), Device::CUDA, DataType::Int64);
            out.set_stream(stream);
            if (count == 0) {
                return out;
            }
            std::vector<std::int64_t> host(count);
            std::iota(host.begin(), host.end(), static_cast<std::int64_t>(start));
            LFS_CUDA_CHECK(cudaMemcpyAsync(
                out.ptr<std::int64_t>(),
                host.data(),
                count * sizeof(std::int64_t),
                cudaMemcpyHostToDevice,
                stream));
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "q16 append dest-index H2D");
            return out;
        }

        [[nodiscard]] Tensor canonical_contiguous(const Tensor& src, cudaStream_t stream) {
            Tensor out = src;
            if (out.device() != Device::CUDA) {
                out = out.cuda();
            }
            if (out.dtype() != DataType::Float32) {
                out = out.to(DataType::Float32);
            }
            if (!out.is_contiguous()) {
                out = out.contiguous();
            }
            if (out.stream() != stream) {
                out.set_stream(stream);
            }
            return out;
        }

        void copy_prefix_bytes(Tensor& dest, const Tensor& src, std::size_t nbytes, cudaStream_t stream) {
            if (nbytes == 0 || !src.is_valid() || src.numel() == 0) {
                return;
            }
            lfs::core::waitForCUDAStream(stream, src.stream());
            lfs::core::waitForCUDAStream(stream, dest.stream());
            LFS_CUDA_CHECK(cudaMemcpyAsync(
                dest.data_ptr(),
                src.data_ptr(),
                nbytes,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        void grow_q16_storage(core::SplatData& splat, std::size_t n_prims, cudaStream_t stream) {
            auto& shN = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            const auto rest = layout_rest(splat);
            const auto n_cells = core::sh_value_quant::sh_value_u16_count(n_prims, rest);
            const auto n_bound_floats = core::sh_value_quant::n_bounds_for_prims(n_prims) * 2;
            const auto cap_prims = prim_capacity(splat);
            const auto cap_cells = core::sh_value_quant::sh_value_u16_count(cap_prims, rest);
            const auto cap_bound_floats = core::sh_value_quant::n_bounds_for_prims(cap_prims) * 2;

            auto grow_1d = [&](Tensor& t,
                               std::size_t logical,
                               std::size_t cap,
                               DataType dtype,
                               std::string_view alloc_name,
                               const char* set_name) {
                if (!t.is_valid()) {
                    if (splat.has_tensor_allocator()) {
                        t = splat.allocate_named_param(
                            TensorShape({logical}), std::max(logical, cap), dtype, alloc_name);
                    } else {
                        t = Tensor::zeros_direct(
                            TensorShape({logical}), std::max(logical, cap), Device::CUDA, dtype);
                    }
                    t.set_name(set_name);
                    t.set_stream(stream);
                    return;
                }
                if (t.stream() != stream) {
                    t.set_stream(stream);
                }
                if (static_cast<std::size_t>(t.numel()) == logical && t.capacity() >= logical) {
                    return;
                }
                if (t.capacity() >= logical && t.capacity() > 0) {
                    if (static_cast<std::size_t>(t.numel()) < logical) {
                        t.append_zeros(logical - static_cast<std::size_t>(t.numel()));
                    }
                    return;
                }
                const std::size_t dest_cap = std::max(logical, cap);
                Tensor grown;
                if (splat.has_tensor_allocator()) {
                    grown = splat.allocate_named_param(
                        TensorShape({logical}), dest_cap, dtype, alloc_name);
                } else {
                    grown = Tensor::zeros_direct(
                        TensorShape({logical}), dest_cap, Device::CUDA, dtype);
                }
                grown.set_name(set_name);
                grown.set_stream(stream);
                if (grown.data_ptr() != t.data_ptr()) {
                    const std::size_t copy_n =
                        std::min(static_cast<std::size_t>(t.numel()), logical);
                    copy_prefix_bytes(grown, t, copy_n * core::dtype_size(dtype), stream);
                }
                t = std::move(grown);
            };

            grow_1d(shN, n_cells, cap_cells, DataType::Float16, "SplatData.shN", "splat.shN");
            grow_1d(bounds,
                    n_bound_floats,
                    cap_bound_floats,
                    DataType::Float32,
                    "SplatData.shN_value_bounds",
                    "splat.shN_value_bounds");
        }

        [[nodiscard]] Tensor make_fp32_chunk(std::uint32_t rest, cudaStream_t stream) {
            constexpr std::size_t kChunk = static_cast<std::size_t>(core::sh_value_quant::kBlockSize);
            const std::size_t chunk_floats = core::sh_swizzled_float_count(kChunk, rest);
            Tensor chunk = Tensor::zeros(TensorShape({chunk_floats}), Device::CUDA, DataType::Float32);
            chunk.set_stream(stream);
            return chunk;
        }

        void encode_chunk_into_q16(
            const Tensor& fp32_chunk,
            Tensor& live_u16,
            Tensor& live_bounds,
            std::size_t block_start,
            std::size_t n_in_block,
            std::uint32_t rest,
            cudaStream_t stream) {
            auto* dest_codes = reinterpret_cast<std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(live_u16));
            auto* dest_mm = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(live_bounds));
            core::sh_value_quant::encode_shN_float4_to_u16(
                fp32_chunk.ptr<float>(),
                dest_codes + core::sh_value_quant::sh_value_u16_count(block_start, rest),
                dest_mm + core::sh_value_quant::n_bounds_for_prims(block_start) * 2,
                n_in_block,
                rest,
                stream);
        }

        void decode_block_to_chunk(
            const Tensor& live_u16,
            const Tensor& live_bounds,
            Tensor& fp32_chunk,
            std::size_t block_start,
            std::size_t n_decode,
            std::size_t n_src,
            std::uint32_t rest,
            cudaStream_t stream) {
            const auto* src_u16 = reinterpret_cast<const std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(live_u16));
            const auto* src_bounds = static_cast<const float*>(
                lfs::core::resolve_exportable_device_ptr(live_bounds));
            core::sh_value_quant::decode_shN_u16_range_to_float4(
                src_u16,
                src_bounds,
                fp32_chunk.ptr<float>(),
                block_start,
                n_decode,
                n_src,
                rest,
                stream);
        }

        void encode_gathered_q16(
            core::SplatData& splat,
            const Tensor& perm,
            std::size_t n_src,
            std::size_t n_dst,
            std::size_t dest_cap_prims,
            cudaStream_t stream) {
            auto& live = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            const auto rest = layout_rest(splat);
            const std::size_t n_cells = core::sh_value_quant::sh_value_u16_count(n_dst, rest);
            const std::size_t n_bound_floats = core::sh_value_quant::n_bounds_for_prims(n_dst) * 2;
            const std::size_t cap_prims = dest_cap_prims > 0 ? dest_cap_prims : n_dst;
            const std::size_t cap_cells = core::sh_value_quant::sh_value_u16_count(cap_prims, rest);
            const std::size_t cap_bound_floats =
                core::sh_value_quant::n_bounds_for_prims(cap_prims) * 2;

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
                TensorShape({n_cells}), std::max(n_cells, cap_cells), Device::CUDA, DataType::Float16);
            dest_u16.set_stream(stream);
            dest_u16.set_name("splat.shN");
            Tensor dest_bounds = Tensor::zeros_direct(
                TensorShape({n_bound_floats}),
                std::max(n_bound_floats, cap_bound_floats),
                Device::CUDA,
                DataType::Float32);
            dest_bounds.set_stream(stream);
            dest_bounds.set_name("splat.shN_value_bounds");

            auto* dest_codes = reinterpret_cast<std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(dest_u16));
            auto* dest_mm = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(dest_bounds));
            core::sh_value_quant::encode_shN_u16_gathered(
                src_u16,
                src_bounds,
                perm_ptr,
                dest_codes,
                dest_mm,
                n_dst,
                n_src,
                rest,
                stream);

            live = std::move(dest_u16);
            bounds = std::move(dest_bounds);
        }

        void reencode_touched_q16_blocks(
            core::SplatData& splat,
            const Tensor& dest_indices_i64,
            const Tensor& src_canonical,
            std::size_t n_prims,
            std::size_t n_decode_src,
            cudaStream_t stream) {
            const std::size_t K = dest_indices_i64.numel();
            if (K == 0) {
                return;
            }
            const auto rest = layout_rest(splat);
            auto& live = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            if (live.stream() != stream) {
                live.set_stream(stream);
            }
            if (bounds.stream() != stream) {
                bounds.set_stream(stream);
            }

            Tensor block_ids = Tensor::zeros(TensorShape({K}), Device::CUDA, DataType::Float32);
            block_ids.set_stream(stream);
            core::sh_value_quant::fill_quant_block_ids_f32(
                dest_indices_i64.ptr<std::int64_t>(), block_ids.ptr<float>(), K, stream);
            auto sorted = block_ids.sort(0, false);
            Tensor order = std::move(sorted.second);
            Tensor sorted_dest = dest_indices_i64.index_select(0, order);
            Tensor sorted_can = src_canonical.index_select(0, order);
            if (!sorted_can.is_contiguous()) {
                sorted_can = sorted_can.contiguous();
            }
            sorted.first.set_stream(stream);
            order.set_stream(stream);
            sorted_dest.set_stream(stream);
            sorted_can.set_stream(stream);

            Tensor unique_blocks = Tensor::empty(TensorShape({K}), Device::CUDA, DataType::Int32);
            Tensor run_offsets = Tensor::empty(TensorShape({K}), Device::CUDA, DataType::Int32);
            Tensor n_runs = Tensor::zeros(TensorShape({1}), Device::CUDA, DataType::Int32);
            unique_blocks.set_stream(stream);
            run_offsets.set_stream(stream);
            n_runs.set_stream(stream);

            core::sh_value_quant::build_sorted_block_runs(
                sorted.first.ptr<float>(),
                unique_blocks.ptr<int>(),
                run_offsets.ptr<int>(),
                n_runs.ptr<int>(),
                K,
                stream);

            auto* codes = reinterpret_cast<std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(live));
            auto* mm = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(bounds));
            core::sh_value_quant::reencode_touched_q16_blocks(
                codes,
                mm,
                sorted_can.ptr<float>(),
                sorted_dest.ptr<std::int64_t>(),
                unique_blocks.ptr<int>(),
                run_offsets.ptr<int>(),
                n_runs.ptr<int>(),
                K,
                n_prims,
                n_decode_src,
                rest,
                stream);
        }
    } // namespace

    using core::DataType;
    using core::Device;
    using core::Tensor;
    using core::TensorShape;

    bool apply_shN_value_quant(core::SplatData& splat) {
        return splat.apply_shN_value_quant();
    }

    bool ensure_shN_fp32_for_mutation(core::SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("ensure_shN_fp32_for_mutation");
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float16)
            return false;

        const auto n = static_cast<std::size_t>(splat.size());
        const auto rest = layout_rest(splat);
        if (n == 0 || rest == 0)
            return false;

        const auto cap = prim_capacity(splat);
        const auto logical_floats = core::sh_swizzled_float_count(n, rest);
        const auto capacity_floats = core::sh_swizzled_float_count(cap, rest);

        // IEEE f16 float4-swizzle (standalone PLY/SOG viewer path, no bounds):
        // element-wise half→float cast. Training exportable is pad-dropped q16
        // (has bounds) and takes the decode path below.
        if (splat.shN_ieee_f16()) {
            Tensor fp32 = shN.to(DataType::Float32);
            if (fp32.device() != Device::CUDA)
                fp32 = fp32.cuda();
            if (!fp32.is_contiguous())
                fp32 = fp32.contiguous();
            // Preserve capacity headroom when possible.
            if (fp32.capacity() < capacity_floats) {
                Tensor room = Tensor::zeros_direct(TensorShape({logical_floats}),
                                                   capacity_floats, Device::CUDA);
                room.set_name("splat.shN");
                room.copy_from(fp32);
                fp32 = std::move(room);
            } else {
                fp32.set_name("splat.shN");
            }
            shN = std::move(fp32);
            splat.shN_value_bounds() = Tensor{};
            return true;
        }

        if (!sh_value_quant_enabled())
            return false;

        Tensor fp32 = Tensor::zeros_direct(TensorShape({logical_floats}),
                                           std::max(logical_floats, capacity_floats),
                                           Device::CUDA);
        fp32.set_name("splat.shN");

        const cudaStream_t stream = core::getCurrentCUDAStream();
        if (fp32.stream() != stream)
            fp32.set_stream(stream);
        if (shN.stream() != stream)
            shN.set_stream(stream);
        auto& bounds = splat.shN_value_bounds();
        if (!bounds.is_valid() ||
            bounds.numel() < core::sh_value_quant::n_bounds_for_prims(n) * 2) {
            // Rebuilding empty or zero bounds makes decode emit all zeros — a silent
            // SH wipe. Fail loud so densify/relocate never zero SH-rest by accident.
            LOG_ERROR("SH value quant expand refused: bounds are short for N={} "
                      "(have={} need={}). Refusing silent SH wipe; restore bounds or "
                      "dequant via a known-good checkpoint.",
                      n,
                      bounds.is_valid() ? bounds.numel() : 0,
                      core::sh_value_quant::n_bounds_for_prims(n) * 2);
            throw std::runtime_error(
                "ensure_shN_fp32_for_mutation: shN_value_bounds short/missing — "
                "refusing silent SH wipe");
        }
        if (bounds.stream() != stream)
            bounds.set_stream(stream);

        lfs::core::sh_value_quant::decode_shN_u16_to_float4(
            reinterpret_cast<const std::uint16_t*>(
                lfs::core::resolve_exportable_device_ptr(shN)),
            static_cast<const float*>(
                lfs::core::resolve_exportable_device_ptr(bounds)),
            fp32.ptr<float>(),
            n,
            rest,
            stream);
        sync_codec_stream(stream);

        shN = std::move(fp32);
        // drop bounds once codes are expanded. Leaving pad-dropped bounds
        // attached to a Float32 float4-swizzle buffer is a dual-representation
        // footgun (FastGS/viewer may treat Float16+bounds as q16; a later
        // partial rebind can re-install codes without matching bounds).
        splat.shN_value_bounds() = Tensor{};
        return true;
    }

    bool commit_shN_after_mutation(core::SplatData& splat) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("commit_shN_after_mutation");
        auto& shN = splat.shN();
        if (!shN.is_valid() || shN.dtype() != DataType::Float32)
            return false;

        if (!sh_value_quant_enabled())
            return false;
        // Single-buffer: rebuild codes+bounds into the live exportable q16 region
        // (allocate_named_param). The caller must already own the mutation guard or
        // trainer render exclusive; this helper's marker only covers nested work.
        return splat.apply_shN_value_quant();
    }

    void gather_shN_to_canonical(core::SplatData& splat,
                                 const Tensor& src_indices,
                                 Tensor& dest_canonical,
                                 std::size_t n_src_primitives) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("gather_shN_to_canonical");
        const auto rest = layout_rest(splat);
        if (rest == 0 || !src_indices.is_valid() || src_indices.numel() == 0) {
            return;
        }
        const cudaStream_t stream = core::getCurrentCUDAStream();
        const std::size_t K = src_indices.numel();
        const std::size_t n_src =
            n_src_primitives > 0 ? n_src_primitives : static_cast<std::size_t>(splat.size());
        Tensor indices = as_i64_indices(src_indices, stream);

        const std::size_t need = K * static_cast<std::size_t>(rest) * 3u;
        const bool in_place = dest_canonical.is_valid() &&
                              dest_canonical.device() == Device::CUDA &&
                              dest_canonical.dtype() == DataType::Float32 &&
                              dest_canonical.numel() >= need;
        Tensor dest = dest_canonical;
        if (!in_place) {
            dest = Tensor::zeros(
                TensorShape({K, static_cast<std::size_t>(rest), std::size_t{3}}),
                Device::CUDA,
                DataType::Float32);
        } else if (!dest.is_contiguous()) {
            dest = dest.contiguous();
        }
        dest.set_stream(stream);

        auto write_dest = [&](float* ptr) {
            if (splat.shN_value_quantized()) {
                auto& live = splat.shN();
                auto& bounds = splat.shN_value_bounds();
                if (!bounds.is_valid() ||
                    bounds.numel() < core::sh_value_quant::n_bounds_for_prims(n_src) * 2) {
                    throw std::runtime_error(
                        "gather_shN_to_canonical: shN_value_bounds short/missing — "
                        "refusing silent SH wipe");
                }
                if (live.stream() != stream) {
                    live.set_stream(stream);
                }
                if (bounds.stream() != stream) {
                    bounds.set_stream(stream);
                }
                core::sh_value_quant::decode_shN_u16_gathered_to_canonical(
                    reinterpret_cast<const std::uint16_t*>(
                        lfs::core::resolve_exportable_device_ptr(live)),
                    static_cast<const float*>(
                        lfs::core::resolve_exportable_device_ptr(bounds)),
                    indices.ptr<std::int64_t>(),
                    ptr,
                    K,
                    n_src,
                    rest,
                    stream);
                return;
            }
            if (splat.shN().dtype() != DataType::Float32) {
                throw std::runtime_error(
                    "gather_shN_to_canonical: expected q16 or fp32 swizzled shN");
            }
            core::shN_swizzled_gather_to_linear_i64(
                splat.shN().ptr<float>(),
                indices.ptr<std::int64_t>(),
                ptr,
                K,
                rest,
                rest);
        };
        write_dest(dest.ptr<float>());
        if (!in_place) {
            dest_canonical = std::move(dest);
        } else if (dest.data_ptr() != dest_canonical.data_ptr()) {
            dest_canonical.copy_from(dest);
        }
    }

    void scatter_canonical_into_shN(core::SplatData& splat,
                                    const Tensor& dest_indices,
                                    const Tensor& src_canonical) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("scatter_canonical_into_shN");
        const auto rest = layout_rest(splat);
        if (rest == 0 || !dest_indices.is_valid() || dest_indices.numel() == 0 ||
            !src_canonical.is_valid() || src_canonical.numel() == 0) {
            return;
        }
        const cudaStream_t stream = core::getCurrentCUDAStream();
        Tensor indices = as_i64_indices(dest_indices, stream);
        Tensor canonical = canonical_contiguous(src_canonical, stream);
        const std::size_t n = static_cast<std::size_t>(splat.size());

        if (splat.shN_value_quantized()) {
            reencode_touched_q16_blocks(splat, indices, canonical, n, n, stream);
            sync_codec_stream(stream);
            return;
        }
        if (splat.shN().dtype() != DataType::Float32) {
            throw std::runtime_error(
                "scatter_canonical_into_shN: expected q16 or fp32 swizzled shN");
        }
        Tensor dest_i32 = indices.dtype() == DataType::Int32 ? indices : indices.to(DataType::Int32);
        core::shN_swizzled_scatter_linear(
            splat.shN().ptr<float>(),
            dest_i32.ptr<int>(),
            canonical.ptr<float>(),
            indices.numel(),
            rest,
            rest);
    }

    void append_canonical_to_shN(core::SplatData& splat,
                                 const Tensor& src_canonical,
                                 std::size_t dest_offset) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("append_canonical_to_shN");
        const auto rest = layout_rest(splat);
        if (rest == 0 || !src_canonical.is_valid() || src_canonical.numel() == 0) {
            return;
        }
        const cudaStream_t stream = core::getCurrentCUDAStream();
        Tensor canonical = canonical_contiguous(src_canonical, stream);
        const std::size_t K = canonical.shape()[0];
        const std::size_t new_n = dest_offset + K;
        const std::size_t old_n = dest_offset;

        if (splat.shN_value_quantized()) {
            grow_q16_storage(splat, new_n, stream);
            auto& live = splat.shN();
            auto& bounds = splat.shN_value_bounds();
            Tensor fp32_chunk = make_fp32_chunk(rest, stream);
            constexpr std::size_t kChunk = static_cast<std::size_t>(core::sh_value_quant::kBlockSize);
            const std::size_t first_block = old_n / kChunk;
            const std::size_t last_block = (new_n - 1) / kChunk;
            for (std::size_t block = first_block; block <= last_block; ++block) {
                const std::size_t block_start = block * kChunk;
                const std::size_t n_in = std::min(kChunk, new_n - block_start);
                LFS_CUDA_CHECK(cudaMemsetAsync(
                    fp32_chunk.data_ptr(), 0, fp32_chunk.bytes(), stream));
                const std::size_t n_decode =
                    old_n > block_start ? std::min(n_in, old_n - block_start) : 0;
                if (n_decode > 0) {
                    decode_block_to_chunk(
                        live, bounds, fp32_chunk, block_start, n_decode, old_n, rest, stream);
                }
                const std::size_t ov_lo = std::max(block_start, dest_offset);
                const std::size_t ov_hi = std::min(block_start + n_in, new_n);
                if (ov_hi > ov_lo) {
                    const std::size_t row0 = ov_lo - dest_offset;
                    const std::size_t n_ov = ov_hi - ov_lo;
                    Tensor rows = canonical.slice(0, row0, row0 + n_ov);
                    if (!rows.is_contiguous()) {
                        rows = rows.contiguous();
                    }
                    core::shN_swizzled_gather_from_linear(
                        fp32_chunk.ptr<float>(),
                        ov_lo - block_start,
                        rows.ptr<float>(),
                        n_ov,
                        rest,
                        rest,
                        stream);
                }
                encode_chunk_into_q16(fp32_chunk, live, bounds, block_start, n_in, rest, stream);
            }
            sync_codec_stream(stream);
            return;
        }

        if (splat.shN().dtype() != DataType::Float32) {
            throw std::runtime_error(
                "append_canonical_to_shN: expected q16 or fp32 swizzled shN");
        }
        auto& shN_buf = splat.shN();
        const std::size_t needed_floats = core::sh_swizzled_float_count(new_n, rest);
        const std::size_t cap_floats = core::sh_swizzled_float_count(prim_capacity(splat), rest);
        if (shN_buf.capacity() < needed_floats) {
            const std::size_t dest_cap = std::max(needed_floats, cap_floats);
            auto grown = Tensor::zeros_direct(
                TensorShape({static_cast<std::size_t>(shN_buf.numel())}), dest_cap,
                shN_buf.device(), shN_buf.dtype());
            copy_prefix_bytes(grown, shN_buf, shN_buf.bytes(), stream);
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "fp32 shN grow copy");
            grown.set_name(shN_buf.name().empty() ? "splat.shN" : shN_buf.name());
            shN_buf = std::move(grown);
        }
        if (shN_buf.numel() < needed_floats) {
            shN_buf.append_zeros(needed_floats - shN_buf.numel());
        }
        core::shN_swizzled_gather_from_linear(
            shN_buf.ptr<float>(), dest_offset, canonical.ptr<float>(), K, rest, rest);
    }

    void zero_shN_at_indices(core::SplatData& splat, const Tensor& dest_indices) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("zero_shN_at_indices");
        const auto rest = layout_rest(splat);
        if (rest == 0 || !dest_indices.is_valid() || dest_indices.numel() == 0) {
            return;
        }
        Tensor zeros = Tensor::zeros(
            TensorShape({dest_indices.numel(), static_cast<std::size_t>(rest), std::size_t{3}}),
            Device::CUDA,
            DataType::Float32);
        scatter_canonical_into_shN(splat, dest_indices, zeros);
    }

    struct ShNMutationBatch::Impl {
        struct Op {
            enum class Kind { Scatter,
                              Zero,
                              Append } kind = Kind::Scatter;
            Tensor dest_indices;
            Tensor canonical;
            std::size_t dest_offset = 0;
        };

        core::SplatData* splat = nullptr;
        std::vector<Op> ops;
        bool flushed = true;
    };

    ShNMutationBatch::ShNMutationBatch(core::SplatData& splat)
        : impl_(std::make_unique<Impl>()) {
        impl_->splat = &splat;
    }

    ShNMutationBatch::ShNMutationBatch(ShNMutationBatch&&) noexcept = default;
    ShNMutationBatch& ShNMutationBatch::operator=(ShNMutationBatch&&) noexcept = default;

    ShNMutationBatch::~ShNMutationBatch() {
        if (!impl_ || impl_->flushed || impl_->ops.empty() || !impl_->splat) {
            return;
        }
        try {
            flush();
        } catch (const std::exception& e) {
            try {
                LOG_ERROR("ShNMutationBatch: flush failed during scope exit: {}", e.what());
            } catch (...) {
            }
        } catch (...) {
            try {
                LOG_ERROR("ShNMutationBatch: flush failed during scope exit with unknown exception");
            } catch (...) {
            }
        }
    }

    void ShNMutationBatch::scatter(const Tensor& dest_indices, const Tensor& src_canonical) {
        if (!impl_ || !impl_->splat) {
            return;
        }
        if (!dest_indices.is_valid() || dest_indices.numel() == 0 ||
            !src_canonical.is_valid() || src_canonical.numel() == 0 ||
            layout_rest(*impl_->splat) == 0) {
            return;
        }
        impl_->flushed = false;
        Impl::Op op;
        op.kind = Impl::Op::Kind::Scatter;
        op.dest_indices = dest_indices;
        op.canonical = src_canonical;
        impl_->ops.push_back(std::move(op));
    }

    void ShNMutationBatch::zero(const Tensor& dest_indices) {
        if (!impl_ || !impl_->splat) {
            return;
        }
        if (!dest_indices.is_valid() || dest_indices.numel() == 0 ||
            layout_rest(*impl_->splat) == 0) {
            return;
        }
        impl_->flushed = false;
        Impl::Op op;
        op.kind = Impl::Op::Kind::Zero;
        op.dest_indices = dest_indices;
        impl_->ops.push_back(std::move(op));
    }

    void ShNMutationBatch::append(const Tensor& src_canonical, std::size_t dest_offset) {
        if (!impl_ || !impl_->splat) {
            return;
        }
        if (!src_canonical.is_valid() || src_canonical.numel() == 0 ||
            layout_rest(*impl_->splat) == 0) {
            return;
        }
        impl_->flushed = false;
        Impl::Op op;
        op.kind = Impl::Op::Kind::Append;
        op.canonical = src_canonical;
        op.dest_offset = dest_offset;
        impl_->ops.push_back(std::move(op));
    }

    void ShNMutationBatch::flush() {
        if (!impl_ || impl_->flushed) {
            return;
        }
        impl_->flushed = true;
        if (!impl_->splat || impl_->ops.empty()) {
            return;
        }
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("ShNMutationBatch::flush");

        core::SplatData& splat = *impl_->splat;
        auto ops = std::move(impl_->ops);
        impl_->ops.clear();

        const auto apply_one = [&](const Impl::Op& op) {
            switch (op.kind) {
            case Impl::Op::Kind::Scatter:
                scatter_canonical_into_shN(splat, op.dest_indices, op.canonical);
                break;
            case Impl::Op::Kind::Zero:
                zero_shN_at_indices(splat, op.dest_indices);
                break;
            case Impl::Op::Kind::Append:
                append_canonical_to_shN(splat, op.canonical, op.dest_offset);
                break;
            }
        };

        if (ops.size() == 1 || !splat.shN_value_quantized()) {
            for (const auto& op : ops) {
                apply_one(op);
            }
            return;
        }

        const cudaStream_t stream = core::getCurrentCUDAStream();
        std::size_t n_prims = static_cast<std::size_t>(splat.size());
        std::size_t n_decode_src = std::numeric_limits<std::size_t>::max();
        std::vector<Tensor> dest_parts;
        std::vector<Tensor> can_parts;
        dest_parts.reserve(ops.size());
        can_parts.reserve(ops.size());

        for (const auto& op : ops) {
            switch (op.kind) {
            case Impl::Op::Kind::Scatter: {
                Tensor indices = as_i64_indices(op.dest_indices, stream);
                Tensor canonical = canonical_contiguous(op.canonical, stream);
                dest_parts.push_back(std::move(indices));
                can_parts.push_back(std::move(canonical));
                break;
            }
            case Impl::Op::Kind::Zero: {
                Tensor indices = as_i64_indices(op.dest_indices, stream);
                Tensor zeros = Tensor::zeros(
                    TensorShape({indices.numel(),
                                 static_cast<std::size_t>(layout_rest(splat)),
                                 std::size_t{3}}),
                    Device::CUDA,
                    DataType::Float32);
                zeros.set_stream(stream);
                dest_parts.push_back(std::move(indices));
                can_parts.push_back(std::move(zeros));
                break;
            }
            case Impl::Op::Kind::Append: {
                Tensor canonical = canonical_contiguous(op.canonical, stream);
                const std::size_t K = canonical.shape()[0];
                n_decode_src = std::min(n_decode_src, op.dest_offset);
                n_prims = std::max(n_prims, op.dest_offset + K);
                dest_parts.push_back(make_range_i64(op.dest_offset, K, stream));
                can_parts.push_back(std::move(canonical));
                break;
            }
            }
        }
        if (n_decode_src == std::numeric_limits<std::size_t>::max()) {
            n_decode_src = n_prims;
        }
        if (dest_parts.empty()) {
            return;
        }

        grow_q16_storage(splat, n_prims, stream);
        Tensor dests = dest_parts.size() == 1 ? dest_parts[0] : Tensor::cat(dest_parts, 0);
        Tensor cans = can_parts.size() == 1 ? can_parts[0] : Tensor::cat(can_parts, 0);
        if (dests.stream() != stream) {
            dests.set_stream(stream);
        }
        if (cans.stream() != stream) {
            cans.set_stream(stream);
        }
        if (!dests.is_contiguous()) {
            dests = dests.contiguous();
        }
        if (!cans.is_contiguous()) {
            cans = cans.contiguous();
        }
        reencode_touched_q16_blocks(splat, dests, cans, n_prims, n_decode_src, stream);
        sync_codec_stream(stream);
    }

    void compact_shN_gather(core::SplatData& splat,
                            const Tensor& keep_indices,
                            std::size_t n_src,
                            std::size_t dest_cap_prims) {
        LFS_ASSERT_LIVE_MODEL_MUTATION_LOCK_HELD();
        LiveModelMutationGuard mutation_guard("compact_shN_gather");
        const auto rest = layout_rest(splat);
        if (rest == 0 || !splat.shN().is_valid() || splat.shN().numel() == 0) {
            return;
        }
        const cudaStream_t stream = core::getCurrentCUDAStream();
        const std::size_t n_dst = keep_indices.is_valid() ? keep_indices.numel() : 0;
        if (n_dst == 0) {
            const std::size_t cap_prims = dest_cap_prims > 0 ? dest_cap_prims : 0;
            const std::size_t cap_cells = core::sh_value_quant::sh_value_u16_count(cap_prims, rest);
            const std::size_t cap_bounds =
                core::sh_value_quant::n_bounds_for_prims(cap_prims) * 2;
            if (splat.shN_value_quantized()) {
                splat.shN() = Tensor::zeros_direct(
                    TensorShape({std::size_t{0}}), cap_cells, Device::CUDA, DataType::Float16);
                splat.shN_value_bounds() = Tensor::zeros_direct(
                    TensorShape({std::size_t{0}}), cap_bounds, Device::CUDA, DataType::Float32);
            } else {
                const std::size_t cap_floats = core::sh_swizzled_float_count(cap_prims, rest);
                splat.shN() = Tensor::zeros_direct(
                    TensorShape({std::size_t{0}}), cap_floats, Device::CUDA, DataType::Float32);
            }
            splat.shN().set_name("splat.shN");
            return;
        }
        Tensor keep = as_i64_indices(keep_indices, stream);
        if (splat.shN_value_quantized()) {
            encode_gathered_q16(splat, keep, n_src, n_dst, dest_cap_prims, stream);
            sync_codec_stream(stream);
            return;
        }
        if (splat.shN().dtype() != DataType::Float32) {
            throw std::runtime_error("compact_shN_gather: expected q16 or fp32 swizzled shN");
        }
        Tensor idx_i32 = keep.to(DataType::Int32);
        const std::size_t cap_rows = dest_cap_prims > 0 ? dest_cap_prims : n_dst;
        const std::size_t cap_floats = core::sh_swizzled_float_count(cap_rows, rest);
        const std::size_t logical_floats = core::sh_swizzled_float_count(n_dst, rest);
        auto fresh = Tensor::zeros_direct(
            TensorShape({logical_floats}), cap_floats, Device::CUDA, DataType::Float32);
        core::shN_swizzled_gather_self(
            splat.shN().ptr<float>(),
            fresh.ptr<float>(),
            idx_i32.ptr<int>(),
            n_dst,
            0,
            rest);
        fresh.set_name("splat.shN");
        splat.shN() = std::move(fresh);
    }

    ShNCommitGuard::~ShNCommitGuard() noexcept {
        if (!expanded_ || !splat_) {
            return;
        }
        try {
            (void)commit_shN_after_mutation(*splat_);
        } catch (const std::exception& e) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit: {}", site_, e.what());
            } catch (...) {
            }
        } catch (...) {
            try {
                LOG_ERROR("{}: SH value commit failed during scope exit with unknown exception", site_);
            } catch (...) {
            }
        }
    }

} // namespace lfs::training::sh_value
