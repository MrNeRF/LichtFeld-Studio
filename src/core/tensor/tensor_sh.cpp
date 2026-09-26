/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_sh.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/detail/tensor_half.hpp"
#include "core/sh_value_quant.hpp"
#include "internal/sh_codec.hpp"
#include "internal/tensor_impl.hpp"
#include <limits>
#include <tbb/parallel_for.h>

namespace lfs::core {
    ShFormat sh_storage_format(const Tensor& values, const Tensor& bounds) {
        return bounds.is_valid() && bounds.numel()   ? ShFormat::Q16
               : values.dtype() == DataType::Float16 ? ShFormat::Float16
                                                     : ShFormat::Float32;
    }

    void sh_codec(const Tensor& source, Tensor& destination, const ShCodec& p,
                  const Tensor* indices, const Tensor* source_bounds, Tensor* destination_bounds) {
        const auto elements = [](size_t rows, uint32_t rest, ShFormat format) {
            return format == ShFormat::Canonical ? rows * rest * 3
                   : format == ShFormat::Q16     ? sh_value_quant::sh_value_u16_count(rows, rest)
                                                 : sh_swizzled_float_count(rows, rest);
        };
        const auto check = [&](const Tensor& t, size_t rows, uint32_t rest, ShFormat format) {
            LFS_ASSERT_MSG(t.is_valid() && t.is_contiguous() && (t.numel() == 0 || !t.has_zero_stride()),
                           "sh_codec requires contiguous storage");
            LFS_ASSERT_MSG(rest <= 15 && rows <= size_t(std::numeric_limits<int32_t>::max()),
                           "sh_codec layout exceeds supported size");
            LFS_ASSERT_MSG(format <= ShFormat::Q16 &&
                               t.dtype() == (format == ShFormat::Float16 || format == ShFormat::Q16
                                                 ? DataType::Float16
                                                 : DataType::Float32) &&
                               t.numel() >= elements(rows, rest, format),
                           "sh_codec storage does not match its declared layout");
        };
        check(source, p.source_rows, p.source_rest, p.source_format);
        check(destination, p.destination_rows, p.destination_rest, p.destination_format);
        LFS_ASSERT_MSG(source.device() == destination.device(), "sh_codec requires the same device");
        internal::require_same_gpu_backend(source, destination, "sh_codec");
        LFS_ASSERT_MSG(p.source_offset <= p.source_rows && p.destination_offset <= p.destination_rows &&
                           (indices && !p.scatter ? p.count <= indices->numel() : p.count <= p.source_rows - p.source_offset) &&
                           (indices && p.scatter ? p.count <= indices->numel() : p.count <= p.destination_rows - p.destination_offset),
                       "sh_codec row range exceeds storage");
        LFS_ASSERT_MSG(!p.scatter || indices, "sh_codec scatter requires indices");
        LFS_ASSERT_MSG(!(p.source_format == ShFormat::Canonical && p.destination_format == ShFormat::Canonical) &&
                           (p.source_format != ShFormat::Canonical || !indices || p.scatter) &&
                           (!p.scatter || (p.source_format == ShFormat::Canonical && p.destination_format != ShFormat::Canonical)),
                       "sh_codec scatter accepts canonical rows into resident storage");
        if (indices) {
            LFS_ASSERT_MSG(indices->is_valid() && indices->is_contiguous() && indices->ndim() == 1 &&
                               (indices->dtype() == DataType::Int32 || indices->dtype() == DataType::Int64),
                           "sh_codec requires contiguous Int32/Int64 indices");
        }
        const auto check_bounds = [](const Tensor* bounds, size_t rows) {
            LFS_ASSERT_MSG(bounds && bounds->is_valid() && bounds->is_contiguous() &&
                               bounds->dtype() == DataType::Float32 &&
                               bounds->numel() >= sh_value_quant::n_bounds_for_prims(rows) * 2,
                           "sh_codec requires per-256-row Float32 bounds");
        };
        if (p.source_format == ShFormat::Q16)
            check_bounds(source_bounds, p.source_rows);
        if (p.destination_format == ShFormat::Q16) {
            check_bounds(destination_bounds, p.destination_rows);
            LFS_ASSERT_MSG(!p.scatter && p.destination_offset == 0 && p.count == p.destination_rows,
                           "sh_codec Q16 output must cover the complete destination");
        }
        for (const auto* input : {indices, source_bounds, static_cast<const Tensor*>(destination_bounds)}) {
            if (!input)
                continue;
            LFS_ASSERT_MSG(input->device() == source.device(), "sh_codec requires the same device");
            internal::require_same_gpu_backend(source, *input, "sh_codec");
        }
        if (destination_bounds) {
            LFS_ASSERT_MSG(!internal::shares_storage(*destination_bounds, destination) &&
                               !internal::shares_storage(*destination_bounds, source) &&
                               (!source_bounds || !internal::shares_storage(*destination_bounds, *source_bounds)) &&
                               (!indices || !internal::shares_storage(*destination_bounds, *indices)),
                           "sh_codec output bounds must not alias any operand");
            internal::preserve_lazy_snapshots_before_write(*destination_bounds);
        }
        internal::preserve_lazy_snapshots_before_write(destination);
        Tensor src = internal::shares_storage(source, destination) ? source.clone() : source;
        Tensor ids = indices ? (internal::shares_storage(*indices, destination) ? indices->clone() : *indices) : Tensor{};
        Tensor bounds = source_bounds ? (internal::shares_storage(*source_bounds, destination) ? source_bounds->clone() : *source_bounds) : Tensor{};
        if (!p.count || !p.destination_rest)
            return;

        if (source.device() == Device::GPU) {
            const GpuBackendScope scope(*gpu_backend_of(source));
            if (gpu_backend_of(source) == GpuBackend::CUDA) {
                if (p.destination_format != ShFormat::Canonical && !p.scatter &&
                    p.destination_offset == 0 && p.count == p.destination_rows && p.destination_rows % 32) {
                    const size_t width = p.destination_format == ShFormat::Q16 ? p.destination_rest * 3 : (p.destination_rest * 3 + 3) / 4 * 4;
                    destination.flatten().slice(0, (p.destination_rows / 32) * 32 * width, ((p.destination_rows + 31) / 32) * 32 * width).zero_();
                }
                if (indices && !p.scatter && p.source_format == ShFormat::Float32 &&
                    p.destination_format == ShFormat::Float32 && p.source_rest != p.destination_rest) {
                    Tensor tmp = internal::allocate_like(source, {p.count, size_t(p.destination_rest), size_t{3}}, DataType::Float32);
                    auto next = p;
                    next.destination_format = ShFormat::Canonical;
                    next.destination_rows = p.count;
                    next.destination_offset = 0;
                    sh_codec(src, tmp, next, &ids);
                    next.source_format = ShFormat::Canonical;
                    next.destination_format = ShFormat::Float32;
                    next.source_rows = p.count;
                    next.source_rest = p.destination_rest;
                    next.source_offset = 0;
                    next.destination_rows = p.destination_rows;
                    next.destination_offset = p.destination_offset;
                    sh_codec(tmp, destination, next);
                    return;
                }
                if (p.destination_format == ShFormat::Float16) {
                    Tensor tmp = !p.scatter && p.destination_offset == 0 && p.count == p.destination_rows
                                     ? internal::allocate_zeros_like(destination, {sh_swizzled_float_count(p.destination_rows, p.destination_rest)}, DataType::Float32)
                                     : destination.to(DataType::Float32);
                    auto next = p;
                    next.destination_format = ShFormat::Float32;
                    sh_codec(src, tmp, next, indices ? &ids : nullptr, source_bounds ? &bounds : nullptr);
                    destination.copy_from(tmp.to(DataType::Float16));
                    return;
                }
                if (p.source_format == ShFormat::Float16 &&
                    !(p.destination_format == ShFormat::Canonical && !indices)) {
                    Tensor tmp = src.to(DataType::Float32);
                    auto next = p;
                    next.source_format = ShFormat::Float32;
                    sh_codec(tmp, destination, next, indices ? &ids : nullptr, nullptr, destination_bounds);
                    return;
                }
                if (p.destination_format == ShFormat::Q16 &&
                    !(p.source_rest == p.destination_rest && !p.scatter &&
                      ((p.source_format == ShFormat::Q16 && indices) ||
                       (p.source_format == ShFormat::Float32 && !indices && p.source_offset == 0)))) {
                    Tensor tmp = internal::allocate_zeros_like(source, {sh_swizzled_float_count(p.count, p.destination_rest)}, DataType::Float32);
                    auto next = p;
                    next.destination_format = ShFormat::Float32;
                    next.destination_rows = p.count;
                    next.destination_offset = 0;
                    sh_codec(src, tmp, next, indices ? &ids : nullptr, source_bounds ? &bounds : nullptr);
                    sh_codec(tmp, destination, {.destination_format = ShFormat::Q16, .source_rows = p.count, .destination_rows = p.count, .count = p.count, .source_rest = p.destination_rest, .destination_rest = p.destination_rest}, nullptr, nullptr, destination_bounds);
                    return;
                }
                if (p.source_format == ShFormat::Q16 &&
                    (p.destination_rest != p.source_rest || p.destination_offset)) {
                    Tensor tmp = internal::allocate_zeros_like(source, {sh_swizzled_float_count(p.count, p.source_rest)}, DataType::Float32);
                    sh_codec(src, tmp, {.source_format = ShFormat::Q16, .source_rows = p.source_rows, .destination_rows = p.count, .count = p.count, .source_rest = p.source_rest, .destination_rest = p.source_rest, .source_offset = p.source_offset}, indices ? &ids : nullptr, &bounds);
                    auto next = p;
                    next.source_format = ShFormat::Float32;
                    next.source_rows = p.count;
                    next.source_offset = 0;
                    sh_codec(tmp, destination, next, nullptr, nullptr, destination_bounds);
                    return;
                }
                if (indices && ((p.source_format == ShFormat::Q16 && ids.dtype() != DataType::Int64) ||
                                (p.scatter && ids.dtype() != DataType::Int32)))
                    ids = ids.to(p.scatter ? DataType::Int32 : DataType::Int64);
            }
            pin_operands({&src, &destination, indices ? &ids : nullptr, source_bounds ? &bounds : nullptr, destination_bounds});
            const auto stream = prepare_inputs_for_stream({&src, &destination});
            if (indices)
                ids.sync_to_stream(stream);
            if (source_bounds)
                bounds.sync_to_stream(stream);
            if (destination_bounds)
                destination_bounds->sync_to_stream(stream);
            destination.set_stream(stream);
            if (destination_bounds)
                destination_bounds->set_stream(stream);
            internal::ShCodecProgram program{.codec = p};
            if (indices)
                program.indices = internal::storage_ref(ids);
            if (source_bounds)
                program.source_bounds = internal::storage_ref(bounds);
            if (destination_bounds)
                program.destination_bounds = internal::storage_ref(*destination_bounds);
            internal::backend_ops_for(destination).sh_codec(internal::storage_ref(src), internal::storage_ref(destination), program, {stream});
            return;
        }

        const auto* source_data = src.data_ptr();
        const auto* source_floats = static_cast<const float*>(source_data);
        const auto* bound_values = source_bounds ? bounds.ptr<float>() : nullptr;
        auto* output_data = destination.data_ptr();
        auto* output_floats = static_cast<float*>(output_data);
        auto* output_bounds = destination_bounds ? destination_bounds->ptr<float>() : nullptr;
        const auto row_index = [&](size_t i) -> size_t {
            if (!indices)
                return i;
            const auto value = ids.dtype() == DataType::Int64 ? ids.ptr<int64_t>()[i] : ids.ptr<int32_t>()[i];
            LFS_ASSERT_MSG(value >= 0 && size_t(value) < (p.scatter ? p.destination_rows : p.source_rows),
                           "sh_codec index out of range");
            return size_t(value);
        };
        const auto read = [&](size_t row, uint32_t c) {
            const auto f = p.source_format;
            const auto width = f == ShFormat::Float32 || f == ShFormat::Float16 ? (p.source_rest * 3 + 3) / 4 * 4 : p.source_rest * 3;
            if (row >= p.source_rows || c >= width)
                return 0.0f;
            const size_t offset = f == ShFormat::Canonical ? row * p.source_rest * 3 + c
                                                           : internal::sh::offset(row, c, p.source_rest, f == ShFormat::Q16);
            if (f == ShFormat::Q16)
                return internal::sh::decode(static_cast<const uint16_t*>(source_data)[offset], bound_values[row / 256 * 2], bound_values[row / 256 * 2 + 1]);
            return f == ShFormat::Float16 ? detail::tensor_half_to_float(static_cast<const detail::tensor_half_t*>(source_data)[offset]) : source_floats[offset];
        };
        if (p.destination_format == ShFormat::Q16) {
            tbb::parallel_for(size_t{0}, sh_value_quant::n_bounds_for_prims(p.count), [&](size_t block) {
                float lo = 1e30f, hi = -1e30f;
                for (size_t i = block * 256; i < std::min(p.count, (block + 1) * 256); ++i)
                    for (uint32_t c = 0; c < p.destination_rest * 3; ++c) {
                        const float v = c < p.source_rest * 3 ? read(indices ? row_index(i) : p.source_offset + i, c) : 0;
                        lo = fminf(lo, v);
                        hi = fmaxf(hi, v);
                    }
                if (lo > hi)
                    lo = hi = 0;
                output_bounds[2 * block] = lo;
                output_bounds[2 * block + 1] = hi;
                for (size_t i = block * 256; i < std::min(sh_swizzled_padded_n(p.count), (block + 1) * 256); ++i)
                    for (uint32_t c = 0; c < p.destination_rest * 3; ++c) {
                        const float v = i < p.count && c < p.source_rest * 3 ? read(indices ? row_index(i) : p.source_offset + i, c) : 0;
                        static_cast<uint16_t*>(output_data)[internal::sh::offset(i, c, p.destination_rest, true)] = i < p.count ? internal::sh::encode(v, lo, hi) : 0;
                    }
            });
            return;
        }
        const uint32_t width = p.destination_format == ShFormat::Canonical ? p.destination_rest * 3 : (p.destination_rest * 3 + 3) / 4 * 4;
        tbb::parallel_for(size_t{0}, p.count, [&](size_t i) {
            const size_t sr = indices && !p.scatter ? row_index(i) : p.source_offset + i;
            const size_t dr = p.scatter ? row_index(i) : p.destination_offset + i;
            for (uint32_t c = 0; c < width; ++c) {
                const bool padding = c >= p.destination_rest * 3 && p.source_format != ShFormat::Float32;
                const float value = padding ? 0 : read(sr, c);
                const size_t offset = p.destination_format == ShFormat::Canonical ? dr * width + c : internal::sh::offset(dr, c, p.destination_rest, false);
                if (p.destination_format == ShFormat::Float16)
                    static_cast<detail::tensor_half_t*>(output_data)[offset] = detail::tensor_float_to_half(value);
                else
                    output_floats[offset] = value;
            }
        });
        if (!p.scatter && p.destination_offset + p.count == p.destination_rows && p.destination_format != ShFormat::Canonical)
            for (size_t row = p.destination_rows; row < sh_swizzled_padded_n(p.destination_rows); ++row)
                for (uint32_t c = 0; c < width; ++c) {
                    const size_t offset = internal::sh::offset(row, c, p.destination_rest, false);
                    if (p.destination_format == ShFormat::Float16)
                        static_cast<uint16_t*>(output_data)[offset] = 0;
                    else
                        output_floats[offset] = 0;
                }
    }
} // namespace lfs::core
