/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/splat_preview.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/splat_data.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "splat_preview_cuda.hpp"
#include <cmath>
#include <cuda_fp16.h>
#include <stdexcept>

namespace lfs::core {
    Tensor splat_sh_preview(const SplatData& model, const Tensor& rows, int degree) {
        if (!rows.is_valid() || rows.device() != Device::CPU || rows.ndim() != 1 ||
            (rows.dtype() != DataType::Int32 && rows.dtype() != DataType::Int64)) {
            throw std::invalid_argument("SH preview rows must be a rank-1 CPU integer tensor");
        }
        if (degree < 0 || degree > model.get_max_sh_degree() || degree > 3) {
            throw std::invalid_argument("SH preview degree is outside the model's supported range");
        }
        const auto indices = rows.to(DataType::Int64).contiguous();
        const auto* selected = indices.ptr<int64_t>();
        const size_t count = indices.numel();
        for (size_t i = 0; i < count; ++i) {
            if (selected[i] < 0 || static_cast<uint64_t>(selected[i]) >= model.size()) {
                throw std::out_of_range("SH preview row is outside the model");
            }
        }
        const auto rest = sh_rest_coefficients_for_degree(degree);
        if (count == 0 || rest == 0) {
            return Tensor::empty_pageable_host({count, rest, size_t{3}}, DataType::Float32);
        }
        const auto layout = static_cast<uint32_t>(model.max_sh_coeffs_rest());
        const auto& source = model.shN_raw();
        const auto& bounds = model.shN_value_bounds();
        const bool quantized = model.shN_value_quantized();
        const bool half = source.dtype() == DataType::Float16;
        const size_t storage_cells = quantized ? sh_swizzled_padded_n(model.size()) * layout * 3u
                                               : sh_swizzled_float_count(model.size(), layout);
        if (!source.is_valid() || !source.is_contiguous() ||
            (!half && source.dtype() != DataType::Float32) || source.numel() < storage_cells ||
            (quantized && (!bounds.is_contiguous() || bounds.dtype() != DataType::Float32 ||
                           bounds.device() != source.device() || bounds.numel() < ((model.size() + 255u) / 256u) * 2u))) {
            throw std::invalid_argument("SH preview requires complete resident SH storage");
        }
        if (source.device() == Device::CUDA) {
            const auto stream = prepare_inputs_for_stream({&source});
            const CUDAStreamGuard guard(stream);
            const auto device_rows = indices.to(Device::CUDA, stream);
            device_rows.sync_to_stream(stream);
            if (quantized)
                bounds.sync_to_stream(stream);
            auto output = Tensor::empty({count, rest, size_t{3}}, Device::CUDA, DataType::Float32);
            detail::gather_preview_sh(source.data_ptr(), quantized ? bounds.ptr<float>() : nullptr, half,
                                      device_rows.ptr<int64_t>(), count, layout, rest, output.ptr<float>(), stream);
            return output.to_pageable_host(stream);
        }
        auto output = Tensor::empty_pageable_host({count, rest, size_t{3}}, DataType::Float32);
        auto* dst = output.ptr<float>();
        for (size_t i = 0; i < count; ++i) {
            const auto row = static_cast<size_t>(selected[i]);
            for (uint32_t cell = 0; cell < rest * 3u; ++cell) {
                float value;
                if (quantized) {
                    const size_t offset = (row / 32u) * layout * 3u * 32u + cell * 32u + row % 32u;
                    const auto q = static_cast<const uint16_t*>(source.data_ptr())[offset];
                    const float lo = bounds.ptr<float>()[(row / 256u) * 2u];
                    const float hi = bounds.ptr<float>()[(row / 256u) * 2u + 1u];
                    const volatile float range = hi - lo;
                    const volatile float normalized = static_cast<float>(q) * (1.0f / 65535.0f);
                    value = std::fma(static_cast<float>(range), static_cast<float>(normalized), lo);
                } else {
                    const size_t offset = static_cast<size_t>(sh_swizzled_index(static_cast<uint32_t>(row), cell / 4u, layout)) * 4u + cell % 4u;
                    value = half ? __half2float(source.ptr<__half>()[offset]) : source.ptr<float>()[offset];
                }
                dst[i * rest * 3u + cell] = value;
            }
        }
        return output;
    }
} // namespace lfs::core
