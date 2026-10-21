/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../../core/tensor/internal/undistort_resample.hpp"
#include <core/tensor.hpp>
#include <core/tensor_image.hpp>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
namespace lfs::training::hip {
    // Reuse the exact CPU area/Lanczos kernel, with original global coordinates
    // and calibration at every pixel. Parallel rows never change footprints,
    // accumulation order or the identity fast path. No GPU dispatch.
    inline core::Tensor undistort_cpu_target(const core::Tensor& input, const core::UndistortParams& params) {
        using namespace core;
        if (!input.is_valid() || input.device() != Device::CPU || (input.dtype() != DataType::Float32 && input.dtype() != DataType::UInt8) ||
            input.ndim() != 3 || input.size(0) != 3 || params.src_width <= 0 || params.src_height <= 0 ||
            params.dst_width <= 0 || params.dst_height <= 0 || input.size(1) != size_t(params.src_height) ||
            input.size(2) != size_t(params.src_width) || size_t(params.src_width) * params.src_height > INT32_MAX ||
            size_t(params.dst_width) * params.dst_height > INT32_MAX || params.num_distortion < 0 || params.num_distortion > 12 ||
            !std::isfinite(params.src_fx) || !std::isfinite(params.src_fy) || !std::isfinite(params.dst_fx) ||
            !std::isfinite(params.dst_fy) || params.src_fx <= 0 || params.src_fy <= 0 || params.dst_fx <= 0 || params.dst_fy <= 0)
            throw std::invalid_argument("Invalid HIP CPU image preparation layout or calibration");
        if (!std::isfinite(params.src_cx) || !std::isfinite(params.src_cy) || !std::isfinite(params.dst_cx) || !std::isfinite(params.dst_cy) ||
            std::max(double(params.src_fx) / params.dst_fx, double(params.src_fy) / params.dst_fy) > 4096)
            throw std::invalid_argument("Invalid or excessive HIP CPU image sampling footprint");
        for (int i = 0; i < params.num_distortion; ++i)
            if (!std::isfinite(params.distortion[i]))
                throw std::invalid_argument("Nonfinite HIP CPU distortion");
        const auto source = input.contiguous();
        if (internal::warp_math::is_identity_resample(params))
            return source.dtype() == DataType::UInt8 ? source.to(DataType::Float32) / 255.f : source.clone();
        auto output = Tensor::empty({3, size_t(params.dst_height), size_t(params.dst_width)}, Device::CPU, DataType::Float32);
        auto* destination = output.ptr<float>();
        const int quadrature = internal::warp_math::area_quadrature(params);
        tbb::task_arena arena(8);
        auto resample = [&](const auto* pixels) {
            arena.execute([&] {
                tbb::parallel_for(tbb::blocked_range<int>(0, params.dst_height, 16), [&](const tbb::blocked_range<int>& rows) {
                    for (int y = rows.begin(); y < rows.end(); ++y)
                        for (int x = 0; x < params.dst_width; ++x)
                            internal::warp_math::undistort_image_kernel(pixels, destination, 3, x, y, quadrature, params);
                });
            });
        };
        // The existing resampler normalizes byte samples when reading each
        // tap. Keep the full-resolution source in bytes, as Media Ingest does.
        if (source.dtype() == DataType::UInt8)
            resample(source.ptr<uint8_t>());
        else
            resample(source.ptr<float>());
        return output;
    }
} // namespace lfs::training::hip
