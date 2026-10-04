/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_vignette.hpp"
#include "vignette_program.hpp"
#include <cmath>
#include <format>
#include <limits>

namespace lfs::core {
    Result<Tensor> vignette_image(uint32_t width, uint32_t height, float intensity, float radius, float softness) {
        if (!width || !height || width > 65535 || height > 65535 ||
            !std::isfinite(intensity) || !std::isfinite(radius) || !std::isfinite(softness))
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::Tensor, .detail = std::format("Invalid vignette extent/parameters: {}x{}, intensity={}, radius={}, softness={}", width, height, intensity, radius, softness), .detection = LFS_SOURCE_SITE_CURRENT()});
        auto image = Tensor::empty({height, width, 4}, Device::GPU, DataType::Float32);
        auto program = GpuKernelModule::load(vignette_program_entries(), *gpu_backend_of(image));
        if (!program)
            return std::move(program).error();
        struct Params {
            uint64_t output = 0;
            uint32_t width, height;
            float intensity, radius, softness;
            uint32_t padding = 0;
        } params{.width = width, .height = height, .intensity = intensity, .radius = radius, .softness = softness};
        const std::array bindings{GpuKernelModule::Binding{0, &image, GpuKernelModule::Access::ReadWrite}};
        auto result = (*program)->dispatch({.function = "vignette", .arguments = {std::as_bytes(std::span(&params, 1)), bindings}, .groups = {GpuKernelModule::groups_for(width, 64), height, 1}});
        if (!result)
            return std::move(result).error();
        return image;
    }
} // namespace lfs::core
