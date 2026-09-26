/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <stdexcept>

namespace lfs::rendering::detail {
    using core::DataType;
    using core::Device;
    using core::GpuBackend;
    using core::GpuBackendScope;
    using core::Tensor;

    [[nodiscard]] inline GpuBackend requireGpuBackend(const Tensor& tensor, const char* message) {
        const auto backend = lfs::core::gpu_backend_of(tensor);
        if (!backend) {
            throw std::runtime_error(message);
        }
        return *backend;
    }

    [[nodiscard]] inline Tensor matchBackend(const Tensor& src, const GpuBackend backend) {
        const auto src_backend = lfs::core::gpu_backend_of(src);
        if (src.device() == Device::GPU && src_backend && *src_backend == backend) {
            return src.is_contiguous() ? src : src.contiguous();
        }
        GpuBackendScope scope(backend);
        if (src.device() == Device::CPU) {
            return src.to(Device::GPU);
        }
        return src.cpu().to(Device::GPU);
    }

    [[nodiscard]] inline Tensor groupInput(const Tensor& source, const GpuBackend backend, const DataType dtype) {
        const auto dense = matchBackend(source, backend).flatten();
        // Tensor::to clones even when the type already matches.
        return dense.dtype() == dtype ? dense : dense.to(dtype);
    }

} // namespace lfs::rendering::detail
