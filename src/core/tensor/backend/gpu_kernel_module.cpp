/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_kernel_module.hpp"

#include "../internal/tensor_impl.hpp"
#include "core/tensor_backend.hpp"
#include "metal/metal_module.hpp"

#include <format>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::core {

    struct GpuKernelModule::Impl {
        GpuBackend backend;
        std::unique_ptr<internal::MetalModule> metal;
    };

    GpuKernelModule::GpuKernelModule(const GpuBackend backend, std::string source, const bool fast_math)
        : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        if (backend != GpuBackend::Metal)
            throw std::invalid_argument(
                std::format("GPU kernel modules are unavailable on the {} backend", gpu_backend_name(backend)));
        impl_->metal = internal::make_metal_module(std::move(source), fast_math);
    }

    GpuKernelModule::~GpuKernelModule() = default;

    GpuBackend GpuKernelModule::backend() const { return impl_->backend; }

    uint64_t GpuKernelModule::address(const Tensor& tensor) const {
        if (!tensor.is_valid() || tensor.numel() == 0)
            return 0;
        if (tensor.device() != Device::GPU || gpu_backend_of(tensor) != impl_->backend)
            throw std::invalid_argument(std::format("GPU kernel operand must live on the {} backend, got {} storage",
                                                    gpu_backend_name(impl_->backend),
                                                    tensor.device() == Device::GPU ? "another GPU backend's" : "CPU"));
        // Kernels index operands densely from this address.
        if (!tensor.is_contiguous()) {
            const auto strides = tensor.strides();
            std::string text;
            for (size_t i = 0; i < strides.rank; ++i)
                text += std::format("{}{}", i ? ", " : "", strides.values[i]);
            throw std::invalid_argument(std::format("GPU kernel operand must be contiguous, got shape {} strides [{}]",
                                                    tensor.shape().str(), text));
        }
        return impl_->metal->address(internal::storage_ref(tensor));
    }

    void GpuKernelModule::launch(const Launch& launch) {
        std::vector<internal::StorageRef> uses;
        uses.reserve(launch.uses.size());
        for (const Tensor* const tensor : launch.uses) {
            if (tensor != nullptr && tensor->is_valid() && tensor->numel() != 0)
                uses.push_back(internal::storage_ref(*tensor));
        }
        impl_->metal->launch(launch.function, launch.constants, uses, launch.params, launch.groups, launch.group);
    }

    uint32_t GpuKernelModule::groups_for(const size_t count, const uint32_t width) {
        const size_t groups = (count + width - 1) / width;
        if (groups > UINT32_MAX)
            throw std::overflow_error(std::format("{} items exceed the dispatch limit at {} per group", count, width));
        return static_cast<uint32_t>(groups);
    }

} // namespace lfs::core
