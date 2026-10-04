/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/private_access.hpp"
#include <vector>

namespace lfs::core::internal {
    struct ProgramArguments {
        std::vector<std::byte> parameters;
        std::vector<const Tensor*> reads;
        std::vector<const Tensor*> writes;
    };

    class GpuProgram {
    public:
        virtual ~GpuProgram() = default;
        virtual bool supports_raster() const = 0;
        virtual uint64_t address(const Tensor&) = 0;
        virtual void dispatch(const GpuKernelModule::Dispatch&, const ProgramArguments&) = 0;
        // draws[i] binds arguments[i]; every draw shares draws[0]'s attachments.
        virtual void draw(std::span<const GpuKernelModule::Draw> draws,
                          std::span<const ProgramArguments> arguments) = 0;
    };

    std::unique_ptr<GpuProgram> make_vulkan_program(std::span<const GpuKernelModule::Entry>);
    std::unique_ptr<GpuProgram> make_metal_program(std::span<const GpuKernelModule::Entry>);
    std::unique_ptr<GpuProgram> make_cuda_program(std::span<const GpuKernelModule::Entry>);
} // namespace lfs::core::internal
