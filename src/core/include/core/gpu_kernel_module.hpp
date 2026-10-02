/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_fwd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core {

    // Compute kernels that a library outside the tensor tree ships as backend
    // source (MSL on Metal). The module compiles on first launch and dispatches
    // on the tensor timeline: a launch runs after earlier work on its storage,
    // and later tensor work on that storage waits for it. Every kernel takes one
    // parameter block at [[buffer(0)]] and reaches tensors through the device
    // addresses the block carries.
    class LFS_CORE_API GpuKernelModule {
    public:
        struct Launch {
            std::string_view function;
            std::span<const std::byte> params;
            // Every tensor the kernel reads or writes.
            std::span<const Tensor* const> uses;
            std::array<uint32_t, 3> groups{1, 1, 1};
            std::array<uint32_t, 3> group{256, 1, 1};
            // Function constants by index; every value is a uint.
            std::span<const std::pair<uint32_t, uint32_t>> constants;
        };

        // fast_math trades IEEE edge cases for speed, as the CUDA kernels built
        // with -use_fast_math do.
        GpuKernelModule(GpuBackend backend, std::string source, bool fast_math);
        ~GpuKernelModule();
        GpuKernelModule(const GpuKernelModule&) = delete;
        GpuKernelModule& operator=(const GpuKernelModule&) = delete;

        [[nodiscard]] GpuBackend backend() const;
        // Device address of the tensor's first element; 0 for an invalid or
        // empty tensor. The tensor must live on this module's backend.
        [[nodiscard]] uint64_t address(const Tensor& tensor) const;
        void launch(const Launch& launch);

        // Threadgroups that cover `count` items at `width` per group.
        [[nodiscard]] static uint32_t groups_for(size_t count, uint32_t width);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::core
