/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

#include <cstdint>
#include <optional>

namespace lfs::core {

    enum class GpuBackend : uint8_t {
        CUDA = 0,
        Vulkan = 1,
    };

    LFS_CORE_API const char* gpu_backend_name(GpuBackend backend);
    LFS_CORE_API std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor);

    // Keep scoped factory selection available to CUDA translation units without
    // pulling in the host-only Result/Error declarations from tensor_backend.hpp.
    class LFS_CORE_API GpuBackendScope {
    public:
        explicit GpuBackendScope(GpuBackend backend);
        ~GpuBackendScope();

        GpuBackendScope(const GpuBackendScope&) = delete;
        GpuBackendScope& operator=(const GpuBackendScope&) = delete;
        GpuBackendScope(GpuBackendScope&&) = delete;
        GpuBackendScope& operator=(GpuBackendScope&&) = delete;

    private:
        std::optional<GpuBackend> previous_;
    };

} // namespace lfs::core
