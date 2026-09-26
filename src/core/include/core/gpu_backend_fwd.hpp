/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace lfs::core {

    enum class GpuBackend : uint8_t {
        CUDA = 0,
        Vulkan = 1,
        Metal = 2,
    };

    // Metal exists only on Apple platforms, so other builds keep two backends.
#ifdef __APPLE__
    inline constexpr std::array kGpuBackends{GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal};
#else
    inline constexpr std::array kGpuBackends{GpuBackend::CUDA, GpuBackend::Vulkan};
#endif
    inline constexpr size_t kGpuBackendCount = kGpuBackends.size();

    LFS_CORE_API const char* gpu_backend_name(GpuBackend backend);
    LFS_CORE_API std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor);
    LFS_CORE_API GpuBackend default_gpu_backend();
    // The backend default_gpu_backend() resolves to, without freezing it.
    LFS_CORE_API GpuBackend configured_gpu_backend();

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
