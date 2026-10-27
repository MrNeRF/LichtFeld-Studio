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
        HIP = 3,
    };

    // Dense enum-index table used by fixed per-backend state. Keep its shape
    // stable even when a backend is not compiled.
    inline constexpr std::array kGpuBackends{GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal, GpuBackend::HIP};
    inline constexpr size_t kGpuBackendCount = kGpuBackends.size();

    // Availability/candidate enumeration excludes backends not compiled into
    // this build. The Vulkan enum remains ABI-stable without being advertised.
#if LFS_HAS_CUDA && defined(LFS_TENSOR_VULKAN)
    inline constexpr std::array kStandardGpuBackends{GpuBackend::CUDA, GpuBackend::Vulkan};
#elif LFS_HAS_CUDA
    inline constexpr std::array kStandardGpuBackends{GpuBackend::CUDA};
#elif defined(LFS_TENSOR_VULKAN) && defined(LFS_TENSOR_METAL)
    inline constexpr std::array kStandardGpuBackends{GpuBackend::Vulkan, GpuBackend::Metal};
#elif defined(LFS_TENSOR_VULKAN)
    inline constexpr std::array kStandardGpuBackends{GpuBackend::Vulkan};
#elif defined(LFS_TENSOR_METAL)
    inline constexpr std::array kStandardGpuBackends{GpuBackend::Metal};
#else
    inline constexpr std::array<GpuBackend, 0> kStandardGpuBackends{};
#endif

#ifdef LFS_TENSOR_HIP
    inline constexpr auto kCompiledGpuBackends = [] {
        std::array<GpuBackend, kStandardGpuBackends.size() + 1> result{};
        for (size_t i = 0; i < kStandardGpuBackends.size(); ++i)
            result[i] = kStandardGpuBackends[i];
        result.back() = GpuBackend::HIP;
        return result;
    }();
#else
    inline constexpr auto kCompiledGpuBackends = kStandardGpuBackends;
#endif

    // Allows optional backend paths to disappear at compile time in default
    // builds without making backend identity or declarations conditional.
    constexpr bool gpu_backend_compiled(GpuBackend backend) {
        for (const auto compiled : kCompiledGpuBackends)
            if (compiled == backend)
                return true;
        return false;
    }

    LFS_CORE_API const char* gpu_backend_name(GpuBackend backend);
    LFS_CORE_API std::optional<GpuBackend> gpu_backend_of(const Tensor& tensor);
    // The configured backend, or without a configuration CUDA where it is
    // built, else Metal where the GPU supports it, else Vulkan. The first call
    // freezes the choice for the process.
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
