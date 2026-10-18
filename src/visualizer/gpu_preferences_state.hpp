/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_device_info.hpp"
#include "core/gpu_device_selection.hpp"
#include "preferences.hpp"

namespace lfs::vis {
    struct GpuRuntimeSelection {
        core::GpuBackend backend = core::GpuBackend::CUDA;
        core::GpuBackend automatic_backend = core::GpuBackend::CUDA;
        core::TensorBackendOptions options;
        bool vulkan_renderer = true;
        std::string cuda_uuid, vulkan_uuid, rendering_uuid;
        std::string automatic_cuda_uuid, automatic_vulkan_uuid, automatic_rendering_uuid;
    };

    inline std::string requestedGpuIdentity(const std::string& requested,
                                            std::span<const core::VulkanHardwareDevice> devices,
                                            const std::string& automatic) {
        if (requested.empty())
            return core::normalize_gpu_identity(automatic);
        const auto selected = core::find_gpu_device<core::VulkanHardwareDevice>(devices, requested);
        return core::normalize_gpu_identity(selected ? devices[*selected].uuid : requested);
    }

    inline std::string gpuPreferenceDeviceName(const std::string& uuid,
                                               std::span<const core::VulkanHardwareDevice> devices,
                                               const std::string& fallback) {
        const auto selected = core::find_gpu_device<core::VulkanHardwareDevice>(devices, uuid);
        if (!selected)
            return fallback;
        const auto& device = devices[*selected];
        if (std::count_if(devices.begin(), devices.end(), [&](const auto& item) { return item.name == device.name; }) < 2)
            return device.name;
        const auto identity = core::normalize_gpu_identity(device.uuid);
        return device.name + " (#" + std::to_string(device.index) + ", " +
               identity.substr(identity.size() > 8 ? identity.size() - 8 : 0) + ")";
    }

    // An unchanged failed startup preference is a fallback, not a pending edit.
    // Explicitly selecting the running device clears a pending edit even when
    // the original preference used Auto, an index, or an unavailable device.
    inline bool gpuPreferencesRequireRestart(const TensorPreferenceState& requested,
                                             const TensorPreferenceState& startup,
                                             const GpuRuntimeSelection& running,
                                             std::span<const core::VulkanHardwareDevice> cuda,
                                             std::span<const core::VulkanHardwareDevice> vulkan) {
        const auto backend = requested.backend.value_or(running.automatic_backend);
        const bool backend_changed = requested.backend != startup.backend;
        const auto device_changed = backend_changed || (backend == core::GpuBackend::CUDA
                                                            ? requested.options.cuda_device != startup.options.cuda_device
                                                        : backend == core::GpuBackend::Vulkan
                                                            ? requested.options.vulkan_device != startup.options.vulkan_device
                                                            : false);
        if ((backend_changed || device_changed) && backend != running.backend)
            return true;
        if (device_changed && backend == core::GpuBackend::CUDA &&
            requestedGpuIdentity(requested.options.cuda_device, cuda, running.automatic_cuda_uuid) != core::normalize_gpu_identity(running.cuda_uuid))
            return true;
        if (device_changed && backend == core::GpuBackend::Vulkan &&
            requestedGpuIdentity(requested.options.vulkan_device, vulkan, running.automatic_vulkan_uuid) != core::normalize_gpu_identity(running.vulkan_uuid))
            return true;
        if (running.vulkan_renderer && requested.options.rendering_vulkan_device != startup.options.rendering_vulkan_device &&
            requestedGpuIdentity(requested.options.rendering_vulkan_device, vulkan, running.automatic_rendering_uuid) != core::normalize_gpu_identity(running.rendering_uuid))
            return true;
        return ((running.vulkan_renderer || backend == core::GpuBackend::Vulkan) &&
                requested.options.vulkan_validation != running.options.vulkan_validation) ||
               (backend == core::GpuBackend::Vulkan &&
                (requested.options.force_fp32_half != running.options.force_fp32_half ||
                 requested.options.force_no_atomic_float != running.options.force_no_atomic_float));
    }
} // namespace lfs::vis
