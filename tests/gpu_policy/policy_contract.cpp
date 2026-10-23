/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/gpu_preferences_state.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lfs::core;
using namespace lfs::vis;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
static std::string& tensorDevice(TensorBackendOptions& options, GpuBackend backend) {
    if (backend == GpuBackend::CUDA) return options.cuda_device;
    if (backend == GpuBackend::Vulkan) return options.vulkan_device;
#ifdef LFS_TENSOR_HIP
    if (backend == GpuBackend::HIP) return options.hip_device;
#endif
    throw std::runtime_error("Backend has no selectable Tensor device");
}
int main() try {
    const std::string first(32, 'a'), second(32, 'b');
    std::vector<VulkanHardwareDevice> devices{
        {.name="Identical GPU", .uuid=first, .index=0},
        {.name="Identical GPU", .uuid=second, .index=1}};
    std::vector<GpuBackend> backends{GpuBackend::CUDA, GpuBackend::Vulkan, GpuBackend::Metal};
#ifdef LFS_TENSOR_HIP
    backends.push_back(GpuBackend::HIP);
#endif
    for (const auto backend : backends) {
        TensorPreferenceState startup;
        startup.backend = backend;
        GpuRuntimeSelection running;
        running.backend = running.automatic_backend = backend;
        running.cuda_uuid = running.automatic_cuda_uuid = first;
        running.vulkan_uuid = running.automatic_vulkan_uuid = first;
        running.rendering_uuid = running.automatic_rendering_uuid = second;
#ifdef LFS_TENSOR_HIP
        running.hip_uuid = running.automatic_hip_uuid = first;
#endif
        const auto pending = [&](const auto& request) {
            return gpuPreferencesRequireRestart(request, startup, running, devices, devices, devices);
        };
        require(!pending(startup), "Unchanged preferences require restart");
        auto request = startup;
        for (const auto candidate : backends) {
            request.backend = candidate;
            require(pending(request) == (candidate != backend), "Backend transition/revert mismatch");
        }
        request = startup;
        request.backend.reset();
        require(!pending(request), "Automatic backend differs from its active resolution");
        if (backend != GpuBackend::Metal) {
            for (const auto& active : {first, std::string("0"), std::string("GPU-{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA}")}) {
                request = startup; tensorDevice(request.options, backend) = active;
                require(!pending(request), "Restoring active GPU identity does not clear restart");
            }
            request = startup; tensorDevice(request.options, backend) = second;
            require(pending(request), "Different GPU on the same backend ignored");
            tensorDevice(request.options, backend) = "removed-device";
            require(pending(request), "Unavailable newly requested GPU ignored");
            startup = request;
            require(!pending(startup), "Unchanged failed startup choice creates a restart loop");
            request = startup; tensorDevice(request.options, backend) = first;
            require(!pending(request), "Selecting actual fallback GPU does not clear restart");
            startup.options = {};
        }
        for (const auto inactive : backends) {
            if (inactive == backend || inactive == GpuBackend::Metal) continue;
            request = startup; tensorDevice(request.options, inactive) = second;
            require(!pending(request), "Inactive backend device changes active restart state");
        }
        request = startup; request.options.rendering_vulkan_device = first;
        require(pending(request), "Independent renderer device change ignored");
        request.options.rendering_vulkan_device = second;
        require(!pending(request), "Restoring active renderer does not clear restart");
        request.options.vulkan_validation = 2;
        require(pending(request), "Vulkan renderer diagnostics change ignored");
        request = startup; request.options.force_fp32_half = true;
        require(pending(request) == (backend == GpuBackend::Vulkan), "Inactive Vulkan Tensor diagnostics affect another backend");
        if (backend == GpuBackend::Metal) {
            running.vulkan_renderer = false;
            request.options.rendering_vulkan_device = first;
            request.options.vulkan_validation = 2;
            require(!pending(request), "Native Metal renderer affected by inactive Vulkan preferences");
        }
    }
    require(gpuPreferenceDeviceName(first, devices, "") != gpuPreferenceDeviceName(second, devices, ""),
            "Duplicate names hide different physical adapters");
    std::swap(devices[0], devices[1]); devices[0].index=0; devices[1].index=1;
    require(find_gpu_device<VulkanHardwareDevice>(devices, first) == 1, "Identity lost on reordering");
    require(!find_gpu_device<VulkanHardwareDevice>(devices, "missing"), "Missing identity resolves to a real GPU");
    require(!find_gpu_device<VulkanHardwareDevice>(devices, "99"), "Unavailable ordinal resolves to a real GPU");
    std::cout << "PASS shared CPU GPU-selection/restart policies: " << backends.size() << " synthetic backends, no SDK/runtime\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
