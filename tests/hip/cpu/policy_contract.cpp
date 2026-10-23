/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/gpu_preferences_state.hpp"
#include "visualizer/gui/hip_memory_status.hpp"
#include "core/tensor/backend/hip/hardware_identity.hpp"
#include "core/tensor/backend/hip/physical_identity.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() try {
    using namespace lfs::core;
    using namespace lfs::vis;
    const std::string first(32, 'a'), second(32, 'b');
    std::vector<VulkanHardwareDevice> devices{
        {.name="Same name", .uuid=first, .index=0},
        {.name="Same name", .uuid=second, .index=1}};
    TensorPreferenceState startup;
    startup.backend = GpuBackend::HIP;
    GpuRuntimeSelection running;
    running.backend = GpuBackend::HIP;
    running.hip_uuid = running.automatic_hip_uuid = first;
    running.rendering_uuid = running.automatic_rendering_uuid = second;
    const auto pending = [&](const auto& request) {
        return gpuPreferencesRequireRestart(request, startup, running, devices, devices, devices);
    };
    auto requested = startup;
    require(!pending(requested), "Unchanged startup requests restart");
    requested.options.hip_device = second;
    require(pending(requested), "HIP device change ignored");
    requested.options.hip_device = "0";
    require(!pending(requested), "Restoring active HIP ordinal did not clear restart");
    requested.options.hip_device = first;
    require(!pending(requested), "Restoring active HIP UUID did not clear restart");
    requested.options.cuda_device = second;
    require(!pending(requested), "Inactive CUDA choice changed HIP restart state");
    requested.options.rendering_vulkan_device = first;
    require(pending(requested), "Independent renderer change ignored");
    requested.options.rendering_vulkan_device = second;
    require(!pending(requested), "Restoring active renderer did not clear restart");
    requested.backend = GpuBackend::Vulkan;
    require(pending(requested), "Backend change ignored");
    require(gpuPreferenceDeviceName(first, devices, "") != gpuPreferenceDeviceName(second, devices, ""),
            "Identical names hide distinct GPUs");
    std::swap(devices[0], devices[1]);
    devices[0].index = 0; devices[1].index = 1;
    require(find_gpu_device<VulkanHardwareDevice>(devices, first) == 1,
            "Stable identity lost after enumeration reordering");

    std::vector<HipHardwareDevice> hip(2);
    for (unsigned i=0; i<2; ++i) {
        hip[i].index = i; hip[i].uuid = "driver-collision";
        hip[i].pci_bus_id = i ? "0000:02:00.0" : "0000:01:00.0";
    }
    internal::disambiguate_hip_hardware(hip);
    require(hip[0].uuid != hip[1].uuid, "Driver UUID collision merges GPUs");
    for (auto& device : hip) { device.uuid.clear(); device.pci_bus_id.clear(); }
    internal::disambiguate_hip_hardware(hip);
    require(hip[0].uuid == "hip-index:0" && hip[1].uuid == "hip-index:1", "Missing identity fallback changed");
    require(internal::pci_physical_id("0000:0A:1f.7") == "pci:0000:0a:1f.7", "PCI normalization changed");
    for (const auto value : {"", "0000:100:00.0", "0000:00:20.0", "0000:00:00.8", "0000:00:00.0tail"})
        require(internal::pci_physical_id(value).empty(), "Invalid PCI identity accepted");

    GpuDeviceInfo device;
    device.name = "Selected HIP GPU"; device.total_memory_bytes = 8ull << 30;
    device.process_memory_budget_bytes = 6ull << 30;
    device.process_memory_used_bytes = 2ull << 30;
    for (const auto* sample : {&device, static_cast<GpuDeviceInfo*>(nullptr)}) {
        auto info = gui::selectHipGpuMemory(sample);
        require(info.native_metrics_unavailable && !info.process_valid && !info.process_budget &&
                !info.process_used && !info.gpu_utilization_valid, "Missing HIP metrics became valid");
    }
    device.supports_process_memory_budget = true;
    auto info = gui::selectHipGpuMemory(&device);
    require(info.native_process_budget && !info.native_metrics_unavailable && info.process_valid &&
            info.process_budget == device.process_memory_budget_bytes && info.process_used == device.process_memory_used_bytes &&
            info.total == device.total_memory_bytes && !info.total_used && !info.gpu_utilization_valid,
            "Native HIP process budget was confused with device/CUDA usage");
    device.process_memory_used_bytes = device.process_memory_budget_bytes + 1;
    require(gui::selectHipGpuMemory(&device).process_over_budget, "Over-budget sample not marked");
    std::cout << "PASS CPU HIP restart, adapter identity and native memory policy\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
