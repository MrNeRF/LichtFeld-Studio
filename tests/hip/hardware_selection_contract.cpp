/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/gpu_device_info.hpp>
#include <core/crash_handler.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "core/tensor/backend/hip/hardware_identity.hpp"
#include "core/tensor/backend/hip/physical_identity.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main() try {
    require(internal::pci_physical_id("0000:0A:1f.7")=="pci:0000:0a:1f.7","PCI identity normalization failed");
    for(const auto raw:{"", "0000:100:00.0", "0000:00:20.0", "0000:00:00.8", "0000:00:00.0tail"})
        require(internal::pci_physical_id(raw).empty(),"Invalid PCI identity was accepted");
    {
        std::vector<HipHardwareDevice> duplicates(2);
        for(unsigned i=0;i<2;++i){duplicates[i].index=i;duplicates[i].uuid="driver-duplicate";duplicates[i].pci_bus_id="0000:"+std::to_string(i)+":00.0";}
        internal::disambiguate_hip_hardware(duplicates);
        require(duplicates[0].uuid=="pci:0000:0:00.0" && duplicates[1].uuid=="pci:0000:1:00.0","Duplicate HIP driver UUIDs select the same adapter");
        for(auto& device:duplicates){device.uuid="driver-duplicate";device.pci_bus_id.clear();}
        internal::disambiguate_hip_hardware(duplicates);
        require(duplicates[0].uuid=="hip-index:0" && duplicates[1].uuid=="hip-index:1","HIP devices without trustworthy identifiers were merged");
    }
    const auto before=configured_gpu_backend();
    require(!gpu_backend_live(GpuBackend::HIP),"HIP inventory test started with a live Tensor context");
    const auto devices=enumerate_hip_hardware();
    const auto vulkan_devices=enumerate_vulkan_hardware();
    for(const auto& device:devices) {
        require(device.physical_id==internal::pci_physical_id(device.pci_bus_id),"HIP physical identity disagrees with native PCI address");
        for(const auto& vk:vulkan_devices) {
            std::printf("Hardware identity HIP %s [%s] / Vulkan %s [%s]: %s\n",device.name.c_str(),device.physical_id.c_str(),vk.name.c_str(),vk.physical_id.c_str(),!device.physical_id.empty()&&device.physical_id==vk.physical_id?"same physical device":"distinct or unverified");
        }
    }
    const auto valid=std::ranges::find_if(devices,[](const auto& device){return device.tensor_unavailable_reason.empty();});
    require(valid!=devices.end(),"No compatible HIP hardware was discovered");
    for(const auto& device:devices) {
        require(!device.name.empty() && !device.uuid.empty() && !device.architecture.empty(),"HIP inventory omitted the driver name, identity or architecture");
        require(device.discrete!=device.shared_memory,"HIP inventory confuses dedicated and shared memory");
        std::printf("HIP %u: %s, architecture=%s, identity=%s, pci=%s, shared_memory=%u, reported_bytes=%zu, status=%s\n",
                    device.index,device.name.c_str(),device.architecture.c_str(),device.uuid.c_str(),device.pci_bus_id.c_str(),device.shared_memory,
                    device.total_memory_bytes,device.tensor_unavailable_reason.empty()?"compatible":device.tensor_unavailable_reason.c_str());
    }
    require(configured_gpu_backend()==before && !gpu_backend_live(GpuBackend::HIP),"HIP discovery changed the configured backend or created its Tensor context");
    auto options=tensor_backend_options_snapshot();options.hip_device=valid->uuid;
    require(bool(set_tensor_backend_options(options)),"HIP UUID selection was rejected");
    require(selected_hip_device()==int(valid->index) && !gpu_backend_live(GpuBackend::HIP),"HIP UUID selection changed device or created a Tensor context");
    options.hip_device=std::to_string(valid->index);
    require(bool(set_tensor_backend_options(options)) && selected_hip_device()==int(valid->index),"HIP ordinal and UUID selection disagree");
    auto invalid=options;invalid.hip_device="not-a-real-gpu";
    require(!set_tensor_backend_options(invalid) && tensor_backend_options_snapshot().hip_device==options.hip_device,
            "Failed HIP selection modified the active options");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Explicit HIP backend selection failed");
    const auto info=gpu_backend_device_info(GpuBackend::HIP);
    require(info && info->name==valid->name && info->total_memory_bytes==valid->total_memory_bytes,
            "Selected HIP device information does not match inventory");
    require(!info->supports_process_memory_budget || info->process_memory_budget_bytes > 0,
            "HIP advertised a process budget without a valid OS sample");
    require(!gpu_backend_live(GpuBackend::HIP),"HIP selected-device query initialized its Tensor context");
    {
        auto values=Tensor::full({17},3.f,Device::GPU);
        require(gpu_backend_of(values)==GpuBackend::HIP && values.cpu().to_vector()==std::vector<float>(17,3.f),"HIP selected device did not execute real Tensor work");
        auto& backend=static_cast<internal::HipBackendOps&>(internal::backend_ops(GpuBackend::HIP));
        const auto queue=backend.execution_lease({});
        require(queue.device==int(valid->index),"HIP execution queue does not belong to the selected GPU");
        require(!set_tensor_backend_options(invalid) && selected_hip_device()==int(valid->index),"HIP device selection was mutable after context creation");
    }
    teardown_gpu_before_exit();
    require(!gpu_backend_live(GpuBackend::HIP),"Application shutdown retained the HIP context after its Tensor owners were released");
    std::puts("PASS HIP hardware: native names and identity, shared-memory classification, lazy UUID/index selection, failed-update preservation and frozen execution device");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"HIP HARDWARE SELECTION FAILED: %s\n",e.what());return 1;}
