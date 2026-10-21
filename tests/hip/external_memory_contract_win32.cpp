/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Native runtime contract: shared Vulkan allocation, no CUDA or HIP SDK linkage.
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#include <vulkan/vulkan.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#ifdef LFS_PROBE_VERIFY_HIP_ABI
#include <hip/hip_runtime_api.h>
#endif

#include "core/tensor/backend/hip/external_memory_api.h"
#ifdef LFS_HIP_CORE_EXTERNAL_CONTRACT
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#endif
static HMODULE runtime;
template<class Fn> Fn load(const char* name) {
    auto result = reinterpret_cast<Fn>(GetProcAddress(runtime, name));
    if (!result) throw std::runtime_error(std::string("Missing HIP entry ") + name);
    return result;
}
static void hip(int result, const char* op) {
    if (result) {
        auto error = load<const char* (*)(int)>("hipGetErrorString");
        throw std::runtime_error(std::string(op) + ": " + error(result) + " (" + std::to_string(result) + ")");
    }
}
static void vk(VkResult result, const char* op) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(op) + ": " + std::to_string(result));
}

int main(int argc, char** argv) try {
    if(argc!=2)throw std::runtime_error("Expected absolute native HIP runtime module path");
    HMODULE native=LoadLibraryExA(argv[1],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!native)throw std::runtime_error("Cannot load native HIP module");
    auto entry=[&]<class T>(const char* name,T){auto fn=reinterpret_cast<T>(GetProcAddress(native,name));if(!fn)throw std::runtime_error(name);return fn;};
#define ENTRY(name) auto name=entry(#name,static_cast<decltype(&::name)>(nullptr))
    ENTRY(lfs_hip_context_create);ENTRY(lfs_hip_context_destroy);ENTRY(lfs_hip_buffer_import_external);
    ENTRY(lfs_hip_buffer_release);ENTRY(lfs_hip_buffer_allocate);ENTRY(lfs_hip_buffer_address);
    ENTRY(lfs_hip_download);ENTRY(lfs_hip_memset);ENTRY(lfs_hip_context_synchronize);
#undef ENTRY
    LfsHipError error{};auto native_check=[&](int code){if(code)throw std::runtime_error(error.message);};
    LfsHipContext* context=nullptr;native_check(lfs_hip_context_create(0,&context,&error));
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* runtime_path = "C:\\Windows\\System32\\amdhip64_7.dll";
    std::printf("Runtime: %s\n", runtime_path);
    runtime = LoadLibraryExA(runtime_path, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!runtime) throw std::runtime_error("HIP 7 DLL load failed");
    hip(load<int (*)(unsigned)>("hipInit")(0), "hipInit");
    int count = 0;
    hip(load<int (*)(int*)>("hipGetDeviceCount")(&count), "hipGetDeviceCount");
    if (count != 1) throw std::runtime_error("Controlled probe expects one HIP adapter");
    char hip_pci[64]{};
    hip(load<int (*)(char*, int, int)>("hipDeviceGetPCIBusId")(hip_pci, sizeof(hip_pci), 0), "HIP PCI identity");
    hip(load<int (*)(int)>("hipSetDevice")(0), "hipSetDevice");
    std::printf("HIP PCI: %s\n", hip_pci);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    VkInstance instance{}; vk(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
    uint32_t n = 0; vk(vkEnumeratePhysicalDevices(instance, &n, nullptr), "enumerate count");
    std::vector<VkPhysicalDevice> devices(n); vk(vkEnumeratePhysicalDevices(instance, &n, devices.data()), "enumerate");
    VkPhysicalDevice physical{};
    for (auto device : devices) {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(device, &props);
        if (props.vendorID == 0x1002) {
            if (physical) throw std::runtime_error("Controlled probe expects one AMD Vulkan adapter");
            physical = device; std::printf("Vulkan: %s\n", props.deviceName);
        }
    }
    if (!physical) throw std::runtime_error("No AMD Vulkan adapter");
    vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, nullptr), "extension count");
    std::vector<VkExtensionProperties> extensions(n);
    vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, extensions.data()), "extensions");
    auto has = [&](const char* name) { return std::any_of(extensions.begin(), extensions.end(), [&](auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
    if (has(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p2.pNext = &pci;
        vkGetPhysicalDeviceProperties2(physical, &p2);
        char bus[64]{}; std::snprintf(bus, sizeof(bus), "%04x:%02x:%02x.%u", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        std::printf("Vulkan PCI: %s\n", bus);
        if (std::strcmp(bus, hip_pci)) throw std::runtime_error("HIP/Vulkan PCI mismatch");
    } else std::puts("PCI extension unavailable: this is a controlled single-AMD-device test only");
    const char* enabled[]{VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    for (auto name : enabled) if (!has(name)) throw std::runtime_error(std::string("Missing extension ") + name);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n); vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, families.data());
    uint32_t family = 0; while (family < n && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
    if (family == n) throw std::runtime_error("No compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex = family; qci.queueCount = 1; qci.pQueuePriorities = &priority;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES}; timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &timeline; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci; dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = enabled;
    VkDevice device{}; vk(vkCreateDevice(physical, &dci, nullptr, &device), "vkCreateDevice");
    VkQueue queue{}; vkGetDeviceQueue(device, family, 0, &queue);
    constexpr VkDeviceSize bytes = 1024 * 1024;
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO}; external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.pNext = &external; bci.size = bytes + 256; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer buffer{}; vk(vkCreateBuffer(device, &bci, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{}; vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_props{}; vkGetPhysicalDeviceMemoryProperties(physical, &memory_props);
    uint32_t memory_type = 0;
    const auto wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    while (memory_type < memory_props.memoryTypeCount && (!(requirements.memoryTypeBits & (1u << memory_type)) || (memory_props.memoryTypes[memory_type].propertyFlags & wanted) != wanted)) ++memory_type;
    if (memory_type == memory_props.memoryTypeCount) throw std::runtime_error("No device-local memory for iGPU probe");
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO}; dedicated.buffer = buffer;
    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO}; export_info.handleTypes = external.handleTypes; export_info.pNext = &dedicated;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.pNext = &export_info; mai.allocationSize = requirements.size; mai.memoryTypeIndex = memory_type;
    VkDeviceMemory memory{}; vk(vkAllocateMemory(device, &mai, nullptr, &memory), "vkAllocateMemory"); vk(vkBindBufferMemory(device, buffer, memory, 0), "bind");
    VkMemoryGetWin32HandleInfoKHR get_handle{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR}; get_handle.memory = memory; get_handle.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle{};
    vk(reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"))(device, &get_handle, &handle), "export memory");
    LfsHipExternalMemory descriptor{sizeof(LfsHipExternalMemory),LFS_HIP_EXTERNAL_WIN32,reinterpret_cast<uintptr_t>(handle),requirements.size,bytes+256,1,0};
    LfsHipBuffer* imported=nullptr;
    auto invalid=descriptor;invalid.mapped_bytes=invalid.allocation_bytes+1;
    if(!lfs_hip_buffer_import_external(context,&invalid,&imported,&error)||imported)throw std::runtime_error("Invalid external range was accepted");
    invalid=descriptor;invalid.handle_type=LFS_HIP_EXTERNAL_FD;
    if(!lfs_hip_buffer_import_external(context,&invalid,&imported,&error)||imported)throw std::runtime_error("Foreign-platform external handle was accepted");
    native_check(lfs_hip_buffer_import_external(context,&descriptor,&imported,&error));
    CloseHandle(handle);
    uint64_t pointer=0;native_check(lfs_hip_buffer_address(context,imported,&pointer,&error));
    if(!pointer)throw std::runtime_error("Null registered external address");
    LfsHipContext* other=nullptr;native_check(lfs_hip_context_create(0,&other,&error));
    unsigned char scratch=0;
    if(!lfs_hip_download(other,imported,0,&scratch,1,&error))throw std::runtime_error("External import accepted a foreign context");
    native_check(lfs_hip_context_destroy(other,&error));
    VkBufferCreateInfo staging_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; staging_info.size = bytes; staging_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer staging{}; vk(vkCreateBuffer(device, &staging_info, nullptr, &staging), "readback buffer");
    VkMemoryRequirements staging_req{}; vkGetBufferMemoryRequirements(device, staging, &staging_req);
    uint32_t staging_type = 0;
    while (staging_type < memory_props.memoryTypeCount && (!(staging_req.memoryTypeBits & (1u << staging_type)) || !(memory_props.memoryTypes[staging_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))) ++staging_type;
    if (staging_type == memory_props.memoryTypeCount) throw std::runtime_error("No readback memory type");
    VkMemoryAllocateInfo staging_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; staging_alloc.allocationSize = staging_req.size; staging_alloc.memoryTypeIndex = staging_type;
    VkDeviceMemory staging_memory{}; vk(vkAllocateMemory(device, &staging_alloc, nullptr, &staging_memory), "readback allocation");
    vk(vkBindBufferMemory(device, staging, staging_memory, 0), "readback bind");
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pool_info.queueFamilyIndex = family; pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool{}; vk(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "command pool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; command_info.commandPool = pool; command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; command_info.commandBufferCount = 1;
    VkCommandBuffer command{}; vk(vkAllocateCommandBuffers(device, &command_info, &command), "command buffer");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence{}; vk(vkCreateFence(device, &fence_info, nullptr, &fence), "fence");
    bool first_submission = true;
    auto submit = [&](bool initial) {
        vk(vkResetCommandBuffer(command, 0), "reset command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vk(vkBeginCommandBuffer(command, &begin), "begin");
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER}; barrier.buffer = buffer; barrier.size = VK_WHOLE_SIZE;
        if (!first_submission) {
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL; barrier.dstQueueFamilyIndex = family;
            barrier.srcAccessMask = 0; barrier.dstAccessMask = initial ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }
        if (initial) vkCmdFillBuffer(command, buffer, 0, bytes, 0x37373737);
        else { VkBufferCopy copy{0, 0, bytes}; vkCmdCopyBuffer(command, buffer, staging, 1, &copy); }
        barrier.srcQueueFamilyIndex = family; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        barrier.srcAccessMask = initial ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT; barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        vk(vkEndCommandBuffer(command), "end");
        VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submission.commandBufferCount = 1; submission.pCommandBuffers = &command;
        vk(vkResetFences(device, 1, &fence), "reset fence"); vk(vkQueueSubmit(queue, 1, &submission, fence), "queue submit");
        vk(vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ull), "queue completion");
        first_submission = false;
    };
    for (int iteration = 0; iteration < 128; ++iteration) {
    submit(true);
    std::vector<uint8_t> check(bytes);
    native_check(lfs_hip_download(context,imported,0,check.data(),bytes,&error));
    if (!std::all_of(check.begin(), check.end(), [](auto value) { return value == 0x37; })) throw std::runtime_error("Vulkan GPU -> HIP alias mismatch");
    std::puts("PASS: Vulkan GPU fill visible through HIP mapping (1 MiB)");
    native_check(lfs_hip_memset(context,{imported,0},bytes,0x5A,&error));
    native_check(lfs_hip_context_synchronize(context,&error));
    submit(false);
    void* host{}; vk(vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &host), "readback map");
    VkMappedMemoryRange invalidate{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE}; invalidate.memory = staging_memory; invalidate.size = VK_WHOLE_SIZE;
    vk(vkInvalidateMappedMemoryRanges(device, 1, &invalidate), "readback invalidate");
    if (!std::all_of(static_cast<uint8_t*>(host), static_cast<uint8_t*>(host) + bytes, [](auto value) { return value == 0x5A; })) throw std::runtime_error("HIP -> Vulkan GPU alias mismatch");
    std::puts("PASS: HIP writes read by Vulkan GPU from the same allocation (1 MiB)");
    vkUnmapMemory(device, staging_memory);
    }
    std::puts("PASS: 128 alternating Vulkan/HIP write/read cycles with host completion waits");


    native_check(lfs_hip_buffer_release(context,imported,&error));
    if(!lfs_hip_download(context,imported,0,&scratch,1,&error))throw std::runtime_error("Released import was accepted");
#ifdef LFS_HIP_CORE_EXTERNAL_CONTRACT
    {
        using namespace lfs::core;
        if(!set_default_gpu_backend(GpuBackend::HIP))throw std::runtime_error("Cannot select Core HIP backend");
        auto& ops=static_cast<internal::HipBackendOps&>(internal::backend_ops(GpuBackend::HIP));
        auto& base_ops=internal::backend_ops(GpuBackend::HIP);
        const auto memory_before=base_ops.stats();
        HANDLE core_handle{};
        vk(reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device,"vkGetMemoryWin32HandleKHR"))(device,&get_handle,&core_handle),"export Core memory");
        descriptor.handle=reinterpret_cast<uintptr_t>(core_handle);
        auto exporter=std::make_shared<int>(7);std::weak_ptr<int> lifetime=exporter;
        auto backing=ops.import_external_memory(descriptor,exporter);
        const auto memory_after=base_ops.stats();
        if(memory_before.pool_used_current!=memory_after.pool_used_current||memory_before.pool_reserved_current!=memory_after.pool_reserved_current)throw std::runtime_error("Shared Vulkan storage was counted again in the HIP pool");
        CloseHandle(core_handle);exporter.reset();
        if(lifetime.expired()||gpu_backend_of(backing)!=GpuBackend::HIP||backing.external_storage_kind()!="hip.external_memory")throw std::runtime_error("Core external ownership or backend metadata was lost");
        auto view=Tensor::view_sharing_storage(backing,16,{257},257,DataType::UInt8,"hip.external_memory");
        {
            auto writable=Tensor::view_sharing_storage(backing,0,{size_t(bytes/4)},size_t(bytes/4),DataType::Float32,"hip.external_memory");
            writable.fill_(25.f);
            base_ops.synchronize_device();
        }
        submit(false);
        void* core_host{};vk(vkMapMemory(device,staging_memory,0,VK_WHOLE_SIZE,0,&core_host),"Core readback map");
        VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};range.memory=staging_memory;range.size=VK_WHOLE_SIZE;
        vk(vkInvalidateMappedMemoryRanges(device,1,&range),"Core readback invalidate");
        if(!std::all_of(static_cast<float*>(core_host),static_cast<float*>(core_host)+bytes/4,[](auto x){return x==25.f;}))throw std::runtime_error("Core HIP writes did not reach the same Vulkan allocation");
        vkUnmapMemory(device,staging_memory);
        submit(true);
        backing={};
        if(lifetime.expired())throw std::runtime_error("Core external exporter was released while a region view remained");
        const auto values=view.to(DataType::Float32).cpu().to_vector();
        if(values!=std::vector<float>(257,55.f))throw std::runtime_error("Vulkan GPU writes did not reach the Core region view");
        view={};
        if(!lifetime.expired())throw std::runtime_error("Core external import created an ownership cycle");
        if(!shutdown_gpu_backend(GpuBackend::HIP))throw std::runtime_error("Core external import retained its native allocation");
        std::puts("PASS Core shared storage: native Tensor writes, Vulkan GPU reads/writes, region views, exporter lifetime and clean shutdown");
    }
#endif
    LfsHipBuffer* regular=nullptr;native_check(lfs_hip_buffer_allocate(context,257,&regular,&error));
    native_check(lfs_hip_memset(context,{regular,0},257,0x19,&error));
    native_check(lfs_hip_download(context,regular,256,&scratch,1,&error));
    if(scratch!=0x19)throw std::runtime_error("Ordinary allocation regressed after external import release");
    native_check(lfs_hip_buffer_release(context,regular,&error));
    native_check(lfs_hip_context_destroy(context,&error));
    vkDestroyFence(device,fence,nullptr);vkDestroyCommandPool(device,pool,nullptr);
    vkDestroyBuffer(device,staging,nullptr);vkFreeMemory(device,staging_memory,nullptr);
    vkDestroyBuffer(device,buffer,nullptr);vkFreeMemory(device,memory,nullptr);
    vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    FreeLibrary(native);FreeLibrary(runtime);
    std::puts("PASS native HIP external memory: 128 bidirectional GPU cycles, bounds, platform, context ownership, stale handles, release and ordinary allocation");return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL native external memory: %s\n",e.what());return 1;}
