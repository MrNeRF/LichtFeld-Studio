/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Standalone Windows HIP/Vulkan feasibility probe, not linked to LichtFeld.
// HIP C ABI shapes checked against ROCm/HIP rocm-7.2.0 hip_runtime_api.h.
#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <d3d12.h>
#include <d3dkmthk.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <hip/hip_runtime_api.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#include <windows.h>
#include <wrl/client.h>
#ifdef LFS_HIP_INTEROP_TRAINING
#include "shared_model_training.hpp"
#endif

struct NativeHandle {
    void* value;
    const void* name;
};
union Handle {
    int fd;
    NativeHandle win;
    const void* object;
};
struct MemoryImport {
    int kind;
    Handle native;
    uint64_t size;
    unsigned flags;
    unsigned padding[16];
};
struct BufferMap {
    uint64_t offset, size;
    unsigned flags;
    unsigned padding[16];
};
struct SemaphoreImport {
    int kind;
    Handle native;
    unsigned flags;
    unsigned padding[16];
};
struct SignalParams {
    uint64_t value, sci, key;
    unsigned inner[12];
    unsigned flags, padding[16];
};
struct WaitParams {
    uint64_t value, sci, key;
    unsigned timeout, align, inner[10];
    unsigned flags, padding[16];
};
static_assert(sizeof(MemoryImport) == 104 && sizeof(BufferMap) == 88);
static_assert(sizeof(SemaphoreImport) == 96 && sizeof(SignalParams) == 144 && sizeof(WaitParams) == 144);

static_assert(sizeof(MemoryImport) == sizeof(hipExternalMemoryHandleDesc));
static_assert(sizeof(BufferMap) == sizeof(hipExternalMemoryBufferDesc));
static_assert(sizeof(SemaphoreImport) == sizeof(hipExternalSemaphoreHandleDesc));
static_assert(sizeof(SignalParams) == sizeof(hipExternalSemaphoreSignalParams));
static_assert(sizeof(WaitParams) == sizeof(hipExternalSemaphoreWaitParams));
static_assert(offsetof(MemoryImport, size) == offsetof(hipExternalMemoryHandleDesc, size));
static_assert(offsetof(SignalParams, flags) == offsetof(hipExternalSemaphoreSignalParams, flags));
static_assert(offsetof(WaitParams, flags) == offsetof(hipExternalSemaphoreWaitParams, flags));
static HMODULE runtime;
template <class Fn>
Fn load(const char* name) {
    auto result = reinterpret_cast<Fn>(GetProcAddress(runtime, name));
    if (!result)
        throw std::runtime_error(std::string("Missing HIP entry ") + name);
    return result;
}
static void hip(int result, const char* op) {
    if (result) {
        auto error = load<const char* (*)(int)>("hipGetErrorString");
        throw std::runtime_error(std::string(op) + ": " + error(result) + " (" + std::to_string(result) + ")");
    }
}
static void vk(VkResult result, const char* op) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(op) + ": " + std::to_string(result));
}

static std::optional<D3DKMT_ADAPTERADDRESS> pci_from_luid(const uint8_t* luid) {
    D3DKMT_OPENADAPTERFROMLUID open{};
    std::memcpy(&open.AdapterLuid, luid, sizeof(open.AdapterLuid));
    if (D3DKMTOpenAdapterFromLuid(&open) != 0)
        return std::nullopt;
    D3DKMT_ADAPTERADDRESS address{};
    D3DKMT_QUERYADAPTERINFO query{};
    query.hAdapter = open.hAdapter;
    query.Type = KMTQAITYPE_ADAPTERADDRESS;
    query.pPrivateDriverData = &address;
    query.PrivateDriverDataSize = sizeof(address);
    const auto result = D3DKMTQueryAdapterInfo(&query);
    D3DKMT_CLOSEADAPTER close{};
    close.hAdapter = open.hAdapter;
    const auto closed = D3DKMTCloseAdapter(&close);
    if (result != 0 || closed != 0)
        return std::nullopt;
    return address;
}

int main(int argc, char** argv) try {
    const bool training_photo = argc > 1 && std::strcmp(argv[1], "--training-shared-photo") == 0;
    const bool training_layout = argc > 1 && std::strcmp(argv[1], "--training-shared-layout") == 0;
    const bool training_shared = training_layout || training_photo || (argc > 1 && std::strcmp(argv[1], "--training-shared-model") == 0);
#ifndef LFS_HIP_INTEROP_TRAINING
    if (training_shared)
        throw std::runtime_error("Shared model training test was not compiled");
#endif
    if (training_shared && argc != 4)
        throw std::runtime_error("Shared training requires runtime and training module paths");
    const bool kmt = argc > 1 && std::strcmp(argv[1], "--kmt") == 0;
    const bool binary = kmt || (argc > 1 && std::strcmp(argv[1], "--binary") == 0);
    const bool wait_value = argc > 1 && std::strcmp(argv[1], "--d3d12-wait-value") == 0;
    const bool native_fence = wait_value || (argc > 1 && std::strcmp(argv[1], "--d3d12-import") == 0);
    const bool d3d12 = native_fence || (argc > 1 && std::strcmp(argv[1], "--d3d12") == 0);
    const bool memory_only = training_shared || (argc > 1 && std::strcmp(argv[1], "--memory-only") == 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    const auto runtime_path = std::filesystem::absolute(
                                  argc > 2 ? argv[2] : "C:\\Windows\\System32\\amdhip64_7.dll")
                                  .make_preferred()
                                  .wstring();
    std::printf("Runtime: %ls\n", runtime_path.c_str());
    runtime = LoadLibraryExW(runtime_path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!runtime)
        throw std::runtime_error("HIP 7 DLL load failed: " + std::to_string(GetLastError()));
    hip(load<int (*)(unsigned)>("hipInit")(0), "hipInit");
    int count = 0;
    hip(load<int (*)(int*)>("hipGetDeviceCount")(&count), "hipGetDeviceCount");
    if (count != 1)
        throw std::runtime_error("Controlled probe expects one HIP adapter");
    char hip_pci[64]{};
    hip(load<int (*)(char*, int, int)>("hipDeviceGetPCIBusId")(hip_pci, sizeof(hip_pci), 0), "HIP PCI identity");
    hip(load<int (*)(int)>("hipSetDevice")(0), "hipSetDevice");
    std::printf("HIP PCI: %s\n", hip_pci);
    unsigned pci_domain = 0, pci_bus = 0, pci_device = 0, pci_function = 0;
    if (sscanf_s(hip_pci, "%x:%x:%x.%x", &pci_domain, &pci_bus, &pci_device, &pci_function) != 4)
        throw std::runtime_error("Cannot parse HIP PCI identity");
    hipDeviceProp_t hip_properties{};
    hip(load<int (*)(hipDeviceProp_t*, int)>("hipGetDevicePropertiesR0600")(&hip_properties, 0), "HIP device properties");
    const bool hip_luid_valid = hip_properties.luidDeviceNodeMask != 0 &&
                                std::any_of(std::begin(hip_properties.luid), std::end(hip_properties.luid), [](char byte) { return byte != 0; });
    std::printf("HIP properties LUID valid: %d, node mask %u\n", hip_luid_valid, hip_properties.luidDeviceNodeMask);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance{};
    vk(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
    uint32_t n = 0;
    vk(vkEnumeratePhysicalDevices(instance, &n, nullptr), "enumerate count");
    std::vector<VkPhysicalDevice> devices(n);
    vk(vkEnumeratePhysicalDevices(instance, &n, devices.data()), "enumerate");
    VkPhysicalDevice physical{};
    for (auto device : devices) {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &ids;
        vkGetPhysicalDeviceProperties2(device, &properties);
        const auto& props = properties.properties;
        const bool same_luid = hip_luid_valid && ids.deviceLUIDValid &&
                               std::memcmp(hip_properties.luid, ids.deviceLUID, sizeof(ids.deviceLUID)) == 0 &&
                               (hip_properties.luidDeviceNodeMask & ids.deviceNodeMask) != 0;
        const auto pci = !hip_luid_valid && ids.deviceLUIDValid ? pci_from_luid(ids.deviceLUID) : std::nullopt;
        const bool same_pci = pci_domain == 0 && pci && props.vendorID == 0x1002 &&
                              pci->BusNumber == pci_bus && pci->DeviceNumber == pci_device && pci->FunctionNumber == pci_function;
        if ((hip_luid_valid && same_luid) || (!hip_luid_valid && same_pci)) {
            if (physical)
                throw std::runtime_error("Controlled probe expects one AMD Vulkan adapter");
            physical = device;
            std::printf("Vulkan: %s\n", props.deviceName);
            if (same_luid)
                std::puts("PASS: HIP/Vulkan physical adapter matched by LUID and node mask");
            else
                std::puts("PASS: HIP PCI matches Vulkan LUID -> Windows adapter PCI address");
        }
    }
    if (!physical)
        throw std::runtime_error("No AMD Vulkan adapter");
    vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, nullptr), "extension count");
    std::vector<VkExtensionProperties> extensions(n);
    vk(vkEnumerateDeviceExtensionProperties(physical, nullptr, &n, extensions.data()), "extensions");
    auto has = [&](const char* name) { return std::any_of(extensions.begin(), extensions.end(), [&](auto& e) { return std::strcmp(e.extensionName, name) == 0; }); };
    if (has(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &pci;
        vkGetPhysicalDeviceProperties2(physical, &p2);
        char bus[64]{};
        std::snprintf(bus, sizeof(bus), "%04x:%02x:%02x.%u", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        std::printf("Vulkan PCI: %s\n", bus);
        if (std::strcmp(bus, hip_pci))
            throw std::runtime_error("HIP/Vulkan PCI mismatch");
    }
    const char* enabled[]{VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    for (auto name : enabled)
        if (!has(name))
            throw std::runtime_error(std::string("Missing extension ") + name);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, families.data());
    uint32_t family = 0;
    while (family < n && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
        ++family;
    if (family == n)
        throw std::runtime_error("No compute queue");
    float priority = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &timeline;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = enabled;
    VkDevice device{};
    vk(vkCreateDevice(physical, &dci, nullptr, &device), "vkCreateDevice");
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    const VkDeviceSize bytes = training_shared ? 64 * 1024 * 1024 : 1024 * 1024;
    VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &external;
    bci.size = bytes + 256;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkBuffer buffer{};
    vk(vkCreateBuffer(device, &bci, nullptr, &buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, buffer, &requirements);
    VkPhysicalDeviceMemoryProperties memory_props{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_props);
    uint32_t memory_type = 0;
    const auto wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    while (memory_type < memory_props.memoryTypeCount && (!(requirements.memoryTypeBits & (1u << memory_type)) || (memory_props.memoryTypes[memory_type].propertyFlags & wanted) != wanted))
        ++memory_type;
    if (memory_type == memory_props.memoryTypeCount)
        throw std::runtime_error("No device-local memory for iGPU probe");
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer = buffer;
    VkExportMemoryAllocateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    export_info.handleTypes = external.handleTypes;
    export_info.pNext = &dedicated;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &export_info;
    mai.allocationSize = requirements.size;
    mai.memoryTypeIndex = memory_type;
    VkDeviceMemory memory{};
    vk(vkAllocateMemory(device, &mai, nullptr, &memory), "vkAllocateMemory");
    vk(vkBindBufferMemory(device, buffer, memory, 0), "bind");
    VkMemoryGetWin32HandleInfoKHR get_handle{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    get_handle.memory = memory;
    get_handle.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE handle{};
    vk(reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"))(device, &get_handle, &handle), "export memory");
    MemoryImport import{};
    import.kind = 2;
    import.native.win.value = handle;
    import.size = requirements.size;
    import.flags = 1;
    void* imported{};
    hip(load<int (*)(void**, const MemoryImport*)>("hipImportExternalMemory")(&imported, &import), "hipImportExternalMemory");
    CloseHandle(handle);
    BufferMap map{};
    map.size = bytes + 256;
    void* pointer{};
    hip(load<int (*)(void**, void*, const BufferMap*)>("hipExternalMemoryGetMappedBuffer")(&pointer, imported, &map), "HIP map buffer");
    VkBufferCreateInfo staging_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    staging_info.size = bytes;
    staging_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer staging{};
    vk(vkCreateBuffer(device, &staging_info, nullptr, &staging), "readback buffer");
    VkMemoryRequirements staging_req{};
    vkGetBufferMemoryRequirements(device, staging, &staging_req);
    uint32_t staging_type = 0;
    while (staging_type < memory_props.memoryTypeCount && (!(staging_req.memoryTypeBits & (1u << staging_type)) || !(memory_props.memoryTypes[staging_type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)))
        ++staging_type;
    if (staging_type == memory_props.memoryTypeCount)
        throw std::runtime_error("No readback memory type");
    VkMemoryAllocateInfo staging_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    staging_alloc.allocationSize = staging_req.size;
    staging_alloc.memoryTypeIndex = staging_type;
    VkDeviceMemory staging_memory{};
    vk(vkAllocateMemory(device, &staging_alloc, nullptr, &staging_memory), "readback allocation");
    vk(vkBindBufferMemory(device, staging, staging_memory, 0), "readback bind");
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool{};
    vk(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "command pool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command{};
    vk(vkAllocateCommandBuffers(device, &command_info, &command), "command buffer");
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence{};
    vk(vkCreateFence(device, &fence_info, nullptr, &fence), "fence");
    bool first_submission = true;
    auto submit = [&](bool initial) {
        vk(vkResetCommandBuffer(command, 0), "reset command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vk(vkBeginCommandBuffer(command, &begin), "begin");
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.buffer = buffer;
        barrier.size = VK_WHOLE_SIZE;
        if (!first_submission) {
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            barrier.dstQueueFamilyIndex = family;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = initial ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }
        if (initial)
            vkCmdFillBuffer(command, buffer, 0, bytes, 0x37373737);
        else {
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(command, buffer, staging, 1, &copy);
        }
        barrier.srcQueueFamilyIndex = family;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        barrier.srcAccessMask = initial ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
        vk(vkEndCommandBuffer(command), "end");
        VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submission.commandBufferCount = 1;
        submission.pCommandBuffers = &command;
        vk(vkResetFences(device, 1, &fence), "reset fence");
        vk(vkQueueSubmit(queue, 1, &submission, fence), "queue submit");
        vk(vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ull), "queue completion");
        first_submission = false;
    };
    for (int iteration = 0; iteration < (training_shared ? 0 : 32); ++iteration) {
        submit(true);
        std::vector<uint8_t> check(bytes);
        hip(load<int (*)(void*, const void*, size_t, int)>("hipMemcpy")(check.data(), pointer, bytes, 2), "HIP read shared buffer");
        if (!std::all_of(check.begin(), check.end(), [](auto value) { return value == 0x37; }))
            throw std::runtime_error("Vulkan GPU -> HIP alias mismatch");
        if (iteration == 0)
            std::puts("PASS: Vulkan GPU fill visible through HIP mapping (1 MiB)");
        hip(load<int (*)(void*, int, size_t)>("hipMemset")(pointer, 0x5A, bytes), "HIP shared memset");
        hip(load<int (*)()>("hipDeviceSynchronize")(), "HIP complete");
        submit(false);
        void* host{};
        vk(vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &host), "readback map");
        VkMappedMemoryRange invalidate{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
        invalidate.memory = staging_memory;
        invalidate.size = VK_WHOLE_SIZE;
        vk(vkInvalidateMappedMemoryRanges(device, 1, &invalidate), "readback invalidate");
        if (!std::all_of(static_cast<uint8_t*>(host), static_cast<uint8_t*>(host) + bytes, [](auto value) { return value == 0x5A; }))
            throw std::runtime_error("HIP -> Vulkan GPU alias mismatch");
        if (iteration == 0)
            std::puts("PASS: HIP writes read by Vulkan GPU from the same allocation (1 MiB)");
        vkUnmapMemory(device, staging_memory);
    }
    if (!training_shared)
        std::puts("PASS: 32 alternating Vulkan/HIP write/read cycles with host completion waits");
#ifdef LFS_HIP_INTEROP_TRAINING
    if (training_shared) {
        VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &ids;
        vkGetPhysicalDeviceProperties2(physical, &properties);
        if (!ids.deviceLUIDValid)
            throw std::runtime_error("Shared training requires Vulkan physical device LUID");
        LfsHipTrainSharedModel shared{};
        shared.struct_bytes = sizeof(shared);
        shared.device_node_mask = ids.deviceNodeMask;
        std::memcpy(shared.device_luid, ids.deviceLUID, sizeof(shared.device_luid));
        HANDLE training_handle{};
        vk(reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"))(
               device, &get_handle, &training_handle),
           "export training model");
        shared.win32_handle = reinterpret_cast<uintptr_t>(training_handle);
        shared.allocation_bytes = requirements.size;
        std::optional<LfsHipTrainSharedModel> other_adapter;
        for (auto candidate : devices) {
            VkPhysicalDeviceIDProperties other_ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 other_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            other_properties.pNext = &other_ids;
            vkGetPhysicalDeviceProperties2(candidate, &other_properties);
            if (candidate != physical && other_ids.deviceLUIDValid) {
                other_adapter = shared;
                std::memcpy(other_adapter->device_luid, other_ids.deviceLUID, sizeof(other_adapter->device_luid));
                other_adapter->device_node_mask = other_ids.deviceNodeMask;
                break;
            }
        }
        submit(true);
        unsigned inspections = 0;
        shared_model_training(std::filesystem::absolute(argv[3]), shared, [&](const LfsHipTrainModelLayout& layout, lfs::training::hip::Session& session) {
                                  submit(false); // GPU copy of the actual shared model, not a HIP host copy.
                                  void* mapped{};
                                  vk(vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped), "shared model readback map");
                                  VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                                  range.memory = staging_memory;
                                  range.size = VK_WHOLE_SIZE;
                                  vk(vkInvalidateMappedMemoryRanges(device, 1, &range), "shared model readback invalidate");
                                  const auto* data = static_cast<const uint8_t*>(mapped);
                                  for (unsigned family = 0; family < 5; ++family) {
                                      std::vector<float> expected(layout.float_counts[family]);
                                      session.read_parameter(family, expected);
                                      if (std::memcmp(data + layout.offsets_bytes[family], expected.data(), expected.size() * sizeof(float)))
                                          throw std::runtime_error("Vulkan sees different trained model parameters from HIP");
                                  }
                                  // Every gap (including reserved rows and leading padding) must
                                  // retain the Vulkan sentinel, not just the final allocation tail.
                                  std::array<std::pair<uint64_t, uint64_t>, 5> intervals{};
                                  for (unsigned family = 0; family < 5; ++family)
                                      intervals[family] = {layout.offsets_bytes[family], layout.offsets_bytes[family] + layout.float_counts[family] * 4};
                                  std::sort(intervals.begin(), intervals.end());
                                  uint64_t cursor = 0;
                                  bool guards = true;
                                  for (auto [begin, end] : intervals) {
                                      guards = guards && std::all_of(data + cursor, data + begin, [](uint8_t value) { return value == 0x37; });
                                      cursor = end;
                                  }
                                  guards = guards && std::all_of(data + cursor, data + bytes, [](uint8_t value) { return value == 0x37; });
                                  if (!guards)
                                      throw std::runtime_error("Training wrote outside the shared parameter layout");
                                  vkUnmapMemory(device, staging_memory);
                                  ++inspections; }, other_adapter ? &*other_adapter : nullptr, training_photo, training_layout, [&](uint64_t offset, uint32_t bits) {
                                  uint32_t previous = 0;
                                  auto copy = load<int (*)(void*, const void*, size_t, int)>("hipMemcpy");
                                  auto* address = static_cast<uint8_t*>(pointer) + offset;
                                  hip(copy(&previous, address, sizeof(previous), 2), "read resident model mutation");
                                  hip(copy(address, &bits, sizeof(bits), 1), "write resident model mutation");
                                  return previous; }, pointer);
        std::printf("PASS: invalid identity and size rejected; different physical GPU rejection %s\n",
                    other_adapter ? "verified" : "not tested (no second adapter)");
        CloseHandle(training_handle);
        if (inspections != (training_layout ? 7 : 6))
            throw std::runtime_error("Missing Vulkan shared model inspection");
        submit(true); // Session teardown must not free the Vulkan owner's allocation.
        uint32_t after = 0;
        hip(load<int (*)(void*, const void*, size_t, int)>("hipMemcpy")(&after, pointer, sizeof(after), 2), "read Vulkan allocation after session teardown");
        if (after != 0x37373737)
            throw std::runtime_error("Vulkan allocation did not survive HIP session destruction");
        std::puts("PASS: Vulkan GPU reads all five trained parameter families byte-for-byte; guard unchanged; owner allocation survives teardown");
    }
#endif

    if (!memory_only) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = binary ? VK_SEMAPHORE_TYPE_BINARY : VK_SEMAPHORE_TYPE_TIMELINE;
        const auto semaphore_handle_type = d3d12 ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT : (kmt ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
        VkPhysicalDeviceExternalSemaphoreInfo sem_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
        sem_query.handleType = semaphore_handle_type;
        sem_query.pNext = &type;
        VkExternalSemaphoreProperties sem_properties{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
        vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &sem_query, &sem_properties);
        std::printf("Semaphore capability: type %u, handle %u, features 0x%x, compatible 0x%x\n", unsigned(type.semaphoreType), unsigned(semaphore_handle_type), sem_properties.externalSemaphoreFeatures, sem_properties.compatibleHandleTypes);
        if (!(sem_properties.externalSemaphoreFeatures & (native_fence ? VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT : VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT)))
            throw std::runtime_error("Vulkan reports this semaphore/handle combination as unsupported for the selected direction");
        VkExportSemaphoreWin32HandleInfoKHR sem_security{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
        sem_security.dwAccess = GENERIC_ALL;
        sem_security.pNext = &type;
        VkExportSemaphoreCreateInfo sexport{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        sexport.handleTypes = d3d12 ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT : (kmt ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
        sexport.pNext = &sem_security;
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        sci.pNext = native_fence ? static_cast<void*>(&type) : static_cast<void*>(&sexport);
        VkSemaphore semaphore{};
        vk(vkCreateSemaphore(device, &sci, nullptr, &semaphore), "timeline create");
        VkSemaphoreGetWin32HandleInfoKHR shi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        shi.semaphore = semaphore;
        shi.handleType = d3d12 ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT : (kmt ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT);
        HANDLE semaphore_handle{};
        Microsoft::WRL::ComPtr<ID3D12Device> native_device;
        Microsoft::WRL::ComPtr<ID3D12Fence> shared_fence;
        if (native_fence) {
            auto hr = [](HRESULT code, const char* operation) { if (FAILED(code)) throw std::runtime_error(std::string(operation) + ": " + std::to_string(uint32_t(code))); };
            VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            p2.pNext = &ids;
            vkGetPhysicalDeviceProperties2(physical, &p2);
            if (!ids.deviceLUIDValid)
                throw std::runtime_error("Vulkan LUID unavailable for native fence adapter matching");
            LUID luid{};
            std::memcpy(&luid, ids.deviceLUID, sizeof(luid));
            Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
            Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;
            hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "DXGI factory");
            hr(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)), "Vulkan LUID -> DXGI adapter");
            hr(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&native_device)), "native fence device");
            hr(native_device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&shared_fence)), "native shared fence");
            hr(native_device->CreateSharedHandle(shared_fence.Get(), nullptr, GENERIC_ALL, nullptr, &semaphore_handle), "export native shared fence");
            VkImportSemaphoreWin32HandleInfoKHR import_info{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
            import_info.semaphore = semaphore;
            import_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
            import_info.handle = semaphore_handle;
            vk(reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkImportSemaphoreWin32HandleKHR"))(device, &import_info), "import native fence into Vulkan timeline");
            std::puts("Native D3D12 fence imported into Vulkan; native adapter matched by Vulkan LUID");
        } else {
            vk(reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"))(device, &shi, &semaphore_handle), "export timeline");
        }
        SemaphoreImport sim{};
        sim.kind = kmt ? 3 : (d3d12 ? 4 : (binary ? 2 : 10));
        sim.native.win.value = semaphore_handle;
        void* hip_semaphore{};
        hip(load<int (*)(void**, const SemaphoreImport*)>("hipImportExternalSemaphore")(&hip_semaphore, &sim), "HIP timeline import");
        if (!kmt)
            CloseHandle(semaphore_handle);
        SignalParams signal{};
        signal.value = binary ? 0 : 1;
        hip(load<int (*)(const void* const*, const SignalParams*, unsigned, void*)>("hipSignalExternalSemaphoresAsync")(&hip_semaphore, &signal, 1, nullptr), "HIP signal semaphore");
        if (binary) {
            VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo wait_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            wait_submit.waitSemaphoreCount = 1;
            wait_submit.pWaitSemaphores = &semaphore;
            wait_submit.pWaitDstStageMask = &stage;
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkFence binary_fence{};
            vk(vkCreateFence(device, &fci, nullptr, &binary_fence), "binary fence");
            vk(vkQueueSubmit(queue, 1, &wait_submit, binary_fence), "Vulkan wait HIP binary signal");
            vk(vkWaitForFences(device, 1, &binary_fence, VK_TRUE, 30'000'000'000ull), "binary wait completion");
            vkDestroyFence(device, binary_fence, nullptr);
            VkSubmitInfo signal_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            signal_submit.signalSemaphoreCount = 1;
            signal_submit.pSignalSemaphores = &semaphore;
            vk(vkQueueSubmit(queue, 1, &signal_submit, VK_NULL_HANDLE), "Vulkan signal binary semaphore");
        } else {
            uint64_t value = 1;
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = 1;
            wait.pSemaphores = &semaphore;
            wait.pValues = &value;
            vk(vkWaitSemaphores(device, &wait, 30'000'000'000ull), "Vulkan wait HIP signal");
            VkSemaphoreSignalInfo vsi{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            vsi.semaphore = semaphore;
            vsi.value = 2;
            vk(vkSignalSemaphore(device, &vsi), "Vulkan signal timeline");
        }
        WaitParams hip_wait{};
        hip_wait.value = binary ? 0 : 2;
        hip(load<int (*)(const void* const*, const WaitParams*, unsigned, void*)>("hipWaitExternalSemaphoresAsync")(&hip_semaphore, &hip_wait, 1, nullptr), "HIP wait Vulkan signal");
        hip(load<int (*)()>("hipDeviceSynchronize")(), "HIP semaphore completion");
        std::printf("PASS: shared %s semaphore signal/wait in both directions\n", binary ? "binary" : "timeline");
        if (native_fence) {
            uint8_t* gpu_check = nullptr;
            hip(load<int (*)(void**, size_t, unsigned)>("hipHostMalloc")(reinterpret_cast<void**>(&gpu_check), bytes, 0), "pinned readback allocation");
            void* explicit_stream = nullptr;
            hip(load<int (*)(void**, unsigned)>("hipStreamCreateWithFlags")(&explicit_stream, 1), "explicit HIP stream");
            int memory_wait_supported = 0;
            hip(load<int (*)(int*, int, int)>("hipDeviceGetAttribute")(&memory_wait_supported, static_cast<int>(hipDeviceAttributeCanUseStreamWaitValue), 0), "stream memory-wait capability");
            std::printf("HIP stream memory-wait supported: %d\n", memory_wait_supported);
            if (wait_value && !memory_wait_supported)
                throw std::runtime_error("HIP stream memory waits unavailable");
            auto* control_word = static_cast<uint8_t*>(pointer) + bytes;
            hip(load<int (*)(void*, int, size_t, void*)>("hipMemsetAsync")(control_word, 0, sizeof(uint64_t), explicit_stream), "control word initialization");
            for (unsigned iteration = 0; iteration < 32; ++iteration) {
                const uint64_t producer_value = 3 + uint64_t(iteration) * 2;
                const uint64_t consumer_value = producer_value + 1;
                hip(load<int (*)(void*, int, size_t, void*)>("hipMemsetAsync")(pointer, 0x37, bytes, explicit_stream), "HIP producer async fill");
                SignalParams producer_signal{};
                producer_signal.value = producer_value;
                hip(load<int (*)(const void* const*, const SignalParams*, unsigned, void*)>("hipSignalExternalSemaphoresAsync")(&hip_semaphore, &producer_signal, 1, explicit_stream), "HIP producer async signal");
                vk(vkResetCommandBuffer(command, 0), "async reset command");
                VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                vk(vkBeginCommandBuffer(command, &begin), "async begin");
                VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                barrier.buffer = buffer;
                barrier.size = VK_WHOLE_SIZE;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
                barrier.dstQueueFamilyIndex = family;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                VkBufferCopy copy{0, 0, bytes};
                vkCmdCopyBuffer(command, buffer, staging, 1, &copy);
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                vkCmdFillBuffer(command, buffer, 0, bytes, 0x5A5A5A5A);
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                vkCmdUpdateBuffer(command, buffer, bytes, sizeof(consumer_value), &consumer_value);
                barrier.srcQueueFamilyIndex = family;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = 0;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                vk(vkEndCommandBuffer(command), "async end");
                VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                VkTimelineSemaphoreSubmitInfo timeline_submit{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timeline_submit.waitSemaphoreValueCount = 1;
                timeline_submit.pWaitSemaphoreValues = &producer_value;
                timeline_submit.signalSemaphoreValueCount = 1;
                timeline_submit.pSignalSemaphoreValues = &consumer_value;
                VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submission.pNext = &timeline_submit;
                submission.waitSemaphoreCount = 1;
                submission.pWaitSemaphores = &semaphore;
                submission.pWaitDstStageMask = &stage;
                submission.commandBufferCount = 1;
                submission.pCommandBuffers = &command;
                submission.signalSemaphoreCount = 1;
                submission.pSignalSemaphores = &semaphore;
                vk(vkResetFences(device, 1, &fence), "async reset fence");
                vk(vkQueueSubmit(queue, 1, &submission, fence), "async consumer submit");
                WaitParams consumer_wait{};
                consumer_wait.value = consumer_value;
                hip(load<int (*)(const void* const*, const WaitParams*, unsigned, void*)>("hipWaitExternalSemaphoresAsync")(&hip_semaphore, &consumer_wait, 1, explicit_stream), "HIP wait async consumer");
                if (wait_value)
                    hip(load<int (*)(void*, void*, uint64_t, unsigned, uint64_t)>("hipStreamWaitValue64")(explicit_stream, control_word, consumer_value, 0, UINT64_MAX), "HIP monotonic control-word GPU wait");
                hip(load<int (*)(void*, const void*, size_t, int, void*)>("hipMemcpyAsync")(gpu_check, pointer, bytes, 2, explicit_stream), "HIP readback after consumer wait");
                hip(load<int (*)(void*)>("hipStreamSynchronize")(explicit_stream), "async roundtrip completion");
                if (!std::all_of(gpu_check, gpu_check + bytes, [](auto value) { return value == 0x5A; })) {
                    uint64_t counter = 0;
                    vkGetSemaphoreCounterValue(device, semaphore, &counter);
                    std::fprintf(stderr, "iteration=%u, HIP first byte=%02x, Vulkan counter=%llu, native counter=%llu\n", iteration, gpu_check[0], counter, shared_fence->GetCompletedValue());
                    throw std::runtime_error("Vulkan queue -> HIP stream mismatch");
                }
                vk(vkWaitForFences(device, 1, &fence, VK_TRUE, 30'000'000'000ull), "async fence completion");
                void* verify{};
                vk(vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &verify), "async readback map");
                VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                range.memory = staging_memory;
                range.size = VK_WHOLE_SIZE;
                vk(vkInvalidateMappedMemoryRanges(device, 1, &range), "async invalidate");
                const bool valid = std::all_of(static_cast<uint8_t*>(verify), static_cast<uint8_t*>(verify) + bytes, [](auto value) { return value == 0x37; });
                vkUnmapMemory(device, staging_memory);
                if (!valid)
                    throw std::runtime_error("HIP stream -> Vulkan queue mismatch");
            }
            hip(load<int (*)(void*)>("hipStreamDestroy")(explicit_stream), "explicit stream cleanup");
            hip(load<int (*)(void*)>("hipHostFree")(gpu_check), "pinned readback cleanup");
            std::puts("PASS: 32 HIP-write -> GPU fence -> Vulkan-read/write -> GPU control-word wait -> HIP-read cycles");
        }
        hip(load<int (*)(void*)>("hipDestroyExternalSemaphore")(hip_semaphore), "HIP semaphore cleanup");
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);
    hip(load<int (*)(void*)>("hipFree")(pointer), "HIP mapping cleanup");
    hip(load<int (*)(void*)>("hipDestroyExternalMemory")(imported), "HIP import cleanup");
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, memory, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    FreeLibrary(runtime);
    std::printf("PROBE PASSED: shared memory%s; training performance not tested\n", memory_only ? " with host waits" : " and semaphore");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "PROBE FAILED: %s\n", error.what());
    return 1; // The OS reclaims this short-lived probe's resources after any failed stage.
}
