/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif
#include <vulkan/vulkan.h>
#include "../tensor_vulkan_interop.hpp"
#include "hip_backend_ops.hpp"
#include "physical_identity.hpp"
#include "external_access.hpp"
#include "core/gpu_device_info.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <atomic>
#include <limits>
#include <unordered_map>

namespace lfs::core::internal {
namespace {
void checked(VkResult result,const char* operation) {
    if(result!=VK_SUCCESS)throw std::runtime_error(std::string(operation)+": Vulkan result "+std::to_string(result));
}
size_t product(size_t a,size_t b) {
    if(b&&a>SIZE_MAX/b)throw std::invalid_argument("HIP/Vulkan shared tensor size overflow");
    return a*b;
}
struct SharedAllocation final:HipExternalStorage {
    VulkanInteropDevice target;
    VkBuffer buffer=VK_NULL_HANDLE;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    VkCommandPool pool=VK_NULL_HANDLE;
    VkCommandBuffer command=VK_NULL_HANDLE;
    VkFence fence=VK_NULL_HANDLE;
    uint64_t bytes=0,allocation_bytes=0,address=0;
    bool vulkan_owned=false,failed=false;
    std::mutex mutex;
    explicit SharedAllocation(VulkanInteropDevice device):target(device){}
    VkDevice device() const{return static_cast<VkDevice>(target.device);}
    ~SharedAllocation() override {
        if(failed) {
            LOG_ERROR("HIP/Vulkan shared allocation retained after incomplete consumer work");
            return;
        }
        if(fence)vkDestroyFence(device(),fence,nullptr);
        if(pool)vkDestroyCommandPool(device(),pool,nullptr);
        if(buffer)vkDestroyBuffer(device(),buffer,nullptr);
        if(memory)vkFreeMemory(device(),memory,nullptr);
    }
    void transfer(bool acquire) {
        std::lock_guard lock(mutex);
        if(failed)throw std::runtime_error("HIP/Vulkan shared allocation has an incomplete ownership transfer");
        if(vulkan_owned==acquire)return;
        std::lock_guard queue_lock(*target.queue_mutex);
        try {
            checked(vkResetCommandBuffer(command,0),"reset shared ownership command");
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            checked(vkBeginCommandBuffer(command,&begin),"begin shared ownership command");
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.buffer=buffer;barrier.size=VK_WHOLE_SIZE;
            const uint32_t local=target.queue_family_count>1?VK_QUEUE_FAMILY_IGNORED:target.queue_families[0];
            barrier.srcQueueFamilyIndex=acquire?VK_QUEUE_FAMILY_EXTERNAL:local;
            barrier.dstQueueFamilyIndex=acquire?local:VK_QUEUE_FAMILY_EXTERNAL;
            barrier.srcAccessMask=acquire?0:VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask=acquire?VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT:0;
            vkCmdPipelineBarrier(command,acquire?VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT:VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                  acquire?VK_PIPELINE_STAGE_ALL_COMMANDS_BIT:VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                  0,0,nullptr,1,&barrier,0,nullptr);
            checked(vkEndCommandBuffer(command),"end shared ownership command");
            checked(vkResetFences(device(),1,&fence),"reset shared ownership fence");
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
            checked(vkQueueSubmit(static_cast<VkQueue>(target.sparse_queue),1,&submit,fence),"submit shared ownership transfer");
            checked(vkWaitForFences(device(),1,&fence,VK_TRUE,30'000'000'000ull),"complete shared ownership transfer");
            vulkan_owned=acquire;
        } catch(...) {failed=true;throw;}
    }
    void release_to_hip() override{transfer(false);}
    void acquire_to_vulkan() override{transfer(true);}
    void create(size_t requested) {
        if(requested>SIZE_MAX-255)throw std::invalid_argument("HIP/Vulkan padding overflows");
        bytes=(std::max<size_t>(1,requested)+255)&~size_t{255};
#ifdef _WIN32
        constexpr auto handle_type=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
        constexpr auto handle_type=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
        VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};external.handleTypes=handle_type;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.pNext=&external;info.size=bytes;
        info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        info.sharingMode=target.queue_family_count>1?VK_SHARING_MODE_CONCURRENT:VK_SHARING_MODE_EXCLUSIVE;
        if(target.queue_family_count>1){info.queueFamilyIndexCount=target.queue_family_count;info.pQueueFamilyIndices=target.queue_families.data();}
        VkPhysicalDeviceExternalBufferInfo query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};query.flags=info.flags;query.usage=info.usage;query.handleType=handle_type;
        VkExternalBufferProperties properties{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(static_cast<VkPhysicalDevice>(target.physical_device),&query,&properties);
        if(!(properties.externalMemoryProperties.externalMemoryFeatures&VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT))
            throw std::runtime_error("Consumer device cannot export HIP-compatible buffer memory");
        checked(vkCreateBuffer(device(),&info,nullptr,&buffer),"create HIP shared buffer");
        VkMemoryRequirements requirements{};vkGetBufferMemoryRequirements(device(),buffer,&requirements);
        VkPhysicalDeviceMemoryProperties types{};vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(target.physical_device),&types);
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<types.memoryTypeCount;++i)if((requirements.memoryTypeBits&(1u<<i))&&(types.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){type=i;break;}
        if(type==UINT32_MAX)throw std::runtime_error("Consumer device has no exportable device-local memory type");
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};dedicated.buffer=buffer;
        VkExportMemoryAllocateInfo exported{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};exported.handleTypes=handle_type;exported.pNext=&dedicated;
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;flags.pNext=&exported;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};allocation.pNext=&flags;allocation.allocationSize=requirements.size;allocation.memoryTypeIndex=type;
        checked(vkAllocateMemory(device(),&allocation,nullptr,&memory),"allocate HIP shared buffer");allocation_bytes=requirements.size;
        checked(vkBindBufferMemory(device(),buffer,memory,0),"bind HIP shared buffer");
        VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};address_info.buffer=buffer;
        address=vkGetBufferDeviceAddress(device(),&address_info);if(!address)throw std::runtime_error("HIP shared buffer has no Vulkan device address");
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool_info.queueFamilyIndex=target.queue_families[0];pool_info.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        checked(vkCreateCommandPool(device(),&pool_info,nullptr,&pool),"create shared ownership pool");
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};command_info.commandPool=pool;command_info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;command_info.commandBufferCount=1;
        checked(vkAllocateCommandBuffers(device(),&command_info,&command),"create shared ownership command");
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};checked(vkCreateFence(device(),&fence_info,nullptr,&fence),"create shared ownership fence");
    }
};

class HipVulkanInterop final:public TensorVulkanInteropBackend {
    VulkanInteropDevice target_;
    std::shared_ptr<TensorVulkanInteropBackend> staging_;
    std::mutex mutex_;
    std::unordered_map<void*,std::weak_ptr<SharedAllocation>> allocations_;
    std::atomic<bool> logged_auxiliary_{false};
    std::vector<std::shared_ptr<HipExternalStorage>> live_storages() {
        std::vector<std::shared_ptr<HipExternalStorage>> result;
        std::lock_guard lock(mutex_);
        for(auto it=allocations_.begin();it!=allocations_.end();) {
            if(auto value=it->second.lock()){result.push_back(std::move(value));++it;}else it=allocations_.erase(it);
        }
        return result;
    }
public:
    explicit HipVulkanInterop(VulkanInteropDevice target):target_(target),staging_(make_host_vulkan_interop(target,GpuBackend::HIP)){}
    void shutdown() override {staging_->shutdown();}
    void drain() override {backend_ops(GpuBackend::HIP).synchronize_device();}
    void release_timeline(void* semaphore) override{staging_->release_timeline(semaphore);}
    std::shared_ptr<void> execution_scope() override{return std::make_shared<GpuBackendScope>(GpuBackend::HIP);}
    std::shared_ptr<void> consumer_scope() override {
        auto storages=live_storages();if(storages.empty())return {};
        struct Scope {
            std::vector<std::shared_ptr<HipExternalStorage>> storages;
            explicit Scope(std::vector<std::shared_ptr<HipExternalStorage>> values):storages(std::move(values)) {
                hip_external_access().begin_consumer(storages,[]{backend_ops(GpuBackend::HIP).synchronize_device();});
            }
            ~Scope(){hip_external_access().end_consumer();}
        };
        return std::make_shared<Scope>(std::move(storages));
    }
    Tensor empty_splat(TensorShape shape,size_t capacity,DataType dtype,std::string_view,bool) override{return empty(std::move(shape),dtype,capacity);}
    Tensor empty(TensorShape shape,DataType dtype,size_t capacity) override {
        size_t elements=shape.elements(),rows=shape.rank()?shape[0]:0;
        if(shape.rank()) {
            capacity=capacity?capacity:rows;
            if(capacity<rows)throw std::invalid_argument("HIP shared capacity is smaller than its logical rows");
            elements=capacity;for(size_t i=1;i<shape.rank();++i)elements=product(elements,shape[i]);
        }
        const size_t requested=product(elements,dtype_size(dtype));
        if(!requested)return staging_->empty(std::move(shape),dtype,capacity);
        auto allocation=std::make_shared<SharedAllocation>(target_);allocation->create(requested);
        LfsHipExternalMemory descriptor{sizeof(LfsHipExternalMemory),0,0,allocation->allocation_bytes,allocation->bytes,1,0};
#ifdef _WIN32
        VkMemoryGetWin32HandleInfoKHR query{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};query.memory=allocation->memory;query.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        auto get_handle=reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(allocation->device(),"vkGetMemoryWin32HandleKHR"));
        if(!get_handle)throw std::runtime_error("Missing Vulkan Win32 external memory export");
        HANDLE handle{};checked(get_handle(allocation->device(),&query,&handle),"export HIP shared buffer");
        struct Handle {HANDLE value;~Handle(){if(value)CloseHandle(value);}} owner{handle};
        descriptor.handle_type=LFS_HIP_EXTERNAL_WIN32;descriptor.handle=reinterpret_cast<uintptr_t>(handle);
#else
        VkMemoryGetFdInfoKHR query{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};query.memory=allocation->memory;query.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto get_handle=reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(allocation->device(),"vkGetMemoryFdKHR"));
        if(!get_handle)throw std::runtime_error("Missing Vulkan FD external memory export");
        int handle=-1;checked(get_handle(allocation->device(),&query,&handle),"export HIP shared buffer");
        struct Handle {int value;~Handle(){if(value>=0)(void)close(value);}} owner{handle};
        descriptor.handle_type=LFS_HIP_EXTERNAL_FD;descriptor.handle=uint64_t(handle);
#endif
        auto& ops=static_cast<HipBackendOps&>(backend_ops(GpuBackend::HIP));
        auto backing=ops.import_external_memory(descriptor,allocation);
        const auto storage=storage_ref(backing);
        hip_external_access().register_storage(allocation);
        {std::lock_guard lock(mutex_);allocations_[storage.data]=allocation;}
        return Tensor::view_sharing_storage(backing,0,std::move(shape),capacity,dtype,"hip.external_memory");
    }
    std::optional<TensorVulkanBuffer> buffer(const Tensor& tensor) override {
        if(!tensor.is_valid()||gpu_backend_of(tensor)!=GpuBackend::HIP)return std::nullopt;
        const auto storage=storage_ref(tensor);
        std::shared_ptr<SharedAllocation> allocation;
        {
            std::lock_guard lock(mutex_);
            if(auto found=allocations_.find(storage.data);found!=allocations_.end())allocation=found->second.lock();
        }
        if(!allocation) {
            if(!logged_auxiliary_.exchange(true)){LOG_WARN("GPU transfer fallback: role=rendering auxiliary buffer, requested=HIP/Vulkan shared storage, effective=synchronized host staging, reason=source allocation was not created by the shared allocator");}
            return staging_->buffer(tensor);
        }
        if(!tensor.is_contiguous()||storage.byte_offset>allocation->bytes||tensor.bytes()>allocation->bytes-storage.byte_offset)
            throw std::invalid_argument("HIP shared renderer view exceeds its contiguous allocation");
        struct Lease{Tensor tensor;std::shared_ptr<SharedAllocation> allocation;};
        auto lease=std::make_shared<Lease>(Lease{tensor,allocation});
        return TensorVulkanBuffer{.buffer=allocation->buffer,.offset=storage.byte_offset,.device_address=allocation->address+storage.byte_offset,
                                  .bytes=tensor.bytes(),.keep_alive=lease,.device=target_.device};
    }
    TensorCompletion ready(std::span<const Tensor* const>) override {backend_ops(GpuBackend::HIP).synchronize_device();return {};}
    void wait(std::span<const Tensor* const> tensors,VulkanTimelinePoint point) override {
        staging_->wait(tensors,point);
        backend_ops(GpuBackend::HIP).synchronize_device();
    }
};

bool same_device(VulkanInteropDevice target) {
    const auto ordinal=selected_hip_device();
    const auto devices=enumerate_hip_hardware();
    auto selected=std::ranges::find_if(devices,[&](const auto& value){return int(value.index)==ordinal;});
    if(selected==devices.end()||selected->physical_id.empty())return false;
    std::string physical;
#ifdef _WIN32
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};properties.pNext=&id;
    vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(target.physical_device),&properties);
    if(id.deviceLUIDValid)physical=luid_physical_id(id.deviceLUID);
#else
    uint32_t count=0;checked(vkEnumerateDeviceExtensionProperties(static_cast<VkPhysicalDevice>(target.physical_device),nullptr,&count,nullptr),"query HIP consumer extensions");
    std::vector<VkExtensionProperties> extensions(count);checked(vkEnumerateDeviceExtensionProperties(static_cast<VkPhysicalDevice>(target.physical_device),nullptr,&count,extensions.data()),"query HIP consumer extensions");
    if(std::ranges::any_of(extensions,[](const auto& extension){return std::string_view(extension.extensionName)==VK_EXT_PCI_BUS_INFO_EXTENSION_NAME;})) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};properties.pNext=&pci;
        vkGetPhysicalDeviceProperties2(static_cast<VkPhysicalDevice>(target.physical_device),&properties);physical=pci_physical_id(pci.pciDomain,pci.pciBus,pci.pciDevice,pci.pciFunction);
    }
#endif
    return !physical.empty()&&physical==selected->physical_id;
}
}
std::shared_ptr<TensorVulkanInteropBackend> make_hip_vulkan_interop(VulkanInteropDevice target) {
    try {
        if(!target.external_memory)throw std::runtime_error("consumer external memory extension is disabled");
        if(!target.sparse_queue||!target.queue_mutex||!target.queue_family_count)throw std::runtime_error("consumer queue coordination is unavailable");
        if(!same_device(target))throw std::runtime_error("HIP and renderer are on different or unverified physical GPUs");
        auto result=std::make_shared<HipVulkanInterop>(target);
        {auto probe=result->empty({1},DataType::Float32,1);}
        LOG_INFO("HIP/Vulkan shared storage active: one allocation per model buffer, queue ownership transfers and host-synchronized consumer submission");
        return result;
    } catch(const std::exception& error) {
        LOG_WARN("GPU transfer fallback: role=rendering, requested=HIP/Vulkan shared storage, effective=synchronized host staging, reason={}; consumer storage is separate",error.what());
        return make_host_vulkan_interop(target,GpuBackend::HIP);
    }
}
}
