/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/headless_vulkan_device.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include "core/tensor/backend/hip/hip_backend_ops.hpp"
#include "training/hip/application_module.hpp"
#include <chrono>
#include <future>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
using namespace std::chrono_literals;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
static void check(VkResult result){require(result==VK_SUCCESS,"Vulkan shared viewer operation failed");}
int main(int argc,char** argv)try {
    require(argc==2,"Argument: absolute HIP training module");
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    auto consumer=HeadlessAdoptedDevice::try_create(true,false,"0");require(bool(consumer),"No external-memory consumer device");
    const auto h=consumer->handles();const auto device=static_cast<VkDevice>(h.device);std::mutex queue_mutex;
    auto module=lfs::training::hip::cached_application_module(argv[1]);
    const auto layout_fn=module->symbol<decltype(&lfs_hip_train_model_layout)>("lfs_hip_train_model_layout");
    {
        TensorVulkanInterop interop({.physical_device=h.physical_device,.device=h.device,.sparse_queue=h.queue,.queue_mutex=&queue_mutex,.queue_families={h.queue_family},.queue_family_count=1,.external_memory=true});
        auto allocator=interop.splat_allocator();
        auto& ops=internal::backend_ops(GpuBackend::HIP);
        auto source=allocator({257},1024,DataType::Float32,"shared viewer model");
        require(source.external_storage_kind()=="hip.external_memory","Same-device model silently fell back to staging");
        const auto memory=ops.stats();require(memory.pool_used_current==0&&memory.pool_reserved_current==0,"Shared model was duplicated in the HIP pool");
        source.fill_(3.f);auto first=interop.buffer(source),same=interop.buffer(source);
        require(first&&same&&first->buffer==same->buffer&&first->device_address==same->device_address,"Shared publication allocated another renderer copy");
        auto region=source.slice(0,3,7);auto sub=interop.buffer(region);
        require(sub&&sub->buffer==first->buffer&&sub->offset==12&&sub->device_address==first->device_address+12,"Shared region offset changed");sub.reset();region={};same.reset();
        VkBuffer staging{};VkDeviceMemory staging_memory{};
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=source.bytes();info.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check(vkCreateBuffer(device,&info,nullptr,&staging));VkMemoryRequirements requirements{};vkGetBufferMemoryRequirements(device,staging,&requirements);
        VkPhysicalDeviceMemoryProperties types{};vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(h.physical_device),&types);uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<types.memoryTypeCount;++i)if((requirements.memoryTypeBits&(1u<<i))&&(types.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
        require(type!=UINT32_MAX,"No coherent consumer readback type");VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};allocation.allocationSize=requirements.size;allocation.memoryTypeIndex=type;
        check(vkAllocateMemory(device,&allocation,nullptr,&staging_memory));check(vkBindBufferMemory(device,staging,staging_memory,0));
        VkCommandPool pool{};VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool_info.queueFamilyIndex=h.queue_family;pool_info.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;check(vkCreateCommandPool(device,&pool_info,nullptr,&pool));
        VkCommandBuffer command{};VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ci.commandPool=pool;ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ci.commandBufferCount=1;check(vkAllocateCommandBuffers(device,&ci,&command));
        auto record=[&](const TensorVulkanBuffer& buffer,bool write){
            check(vkResetCommandBuffer(command,0));VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(command,&begin));
            VkBufferCopy copy{buffer.offset,0,buffer.bytes};vkCmdCopyBuffer(command,static_cast<VkBuffer>(buffer.buffer),staging,1,&copy);
            if(write){
                VkBufferMemoryBarrier war{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};war.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;war.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;war.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;war.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;war.buffer=static_cast<VkBuffer>(buffer.buffer);war.offset=buffer.offset;war.size=buffer.bytes;
                vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&war,0,nullptr);
                vkCmdFillBuffer(command,static_cast<VkBuffer>(buffer.buffer),buffer.offset,buffer.bytes,0x41100000u);
            }
            VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};host_read.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host_read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host_read,0,nullptr,0,nullptr);
            check(vkEndCommandBuffer(command));
        };
        auto submit=[&]{VkSubmitInfo s{VK_STRUCTURE_TYPE_SUBMIT_INFO};s.commandBufferCount=1;s.pCommandBuffers=&command;std::lock_guard lock(queue_mutex);check(vkQueueSubmit(static_cast<VkQueue>(h.queue),1,&s,VK_NULL_HANDLE));};
        std::future<void> native_call;std::promise<void> native_started;auto started=native_started.get_future();
        {
            auto scope=interop.consumer_scope();require(bool(scope),"Shared viewer did not protect consumer submission");
            native_call=std::async(std::launch::async,[&]{native_started.set_value();LfsHipTrainModelLayout layout{};layout.struct_bytes=sizeof(layout);LfsHipError error{};require(layout_fn(37,&layout,&error)==0,error.message);});
            started.wait();require(native_call.wait_for(30ms)==std::future_status::timeout,"Native trainer bypassed consumer recording exclusion");
            record(*first,true);submit();
        }
        require(native_call.wait_for(35s)==std::future_status::ready,"Native trainer did not resume after consumer submission");native_call.get();
        require(source.cpu().to_vector()==std::vector<float>(257,9.f),"Vulkan writes did not reach the HIP model");
        void* host{};check(vkMapMemory(device,staging_memory,0,source.bytes(),0,&host));for(size_t i=0;i<257;++i)require(static_cast<float*>(host)[i]==3.f,"Vulkan did not read HIP writes from the shared allocation");vkUnmapMemory(device,staging_memory);
        bool rejected=false;try{source.reserve(2048);}catch(const std::exception&){rejected=true;}require(rejected,"Shared allocation grew without its owner");
        auto replacement=allocator({257},2048,DataType::Float32,"replacement shared model");replacement.copy_from(source);replacement.fill_(7.f);source=std::move(replacement);
        auto second=interop.buffer(source);require(second&&second->buffer!=first->buffer,"Model replacement invalidated an older publication lease");
        {
            auto scope=interop.consumer_scope();record(*first,false);submit();
        }
        ops.synchronize_device();check(vkMapMemory(device,staging_memory,0,source.bytes(),0,&host));for(size_t i=0;i<257;++i)require(static_cast<float*>(host)[i]==9.f,"Retired shared model changed after replacement");vkUnmapMemory(device,staging_memory);
        require(source.cpu().to_vector()==std::vector<float>(257,7.f),"Replacement model changed");
        for(unsigned frame=0;frame<32;++frame) {
            const float value=float(frame)+.25f;source.fill_(value);
            {auto scope=interop.consumer_scope();record(*second,false);submit();}
            ops.synchronize_device();
            check(vkMapMemory(device,staging_memory,0,source.bytes(),0,&host));
            for(size_t i=0;i<257;++i)require(static_cast<float*>(host)[i]==value,"Repeated shared viewer cycle lost HIP writes");
            vkUnmapMemory(device,staging_memory);
        }
        first.reset();second.reset();source={};interop.drain(GpuBackend::HIP);
        const auto after=ops.stats();require(after.pool_used_current==0,"Shared viewer retained native Tensor allocations");
        vkDestroyCommandPool(device,pool,nullptr);vkDestroyBuffer(device,staging,nullptr);vkFreeMemory(device,staging_memory,nullptr);
    }
    module.reset();require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"Shared viewer retained HIP storage or queues");
    std::puts("PASS shared HIP viewer: one allocation, bidirectional GPU access, native dispatch exclusion, offsets, owner growth, retired generations and clean shutdown");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL shared HIP viewer: %s\n",e.what());return 1;}
