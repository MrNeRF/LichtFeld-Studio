/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/headless_vulkan_device.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_upload.hpp>
#include <core/logger.hpp>
#include <cstdio>
#include <core/detail/tensor_half.hpp>
#include <stdexcept>
#include "../../src/visualizer/rendering/hip_input_publications.hpp"
using namespace lfs::core;
static void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static void check(VkResult v){require(v==VK_SUCCESS,"Vulkan contract operation failed");}
int main(int argc,char**argv) try {
    Logger::get().init(LogLevel::Info);
    require(bool(set_default_gpu_backend(GpuBackend::HIP)),"Cannot select HIP");
    require(!tensor_backend_shares_vulkan_device() && tensor_backend_needs_vulkan_interop(),"HIP publication capabilities are incorrect");
    auto consumer=HeadlessAdoptedDevice::try_create(true,false,argc>1?argv[1]:"");
    require(consumer.has_value(),"No suitable Vulkan consumer");
    const auto h=consumer->handles();
    const auto device=static_cast<VkDevice>(h.device);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(static_cast<VkPhysicalDevice>(h.physical_device),&properties);
    std::printf("Vulkan consumer: %s\n",properties.deviceName);
    {
        std::vector<unsigned char> flags(257);
        std::vector<float> values(257);
        for(size_t i=0;i<257;++i){flags[i]=i%3==0;values[i]=float(int(i)-128)/16.f;}
        auto mask=Tensor::from_blob(flags.data(),{257},Device::CPU,DataType::Bool).to(Device::GPU);
        for(const auto dtype:{DataType::Float32,DataType::Float16}) {
            auto x=Tensor::from_vector(values,{257},Device::GPU).to(dtype);
            auto y=Tensor::empty({257},Device::GPU,dtype);
            where_into(y,mask,-2.25f,x);
            auto actual=y.to(DataType::Float32).cpu().to_vector();
            for(size_t i=0;i<257;++i)require(actual[i]==(flags[i]?-2.25f:values[i]),"HIP renderer selection differs from CPU reference");
            where_into(x,mask,4.5f,x);
            actual=x.to(DataType::Float32).cpu().to_vector();
            for(size_t i=0;i<257;++i)require(actual[i]==(flags[i]?4.5f:values[i]),"HIP in-place renderer selection differs from CPU reference");
            auto backing=Tensor::full({259},99.f,Device::GPU,dtype);
            auto src=backing.slice(0,0,257),dst=backing.slice(0,1,258);
            where_into(dst,mask,6.f,src);
            actual=backing.to(DataType::Float32).cpu().to_vector();
            require(actual.front()==99.f && actual.back()==99.f,"HIP renderer selection overwrote guards");
            for(size_t i=0;i<257;++i)require(actual[i+1]==(flags[i]?6.f:99.f),"HIP overlapping renderer selection lost its source snapshot");
        }
    }
    {
        std::mutex consumer_queue_mutex;
        TensorVulkanInterop interop({.physical_device=h.physical_device,.device=h.device,.sparse_queue=h.queue,.queue_mutex=&consumer_queue_mutex,.queue_families={h.queue_family},.queue_family_count=1,.external_memory=true});
        const size_t publication_count=argc>2 && std::string_view(argv[2])=="--large" ? 300001 : 257;
        auto source=Tensor::full({publication_count},3.f,Device::GPU);
        lfs::vis::HipInputPublications cache;
        cache.resize(2);
        unsigned publications=0;
        auto publish=[&]{++publications;return interop.buffer(source);};
        auto first=cache.get(0,0,publish);
        require(first.has_value(),"Initial HIP publication failed");
        const auto repeated=cache.get(0,0,publish);
        require(publications==1 && repeated && repeated->buffer==first->buffer,
                "Unchanged HIP publication performed another transfer");
        unsigned retries=0;
        require(!cache.get(1,7,[&]()->std::optional<TensorVulkanBuffer>{++retries;return std::nullopt;}),
                "Failed publication unexpectedly cached a buffer");
        require(bool(cache.get(1,7,[&]{++retries;return interop.buffer(source);})) && retries==2,
                "Failed publication poisoned a later retry");
        cache.invalidate(1);
        require(bool(cache.get(0,0,publish)) && publications==1,
                "Retiring one view invalidated another view's publication");
        require(bool(cache.get(1,7,[&]{++retries;return interop.buffer(source);})) && retries==3,
                "Retired view reused its previous publication");
        {
            lfs::vis::HipInputPublications bounded(source.bytes());
            bounded.resize(2);
            unsigned loads=0;
            auto load=[&]{++loads;return interop.buffer(source);};
            auto retained=bounded.get(0,0,load);
            require(bool(retained),"Bounded publication failed");
            require(bool(bounded.get(1,0,load)) && bool(bounded.get(1,0,load)) && loads==3,
                    "Publication cache exceeded its retained-payload budget");
            bounded.resize(0);bounded.resize(1);
            require(bool(bounded.get(0,0,load)) && bool(bounded.get(0,0,load)) && loads==4,
                    "Shrinking publication slots did not release the cache budget");
        }
        require(first && first->device==h.device && first->bytes==publication_count*sizeof(float) && first->device_address,"Invalid published buffer");
        source.reserve(publication_count*2); // Publication must retain the allocation preceding a model-capacity change.
        source.fill_(7.f);
        cache.invalidate(); // Native/model mutations and training bypass invalidate all slots.
        auto second=cache.get(0,0,publish);
        require(publications==2,"Changed HIP model reused an outdated publication");
        require(second && second->buffer!=first->buffer,"A live publication lease was overwritten");
        auto cpu_source=Tensor::full({publication_count},23.f,Device::CPU);
        auto cpu_snapshot=interop.host_snapshot_buffer(cpu_source);
        require(cpu_snapshot && cpu_snapshot->bytes==source.bytes(),"Completed CPU snapshot was not published");
        cpu_source.fill_(55.f);
        cpu_source={}; // Consumer copy must survive source mutation and release.
        // Read the first immutable snapshot and overwrite the second with an actual consumer command.
        VkBuffer staging{};VkDeviceMemory memory{};
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=source.bytes()*2;
        bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device,&bi,nullptr,&staging));
        VkMemoryRequirements requirements{};vkGetBufferMemoryRequirements(device,staging,&requirements);
        VkPhysicalDeviceMemoryProperties types{};vkGetPhysicalDeviceMemoryProperties(static_cast<VkPhysicalDevice>(h.physical_device),&types);
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<types.memoryTypeCount;++i)
            if((requirements.memoryTypeBits&(1u<<i)) && (types.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
        require(type!=UINT32_MAX,"Missing coherent consumer readback memory");
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};ai.allocationSize=requirements.size;ai.memoryTypeIndex=type;
        check(vkAllocateMemory(device,&ai,nullptr,&memory));check(vkBindBufferMemory(device,staging,memory,0));
        VkCommandPool pool{};VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=h.queue_family;
        check(vkCreateCommandPool(device,&pi,nullptr,&pool));
        VkCommandBuffer command{};VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ci.commandPool=pool;ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ci.commandBufferCount=1;
        check(vkAllocateCommandBuffers(device,&ci,&command));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};check(vkBeginCommandBuffer(command,&begin));
        VkBufferCopy copy{0,0,source.bytes()};vkCmdCopyBuffer(command,static_cast<VkBuffer>(first->buffer),staging,1,&copy);
        VkBufferCopy cpu_copy{cpu_snapshot->offset,source.bytes(),source.bytes()};
        vkCmdCopyBuffer(command,static_cast<VkBuffer>(cpu_snapshot->buffer),staging,1,&cpu_copy);
        vkCmdFillBuffer(command,static_cast<VkBuffer>(second->buffer),0,source.bytes(),0x41100000u); // 9.f
        VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_read.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host_read.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&host_read,0,nullptr,0,nullptr);
        check(vkEndCommandBuffer(command));
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};ti.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};si.pNext=&ti;
        VkSemaphore done{};check(vkCreateSemaphore(device,&si,nullptr,&done));
        const uint64_t value=1;VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};timeline.signalSemaphoreValueCount=1;timeline.pSignalSemaphoreValues=&value;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.pNext=&timeline;submit.commandBufferCount=1;submit.pCommandBuffers=&command;submit.signalSemaphoreCount=1;submit.pSignalSemaphores=&done;
        check(vkQueueSubmit(static_cast<VkQueue>(h.queue),1,&submit,VK_NULL_HANDLE));
        const Tensor* tensors[]={&source};interop.wait(tensors,{done,value});
        require(source.cpu().to_vector()==std::vector<float>(publication_count,9.f),"Consumer edits were not returned to HIP");
        void* mapped=nullptr;check(vkMapMemory(device,memory,0,source.bytes()*2,0,&mapped));
        for(size_t i=0;i<publication_count;++i)require(static_cast<float*>(mapped)[i]==3.f,"Older publication lease changed after HIP mutation");
        for(size_t i=0;i<publication_count;++i)require(static_cast<float*>(mapped)[publication_count+i]==23.f,"Completed CPU snapshot changed or was not visible to the consumer GPU");
        vkUnmapMemory(device,memory);
        {
            TensorWorkQueue queue(GpuBackend::HIP,h.device,done,false);
            require(queue.timeline()==nullptr,"Host-synchronized HIP queue exposes a foreign GPU timeline");
            queue.execute([&]{source.fill_(11.f);},value).wait();
            require(source.cpu().to_vector()==std::vector<float>(publication_count,11.f),"HIP consumer queue ordering failed");
            // The real viewport replaces its consumer timeline after training
            // initialization; a constructor-only test misses this transition.
            VkSemaphore replacement{};check(vkCreateSemaphore(device,&si,nullptr,&replacement));
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore=replacement;signal.value=2;check(vkSignalSemaphore(device,&signal));
            queue.set_consumer_timeline(h.device,{replacement,2});
            queue.execute([&]{source.fill_(13.f);},2).wait();
            require(source.cpu().to_vector()==std::vector<float>(publication_count,13.f),"HIP replaced consumer timeline ordering failed");
            queue.set_consumer_timeline(h.device,{done,value});
            bool rejected=false;
            try { queue.set_consumer_timeline(nullptr,{done,value}); }
            catch(const std::invalid_argument&) {rejected=true;}
            require(rejected,"HIP accepted a consumer timeline without its device");
            vkDestroySemaphore(device,replacement,nullptr);
        }
        interop.release_timeline(done);interop.drain(GpuBackend::HIP);
        cache.clear();
        first.reset();second.reset();
        vkDestroySemaphore(device,done,nullptr);vkDestroyCommandPool(device,pool,nullptr);
        vkDestroyBuffer(device,staging,nullptr);vkFreeMemory(device,memory,nullptr);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)),"HIP publication leaked Tensor storage");
    std::puts("PASS HIP viewer fallback: immutable leases, Vulkan GPU read/write, consumer timeline ordering, copyback and teardown");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"HIP VIEWER INTEROP FAILED: %s\n",e.what());return 1;}
