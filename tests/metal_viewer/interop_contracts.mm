/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Actual Metal -> Vulkan texture and timeline transfer using the installed ICD.
#import <Metal/Metal.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_metal.h>
static void check(VkResult r, const char* operation) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(r));
}
static void run() {
    const char* instance_extensions[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    // Export intent is required by VK_EXT_metal_objects, including Metal device.
    VkExportMetalObjectCreateInfoEXT export_device{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT};
    export_device.exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_DEVICE_BIT_EXT;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pNext = &export_device;
    create.pApplicationInfo = &app;
    create.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    create.enabledExtensionCount = 1;
    create.ppEnabledExtensionNames = instance_extensions;
    VkInstance instance;
    check(vkCreateInstance(&create, nullptr, &instance), "vkCreateInstance");
    uint32_t n = 0;
    check(vkEnumeratePhysicalDevices(instance, &n, nullptr), "physical count");
    if (!n)
        throw std::runtime_error("No Vulkan physical device");
    std::vector<VkPhysicalDevice> devices(n);
    check(vkEnumeratePhysicalDevices(instance, &n, devices.data()), "physical devices");
    const auto physical = devices[0];
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    std::printf("Interop device: %s, driver %u\n", properties.deviceName, properties.driverVersion);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &n, families.data());
    uint32_t family = 0;
    while (family < n && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        ++family;
    if (family == n)
        throw std::runtime_error("No graphics queue");
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    const char* extensions[] = {VK_EXT_METAL_OBJECTS_EXTENSION_NAME, "VK_KHR_portability_subset"};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    timeline.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.pNext = &timeline;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 2;
    device_info.ppEnabledExtensionNames = extensions;
    VkDevice device;
    check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice");
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(vkGetDeviceProcAddr(device, "vkExportMetalObjectsEXT"));
    if (!export_objects)
        throw std::runtime_error("Metal objects extension unavailable");
    VkExportMetalDeviceInfoEXT metal_device{VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
    VkExportMetalObjectsInfoEXT exports{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
    exports.pNext = &metal_device;
    export_objects(device, &exports);
    id<MTLDevice> metal = metal_device.mtlDevice;
    if (!metal)
        throw std::runtime_error("Metal device export failed");
    auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:8 height:8 mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    auto texture = [metal newTextureWithDescriptor:desc];
    if (!texture)
        throw std::runtime_error("Texture allocation failed");
    VkImportMetalTextureInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_METAL_TEXTURE_INFO_EXT};
    import.plane = VK_IMAGE_ASPECT_PLANE_0_BIT;
    import.mtlTexture = texture;
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.pNext = &import;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    image_info.extent = {8, 8, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage image;
    check(vkCreateImage(device, &image_info, nullptr, &image), "import Metal texture");
    // Imported Metal textures already have backing memory; do not allocate/bind a second backing.
    auto event = [metal newSharedEvent];
    VkImportMetalSharedEventInfoEXT import_event{VK_STRUCTURE_TYPE_IMPORT_METAL_SHARED_EVENT_INFO_EXT};
    import_event.mtlSharedEvent = event;
    VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    type.pNext = &import_event;
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sem_info.pNext = &type;
    VkSemaphore semaphore;
    check(vkCreateSemaphore(device, &sem_info, nullptr, &semaphore), "import Metal event");
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool pool;
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "command pool");
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command;
    check(vkAllocateCommandBuffers(device, &command_info, &command), "command buffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin), "begin initialization");
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    check(vkEndCommandBuffer(command), "end initialization");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "submit initialization");
    check(vkQueueWaitIdle(queue), "initialize layout");
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = 8 * 8 * 4 * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buffer;
    check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer), "readback buffer");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t index = 0;
    constexpr auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    while (index < memory.memoryTypeCount && (!(req.memoryTypeBits & (1u << index)) || (memory.memoryTypes[index].propertyFlags & host) != host))
        ++index;
    if (index == memory.memoryTypeCount)
        throw std::runtime_error("No coherent host memory");
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = req.size;
    allocation.memoryTypeIndex = index;
    VkDeviceMemory storage;
    check(vkAllocateMemory(device, &allocation, nullptr, &storage), "allocate readback");
    check(vkBindBufferMemory(device, buffer, storage, 0), "bind readback");
    std::array<float, 8 * 8 * 4> expected;
    for (size_t i = 0; i < expected.size(); ++i)
        expected[i] = float(i) / 256;
    auto source = [metal newBufferWithBytes:expected.data() length:sizeof(expected) options:MTLResourceStorageModeShared];
    auto metal_queue = [metal newCommandQueue];
    auto native = [metal_queue commandBuffer];
    auto blit = [native blitCommandEncoder];
    [blit copyFromBuffer:source
               sourceOffset:0
          sourceBytesPerRow:8 * 16
        sourceBytesPerImage:sizeof(expected)
                 sourceSize:MTLSizeMake(8, 8, 1)
                  toTexture:texture
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    [native encodeSignalEvent:event value:1];
    [native commit];
    check(vkResetCommandBuffer(command, 0), "reset command");
    check(vkBeginCommandBuffer(command, &begin), "begin readback");
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {8, 8, 1};
    vkCmdCopyImageToBuffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    check(vkEndCommandBuffer(command), "end readback");
    uint64_t value = 1;
    VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo timeline_submit{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timeline_submit.waitSemaphoreValueCount = 1;
    timeline_submit.pWaitSemaphoreValues = &value;
    submit.pNext = &timeline_submit;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &semaphore;
    submit.pWaitDstStageMask = &stage;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    check(vkCreateFence(device, &fence_info, nullptr, &fence), "fence");
    check(vkQueueSubmit(queue, 1, &submit, fence), "submit interop readback");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull), "Metal to Vulkan completion");
    void* bytes;
    check(vkMapMemory(device, storage, 0, sizeof(expected), 0, &bytes), "map output");
    if (std::memcmp(bytes, expected.data(), sizeof(expected)))
        throw std::runtime_error("Native texture data differs after Vulkan import");
    vkUnmapMemory(device, storage);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyBuffer(device, buffer, nullptr);
    vkFreeMemory(device, storage, nullptr);
    vkDestroyImage(device, image, nullptr);
    vkDestroySemaphore(device, semaphore, nullptr);
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::puts("Metal/Vulkan texture and GPU timeline contract passed.");
}
int main() {
    @autoreleasepool {
        try {
            run();
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}
