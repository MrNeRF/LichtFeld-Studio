#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

    constexpr VkDeviceSize MiB = 1024ull * 1024ull;
    constexpr VkDeviceSize kPoolBlockSize = 64ull * MiB;
    constexpr VkDeviceSize kStagingRingSize = 64ull * MiB;
    constexpr uint32_t kWorkgroupSize = 256;

    [[noreturn]] void fail(std::string_view message) {
        throw std::runtime_error(std::string(message));
    }

    void check(VkResult result, std::string_view operation) {
        if (result != VK_SUCCESS) {
            std::ostringstream stream;
            stream << operation << " failed with VkResult " << result;
            fail(stream.str());
        }
    }

    std::vector<uint32_t> readSpirv(const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
            fail("cannot open SPIR-V file: " + path.string());
        const auto bytes = stream.tellg();
        if (bytes <= 0 || bytes % 4 != 0)
            fail("invalid SPIR-V byte length: " + path.string());
        std::vector<uint32_t> words(static_cast<size_t>(bytes) / sizeof(uint32_t));
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(words.data()), bytes);
        if (!stream || words.front() != 0x07230203u)
            fail("invalid SPIR-V module: " + path.string());
        return words;
    }

    struct SpirvFacts {
        std::set<uint32_t> capabilities;
        std::set<uint32_t> specializationIds;
        bool hasSpecConstant = false;
        uint32_t addressingModel = std::numeric_limits<uint32_t>::max();
    };

    SpirvFacts inspectSpirv(std::span<const uint32_t> words) {
        SpirvFacts facts;
        for (size_t offset = 5; offset < words.size();) {
            const uint32_t instruction = words[offset];
            const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16u);
            const uint16_t opcode = static_cast<uint16_t>(instruction & 0xffffu);
            if (wordCount == 0 || offset + wordCount > words.size())
                fail("malformed SPIR-V instruction stream");
            if (opcode == 17 && wordCount >= 2)
                facts.capabilities.insert(words[offset + 1]);
            if (opcode == 14 && wordCount >= 3)
                facts.addressingModel = words[offset + 1];
            if (opcode == 50)
                facts.hasSpecConstant = true;
            if (opcode == 71 && wordCount >= 4 && words[offset + 2] == 1)
                facts.specializationIds.insert(words[offset + 3]);
            offset += wordCount;
        }
        return facts;
    }

    std::string capabilityName(uint32_t capability) {
        switch (capability) {
        case 1:
            return "Shader";
        case 11:
            return "Int64";
        case 22:
            return "Int16";
        case 61:
            return "GroupNonUniform";
        case 63:
            return "GroupNonUniformArithmetic";
        case 4433:
            return "StorageBuffer16BitAccess";
        case 5347:
            return "PhysicalStorageBufferAddresses";
        default:
            return "Capability" + std::to_string(capability);
        }
    }

    std::string joinCapabilities(const std::set<uint32_t>& capabilities) {
        std::ostringstream stream;
        bool first = true;
        for (const uint32_t capability : capabilities) {
            if (!first)
                stream << ',';
            first = false;
            stream << capabilityName(capability) << '(' << capability << ')';
        }
        return stream.str();
    }

    bool hasLayer(std::string_view name) {
        uint32_t count = 0;
        check(vkEnumerateInstanceLayerProperties(&count, nullptr),
              "vkEnumerateInstanceLayerProperties(count)");
        std::vector<VkLayerProperties> layers(count);
        check(vkEnumerateInstanceLayerProperties(&count, layers.data()),
              "vkEnumerateInstanceLayerProperties(data)");
        return std::ranges::any_of(layers, [name](const auto& layer) {
            return name == layer.layerName;
        });
    }

    bool hasInstanceExtension(std::string_view name) {
        uint32_t count = 0;
        check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr),
              "vkEnumerateInstanceExtensionProperties(count)");
        std::vector<VkExtensionProperties> extensions(count);
        check(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()),
              "vkEnumerateInstanceExtensionProperties(data)");
        return std::ranges::any_of(extensions, [name](const auto& extension) {
            return name == extension.extensionName;
        });
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT,
                                                 const VkDebugUtilsMessengerCallbackDataEXT* data,
                                                 void* userData) {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
            auto& messages = *static_cast<std::vector<std::string>*>(userData);
            messages.emplace_back(data != nullptr && data->pMessage != nullptr ? data->pMessage
                                                                               : "empty validation message");
        }
        return VK_FALSE;
    }

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo info{};
        VkDeviceSize size = 0;
    };

    struct Pipeline {
        VkShaderModule shader = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        double creationMilliseconds = 0.0;
    };

    class VulkanContext {
    public:
        VulkanContext() { initialize(); }

        ~VulkanContext() {
            if (device_ != VK_NULL_HANDLE)
                vkDeviceWaitIdle(device_);
            if (timeline_ != VK_NULL_HANDLE)
                vkDestroySemaphore(device_, timeline_, nullptr);
            if (commandPool_ != VK_NULL_HANDLE)
                vkDestroyCommandPool(device_, commandPool_, nullptr);
            if (allocator_ != VK_NULL_HANDLE)
                vmaDestroyAllocator(allocator_);
            if (device_ != VK_NULL_HANDLE)
                vkDestroyDevice(device_, nullptr);
            if (debugMessenger_ != VK_NULL_HANDLE) {
                const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
                if (destroy != nullptr)
                    destroy(instance_, debugMessenger_, nullptr);
            }
            if (instance_ != VK_NULL_HANDLE)
                vkDestroyInstance(instance_, nullptr);
        }

        VulkanContext(const VulkanContext&) = delete;
        VulkanContext& operator=(const VulkanContext&) = delete;

        [[nodiscard]] VkDevice device() const { return device_; }
        [[nodiscard]] VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
        [[nodiscard]] VmaAllocator allocator() const { return allocator_; }
        [[nodiscard]] const VkPhysicalDeviceProperties& properties() const { return properties_; }
        [[nodiscard]] const VkPhysicalDeviceMemoryProperties& memoryProperties() const {
            return memoryProperties_;
        }
        [[nodiscard]] uint32_t subgroupSize() const { return subgroupProperties_.subgroupSize; }
        [[nodiscard]] uint32_t minSubgroupSize() const { return vulkan13Properties_.minSubgroupSize; }
        [[nodiscard]] uint32_t maxSubgroupSize() const { return vulkan13Properties_.maxSubgroupSize; }
        [[nodiscard]] bool memoryBudgetEnabled() const { return memoryBudgetEnabled_; }
        [[nodiscard]] bool shaderAtomicFloatEnabled() const { return shaderAtomicFloatEnabled_; }
        [[nodiscard]] bool validationEnabled() const { return validationEnabled_; }
        [[nodiscard]] const std::vector<std::string>& validationMessages() const {
            return validationMessages_;
        }

        Buffer createBuffer(VkDeviceSize size,
                            VkBufferUsageFlags usage,
                            VmaMemoryUsage memoryUsage,
                            VmaAllocationCreateFlags flags = 0,
                            VkMemoryPropertyFlags requiredFlags = 0,
                            VmaPool pool = VK_NULL_HANDLE) const {
            VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo allocationInfo{};
            allocationInfo.flags = flags;
            allocationInfo.usage = memoryUsage;
            allocationInfo.requiredFlags = requiredFlags;
            allocationInfo.pool = pool;
            Buffer result;
            result.size = size;
            check(vmaCreateBuffer(
                      allocator_, &bufferInfo, &allocationInfo, &result.buffer, &result.allocation, &result.info),
                  "vmaCreateBuffer");
            return result;
        }

        void destroyBuffer(Buffer& buffer) const {
            if (buffer.buffer != VK_NULL_HANDLE)
                vmaDestroyBuffer(allocator_, buffer.buffer, buffer.allocation);
            buffer = {};
        }

        [[nodiscard]] VkDeviceAddress address(const Buffer& buffer) const {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = buffer.buffer;
            const VkDeviceAddress result = vkGetBufferDeviceAddress(device_, &info);
            if (result == 0)
                fail("vkGetBufferDeviceAddress returned zero");
            return result;
        }

        Pipeline createPipeline(const std::filesystem::path& path,
                                std::span<const uint32_t> specializationValues = {}) const {
            const auto words = readSpirv(path);
            VkShaderModuleCreateInfo shaderInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shaderInfo.codeSize = words.size() * sizeof(uint32_t);
            shaderInfo.pCode = words.data();
            Pipeline result;
            check(vkCreateShaderModule(device_, &shaderInfo, nullptr, &result.shader),
                  "vkCreateShaderModule");

            VkPushConstantRange pushRange{};
            pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushRange.size = 48;
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.pushConstantRangeCount = 1;
            layoutInfo.pPushConstantRanges = &pushRange;
            check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &result.layout),
                  "vkCreatePipelineLayout");

            std::vector<VkSpecializationMapEntry> entries;
            entries.reserve(specializationValues.size());
            for (uint32_t i = 0; i < specializationValues.size(); ++i)
                entries.push_back(
                    {i, static_cast<uint32_t>(i * sizeof(uint32_t)), sizeof(uint32_t)});
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = static_cast<uint32_t>(entries.size());
            specialization.pMapEntries = entries.data();
            specialization.dataSize = specializationValues.size_bytes();
            specialization.pData = specializationValues.data();
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = result.shader;
            stage.pName = "main";
            stage.pSpecializationInfo = specializationValues.empty() ? nullptr : &specialization;
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = stage;
            pipelineInfo.layout = result.layout;
            const auto start = std::chrono::steady_clock::now();
            check(vkCreateComputePipelines(
                      device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &result.pipeline),
                  "vkCreateComputePipelines");
            const auto end = std::chrono::steady_clock::now();
            result.creationMilliseconds =
                std::chrono::duration<double, std::milli>(end - start).count();
            return result;
        }

        void destroyPipeline(Pipeline& pipeline) const {
            if (pipeline.pipeline != VK_NULL_HANDLE)
                vkDestroyPipeline(device_, pipeline.pipeline, nullptr);
            if (pipeline.layout != VK_NULL_HANDLE)
                vkDestroyPipelineLayout(device_, pipeline.layout, nullptr);
            if (pipeline.shader != VK_NULL_HANDLE)
                vkDestroyShaderModule(device_, pipeline.shader, nullptr);
            pipeline = {};
        }

        VkCommandBuffer beginCommands() const {
            VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocateInfo.commandPool = commandPool_;
            allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocateInfo.commandBufferCount = 1;
            VkCommandBuffer command = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(device_, &allocateInfo, &command),
                  "vkAllocateCommandBuffers");
            VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(command, &beginInfo), "vkBeginCommandBuffer");
            return command;
        }

        uint64_t submitAndWait(VkCommandBuffer command) {
            check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
            const uint64_t value = ++timelineValue_;
            VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            commandInfo.commandBuffer = command;
            VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            signalInfo.semaphore = timeline_;
            signalInfo.value = value;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &commandInfo;
            submit.signalSemaphoreInfoCount = 1;
            submit.pSignalSemaphoreInfos = &signalInfo;
            check(vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit2");
            waitTimeline(value);
            vkFreeCommandBuffers(device_, commandPool_, 1, &command);
            return value;
        }

        void waitTimeline(uint64_t value) const {
            VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores = &timeline_;
            waitInfo.pValues = &value;
            check(vkWaitSemaphores(device_, &waitInfo, std::numeric_limits<uint64_t>::max()),
                  "vkWaitSemaphores");
        }

        [[nodiscard]] uint64_t completedTimeline() const {
            uint64_t value = 0;
            check(vkGetSemaphoreCounterValue(device_, timeline_, &value),
                  "vkGetSemaphoreCounterValue");
            return value;
        }

        static void transferToComputeBarrier(VkCommandBuffer command) {
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

        static void computeToTransferBarrier(VkCommandBuffer command) {
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

        static void computeToComputeBarrier(VkCommandBuffer command) {
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.memoryBarrierCount = 1;
            dependency.pMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

    private:
        void initialize() {
            validationEnabled_ = hasLayer("VK_LAYER_KHRONOS_validation");
            const bool debugUtils = hasInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            std::vector<const char*> layers;
            std::vector<const char*> instanceExtensions;
            if (validationEnabled_)
                layers.push_back("VK_LAYER_KHRONOS_validation");
            if (debugUtils)
                instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.pApplicationName = "LichtFeld Vulkan P0 probes";
            application.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instanceInfo.pApplicationInfo = &application;
            instanceInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
            instanceInfo.ppEnabledLayerNames = layers.data();
            instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
            instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
            check(vkCreateInstance(&instanceInfo, nullptr, &instance_), "vkCreateInstance");

            if (debugUtils) {
                VkDebugUtilsMessengerCreateInfoEXT debugInfo{
                    VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
                debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                debugInfo.pfnUserCallback = debugCallback;
                debugInfo.pUserData = &validationMessages_;
                const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
                if (create != nullptr)
                    check(create(instance_, &debugInfo, nullptr, &debugMessenger_),
                          "vkCreateDebugUtilsMessengerEXT");
            }

            selectPhysicalDevice();
            queryFeaturesAndCreateDevice();
            vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);

            VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = queueFamily_;
            check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
                  "vkCreateCommandPool");

            VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            semaphoreInfo.pNext = &typeInfo;
            check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &timeline_),
                  "vkCreateSemaphore");

            VmaAllocatorCreateInfo allocatorInfo{};
            allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
            if (memoryBudgetEnabled_)
                allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
            allocatorInfo.physicalDevice = physicalDevice_;
            allocatorInfo.device = device_;
            allocatorInfo.instance = instance_;
            allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
            check(vmaCreateAllocator(&allocatorInfo, &allocator_), "vmaCreateAllocator");
        }

        void selectPhysicalDevice() {
            uint32_t count = 0;
            check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                  "vkEnumeratePhysicalDevices(count)");
            if (count == 0)
                fail("no Vulkan physical devices");
            std::vector<VkPhysicalDevice> devices(count);
            check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                  "vkEnumeratePhysicalDevices(data)");
            physicalDevice_ = devices.front();
            for (const auto device : devices) {
                VkPhysicalDeviceProperties candidate{};
                vkGetPhysicalDeviceProperties(device, &candidate);
                if (candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                    physicalDevice_ = device;
                    break;
                }
            }
            vkGetPhysicalDeviceProperties(physicalDevice_, &properties_);

            uint32_t queueCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueCount, nullptr);
            std::vector<VkQueueFamilyProperties> queues(queueCount);
            vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueCount, queues.data());
            for (uint32_t i = 0; i < queueCount; ++i) {
                if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    queueFamily_ = i;
                    return;
                }
            }
            fail("no Vulkan compute queue family");
        }

        void queryFeaturesAndCreateDevice() {
            uint32_t extensionCount = 0;
            check(vkEnumerateDeviceExtensionProperties(
                      physicalDevice_, nullptr, &extensionCount, nullptr),
                  "vkEnumerateDeviceExtensionProperties(count)");
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            check(vkEnumerateDeviceExtensionProperties(
                      physicalDevice_, nullptr, &extensionCount, extensionProperties.data()),
                  "vkEnumerateDeviceExtensionProperties(data)");
            std::unordered_set<std::string> availableExtensions;
            for (const auto& extension : extensionProperties)
                availableExtensions.emplace(extension.extensionName);
            memoryBudgetEnabled_ = availableExtensions.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            shaderAtomicFloatEnabled_ =
                availableExtensions.contains(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
            VkPhysicalDeviceVulkan13Features vulkan13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceVulkan12Features vulkan12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceVulkan11Features vulkan11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &vulkan11;
            vulkan11.pNext = &vulkan12;
            vulkan12.pNext = &vulkan13;
            vulkan13.pNext = &atomicFloat;
            vkGetPhysicalDeviceFeatures2(physicalDevice_, &features);
            if (!features.features.shaderInt64 || !features.features.shaderInt16 ||
                !vulkan11.storageBuffer16BitAccess || !vulkan12.storageBuffer8BitAccess ||
                !vulkan12.timelineSemaphore || !vulkan12.bufferDeviceAddress ||
                !vulkan13.synchronization2)
                fail("device is missing a D12 required Vulkan feature");

            features.features = {};
            features.features.shaderInt64 = VK_TRUE;
            features.features.shaderInt16 = VK_TRUE;
            vulkan11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            vulkan12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            vulkan13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            atomicFloat = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
            features.pNext = &vulkan11;
            vulkan11.pNext = &vulkan12;
            vulkan12.pNext = &vulkan13;
            vulkan13.pNext = shaderAtomicFloatEnabled_ ? &atomicFloat : nullptr;
            vulkan11.storageBuffer16BitAccess = VK_TRUE;
            vulkan12.storageBuffer8BitAccess = VK_TRUE;
            vulkan12.shaderFloat16 = VK_TRUE;
            vulkan12.timelineSemaphore = VK_TRUE;
            vulkan12.bufferDeviceAddress = VK_TRUE;
            vulkan13.synchronization2 = VK_TRUE;
            if (shaderAtomicFloatEnabled_)
                atomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;

            std::vector<const char*> extensions;
            if (memoryBudgetEnabled_)
                extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
            if (shaderAtomicFloatEnabled_)
                extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueInfo.queueFamilyIndex = queueFamily_;
            queueInfo.queueCount = 1;
            queueInfo.pQueuePriorities = &priority;
            VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            deviceInfo.pNext = &features;
            deviceInfo.queueCreateInfoCount = 1;
            deviceInfo.pQueueCreateInfos = &queueInfo;
            deviceInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            deviceInfo.ppEnabledExtensionNames = extensions.data();
            check(vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_), "vkCreateDevice");
            vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);

            vulkan13Properties_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
            subgroupProperties_ = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties2.pNext = &subgroupProperties_;
            subgroupProperties_.pNext = &vulkan13Properties_;
            vkGetPhysicalDeviceProperties2(physicalDevice_, &properties2);
        }

        VkInstance instance_ = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
        VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue queue_ = VK_NULL_HANDLE;
        uint32_t queueFamily_ = 0;
        VkCommandPool commandPool_ = VK_NULL_HANDLE;
        VkSemaphore timeline_ = VK_NULL_HANDLE;
        uint64_t timelineValue_ = 0;
        VmaAllocator allocator_ = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties properties_{};
        VkPhysicalDeviceMemoryProperties memoryProperties_{};
        VkPhysicalDeviceSubgroupProperties subgroupProperties_{};
        VkPhysicalDeviceVulkan13Properties vulkan13Properties_{};
        bool memoryBudgetEnabled_ = false;
        bool shaderAtomicFloatEnabled_ = false;
        bool validationEnabled_ = false;
        std::vector<std::string> validationMessages_;
    };

    struct AbiPush {
        uint64_t inputAddress;
        uint64_t outputAddress;
        uint32_t count;
        uint32_t padding;
    };

    struct ReductionPush {
        uint64_t inputAddress;
        uint64_t outputAddress;
        uint32_t totalColumns;
        uint32_t rowCount;
        uint32_t rowStride;
        uint32_t segmentSize;
        uint32_t segmentCount;
        uint32_t padding;
    };

    static_assert(sizeof(AbiPush) == 24);
    static_assert(sizeof(ReductionPush) == 40);

    template <typename T>
    std::vector<T> executeAbi(VulkanContext& context,
                              const Pipeline& pipeline,
                              std::span<const T> input,
                              uint32_t inputOffsetElements = 0,
                              uint32_t outputOffsetElements = 0,
                              VmaPool pool = VK_NULL_HANDLE,
                              uint64_t* timelineValue = nullptr) {
        const VkDeviceSize inputBytes = (input.size() + inputOffsetElements) * sizeof(T);
        const VkDeviceSize outputBytes = (input.size() + outputOffsetElements) * sizeof(T);
        const VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        Buffer deviceInput = context.createBuffer(
            inputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, pool);
        Buffer deviceOutput = context.createBuffer(
            outputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, pool);
        Buffer staging = context.createBuffer(
            inputBytes + outputBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto* mapped = static_cast<std::byte*>(staging.info.pMappedData);
        std::memset(mapped, 0, static_cast<size_t>(inputBytes + outputBytes));
        std::memcpy(mapped + inputOffsetElements * sizeof(T), input.data(), input.size_bytes());

        VkCommandBuffer command = context.beginCommands();
        VkBufferCopy upload{};
        upload.size = inputBytes;
        vkCmdCopyBuffer(command, staging.buffer, deviceInput.buffer, 1, &upload);
        VulkanContext::transferToComputeBarrier(command);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        const AbiPush push{context.address(deviceInput) + inputOffsetElements * sizeof(T),
                           context.address(deviceOutput) + outputOffsetElements * sizeof(T),
                           static_cast<uint32_t>(input.size()),
                           0};
        vkCmdPushConstants(command,
                           pipeline.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        vkCmdDispatch(command,
                      (static_cast<uint32_t>(input.size()) + kWorkgroupSize - 1) / kWorkgroupSize,
                      1,
                      1);
        VulkanContext::computeToTransferBarrier(command);
        VkBufferCopy download{};
        download.srcOffset = 0;
        download.dstOffset = inputBytes;
        download.size = outputBytes;
        vkCmdCopyBuffer(command, deviceOutput.buffer, staging.buffer, 1, &download);
        const uint64_t completed = context.submitAndWait(command);
        if (timelineValue != nullptr)
            *timelineValue = completed;

        std::vector<T> result(input.size());
        std::memcpy(result.data(),
                    mapped + inputBytes + outputOffsetElements * sizeof(T),
                    result.size() * sizeof(T));
        context.destroyBuffer(staging);
        context.destroyBuffer(deviceOutput);
        context.destroyBuffer(deviceInput);
        return result;
    }

    std::vector<float> executeReduction(VulkanContext& context,
                                        const Pipeline& pipeline,
                                        std::span<const float> input,
                                        uint32_t columns,
                                        uint32_t rows,
                                        uint32_t rowStride) {
        const VkDeviceSize inputBytes = input.size_bytes();
        const VkDeviceSize outputBytes = rows * sizeof(float);
        const VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        Buffer deviceInput = context.createBuffer(
            inputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer deviceOutput = context.createBuffer(
            outputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer staging = context.createBuffer(
            inputBytes + outputBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto* mapped = static_cast<std::byte*>(staging.info.pMappedData);
        std::memcpy(mapped, input.data(), inputBytes);

        VkCommandBuffer command = context.beginCommands();
        VkBufferCopy upload{0, 0, inputBytes};
        vkCmdCopyBuffer(command, staging.buffer, deviceInput.buffer, 1, &upload);
        VulkanContext::transferToComputeBarrier(command);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        const ReductionPush push{context.address(deviceInput),
                                 context.address(deviceOutput),
                                 columns,
                                 rows,
                                 rowStride,
                                 columns,
                                 1,
                                 0};
        vkCmdPushConstants(command,
                           pipeline.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        vkCmdDispatch(command, rows, 1, 1);
        VulkanContext::computeToTransferBarrier(command);
        VkBufferCopy download{0, inputBytes, outputBytes};
        vkCmdCopyBuffer(command, deviceOutput.buffer, staging.buffer, 1, &download);
        context.submitAndWait(command);

        std::vector<float> result(rows);
        std::memcpy(result.data(), mapped + inputBytes, outputBytes);
        context.destroyBuffer(staging);
        context.destroyBuffer(deviceOutput);
        context.destroyBuffer(deviceInput);
        return result;
    }

    std::vector<float> executeTwoPassAxisReduction(VulkanContext& context,
                                                   const Pipeline& pipeline,
                                                   std::span<const float> input,
                                                   uint32_t columns,
                                                   uint32_t rows) {
        constexpr uint32_t segmentSize = 256;
        const uint32_t segmentCount = (columns + segmentSize - 1) / segmentSize;
        const VkDeviceSize inputBytes = input.size_bytes();
        const VkDeviceSize partialBytes =
            static_cast<VkDeviceSize>(rows) * segmentCount * sizeof(float);
        const VkDeviceSize outputBytes = static_cast<VkDeviceSize>(rows) * sizeof(float);
        const VkBufferUsageFlags deviceUsage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        Buffer deviceInput = context.createBuffer(
            inputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer devicePartials = context.createBuffer(
            partialBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer deviceOutput = context.createBuffer(
            outputBytes, deviceUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
        Buffer staging = context.createBuffer(
            inputBytes + outputBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto* mapped = static_cast<std::byte*>(staging.info.pMappedData);
        std::memcpy(mapped, input.data(), inputBytes);

        VkCommandBuffer command = context.beginCommands();
        VkBufferCopy upload{0, 0, inputBytes};
        vkCmdCopyBuffer(command, staging.buffer, deviceInput.buffer, 1, &upload);
        VulkanContext::transferToComputeBarrier(command);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        const ReductionPush firstPass{context.address(deviceInput),
                                      context.address(devicePartials),
                                      columns,
                                      rows,
                                      columns,
                                      segmentSize,
                                      segmentCount,
                                      0};
        vkCmdPushConstants(command,
                           pipeline.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(firstPass),
                           &firstPass);
        vkCmdDispatch(command, rows * segmentCount, 1, 1);
        VulkanContext::computeToComputeBarrier(command);
        const ReductionPush secondPass{context.address(devicePartials),
                                       context.address(deviceOutput),
                                       segmentCount,
                                       rows,
                                       segmentCount,
                                       segmentCount,
                                       1,
                                       0};
        vkCmdPushConstants(command,
                           pipeline.layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(secondPass),
                           &secondPass);
        vkCmdDispatch(command, rows, 1, 1);
        VulkanContext::computeToTransferBarrier(command);
        VkBufferCopy download{0, inputBytes, outputBytes};
        vkCmdCopyBuffer(command, deviceOutput.buffer, staging.buffer, 1, &download);
        context.submitAndWait(command);

        std::vector<float> result(rows);
        std::memcpy(result.data(), mapped + inputBytes, outputBytes);
        context.destroyBuffer(staging);
        context.destroyBuffer(deviceOutput);
        context.destroyBuffer(devicePartials);
        context.destroyBuffer(deviceInput);
        return result;
    }

    template <typename T>
    void requireEqual(std::span<const T> actual, std::span<const T> expected, std::string_view label) {
        if (actual.size() != expected.size())
            fail(std::string(label) + ": size mismatch");
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &expected[i], sizeof(T)) != 0) {
                std::ostringstream stream;
                stream << label << ": mismatch at " << i;
                fail(stream.str());
            }
        }
    }

    void probeAbi(VulkanContext& context, const std::filesystem::path& shaderPath) {
        const auto facts = inspectSpirv(readSpirv(shaderPath));
        if (!facts.hasSpecConstant || facts.specializationIds != std::set<uint32_t>{0, 1})
            fail("ABI shader did not contain the expected OpSpecConstant IDs 0 and 1");
        if (!facts.capabilities.contains(5347) || facts.addressingModel != 5348)
            fail("ABI shader did not declare PhysicalStorageBuffer64 addressing");
        std::cout << "PROBE1 spirv_capabilities=" << joinCapabilities(facts.capabilities)
                  << " addressing_model=PhysicalStorageBuffer64(5348) spec_ids=0,1\n";

        const std::array<uint32_t, 2> floatSpecialization{0, 0};
        const std::array<uint32_t, 2> intSpecialization{1, 1};
        const std::array<uint32_t, 2> u16Specialization{0, 2};
        Pipeline floatPipeline = context.createPipeline(shaderPath, floatSpecialization);
        Pipeline intPipeline = context.createPipeline(shaderPath, intSpecialization);
        Pipeline u16Pipeline = context.createPipeline(shaderPath, u16Specialization);
        std::cout << std::fixed << std::setprecision(3)
                  << "PROBE1 pipeline_creation_ms float_add=" << floatPipeline.creationMilliseconds
                  << " int_xor=" << intPipeline.creationMilliseconds
                  << " u16_add=" << u16Pipeline.creationMilliseconds << '\n';

        constexpr size_t count = 1'048'581;
        std::vector<float> floatInput(count);
        std::vector<float> floatExpected(count);
        for (size_t i = 0; i < count; ++i) {
            floatInput[i] = static_cast<float>(static_cast<int>(i % 4096) - 2048);
            floatExpected[i] = floatInput[i] + 1.0f;
        }
        requireEqual<float>(executeAbi<float>(context, floatPipeline, floatInput),
                            floatExpected,
                            "probe1 float32");

        std::vector<int32_t> intInput(count);
        std::vector<int32_t> intExpected(count);
        for (size_t i = 0; i < count; ++i) {
            intInput[i] = static_cast<int32_t>(i * 2654435761u);
            intExpected[i] = intInput[i] ^ static_cast<int32_t>(0x5a5a5a5a);
        }
        requireEqual<int32_t>(executeAbi<int32_t>(context, intPipeline, intInput),
                              intExpected,
                              "probe1 int32");

        const std::vector<float> offsetFloatInput{0.0f, 1.0f, -2.0f, 17.0f, 1024.0f, -0.0f, 8.0f};
        std::vector<float> offsetFloatExpected(offsetFloatInput.size());
        std::ranges::transform(offsetFloatInput, offsetFloatExpected.begin(), [](float value) {
            return value + 1.0f;
        });
        requireEqual<float>(executeAbi<float>(context, floatPipeline, offsetFloatInput, 3, 3),
                            offsetFloatExpected,
                            "probe1 float32 odd element offset");

        const std::vector<uint16_t> offsetU16Input{0, 1, 2, 17, 65530, 42, 7};
        std::vector<uint16_t> offsetU16Expected(offsetU16Input.size());
        std::ranges::transform(offsetU16Input, offsetU16Expected.begin(), [](uint16_t value) {
            return static_cast<uint16_t>(value + 1);
        });
        requireEqual<uint16_t>(executeAbi<uint16_t>(context, u16Pipeline, offsetU16Input, 3, 3),
                               offsetU16Expected,
                               "probe1 uint16 odd element offset");

        context.destroyPipeline(u16Pipeline);
        context.destroyPipeline(intPipeline);
        context.destroyPipeline(floatPipeline);
        std::cout << "PROBE1 PASS device=" << context.properties().deviceName
                  << " count=1048581 float32=bit-exact int32=bit-exact"
                     " offset4_element3=bit-exact offset2_element3=bit-exact\n";
    }

    void probeReduction(VulkanContext& context, const std::filesystem::path& shaderPath) {
        const auto facts = inspectSpirv(readSpirv(shaderPath));
        if (!facts.capabilities.contains(61) || !facts.capabilities.contains(63))
            fail("reduction shader lacks subgroup arithmetic capabilities");
        Pipeline pipeline = context.createPipeline(shaderPath);
        constexpr std::array<uint32_t, 8> counts{1, 7, 31, 32, 33, 1000, 65537, 1048583};
        for (const uint32_t count : counts) {
            std::vector<float> input(count);
            for (uint32_t i = 0; i < count; ++i)
                input[i] = static_cast<float>(i % 4);
            const float expected = std::accumulate(input.begin(), input.end(), 0.0f);
            const auto actual = executeReduction(context, pipeline, input, count, 1, count);
            requireEqual<float>(actual, std::span<const float>(&expected, 1), "probe2 vector reduction");
        }

        constexpr uint32_t rows = 1031;
        constexpr uint32_t columns = 517;
        std::vector<float> matrix(static_cast<size_t>(rows) * columns);
        std::vector<float> expected(rows, 0.0f);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < columns; ++column) {
                const float value = static_cast<float>((row + column) % 3);
                matrix[static_cast<size_t>(row) * columns + column] = value;
                expected[row] += value;
            }
        }
        requireEqual<float>(executeTwoPassAxisReduction(context, pipeline, matrix, columns, rows),
                            expected,
                            "probe2 axis reduction");
        context.destroyPipeline(pipeline);
        std::cout << "PROBE2 PASS device=" << context.properties().deviceName
                  << " subgroup_default=" << context.subgroupSize()
                  << " subgroup_min=" << context.minSubgroupSize()
                  << " subgroup_max=" << context.maxSubgroupSize()
                  << " vector_counts=1,7,31,32,33,1000,65537,1048583"
                     " axis=1031x517 two-pass bit-exact no-hardcoded-subgroup-size\n";
    }

    struct BudgetSnapshot {
        VkDeviceSize usage = 0;
        VkDeviceSize budget = 0;
    };

    VmaStatistics allocatorStatistics(const VulkanContext& context) {
        VmaTotalStatistics statistics{};
        vmaCalculateStatistics(context.allocator(), &statistics);
        return statistics.total.statistics;
    }

    BudgetSnapshot deviceLocalBudget(const VulkanContext& context) {
        std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
        vmaGetHeapBudgets(context.allocator(), budgets.data());
        BudgetSnapshot result;
        const auto& memory = context.memoryProperties();
        for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
            if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                result.usage += budgets[i].usage;
                result.budget += budgets[i].budget;
            }
        }
        return result;
    }

    bool hasHostVisibleDeviceLocalMemory(const VulkanContext& context) {
        const auto& memory = context.memoryProperties();
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags flags = memory.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0)
                return true;
        }
        return false;
    }

    VmaPool createDevicePool(const VulkanContext& context) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = 256;
        bufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo allocationInfo{};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        allocationInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t memoryTypeIndex = 0;
        check(vmaFindMemoryTypeIndexForBufferInfo(
                  context.allocator(), &bufferInfo, &allocationInfo, &memoryTypeIndex),
              "vmaFindMemoryTypeIndexForBufferInfo");
        VmaPoolCreateInfo poolInfo{};
        poolInfo.memoryTypeIndex = memoryTypeIndex;
        poolInfo.blockSize = kPoolBlockSize;
        VmaPool pool = VK_NULL_HANDLE;
        check(vmaCreatePool(context.allocator(), &poolInfo, &pool), "vmaCreatePool");
        return pool;
    }

    void probeVma(VulkanContext& context, const std::filesystem::path& shaderPath) {
        const BudgetSnapshot before = deviceLocalBudget(context);
        const VmaStatistics allocatorBefore = allocatorStatistics(context);
        VmaPool pool = createDevicePool(context);
        constexpr std::array<VkDeviceSize, 16> sizes{
            256, 512, 1024, 4096, 16384, 65536, 131072, 262144,
            524288, MiB, 2 * MiB, 4 * MiB, 8 * MiB, 768, 32768, 3 * MiB};
        const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        std::vector<Buffer> allocations;
        allocations.reserve(200);
        std::map<uint32_t, uint32_t> alignmentHistogram;
        std::unordered_set<VkDeviceAddress> initialAddresses;
        for (uint32_t i = 0; i < 200; ++i) {
            allocations.push_back(context.createBuffer(
                sizes[i % sizes.size()], usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, pool));
            const VkDeviceAddress address = context.address(allocations.back());
            initialAddresses.insert(address);
            const uint32_t lowBit = std::countr_zero(address);
            ++alignmentHistogram[lowBit];
        }
        const BudgetSnapshot peak = deviceLocalBudget(context);
        const VmaStatistics allocatorPeak = allocatorStatistics(context);
        std::cout << "PROBE3 address_lowest_set_bit_histogram";
        for (const auto [bit, count] : alignmentHistogram)
            std::cout << " 2^" << bit << "=" << count;
        std::cout << '\n';
        for (auto& allocation : allocations)
            context.destroyBuffer(allocation);
        allocations.clear();

        Buffer stagingRing = context.createBuffer(
            kStagingRingSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto* ringData = static_cast<std::byte*>(stagingRing.info.pMappedData);
        VkDeviceSize ringHead = 0;
        uint32_t ringWraps = 0;
        uint32_t reusedDeviceAddresses = 0;
        uint64_t lastRetireValue = 0;
        std::unordered_set<VkDeviceAddress> retiredAddresses;
        const std::array<uint32_t, 2> specialization{0, 1};
        Pipeline pipeline = context.createPipeline(shaderPath, specialization);
        constexpr uint32_t elementCount = 262144;
        constexpr VkDeviceSize payloadBytes = elementCount * sizeof(int32_t);

        for (uint32_t iteration = 0; iteration < 50; ++iteration) {
            if (ringHead + 2 * payloadBytes > kStagingRingSize) {
                if (lastRetireValue != 0)
                    context.waitTimeline(lastRetireValue);
                ringHead = 0;
                ++ringWraps;
            }
            const VkDeviceSize uploadOffset = ringHead;
            const VkDeviceSize downloadOffset = ringHead + payloadBytes;
            ringHead += 2 * payloadBytes;
            auto* upload = reinterpret_cast<int32_t*>(ringData + uploadOffset);
            for (uint32_t i = 0; i < elementCount; ++i)
                upload[i] = static_cast<int32_t>((iteration * 131u + i) & 0x7fffffffu);

            Buffer input = context.createBuffer(
                payloadBytes, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, pool);
            Buffer output = context.createBuffer(
                payloadBytes, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 0, pool);
            const VkDeviceAddress inputAddress = context.address(input);
            const VkDeviceAddress outputAddress = context.address(output);
            reusedDeviceAddresses += retiredAddresses.contains(inputAddress) ? 1 : 0;
            reusedDeviceAddresses += retiredAddresses.contains(outputAddress) ? 1 : 0;

            VkCommandBuffer command = context.beginCommands();
            VkBufferCopy uploadCopy{uploadOffset, 0, payloadBytes};
            vkCmdCopyBuffer(command, stagingRing.buffer, input.buffer, 1, &uploadCopy);
            VulkanContext::transferToComputeBarrier(command);
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            const AbiPush push{inputAddress, outputAddress, elementCount, 0};
            vkCmdPushConstants(command,
                               pipeline.layout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(push),
                               &push);
            vkCmdDispatch(command, (elementCount + kWorkgroupSize - 1) / kWorkgroupSize, 1, 1);
            VulkanContext::computeToTransferBarrier(command);
            VkBufferCopy downloadCopy{0, downloadOffset, payloadBytes};
            vkCmdCopyBuffer(command, output.buffer, stagingRing.buffer, 1, &downloadCopy);
            lastRetireValue = context.submitAndWait(command);
            if (context.completedTimeline() < lastRetireValue)
                fail("timeline retirement value was not complete before host reuse");

            const auto* download = reinterpret_cast<const int32_t*>(ringData + downloadOffset);
            for (uint32_t i = 0; i < elementCount; ++i) {
                const int32_t expected = upload[i] + 1;
                if (download[i] != expected)
                    fail("probe3 roundtrip mismatch");
            }
            retiredAddresses.insert(inputAddress);
            retiredAddresses.insert(outputAddress);
            context.destroyBuffer(output);
            context.destroyBuffer(input);
        }
        context.destroyPipeline(pipeline);
        context.destroyBuffer(stagingRing);
        vmaDestroyPool(context.allocator(), pool);
        const BudgetSnapshot after = deviceLocalBudget(context);
        const VmaStatistics allocatorAfter = allocatorStatistics(context);

        std::cout << "PROBE3 budget_bytes before_usage=" << before.usage
                  << " before_budget=" << before.budget << " peak_usage=" << peak.usage
                  << " peak_budget=" << peak.budget << " after_usage=" << after.usage
                  << " after_budget=" << after.budget << '\n';
        std::cout << "PROBE3 vma_bytes before_allocation=" << allocatorBefore.allocationBytes
                  << " before_blocks=" << allocatorBefore.blockBytes
                  << " peak_allocation=" << allocatorPeak.allocationBytes
                  << " peak_blocks=" << allocatorPeak.blockBytes
                  << " after_allocation=" << allocatorAfter.allocationBytes
                  << " after_blocks=" << allocatorAfter.blockBytes << '\n';
        std::cout << "PROBE3 PASS device=" << context.properties().deviceName
                  << " allocations=200 pool_block_bytes=" << kPoolBlockSize
                  << " staging_ring_bytes=" << kStagingRingSize << " iterations=50"
                  << " ring_wraps=" << ringWraps
                  << " reused_device_addresses=" << reusedDeviceAddresses
                  << " host_visible_device_local="
                  << (hasHostVisibleDeviceLocalMemory(context) ? "yes" : "no")
                  << " memory_budget_extension=" << (context.memoryBudgetEnabled() ? "present" : "absent")
                  << " validation_layer=" << (context.validationEnabled() ? "enabled" : "unavailable")
                  << " validation_messages=" << context.validationMessages().size() << '\n';
    }

} // namespace

int main() {
    try {
        VulkanContext context;
        std::cout << "DEVICE name=" << context.properties().deviceName
                  << " api=" << VK_API_VERSION_MAJOR(context.properties().apiVersion) << '.'
                  << VK_API_VERSION_MINOR(context.properties().apiVersion) << '.'
                  << VK_API_VERSION_PATCH(context.properties().apiVersion)
                  << " subgroup=" << context.subgroupSize()
                  << " validation=" << (context.validationEnabled() ? "enabled" : "unavailable")
                  << " spirv_val=" << PROBE_SPIRV_VAL << " slangc=" << PROBE_SLANGC << '\n';
        const std::filesystem::path shaderDirectory(PROBE_SHADER_DIR);
        bool allPassed = true;
        const auto runProbe = [&context, &allPassed](std::string_view name, auto&& function) {
            try {
                function();
            } catch (const std::exception& error) {
                allPassed = false;
                std::cerr << name << " FAIL device=" << context.properties().deviceName
                          << " reason=" << error.what() << '\n';
            }
        };
        runProbe("PROBE1", [&] { probeAbi(context, shaderDirectory / "abi_probe.spv"); });
        runProbe("PROBE2", [&] { probeReduction(context, shaderDirectory / "reduction_probe.spv"); });
        runProbe("PROBE3", [&] { probeVma(context, shaderDirectory / "abi_probe.spv"); });
        if (!context.validationMessages().empty()) {
            for (const auto& message : context.validationMessages())
                std::cerr << "VALIDATION " << message << '\n';
            allPassed = false;
        }
        std::cout << "ALL PROBES " << (allPassed ? "PASS" : "FAIL")
                  << " device=" << context.properties().deviceName << '\n';
        return allPassed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ALL PROBES FAIL reason=" << error.what() << '\n';
        return 1;
    }
}
