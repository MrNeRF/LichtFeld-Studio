/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_context.hpp"

#include "core/assert.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/user_paths.hpp"
#include "vk_memory.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>

namespace lfs::core::internal {
    namespace {
        // Test hook: pretend the device has no float atomics so the deterministic
        // index_add fallback is exercised on hardware that has the extension.
        bool force_no_atomic_float() {
            const char* const value = std::getenv("LFS_VULKAN_FORCE_NO_ATOMIC_FLOAT");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }

        constexpr std::string_view kValidationLayer = "VK_LAYER_KHRONOS_validation";
        constexpr std::array<std::string_view, 2> kValidationAllowlist{
            "loader_add_layer_properties",
            "linux_read_sorted_physical_devices",
        };

        struct ContextSlot {
            std::once_flag once;
            std::shared_ptr<VulkanContext> context;
            std::exception_ptr failure;
        };

        std::mutex g_context_mutex;
        std::shared_ptr<ContextSlot> g_context_slot = std::make_shared<ContextSlot>();
        std::mutex g_validation_mutex;
        std::vector<std::string> g_validation_messages;
        std::atomic<uint64_t> g_next_context_id{1};

        bool environment_flag(const char* const name) {
            const char* const value = std::getenv(name);
            return value != nullptr &&
                   (std::string_view(value) == "1" ||
                    std::string_view(value) == "true" ||
                    std::string_view(value) == "TRUE");
        }

        bool has_layer(const std::string_view name) {
            uint32_t count = 0;
            if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) {
                return false;
            }
            std::vector<VkLayerProperties> layers(count);
            if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS) {
                return false;
            }
            return std::ranges::any_of(layers, [name](const VkLayerProperties& layer) {
                return name == layer.layerName;
            });
        }

        bool has_instance_extension(const std::string_view name) {
            uint32_t count = 0;
            if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS) {
                return false;
            }
            std::vector<VkExtensionProperties> extensions(count);
            if (vkEnumerateInstanceExtensionProperties(
                    nullptr, &count, extensions.data()) != VK_SUCCESS) {
                return false;
            }
            return std::ranges::any_of(
                extensions, [name](const VkExtensionProperties& extension) {
                    return name == extension.extensionName;
                });
        }

        bool allowlisted_validation_message(const std::string_view message) {
            return std::ranges::any_of(kValidationAllowlist,
                                       [message](const std::string_view fragment) {
                                           return message.contains(fragment);
                                       });
        }

        VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(
            const VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT,
            const VkDebugUtilsMessengerCallbackDataEXT* const callback,
            void*) {
            if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
                return VK_FALSE;
            }
            const std::string message =
                callback != nullptr && callback->pMessage != nullptr
                    ? callback->pMessage
                    : "empty Vulkan validation message";
            if (!allowlisted_validation_message(message)) {
                std::lock_guard lock(g_validation_mutex);
                g_validation_messages.push_back(message);
            }
            return VK_FALSE;
        }

        std::string uuid_string(const std::array<uint8_t, VK_UUID_SIZE>& uuid) {
            std::ostringstream stream;
            stream << std::hex << std::setfill('0');
            for (const uint8_t byte : uuid) {
                stream << std::setw(2) << static_cast<unsigned>(byte);
            }
            return stream.str();
        }

        std::string normalized_uuid(std::string value) {
            std::erase_if(value, [](const char character) {
                return character == '-' || character == '{' || character == '}';
            });
            std::ranges::transform(value, value.begin(), [](const unsigned char character) {
                return static_cast<char>(std::tolower(character));
            });
            return value;
        }

        std::vector<char> read_binary_file(const std::filesystem::path& path) {
            std::ifstream stream(path, std::ios::binary | std::ios::ate);
            if (!stream) {
                return {};
            }
            const std::streamoff size = stream.tellg();
            if (size <= 0) {
                return {};
            }
            std::vector<char> data(static_cast<size_t>(size));
            stream.seekg(0);
            if (!stream.read(data.data(), size)) {
                return {};
            }
            return data;
        }

        bool valid_pipeline_cache(const std::vector<char>& data,
                                  const VkPhysicalDeviceProperties& properties) {
            if (data.size() < sizeof(VkPipelineCacheHeaderVersionOne)) {
                return false;
            }
            VkPipelineCacheHeaderVersionOne header{};
            std::memcpy(&header, data.data(), sizeof(header));
            return header.headerVersion == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
                   header.vendorID == properties.vendorID &&
                   header.deviceID == properties.deviceID &&
                   std::memcmp(header.pipelineCacheUUID,
                               properties.pipelineCacheUUID, VK_UUID_SIZE) == 0;
        }

        std::optional<uint32_t> parse_device_index(const std::string_view value) {
            uint32_t result = 0;
            const auto [end, error] = std::from_chars(
                value.data(), value.data() + value.size(), result);
            if (error == std::errc{} && end == value.data() + value.size()) {
                return result;
            }
            return std::nullopt;
        }

        bool device_has_required_features(VkPhysicalDevice device,
                                          VkDeviceCaps* caps = nullptr) {
            VkPhysicalDeviceVulkan13Features features13{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceVulkan11Features features11{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &features11;
            features11.pNext = &features12;
            features12.pNext = &features13;
            vkGetPhysicalDeviceFeatures2(device, &features);

            VkPhysicalDeviceSubgroupProperties subgroup{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &subgroup;
            subgroup.pNext = &ids;
            vkGetPhysicalDeviceProperties2(device, &properties);
            const VkSubgroupFeatureFlags subgroup_required =
                VK_SUBGROUP_FEATURE_BASIC_BIT |
                VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
                VK_SUBGROUP_FEATURE_BALLOT_BIT |
                VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            const bool subgroup_supported =
                (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
                (subgroup.supportedOperations & subgroup_required) == subgroup_required;
            const bool available =
                VK_API_VERSION_MAJOR(properties.properties.apiVersion) > 1 ||
                (VK_API_VERSION_MAJOR(properties.properties.apiVersion) == 1 &&
                 VK_API_VERSION_MINOR(properties.properties.apiVersion) >= 3);
            const bool required =
                available && features.features.shaderInt64 &&
                features.features.shaderInt16 && features11.storageBuffer16BitAccess &&
                features12.storageBuffer8BitAccess && features12.timelineSemaphore &&
                features12.bufferDeviceAddress && features13.synchronization2 &&
                subgroup_supported;
            if (required && caps != nullptr) {
                std::copy_n(ids.deviceUUID, VK_UUID_SIZE, caps->device_uuid.begin());
                std::copy_n(ids.driverUUID, VK_UUID_SIZE, caps->driver_uuid.begin());
                caps->subgroup_size = subgroup.subgroupSize;
                caps->max_workgroup_invocations =
                    properties.properties.limits.maxComputeWorkGroupInvocations;
                std::copy_n(properties.properties.limits.maxComputeWorkGroupSize, 3,
                            caps->max_workgroup_size.begin());
                std::copy_n(properties.properties.limits.maxComputeWorkGroupCount, 3,
                            caps->max_workgroup_count.begin());
                caps->shared_memory_size =
                    properties.properties.limits.maxComputeSharedMemorySize;
                caps->timestamp_period = properties.properties.limits.timestampPeriod;
                caps->shader_float16 = features12.shaderFloat16;
            }
            return required;
        }

        bool has_compute_queue(VkPhysicalDevice device) {
            uint32_t count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
            std::vector<VkQueueFamilyProperties> queues(count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &count, queues.data());
            return std::ranges::any_of(queues, [](const VkQueueFamilyProperties& queue) {
                return (queue.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            });
        }
    } // namespace

    void vk_check(VulkanContext* const context, const VkResult result,
                  const char* const operation) {
        if (result == VK_SUCCESS) {
            return;
        }
        if (result == VK_ERROR_DEVICE_LOST && context != nullptr) {
            context->mark_device_lost_once();
        }
        ErrorCode code = ErrorCode::Internal;
        if (result == VK_ERROR_DEVICE_LOST) {
            code = ErrorCode::DeviceLost;
        } else if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                   result == VK_ERROR_OUT_OF_HOST_MEMORY ||
                   result == VK_ERROR_OUT_OF_POOL_MEMORY ||
                   result == VK_ERROR_FRAGMENTED_POOL) {
            code = ErrorCode::ResourceExhausted;
        }
        throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = std::format("Vulkan call failed: {}", operation),
            .detail = std::format("{} returned VkResult {}", operation,
                                  static_cast<int>(result)),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .native = lfs::NativeError{
                .domain = lfs::ErrorDomain::Vulkan,
                .code = static_cast<int64_t>(result),
                .name = std::format("VkResult({})", static_cast<int>(result)),
            },
        }));
    }

    VulkanContext::VulkanContext()
        : context_id_(g_next_context_id.fetch_add(1, std::memory_order_relaxed)) {
        try {
            {
                std::lock_guard lock(g_validation_mutex);
                g_validation_messages.clear();
            }
            create_instance();
            select_physical_device();
            create_device();
            create_allocator();

            VkSemaphoreTypeCreateInfo type_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo semaphore_info{
                VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            semaphore_info.pNext = &type_info;
            vk_check(this,
                     vkCreateSemaphore(device_, &semaphore_info, nullptr, &timeline_),
                     "vkCreateSemaphore(timeline)");
            create_pipeline_cache();
            recorders_ = std::make_unique<VulkanRecorderRegistry>(*this);
            memory_ = std::make_unique<VulkanMemory>(*this);
            pipelines_ = std::make_unique<VulkanPipelines>(*this);
            create_fault_buffer();
            LOG_INFO("Tensor Vulkan backend initialized: {} (device {}, UUID {}, subgroup {}, "
                     "host upload path={})",
                     properties_.deviceName, device_index_,
                     uuid_string(caps_.device_uuid), caps_.subgroup_size,
                     caps_.direct_host_uploads ? "direct" : "staging-ring");
        } catch (...) {
            try {
                shutdown();
            } catch (...) {
            }
            throw;
        }
    }

    VulkanContext::~VulkanContext() {
        try {
            shutdown();
        } catch (...) {
        }
        pipelines_.reset();
        memory_.reset();
        recorders_.reset();
    }

    void VulkanContext::create_instance() {
        const bool validation_requested = environment_flag("LFS_VULKAN_VALIDATION");
        const bool validation_available = has_layer(kValidationLayer);
        const bool debug_utils = has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::vector<const char*> layers;
        std::vector<const char*> extensions;
        if (validation_requested && validation_available) {
            layers.push_back(kValidationLayer.data());
            if (debug_utils) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }
        }
        VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        application.pApplicationName = "LichtFeld Tensor Vulkan";
        application.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        application.pEngineName = "LichtFeld";
        application.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo create_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create_info.pApplicationInfo = &application;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
        vk_check(this, vkCreateInstance(&create_info, nullptr, &instance_),
                 "vkCreateInstance");
        if (!extensions.empty()) {
            VkDebugUtilsMessengerCreateInfoEXT debug_info{
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
            debug_info.messageSeverity =
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug_info.pfnUserCallback = validation_callback;
            const auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
            if (create != nullptr) {
                vk_check(this, create(instance_, &debug_info, nullptr, &debug_messenger_),
                         "vkCreateDebugUtilsMessengerEXT");
            }
        }
    }

    void VulkanContext::select_physical_device() {
        uint32_t count = 0;
        vk_check(this, vkEnumeratePhysicalDevices(instance_, &count, nullptr),
                 "vkEnumeratePhysicalDevices(count)");
        LFS_ASSERT_MSG(count != 0, "Vulkan backend: no physical devices available");
        std::vector<VkPhysicalDevice> devices(count);
        vk_check(this, vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
                 "vkEnumeratePhysicalDevices(data)");

        std::optional<uint32_t> selected;
        if (const char* const override_value = std::getenv("LFS_VULKAN_DEVICE")) {
            const std::string_view value(override_value);
            if (const auto index = parse_device_index(value)) {
                LFS_ASSERT_MSG(*index < devices.size(),
                               "LFS_VULKAN_DEVICE index is out of range");
                selected = *index;
            } else {
                const std::string requested = normalized_uuid(std::string(value));
                for (uint32_t index = 0; index < devices.size(); ++index) {
                    VkDeviceCaps candidate_caps{};
                    if (device_has_required_features(devices[index], &candidate_caps) &&
                        uuid_string(candidate_caps.device_uuid) == requested) {
                        selected = index;
                        break;
                    }
                }
                LFS_ASSERT_MSG(selected.has_value(),
                               "LFS_VULKAN_DEVICE UUID did not match a Vulkan device");
            }
        }
        if (!selected) {
            for (uint32_t index = 0; index < devices.size(); ++index) {
                VkPhysicalDeviceProperties candidate{};
                vkGetPhysicalDeviceProperties(devices[index], &candidate);
                if (candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
                    device_has_required_features(devices[index]) &&
                    has_compute_queue(devices[index])) {
                    selected = index;
                    break;
                }
            }
        }
        if (!selected) {
            for (uint32_t index = 0; index < devices.size(); ++index) {
                if (device_has_required_features(devices[index]) &&
                    has_compute_queue(devices[index])) {
                    selected = index;
                    break;
                }
            }
        }
        LFS_ASSERT_MSG(selected.has_value(),
                       "Vulkan backend: no device meets the required feature set");
        physical_device_ = devices[*selected];
        device_index_ = *selected;
        caps_.device_index = device_index_;
        vkGetPhysicalDeviceProperties(physical_device_, &properties_);
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);
        LFS_ASSERT_MSG(device_has_required_features(physical_device_, &caps_),
                       "Selected Vulkan device no longer reports required features");

        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_count,
                                                 queues.data());
        std::optional<uint32_t> fallback;
        for (uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
                continue;
            }
            if (!fallback) {
                fallback = index;
            }
            if ((queues[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
                queue_family_ = index;
                return;
            }
        }
        LFS_ASSERT_MSG(fallback.has_value(),
                       "Vulkan backend: selected device has no compute queue");
        queue_family_ = *fallback;
    }

    void VulkanContext::create_device() {
        uint32_t extension_count = 0;
        vk_check(this, vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, nullptr),
                 "vkEnumerateDeviceExtensionProperties(count)");
        std::vector<VkExtensionProperties> extension_properties(extension_count);
        vk_check(this, vkEnumerateDeviceExtensionProperties(physical_device_, nullptr, &extension_count, extension_properties.data()),
                 "vkEnumerateDeviceExtensionProperties(data)");
        std::unordered_set<std::string> extensions_available;
        for (const auto& extension : extension_properties) {
            extensions_available.emplace(extension.extensionName);
        }
        caps_.memory_budget =
            extensions_available.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        VkPhysicalDeviceVulkan13Features query13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features query12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features query11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        query.pNext = &query11;
        query11.pNext = &query12;
        query12.pNext = &query13;
        query13.pNext = &atomic_float;
        vkGetPhysicalDeviceFeatures2(physical_device_, &query);
        caps_.shader_float16 = query12.shaderFloat16;
        caps_.shader_atomic_float =
            extensions_available.contains(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) &&
            atomic_float.shaderBufferFloat32AtomicAdd &&
            !force_no_atomic_float();
        caps_.host_visible_device_local = false;
        for (uint32_t index = 0; index < memory_properties_.memoryTypeCount; ++index) {
            const VkMemoryPropertyFlags flags =
                memory_properties_.memoryTypes[index].propertyFlags;
            caps_.host_visible_device_local |=
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
                (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        }
        caps_.direct_host_uploads = false;

        VkPhysicalDeviceVulkan13Features features13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan12Features features12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan11Features features11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features.features.shaderInt64 = VK_TRUE;
        features.features.shaderInt16 = VK_TRUE;
        features.pNext = &features11;
        features11.storageBuffer16BitAccess = VK_TRUE;
        features11.pNext = &features12;
        features12.storageBuffer8BitAccess = VK_TRUE;
        features12.timelineSemaphore = VK_TRUE;
        features12.bufferDeviceAddress = VK_TRUE;
        features12.shaderFloat16 = caps_.shader_float16;
        features12.pNext = &features13;
        features13.synchronization2 = VK_TRUE;
        features13.pNext = caps_.shader_atomic_float ? &atomic_float : nullptr;
        atomic_float = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
        atomic_float.shaderBufferFloat32AtomicAdd = caps_.shader_atomic_float;

        std::vector<const char*> enabled_extensions;
        if (caps_.memory_budget) {
            enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
        if (caps_.shader_atomic_float) {
            enabled_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family_;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo create_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        create_info.pNext = &features;
        create_info.queueCreateInfoCount = 1;
        create_info.pQueueCreateInfos = &queue_info;
        create_info.enabledExtensionCount =
            static_cast<uint32_t>(enabled_extensions.size());
        create_info.ppEnabledExtensionNames = enabled_extensions.data();
        vk_check(this, vkCreateDevice(physical_device_, &create_info, nullptr, &device_),
                 "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    }

    void VulkanContext::create_allocator() {
        VmaAllocatorCreateInfo create_info{};
        create_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        if (caps_.memory_budget) {
            create_info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        }
        create_info.instance = instance_;
        create_info.physicalDevice = physical_device_;
        create_info.device = device_;
        create_info.vulkanApiVersion = VK_API_VERSION_1_3;
        vk_check(this, vmaCreateAllocator(&create_info, &allocator_),
                 "vmaCreateAllocator");
    }

    void VulkanContext::create_pipeline_cache() {
        std::vector<char> initial_data;
        if (const auto paths = UserPaths::resolve()) {
            const std::string feature_key = std::format(
                "f16{}_atomic{}_budget{}", caps_.shader_float16,
                caps_.shader_atomic_float, caps_.memory_budget);
            const std::filesystem::path directory =
                paths->cacheDir() / "tensor_vulkan";
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            pipeline_cache_path_ =
                (directory / std::format("{}_{}_{}.bin",
                                         uuid_string(caps_.device_uuid),
                                         uuid_string(caps_.driver_uuid), feature_key))
                    .string();
            initial_data = read_binary_file(pipeline_cache_path_);
            if (!valid_pipeline_cache(initial_data, properties_)) {
                initial_data.clear();
            }
        }
        VkPipelineCacheCreateInfo create_info{
            VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        create_info.initialDataSize = initial_data.size();
        create_info.pInitialData = initial_data.data();
        vk_check(this, vkCreatePipelineCache(device_, &create_info, nullptr, &pipeline_cache_),
                 "vkCreatePipelineCache");
    }

    void VulkanContext::save_pipeline_cache() noexcept {
        try {
            if (pipeline_cache_path_.empty() || pipeline_cache_ == VK_NULL_HANDLE) {
                return;
            }
            size_t size = 0;
            if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr) !=
                    VK_SUCCESS ||
                size == 0) {
                return;
            }
            std::vector<char> data(size);
            if (vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data()) !=
                VK_SUCCESS) {
                return;
            }
            // A per-context temporary plus rename is atomic on its own; a lock
            // directory left behind by a crash would silence every later save.
            const std::filesystem::path destination(pipeline_cache_path_);
            std::error_code error;
            const std::filesystem::path temporary =
                destination.string() + std::format(".tmp.{}", context_id_);
            {
                std::ofstream stream(temporary,
                                     std::ios::binary | std::ios::trunc);
                stream.write(data.data(), static_cast<std::streamsize>(data.size()));
                if (!stream) {
                    std::filesystem::remove(temporary, error);
                    return;
                }
            }
            std::filesystem::rename(temporary, destination, error);
            if (error) {
                std::filesystem::remove(destination, error);
                error.clear();
                std::filesystem::rename(temporary, destination, error);
            }
        } catch (...) {
        }
    }

    void VulkanContext::create_fault_buffer() {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = 16;
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocation_info{};
        allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
        allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VmaAllocationInfo info{};
        vk_check(this, vmaCreateBuffer(allocator_, &buffer_info, &allocation_info, &fault_buffer_, &fault_allocation_, &info),
                 "vmaCreateBuffer(fault)");
        fault_mapped_ = info.pMappedData;
        std::memset(fault_mapped_, 0, 16);
        VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address_info.buffer = fault_buffer_;
        fault_address_ = vkGetBufferDeviceAddress(device_, &address_info);
        LFS_ASSERT_MSG(fault_address_ != 0,
                       "vkGetBufferDeviceAddress returned zero for the fault buffer");
    }

    void VulkanContext::destroy_fault_buffer() noexcept {
        if (fault_buffer_ != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, fault_buffer_, fault_allocation_);
            fault_buffer_ = VK_NULL_HANDLE;
            fault_allocation_ = VK_NULL_HANDLE;
            fault_mapped_ = nullptr;
            fault_address_ = 0;
        }
    }

    uint64_t VulkanContext::reserve_timeline_value() {
        LFS_ASSERT_MSG(accepting_work() && !dead(),
                       "Vulkan backend is not accepting new work");
        return next_timeline_.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    void VulkanContext::submit(const VkCommandBuffer command,
                               const uint64_t signal_value) {
        LFS_ASSERT_MSG(!dead(), "Vulkan backend device is lost");
        VkCommandBufferSubmitInfo command_info{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        command_info.commandBuffer = command;
        VkSemaphoreSubmitInfo signal_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signal_info.semaphore = timeline_;
        signal_info.value = signal_value;
        signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkSubmitInfo2 submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submit_info.commandBufferInfoCount = 1;
        submit_info.pCommandBufferInfos = &command_info;
        submit_info.signalSemaphoreInfoCount = 1;
        submit_info.pSignalSemaphoreInfos = &signal_info;
        std::lock_guard lock(queue_mutex_);
        vk_check(this, vkQueueSubmit2(queue_, 1, &submit_info, VK_NULL_HANDLE),
                 "vkQueueSubmit2");
    }

    void VulkanContext::wait(const uint64_t value) {
        if (value == 0) {
            return;
        }
        LFS_ASSERT_MSG(!dead(), "Vulkan backend device is lost");
        VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_;
        wait_info.pValues = &value;
        constexpr uint64_t kSliceNanoseconds = 2'000'000'000ull;
        for (uint32_t slice = 0;; ++slice) {
            const VkResult result = vkWaitSemaphores(device_, &wait_info, kSliceNanoseconds);
            if (result != VK_TIMEOUT) {
                vk_check(this, result, "vkWaitSemaphores");
                return;
            }
            LFS_ASSERT_MSG(!dead(), "Vulkan backend device is lost");
            if (slice == 14) {
                LOG_WARN("Vulkan timeline value {} not signalled after 30 s", value);
            }
            if (slice == 59) {
                mark_device_lost_once();
                LFS_ASSERT_MSG(false,
                               std::format("Vulkan timeline value {} not signalled after 120 s; "
                                           "treating the device as lost",
                                           value));
            }
        }
    }

    uint64_t VulkanContext::completed_timeline() const {
        if (timeline_ == VK_NULL_HANDLE || dead()) {
            return 0;
        }
        uint64_t value = 0;
        vk_check(const_cast<VulkanContext*>(this),
                 vkGetSemaphoreCounterValue(device_, timeline_, &value),
                 "vkGetSemaphoreCounterValue");
        return value;
    }

    // A recorded fault surfaces at the next synchronization as a Vulkan-domain
    // error; the record is cleared so later synchronizations run clean.
    void VulkanContext::check_fault_buffer() {
        const std::array<uint32_t, 4> record = consume_fault_record();
        if (record[0] == 0) {
            return;
        }
        throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
            .code = ErrorCode::BoundsViolation,
            .domain = lfs::ErrorDomain::Vulkan,
            .user_message = "A tensor index was out of range on the Vulkan backend",
            .detail = std::format("device fault code {}: index {} is outside the extent {} "
                                  "(operation {})",
                                  record[0], static_cast<int32_t>(record[1]), record[2],
                                  record[3]),
            .detection = LFS_SOURCE_SITE_CURRENT(),
        }));
    }

    std::array<uint32_t, 4> VulkanContext::consume_fault_record() noexcept {
        std::array<uint32_t, 4> record{};
        if (fault_mapped_ == nullptr) {
            return record;
        }
        std::memcpy(record.data(), fault_mapped_, sizeof(record));
        std::memset(fault_mapped_, 0, sizeof(record));
        return record;
    }

    void VulkanContext::mark_device_lost_once() {
        dead_.store(true, std::memory_order_release);
        if (!device_loss_reported_.exchange(true, std::memory_order_acq_rel)) {
            LOG_ERROR("Tensor Vulkan backend device lost");
        }
    }

    VulkanMemory& VulkanContext::memory() {
        LFS_ASSERT_MSG(memory_ != nullptr, "Vulkan memory service is unavailable");
        return *memory_;
    }

    VulkanRecorderRegistry& VulkanContext::recorders() {
        LFS_ASSERT_MSG(recorders_ != nullptr, "Vulkan recorder service is unavailable");
        return *recorders_;
    }

    VulkanPipelines& VulkanContext::pipelines() {
        LFS_ASSERT_MSG(pipelines_ != nullptr, "Vulkan pipeline service is unavailable");
        return *pipelines_;
    }

    void VulkanContext::shutdown() {
        std::lock_guard shutdown_lock(shutdown_mutex_);
        if (instance_ == VK_NULL_HANDLE) {
            return;
        }
        accepting_work_.store(false, std::memory_order_release);
        if (!dead() && recorders_) {
            try {
                recorders_->wait_all();
            } catch (...) {
                mark_device_lost_once();
            }
        }
        if (memory_) {
            memory_->shutdown();
        }
        if (pipelines_) {
            pipelines_->shutdown();
        }
        if (recorders_) {
            recorders_->shutdown();
        }
        destroy_fault_buffer();
        save_pipeline_cache();
        if (pipeline_cache_ != VK_NULL_HANDLE) {
            vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
            pipeline_cache_ = VK_NULL_HANDLE;
        }
        if (allocator_ != nullptr) {
            vmaDestroyAllocator(allocator_);
            allocator_ = nullptr;
        }
        if (timeline_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, timeline_, nullptr);
            timeline_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (debug_messenger_ != VK_NULL_HANDLE) {
            const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroy != nullptr) {
                destroy(instance_, debug_messenger_, nullptr);
            }
            debug_messenger_ = VK_NULL_HANDLE;
        }
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    bool vulkan_backend_probe_available() noexcept {
        if (const auto live = try_live_vulkan_context()) {
            return !live->dead();
        }
        static const bool available = [] {
            try {
                VulkanContext probe;
                probe.shutdown();
                return true;
            } catch (...) {
                return false;
            }
        }();
        return available;
    }

    std::shared_ptr<VulkanContext> acquire_vulkan_context() {
        std::shared_ptr<ContextSlot> slot;
        {
            std::lock_guard lock(g_context_mutex);
            slot = g_context_slot;
        }
        std::call_once(slot->once, [slot] {
            try {
                slot->context = std::make_shared<VulkanContext>();
            } catch (...) {
                slot->failure = std::current_exception();
            }
        });
        if (slot->failure) {
            std::rethrow_exception(slot->failure);
        }
        LFS_ASSERT_MSG(slot->context != nullptr,
                       "Vulkan backend initialization produced no context");
        return slot->context;
    }

    std::shared_ptr<VulkanContext> try_live_vulkan_context() noexcept {
        std::lock_guard lock(g_context_mutex);
        return g_context_slot->context;
    }

    void shutdown_vulkan_context() {
        std::shared_ptr<VulkanContext> context;
        {
            std::lock_guard lock(g_context_mutex);
            context = std::move(g_context_slot->context);
            g_context_slot = std::make_shared<ContextSlot>();
        }
        if (context) {
            context->shutdown();
        } else {
            // Nothing can be outstanding from a context that no longer exists; a
            // deleter that runs later is a no-op by the context-id check.
            VulkanMemory::reset_last_shutdown_live_allocations();
        }
    }

    std::vector<std::string> vulkan_validation_messages_for_testing() {
        std::lock_guard lock(g_validation_mutex);
        return g_validation_messages;
    }

    VkDeviceCaps vulkan_device_caps_for_testing() {
        return acquire_vulkan_context()->caps();
    }

    uint64_t vulkan_live_vma_objects_for_testing() noexcept {
        try {
            if (const auto context = try_live_vulkan_context()) {
                return context->memory().live_object_count() +
                       (context->allocator() != nullptr ? 1 : 0);
            }
        } catch (...) {
        }
        // No live context: report what the last shutdown found still allocated,
        // so a leaked tensor is detected instead of hidden by the slot replacement.
        return VulkanMemory::last_shutdown_live_allocations();
    }

    uint64_t vulkan_completed_timeline_for_testing() {
        return acquire_vulkan_context()->completed_timeline();
    }

    size_t vulkan_dead_recorder_count_for_testing() {
        return acquire_vulkan_context()->recorders().dead_recorder_count();
    }

} // namespace lfs::core::internal
