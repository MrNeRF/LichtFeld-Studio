/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/environment.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include "core/vulkan_device_selection.hpp"
#include "core/vulkan_shader_requirements.hpp"
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::core {
    inline bool headless_sparse_binding_supported(VkPhysicalDevice physical, uint32_t family) {
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(physical, &features);
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        return features.sparseBinding && family < count &&
               (queues[family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
    }

    inline bool has_device_extension(const VkPhysicalDevice physical, const char* const name) {
        uint32_t count = 0;
        if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr) !=
            VK_SUCCESS) {
            return false;
        }
        std::vector<VkExtensionProperties> extensions(count);
        if (vkEnumerateDeviceExtensionProperties(
                physical, nullptr, &count, extensions.data()) != VK_SUCCESS) {
            return false;
        }
        for (const VkExtensionProperties& extension : extensions) {
            if (std::strcmp(extension.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    }

    inline bool vulkan_api_at_least_1_3(const uint32_t api_version) {
        return VK_API_VERSION_MAJOR(api_version) > 1 ||
               (VK_API_VERSION_MAJOR(api_version) == 1 &&
                VK_API_VERSION_MINOR(api_version) >= 3);
    }

    inline bool device_has_required_features(const VkPhysicalDevice physical,
                                             uint32_t* const queue_family,
                                             bool* const shader_float16,
                                             bool* const shader_atomic_float) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physical, &properties);
        if (!vulkan_api_at_least_1_3(properties.apiVersion)) {
            return false;
        }

        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, queues.data());
        std::optional<uint32_t> family;
        for (uint32_t index = 0; index < queue_count; ++index) {
            if ((queues[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                family = index;
                break;
            }
        }
        if (!family.has_value()) {
            return false;
        }

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
        vkGetPhysicalDeviceFeatures2(physical, &features);

        VkPhysicalDeviceFloatControlsProperties float_controls{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
        VkPhysicalDeviceSubgroupProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 properties2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties2.pNext = &subgroup;
        subgroup.pNext = &float_controls;
        vkGetPhysicalDeviceProperties2(physical, &properties2);
        const VkSubgroupFeatureFlags subgroup_required =
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
            VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        const bool subgroup_supported =
            (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
            (subgroup.supportedOperations & subgroup_required) == subgroup_required;
        if (!features.features.shaderInt64 || !features.features.shaderInt16 ||
            !features11.storageBuffer16BitAccess || !features12.storageBuffer8BitAccess ||
            !features12.timelineSemaphore || !features12.bufferDeviceAddress ||
            !features13.synchronization2 ||
            !float_controls.shaderSignedZeroInfNanPreserveFloat32 || !subgroup_supported) {
            return false;
        }

        *queue_family = *family;
        *shader_float16 = features12.shaderFloat16 == VK_TRUE;
        *shader_atomic_float = false;
        if (has_device_extension(physical, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)) {
            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 atomic_query{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            atomic_query.pNext = &atomic_float;
            vkGetPhysicalDeviceFeatures2(physical, &atomic_query);
            *shader_atomic_float = atomic_float.shaderBufferFloat32AtomicAdd == VK_TRUE;
        }
        return true;
    }

    class HeadlessAdoptedDevice {
    public:
        static std::optional<HeadlessAdoptedDevice> try_create(bool external_interop = false, bool push_descriptors = false,
                                                               std::string_view requested_device = {},
                                                               const std::optional<VulkanDeviceUuid>& cuda_uuid = std::nullopt) {
            HeadlessAdoptedDevice device;
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.pApplicationName = "LichtFeld Tensor Vulkan Adoption";
            application.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
            application.pEngineName = "LichtFeld";
            application.apiVersion = VK_API_VERSION_1_3;
            VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            instance_info.pApplicationInfo = &application;
            if (vkCreateInstance(&instance_info, nullptr, &device.instance_) != VK_SUCCESS) {
                return std::nullopt;
            }

            uint32_t count = 0;
            if (vkEnumeratePhysicalDevices(device.instance_, &count, nullptr) != VK_SUCCESS ||
                count == 0) {
                return std::nullopt;
            }
            std::vector<VkPhysicalDevice> physical_devices(count);
            if (vkEnumeratePhysicalDevices(
                    device.instance_, &count, physical_devices.data()) != VK_SUCCESS) {
                return std::nullopt;
            }

            std::vector<const char*> extensions;
            if (push_descriptors)
                extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
            if (external_interop) {
#ifdef _WIN32
                extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
                extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
                extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
                extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
            }
            struct Features {
                uint32_t queue_family = 0;
                bool shader_float16 = false;
                bool shader_atomic_float = false;
            };
            std::vector<Features> features_by_device(count);
            std::vector<VulkanDeviceCandidate> candidates(count);
            for (std::size_t index = 0; index < count; ++index) {
                const auto candidate = physical_devices[index];
                auto& support = features_by_device[index];
                auto& info = candidates[index];
                info.required_features = device_has_required_features(
                    candidate, &support.queue_family, &support.shader_float16, &support.shader_atomic_float);
                if (push_descriptors) {
                    const auto missing = missing_viewer_shader_features(candidate);
                    if (!missing.empty())
                        std::fprintf(stderr, "Skipping headless Vulkan device %zu: viewer shaders require %s\n", index, missing.c_str());
                    info.required_features &= missing.empty();
                }
                info.required_extensions = true;
                for (const char* extension : extensions)
                    info.required_extensions &= has_device_extension(candidate, extension);
                VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                properties.pNext = &id;
                vkGetPhysicalDeviceProperties2(candidate, &properties);
                std::memcpy(info.uuid.data(), id.deviceUUID, info.uuid.size());
                info.discrete = properties.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
            }
            const auto selected = select_headless_vulkan_device(candidates, requested_device, cuda_uuid);
            if (!selected)
                return std::nullopt;
            const auto physical = physical_devices[*selected];
            const auto [queue_family, shader_float16, shader_atomic_float] = features_by_device[*selected];
            if (shader_atomic_float)
                extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

            // Same predicate extension the windowed viewer enables. Without it an
            // off-screen export has to read the instance count back to the CPU
            // before it can bound the depth waves.
            bool conditional_rendering = false;
            if (has_device_extension(physical, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
                VkPhysicalDeviceConditionalRenderingFeaturesEXT supported{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                query.pNext = &supported;
                vkGetPhysicalDeviceFeatures2(physical, &query);
                const auto cuda = gpu_backend_device_info(GpuBackend::CUDA, 0);
                const bool pre_volta = cuda && cuda->compute_capability_major > 0 &&
                                       cuda->compute_capability_major < 7;
                if (supported.conditionalRendering == VK_TRUE &&
                    !environment::flag("LFS_VK_DISABLE_CONDITIONAL_RENDERING", pre_volta)) {
                    conditional_rendering = true;
                    extensions.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
                }
            }

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
            atomic_float.shaderBufferFloat32AtomicAdd =
                shader_atomic_float ? VK_TRUE : VK_FALSE;
            VkPhysicalDeviceVulkan13Features features13{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            features13.synchronization2 = VK_TRUE;
            if (push_descriptors) {
                VkPhysicalDeviceVulkan13Features supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                query.pNext = &supported;
                vkGetPhysicalDeviceFeatures2(physical, &query);
                features13.subgroupSizeControl = supported.subgroupSizeControl;
                features13.computeFullSubgroups = supported.computeFullSubgroups;
            }
            features13.pNext = shader_atomic_float ? &atomic_float : nullptr;
            VkPhysicalDeviceVulkan12Features features12{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            features12.storageBuffer8BitAccess = VK_TRUE;
            features12.timelineSemaphore = VK_TRUE;
            features12.bufferDeviceAddress = VK_TRUE;
            features12.shaderFloat16 = shader_float16 ? VK_TRUE : VK_FALSE;
            features12.pNext = &features13;
            VkPhysicalDeviceVulkan11Features features11{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
            features11.storageBuffer16BitAccess = VK_TRUE;
            features11.uniformAndStorageBuffer16BitAccess = push_descriptors ? VK_TRUE : VK_FALSE;
            VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional_features{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
            conditional_features.conditionalRendering = conditional_rendering ? VK_TRUE : VK_FALSE;
            if (conditional_rendering) {
                conditional_features.pNext = &features12;
                features11.pNext = &conditional_features;
            } else {
                features11.pNext = &features12;
            }
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.features.shaderInt64 = VK_TRUE;
            features.features.shaderInt16 = VK_TRUE;
            features.features.sparseBinding = external_interop &&
                                              headless_sparse_binding_supported(physical, queue_family);
            features.pNext = &features11;

            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue_info.queueFamilyIndex = queue_family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            VkDeviceCreateInfo create_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
            create_info.pNext = &features;
            create_info.queueCreateInfoCount = 1;
            create_info.pQueueCreateInfos = &queue_info;
            create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            create_info.ppEnabledExtensionNames =
                extensions.empty() ? nullptr : extensions.data();
            if (vkCreateDevice(physical, &create_info, nullptr, &device.device_) !=
                VK_SUCCESS) {
                return std::nullopt;
            }
            vkGetDeviceQueue(device.device_, queue_family, 0, &device.queue_);
            device.physical_device_ = physical;
            device.queue_family_ = queue_family;
            device.shader_atomic_float_ = shader_atomic_float;
            device.shader_float16_ = shader_float16;
            return device;
        }

        HeadlessAdoptedDevice(HeadlessAdoptedDevice&& other) noexcept {
            instance_ = other.instance_;
            physical_device_ = other.physical_device_;
            device_ = other.device_;
            queue_ = other.queue_;
            queue_family_ = other.queue_family_;
            shader_atomic_float_ = other.shader_atomic_float_;
            shader_float16_ = other.shader_float16_;
            other.instance_ = VK_NULL_HANDLE;
            other.physical_device_ = VK_NULL_HANDLE;
            other.device_ = VK_NULL_HANDLE;
            other.queue_ = VK_NULL_HANDLE;
        }

        HeadlessAdoptedDevice(const HeadlessAdoptedDevice&) = delete;
        HeadlessAdoptedDevice& operator=(const HeadlessAdoptedDevice&) = delete;
        HeadlessAdoptedDevice& operator=(HeadlessAdoptedDevice&&) = delete;

        ~HeadlessAdoptedDevice() {
            if (device_ != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(device_);
                vkDestroyDevice(device_, nullptr);
            }
            if (instance_ != VK_NULL_HANDLE) {
                vkDestroyInstance(instance_, nullptr);
            }
        }

        [[nodiscard]] VulkanDeviceHandles handles() const {
            return VulkanDeviceHandles{
                .instance = instance_,
                .physical_device = physical_device_,
                .device = device_,
                .queue = queue_,
                .queue_family = queue_family_,
                .shader_atomic_float = shader_atomic_float_,
                .memory_budget = false,
                .shader_float16 = shader_float16_,
            };
        }

        VulkanDeviceHandles release() {
            const auto result = handles();
            instance_ = VK_NULL_HANDLE;
            device_ = VK_NULL_HANDLE;
            return result;
        }

    private:
        HeadlessAdoptedDevice() = default;

        VkInstance instance_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue queue_ = VK_NULL_HANDLE;
        uint32_t queue_family_ = 0;
        bool shader_atomic_float_ = false;
        bool shader_float16_ = false;
    };

} // namespace lfs::core
