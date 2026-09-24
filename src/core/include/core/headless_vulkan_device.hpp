/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/environment.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend.hpp"
#include "core/vulkan_device_selection.hpp"
#include "core/vulkan_helpers.hpp"
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
                                             bool* const shader_atomic_float,
                                             const bool viewer_shaders,
                                             std::string* const missing = nullptr) {
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

        const auto feature_check = check_vulkan_feature_requirements(
            physical, {.viewer_shaders = viewer_shaders});
        if (missing != nullptr)
            *missing = feature_check.missing;
        if (!feature_check.supported()) {
            return false;
        }

        *queue_family = *family;
        *shader_float16 = feature_check.shader_float16;
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
            if (create_vulkan_instance(application, {}, {}, nullptr, 0, &device.instance_) != VK_SUCCESS) {
                return std::nullopt;
            }

            const auto enumeration = enumerate_vulkan_physical_devices(device.instance_);
            if (enumeration.count_result != VK_SUCCESS ||
                enumeration.devices_result != VK_SUCCESS || enumeration.devices.empty()) {
                return std::nullopt;
            }
            const auto& physical_devices = enumeration.devices;
            const auto count = static_cast<uint32_t>(physical_devices.size());

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
                std::string missing;
                info.required_features = device_has_required_features(
                    candidate, &support.queue_family, &support.shader_float16,
                    &support.shader_atomic_float, push_descriptors, &missing);
                if (push_descriptors) {
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
            VulkanDeviceFeatureEnableChain required_features;
            auto& features = required_features.features;
            auto& features11 = required_features.features11;
            auto& features12 = required_features.features12;
            auto& features13 = required_features.features13;
            if (push_descriptors) {
                VkPhysicalDeviceVulkan13Features supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
                VkPhysicalDeviceFeatures2 query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                query.pNext = &supported;
                vkGetPhysicalDeviceFeatures2(physical, &query);
                features13.subgroupSizeControl = supported.subgroupSizeControl;
                features13.computeFullSubgroups = supported.computeFullSubgroups;
            }
            features13.pNext = shader_atomic_float ? &atomic_float : nullptr;
            features12.shaderFloat16 = shader_float16 ? VK_TRUE : VK_FALSE;
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
            features.features.sparseBinding = external_interop &&
                                              headless_sparse_binding_supported(physical, queue_family);

            const float priority = 1.0f;
            VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue_info.queueFamilyIndex = queue_family;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &priority;
            if (create_vulkan_device(physical, {queue_info}, extensions, &features,
                                     nullptr, &device.device_) != VK_SUCCESS) {
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
