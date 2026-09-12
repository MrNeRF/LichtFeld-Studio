/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
    using namespace lfs::core;

    bool has_device_extension(const VkPhysicalDevice physical, const char* const name) {
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

    bool vulkan_api_at_least_1_3(const uint32_t api_version) {
        return VK_API_VERSION_MAJOR(api_version) > 1 ||
               (VK_API_VERSION_MAJOR(api_version) == 1 &&
                VK_API_VERSION_MINOR(api_version) >= 3);
    }

    bool device_has_required_features(const VkPhysicalDevice physical,
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
        static std::optional<HeadlessAdoptedDevice> try_create() {
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

            uint32_t queue_family = 0;
            bool shader_float16 = false;
            bool shader_atomic_float = false;
            VkPhysicalDevice physical = VK_NULL_HANDLE;
            for (const VkPhysicalDevice candidate : physical_devices) {
                if (device_has_required_features(
                        candidate, &queue_family, &shader_float16, &shader_atomic_float)) {
                    physical = candidate;
                    break;
                }
            }
            if (physical == VK_NULL_HANDLE) {
                return std::nullopt;
            }

            VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
            atomic_float.shaderBufferFloat32AtomicAdd =
                shader_atomic_float ? VK_TRUE : VK_FALSE;
            VkPhysicalDeviceVulkan13Features features13{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
            features13.synchronization2 = VK_TRUE;
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
            features11.pNext = &features12;
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.features.shaderInt64 = VK_TRUE;
            features.features.shaderInt16 = VK_TRUE;
            features.pNext = &features11;

            std::vector<const char*> extensions;
            if (shader_atomic_float) {
                extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            }
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

    std::vector<float> pattern(const size_t count, const size_t salt) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = 0.05f + 0.95f * static_cast<float>(((i + salt) * 7919) % 1000) / 1000.0f;
        }
        return values;
    }

    std::vector<double> matmul_reference(const std::vector<float>& a,
                                         const std::vector<float>& b,
                                         const size_t n) {
        std::vector<double> c(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += static_cast<double>(a[i * n + k]) * static_cast<double>(b[k * n + j]);
                }
                c[i * n + j] = sum;
            }
        }
        return c;
    }

    float expected_sum(const std::vector<float>& values) {
        double sum = 0.0;
        for (const float value : values) {
            sum += static_cast<double>(value);
        }
        return static_cast<float>(sum);
    }

} // namespace

TEST(TensorVulkanAdoption, BackendRunsOnAnAdoptedDevice) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan tensor backend is unavailable";
    }
    ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
    {
        auto adopted = HeadlessAdoptedDevice::try_create();
        if (!adopted.has_value()) {
            GTEST_SKIP() << "no Vulkan 1.3 device with the tensor backend features";
        }
        const VulkanDeviceHandles handles = adopted->handles();
        const auto status = adopt_vulkan_device(handles);
        ASSERT_TRUE(status) << lfs::format_for_developer(status.error());
        EXPECT_TRUE(vulkan_backend_adopted());

        const std::vector<float> values = pattern(4096, 1);
        const std::vector<float> left = pattern(64 * 64, 2);
        const std::vector<float> right = pattern(64 * 64, 3);
        std::vector<float> sorted_expected = values;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        const std::vector<double> matmul_expected = matmul_reference(left, right, 64);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor uploaded = Tensor::from_vector(values, {values.size()}, Device::GPU);
            EXPECT_NEAR(uploaded.sum_scalar(), expected_sum(values), 1e-3f);

            const Tensor product =
                Tensor::from_vector(left, {64, 64}, Device::GPU)
                    .matmul(Tensor::from_vector(right, {64, 64}, Device::GPU));
            const std::vector<float> product_values = product.cpu().to_vector();
            ASSERT_EQ(product_values.size(), matmul_expected.size());
            for (size_t i = 0; i < product_values.size(); ++i) {
                EXPECT_NEAR(product_values[i], matmul_expected[i], 1e-3) << "index=" << i;
            }

            const auto [sorted, order] =
                Tensor::from_vector(values, {values.size()}, Device::GPU).sort(0, false);
            (void)order;
            EXPECT_EQ(sorted.cpu().to_vector(), sorted_expected);
        }

        const auto second = adopt_vulkan_device(handles);
        EXPECT_FALSE(second);
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        EXPECT_FALSE(vulkan_backend_adopted());
    }

    GpuBackendScope scope(GpuBackend::Vulkan);
    const Tensor recovered = Tensor::from_vector(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
                                                 {4}, Device::GPU);
    EXPECT_FLOAT_EQ(recovered.sum_scalar(), 10.0f);
}

TEST(TensorVulkanAdoption, RejectsIncompleteHandlesAndStaysUsable) {
    if (!gpu_backend_available(GpuBackend::Vulkan)) {
        GTEST_SKIP() << "Vulkan tensor backend is unavailable";
    }
    ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
    VulkanDeviceHandles handles{};
    const auto status = adopt_vulkan_device(handles);
    EXPECT_FALSE(status);
    EXPECT_FALSE(vulkan_backend_adopted());

    GpuBackendScope scope(GpuBackend::Vulkan);
    const Tensor tensor = Tensor::from_vector(std::vector<float>{5.0f, 7.0f}, {2}, Device::GPU);
    EXPECT_FLOAT_EQ(tensor.sum_scalar(), 12.0f);
}
