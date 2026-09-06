/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/gpu_backend_ops.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
    using namespace lfs::core;

    class TensorVulkanCudaBridge : public testing::Test {
    protected:
        void SetUp() override {
            ASSERT_TRUE(gpu_backend_available(GpuBackend::Vulkan));
        }

        void TearDown() override {
            const auto status = shutdown_gpu_backend(GpuBackend::Vulkan);
            EXPECT_TRUE(status.has_value());
            EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
            EXPECT_EQ(internal::vulkan_cuda_import_count_for_testing(), 0u);
            for (const std::string& message :
                 internal::vulkan_validation_messages_for_testing()) {
                ADD_FAILURE() << message;
            }
        }

        static void force_vulkan_context() {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({8}, Device::CUDA);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 8.0f);
        }
    };

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
            application.pApplicationName = "LichtFeld Tensor Vulkan CUDA Bridge";
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

            const bool external_memory =
                has_device_extension(physical, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            const bool external_semaphore =
                has_device_extension(physical, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
            if (!external_memory || !external_semaphore) {
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
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
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
            create_info.ppEnabledExtensionNames = extensions.data();
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
                .external_memory = true,
                .external_semaphore = true,
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

    TEST_F(TensorVulkanCudaBridge, ViewReadsPendingVulkanWrite) {
        force_vulkan_context();
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "Vulkan tensor backend is not exporting memory for CUDA";
        }

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        const size_t count = 8u * 1024u * 1024u;
        std::vector<float> host(count, 0.0f);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const uint64_t live_before = internal::vulkan_live_vma_objects_for_testing();
            const Tensor ones = Tensor::ones({count}, Device::CUDA);
            internal::backend_ops(GpuBackend::Vulkan).synchronize_device();
            const Tensor result = ones.mul(42.0f);
            const auto view = cuda_view_of_vulkan_tensor(result, stream);
            ASSERT_TRUE(view) << lfs::format_for_developer(view.error());
            EXPECT_EQ(gpu_backend_of(*view), GpuBackend::CUDA);
            EXPECT_EQ(view->shape(), result.shape());
            EXPECT_EQ(view->dtype(), result.dtype());
            EXPECT_TRUE(view->is_contiguous());
            ASSERT_EQ(cudaMemcpyAsync(host.data(), view->data_ptr(), result.bytes(),
                                      cudaMemcpyDeviceToHost, stream),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            for (size_t i = 0; i < count; i += count / 16) {
                EXPECT_FLOAT_EQ(host[i], 42.0f) << "index=" << i;
            }
            EXPECT_FLOAT_EQ(host.front(), 42.0f);
            EXPECT_FLOAT_EQ(host.back(), 42.0f);
            std::cout << "live_vma_before=" << live_before
                      << " live_vma_with_view=" << internal::vulkan_live_vma_objects_for_testing()
                      << " cuda_imports=" << internal::vulkan_cuda_import_count_for_testing()
                      << std::endl;
        }
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorVulkanCudaBridge, SliceViewHasTensorByteOffset) {
        force_vulkan_context();
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "Vulkan tensor backend is not exporting memory for CUDA";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::CUDA);
        const Tensor sliced = base.slice(0, 1, 2);
        const auto view = cuda_view_of_vulkan_tensor(sliced, stream);
        ASSERT_TRUE(view) << lfs::format_for_developer(view.error());
        std::vector<float> host(3, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(host.data(), view->data_ptr(), sliced.bytes(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(host, (std::vector<float>{4, 5, 6}));
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorVulkanCudaBridge, ViewKeepsVulkanStorageAlive) {
        force_vulkan_context();
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "Vulkan tensor backend is not exporting memory for CUDA";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({8}, Device::CUDA);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 8.0f);
        }
        Tensor::trim_memory_pool();
        const uint64_t baseline = internal::vulkan_live_vma_objects_for_testing();
        Tensor view;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            Tensor tensor = Tensor::ones({4096}, Device::CUDA);
            auto created = cuda_view_of_vulkan_tensor(tensor, stream);
            ASSERT_TRUE(created) << lfs::format_for_developer(created.error());
            view = *created;
            tensor = Tensor();
            Tensor::trim_memory_pool();
            EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        }
        Tensor::trim_memory_pool();
        EXPECT_GT(internal::vulkan_live_vma_objects_for_testing(), baseline);
        view = Tensor();
        Tensor::trim_memory_pool();
        EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), baseline);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorVulkanCudaBridge, CpuAndCudaSourcesReturnError) {
        force_vulkan_context();
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "Vulkan tensor backend is not exporting memory for CUDA";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        {
            GpuBackendScope cuda_scope(GpuBackend::CUDA);
            const Tensor cuda = Tensor::ones({8}, Device::CUDA);
            EXPECT_EQ(gpu_backend_of(cuda), GpuBackend::CUDA);
            const auto view = cuda_view_of_vulkan_tensor(cuda, stream);
            EXPECT_FALSE(view);
            EXPECT_EQ(view.error().code(), lfs::ErrorCode::InvalidArgument);
        }
        const Tensor cpu = Tensor::ones({8}, Device::CPU);
        const auto cpu_view = cuda_view_of_vulkan_tensor(cpu, stream);
        EXPECT_FALSE(cpu_view);
        EXPECT_EQ(cpu_view.error().code(), lfs::ErrorCode::InvalidArgument);

        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {2, 4}, Device::CUDA);
        const Tensor column = base.slice(1, 0, 1);
        ASSERT_FALSE(column.is_contiguous());
        const auto strided = cuda_view_of_vulkan_tensor(column, stream);
        EXPECT_FALSE(strided);
        EXPECT_EQ(strided.error().code(), lfs::ErrorCode::InvalidArgument);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorVulkanCudaBridge, ShutdownReleasesImportsWhileViewAlive) {
        force_vulkan_context();
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "Vulkan tensor backend is not exporting memory for CUDA";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        Tensor view;
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor tensor = Tensor::ones({4096}, Device::CUDA);
            auto created = cuda_view_of_vulkan_tensor(tensor, stream);
            ASSERT_TRUE(created) << lfs::format_for_developer(created.error());
            view = *created;
            EXPECT_GT(internal::vulkan_cuda_import_count_for_testing(), 0u);
        }
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan).has_value());
        EXPECT_EQ(internal::vulkan_cuda_import_count_for_testing(), 0u);
        view = Tensor();
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    TEST_F(TensorVulkanCudaBridge, AdoptedDeviceWithExternalMemoryTakesView) {
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        auto adopted = HeadlessAdoptedDevice::try_create();
        if (!adopted.has_value()) {
            GTEST_SKIP() << "no Vulkan 1.3 device with external-memory extensions";
        }
        const auto status = adopt_vulkan_device(adopted->handles());
        ASSERT_TRUE(status) << lfs::format_for_developer(status.error());
        EXPECT_TRUE(vulkan_backend_adopted());
        {
            GpuBackendScope scope(GpuBackend::Vulkan);
            const Tensor warm = Tensor::ones({8}, Device::CUDA);
            ASSERT_FLOAT_EQ(warm.sum_scalar(), 8.0f);
        }
        if (!vulkan_backend_exports_memory()) {
            GTEST_SKIP() << "adopted device did not export memory for CUDA";
        }
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        GpuBackendScope scope(GpuBackend::Vulkan);
        const Tensor source = Tensor::full({1024}, 3.5f, Device::CUDA);
        const Tensor result = source.mul(2.0f);
        const auto view = cuda_view_of_vulkan_tensor(result, stream);
        ASSERT_TRUE(view) << lfs::format_for_developer(view.error());
        std::vector<float> host(1024, 0.0f);
        ASSERT_EQ(cudaMemcpyAsync(host.data(), view->data_ptr(), result.bytes(),
                                  cudaMemcpyDeviceToHost, stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_FLOAT_EQ(host.front(), 7.0f);
        EXPECT_FLOAT_EQ(host.back(), 7.0f);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
        ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
        EXPECT_FALSE(vulkan_backend_adopted());
    }

} // namespace

TEST_F(TensorVulkanCudaBridge, DirectLargeAllocationsAreNotExportableButStayUsable) {
    if (!vulkan_backend_exports_memory()) {
        GTEST_SKIP() << "Vulkan tensor backend does not export memory";
    }
    GpuBackendScope scope(GpuBackend::Vulkan);
    constexpr size_t count = size_t{100} * 1024 * 1024 / sizeof(float);
    Tensor large = Tensor::ones({count}, Device::CUDA, DataType::Float32);
    EXPECT_FLOAT_EQ(large.sum_scalar(), static_cast<float>(count));
    const auto view = cuda_view_of_vulkan_tensor(large, nullptr);
    ASSERT_FALSE(view.has_value());
    EXPECT_EQ(view.error().code(), lfs::ErrorCode::Unsupported);
    EXPECT_FLOAT_EQ((large * 2.0f).sum_scalar(), 2.0f * static_cast<float>(count));
}
