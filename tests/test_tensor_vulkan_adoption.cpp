/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/headless_vulkan_device.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_splat.hpp"
#include <chrono>
#include <cstdio>
#include <set>
#ifdef __APPLE__
#include <vulkan/vulkan_metal.h>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace {
    using namespace lfs::core;

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
        EXPECT_GT(gpu_device_count(GpuBackend::Vulkan), 0);
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
            EXPECT_FALSE(handles.shader_float64);
            auto scales = Tensor::zeros({33, 3}, Device::GPU);
            auto rotations = Tensor::zeros({33, 4}, Device::GPU);
            affine_splat_geometry({{2, 0, 0, 0, 2, 0, 0, 0, 2}}, scales, rotations, scales, rotations);
            for (const float scale : scales.to_vector())
                EXPECT_NEAR(scale, std::log(2.0f), 1e-6f);
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

#ifdef __APPLE__
TEST(TensorVulkanAdoption, ExportableHostReadbacksShareBackingAcrossSizeSweep) {
    ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
    auto adopted = HeadlessAdoptedDevice::try_create(false, true);
    if (!adopted || !adopted->handles().metal_objects)
        GTEST_SKIP() << "A Vulkan device exporting Metal memory is required";
    ASSERT_TRUE(adopt_vulkan_device(adopted->handles()));
    {
        auto context = internal::acquire_vulkan_context();
        auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
            vkGetDeviceProcAddr(context->device(), "vkExportMetalObjectsEXT"));
        ASSERT_NE(export_objects, nullptr);
        for (const size_t bytes : {size_t{4}, size_t{256}, size_t{4096}, size_t{65536}, size_t{1} << 20}) {
            SCOPED_TRACE(bytes);
            struct Readbacks {
                internal::VulkanMemory& memory;
                std::vector<internal::StorageRef> storage;
                ~Readbacks() {
                    for (auto ref : storage)
                        memory.deallocate(ref);
                }
            } readbacks{context->memory(), {}};
            std::set<VkDeviceMemory> backing;
            const auto started = std::chrono::steady_clock::now();
            for (size_t index = 0; index < 128; ++index) {
                const auto storage = context->memory().allocate_readback(bytes);
                readbacks.storage.push_back(storage);
                VmaAllocationInfo2 info{};
                vmaGetAllocationInfo2(context->allocator(),
                                      reinterpret_cast<VmaAllocation>(static_cast<uintptr_t>(storage.meta->gpu_descriptor.native_allocation)), &info);
                EXPECT_FALSE(info.dedicatedMemory) << "Small exported readbacks must share the host-visible pool";
                backing.insert(info.allocationInfo.deviceMemory);
                VkExportMetalBufferInfoEXT metal{VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT};
                metal.memory = info.allocationInfo.deviceMemory;
                VkExportMetalObjectsInfoEXT objects{VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
                objects.pNext = &metal;
                export_objects(context->device(), &objects);
                EXPECT_NE(metal.mtlBuffer, nullptr) << "The shared pool lost its Metal export contract";
                EXPECT_EQ(*static_cast<const std::byte*>(info.allocationInfo.pMappedData), std::byte{0});
            }
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            EXPECT_LT(backing.size(), readbacks.storage.size() / 4);
            std::printf("Exported host allocator sweep: bytes=%zu allocations=%zu backing_blocks=%zu allocation_ms=%.3f\n",
                        bytes, readbacks.storage.size(), backing.size(), elapsed);
        }
    }
    ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
    EXPECT_EQ(internal::vulkan_live_vma_objects_for_testing(), 0u);
}
#endif
