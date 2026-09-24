/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/headless_vulkan_device.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_splat.hpp"

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
