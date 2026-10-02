/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/sh_layout.cuh"
#include "core/headless_vulkan_device.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/rasterizer/vulkan/src/buffer.h"
#include "visualizer/rendering/vksplat_input_packer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;
    using lfs::vis::vksplat::rawDeviceInputLayout;

    class VkSplatDeletedMask : public testing::TestWithParam<std::tuple<lfs::core::GpuBackend, std::size_t>> {
    protected:
        std::optional<lfs::core::HeadlessAdoptedDevice> device_;

        void SetUp() override {
            using namespace lfs::core;
            if (!gpu_backend_available(std::get<0>(GetParam())))
                GTEST_SKIP() << "Tensor backend unavailable";
            ASSERT_TRUE(shutdown_gpu_backend(GpuBackend::Vulkan));
            auto candidate = HeadlessAdoptedDevice::try_create(true);
            if (!candidate)
                GTEST_SKIP() << "Vulkan interop unavailable";
            device_.emplace(std::move(*candidate));
            ASSERT_TRUE(adopt_vulkan_device(device_->handles()));
        }

        void TearDown() override {
            EXPECT_TRUE(lfs::core::shutdown_gpu_backend(lfs::core::GpuBackend::Vulkan));
            device_.reset();
        }
    };

    TEST_P(VkSplatDeletedMask, BindsVisibleStorageAndRefreshesOnlyWhenRequested) {
        using namespace lfs::core;
        using lfs::vis::vksplat::prepareDeletedMask;
        const auto [backend, n] = GetParam();
        GpuBackendScope scope(backend);
        const auto handles = device_->handles();
        TensorVulkanInterop interop(VulkanInteropDevice{
            .physical_device = handles.physical_device,
            .device = handles.device,
            .queue_families = {handles.queue_family},
            .queue_family_count = 1,
            .external_memory = true,
            .external_semaphore = true,
            .metal_objects = handles.metal_objects});
        auto input = Tensor::ones_bool({n}, Device::GPU);
        if (backend == GpuBackend::CUDA)
            ASSERT_FALSE(interop.buffer(input));
        Tensor slot;
        auto bound = prepareDeletedMask(interop, input, backend, true, slot);
        ASSERT_TRUE(interop.buffer(bound));
        EXPECT_EQ(bound.numel(), (n + 3u) & ~std::size_t{3u});
        auto values = bound.cpu();
        for (std::size_t i = 0; i < bound.numel(); ++i)
            EXPECT_EQ(values.ptr<std::uint8_t>()[i], i < n ? 1 : 0);
        const auto* allocation = bound.data_ptr();
        input.zero_();
        bound = prepareDeletedMask(interop, input, backend, false, slot);
        EXPECT_EQ(bound.data_ptr(), allocation);
        if (slot.is_valid()) {
            // No upload was requested: a retained copy must still have the old bytes.
            EXPECT_EQ(bound.cpu().ptr<std::uint8_t>()[0], 1);
        }
        bound = prepareDeletedMask(interop, input, backend, true, slot);
        EXPECT_EQ(bound.data_ptr(), allocation);
        EXPECT_EQ(bound.cpu().ptr<std::uint8_t>()[0], 0);

        // Already visible, aligned storage must bind directly even on refresh.
        auto visible = interop.empty({8}, DataType::Bool, backend);
        visible.zero_();
        bound = prepareDeletedMask(interop, visible, backend, true, slot);
        EXPECT_EQ(bound.data_ptr(), visible.data_ptr());
        EXPECT_FALSE(slot.is_valid());
        bound = prepareDeletedMask(interop, visible, backend, true, slot);
        EXPECT_EQ(bound.data_ptr(), visible.data_ptr());
        EXPECT_FALSE(slot.is_valid());
        interop.drain(backend);
    }

    INSTANTIATE_TEST_SUITE_P(Storage, VkSplatDeletedMask,
                             testing::Combine(testing::Values(lfs::core::GpuBackend::CUDA,
                                                              lfs::core::GpuBackend::Vulkan),
                                              testing::Values(std::size_t{4}, std::size_t{5})));

    enum class RestStorage { Float32,
                             Float16,
                             Quantized16 };

    class RawInputLayout : public testing::TestWithParam<std::tuple<int, RestStorage>> {};

    template <typename Handle>
    Handle fakeHandle(std::uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>)
            return reinterpret_cast<Handle>(value);
        else
            return static_cast<Handle>(value);
    }

    TEST(VulkanBufferView, RejectsRangesOutsideTheBoundRegion) {
        _VulkanBuffer view{};
        view.buffer = fakeHandle<VkBuffer>(1);
        view.allocSize = 1024;
        view.offset = 256;
        view.capacity = 128;
        view.size = 64;
        EXPECT_TRUE(view.hasValidViewBounds());
        EXPECT_TRUE(view.containsRange(0, 64));
        EXPECT_FALSE(view.containsRange(128, 1));
        view.offset = 960;
        EXPECT_FALSE(view.hasValidViewBounds());
    }

    TEST_P(RawInputLayout, ReportsBytesForResidentTensors) {
        constexpr std::size_t n = 5;
        const auto [degree, storage] = GetParam();
        const std::size_t rest = static_cast<std::size_t>((degree + 1) * (degree + 1) - 1);
        auto f32 = [](std::size_t count, lfs::core::TensorShape shape) {
            return Tensor::from_vector(std::vector<float>(count, 0.0f), shape, Device::GPU);
        };
        Tensor shn = rest ? f32(n * rest * 3, {n, rest, 3}) : Tensor{};
        SplatData splat(degree,
                        f32(n * 3, {n, 3}), f32(n * 3, {n, 1, 3}), std::move(shn),
                        f32(n * 3, {n, 3}), f32(n * 4, {n, 4}), f32(n, {n, 1}),
                        1.0f);
        if (storage == RestStorage::Float16)
            splat.shN() = splat.shN().to(DataType::Float16);
        if (storage == RestStorage::Quantized16) {
            splat.shN() = Tensor::zeros({lfs::core::sh_value_quant::sh_value_u16_count(n, rest)},
                                        Device::GPU, DataType::Float16);
            splat.shN_value_bounds() = f32(2, {2});
        }

        const auto layout = rawDeviceInputLayout(splat);
        ASSERT_TRUE(layout) << layout.error();
        EXPECT_EQ(layout->num_splats, n);
        EXPECT_EQ(layout->xyz_bytes, n * 3 * sizeof(float));
        EXPECT_EQ(layout->rotations_bytes, n * 4 * sizeof(float));
        EXPECT_EQ(layout->scaling_bytes, n * 3 * sizeof(float));
        EXPECT_EQ(layout->opacity_bytes, n * sizeof(float));
        EXPECT_EQ(layout->non_sh_bytes, n * 11 * sizeof(float));
        EXPECT_EQ(layout->omits_shN, degree == 0);
        EXPECT_EQ(layout->shN_f16, storage == RestStorage::Float16);
        EXPECT_EQ(layout->shN_q16, storage == RestStorage::Quantized16);
        if (degree == 0) {
            EXPECT_EQ(layout->shN_bytes, 4 * sizeof(float));
        } else if (storage == RestStorage::Quantized16) {
            EXPECT_EQ(layout->shN_bytes,
                      lfs::core::sh_value_quant::sh_value_u16_count(n, rest) * sizeof(std::uint16_t));
            EXPECT_EQ(layout->shN_bounds_bytes, 2 * sizeof(float));
        } else {
            const std::size_t slots = (rest * 3 + 3) / 4;
            const std::size_t groups = (n + lfs::core::kShReorderSize - 1) /
                                       lfs::core::kShReorderSize;
            const std::size_t expected_elements = groups * slots *
                                                  lfs::core::kShReorderSize * 4;
            EXPECT_EQ(layout->shN_bytes, expected_elements *
                                             (storage == RestStorage::Float16 ? 2 : 4));
        }
    }

    INSTANTIATE_TEST_SUITE_P(ResidentFormats, RawInputLayout,
                             testing::Values(std::tuple{0, RestStorage::Float32},
                                             std::tuple{3, RestStorage::Float32},
                                             std::tuple{3, RestStorage::Float16},
                                             std::tuple{3, RestStorage::Quantized16}));
} // namespace
