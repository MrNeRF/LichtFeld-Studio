/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/error.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include <array>
#include <gtest/gtest.h>
#include <string>
#include <vector>
#if defined(LFS_TEST_TENSOR_VULKAN)
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "training/vulkan/fast_state.hpp"

namespace {
    TEST(TensorVulkanTrainingFault, InvalidTileDataSurfacesAtSynchronization) {
        using namespace lfs;
        using core::DataType;
        using core::Device;
        using core::Tensor;
        namespace vk = training::vulkan;
        if (!core::gpu_backend_available(core::GpuBackend::Vulkan))
            GTEST_SKIP();
        core::GpuBackendScope scope(core::GpuBackend::Vulkan);
        const auto context = core::internal::acquire_vulkan_context();
        for (const uint32_t kind : {0u, 1u, 2u}) {
            SCOPED_TRACE(kind);
            auto keys = Tensor::full({1}, kind == 0 ? 1 : 0, Device::GPU, DataType::Int32);
            auto values = Tensor::full({1}, kind == 1 ? 1 : 0, Device::GPU, DataType::Int32);
            auto ranges = Tensor::zeros({2}, Device::GPU, DataType::Int32);
            auto offsets = Tensor::ones({1}, Device::GPU, DataType::Int64);
            auto projected = Tensor::zeros({sizeof(vk::Projected) / 4}, Device::GPU);
            vk::FastPush p{};
            p.keys = vk::address(keys);
            p.values = vk::address(values);
            p.ranges = vk::address(ranges);
            p.offsets = vk::address(offsets);
            p.projected = vk::address(projected);
            p.status = context->fault_address();
            p.instances = p.visible = p.grid_width = p.grid_height = 1;
            p.stage = kind == 2 ? 3 : 4;
            const std::array reads{vk::ref(keys), vk::ref(values), vk::ref(offsets), vk::ref(projected)};
            const std::array writes{vk::ref(ranges)};
            // A zero-size projected tile with a nonzero offset tests emission
            // mismatch; the other cases test invalid tile and primitive ids.
            ASSERT_NO_THROW(vk::dispatch("fast_forward", p, reads, writes, 1, vk::specialization(p, p.stage)));
            try {
                (void)ranges.cpu();
                FAIL() << "Expected a deferred rasterizer fault";
            } catch (const lfs::Exception& error) {
                EXPECT_EQ(error.error().code(), ErrorCode::BoundsViolation);
                EXPECT_NE(lfs::format_for_developer(error.error()).find("invalid tile data"), std::string::npos);
            }
            EXPECT_NO_THROW((void)ranges.cpu());
            EXPECT_EQ(ranges.to_vector_int(), (std::vector<int>{0, 0}));
        }
    }
} // namespace
#endif
