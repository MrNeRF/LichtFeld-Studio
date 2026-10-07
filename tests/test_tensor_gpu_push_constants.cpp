/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_shader_table.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/rasterizer/vulkan/src/gs_renderer.h"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>

namespace {
    void check_programs(const uint32_t limit) {
        std::ifstream inventory(LFS_PROGRAM_INVENTORY);
        ASSERT_TRUE(inventory.is_open());
        size_t count = 0;
        for (std::string path; std::getline(inventory, path);) {
            if (path.empty())
                continue;
            SCOPED_TRACE(path);
            std::ifstream input(path);
            ASSERT_TRUE(input.is_open());
            const auto reflection = nlohmann::json::parse(input);
            const auto bytes = reflection.at("parameters").at(0).at("type").at("elementVarLayout").at("binding").at("size").get<uint32_t>();
            EXPECT_LE(bytes, limit);
            ++count;
        }
        EXPECT_GT(count, 0u);
#ifdef LFS_TRAINING_MANIFEST_DIR
        size_t training_count = 0;
        for (const auto& file : std::filesystem::directory_iterator(LFS_TRAINING_MANIFEST_DIR)) {
            if (file.path().extension() != ".json")
                continue;
            SCOPED_TRACE(file.path().string());
            std::ifstream input(file.path());
            ASSERT_TRUE(input.is_open());
            const auto reflection = nlohmann::json::parse(input);
            EXPECT_LE(reflection.at("push_constant_size").get<uint32_t>(), limit);
            ++training_count;
        }
        EXPECT_GT(training_count, 0u);
#endif
        // Native rasterizer pipelines reserve this common range and reject
        // dispatches larger than it before recording push constants.
        EXPECT_LE(sizeof(VulkanGSRendererUniforms), limit);
        for (const auto& shader : lfs::core::internal::embedded_shaders())
            EXPECT_LE(shader.push_constant_size, limit) << shader.name;
    }

    TEST(GpuPushConstantContracts, StaticLimit) {
        check_programs(256);
    }

    TEST(GpuPushConstantContracts, DeviceLimit) {
        using namespace lfs::core;
        if (!gpu_backend_available(GpuBackend::Vulkan))
            GTEST_SKIP() << "Vulkan device unavailable";
        const auto context = internal::acquire_vulkan_context();
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
        check_programs(properties.limits.maxPushConstantsSize);
    }
} // namespace
