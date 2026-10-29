/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/vksplat_viewport_renderer.hpp"
#include "user_shader_images.hpp"
#include "window/vulkan_context.hpp"
#include <cstring>
#include <gtest/gtest.h>

namespace lfs::vis {
    struct UserShaderTestAccess {
        using Source = VulkanGSRenderer::ComputeProgramSource;
        inline static std::unordered_map<VksplatViewportRenderer*, VulkanGSRenderer::ProgramSetHandle> fixture_sets;

        static VulkanGSRenderer& renderer(VksplatViewportRenderer& viewport) { return viewport.renderer_; }
        static void publish(VksplatViewportRenderer& viewport, std::vector<Source> programs) {
            auto& renderer = viewport.renderer_;
            auto requested = renderer.prepareProgramSet(std::move(programs));
            ASSERT_TRUE(requested) << lfs::format_for_developer(requested.error());
            renderer.waitForProgramSetBuilds();
            ASSERT_TRUE(renderer.programSetReady(*requested).value_or(false));
            fixture_sets[&viewport] = *requested;
            renderer.useProgramSet(*requested);
        }
        static void release(VksplatViewportRenderer& viewport) {
            viewport.renderer_.useProgramSet({});
            fixture_sets.erase(&viewport);
            viewport.renderer_.drainRetiredProgramSets();
        }
        static std::size_t retiredSets(VksplatViewportRenderer& viewport) {
            viewport.renderer_.drainRetiredProgramSets();
            return viewport.renderer_.retiredProgramSetCount();
        }
        static const _VulkanBuffer& geometry(VksplatViewportRenderer& viewport) {
            return viewport.buffers_.custom_geometry.deviceBuffer;
        }
        static std::array<bool, 2> encodedSh(const VksplatViewportRenderer& viewport) {
            return {viewport.buffers_.shN_f16, viewport.buffers_.shN_q16};
        }
        static void warmup(VksplatViewportRenderer& viewport, bool pending) { viewport.macro_chain_warmup_pending_ = pending; }
        static bool quantPool(const VksplatViewportRenderer& viewport) { return viewport.buffers_.quant_pool; }
        // The last projection ran the effect's hosts.
        static bool customProjection(const VksplatViewportRenderer& viewport) {
            return viewport.resident_projection_custom_;
        }
        // The projected colors of the first `count` render slots.
        static std::vector<float> colors(VksplatViewportRenderer& viewport, const VmaAllocator allocator,
                                         const std::size_t count) {
            auto& renderer = viewport.renderer_;
            const auto& rgb = viewport.buffers_.rgb.deviceBuffer;
            if (rgb.buffer == VK_NULL_HANDLE) {
                ADD_FAILURE() << "no projected colors to read";
                return {};
            }
            return lfs::test::readBuffer(renderer, rgb, allocator, count * 3);
        }
        inline static std::size_t dispatches = 0;

        static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer command, std::uint32_t x, std::uint32_t y,
                                                   std::uint32_t z) {
            ++dispatches;
            vkCmdDispatch(command, x, y, z);
        }
        static VKAPI_ATTR void VKAPI_CALL dispatchIndirect(VkCommandBuffer command, VkBuffer buffer,
                                                           VkDeviceSize offset) {
            ++dispatches;
            vkCmdDispatchIndirect(command, buffer, offset);
        }
    };
} // namespace lfs::vis
