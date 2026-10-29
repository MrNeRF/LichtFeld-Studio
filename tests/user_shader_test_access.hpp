/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/rendering_manager.hpp"
#include "rendering/vksplat_viewport_renderer.hpp"
#include "user_shader_images.hpp"
#include "window/vulkan_context.hpp"
#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>

namespace lfs::vis {
    // Fixture commands use production validation and installation without adding UI history or routing a timeline.
    struct ShaderCommands {
        shaders::ShaderService& service;
        auto command(const shaders::ShaderCommand& cmd) {
            return service.applyCommand(cmd, {}, false);
        }
        auto createAsset(std::string name, std::string source) { return command(shaders::SetShaderSource{{}, std::move(source), std::move(name), {}}); }
        auto updateSource(core::Uuid id, std::string source) { return command(shaders::SetShaderSource{id, std::move(source), {}, {}}); }
        auto renameAsset(core::Uuid id, std::string name) { return command(shaders::RenameShader{id, std::move(name)}); }
        auto removeAsset(core::Uuid id) { return command(shaders::RemoveShader{id}); }
        auto setParameter(core::Uuid id, std::string_view name, std::span<const float> values) { return command(shaders::SetShaderParameter{id, std::string(name), {values.begin(), values.end()}, {}}); }
        auto bind(shaders::ShaderSlot slot, core::Uuid id) { return command(shaders::BindShader{slot, id}); }
        auto unbind(shaders::ShaderSlot slot) { return command(shaders::UnbindShader{slot, {}}); }
    };
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
        static void waitForBuilds(VksplatViewportRenderer& viewport) { viewport.renderer_.waitForProgramSetBuilds(); }
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
        using Slot = VksplatViewportRenderer::OutputSlot;
        inline static std::size_t binds = 0;
        inline static std::size_t dispatches = 0;
        inline static std::atomic<bool> fail_pipeline = false;
        inline static std::thread::id owner;

        static VKAPI_ATTR VkResult VKAPI_CALL createPipelines(
            VkDevice device, VkPipelineCache cache, uint32_t count, const VkComputePipelineCreateInfo* infos,
            const VkAllocationCallbacks* allocator, VkPipeline* out) {
            if (std::this_thread::get_id() != owner && fail_pipeline.exchange(false))
                return VK_ERROR_OUT_OF_DEVICE_MEMORY;
            return vkCreateComputePipelines(device, cache, count, infos, allocator, out);
        }

        static VKAPI_ATTR void VKAPI_CALL bind(VkCommandBuffer command, VkPipelineBindPoint point,
                                               VkPipeline pipeline) {
            ++binds;
            vkCmdBindPipeline(command, point, pipeline);
        }
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
        // Counts the renderer's pipeline binds and dispatches.
        static void install(RenderingManager& manager, VulkanContext& context) {
            owner = std::this_thread::get_id();
            manager.last_vulkan_context_ = &context;
            manager.vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
            auto table = rendering::VulkanDispatch::real();
            table.create_compute_pipelines = &createPipelines;
            table.cmd_bind_pipeline = &bind;
            table.cmd_dispatch = &dispatch;
            table.cmd_dispatch_indirect = &dispatchIndirect;
            manager.vksplat_viewport_renderer_->renderer_.setVulkanDispatch(table);
        }
        // The programs the latest on-screen frame adopted, whether or not a draw ran them.
        static std::shared_ptr<const shaders::ShaderPrograms> displayPrograms(RenderingManager& manager) {
            const auto& record = manager.display_record_;
            return record && record->snapshot ? record->snapshot->programs : nullptr;
        }
        // The device set of the latest on-screen frame.
        static rendering::ProgramSetHandle displaySet(RenderingManager& manager) {
            return manager.display_record_ ? manager.display_record_->programs : nullptr;
        }
        static VksplatViewportRenderer& renderer(RenderingManager& manager) {
            return *manager.vksplat_viewport_renderer_;
        }
        static Viewport& secondaryViewport(RenderingManager& manager) {
            return manager.split_view_service_.secondaryViewport();
        }
        static void resetRenderer(RenderingManager& manager) { manager.releaseSceneRenderResources(); }
        static std::size_t retiredSets(RenderingManager& manager) {
            return renderer(manager).renderer_.retiredProgramSetCount();
        }
        static std::uint64_t programSetSerial(RenderingManager& manager) { return renderer(manager).programSetSerial(); }
        // Whether every custom draw of the latest frame found its program.
        static bool ranEveryProgram(RenderingManager& manager) {
            return manager.custom_frame_complete_;
        }
        // RGB of the latest render into `slot`.
        static std::vector<float> image(RenderingManager& manager, VulkanContext& context, const Slot slot) {
            auto rgba = renderer(manager).readOutputImageRgba(context, slot);
            if (!rgba) {
                ADD_FAILURE() << rgba.error();
                return {};
            }
            return lfs::test::rgbChannels((*rgba)->ptr<float>(), (*rgba)->numel() / 4, 4);
        }
    };
} // namespace lfs::vis
