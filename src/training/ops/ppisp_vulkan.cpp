/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/ppisp_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <string_view>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        struct Push {
            uint64_t exposure, vignetting, color, crf, rgb, grad_output;
            uint64_t output, grad_exposure, grad_vignetting, grad_color, grad_crf, grad_rgb;
            uint64_t moment1, moment2, gradient, loss;
            uint32_t count, cameras, frames, height, width, y_offset, full_height;
            int32_t camera_index, frame_index;
            float lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps, center, channel, non_positive;
        };
        static_assert(sizeof(Push) == 200);
        static_assert(offsetof(Push, count) == 128);
        static_assert(offsetof(Push, lr) == 164);

        struct Pipeline {
            std::shared_ptr<VulkanContext> context;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkPipeline handle = VK_NULL_HANDLE;
            ~Pipeline() {
                if (!context || context->device() == VK_NULL_HANDLE)
                    return;
                if (handle != VK_NULL_HANDLE)
                    vkDestroyPipeline(context->device(), handle, nullptr);
                if (layout != VK_NULL_HANDLE)
                    vkDestroyPipelineLayout(context->device(), layout, nullptr);
            }
        };

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, uint32_t op) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), op};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("ppisp"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan PPISP shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.ppisp)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize,
                           "Vulkan PPISP parameters exceed device push-constant limit");
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.ppisp)");
            const VkSpecializationMapEntry entry{0, 0, sizeof(op)};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = 1;
            specialization.pMapEntries = &entry;
            specialization.dataSize = sizeof(op);
            specialization.pData = &op;
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.ppisp)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan, "Vulkan PPISP op received storage from another backend");
            return storage;
        }
        void append(std::vector<StorageRef>& refs, const Tensor& tensor) {
            if (tensor.is_valid() && tensor.numel())
                refs.push_back(ref(tensor));
        }
        void launch(const std::shared_ptr<VulkanContext>& context, const Push& p, uint32_t op,
                    const std::vector<StorageRef>& reads, const std::vector<StorageRef>& writes, uint32_t groups) {
            if (!groups)
                return;
            const auto pipeline = pipeline_for(context, op);
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(command, groups, 1, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }
        uint32_t count32(size_t count) {
            LFS_ASSERT_MSG(count <= UINT32_MAX, "PPISP tensor exceeds 32-bit indexing");
            return static_cast<uint32_t>(count);
        }

        void forward(const PPISPInputs& pp, In rgb, Out corrected, const PPISPRegion& r) {
            LFS_ASSERT_MSG(rgb.ndim() == 3 && rgb.shape()[0] == 3, "PPISP expects CHW RGB");
            const uint32_t h = count32(rgb.shape()[1]), w = count32(rgb.shape()[2]);
            LFS_ASSERT_MSG(h && w && r.y_offset >= 0 && r.full_height >= static_cast<int>(h) + r.y_offset,
                           "PPISP region is outside full image");
            Push p{};
            p.exposure = vk::address(ref(pp.exposure));
            p.vignetting = vk::address(ref(pp.vignetting));
            p.color = vk::address(ref(pp.color));
            p.crf = vk::address(ref(pp.crf));
            p.rgb = vk::address(ref(rgb));
            p.output = vk::address(ref(corrected));
            p.height = h;
            p.width = w;
            p.y_offset = r.y_offset;
            p.full_height = r.full_height;
            p.camera_index = r.camera_index;
            p.frame_index = r.frame_index;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 0, {ref(pp.exposure), ref(pp.vignetting), ref(pp.color), ref(pp.crf), ref(rgb)},
                   {ref(corrected)}, vk::dispatch_groups(*ctx, static_cast<size_t>(h) * w));
        }

        void backward(const PPISPInputs& pp, In rgb, In grad_output, const PPISPOutputs& g, Out grad_rgb,
                      int, int, int camera, int frame) {
            const uint32_t h = count32(rgb.shape()[1]), w = count32(rgb.shape()[2]);
            Push p{};
            p.exposure = vk::address(ref(pp.exposure));
            p.vignetting = vk::address(ref(pp.vignetting));
            p.color = vk::address(ref(pp.color));
            p.crf = vk::address(ref(pp.crf));
            p.rgb = vk::address(ref(rgb));
            p.grad_output = vk::address(ref(grad_output));
            p.grad_exposure = vk::address(ref(g.exposure));
            p.grad_vignetting = vk::address(ref(g.vignetting));
            p.grad_color = vk::address(ref(g.color));
            p.grad_crf = vk::address(ref(g.crf));
            p.grad_rgb = vk::address(ref(grad_rgb));
            p.height = h;
            p.width = w;
            p.camera_index = camera;
            p.frame_index = frame;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 1, {ref(pp.exposure), ref(pp.vignetting), ref(pp.color), ref(pp.crf), ref(rgb), ref(grad_output), ref(g.exposure), ref(g.vignetting), ref(g.color), ref(g.crf)},
                   {ref(grad_rgb), ref(g.exposure), ref(g.vignetting), ref(g.color), ref(g.crf)}, vk::dispatch_groups(*ctx, static_cast<size_t>(h) * w));
        }

        void adam(const PPISPAdamGroup& group, const PPISPAdamUpdateParams& a) {
            if (!group.parameter.is_valid() || !group.parameter.numel())
                return;
            Push p{};
            p.output = vk::address(ref(group.parameter));
            p.moment1 = vk::address(ref(group.moment1));
            p.moment2 = vk::address(ref(group.moment2));
            p.gradient = vk::address(ref(group.gradient));
            p.count = count32(group.parameter.numel());
            p.lr = a.lr;
            p.beta1 = a.beta1;
            p.beta2 = a.beta2;
            p.bc1_rcp = a.bc1_rcp;
            p.bc2_sqrt_rcp = a.bc2_sqrt_rcp;
            p.eps = a.eps;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 2, {ref(group.parameter), ref(group.moment1), ref(group.moment2), ref(group.gradient)},
                   {ref(group.parameter), ref(group.moment1), ref(group.moment2)}, vk::dispatch_groups(*ctx, p.count));
        }
        void adam_batch(const std::array<PPISPAdamGroup, 4>& groups, const PPISPAdamUpdateParams& p) {
            for (const auto& g : groups)
                adam(g, p);
        }

        void vignetting_regularization(In parameters, Out gradient, Out loss, float center, float channel, float non_positive) {
            const size_t cameras = parameters.numel() / 15;
            if (!cameras)
                return;
            Push p{};
            p.vignetting = vk::address(ref(parameters));
            p.grad_vignetting = gradient.is_valid() ? vk::address(ref(gradient)) : 0;
            p.loss = loss.is_valid() ? vk::address(ref(loss)) : 0;
            p.cameras = count32(cameras);
            p.center = center;
            p.channel = channel;
            p.non_positive = non_positive;
            const auto ctx = acquire_vulkan_context();
            std::vector<StorageRef> reads{ref(parameters)}, writes;
            append(reads, gradient);
            append(reads, loss);
            if (gradient.is_valid())
                writes.push_back(ref(gradient));
            if (loss.is_valid())
                writes.push_back(ref(loss));
            launch(ctx, p, 3, reads, writes, vk::dispatch_groups(*ctx, cameras));
        }

        void project_mean(Out exposure, Out color) {
            const size_t frames = exposure.numel();
            if (!frames)
                return;
            Push p{};
            p.exposure = vk::address(ref(exposure));
            p.color = vk::address(ref(color));
            p.frames = count32(frames);
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 4, {ref(exposure), ref(color)}, {ref(exposure), ref(color)}, 9);
        }

        void initialize(const PPISPOutputs& outputs) {
            const size_t cameras = outputs.vignetting.numel() / 15, frames = outputs.exposure.numel();
            Push p{};
            p.exposure = vk::address(ref(outputs.exposure));
            p.vignetting = vk::address(ref(outputs.vignetting));
            p.color = vk::address(ref(outputs.color));
            p.crf = vk::address(ref(outputs.crf));
            p.frames = count32(frames);
            p.cameras = count32(cameras);
            p.count = count32(std::max({frames, cameras * 15, frames * 8, cameras * 12}));
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 5, {ref(outputs.exposure), ref(outputs.vignetting), ref(outputs.color), ref(outputs.crf)},
                   {ref(outputs.exposure), ref(outputs.vignetting), ref(outputs.color), ref(outputs.crf)}, vk::dispatch_groups(*ctx, p.count));
        }

        const PPISPOps kOps{forward, backward, adam, adam_batch, vignetting_regularization, project_mean, initialize};
    } // namespace
    const gpu_ops::PPISPOps& vulkan_ppisp_ops() { return kOps; }
} // namespace lfs::training
