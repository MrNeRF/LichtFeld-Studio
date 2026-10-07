/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/training_image_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <mutex>
#include <ranges>
#include <string_view>
#include <tuple>

namespace lfs::training {
    core::Tensor vulkan_flip_error_map(const core::Tensor&, const core::Tensor&, float);
    core::Tensor vulkan_flip_error_image(const core::Tensor&);

    namespace {
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        constexpr uint32_t kHeatmap = 0, kRoi = 1, kResize = 2, kRandom = 3, kCanny = 4, kNormalize = 5;
        struct alignas(16) Float3 {
            float x, y, z, pad;
        };
        struct alignas(16) Push {
            uint64_t input, output, secondary, tertiary;
            uint64_t camera, unused0, unused1, unused2;
            uint32_t count, width, height, source_width;
            uint32_t source_height, channels, slot, slot_count;
            float fx, fy, cx, cy;
            float outside_weight, ema_alpha, skip_below, scalar0;
            std::array<std::array<float, 4>, 3> crop_rows;
            Float3 crop_min;
            uint32_t inverse;
            Float3 crop_max;
            uint32_t seed;
        };
        static_assert(offsetof(Push, crop_rows) == 128);
        static_assert(offsetof(Push, crop_min) == 176);
        static_assert(offsetof(Push, crop_max) == 208);
        static_assert(sizeof(Push) == 240);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context,
                                               uint32_t operation, uint32_t bytes = 0) {
            static std::mutex mutex;
            static std::map<std::tuple<uint64_t, uint32_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::tuple{context->context_id(), operation, bytes};
            std::lock_guard lock(mutex);
            if (auto it = cache.find(key); it != cache.end())
                return it->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("training_image"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan training-image shader is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader), "vkCreateShaderModule(training.image)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize, "Vulkan training-image parameters exceed push-constant limit");
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout), "vkCreatePipelineLayout(training.image)");
            const std::array<VkSpecializationMapEntry, 2> entries{{{0, 0, sizeof(operation)}, {1, sizeof(operation), sizeof(bytes)}}};
            const std::array<uint32_t, 2> values{operation, bytes};
            VkSpecializationInfo specialization{static_cast<uint32_t>(entries.size()), entries.data(), sizeof(values), values.data()};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            info.stage = stage;
            info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &info, nullptr, &pipeline->handle), "vkCreateComputePipelines(training.image)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            vulkan::release_at_shutdown(*context, mutex, cache);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const auto storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan, "Vulkan training-image op received non-Vulkan storage");
            return storage;
        }
        void launch(Push push, uint32_t operation, uint32_t bytes,
                    std::span<const StorageRef> reads, std::span<const StorageRef> writes,
                    uint32_t groups_x, uint32_t groups_y = 1) {
            if (groups_x == 0 || groups_y == 0)
                return;
            const auto context = acquire_vulkan_context();
            auto pipeline = pipeline_for(context, operation, bytes);
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, groups_x, groups_y, 1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }
        uint32_t count32(size_t n) {
            LFS_ASSERT_MSG(n <= UINT32_MAX, "Vulkan training-image count exceeds uint32");
            return static_cast<uint32_t>(n);
        }

        void heatmap(In loss, Out latest, Out ema, int slot, float alpha) {
            Push p{};
            p.input = vk::address(ref(loss));
            p.output = vk::address(ref(latest));
            p.secondary = vk::address(ref(ema));
            p.slot = static_cast<uint32_t>(slot);
            p.slot_count = count32(latest.numel());
            p.ema_alpha = alpha;
            auto a = ref(loss), b = ref(latest), c = ref(ema);
            const std::array reads{a, b, c};
            const std::array writes{b, c};
            launch(p, kHeatmap, 0, reads, writes, 1);
        }
        void roi(In view, In camera, Out weights, const RoiParams& params) {
            LFS_ASSERT_MSG(params.image.w > 0 && params.image.h > 0 && params.intrinsics.fx > 0 && params.intrinsics.fy > 0, "ROI dimensions and focal lengths must be positive");
            Push p{};
            p.input = vk::address(ref(view));
            p.output = vk::address(ref(weights));
            p.camera = vk::address(ref(camera));
            p.width = static_cast<uint32_t>(params.image.w);
            p.height = static_cast<uint32_t>(params.image.h);
            p.count = count32(static_cast<size_t>(p.width) * p.height);
            p.fx = params.intrinsics.fx;
            p.fy = params.intrinsics.fy;
            p.cx = params.intrinsics.cx;
            p.cy = params.intrinsics.cy;
            p.outside_weight = params.outside_weight;
            p.inverse = params.inverse ? 1u : 0u;
            const auto& m = params.world_to_cropbox;
            p.crop_rows = {{{m[0], m[4], m[8], m[12]}, {m[1], m[5], m[9], m[13]}, {m[2], m[6], m[10], m[14]}}};
            p.crop_min = {params.minimum[0], params.minimum[1], params.minimum[2], 0};
            p.crop_max = {params.maximum[0], params.maximum[1], params.maximum[2], 0};
            const auto a = ref(view), b = ref(camera), c = ref(weights);
            const std::array reads{a, b};
            const std::array writes{c};
            const auto context = acquire_vulkan_context();
            const uint32_t gx = static_cast<uint32_t>((p.count + 255) / 256);
            LFS_ASSERT_MSG(gx <= context->caps().max_workgroup_count[0], "ROI dispatch exceeds Vulkan limit");
            launch(p, kRoi, 0, reads, writes, gx);
        }
        void resize_background(In source, Out destination) {
            Push p{};
            p.input = vk::address(ref(source));
            p.output = vk::address(ref(destination));
            p.channels = count32(source.shape()[0]);
            p.source_height = count32(source.shape()[1]);
            p.source_width = count32(source.shape()[2]);
            p.height = count32(destination.shape()[1]);
            p.width = count32(destination.shape()[2]);
            p.count = count32(destination.shape()[1] * destination.shape()[2]);
            const auto a = ref(source), b = ref(destination);
            const std::array reads{a};
            const std::array writes{b};
            const auto context = acquire_vulkan_context();
            uint32_t gx = static_cast<uint32_t>((p.count + 255) / 256);
            LFS_ASSERT_MSG(gx <= context->caps().max_workgroup_count[0], "resize dispatch exceeds Vulkan limit");
            launch(p, kResize, 0, reads, writes, gx);
        }
        void random_background(Out destination, uint64_t seed) {
            const size_t plane = destination.shape()[1] * destination.shape()[2];
            Push p{};
            p.output = vk::address(ref(destination));
            p.count = count32(plane);
            p.seed = static_cast<uint32_t>(seed);
            const auto out = ref(destination);
            const std::array writes{out};
            const auto context = acquire_vulkan_context();
            uint32_t gx = static_cast<uint32_t>((p.count + 255) / 256);
            LFS_ASSERT_MSG(gx <= context->caps().max_workgroup_count[0], "random background dispatch exceeds Vulkan limit");
            launch(p, kRandom, 0, {}, writes, gx);
        }
        void canny(In image, Out edges) {
            Push p{};
            p.input = vk::address(ref(image));
            p.output = vk::address(ref(edges));
            p.height = count32(image.shape()[1]);
            p.width = count32(image.shape()[2]);
            p.count = p.height * p.width;
            const uint32_t bytes = image.dtype() == core::DataType::UInt8 ? 1u : 0u;
            const auto input = ref(image), output = ref(edges);
            const std::array reads{input};
            const std::array writes{output};
            const auto context = acquire_vulkan_context();
            const uint32_t gx = (p.width + 15) / 16, gy = (p.height + 15) / 16;
            LFS_ASSERT_MSG(gx <= context->caps().max_workgroup_count[0] && gy <= context->caps().max_workgroup_count[1], "Canny dispatch exceeds Vulkan limit");
            launch(p, kCanny, bytes, reads, writes, gx, gy);
        }
        void normalize_scalar(Out values, In scalar, float skip_below) {
            if (values.numel() == 0)
                return;
            Push p{};
            p.output = vk::address(ref(values));
            p.secondary = vk::address(ref(scalar));
            p.count = count32(values.numel());
            p.skip_below = skip_below;
            const auto a = ref(values), b = ref(scalar);
            const std::array reads{a, b};
            const std::array writes{a};
            const auto context = acquire_vulkan_context();
            uint32_t gx = (p.count + 255) / 256;
            LFS_ASSERT_MSG(gx <= context->caps().max_workgroup_count[0], "scalar normalization dispatch exceeds Vulkan limit");
            launch(p, kNormalize, 0, reads, writes, gx);
        }
        Tensor upload_image_chw(In cpu_hwc) {
            return cpu_hwc.to(core::Device::GPU).permute({2, 0, 1}).contiguous();
        }
        const TrainingImageOps kVulkanTrainingImageOps{.heatmap = heatmap, .roi = roi, .resize_background = resize_background, .random_background = random_background, .canny = canny, .normalize_scalar = normalize_scalar, .upload_image_chw = upload_image_chw, .flip_error_map = vulkan_flip_error_map, .flip_error_image = vulkan_flip_error_image};
    } // namespace
    const TrainingImageOps& vulkan_training_image_ops() { return kVulkanTrainingImageOps; }
} // namespace lfs::training
