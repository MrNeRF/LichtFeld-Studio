/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/geometry_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <cstring>
#include <format>
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
        namespace k = lfs::training::kernels;

        constexpr uint32_t kMaxAnchorSamples = 262144;

        struct Push {
            uint64_t normal, depth, alpha, target, weight;
            uint64_t grad_normal, grad_depth, grad_alpha, loss, partials;
            uint64_t points, view, pairs, sample_count;
            float lo_x, lo_y, lo_z, hi_x, hi_y, hi_z;
            uint32_t width, height, blocks, has_weight, samples, stride, capacity;
            int32_t anchor_model;
            uint32_t prior;
            float fx, fy, cx, cy, loss_weight, lambda_grad, quant_step, floor_override;
            float anchor_scale, anchor_shift, anchor_floor, min_count, min_weight, near_plane;
        };
        static_assert(sizeof(Push) == 232);
        static_assert(offsetof(Push, width) == 136);
        static_assert(offsetof(Push, fx) == 172);

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
            const std::string_view name = context->caps().shader_float64 ? "geometry" : "geometry_fp32";
            const auto module = std::ranges::find(modules, name, &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan Geometry shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.geometry)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.geometry)");
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
                     "vkCreateComputePipelines(training.geometry)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan Geometry op received storage from another backend");
            return storage;
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
            LFS_ASSERT_MSG(count <= UINT32_MAX, "Vulkan Geometry input exceeds 32-bit indexing");
            return static_cast<uint32_t>(count);
        }

        Push base(In alpha, In pixel_weight, Out loss, Out partials, int width, int height, size_t slot_count,
                  size_t required, std::string_view name) {
            const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
            LFS_ASSERT_MSG(pixels && pixels <= UINT32_MAX, "Vulkan Geometry image size is outside 32-bit indexing");
            LFS_ASSERT_MSG(alpha.numel() == pixels, "Vulkan Geometry alpha shape does not match image");
            LFS_ASSERT_MSG(!pixel_weight.is_valid() || pixel_weight.numel() == pixels,
                           "Vulkan Geometry pixel weight shape does not match image");
            LFS_ASSERT_MSG(loss.is_valid() && loss.numel() >= 1, "Vulkan Geometry loss output must be non-empty");
            LFS_ASSERT_MSG(partials.is_valid() && partials.dtype() == core::DataType::Float32 && partials.numel() >= required,
                           std::format("Vulkan Geometry {} partials need {} Float32 values", name, required));
            Push p{};
            p.alpha = vk::address(ref(alpha));
            p.weight = pixel_weight.is_valid() ? vk::address(ref(pixel_weight)) : 0;
            p.loss = vk::address(ref(loss));
            p.partials = vk::address(ref(partials));
            p.width = count32(static_cast<size_t>(width));
            p.height = count32(static_cast<size_t>(height));
            p.blocks = static_cast<uint32_t>(k::geometry_loss_block_count(pixels));
            p.has_weight = pixel_weight.is_valid() ? 1u : 0u;
            // The shader computes its double-statistics base from this float-slot prefix.
            (void)slot_count;
            return p;
        }

        std::vector<StorageRef> optional_read(const Tensor& value) {
            return value.is_valid() && value.numel() ? std::vector<StorageRef>{ref(value)} : std::vector<StorageRef>{};
        }

        void depth(In depth_map, In alpha, In target, In pixel_weight, Out grad_depth, Out grad_alpha, Out loss,
                   Out partials, const DepthParams& params) {
            const uint32_t width = count32(depth_map.shape()[1]), height = count32(depth_map.shape()[0]);
            Push p = base(alpha, pixel_weight, loss, partials, width, height, k::depth_loss_slots::kSlotCount,
                          k::depth_loss_partial_count(static_cast<size_t>(width) * height), "depth");
            p.depth = vk::address(ref(depth_map));
            p.target = vk::address(ref(target));
            p.grad_depth = vk::address(ref(grad_depth));
            p.grad_alpha = vk::address(ref(grad_alpha));
            p.loss_weight = params.weight;
            p.lambda_grad = params.gradient_weight;
            p.quant_step = std::max(params.prior_quantization_step, 0.0f);
            const bool anchor = params.anchor && params.anchor->valid;
            p.anchor_model = anchor ? params.anchor->model : 0;
            p.anchor_scale = anchor ? params.anchor->scale : 0.0f;
            p.anchor_shift = anchor ? params.anchor->shift : 0.0f;
            p.anchor_floor = anchor ? params.anchor->floor : 0.0f;
            p.floor_override = p.anchor_floor;
            const auto context = acquire_vulkan_context();
            std::vector<StorageRef> reads{ref(depth_map), ref(alpha), ref(target), ref(partials)};
            if (pixel_weight.is_valid())
                reads.push_back(ref(pixel_weight));
            const std::vector<StorageRef> outputs{ref(grad_depth), ref(grad_alpha), ref(loss), ref(partials)};
            launch(context, p, 0, reads, outputs, p.blocks);
            launch(context, p, 1, {ref(partials)}, outputs, 1);
            launch(context, p, 2, reads, outputs, p.blocks);
            launch(context, p, 3, {ref(partials)}, outputs, 1);
            launch(context, p, 4, reads, outputs, p.blocks);
            launch(context, p, 5, {ref(partials)}, outputs, 1);
        }

        void normal(In normal_map, In alpha, In target, In pixel_weight, Out grad_normal, Out loss, Out partials,
                    float weight) {
            const uint32_t width = count32(normal_map.shape()[2]), height = count32(normal_map.shape()[1]);
            Push p = base(alpha, pixel_weight, loss, partials, width, height, k::normal_loss_slots::kSlotCount,
                          k::normal_loss_partial_count(static_cast<size_t>(width) * height), "normal");
            p.normal = vk::address(ref(normal_map));
            p.target = vk::address(ref(target));
            p.grad_normal = vk::address(ref(grad_normal));
            p.loss_weight = weight;
            p.min_count = k::kNormalLossMinValidCount;
            p.min_weight = k::kNormalLossMinValidWeight;
            const auto context = acquire_vulkan_context();
            std::vector<StorageRef> reads{ref(normal_map), ref(alpha), ref(target), ref(partials)};
            if (pixel_weight.is_valid())
                reads.push_back(ref(pixel_weight));
            const std::vector<StorageRef> outputs{ref(grad_normal), ref(loss), ref(partials)};
            launch(context, p, 6, reads, outputs, p.blocks);
            launch(context, p, 7, {ref(partials)}, outputs, 1);
            launch(context, p, 8, reads, outputs, p.blocks);
            launch(context, p, 9, {ref(partials)}, outputs, 1);
        }

        void depth_normal(In normal_map, In depth_map, In alpha, In pixel_weight, Out grad_normal, Out grad_depth,
                          Out grad_alpha, Out loss, Out partials, const Intrinsics& intrinsics, float weight, bool prior) {
            const uint32_t width = count32(depth_map.shape()[1]), height = count32(depth_map.shape()[0]);
            Push p = base(alpha, pixel_weight, loss, partials, width, height, k::normal_consistency_slots::kSlotCount,
                          k::normal_consistency_partial_count(static_cast<size_t>(width) * height), "normal consistency");
            p.normal = normal_map.is_valid() ? vk::address(ref(normal_map)) : 0;
            p.depth = vk::address(ref(depth_map));
            p.grad_normal = grad_normal.is_valid() ? vk::address(ref(grad_normal)) : 0;
            p.grad_depth = vk::address(ref(grad_depth));
            p.grad_alpha = vk::address(ref(grad_alpha));
            p.fx = intrinsics.fx;
            p.fy = intrinsics.fy;
            p.cx = intrinsics.cx;
            p.cy = intrinsics.cy;
            p.loss_weight = weight;
            p.prior = prior ? 1u : 0u;
            p.min_count = k::kNormalConsistencyMinValidCount;
            p.min_weight = k::kNormalConsistencyMinValidWeight;
            const auto context = acquire_vulkan_context();
            std::vector<StorageRef> reads{ref(depth_map), ref(alpha), ref(partials)};
            if (normal_map.is_valid())
                reads.push_back(ref(normal_map));
            if (pixel_weight.is_valid())
                reads.push_back(ref(pixel_weight));
            std::vector<StorageRef> outputs{ref(grad_depth), ref(grad_alpha), ref(loss), ref(partials)};
            if (grad_normal.is_valid())
                outputs.push_back(ref(grad_normal));
            launch(context, p, 10, reads, outputs, p.blocks);
            launch(context, p, 11, {ref(partials)}, outputs, 1);
            launch(context, p, 12, reads, outputs, p.blocks);
            launch(context, p, 13, {ref(partials)}, outputs, 1);
        }

        void consistency(In normal_map, In depth_map, In alpha, In pixel_weight, Out grad_normal, Out grad_depth,
                         Out grad_alpha, Out loss, Out partials, Intrinsics intrinsics, float weight) {
            depth_normal(normal_map, depth_map, alpha, pixel_weight, grad_normal, grad_depth, grad_alpha, loss, partials, intrinsics, weight, false);
        }
        void prior_depth(In prior_normal, In depth_map, In alpha, In pixel_weight, Out grad_depth, Out grad_alpha,
                         Out loss, Out partials, Intrinsics intrinsics, float weight) {
            Tensor absent_gradient;
            depth_normal(prior_normal, depth_map, alpha, pixel_weight, absent_gradient, grad_depth, grad_alpha, loss, partials, intrinsics, weight, true);
        }

        std::vector<AnchorSample> collect_anchor_samples(In points, In view, In prior, const AnchorParams& params) {
            const size_t count = points.shape()[0];
            if (!count)
                return {};
            LFS_ASSERT_MSG(count <= UINT32_MAX / 3, "Vulkan anchor point list exceeds 32-bit indexing");
            const size_t stride = std::max<size_t>(1, count / kMaxAnchorSamples), samples = (count + stride - 1) / stride;
            Tensor pairs = Tensor::empty({kMaxAnchorSamples, 2}, core::Device::GPU);
            Tensor sample_count = Tensor::zeros({1}, core::Device::GPU, core::DataType::Int32);
            Push p{};
            p.points = vk::address(ref(points));
            p.view = vk::address(ref(view));
            p.target = vk::address(ref(prior));
            p.pairs = vk::address(ref(pairs));
            p.sample_count = vk::address(ref(sample_count));
            p.lo_x = params.aabb_lo[0];
            p.lo_y = params.aabb_lo[1];
            p.lo_z = params.aabb_lo[2];
            p.hi_x = params.aabb_hi[0];
            p.hi_y = params.aabb_hi[1];
            p.hi_z = params.aabb_hi[2];
            p.width = count32(prior.shape()[1]);
            p.height = count32(prior.shape()[0]);
            p.samples = count32(samples);
            p.stride = count32(stride);
            p.capacity = kMaxAnchorSamples;
            p.fx = params.intrinsics.fx;
            p.fy = params.intrinsics.fy;
            p.cx = params.intrinsics.cx;
            p.cy = params.intrinsics.cy;
            p.near_plane = params.near_plane;
            const auto context = acquire_vulkan_context();
            std::vector<StorageRef> reads{ref(points), ref(view), ref(prior)}, writes{ref(pairs), ref(sample_count)};
            launch(context, p, 14, reads, writes, vk::dispatch_groups(*context, samples));
            const int found = std::min(sample_count.item<int>(), static_cast<int>(kMaxAnchorSamples));
            if (found < k::kMinAnchorSamples)
                return {};
            const Tensor host = pairs.slice(0, 0, static_cast<size_t>(found)).cpu().contiguous();
            std::vector<AnchorSample> result(static_cast<size_t>(found));
            std::memcpy(result.data(), host.data_ptr(), result.size() * sizeof(AnchorSample));
            return result;
        }
    } // namespace

    const lfs::gpu_ops::GeometryLossOps& vulkan_geometry_ops() {
        static const lfs::gpu_ops::GeometryLossOps ops{depth, normal, consistency, prior_depth, collect_anchor_samples};
        return ops;
    }
} // namespace lfs::training
