/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/refine_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "training_shader_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <string_view>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;
        constexpr uint32_t kSplit = 0, kFill = 1, kCounts = 2, kSanitize = 3, kDivide = 4, kClip = 5, kOversize = 6, kDead = 7, kRotation = 8, kMedianHistogram = 9, kMedianSelect = 10;
        struct Push {
            uint64_t a, b, c, d, e, f, g, h, i, j, k, l;
            uint32_t count, count_a, count_b, count_c, count_d, rows, opacity_columns, reserved;
            float minimum_opacity, limit, median, unused;
        };
        static_assert(sizeof(Push) == 144 && offsetof(Push, count) == 96 && offsetof(Push, minimum_opacity) == 128);
        struct Pipeline {
            std::shared_ptr<VulkanContext> context;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkPipeline handle = VK_NULL_HANDLE;
            ~Pipeline() {
                if (!context || context->device() == VK_NULL_HANDLE)
                    return;
                if (handle)
                    vkDestroyPipeline(context->device(), handle, nullptr);
                if (layout)
                    vkDestroyPipelineLayout(context->device(), layout, nullptr);
            }
        };
        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, uint32_t op) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            auto key = std::pair{context->context_id(), op};
            std::lock_guard lock(mutex);
            if (auto it = cache.find(key); it != cache.end())
                return it->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("refine"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan Refine shader is missing");
            VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            si.codeSize = module->words.size_bytes();
            si.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &si, nullptr, &shader), "vkCreateShaderModule(training.refine)");
            auto result = std::make_shared<Pipeline>();
            result->context = context;
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &props);
            LFS_ASSERT_MSG(sizeof(Push) <= props.limits.maxPushConstantsSize, "Refine parameters exceed Vulkan push-constant limit");
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &li, nullptr, &result->layout), "vkCreatePipelineLayout(training.refine)");
            VkSpecializationMapEntry entry{0, 0, sizeof(op)};
            VkSpecializationInfo spec{1, &entry, sizeof(op), &op};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &spec;
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = stage;
            ci.layout = result->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &ci, nullptr, &result->handle), "vkCreateComputePipelines(training.refine)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, result);
            return result;
        }
        StorageRef ref(const Tensor& t) {
            auto s = storage_ref(t);
            LFS_ASSERT_MSG(s.backend == core::GpuBackend::Vulkan, "Vulkan Refine op received non-Vulkan storage");
            return s;
        }
        uint32_t n32(size_t n) {
            LFS_ASSERT_MSG(n <= UINT32_MAX, "Vulkan Refine count exceeds uint32 indexing");
            return static_cast<uint32_t>(n);
        }
        void launch(const Push& p, uint32_t op, std::span<const StorageRef> reads, std::span<const StorageRef> writes, uint32_t groups) {
            if (!groups)
                return;
            auto context = acquire_vulkan_context();
            auto pipeline = pipeline_for(context, op);
            LFS_ASSERT_MSG(groups <= context->caps().max_workgroup_count[0], "Vulkan Refine dispatch exceeds device limit");
            context->recorders().record(reads, writes, [&](VkCommandBuffer cmd) {vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline->handle);vkCmdPushConstants(cmd,pipeline->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);vkCmdDispatch(cmd,groups,1,1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }
        uint32_t groups(size_t n) { return static_cast<uint32_t>((n + 255) / 256); }
        void split(const RefineOutputs& parents, const RefineOutputs& children, In indices) {
            size_t n = indices.numel();
            if (!n)
                return;
            LFS_ASSERT_MSG(n <= static_cast<size_t>(INT_MAX), "Refine split exceeds INT_MAX");
            Push p{};
            p.a = vk::address(ref(parents.means));
            p.b = vk::address(ref(parents.rotations));
            p.c = vk::address(ref(parents.scales));
            p.d = vk::address(ref(parents.sh0));
            p.e = vk::address(ref(parents.opacity));
            p.f = vk::address(ref(children.means));
            p.g = vk::address(ref(children.rotations));
            p.h = vk::address(ref(children.scales));
            p.i = vk::address(ref(children.sh0));
            p.j = vk::address(ref(children.opacity));
            p.k = vk::address(ref(indices));
            p.count = n32(n);
            std::array reads{ref(parents.means), ref(parents.rotations), ref(parents.scales), ref(parents.sh0), ref(parents.opacity), ref(indices)};
            std::array writes{ref(parents.means), ref(parents.scales), ref(parents.opacity), ref(children.means), ref(children.rotations), ref(children.scales), ref(children.sh0), ref(children.opacity)};
            launch(p, kSplit, reads, writes, groups(n));
        }
        void fill_slots(In indices, const RefineInputs& source, const RefineOutputs& dst, Out free_mask) {
            size_t n = indices.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(indices));
            p.b = vk::address(ref(source.means));
            p.c = vk::address(ref(source.rotations));
            p.d = vk::address(ref(source.scales));
            p.e = vk::address(ref(source.sh0));
            p.f = vk::address(ref(source.opacity));
            p.g = vk::address(ref(dst.means));
            p.h = vk::address(ref(dst.rotations));
            p.i = vk::address(ref(dst.scales));
            p.j = vk::address(ref(dst.sh0));
            p.k = vk::address(ref(dst.opacity));
            p.l = free_mask.is_valid() ? vk::address(ref(free_mask)) : 0;
            p.count = n32(n);
            p.rows = n32(dst.means.shape()[0]);
            std::array reads{ref(indices), ref(source.means), ref(source.rotations), ref(source.scales), ref(source.sh0), ref(source.opacity)};
            std::vector<StorageRef> writes{ref(dst.means), ref(dst.rotations), ref(dst.scales), ref(dst.sh0), ref(dst.opacity)};
            if (free_mask.is_valid())
                writes.push_back(ref(free_mask));
            launch(p, kFill, reads, writes, groups(n));
        }
        uint32_t opt_count(In t) { return t.is_valid() ? n32(t.numel()) : 0; }
        void counts(In b0, In b1, In f0, In f1, Out out) {
            Push p{};
            p.a = b0.is_valid() ? vk::address(ref(b0)) : 0;
            p.b = b1.is_valid() ? vk::address(ref(b1)) : 0;
            p.c = f0.is_valid() ? vk::address(ref(f0)) : 0;
            p.d = f1.is_valid() ? vk::address(ref(f1)) : 0;
            p.e = vk::address(ref(out));
            p.count_a = opt_count(b0);
            p.count_b = opt_count(b1);
            p.count_c = opt_count(f0);
            p.count_d = opt_count(f1);
            std::vector<StorageRef> reads;
            if (b0.is_valid())
                reads.push_back(ref(b0));
            if (b1.is_valid())
                reads.push_back(ref(b1));
            if (f0.is_valid())
                reads.push_back(ref(f0));
            if (f1.is_valid())
                reads.push_back(ref(f1));
            const std::array writes{ref(out)};
            launch(p, kCounts, reads, writes, 1);
        }
        void normalize_positive_median(Out values) {
            size_t n = values.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(values));
            p.count = n32(n);
            auto storage = ref(values);
            std::array rw{storage};
            launch(p, kSanitize, rw, rw, groups(n));
            // Positive float bit patterns sort in numeric order. Select the
            // upper median one byte at a time, without downloading the values.
            auto selection = Tensor::zeros({259}, core::Device::GPU, core::DataType::UInt32);
            p.b = vk::address(ref(selection));
            const std::array selection_rw{ref(selection)};
            const std::array reads{storage, ref(selection)};
            for (uint32_t pass = 0; pass < 4; ++pass) {
                p.reserved = 24 - pass * 8;
                launch(p, kMedianHistogram, reads, selection_rw, std::min(groups(n), 256u));
                launch(p, kMedianSelect, selection_rw, selection_rw, 1);
            }
            launch(p, kDivide, reads, rw, groups(n));
        }
        void clip_scales(Out scales, In shares, In frozen, float limit) {
            size_t n = scales.shape()[0];
            if (!n || !(limit > 0.0f && limit < 1.0f))
                return;
            Push p{};
            p.a = frozen.is_valid() ? vk::address(ref(frozen)) : 0;
            p.b = vk::address(ref(scales));
            p.c = vk::address(ref(shares));
            p.count = n32(n);
            p.count_a = opt_count(frozen);
            p.limit = limit;
            std::vector<StorageRef> reads{ref(shares)};
            std::vector<StorageRef> writes{ref(scales)};
            if (frozen.is_valid())
                reads.push_back(ref(frozen));
            launch(p, kClip, reads, writes, groups(n));
        }
        void oversize_scores(In error, In shares, In frozen, Out scores, float limit) {
            size_t n = error.numel();
            if (!n)
                return;
            Push p{};
            p.a = frozen.is_valid() ? vk::address(ref(frozen)) : 0;
            p.b = vk::address(ref(error));
            p.c = vk::address(ref(shares));
            p.d = vk::address(ref(scores));
            p.count = n32(n);
            p.count_a = opt_count(frozen);
            p.limit = limit;
            std::vector<StorageRef> reads{ref(error), ref(shares)}, writes{ref(scores)};
            if (frozen.is_valid())
                reads.push_back(ref(frozen));
            launch(p, kOversize, reads, writes, groups(n));
        }
        void dead_mask(In opacity, In rotations, Out mask, float minimum) {
            size_t n = opacity.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(opacity));
            p.b = vk::address(ref(rotations));
            p.c = vk::address(ref(mask));
            p.count = n32(n);
            p.minimum_opacity = minimum;
            const std::array reads{ref(opacity), ref(rotations)};
            const std::array writes{ref(mask)};
            launch(p, kDead, reads, writes, groups(n));
        }
        void rotation_mask(In rotations, Out mask) {
            size_t n = rotations.shape()[0];
            if (!n)
                return;
            Push p{};
            p.b = vk::address(ref(rotations));
            p.c = vk::address(ref(mask));
            p.count = n32(n);
            const std::array reads{ref(rotations)};
            const std::array writes{ref(mask)};
            launch(p, kRotation, reads, writes, groups(n));
        }
        const RefineOps kOps{.split = split, .fill_slots = fill_slots, .counts = counts, .normalize_positive_median = normalize_positive_median, .clip_scales = clip_scales, .oversize_scores = oversize_scores, .dead_mask = dead_mask, .rotation_mask = rotation_mask};
    } // namespace
    const RefineOps& vulkan_refine_ops() { return kOps; }
} // namespace lfs::training
