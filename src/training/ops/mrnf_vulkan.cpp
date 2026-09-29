/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mrnf_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_memory.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "lfs/training/ops/pair_sort_vulkan.hpp"
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
        constexpr uint32_t kNoise = 0, kDecay = 1, kFold = 2, kFoldError = 3, kProject = 4, kGatherCenter = 5, kFar = 6, kMae = 7, kSeed = 8, kGatherSeeds = 9, kStarvation = 10, kPrune = 11, kParent = 12, kGumbel = 13, kGatherIndices = 14, kGeomean = 15;
        struct Push {
            uint64_t a, b, c, d, e, f, g, h, i, j, k, l, seed;
            uint32_t count, count_a, count_b, count_c, count_d, width, height, channels;
            float x, y, z, w, u0, u1, scalar0, scalar1;
        };
        static_assert(sizeof(Push) == 168 && offsetof(Push, seed) == 96 && offsetof(Push, count) == 104 && offsetof(Push, x) == 136);
        struct Pipeline {
            std::shared_ptr<VulkanContext> context;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            VkPipeline handle = VK_NULL_HANDLE;
            ~Pipeline() {
                if (!context || !context->device())
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
            const auto module = std::ranges::find(modules, std::string_view("mrnf"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan MRNF shader is missing");
            VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            si.codeSize = module->words.size_bytes();
            si.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &si, nullptr, &shader), "vkCreateShaderModule(training.mrnf)");
            auto out = std::make_shared<Pipeline>();
            out->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize, "MRNF parameters exceed Vulkan push-constant limit");
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &li, nullptr, &out->layout), "vkCreatePipelineLayout(training.mrnf)");
            VkSpecializationMapEntry entry{0, 0, sizeof(op)};
            VkSpecializationInfo spec{1, &entry, sizeof(op), &op};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &spec;
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = stage;
            ci.layout = out->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &ci, nullptr, &out->handle), "vkCreateComputePipelines(training.mrnf)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, out);
            return out;
        }
        StorageRef ref(const Tensor& t) {
            auto s = storage_ref(t);
            LFS_ASSERT_MSG(s.backend == core::GpuBackend::Vulkan, "Vulkan MRNF op received non-Vulkan storage");
            return s;
        }
        uint32_t n32(size_t n) {
            LFS_ASSERT_MSG(n <= UINT32_MAX, "Vulkan MRNF count exceeds uint32 indexing");
            return static_cast<uint32_t>(n);
        }
        uint32_t groups(size_t n) { return static_cast<uint32_t>((n + 255) / 256); }
        void launch(const Push& p, uint32_t op, std::span<const StorageRef> reads, std::span<const StorageRef> writes, uint32_t group_count) {
            if (!group_count)
                return;
            auto context = acquire_vulkan_context();
            LFS_ASSERT_MSG(group_count <= context->caps().max_workgroup_count[0], "Vulkan MRNF dispatch exceeds device limit");
            auto pipeline = pipeline_for(context, op);
            context->recorders().record(reads, writes, [&](VkCommandBuffer cmd) {vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline->handle);vkCmdPushConstants(cmd,pipeline->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);vkCmdDispatch(cmd,group_count,1,1); }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }
        uint32_t opt_count(In t) { return t.is_valid() ? n32(t.numel()) : 0; }
        std::vector<float> read_floats(In t) {
            auto cpu = t.cpu().contiguous();
            const auto* p = cpu.ptr<float>();
            return {p, p + cpu.numel()};
        }
        bool finite(float x) { return std::isfinite(x); }
        void noise(Out means, In opacity, In visibility, In frozen, const MrnfNoiseParams& args) {
            size_t n = means.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(means));
            p.b = vk::address(ref(opacity));
            p.c = vk::address(ref(visibility));
            p.d = frozen.is_valid() ? vk::address(ref(frozen)) : 0;
            p.seed = args.seed;
            p.count = n32(n);
            p.count_a = opt_count(frozen);
            p.x = args.lr_mean;
            p.y = args.noise_weight;
            p.z = args.median_scale;
            std::vector<StorageRef> reads{ref(opacity), ref(visibility)}, writes{ref(means)};
            if (frozen.is_valid())
                reads.push_back(ref(frozen));
            launch(p, kNoise, reads, writes, groups(n));
        }
        void decay(Out opacity, Out scales, In frozen, In far, const DecayParams& args) {
            size_t n = scales.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(opacity));
            p.b = vk::address(ref(scales));
            p.c = frozen.is_valid() ? vk::address(ref(frozen)) : 0;
            p.d = far.is_valid() ? vk::address(ref(far)) : 0;
            p.count = n32(n);
            p.count_a = opt_count(frozen);
            p.count_b = opt_count(far);
            p.x = args.opacity_decay;
            p.y = args.scale_decay;
            p.z = args.far_decay_scale;
            p.w = args.train_t;
            std::vector<StorageRef> rw{ref(opacity), ref(scales)}, reads;
            if (frozen.is_valid())
                reads.push_back(ref(frozen));
            if (far.is_valid())
                reads.push_back(ref(far));
            reads.insert(reads.end(), rw.begin(), rw.end());
            launch(p, kDecay, reads, rw, groups(n));
        }
        Bounds percentile_bounds(In means, float percentile) {
            size_t n = means.shape()[0];
            LFS_ASSERT_MSG(n > 0 && n <= static_cast<size_t>(INT_MAX), "MRNF bounds require 1..INT_MAX rows");
            LFS_ASSERT_MSG(finite(percentile) && percentile >= 0.0f && percentile <= 1.0f, "MRNF percentile must be finite and within [0,1]");
            auto values = read_floats(means);
            Bounds out{};
            size_t low = static_cast<size_t>(std::floor((1.0f - percentile) * 0.5f * float(n - 1))), high = static_cast<size_t>(std::floor((1.0f - (1.0f - percentile) * 0.5f) * float(n - 1)));
            std::array<float, 3> ext{};
            for (size_t axis = 0; axis < 3; ++axis) {
                std::vector<float> v(n);
                for (size_t row = 0; row < n; ++row)
                    v[row] = values[row * 3 + axis];
                std::sort(v.begin(), v.end());
                out.center[axis] = (v[low] + v[high]) * 0.5f;
                out.extent[axis] = (v[high] - v[low]) * 0.5f;
                ext[axis] = out.extent[axis];
            }
            std::sort(ext.begin(), ext.end());
            out.median_size = ext[1] * 2.0f;
            out.max_extent = ext[2];
            return out;
        }
        ScalarValidity median_extent(In scales) {
            if (!scales.is_valid() || !scales.numel())
                return {};
            size_t n = scales.shape()[0];
            Tensor ext = Tensor::empty({n}, core::Device::GPU, core::DataType::Float32);
            Push p{};
            p.a = vk::address(ref(scales));
            p.b = vk::address(ref(ext));
            p.count = n32(n);
            const std::array reads{ref(scales)};
            const std::array writes{ref(ext)};
            launch(p, kGeomean, reads, writes, groups(n));
            auto values = read_floats(ext);
            values.erase(std::remove_if(values.begin(), values.end(), [](float x) { return !std::isfinite(x) || !(x > 0.0f); }), values.end());
            if (values.empty())
                return {};
            std::sort(values.begin(), values.end());
            float median = values[values.size() / 2];
            return {median, std::isfinite(median) && median > 0.0f};
        }
        void gumbel(GumbelTopKScratch*, In weights, Out indices, const GumbelParams& args) {
            size_t n = weights.numel(), k = indices.numel();
            LFS_ASSERT_MSG(k <= n, "MRNF Gumbel top-k requires k <= n");
            if (!k)
                return;
            LFS_ASSERT_MSG(n <= static_cast<size_t>(INT_MAX), "MRNF Gumbel count exceeds INT_MAX");
            if (args.compact_sparse)
                LFS_ASSERT_MSG(args.known_nnz <= n, "MRNF known_nnz exceeds item count");
            Tensor sources;
            size_t active = n;
            if (args.compact_sparse) {
                Tensor positive = weights > 0.0f;
                active = args.known_nnz ? args.known_nnz : positive.count_nonzero();
                if (active != n)
                    sources = positive.nonzero().reshape({static_cast<int>(active)});
                LFS_ASSERT_MSG(args.known_nnz == 0 || args.known_nnz == active, "MRNF known_nnz differs from positive-weight count");
            }
            bool compact = args.compact_sparse && active >= k && active < n;
            size_t sort_n = compact ? active : n;
            if (k == n) {
                Push p{};
                p.c = vk::address(ref(indices));
                p.count = n32(k);
                const std::array writes{ref(indices)};
                launch(p, kGatherIndices, {}, writes, groups(k));
                return;
            }
            if (compact)
                sources = sources.reshape({static_cast<int>(sort_n)});
            Tensor keyA = Tensor::empty({sort_n}, core::Device::GPU, core::DataType::UInt32), keyB = Tensor::empty({sort_n}, core::Device::GPU, core::DataType::UInt32), valueA = Tensor::empty({sort_n}, core::Device::GPU, core::DataType::UInt32), valueB = Tensor::empty({sort_n}, core::Device::GPU, core::DataType::UInt32);
            Push p{};
            p.a = vk::address(ref(weights));
            p.b = compact ? vk::address(ref(sources)) : 0;
            p.c = vk::address(ref(keyA));
            p.d = vk::address(ref(valueA));
            p.seed = args.seed;
            p.count = n32(sort_n);
            p.count_a = n32(n);
            p.count_b = compact ? 1u : 0u;
            std::vector<StorageRef> reads{ref(weights)}, writes{ref(keyA), ref(valueA)};
            if (compact)
                reads.push_back(ref(sources));
            launch(p, kGumbel, reads, writes, groups(sort_n));
            const bool in_a = vulkan_pair_sort({&keyA, &keyB, &valueA, &valueB}, n32(sort_n), 0, 32, false);
            Tensor& sorted = in_a ? valueA : valueB;
            p = {};
            p.a = vk::address(ref(sorted));
            p.b = compact ? vk::address(ref(sources)) : 0;
            p.c = vk::address(ref(indices));
            p.count = n32(k);
            std::vector<StorageRef> gather_reads{ref(sorted)}, gather_writes{ref(indices)};
            if (compact)
                gather_reads.push_back(ref(sources));
            launch(p, kGatherIndices, gather_reads, gather_writes, groups(k));
        }
        void fold(Out vis, Out maxw, Out dens, Out ratio, float power) {
            size_t n = vis.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(vis));
            p.b = vk::address(ref(maxw));
            p.c = vk::address(ref(dens));
            p.d = ratio.is_valid() ? vk::address(ref(ratio)) : 0;
            p.count = n32(n);
            p.y = power;
            std::vector<StorageRef> reads{ref(vis), ref(maxw), ref(dens)}, writes{ref(vis), ref(maxw), ref(dens)};
            if (ratio.is_valid()) {
                reads.push_back(ref(ratio));
                writes.push_back(ref(ratio));
            }
            launch(p, kFold, reads, writes, groups(n));
        }
        void fold_error(Out maxw, Out dens) {
            size_t n = maxw.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(maxw));
            p.b = vk::address(ref(dens));
            p.count = n32(n);
            const std::array reads{ref(maxw), ref(dens)};
            const std::array writes{ref(maxw), ref(dens)};
            launch(p, kFoldError, reads, writes, groups(n));
        }
        void project(In means, In view, Out xy, Out radii, const ProjectParams& args) {
            size_t n = means.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(means));
            p.b = vk::address(ref(view));
            p.c = vk::address(ref(xy));
            p.d = vk::address(ref(radii));
            p.count = n32(n);
            p.width = n32(args.image.w);
            p.height = n32(args.image.h);
            p.x = args.near_plane;
            p.y = args.intrinsics.fx;
            p.z = args.intrinsics.fy;
            p.u0 = args.intrinsics.cx;
            p.u1 = args.intrinsics.cy;
            const std::array reads{ref(means), ref(view)};
            const std::array writes{ref(xy), ref(radii)};
            launch(p, kProject, reads, writes, groups(n));
        }
        void gather_center(In xy, In radii, In error, Out scores) {
            size_t n = xy.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(xy));
            p.b = vk::address(ref(radii));
            p.c = vk::address(ref(error));
            p.d = vk::address(ref(scores));
            p.count = n32(n);
            p.height = n32(error.shape()[0]);
            p.width = n32(error.shape()[1]);
            const std::array reads{ref(xy), ref(radii), ref(error)};
            const std::array writes{ref(scores)};
            launch(p, kGatherCenter, reads, writes, groups(n));
        }
        void far_mask(In means, Out mask, std::array<float, 3> center, float radius) {
            size_t n = means.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(means));
            p.b = vk::address(ref(mask));
            p.count = n32(n);
            p.x = center[0];
            p.y = center[1];
            p.z = center[2];
            p.w = radius * radius;
            const std::array reads{ref(means)};
            const std::array writes{ref(mask)};
            launch(p, kFar, reads, writes, groups(n));
        }
        void mean_abs_error(In pred, In target, Out error) {
            size_t c = pred.shape()[0], pixels = pred.shape()[1] * pred.shape()[2];
            if (!c || !pixels)
                return;
            Push p{};
            p.a = vk::address(ref(pred));
            p.b = vk::address(ref(target));
            p.c = vk::address(ref(error));
            p.count = n32(pixels);
            p.channels = n32(c);
            const std::array reads{ref(pred), ref(target)};
            const std::array writes{ref(error)};
            launch(p, kMae, reads, writes, groups(pixels));
        }
        void seed_weights(In error, In alpha, Out weights) {
            size_t n = weights.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(error));
            p.b = vk::address(ref(alpha));
            p.c = vk::address(ref(weights));
            p.count = n32(n);
            const std::array reads{ref(error), ref(alpha)};
            const std::array writes{ref(weights)};
            launch(p, kSeed, reads, writes, groups(n));
        }
        void gather_seeds(In indices, In target, In alpha, In depth, Out rgb, Out sampled_alpha, Out sampled_depth) {
            size_t n = indices.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(indices));
            p.b = vk::address(ref(target));
            p.c = vk::address(ref(alpha));
            p.d = depth.is_valid() ? vk::address(ref(depth)) : 0;
            p.e = vk::address(ref(rgb));
            p.f = vk::address(ref(sampled_alpha));
            p.g = vk::address(ref(sampled_depth));
            p.count = n32(n);
            p.count_a = n32(alpha.numel());
            p.channels = n32(target.shape()[0]);
            std::vector<StorageRef> reads{ref(indices), ref(target), ref(alpha)}, writes{ref(rgb), ref(sampled_alpha), ref(sampled_depth)};
            if (depth.is_valid())
                reads.push_back(ref(depth));
            launch(p, kGatherSeeds, reads, writes, groups(n));
        }
        float sorted_median(In values) {
            if (!values.is_valid() || !values.numel())
                return 0.0f;
            auto v = read_floats(values);
            std::sort(v.begin(), v.end(), [](float a, float b) {if(std::isnan(a))return false;if(std::isnan(b))return true;return a<b; });
            float median = v[v.size() / 2];
            return std::isfinite(median) ? median : 0.0f;
        }
        void starvation(Out weights, In visibility, float median) {
            size_t n = weights.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(weights));
            p.b = vk::address(ref(visibility));
            p.count = n32(n);
            p.x = median;
            const std::array reads{ref(weights), ref(visibility)};
            const std::array writes{ref(weights)};
            launch(p, kStarvation, reads, writes, groups(n));
        }
        size_t compact_bool(In mask, Out indices, size_t count) {
            if (!mask.numel() || !count)
                return 0;
            Tensor found = mask.nonzero();
            size_t total = found.shape()[0], written = std::min({total, count, indices.numel()});
            if (written)
                indices.slice(0, 0, written).copy_from(found.reshape({static_cast<int>(total)}).slice(0, 0, written));
            return total;
        }
        void prune(In means, In scale, Out mask, std::array<float, 3> center, float maximum, float logmax) {
            size_t n = means.shape()[0];
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(means));
            p.b = vk::address(ref(scale));
            p.c = vk::address(ref(mask));
            p.count = n32(n);
            p.x = center[0];
            p.y = center[1];
            p.z = center[2];
            p.w = maximum;
            p.u0 = logmax;
            const std::array reads{ref(means), ref(scale), ref(mask)};
            const std::array writes{ref(mask)};
            launch(p, kPrune, reads, writes, groups(n));
        }
        void parent_weights(In opacity, In vis, In active, In trainable, In edge, Out weights) {
            size_t n = opacity.numel();
            if (!n)
                return;
            Push p{};
            p.a = vk::address(ref(opacity));
            p.b = vk::address(ref(vis));
            p.c = active.is_valid() ? vk::address(ref(active)) : 0;
            p.d = trainable.is_valid() ? vk::address(ref(trainable)) : 0;
            p.e = edge.is_valid() ? vk::address(ref(edge)) : 0;
            p.f = vk::address(ref(weights));
            p.count = n32(n);
            p.count_a = opt_count(active);
            p.count_b = opt_count(trainable);
            std::vector<StorageRef> reads{ref(opacity), ref(vis)}, writes{ref(weights)};
            if (active.is_valid())
                reads.push_back(ref(active));
            if (trainable.is_valid())
                reads.push_back(ref(trainable));
            if (edge.is_valid())
                reads.push_back(ref(edge));
            launch(p, kParent, reads, writes, groups(n));
        }
        const MrnfOps kOps{.noise = noise, .decay = decay, .percentile_bounds = percentile_bounds, .median_extent = median_extent, .gumbel = gumbel, .fold = fold, .fold_error = fold_error, .project_centers = project, .gather_center_error = gather_center, .far_mask = far_mask, .mean_abs_error = mean_abs_error, .seed_weights = seed_weights, .gather_seeds = gather_seeds, .sorted_median = sorted_median, .starvation_weights = starvation, .compact_bool_indices = compact_bool, .prune_bounds = prune, .replace_parent_weights = parent_weights};
    } // namespace
    const MrnfOps& vulkan_mrnf_ops() { return kOps; }
} // namespace lfs::training
