/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/bilateral_vulkan.hpp"

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
            uint64_t grid, rgb, grad_output, offset, output, grad_grid, partials;
            uint32_t count;
            int32_t N, C, L, H, W, h, w, norm_n, partial_count;
            uint32_t chw, exposure_chroma, per_image;
            float grad_loss, lr, beta1, beta2, bc1_rcp, bc2_sqrt_rcp, eps;
            float scale1, scale2, spatial, inv_spatial;
        };
        static_assert(sizeof(Push) == 152);
        static_assert(offsetof(Push, count) == 56);
        static_assert(offsetof(Push, grad_loss) == 108);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, uint32_t operation) {
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), operation};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const auto modules = vulkan::embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view("bilateral"), &vulkan::EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan bilateral shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader),
                     "vkCreateShaderModule(training.bilateral)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(context->physical_device(), &properties);
            LFS_ASSERT_MSG(sizeof(Push) <= properties.limits.maxPushConstantsSize,
                           "Vulkan bilateral parameters exceed device push-constant limit");
            VkPushConstantRange range{};
            range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            range.size = sizeof(Push);
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout),
                     "vkCreatePipelineLayout(training.bilateral)");
            const VkSpecializationMapEntry entry{0, 0, sizeof(operation)};
            VkSpecializationInfo specialization{};
            specialization.mapEntryCount = 1;
            specialization.pMapEntries = &entry;
            specialization.dataSize = sizeof(operation);
            specialization.pData = &operation;
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle),
                     "vkCreateComputePipelines(training.bilateral)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            return pipeline;
        }

        StorageRef ref(const Tensor& tensor) {
            const StorageRef storage = storage_ref(tensor);
            LFS_ASSERT_MSG(storage.backend == core::GpuBackend::Vulkan,
                           "Vulkan bilateral op received storage from another backend");
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
        void append(std::vector<StorageRef>& refs, const Tensor& tensor) {
            if (tensor.is_valid() && tensor.numel())
                refs.push_back(ref(tensor));
        }
        uint32_t count32(size_t count, const char* what) {
            LFS_ASSERT_MSG(count <= UINT32_MAX, what);
            return static_cast<uint32_t>(count);
        }

        Push slice_params(In grid, In rgb, In grad_output, In offset, Out output, Out grad_grid,
                          const GridSliceParams& params) {
            const auto& s = grid.shape();
            LFS_ASSERT_MSG(s.rank() == 5 && rgb.ndim() == 3, "bilateral slice expects a [N,C,L,H,W] grid and 3D RGB image");
            const bool chw = params.layout == Layout::CHW;
            const int h = static_cast<int>(rgb.shape()[chw ? 1 : 0]);
            const int w = static_cast<int>(rgb.shape()[chw ? 2 : 1]);
            LFS_ASSERT_MSG(rgb.shape()[chw ? 0 : 2] == 3, "bilateral slice expects three RGB channels");
            const bool exposure_chroma = params.transform == GridTransform::ExposureChroma;
            LFS_ASSERT_MSG(s[1] >= (exposure_chroma ? 9u : 12u),
                           exposure_chroma ? "bilateral exposure-chroma slice requires 9 grid channels"
                                           : "bilateral affine slice requires 12 grid channels");
            const StorageRef g = ref(grid), image = ref(rgb), out = ref(output);
            const StorageRef grad = grad_output.is_valid() ? ref(grad_output) : StorageRef{};
            const StorageRef off = offset.is_valid() ? ref(offset) : StorageRef{};
            const StorageRef gg = grad_grid.is_valid() ? ref(grad_grid) : StorageRef{};
            Push p{};
            p.grid = vk::address(g);
            p.rgb = vk::address(image);
            p.grad_output = grad_output.is_valid() ? vk::address(grad) : 0;
            p.offset = offset.is_valid() ? vk::address(off) : 0;
            p.output = vk::address(out);
            p.grad_grid = grad_grid.is_valid() ? vk::address(gg) : 0;
            p.N = static_cast<int32_t>(s[0]);
            p.C = static_cast<int32_t>(s[1]);
            p.L = static_cast<int32_t>(s[2]);
            p.H = static_cast<int32_t>(s[3]);
            p.W = static_cast<int32_t>(s[4]);
            p.h = h;
            p.w = w;
            p.chw = chw ? 1u : 0u;
            p.exposure_chroma = exposure_chroma ? 1u : 0u;
            return p;
        }

        void slice_forward(In grid, In rgb, In offset, Out output, const GridSliceParams& params) {
            Tensor no_gradient, no_grid_gradient;
            Push p = slice_params(grid, rgb, no_gradient, offset, output, no_grid_gradient, params);
            const uint32_t pixels = count32(static_cast<size_t>(p.h) * p.w, "bilateral image exceeds 32-bit indexing");
            p.count = pixels;
            std::vector<StorageRef> reads{ref(grid), ref(rgb)};
            append(reads, offset);
            launch(acquire_vulkan_context(), p, 0, reads, {ref(output)}, vk::dispatch_groups(*acquire_vulkan_context(), pixels));
        }

        void slice_backward(In grid, In rgb, In grad_output, In offset, Out grad_grid, Out grad_rgb,
                            const GridSliceParams& params) {
            Push p = slice_params(grid, rgb, grad_output, offset, grad_rgb, grad_grid, params);
            const uint32_t pixels = count32(static_cast<size_t>(p.h) * p.w, "bilateral image exceeds 32-bit indexing");
            p.count = pixels;
            std::vector<StorageRef> reads{ref(grid), ref(rgb), ref(grad_output)};
            append(reads, offset);
            append(reads, grad_grid);
            launch(acquire_vulkan_context(), p, 2, reads, {ref(grad_rgb), ref(grad_grid)}, vk::dispatch_groups(*acquire_vulkan_context(), pixels));
        }

        void tv_forward(In grids, Out loss, Out reduction_temp, int norm_n) {
            const auto& s = grids.shape();
            LFS_ASSERT_MSG(s.rank() == 5, "bilateral TV expects a [N,C,L,H,W] grid");
            const size_t cells = s[0] * s[2] * s[3] * s[4];
            const uint32_t groups = static_cast<uint32_t>(std::min((cells + 255) / 256, size_t{2048}));
            LFS_ASSERT_MSG(reduction_temp.numel() >= groups, "bilateral TV reduction buffer is too small");
            Push p{};
            p.grid = vk::address(ref(grids));
            p.partials = vk::address(ref(reduction_temp));
            p.output = vk::address(ref(loss));
            p.N = s[0];
            p.C = s[1];
            p.L = s[2];
            p.H = s[3];
            p.W = s[4];
            p.norm_n = norm_n > 0 ? norm_n : p.N;
            p.partial_count = groups;
            p.count = count32(cells, "bilateral TV grid too large");
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 3, {ref(grids)}, {ref(reduction_temp)}, groups);
            launch(ctx, p, 4, {ref(reduction_temp)}, {ref(loss)}, 1);
        }
        void tv_backward(In grids, Out gradients, float grad_loss, int norm_n) {
            const auto& s = grids.shape();
            LFS_ASSERT_MSG(s.rank() == 5, "bilateral TV expects a [N,C,L,H,W] grid");
            const size_t cells = s[0] * s[2] * s[3] * s[4];
            Push p{};
            p.grid = vk::address(ref(grids));
            p.output = vk::address(ref(gradients));
            p.N = s[0];
            p.C = s[1];
            p.L = s[2];
            p.H = s[3];
            p.W = s[4];
            p.norm_n = norm_n > 0 ? norm_n : p.N;
            p.count = count32(cells, "bilateral TV grid too large");
            p.grad_loss = grad_loss;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 5, {ref(grids), ref(gradients)}, {ref(gradients)}, vk::dispatch_groups(*ctx, cells));
        }
        void project_mean(Out grids, In mean, In identity, int per_image) {
            const auto& s = grids.shape();
            LFS_ASSERT_MSG(s.rank() == 5, "bilateral projection expects [N,C,L,H,W]");
            Push p{};
            p.output = vk::address(ref(grids));
            p.partials = vk::address(ref(mean));
            p.offset = vk::address(ref(identity));
            p.N = s[0];
            p.C = s[1];
            p.L = s[2];
            p.H = s[3];
            p.W = s[4];
            p.per_image = per_image != 0;
            p.count = count32(grids.numel(), "bilateral grid too large");
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 6, {ref(grids), ref(mean), ref(identity)}, {ref(grids)}, vk::dispatch_groups(*ctx, p.count));
        }
        void update_offset(Out channel_sum, Out shared_offset, In identity, In old_mean, In new_mean,
                           float spatial, float inv_n_spatial) {
            Push p{};
            p.grid = vk::address(ref(channel_sum));
            p.offset = vk::address(ref(shared_offset));
            p.grad_output = vk::address(ref(identity));
            p.rgb = vk::address(ref(old_mean));
            p.partials = vk::address(ref(new_mean));
            p.count = count32(channel_sum.numel(), "bilateral channels too large");
            p.spatial = spatial;
            p.inv_spatial = inv_n_spatial;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 7, {ref(channel_sum), ref(shared_offset), ref(identity), ref(old_mean), ref(new_mean)}, {ref(channel_sum), ref(shared_offset)}, vk::dispatch_groups(*ctx, p.count));
        }
        void adam(Out grid, Out moment1, Out moment2, In gradient, const AdamUpdateParams& a) {
            Push p{};
            p.grid = vk::address(ref(grid));
            p.rgb = vk::address(ref(moment1));
            p.offset = vk::address(ref(moment2));
            p.grad_output = vk::address(ref(gradient));
            p.count = count32(grid.numel(), "bilateral Adam tensor too large");
            p.lr = a.lr;
            p.beta1 = a.beta1;
            p.beta2 = a.beta2;
            p.bc1_rcp = a.bc1_rcp;
            p.bc2_sqrt_rcp = a.bc2_sqrt_rcp;
            p.eps = a.eps;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 8, {ref(grid), ref(moment1), ref(moment2), ref(gradient)}, {ref(grid), ref(moment1), ref(moment2)}, vk::dispatch_groups(*ctx, p.count));
        }
        void scale_moments(Out moment1, Out moment2, float scale1, float scale2) {
            if (!moment1.numel())
                return;
            Push p{};
            p.grid = vk::address(ref(moment1));
            p.rgb = vk::address(ref(moment2));
            p.count = count32(moment1.numel(), "bilateral moments too large");
            p.scale1 = scale1;
            p.scale2 = scale2;
            const auto ctx = acquire_vulkan_context();
            launch(ctx, p, 9, {ref(moment1), ref(moment2)}, {ref(moment1), ref(moment2)}, vk::dispatch_groups(*ctx, p.count));
        }
        void copy_slice(Out source, Out destination, size_t source_offset, size_t destination_offset, size_t elements) {
            LFS_ASSERT_MSG(source_offset <= source.numel() && elements <= source.numel() - source_offset &&
                               destination_offset <= destination.numel() && elements <= destination.numel() - destination_offset,
                           "bilateral transfer slice exceeds tensor bounds");
            destination.flatten().slice(0, destination_offset, destination_offset + elements).copy_from(source.flatten().slice(0, source_offset, source_offset + elements));
        }
        void upload_slice(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements) { copy_slice(host, device, host_offset, device_offset, elements); }
        void download_slice(Out host, Out device, size_t host_offset, size_t device_offset, size_t elements) { copy_slice(device, host, device_offset, host_offset, elements); }

        const BilateralOps kVulkanBilateralOps{.slice_forward = slice_forward, .slice_backward = slice_backward, .tv_forward = tv_forward, .tv_backward = tv_backward, .project_mean = project_mean, .update_offset = update_offset, .adam = adam, .scale_moments = scale_moments, .upload_slice = upload_slice, .download_slice = download_slice};
    } // namespace
    const gpu_ops::BilateralOps& vulkan_bilateral_ops() { return kVulkanBilateralOps; }
} // namespace lfs::training
