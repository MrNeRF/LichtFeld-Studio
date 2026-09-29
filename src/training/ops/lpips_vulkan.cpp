/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/lpips_vulkan.hpp"

#include "core/assert.hpp"
#include "core/tensor/backend/vulkan/vk_context.hpp"
#include "core/tensor/backend/vulkan/vk_ops_common.hpp"
#include "core/tensor/backend/vulkan/vk_recorder.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/vulkan_helpers.hpp"
#include "training_shader_table.hpp"

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace lfs::training {
    namespace {
        using namespace lfs::core;
        using namespace lfs::core::internal;
        using namespace lfs::gpu_ops;

        struct Push {
            uint64_t a, b, c, d, e, f, g;
            uint32_t count, channels, height, width, out_channels, out_height, out_width;
            uint32_t stride, padding, dilation, flags;
            float shift0, shift1, shift2, scale0, scale1, scale2, inverse_count;
        };
        static_assert(sizeof(Push) == 128);

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

        std::shared_ptr<Pipeline> pipeline_for(const std::shared_ptr<VulkanContext>& context, const uint32_t operation) {
            using namespace lfs::training::vulkan;
            static std::mutex mutex;
            static std::map<std::pair<uint64_t, uint32_t>, std::shared_ptr<Pipeline>> cache;
            const auto key = std::pair{context->context_id(), operation};
            std::lock_guard lock(mutex);
            if (const auto found = cache.find(key); found != cache.end())
                return found->second;
            const auto modules = embedded_training_shaders();
            const auto module = std::ranges::find(modules, std::string_view(operation == 6 ? "lpips_halo" : operation == 5 ? "lpips_coop"
                                                                                                                           : "lpips"),
                                                  &EmbeddedShader::name);
            LFS_ASSERT_MSG(module != modules.end(), "Vulkan LPIPS shader module is missing");
            VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            shader_info.codeSize = module->words.size_bytes();
            shader_info.pCode = module->words.data();
            VkShaderModule shader = VK_NULL_HANDLE;
            vk_check(context.get(), vkCreateShaderModule(context->device(), &shader_info, nullptr, &shader), "vkCreateShaderModule(training.lpips)");
            auto pipeline = std::make_shared<Pipeline>();
            pipeline->context = context;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
            VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layout_info.pushConstantRangeCount = 1;
            layout_info.pPushConstantRanges = &range;
            vk_check(context.get(), vkCreatePipelineLayout(context->device(), &layout_info, nullptr, &pipeline->layout), "vkCreatePipelineLayout(training.lpips)");
            const VkSpecializationMapEntry entry{0, 0, sizeof(operation)};
            const VkSpecializationInfo specialization{1, &entry, sizeof(operation), &operation};
            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shader;
            stage.pName = "main";
            stage.pSpecializationInfo = &specialization;
            VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipeline_info.stage = stage;
            pipeline_info.layout = pipeline->layout;
            vk_check(context.get(), vkCreateComputePipelines(context->device(), context->pipeline_cache(), 1, &pipeline_info, nullptr, &pipeline->handle), "vkCreateComputePipelines(training.lpips)");
            vkDestroyShaderModule(context->device(), shader, nullptr);
            cache.emplace(key, pipeline);
            release_at_shutdown(*context, mutex, cache);
            return pipeline;
        }

        uint32_t pack(const int low, const int high) {
            if (low < 0 || high < 0 || low > 65535 || high > 65535)
                throw std::invalid_argument("LPIPS convolution parameter is out of range");
            return uint32_t(low) | (uint32_t(high) << 16);
        }

        void dispatch(const std::shared_ptr<VulkanContext>& context, const uint32_t operation, const Push& push,
                      const std::span<const StorageRef> reads, const std::span<const StorageRef> writes, const size_t count) {
            if (count == 0)
                return;
            const auto pipeline = pipeline_for(context, operation);
            const size_t kItemsPerWorkgroup = operation >= 5 ? 512 : 256;
            const size_t max_items = static_cast<size_t>(context->caps().max_workgroup_count[0]) *
                                     kItemsPerWorkgroup;
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->handle);
                for (size_t base = 0; base < count; base += max_items) {
                    Push chunk = push;
                    chunk.g = base;
                    const size_t chunk_count = std::min(count - base, max_items);
                    vkCmdPushConstants(command, pipeline->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(chunk), &chunk);
                    vkCmdDispatch(command, static_cast<uint32_t>((chunk_count + kItemsPerWorkgroup - 1) / kItemsPerWorkgroup), 1, 1);
                    if (base + chunk_count < count) {
                        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                        dependency.memoryBarrierCount = 1;
                        dependency.pMemoryBarriers = &barrier;
                        vkCmdPipelineBarrier2(command, &dependency);
                    }
                } }, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_WHOLE_SIZE, pipeline);
        }

        void weight_taps(const Tensor& weight, Tensor& taps) {
            if (weight.ndim() != 4 || weight.size(2) != 3 || weight.size(3) != 3 || weight.dtype() != DataType::Float16)
                throw std::invalid_argument("LPIPS weight_taps expects fp16 3x3 weights");
            const auto context = acquire_vulkan_context();
            Push p{};
            p.a = vk::address(storage_ref(weight));
            p.d = vk::address(storage_ref(taps));
            p.count = vk::checked_u32(weight.numel(), "LPIPS tap count exceeds uint32");
            p.channels = vk::checked_u32(weight.size(1), "LPIPS input channels exceed uint32");
            p.out_channels = vk::checked_u32(weight.size(0), "LPIPS output channels exceed uint32");
            const std::array reads{storage_ref(weight)};
            const std::array writes{storage_ref(taps)};
            dispatch(context, 0, p, reads, writes, weight.numel());
        }

        void rgb_conv(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output, const RGBConvParams& params) {
            if (input.ndim() != 4 || input.size(1) != 3 || weight.ndim() != 4 || weight.size(1) != 3 || weight.size(2) != 3 || weight.size(3) != 3)
                throw std::invalid_argument("LPIPS RGB convolution expects NCHW RGB and 3x3 weights");
            const auto context = acquire_vulkan_context();
            Push p{};
            p.a = vk::address(storage_ref(input));
            p.b = vk::address(storage_ref(weight));
            p.c = vk::address(storage_ref(bias));
            p.d = vk::address(storage_ref(output));
            p.count = vk::checked_u32(output.numel(), "LPIPS RGB output exceeds uint32");
            p.channels = 3;
            p.height = vk::checked_u32(input.size(2), "LPIPS height exceeds uint32");
            p.width = vk::checked_u32(input.size(3), "LPIPS width exceeds uint32");
            p.out_channels = vk::checked_u32(weight.size(0), "LPIPS channels exceed uint32");
            p.out_height = vk::checked_u32(output.size(2), "LPIPS output height exceeds uint32");
            p.out_width = vk::checked_u32(output.size(3), "LPIPS output width exceeds uint32");
            p.stride = pack(1, 1);
            p.padding = pack(1, 1);
            p.dilation = pack(1, 1);
            p.flags = 1u | (params.official_scaling ? 1u << 16 : 0u);
            p.shift0 = params.shift[0];
            p.shift1 = params.shift[1];
            p.shift2 = params.shift[2];
            p.scale0 = params.scale[0];
            p.scale1 = params.scale[1];
            p.scale2 = params.scale[2];
            const std::array reads{storage_ref(input), storage_ref(weight), storage_ref(bias)};
            const std::array writes{storage_ref(output)};
            const auto& caps = context->caps();
            const bool cooperative = caps.cooperative_matrix && caps.shader_float16 &&
                                     caps.vulkan_memory_model && caps.subgroup_size == 32 &&
                                     caps.max_workgroup_invocations >= 512 && caps.max_workgroup_size[0] >= 512 &&
                                     caps.shared_memory_size >= 32768;
            const size_t work = ((size_t(p.out_height) * p.out_width + 63) / 64) *
                                ((size_t(p.out_channels) + 63) / 64) * input.size(0) * 512;
            dispatch(context, cooperative ? 5 : 1, p, reads, writes, cooperative ? work : output.numel());
        }

        void convolution(const Tensor& input, const Tensor& weight, const Tensor& taps, const Tensor& bias, Tensor& output, Tensor& scratch, const ConvParams& params) {
            (void)scratch;
            if (input.ndim() != 4 || weight.ndim() != 4 ||
                (input.dtype() != DataType::Float16 && input.dtype() != DataType::Float32) ||
                weight.dtype() != input.dtype() || output.dtype() != input.dtype() || weight.size(2) != 3 || weight.size(3) != 3)
                throw std::invalid_argument("LPIPS convolution expects matching fp16 or fp32 NCHW tensors and 3x3 weights");
            const auto context = acquire_vulkan_context();
            Push p{};
            p.a = vk::address(storage_ref(input));
            p.b = vk::address(storage_ref(weight));
            p.c = bias.is_valid() ? vk::address(storage_ref(bias)) : 0;
            p.d = vk::address(storage_ref(output));
            p.count = vk::checked_u32(output.numel(), "LPIPS convolution output exceeds uint32");
            p.channels = vk::checked_u32(input.size(1), "LPIPS channels exceed uint32");
            p.height = vk::checked_u32(input.size(2), "LPIPS height exceeds uint32");
            p.width = vk::checked_u32(input.size(3), "LPIPS width exceeds uint32");
            p.out_channels = vk::checked_u32(weight.size(0), "LPIPS output channels exceed uint32");
            p.out_height = vk::checked_u32(output.size(2), "LPIPS output height exceeds uint32");
            p.out_width = vk::checked_u32(output.size(3), "LPIPS output width exceeds uint32");
            p.stride = pack(params.stride_h, params.stride_w);
            p.padding = pack(params.pad_h, params.pad_w);
            p.dilation = pack(params.dilation_h, params.dilation_w);
            p.flags = uint32_t(params.activation) | (params.pad_mode == core::nn::ConvPadMode::Replicate ? 1u << 8 : 0u) |
                      (input.dtype() == DataType::Float16 ? 1u << 9 : 0u);
            std::vector<StorageRef> reads{storage_ref(input), storage_ref(weight)};
            const std::array writes{storage_ref(output)};
            const auto& caps = context->caps();
            const bool cooperative = input.dtype() == DataType::Float16 && caps.cooperative_matrix &&
                                     caps.shader_float16 && caps.vulkan_memory_model && caps.subgroup_size == 32 &&
                                     caps.max_workgroup_invocations >= 512 && caps.max_workgroup_size[0] >= 512 &&
                                     caps.shared_memory_size >= 32768;
            const bool halo = cooperative && params.stride_h == 1 && params.stride_w == 1 &&
                              params.dilation_h == 1 && params.dilation_w == 1 && p.channels % 32 == 0;
            if (halo && taps.is_valid() && taps.numel()) {
                if (taps.dtype() != DataType::Float16 || !taps.is_contiguous() ||
                    taps.ndim() != 3 || taps.size(0) != 9 ||
                    taps.size(1) != p.out_channels || taps.size(2) != p.channels)
                    throw std::invalid_argument("LPIPS cached taps must be contiguous fp16 tap-major weights");
                p.e = vk::address(storage_ref(taps));
                reads.push_back(storage_ref(taps));
            }
            const uint32_t operation = halo ? 6 : cooperative ? 5
                                                              : 4;
            const size_t tile_pixels = 64;
            const size_t tile_channels = cooperative ? 64 : 16;
            const size_t spatial_tiles = halo ? ((size_t(p.out_height) + 7) / 8) * ((size_t(p.out_width) + 7) / 8)
                                              : (size_t(p.out_height) * p.out_width + tile_pixels - 1) / tile_pixels;
            const size_t work = spatial_tiles *
                                ((size_t(p.out_channels) + tile_channels - 1) / tile_channels) * input.size(0) * (cooperative ? 512 : 256);
            if (bias.is_valid()) {
                reads.push_back(storage_ref(bias));
                dispatch(context, operation, p, reads, writes, work);
            } else
                dispatch(context, operation, p, reads, writes, work);
        }

        void pool_reduce(const Tensor& x, const Tensor& y, const Tensor& weight, Tensor& score, Tensor& pooled_x, Tensor& pooled_y, const PoolReduceParams& params) {
            if (x.ndim() != 4 || y.shape() != x.shape() || x.dtype() != DataType::Float16 || y.dtype() != DataType::Float16 || weight.dtype() != DataType::Float16 || x.size(1) % 8 != 0)
                throw std::invalid_argument("LPIPS pool_reduce expects matching fp16 NCHW tensors and channels divisible by eight");
            const auto context = acquire_vulkan_context();
            Push p{};
            p.a = vk::address(storage_ref(x));
            p.b = vk::address(storage_ref(y));
            p.c = vk::address(storage_ref(weight));
            p.d = vk::address(storage_ref(score));
            const bool do_pool = pooled_x.is_valid() && pooled_y.is_valid();
            p.e = do_pool ? vk::address(storage_ref(pooled_x)) : 0;
            p.f = do_pool ? vk::address(storage_ref(pooled_y)) : 0;
            p.count = vk::checked_u32(x.numel() / x.size(1), "LPIPS pool pixel count exceeds uint32");
            p.channels = vk::checked_u32(x.size(1), "LPIPS channels exceed uint32");
            p.height = vk::checked_u32(x.size(2), "LPIPS height exceeds uint32");
            p.width = vk::checked_u32(x.size(3), "LPIPS width exceeds uint32");
            p.stride = pack(params.y0, params.y1);
            p.padding = pack(params.x0, params.x1);
            p.inverse_count = params.inverse_count;
            size_t dispatch_count = p.count;
            if (do_pool)
                dispatch_count = std::max(dispatch_count, pooled_x.numel());
            std::vector<StorageRef> reads{storage_ref(x), storage_ref(y), storage_ref(weight), storage_ref(score)};
            std::vector<StorageRef> writes{storage_ref(score)};
            if (do_pool) {
                reads.push_back(storage_ref(pooled_x));
                reads.push_back(storage_ref(pooled_y));
                writes.push_back(storage_ref(pooled_x));
                writes.push_back(storage_ref(pooled_y));
            }
            dispatch(context, 3, p, reads, writes, dispatch_count);
        }
    } // namespace

    const gpu_ops::LpipsOps& vulkan_lpips_ops() {
        static const gpu_ops::LpipsOps ops{{.weight_taps = weight_taps, .rgb_conv = rgb_conv, .convolution = convolution, .pool_reduce = pool_reduce, .prefer_independent_queue = true}};
        return ops;
    }
} // namespace lfs::training
