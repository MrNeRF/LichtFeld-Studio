/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {
    namespace {
        struct Push {
            uint64_t input, output;
            float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
            int sw, sh, dw, dh;
            float distortion[12];
            int model, num_distortion, channels, padding;
        };
        static_assert(sizeof(Push) == 128);
        Tensor dispatch(const Tensor& input, Push p, bool mask, uint32_t phase) {
            const GpuBackendScope scope(GpuBackend::Vulkan);
            auto output = Tensor::empty(mask ? TensorShape{size_t(p.dh), size_t(p.dw)} : TensorShape{size_t(p.channels), size_t(p.dh), size_t(p.dw)}, Device::GPU);
            const auto source = storage_ref(input), destination = storage_ref(output);
            p.input = vk::address(source);
            p.output = vk::address(destination);
            const auto context = acquire_vulkan_context();
            const std::array constants{phase};
            const auto& pipeline = context->pipelines().specialized("image_resample", sizeof(p), constants);
            const std::array reads{source}, writes{destination};
            context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(command, vk::dispatch_groups(*context, size_t(p.dw) * p.dh), 1, 1);
            });
            return output;
        }
    } // namespace
    Tensor VulkanBackendOps::image_undistort(const Tensor& input, const UndistortParams& p, bool mask, ExecContext) {
        LFS_FACADE_TRACE(image_undistort);
        Push push{.src_fx = p.src_fx, .src_fy = p.src_fy, .src_cx = p.src_cx, .src_cy = p.src_cy, .dst_fx = p.dst_fx, .dst_fy = p.dst_fy, .dst_cx = p.dst_cx, .dst_cy = p.dst_cy, .sw = p.src_width, .sh = p.src_height, .dw = p.dst_width, .dh = p.dst_height, .model = int(p.model_type), .num_distortion = p.num_distortion, .channels = mask ? 1 : int(input.size(0))};
        std::copy_n(p.distortion, 12, push.distortion);
        return dispatch(input, push, mask, 0);
    }
    Tensor VulkanBackendOps::image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext) {
        LFS_FACADE_TRACE(image_resize_prior);
        Push push{.sw = int(input.size(input.ndim() - 1)), .sh = int(input.size(input.ndim() - 2)), .dw = width, .dh = height, .channels = normal ? 3 : 1};
        return dispatch(input, push, !normal, normal ? 2 : 1);
    }
} // namespace lfs::core::internal
