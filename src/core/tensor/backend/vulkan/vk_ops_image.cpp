/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"
#include <cmath>

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
    Tensor VulkanBackendOps::image_warp(const Tensor& input, const UndistortParams& p, int mode, bool inverse, Tensor* validity, ExecContext) {
        const GpuBackendScope scope(GpuBackend::Vulkan);
        const int width = inverse ? p.src_width : p.dst_width;
        const int height = inverse ? p.src_height : p.dst_height;
        const int channels = input.ndim() == 2 ? 1 : int(input.size(0));
        auto output = Tensor::zeros(mode == 4 ? TensorShape{size_t(height), size_t(width), 2} : input.ndim() == 2 ? TensorShape{size_t(height), size_t(width)}
                                                                                                                  : TensorShape{size_t(channels), size_t(height), size_t(width)},
                                    Device::GPU);
        Tensor mask;
        if (validity) {
            // Use 32-bit stores so the warp does not require shaderInt8.
            mask = Tensor::zeros({size_t(height), size_t(width)}, Device::GPU, DataType::Int32);
        }
        struct CameraParams {
            float src_fx, src_fy, src_cx, src_cy, dst_fx, dst_fy, dst_cx, dst_cy;
            int src_width, src_height, dst_width, dst_height, model_type;
            float distortion[12];
            int num_distortion;
        } camera{p.src_fx, p.src_fy, p.src_cx, p.src_cy, p.dst_fx, p.dst_fy, p.dst_cx, p.dst_cy, p.src_width, p.src_height, p.dst_width, p.dst_height, int(p.model_type), {}, p.num_distortion};
        std::copy_n(p.distortion, 12, camera.distortion);
        const auto parameters = Tensor::from_blob(&camera, {sizeof(camera)}, Device::CPU, DataType::UInt8).to(Device::GPU);
        const auto source = storage_ref(input), destination = storage_ref(output), camera_storage = storage_ref(parameters);
        struct WarpPush {
            uint64_t input, output, validity, camera;
            int width, height, channels, mode, inverse, quadrature;
        } push{
            vk::address(source), vk::address(destination), mask.is_valid() ? vk::address(storage_ref(mask)) : 0,
            vk::address(camera_storage), width, height, channels, mode, inverse,
            std::max(8, int(std::ceil(std::max(p.src_fx / p.dst_fx, p.src_fy / p.dst_fy))))};
        const auto context = acquire_vulkan_context();
        const auto& pipeline = context->pipelines().specialized("image_warp", sizeof(push), {});
        const std::array reads{source, camera_storage};
        std::vector<StorageRef> writes{destination};
        if (mask.is_valid())
            writes.push_back(storage_ref(mask));
        context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, std::min(65535u, (uint32_t(width) * uint32_t(height) + 255) / 256), 1, 1);
        });
        if (validity)
            *validity = mask.to(DataType::UInt8);
        return output;
    }
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
