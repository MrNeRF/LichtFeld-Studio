/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "core/cuda/memory_arena.hpp"
#include "cuda_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fast_lfs::rasterization {

    namespace {
        thread_local std::string last_forward_error;
        thread_local std::string last_backward_error;

        void free_sorted_primitive_indices(void* ptr) noexcept {
            if (!ptr) {
                return;
            }
#if CUDART_VERSION >= 11020
            cudaFreeAsync(ptr, nullptr);
#else
            cudaFree(ptr);
#endif
        }
    } // namespace

    ForwardContext forward_raw(
        const float* means_ptr,
        const float* scales_raw_ptr,
        const float* rotations_raw_ptr,
        const float* opacities_raw_ptr,
        const float* sh_coefficients_0_ptr,
        const float* sh_coefficients_rest_ptr,
        const float* w2c_ptr,
        const float* cam_position_ptr,
        float* image_ptr,
        float* alpha_ptr,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter) {

        // Validate inputs using pure CUDA validation
        CHECK_CUDA_PTR(means_ptr, "means_ptr");
        CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
        CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
        CHECK_CUDA_PTR(opacities_raw_ptr, "opacities_raw_ptr");
        CHECK_CUDA_PTR(sh_coefficients_0_ptr, "sh_coefficients_0_ptr");
        if (total_bases_sh_rest > 0) {
            CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
        }
        CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
        CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");
        CHECK_CUDA_PTR(image_ptr, "image_ptr");
        CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");

        if (n_primitives <= 0 || width <= 0 || height <= 0) {
            throw std::runtime_error("Invalid dimensions in forward_raw");
        }

        // Calculate grid dimensions
        const dim3 grid(div_round_up(width, config::tile_width),
                        div_round_up(height, config::tile_height), 1);
        const int n_tiles = grid.x * grid.y;

        // Get global arena and begin frame
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        uint64_t frame_id = arena.begin_frame();

        // Get arena allocator for this frame
        auto arena_allocator = arena.get_allocator(frame_id);

        // Allocate buffers through arena
        size_t per_primitive_size = required<PerPrimitiveBuffers>(n_primitives);
        size_t per_tile_size = required<PerTileBuffers>(n_tiles);

        char* per_primitive_buffers_blob = arena_allocator(per_primitive_size);
        char* per_tile_buffers_blob = arena_allocator(per_tile_size);

        if (!per_primitive_buffers_blob || !per_tile_buffers_blob) {
            arena.end_frame(frame_id);
            ForwardContext error_ctx = {};
            error_ctx.success = false;
            error_ctx.error_message = "OUT_OF_MEMORY: Failed to allocate initial buffers from arena";
            error_ctx.frame_id = frame_id; // Set frame_id so caller knows it's ended
            return error_ctx;
        }

        // Allocate helper buffers for backward pass upfront to avoid allocation failures later
        const size_t grad_mean2d_size = n_primitives * 2 * sizeof(float);
        const size_t grad_conic_size = n_primitives * 3 * sizeof(float);
        char* grad_mean2d_helper = arena_allocator(grad_mean2d_size);
        char* grad_conic_helper = arena_allocator(grad_conic_size);

        if (!grad_mean2d_helper || !grad_conic_helper) {
            arena.end_frame(frame_id);
            ForwardContext error_ctx = {};
            error_ctx.success = false;
            error_ctx.error_message = "OUT_OF_MEMORY: Failed to allocate backward helper buffers from arena";
            error_ctx.frame_id = frame_id;
            return error_ctx;
        }

        // Create allocation wrappers
        std::function<char*(size_t)> per_primitive_buffers_func =
            [&per_primitive_buffers_blob](size_t size) -> char* {
            // Already allocated, just return the pointer
            return per_primitive_buffers_blob;
        };

        std::function<char*(size_t)> per_tile_buffers_func =
            [&per_tile_buffers_blob](size_t size) -> char* {
            return per_tile_buffers_blob;
        };

        try {
            // Call the actual forward implementation
            ForwardResult forward_result = forward(per_primitive_buffers_func,
                                                   per_tile_buffers_func,
                                                   reinterpret_cast<const float3*>(means_ptr),
                                                   reinterpret_cast<const float3*>(scales_raw_ptr),
                                                   reinterpret_cast<const float4*>(rotations_raw_ptr),
                                                   opacities_raw_ptr,
                                                   reinterpret_cast<const float3*>(sh_coefficients_0_ptr),
                                                   reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
                                                   reinterpret_cast<const float4*>(w2c_ptr),
                                                   reinterpret_cast<const float3*>(cam_position_ptr),
                                                   image_ptr,
                                                   alpha_ptr,
                                                   n_primitives,
                                                   active_sh_bases,
                                                   total_bases_sh_rest,
                                                   width,
                                                   height,
                                                   focal_x,
                                                   focal_y,
                                                   center_x,
                                                   center_y,
                                                   near_plane,
                                                   far_plane,
                                                   mip_filter);

            // Verify allocations happened
            if (forward_result.n_instances > 0 && !forward_result.sorted_primitive_indices) {
                arena.end_frame(frame_id);
                ForwardContext error_ctx = {};
                error_ctx.success = false;
                error_ctx.error_message = "OUT_OF_MEMORY: Sorted primitive indices were not allocated despite n_instances > 0";
                error_ctx.frame_id = frame_id;
                return error_ctx;
            }
            // Create and return context
            ForwardContext ctx;
            ctx.per_primitive_buffers = per_primitive_buffers_blob;
            ctx.per_tile_buffers = per_tile_buffers_blob;
            ctx.sorted_primitive_indices = forward_result.sorted_primitive_indices;
            ctx.per_primitive_buffers_size = per_primitive_size;
            ctx.per_tile_buffers_size = per_tile_size;
            ctx.sorted_primitive_indices_size = forward_result.sorted_primitive_indices_size;
            ctx.per_instance_sort_scratch_size = forward_result.per_instance_sort_scratch_size;
            ctx.per_instance_sort_total_size = forward_result.per_instance_sort_total_size;
            ctx.n_instances = forward_result.n_instances;
            ctx.frame_id = frame_id;
            ctx.grad_mean2d_helper = grad_mean2d_helper;
            ctx.grad_conic_helper = grad_conic_helper;
            ctx.grad_opacity_helper = nullptr;
            ctx.grad_color_helper = nullptr;
            ctx.success = true;
            ctx.error_message = nullptr;

            return ctx;

        } catch (const std::exception& e) {
            // Clean up frame on error and return error context instead of throwing
            arena.end_frame(frame_id);
            last_forward_error = e.what();
            ForwardContext error_ctx = {};
            error_ctx.success = false;
            error_ctx.error_message = last_forward_error.c_str();
            error_ctx.frame_id = frame_id;
            return error_ctx;
        }
    }

    void release_forward_context(const ForwardContext& forward_ctx) {
        if (!forward_ctx.success) {
            return;
        }
        free_sorted_primitive_indices(forward_ctx.sorted_primitive_indices);
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(forward_ctx.frame_id);
    }

    BackwardOutputs backward_raw(
        float* densification_info_ptr,
        const float* densification_error_map_ptr,
        const float* grad_image_ptr,
        const float* grad_alpha_ptr,
        const float* image_ptr,
        const float* alpha_ptr,
        const float* means_ptr,
        const float* scales_raw_ptr,
        const float* rotations_raw_ptr,
        const float* raw_opacities_ptr,
        const float* sh_coefficients_rest_ptr,
        const float* w2c_ptr,
        const float* cam_position_ptr,
        const ForwardContext& forward_ctx,
        float* grad_w2c_ptr,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter,
        DensificationType densification_type,
        const FusedAdamSettings* fused_adam) {

        BackwardOutputs outputs;
        outputs.success = false;
        outputs.error_message = nullptr;
        if (fused_adam == nullptr || !fused_adam->enabled) {
            release_forward_context(forward_ctx);
            outputs.error_message = "FastGS backward requires fused Adam settings";
            return outputs;
        }

        // Validate required inputs using pure CUDA validation
        CHECK_CUDA_PTR(grad_image_ptr, "grad_image_ptr");
        CHECK_CUDA_PTR(grad_alpha_ptr, "grad_alpha_ptr");
        CHECK_CUDA_PTR(image_ptr, "image_ptr");
        CHECK_CUDA_PTR(alpha_ptr, "alpha_ptr");
        CHECK_CUDA_PTR(means_ptr, "means_ptr");
        CHECK_CUDA_PTR(scales_raw_ptr, "scales_raw_ptr");
        CHECK_CUDA_PTR(rotations_raw_ptr, "rotations_raw_ptr");
        if (total_bases_sh_rest > 0) {
            CHECK_CUDA_PTR(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
        }
        CHECK_CUDA_PTR(w2c_ptr, "w2c_ptr");
        CHECK_CUDA_PTR(cam_position_ptr, "cam_position_ptr");

        // Optional pointer
        CHECK_CUDA_PTR_OPTIONAL(densification_info_ptr, "densification_info_ptr");
        CHECK_CUDA_PTR_OPTIONAL(densification_error_map_ptr, "densification_error_map_ptr");
        CHECK_CUDA_PTR_OPTIONAL(grad_w2c_ptr, "grad_w2c_ptr");

        // Validate forward context
        if (!forward_ctx.per_primitive_buffers || !forward_ctx.per_tile_buffers) {
            release_forward_context(forward_ctx);
            outputs.error_message = "Invalid forward context buffers";
            return outputs;
        }

        if (forward_ctx.n_instances > 0 && !forward_ctx.sorted_primitive_indices) {
            release_forward_context(forward_ctx);
            outputs.error_message = "Missing sorted primitive indices in forward context";
            return outputs;
        }

        // Use pre-allocated helper buffers from forward context
        if (!forward_ctx.grad_mean2d_helper || !forward_ctx.grad_conic_helper) {
            release_forward_context(forward_ctx);
            outputs.error_message = "Missing pre-allocated helper buffers in forward context";
            return outputs;
        }
        float* grad_mean2d_helper = static_cast<float*>(forward_ctx.grad_mean2d_helper);
        float* grad_conic_helper = static_cast<float*>(forward_ctx.grad_conic_helper);
        float* grad_opacity_helper = nullptr;
        float* grad_color_helper = nullptr;

        try {
            grad_opacity_helper = static_cast<float*>(forward_ctx.grad_opacity_helper);
            grad_color_helper = static_cast<float*>(forward_ctx.grad_color_helper);
            if (!grad_opacity_helper || !grad_color_helper) {
                auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                auto arena_allocator = arena.get_allocator(forward_ctx.frame_id);
                grad_opacity_helper = reinterpret_cast<float*>(arena_allocator(n_primitives * sizeof(float)));
                grad_color_helper = reinterpret_cast<float*>(arena_allocator(n_primitives * 3 * sizeof(float)));
                if (!grad_opacity_helper || !grad_color_helper) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate fused Adam helper buffers from arena");
                }
            }

            // Zero out helper buffers
            const size_t grad_mean2d_size = n_primitives * 2 * sizeof(float);
            const size_t grad_conic_size = n_primitives * 3 * sizeof(float);
            cudaMemset(grad_mean2d_helper, 0, grad_mean2d_size);
            cudaMemset(grad_conic_helper, 0, grad_conic_size);
            cudaMemset(grad_opacity_helper, 0, n_primitives * sizeof(float));
            cudaMemset(grad_color_helper, 0, n_primitives * 3 * sizeof(float));

            if (grad_w2c_ptr) {
                cudaMemset(grad_w2c_ptr, 0, 4 * 4 * sizeof(float));
            }

            // Call the actual backward implementation
            backward(
                densification_error_map_ptr,
                grad_image_ptr,
                grad_alpha_ptr,
                image_ptr,
                alpha_ptr,
                reinterpret_cast<const float3*>(means_ptr),
                reinterpret_cast<const float3*>(scales_raw_ptr),
                reinterpret_cast<const float4*>(rotations_raw_ptr),
                raw_opacities_ptr,
                reinterpret_cast<const float3*>(sh_coefficients_rest_ptr),
                reinterpret_cast<const float4*>(w2c_ptr),
                reinterpret_cast<const float3*>(cam_position_ptr),
                static_cast<char*>(forward_ctx.per_primitive_buffers),
                static_cast<char*>(forward_ctx.per_tile_buffers),
                static_cast<const uint*>(forward_ctx.sorted_primitive_indices),
                grad_opacity_helper,
                reinterpret_cast<float3*>(grad_color_helper),
                reinterpret_cast<float2*>(grad_mean2d_helper),
                grad_conic_helper,
                grad_w2c_ptr ? reinterpret_cast<float4*>(grad_w2c_ptr) : nullptr,
                densification_info_ptr,
                n_primitives,
                forward_ctx.n_instances,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                mip_filter,
                densification_type,
                *fused_adam);

            // Mark frame as complete
            release_forward_context(forward_ctx);

            outputs.success = true;
            return outputs;

        } catch (const std::exception& e) {
            // Clean up on error
            release_forward_context(forward_ctx);

            last_backward_error = e.what();
            outputs.error_message = last_backward_error.c_str();
            return outputs;
        }
    }

    void warmup_kernels() {
        // Pre-compile rasterization kernels via minimal forward+backward pass.
        // All allocated memory is released before returning.

        constexpr int NUM_GAUSSIANS = 100;
        constexpr int IMG_WIDTH = 64;
        constexpr int IMG_HEIGHT = 64;
        constexpr float FOCAL = 50.0f;
        constexpr float CENTER_X = IMG_WIDTH / 2.0f;
        constexpr float CENTER_Y = IMG_HEIGHT / 2.0f;

        // Allocate all buffers in one block for efficiency
        constexpr size_t INPUT_SIZE = NUM_GAUSSIANS * (3 + 3 + 4 + 1 + 3) * sizeof(float) // means, scales, rotations, opacities, sh0
                                      + 16 * sizeof(float)                                // w2c
                                      + 3 * sizeof(float)                                 // cam_pos
                                      + IMG_WIDTH * IMG_HEIGHT * 4 * sizeof(float);       // image + alpha

        char* buffer;
        cudaMalloc(&buffer, INPUT_SIZE);
        cudaMemset(buffer, 0, INPUT_SIZE);

        float* const means = reinterpret_cast<float*>(buffer);
        float* const scales = means + NUM_GAUSSIANS * 3;
        float* const rotations = scales + NUM_GAUSSIANS * 3;
        float* const opacities = rotations + NUM_GAUSSIANS * 4;
        float* const sh0 = opacities + NUM_GAUSSIANS;
        float* const w2c = sh0 + NUM_GAUSSIANS * 3;
        float* const cam_pos = w2c + 16;
        float* const image = cam_pos + 3;
        float* const alpha = image + IMG_WIDTH * IMG_HEIGHT * 3;

        // Initialize rotations to identity quaternion
        std::vector<float> rot_data(NUM_GAUSSIANS * 4);
        for (int i = 0; i < NUM_GAUSSIANS; ++i) {
            rot_data[i * 4] = 1.0f; // w=1, x=y=z=0
        }
        cudaMemcpy(rotations, rot_data.data(), rot_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Initialize w2c to identity and camera position
        const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const float cam[3] = {0.0f, 0.0f, 5.0f};
        cudaMemcpy(w2c, identity, sizeof(identity), cudaMemcpyHostToDevice);
        cudaMemcpy(cam_pos, cam, sizeof(cam), cudaMemcpyHostToDevice);

        // Forward pass compiles forward kernels
        const auto ctx = forward_raw(
            means, scales, rotations, opacities, sh0, nullptr,
            w2c, cam_pos, image, alpha,
            NUM_GAUSSIANS, 1, 0,
            IMG_WIDTH, IMG_HEIGHT,
            FOCAL, FOCAL, CENTER_X, CENTER_Y,
            0.01f, 100.0f);

        if (ctx.success) {
            // Allocate image/alpha gradients plus Adam moments. Gaussian gradients are fused.
            constexpr int PARAM_ELEMENTS = NUM_GAUSSIANS * (3 + 3 + 4 + 1 + 3);
            constexpr size_t GRAD_SIZE = IMG_WIDTH * IMG_HEIGHT * 4 * sizeof(float);
            constexpr size_t MOMENT_SIZE = PARAM_ELEMENTS * 2 * sizeof(float);
            char* grad_buffer;
            cudaMalloc(&grad_buffer, GRAD_SIZE + MOMENT_SIZE);
            cudaMemset(grad_buffer, 0, GRAD_SIZE + MOMENT_SIZE);

            float* const grad_image = reinterpret_cast<float*>(grad_buffer);
            float* const grad_alpha = grad_image + IMG_WIDTH * IMG_HEIGHT * 3;
            float* const exp_avg = grad_alpha + IMG_WIDTH * IMG_HEIGHT;
            float* const exp_avg_sq = exp_avg + PARAM_ELEMENTS;

            int moment_offset = 0;
            auto make_param = [&](float* param, const int n_elements, const int n_attributes) {
                FusedAdamParam adam_param;
                adam_param.param = param;
                adam_param.exp_avg = exp_avg + moment_offset;
                adam_param.exp_avg_sq = exp_avg_sq + moment_offset;
                adam_param.n_elements = n_elements;
                adam_param.n_attributes = n_attributes;
                adam_param.step_size = 0.0f;
                adam_param.bias_correction2_sqrt_rcp = 1.0f;
                adam_param.enabled = true;
                moment_offset += n_elements;
                return adam_param;
            };

            FusedAdamSettings warmup_adam;
            warmup_adam.enabled = true;
            warmup_adam.means = make_param(means, NUM_GAUSSIANS * 3, 3);
            warmup_adam.scaling = make_param(scales, NUM_GAUSSIANS * 3, 3);
            warmup_adam.rotation = make_param(rotations, NUM_GAUSSIANS * 4, 4);
            warmup_adam.opacity = make_param(opacities, NUM_GAUSSIANS, 1);
            warmup_adam.sh0 = make_param(sh0, NUM_GAUSSIANS * 3, 3);

            // Backward pass compiles backward kernels (also releases arena)
            backward_raw(
                nullptr, nullptr, grad_image, grad_alpha, image, alpha,
                means, scales, rotations, opacities, nullptr, w2c, cam_pos, ctx,
                nullptr,
                NUM_GAUSSIANS, 1, 0,
                IMG_WIDTH, IMG_HEIGHT, FOCAL, FOCAL, CENTER_X, CENTER_Y, true,
                DensificationType::None, &warmup_adam);

            cudaFree(grad_buffer);
        } else {
            lfs::core::GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
        }

        cudaFree(buffer);
        // Note: cudaFree is synchronous, no need for cudaDeviceSynchronize
    }

} // namespace fast_lfs::rasterization
