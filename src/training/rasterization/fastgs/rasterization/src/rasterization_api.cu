/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "backward.h"
#include "buffer_utils.h"
#include "core/cuda/memory_arena.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "forward.h"
#include "helper_math.h"
#include "lfs/cuda_scratch.hpp"
#include "rasterization_api.h"
#include "rasterization_config.h"
#include "utils.h"
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fast_lfs::rasterization {

    namespace {
        thread_local std::string last_forward_error;
        thread_local std::string last_backward_error;

        void free_sorted_primitive_indices(void* ptr, cudaStream_t stream) noexcept {
            if (!ptr) {
                return;
            }
            lfs::diagnostics::VramProfiler::instance().recordDeallocation(ptr);
#if CUDART_VERSION >= 11020
            const cudaError_t status = cudaFreeAsync(ptr, stream);
#else
            (void)stream;
            const cudaError_t status = cudaFree(ptr);
#endif
            if (status != cudaSuccess) {
                lfs::core::ensure_cuda_success(
                    status, "FastGS sorted-index buffer free",
                    lfs::core::detail::format_cuda_safe(
                        "ptr={}, stream={}", ptr, static_cast<void*>(stream)),
                    LFS_SOURCE_SITE_CURRENT(),
                    lfs::core::CudaFailureDisposition::LogOnly);
            }
        }

        const char* cuda_memory_type_name(cudaMemoryType type) {
            switch (type) {
            case cudaMemoryTypeHost: return "host";
            case cudaMemoryTypeDevice: return "device";
            case cudaMemoryTypeManaged: return "managed";
            default: return "unknown";
            }
        }

        int checked_current_cuda_device(const char* phase) {
            int device_count = 0;
            const cudaError_t count_err = cudaGetDeviceCount(&device_count);
            if (count_err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    count_err, "cudaGetDeviceCount(FastGS preflight)", phase);
            }
            LFS_ASSERT_MSG(device_count > 0,
                           lfs::core::detail::format_cuda_safe(
                               "{}: no CUDA devices are visible", phase));

            int current_device = -1;
            const cudaError_t device_err = cudaGetDevice(&current_device);
            if (device_err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    device_err, "cudaGetDevice(FastGS preflight)",
                    lfs::core::detail::format_cuda_safe(
                        "phase={}, device_count={}", phase, device_count));
            }
            LFS_ASSERT_MSG(
                current_device >= 0 && current_device < device_count,
                lfs::core::detail::format_cuda_safe(
                    "{}: current CUDA device ordinal is out of range "
                    "(current_device={}, device_count={})",
                    phase, current_device, device_count));
            return current_device;
        }

        void checked_no_pending_cuda_error(const char* phase) {
            const cudaError_t pending_err = cudaPeekAtLastError();
            if (pending_err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    pending_err, "cudaPeekAtLastError(FastGS preflight)",
                    lfs::core::detail::format_cuda_safe(
                        "{}: pending CUDA error before buffer sizing", phase));
            }
        }

        void checked_device_pointer_on_current_device(
            const void* ptr,
            const char* name,
            int current_device) {
            LFS_ASSERT_MSG(ptr != nullptr,
                           lfs::core::detail::format_cuda_safe(
                               "FastGS forward preflight: {} is null", name));

            cudaPointerAttributes attrs{};
            const cudaError_t attr_err = cudaPointerGetAttributes(&attrs, ptr);
            if (attr_err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    attr_err, "cudaPointerGetAttributes(FastGS preflight)", name);
            }
            LFS_ASSERT_MSG(
                attrs.type == cudaMemoryTypeDevice,
                lfs::core::detail::format_cuda_safe(
                    "FastGS forward preflight: {} is not device memory (type={})",
                    name, cuda_memory_type_name(attrs.type)));
            LFS_ASSERT_MSG(
                attrs.device == current_device,
                lfs::core::detail::format_cuda_safe(
                    "FastGS forward preflight: {} is allocated on CUDA device {} "
                    "but the current CUDA device is {}",
                    name, attrs.device, current_device));
        }

        void validate_fastgs_forward_cuda_preflight(
            const float* means_ptr,
            const float* scales_raw_ptr,
            const float* rotations_raw_ptr,
            const float* opacities_raw_ptr,
            const float* sh_coefficients_0_ptr,
            const float* sh_coefficients_rest_ptr,
            const float* w2c_ptr,
            const float* cam_position_ptr,
            const float* image_ptr,
            const float* alpha_ptr,
            const float* depth_ptr,
            int n_primitives,
            int active_sh_bases,
            int sh_layout_bases,
            int width,
            int height,
            int n_tiles) {
            const int current_device = checked_current_cuda_device("FastGS forward preflight");
            checked_no_pending_cuda_error("FastGS forward preflight");

            if (n_primitives <= 0 || active_sh_bases <= 0 || active_sh_bases > 16 ||
                sh_layout_bases <= 0 || sh_layout_bases > 16 ||
                width <= 0 || height <= 0 || n_tiles <= 0) {
                throw std::runtime_error(
                    "FastGS forward preflight: invalid dimensions"
                    " (n_primitives=" +
                    std::to_string(n_primitives) +
                    ", active_sh_bases=" + std::to_string(active_sh_bases) +
                    ", sh_layout_bases=" + std::to_string(sh_layout_bases) +
                    ", width=" + std::to_string(width) +
                    ", height=" + std::to_string(height) +
                    ", n_tiles=" + std::to_string(n_tiles) + ")");
            }
            if (sh_layout_bases < active_sh_bases) {
                throw std::runtime_error(
                    "FastGS forward preflight: SH layout is smaller than active SH degree"
                    " (active_sh_bases=" +
                    std::to_string(active_sh_bases) +
                    ", sh_layout_bases=" + std::to_string(sh_layout_bases) + ")");
            }

            checked_device_pointer_on_current_device(means_ptr, "means_ptr", current_device);
            checked_device_pointer_on_current_device(scales_raw_ptr, "scales_raw_ptr", current_device);
            checked_device_pointer_on_current_device(rotations_raw_ptr, "rotations_raw_ptr", current_device);
            checked_device_pointer_on_current_device(opacities_raw_ptr, "opacities_raw_ptr", current_device);
            checked_device_pointer_on_current_device(sh_coefficients_0_ptr, "sh_coefficients_0_ptr", current_device);
            if (active_sh_bases > 1) {
                checked_device_pointer_on_current_device(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr", current_device);
            }
            checked_device_pointer_on_current_device(w2c_ptr, "w2c_ptr", current_device);
            checked_device_pointer_on_current_device(cam_position_ptr, "cam_position_ptr", current_device);
            checked_device_pointer_on_current_device(image_ptr, "image_ptr", current_device);
            checked_device_pointer_on_current_device(alpha_ptr, "alpha_ptr", current_device);
            checked_device_pointer_on_current_device(depth_ptr, "depth_ptr", current_device);
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
        float* depth_ptr,
        float* normal_ptr,
        int n_primitives,
        int active_sh_bases,
        int sh_layout_bases,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane,
        bool mip_filter,
        cudaStream_t stream) {

        if (stream == nullptr) {
            stream = lfs::core::getCurrentCUDAStream();
        }

        lfs::core::RasterizerMemoryArena* arena = nullptr;
        uint64_t frame_id = 0;
        bool frame_started = false;
        try {
            auto fail = [&](std::string message) {
                if (frame_started && arena) {
                    arena->end_frame(frame_id);
                    frame_started = false;
                }
                last_forward_error = std::move(message);
                ForwardContext error_ctx = {};
                error_ctx.success = false;
                error_ctx.error_message = last_forward_error.c_str();
                error_ctx.frame_id = frame_id;
                return error_ctx;
            };

            if (n_primitives <= 0 || width <= 0 || height <= 0) {
                return fail("FastGS forward preflight: invalid dimensions"
                            " (n_primitives=" +
                            std::to_string(n_primitives) +
                            ", width=" + std::to_string(width) +
                            ", height=" + std::to_string(height) + ")");
            }

            // Calculate grid dimensions
            const dim3 grid(div_round_up(width, config::tile_width),
                            div_round_up(height, config::tile_height), 1);
            const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
            const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");

            // Get global arena and begin frame before buffer sizing. begin_frame
            // synchronizes prior arena users, so asynchronous CUDA failures are
            // attributed before the CUB workspace query below.
            arena = &lfs::core::GlobalArenaManager::instance().get_arena();
            frame_id = arena->begin_frame(stream);
            frame_started = true;

            validate_fastgs_forward_cuda_preflight(
                means_ptr,
                scales_raw_ptr,
                rotations_raw_ptr,
                opacities_raw_ptr,
                sh_coefficients_0_ptr,
                sh_coefficients_rest_ptr,
                w2c_ptr,
                cam_position_ptr,
                image_ptr,
                alpha_ptr,
                depth_ptr,
                n_primitives,
                active_sh_bases,
                sh_layout_bases,
                width,
                height,
                n_tiles);

            // Get arena allocator for this frame
            auto arena_allocator = arena->get_allocator(frame_id);

            // Allocate buffers through arena
            const size_t per_primitive_size = required<PerPrimitiveBuffers>(n_primitives);
            const size_t per_tile_size = required<PerTileBuffers>(n_tiles);

            char* per_primitive_buffers_blob = arena_allocator(per_primitive_size);
            char* per_tile_buffers_blob = arena_allocator(per_tile_size);

            if (!per_primitive_buffers_blob || !per_tile_buffers_blob) {
                return fail("OUT_OF_MEMORY: Failed to allocate initial buffers from arena");
            }

            // Allocate helper buffers for backward pass upfront to avoid allocation failures later
            const size_t grad_mean2d_size = static_cast<size_t>(n_primitives) * 2 * sizeof(float);
            const size_t grad_conic_size = static_cast<size_t>(n_primitives) * 3 * sizeof(float);
            const size_t grad_depth_size = static_cast<size_t>(n_primitives) * sizeof(float);
            char* grad_mean2d_helper = arena_allocator(grad_mean2d_size);
            char* grad_conic_helper = arena_allocator(grad_conic_size);
            char* grad_depth_helper = arena_allocator(grad_depth_size);

            if (!grad_mean2d_helper || !grad_conic_helper || !grad_depth_helper) {
                return fail("OUT_OF_MEMORY: Failed to allocate backward helper buffers from arena");
            }

            float3* primitive_normals = nullptr;
            if (normal_ptr != nullptr) {
                checked_device_pointer_on_current_device(normal_ptr, "normal_ptr",
                                                         checked_current_cuda_device("FastGS forward preflight"));
                primitive_normals = reinterpret_cast<float3*>(
                    arena_allocator(static_cast<size_t>(n_primitives) * sizeof(float3)));
                if (!primitive_normals) {
                    return fail("OUT_OF_MEMORY: Failed to allocate primitive normal buffer from arena");
                }
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

            // Call the actual forward implementation
            ForwardResult forward_result = forward(per_primitive_buffers_func,
                                                   per_tile_buffers_func,
                                                   reinterpret_cast<const float3*>(means_ptr),
                                                   reinterpret_cast<const float3*>(scales_raw_ptr),
                                                   reinterpret_cast<const float4*>(rotations_raw_ptr),
                                                   opacities_raw_ptr,
                                                   reinterpret_cast<const float3*>(sh_coefficients_0_ptr),
                                                   reinterpret_cast<const float4*>(sh_coefficients_rest_ptr),
                                                   reinterpret_cast<const float4*>(w2c_ptr),
                                                   reinterpret_cast<const float3*>(cam_position_ptr),
                                                   image_ptr,
                                                   alpha_ptr,
                                                   depth_ptr,
                                                   normal_ptr,
                                                   primitive_normals,
                                                   n_primitives,
                                                   active_sh_bases,
                                                   sh_layout_bases,
                                                   width,
                                                   height,
                                                   focal_x,
                                                   focal_y,
                                                   center_x,
                                                   center_y,
                                                   near_plane,
                                                   far_plane,
                                                   mip_filter,
                                                   stream);

            // Verify allocations happened
            if (forward_result.n_instances > 0 && !forward_result.sorted_primitive_indices) {
                return fail("OUT_OF_MEMORY: Sorted primitive indices were not allocated despite n_instances > 0");
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
            ctx.sh_layout_bases = sh_layout_bases;
            ctx.frame_id = frame_id;
            ctx.stream = stream;
            ctx.grad_mean2d_helper = grad_mean2d_helper;
            ctx.grad_conic_helper = grad_conic_helper;
            ctx.grad_depth_helper = grad_depth_helper;
            ctx.grad_opacity_helper = nullptr;
            ctx.grad_color_helper = nullptr;
            ctx.primitive_normals = primitive_normals;
            ctx.success = true;
            ctx.error_message = nullptr;

            return ctx;

        } catch (const std::exception& e) {
            // Clean up frame on error and return error context instead of throwing
            if (frame_started && arena) {
                arena->end_frame(frame_id);
                frame_started = false;
            }
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
        // Release on the context's stream, not the caller's current one —
        // robust against unwind paths on threads whose guard already popped.
        free_sorted_primitive_indices(forward_ctx.sorted_primitive_indices, forward_ctx.stream);
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(forward_ctx.frame_id, forward_ctx.stream);
    }

    BackwardOutputs backward_raw(
        float* densification_info_ptr,
        const float* densification_error_map_ptr,
        const float* grad_image_ptr,
        const float* grad_alpha_ptr,
        const float* grad_depth_ptr,
        const float* grad_normal_ptr,
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
        int sh_layout_bases,
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        bool mip_filter,
        DensificationType densification_type,
        const FusedAdamSettings* fused_adam) {

        // The forward chose the stream and chained the arena frame on it; the
        // backward shares the same context/arena frame and must match.
        const cudaStream_t stream = forward_ctx.stream;

        BackwardOutputs outputs;
        outputs.success = false;
        outputs.error_message = nullptr;
        if (fused_adam == nullptr || !fused_adam->enabled) {
            release_forward_context(forward_ctx);
            outputs.error_message = "FastGS backward requires fused Adam settings";
            return outputs;
        }
        if (n_primitives <= 0 || width <= 0 || height <= 0 || forward_ctx.n_instances < 0) {
            release_forward_context(forward_ctx);
            outputs.error_message = "Invalid dimensions in backward_raw";
            return outputs;
        }
        if (active_sh_bases <= 0 || active_sh_bases > 16 ||
            sh_layout_bases <= 0 || sh_layout_bases > 16 ||
            sh_layout_bases < active_sh_bases) {
            release_forward_context(forward_ctx);
            last_backward_error =
                "Invalid SH layout in backward_raw (active_sh_bases=" +
                std::to_string(active_sh_bases) +
                ", sh_layout_bases=" + std::to_string(sh_layout_bases) + ")";
            outputs.error_message = last_backward_error.c_str();
            return outputs;
        }

        try {
            // Validate required inputs using pure CUDA validation
            LFS_VALIDATE_CUDA_DEVICE_POINTER(grad_image_ptr, "grad_image_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(grad_alpha_ptr, "grad_alpha_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(grad_depth_ptr, "grad_depth_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(grad_normal_ptr, "grad_normal_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(image_ptr, "image_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(alpha_ptr, "alpha_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(means_ptr, "means_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(scales_raw_ptr, "scales_raw_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(rotations_raw_ptr, "rotations_raw_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(raw_opacities_ptr, "raw_opacities_ptr");
            if (active_sh_bases > 1) {
                LFS_VALIDATE_CUDA_DEVICE_POINTER(sh_coefficients_rest_ptr, "sh_coefficients_rest_ptr");
            }
            LFS_VALIDATE_CUDA_DEVICE_POINTER(w2c_ptr, "w2c_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER(cam_position_ptr, "cam_position_ptr");

            // Optional pointer
            LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(densification_info_ptr, "densification_info_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(densification_error_map_ptr, "densification_error_map_ptr");
            LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(grad_w2c_ptr, "grad_w2c_ptr");
        } catch (const std::exception& e) {
            release_forward_context(forward_ctx);
            last_backward_error = e.what();
            outputs.error_message = last_backward_error.c_str();
            return outputs;
        }

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
        if (!forward_ctx.grad_mean2d_helper || !forward_ctx.grad_conic_helper || !forward_ctx.grad_depth_helper) {
            release_forward_context(forward_ctx);
            outputs.error_message = "Missing pre-allocated helper buffers in forward context";
            return outputs;
        }
        float* grad_mean2d_helper = static_cast<float*>(forward_ctx.grad_mean2d_helper);
        float* grad_conic_helper = static_cast<float*>(forward_ctx.grad_conic_helper);
        float* grad_depth_helper = static_cast<float*>(forward_ctx.grad_depth_helper);
        float* grad_opacity_helper = nullptr;
        float* grad_color_helper = nullptr;
        float3* grad_normal_helper = nullptr;
        const float3* primitive_normals = nullptr;

        try {
            grad_opacity_helper = static_cast<float*>(forward_ctx.grad_opacity_helper);
            grad_color_helper = static_cast<float*>(forward_ctx.grad_color_helper);
            if (!grad_opacity_helper || !grad_color_helper) {
                auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                auto arena_allocator = arena.get_allocator(forward_ctx.frame_id);
                grad_opacity_helper = reinterpret_cast<float*>(arena_allocator(static_cast<size_t>(n_primitives) * sizeof(float)));
                grad_color_helper = reinterpret_cast<float*>(arena_allocator(static_cast<size_t>(n_primitives) * 3 * sizeof(float)));
                if (!grad_opacity_helper || !grad_color_helper) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate fused Adam helper buffers from arena");
                }
            }
            if (grad_normal_ptr != nullptr) {
                primitive_normals = static_cast<const float3*>(forward_ctx.primitive_normals);
                if (!primitive_normals) {
                    throw std::runtime_error("Missing primitive normal buffer in forward context");
                }
                auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                auto arena_allocator = arena.get_allocator(forward_ctx.frame_id);
                grad_normal_helper = reinterpret_cast<float3*>(
                    arena_allocator(static_cast<size_t>(n_primitives) * sizeof(float3)));
                if (!grad_normal_helper) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate normal gradient helper buffer from arena");
                }
                LFS_CUDA_CHECK_MSG(
                    cudaMemsetAsync(grad_normal_helper, 0, static_cast<size_t>(n_primitives) * sizeof(float3), stream),
                    "cudaMemsetAsync(grad_normal_helper)");
            }

            // Zero out helper buffers
            const size_t grad_mean2d_size = static_cast<size_t>(n_primitives) * 2 * sizeof(float);
            const size_t grad_conic_size = static_cast<size_t>(n_primitives) * 3 * sizeof(float);
            const size_t grad_depth_size = static_cast<size_t>(n_primitives) * sizeof(float);
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_mean2d_helper, 0, grad_mean2d_size, stream),
                                 "cudaMemsetAsync(grad_mean2d_helper)");
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_conic_helper, 0, grad_conic_size, stream),
                                 "cudaMemsetAsync(grad_conic_helper)");
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_depth_helper, 0, grad_depth_size, stream),
                                 "cudaMemsetAsync(grad_depth_helper)");
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_opacity_helper, 0, static_cast<size_t>(n_primitives) * sizeof(float), stream),
                                 "cudaMemsetAsync(grad_opacity_helper)");
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_color_helper, 0, static_cast<size_t>(n_primitives) * 3 * sizeof(float), stream),
                                 "cudaMemsetAsync(grad_color_helper)");

            if (grad_w2c_ptr) {
                LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(grad_w2c_ptr, 0, 4 * 4 * sizeof(float), stream),
                                     "cudaMemsetAsync(grad_w2c)");
            }

            // Call the actual backward implementation
            backward(
                densification_error_map_ptr,
                grad_image_ptr,
                grad_alpha_ptr,
                grad_depth_ptr,
                grad_normal_ptr,
                image_ptr,
                alpha_ptr,
                reinterpret_cast<const float3*>(means_ptr),
                reinterpret_cast<const float3*>(scales_raw_ptr),
                reinterpret_cast<const float4*>(rotations_raw_ptr),
                raw_opacities_ptr,
                reinterpret_cast<const float4*>(sh_coefficients_rest_ptr),
                reinterpret_cast<const float4*>(w2c_ptr),
                reinterpret_cast<const float3*>(cam_position_ptr),
                primitive_normals,
                static_cast<char*>(forward_ctx.per_primitive_buffers),
                static_cast<char*>(forward_ctx.per_tile_buffers),
                static_cast<const uint*>(forward_ctx.sorted_primitive_indices),
                grad_opacity_helper,
                reinterpret_cast<float3*>(grad_color_helper),
                reinterpret_cast<float2*>(grad_mean2d_helper),
                grad_conic_helper,
                grad_depth_helper,
                grad_normal_helper,
                grad_w2c_ptr ? reinterpret_cast<float4*>(grad_w2c_ptr) : nullptr,
                densification_info_ptr,
                n_primitives,
                forward_ctx.n_instances,
                active_sh_bases,
                sh_layout_bases,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                mip_filter,
                densification_type,
                *fused_adam,
                stream);

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
                                      + IMG_WIDTH * IMG_HEIGHT * 5 * sizeof(float);       // image + alpha + depth

        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        lfs::training::cuda_scratch::DeviceBuffer input_buffer(
            INPUT_SIZE, stream, "fastgs.warmup.input");
        char* const buffer = input_buffer.as<char>();
        LFS_CUDA_CHECK_MSG(
            cudaMemsetAsync(buffer, 0, INPUT_SIZE, stream),
            "FastGS warmup input initialization");

        float* const means = reinterpret_cast<float*>(buffer);
        float* const scales = means + NUM_GAUSSIANS * 3;
        float* const rotations = scales + NUM_GAUSSIANS * 3;
        float* const opacities = rotations + NUM_GAUSSIANS * 4;
        float* const sh0 = opacities + NUM_GAUSSIANS;
        float* const w2c = sh0 + NUM_GAUSSIANS * 3;
        float* const cam_pos = w2c + 16;
        float* const image = cam_pos + 3;
        float* const alpha = image + IMG_WIDTH * IMG_HEIGHT * 3;
        float* const depth = alpha + IMG_WIDTH * IMG_HEIGHT;

        // Initialize rotations to identity quaternion
        std::vector<float> rot_data(NUM_GAUSSIANS * 4);
        for (int i = 0; i < NUM_GAUSSIANS; ++i) {
            rot_data[i * 4] = 1.0f; // w=1, x=y=z=0
        }
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(rotations, rot_data.data(), rot_data.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "FastGS warmup rotation upload");

        // Initialize w2c to identity and camera position
        const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const float cam[3] = {0.0f, 0.0f, 5.0f};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(w2c, identity, sizeof(identity), cudaMemcpyHostToDevice, stream),
            "FastGS warmup transform upload");
        LFS_CUDA_CHECK_MSG(
            cudaMemcpyAsync(cam_pos, cam, sizeof(cam), cudaMemcpyHostToDevice, stream),
            "FastGS warmup camera upload");
        LFS_CUDA_CHECK_MSG(
            cudaStreamSynchronize(stream),
            "FastGS warmup input upload sync");

        // Forward pass compiles forward kernels
        const auto ctx = forward_raw(
            means, scales, rotations, opacities, sh0, nullptr,
            w2c, cam_pos, image, alpha, depth, nullptr,
            NUM_GAUSSIANS, 1, 1,
            IMG_WIDTH, IMG_HEIGHT,
            FOCAL, FOCAL, CENTER_X, CENTER_Y,
            0.01f, 100.0f);

        if (ctx.success) {
            // Allocate image/alpha gradients plus quantised Adam moments (uint8 m/v + per-primitive
            // fp32 scales). Gaussian gradients are fused.
            constexpr int PARAM_ELEMENTS = NUM_GAUSSIANS * (3 + 3 + 4 + 1 + 3);
            constexpr int PARAM_ROWS = NUM_GAUSSIANS * 5; // means,scaling,rotation,opacity,sh0
            constexpr size_t GRAD_SIZE = IMG_WIDTH * IMG_HEIGHT * 4 * sizeof(float);
            constexpr size_t MOMENT_Q_SIZE = PARAM_ELEMENTS * 2 * sizeof(std::uint8_t);
            constexpr size_t SCALE_SIZE = PARAM_ROWS * 2 * sizeof(float);
            constexpr size_t GRAD_BUFFER_SIZE = GRAD_SIZE + MOMENT_Q_SIZE + SCALE_SIZE;
            lfs::training::cuda_scratch::DeviceBuffer gradient_buffer(
                GRAD_BUFFER_SIZE, stream, "fastgs.warmup.gradients");
            char* const grad_buffer = gradient_buffer.as<char>();
            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(grad_buffer, 0, GRAD_BUFFER_SIZE, stream),
                "FastGS warmup gradient initialization");

            float* const grad_image = reinterpret_cast<float*>(grad_buffer);
            float* const grad_alpha = grad_image + IMG_WIDTH * IMG_HEIGHT * 3;
            std::uint8_t* const exp_avg_q = reinterpret_cast<std::uint8_t*>(grad_buffer + GRAD_SIZE);
            std::uint8_t* const exp_avg_sq_q = exp_avg_q + PARAM_ELEMENTS;
            float* const exp_avg_scale = reinterpret_cast<float*>(grad_buffer + GRAD_SIZE + MOMENT_Q_SIZE);
            float* const exp_avg_sq_scale = exp_avg_scale + PARAM_ROWS;

            int moment_offset = 0;
            int scale_offset = 0;
            auto make_param = [&](float* param, const int n_elements, const int n_attributes) {
                FusedAdamParam adam_param;
                adam_param.param = param;
                adam_param.exp_avg_q = exp_avg_q + moment_offset;
                adam_param.exp_avg_sq_q = exp_avg_sq_q + moment_offset;
                adam_param.exp_avg_scale = exp_avg_scale + scale_offset;
                adam_param.exp_avg_sq_scale = exp_avg_sq_scale + scale_offset;
                adam_param.n_elements = n_elements;
                adam_param.n_attributes = n_attributes;
                adam_param.step_size = 0.0f;
                adam_param.bias_correction2_sqrt_rcp = 1.0f;
                adam_param.enabled = true;
                moment_offset += n_elements;
                scale_offset += n_elements / n_attributes;
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
                nullptr, nullptr, grad_image, grad_alpha, nullptr, nullptr, image, alpha,
                means, scales, rotations, opacities, nullptr, w2c, cam_pos, ctx,
                nullptr,
                NUM_GAUSSIANS, 1, 1,
                IMG_WIDTH, IMG_HEIGHT, FOCAL, FOCAL, CENTER_X, CENTER_Y, true,
                DensificationType::None, &warmup_adam);

        } else {
            lfs::core::GlobalArenaManager::instance().get_arena().end_frame(ctx.frame_id);
        }
    }

} // namespace fast_lfs::rasterization
