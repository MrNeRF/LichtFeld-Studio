/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gsplat_rasterizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "gsplat/Ops.h"
#include "training/kernels/grad_alpha.hpp"
#include <spdlog/spdlog.h>

// Debug macro for CUDA sync points - only active in debug builds
// Disabled by default for performance (saves ~50% API overhead)
#ifdef GSPLAT_DEBUG_SYNC_ENABLED
#define GSPLAT_DEBUG_SYNC(msg)                                   \
    do {                                                         \
        cudaDeviceSynchronize();                                 \
        auto err = cudaGetLastError();                           \
        if (err != cudaSuccess) {                                \
            LOG_ERROR("CUDA error {}: {}", msg, cudaGetErrorString(err)); \
        }                                                        \
    } while (0)
#else
#define GSPLAT_DEBUG_SYNC(msg) ((void)0)
#endif

namespace lfs::training {

    std::expected<std::pair<RenderOutput, GsplatRasterizeContext>, std::string> gsplat_rasterize_forward(
        core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color,
        int tile_x_offset,
        int tile_y_offset,
        int tile_width,
        int tile_height,
        float scaling_modifier,
        bool antialiased,
        GsplatRenderMode render_mode,
        bool use_gut) {

        // Begin arena frame for memory allocation
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        uint64_t frame_id = arena.begin_frame();
        auto arena_allocator = arena.get_allocator(frame_id);

        // Full image dimensions
        const uint32_t full_image_height = static_cast<uint32_t>(viewpoint_camera.image_height());
        const uint32_t full_image_width = static_cast<uint32_t>(viewpoint_camera.image_width());

        // Render dimensions (0 = full image)
        const uint32_t image_width = (tile_width > 0) ? static_cast<uint32_t>(tile_width) : full_image_width;
        const uint32_t image_height = (tile_height > 0) ? static_cast<uint32_t>(tile_height) : full_image_height;

        const float* viewmat_ptr = viewpoint_camera.world_view_transform_ptr();

        // Adjust K matrix principal point (cx, cy) for tile offset
        core::Tensor K_tensor;
        if (tile_x_offset != 0 || tile_y_offset != 0) {
            auto K_cpu = viewpoint_camera.K().cpu().contiguous();
            auto K_acc = K_cpu.accessor<float, 3>();
            K_acc(0, 0, 2) -= static_cast<float>(tile_x_offset);
            K_acc(0, 1, 2) -= static_cast<float>(tile_y_offset);
            K_tensor = K_cpu.to(core::Device::CUDA).contiguous();
        } else {
            K_tensor = viewpoint_camera.K().contiguous();
        }
        const float* K_ptr = K_tensor.ptr<float>();

        // Get Gaussian parameters (activated) - ensure contiguous
        auto means = gaussian_model.get_means().contiguous();
        auto opacities = gaussian_model.get_opacity().contiguous(); // [N] sigmoid applied
        auto scales = gaussian_model.get_scaling().contiguous();    // [N, 3] exp applied
        auto quats = gaussian_model.get_rotation().contiguous();    // [N, 4] normalized
        auto sh_coeffs = gaussian_model.get_shs().contiguous();     // [N, K, 3]
        const uint32_t sh_degree = static_cast<uint32_t>(gaussian_model.get_active_sh_degree());

        // Squeeze opacities if needed
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
            opacities = opacities.squeeze(-1);
        }

        // Get raw pointers
        const float* means_ptr = means.ptr<float>();
        const float* opacities_ptr = opacities.ptr<float>();
        const float* scales_ptr = scales.ptr<float>();
        const float* quats_ptr = quats.ptr<float>();
        const float* sh_coeffs_ptr = sh_coeffs.ptr<float>();

        // Background color
        const float* bg_ptr = nullptr;
        if (bg_color.is_valid() && bg_color.numel() > 0) {
            bg_ptr = bg_color.ptr<float>();
        }

        // Settings
        constexpr float eps2d = 0.3f;
        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 10000.0f;
        constexpr float radius_clip = 0.0f;
        constexpr uint32_t tile_size = 16;
        const bool calc_compensations = antialiased;
        // Convert from lfs::core::CameraModelType (enum class) to global CameraModelType (plain enum) for CUDA kernels
        const ::CameraModelType camera_model = static_cast<::CameraModelType>(
            static_cast<int>(viewpoint_camera.camera_model_type()));

        // Distortion coefficients
        const core::Tensor radial_dist = viewpoint_camera.radial_distortion();
        const core::Tensor tangential_dist = viewpoint_camera.tangential_distortion();
        core::Tensor radial_cuda, tangential_cuda, thin_prism_cuda;
        const float* radial_ptr = nullptr;
        const float* tangential_ptr = nullptr;
        const float* thin_prism_ptr = nullptr;

        // Helper to copy tensor to CUDA
        auto to_cuda = [](const core::Tensor& t) {
            return t.to(core::Device::CUDA).contiguous();
        };

        switch (camera_model) {
        case CameraModelType::THIN_PRISM_FISHEYE:
            if (radial_dist.is_valid() && radial_dist.numel() == 4) {
                radial_cuda = to_cuda(radial_dist);
                radial_ptr = radial_cuda.ptr<float>();
            }
            if (tangential_dist.is_valid() && tangential_dist.numel() == 4) {
                thin_prism_cuda = to_cuda(tangential_dist);
                thin_prism_ptr = thin_prism_cuda.ptr<float>();
            }
            break;
        case CameraModelType::FISHEYE:
            if (radial_dist.is_valid() && radial_dist.numel() >= 4) {
                radial_cuda = to_cuda(radial_dist.numel() == 4 ? radial_dist : radial_dist.slice(0, 0, 4));
                radial_ptr = radial_cuda.ptr<float>();
            }
            break;
        case CameraModelType::PINHOLE:
            if (radial_dist.is_valid() && radial_dist.numel() > 0) {
                radial_cuda = to_cuda(radial_dist.numel() == 6 ? radial_dist : radial_dist.slice(0, 0, std::min(radial_dist.numel(), size_t(6))));
                radial_ptr = radial_cuda.ptr<float>();
            }
            if (tangential_dist.is_valid() && tangential_dist.numel() >= 2) {
                tangential_cuda = to_cuda(tangential_dist.numel() == 2 ? tangential_dist : tangential_dist.slice(0, 0, 2));
                tangential_ptr = tangential_cuda.ptr<float>();
            }
            break;
        default:
            break;
        }

        UnscentedTransformParameters ut_params;

        // Calculate buffer dimensions
        const uint32_t N = static_cast<uint32_t>(means.shape()[0]);
        const uint32_t C = 1; // Single camera
        const uint32_t K = (sh_coeffs.is_valid() && sh_coeffs.ndim() >= 2)
                               ? static_cast<uint32_t>(sh_coeffs.shape()[1])
                               : 0; // SH coefficients
        const uint32_t H = image_height;
        const uint32_t W = image_width;
        const uint32_t num_tiles_y = (H + tile_size - 1) / tile_size;
        const uint32_t num_tiles_x = (W + tile_size - 1) / tile_size;

        // Determine channels based on render mode
        uint32_t channels = 3;
        if (render_mode == GsplatRenderMode::D || render_mode == GsplatRenderMode::ED) {
            channels = 1;
        } else if (render_mode == GsplatRenderMode::RGB_D || render_mode == GsplatRenderMode::RGB_ED) {
            channels = 4;
        }

        // Calculate total memory needed (with alignment)
        auto align = [](size_t size, size_t alignment = 128) {
            return (size + alignment - 1) & ~(alignment - 1);
        };

        // Buffer sizes in bytes
        const size_t radii_size = align(C * N * 2 * sizeof(int32_t));
        const size_t means2d_size = align(C * N * 2 * sizeof(float));
        const size_t depths_size = align(C * N * sizeof(float));
        const size_t dirs_size = align(C * N * 3 * sizeof(float));
        const size_t conics_size = align(C * N * 3 * sizeof(float));
        const size_t compensations_size = calc_compensations ? align(C * N * sizeof(float)) : 0;
        const size_t tiles_per_gauss_size = align(C * N * sizeof(int32_t));
        const size_t tile_offsets_size = align(C * num_tiles_y * num_tiles_x * sizeof(int32_t));
        const size_t colors_size = align(C * N * channels * sizeof(float));
        const size_t render_colors_size = align(C * H * W * channels * sizeof(float));
        const size_t render_alphas_size = align(C * H * W * sizeof(float));
        const size_t last_ids_size = align(C * H * W * sizeof(int32_t));

        const size_t total_size = radii_size + means2d_size + depths_size + dirs_size +
                                  conics_size + compensations_size + tiles_per_gauss_size +
                                  tile_offsets_size + colors_size + render_colors_size +
                                  render_alphas_size + last_ids_size;

        // Allocate from arena
        char* blob = arena_allocator(total_size);

        // Carve out buffers (aligned)
        char* ptr = blob;
        auto* radii_ptr_out = reinterpret_cast<int32_t*>(ptr);
        ptr += radii_size;
        auto* means2d_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += means2d_size;
        auto* depths_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += depths_size;
        auto* dirs_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += dirs_size;
        auto* conics_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += conics_size;
        float* compensations_ptr_out = nullptr;
        if (calc_compensations) {
            compensations_ptr_out = reinterpret_cast<float*>(ptr);
            ptr += compensations_size;
        }
        auto* tiles_per_gauss_ptr = reinterpret_cast<int32_t*>(ptr);
        ptr += tiles_per_gauss_size;
        auto* tile_offsets_ptr_out = reinterpret_cast<int32_t*>(ptr);
        ptr += tile_offsets_size;
        auto* colors_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += colors_size;
        auto* render_colors_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += render_colors_size;
        auto* render_alphas_ptr_out = reinterpret_cast<float*>(ptr);
        ptr += render_alphas_size;
        auto* last_ids_ptr_out = reinterpret_cast<int32_t*>(ptr);

        // Setup result struct
        gsplat_lfs::RasterizeWithSHResult result{
            .render_colors = render_colors_ptr_out,
            .render_alphas = render_alphas_ptr_out,
            .radii = radii_ptr_out,
            .means2d = means2d_ptr_out,
            .depths = depths_ptr_out,
            .colors = colors_ptr_out,
            .dirs = dirs_ptr_out,
            .conics = conics_ptr_out,
            .tiles_per_gauss = tiles_per_gauss_ptr,
            .tile_offsets = tile_offsets_ptr_out,
            .last_ids = last_ids_ptr_out,
            .compensations = compensations_ptr_out,
            .isect_ids = nullptr,
            .flatten_ids = nullptr,
            .n_isects = 0};

        // Call raw pointer forward API
        gsplat_lfs::rasterize_from_world_with_sh_fwd(
            means_ptr,
            quats_ptr,
            scales_ptr,
            opacities_ptr,
            sh_coeffs_ptr,
            sh_degree,
            bg_ptr,
            nullptr, // masks
            N,
            C,
            K,
            image_width,
            image_height,
            tile_size,
            viewmat_ptr,
            nullptr, // viewmats1 (rolling shutter)
            K_ptr,
            camera_model,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            scaling_modifier,
            calc_compensations,
            static_cast<int>(render_mode),
            ut_params,
            ShutterType::GLOBAL,
            radial_ptr,
            tangential_ptr,
            thin_prism_ptr,
            result,
            nullptr);

        // Build RenderOutput - wrap raw pointers in tensor views
        RenderOutput render_output;

        // Create tensor views over arena memory for output
        auto render_colors_tensor = core::Tensor::from_blob(
            render_colors_ptr_out, {static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W), static_cast<size_t>(channels)},
            core::Device::CUDA, core::DataType::Float32);
        auto render_alphas_tensor = core::Tensor::from_blob(
            render_alphas_ptr_out, {static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W), 1UL},
            core::Device::CUDA, core::DataType::Float32);

        // Process based on render mode
        core::Tensor final_image, final_depth;

        switch (render_mode) {
        case GsplatRenderMode::RGB:
            final_image = render_colors_tensor;
            break;

        case GsplatRenderMode::D:
            final_depth = render_colors_tensor;
            break;

        case GsplatRenderMode::ED:
            final_depth = render_colors_tensor.div(render_alphas_tensor.clamp_min(1e-10f));
            break;

        case GsplatRenderMode::RGB_D:
            final_image = render_colors_tensor.slice(-1, 0, 3);
            final_depth = render_colors_tensor.slice(-1, 3, 4);
            break;

        case GsplatRenderMode::RGB_ED:
            final_image = render_colors_tensor.slice(-1, 0, 3);
            auto accum_depth = render_colors_tensor.slice(-1, 3, 4);
            final_depth = accum_depth.div(render_alphas_tensor.clamp_min(1e-10f));
            break;
        }

        // Convert from [1, H, W, C] to [C, H, W] format
        // IMPORTANT: squeeze(0).permute({2,0,1}).contiguous() copies data out of arena
        if (final_image.is_valid() && final_image.numel() > 0) {
            auto pre_clamp = final_image.squeeze(0).permute({2, 0, 1}).contiguous();
            render_output.image = pre_clamp.clamp(0.0f, 1.0f);
        }

        render_output.alpha = render_alphas_tensor.squeeze(0).permute({2, 0, 1}).contiguous();

        if (final_depth.is_valid() && final_depth.numel() > 0) {
            render_output.depth = final_depth.squeeze(0).permute({2, 0, 1}).contiguous();
        }

        render_output.width = static_cast<int>(image_width);
        render_output.height = static_cast<int>(image_height);

        // Build context for backward - store raw pointers
        GsplatRasterizeContext ctx;

        // Store raw pointers directly (arena memory stays valid until end_frame)
        ctx.render_colors_ptr = render_colors_ptr_out;
        ctx.render_alphas_ptr = render_alphas_ptr_out;
        ctx.radii_ptr = radii_ptr_out;
        ctx.means2d_ptr = means2d_ptr_out;
        ctx.depths_ptr = depths_ptr_out;
        ctx.colors_ptr = colors_ptr_out;
        ctx.tile_offsets_ptr = tile_offsets_ptr_out;
        ctx.last_ids_ptr = last_ids_ptr_out;
        ctx.compensations_ptr = compensations_ptr_out;

        // Store flatten_ids from result (allocated by gsplat, must be freed later)
        ctx.isect_ids_ptr = result.isect_ids;
        ctx.flatten_ids_ptr = result.flatten_ids;
        ctx.n_isects = result.n_isects;

        // Save input tensors for backward (these are references, not copies)
        ctx.means = means;
        ctx.quats = quats;
        ctx.scales = scales;
        ctx.opacities = opacities;
        ctx.sh_coeffs = sh_coeffs;

        // Store camera pointers
        ctx.viewmat_ptr = viewmat_ptr;
        ctx.K_ptr = K_ptr;
        ctx.K_tensor = K_tensor;
        ctx.bg_color = bg_color;

        // Distortion coefficients
        ctx.radial_ptr = radial_ptr;
        ctx.tangential_ptr = tangential_ptr;
        ctx.thin_prism_ptr = thin_prism_ptr;
        ctx.radial_cuda = radial_cuda;
        ctx.tangential_cuda = tangential_cuda;
        ctx.thin_prism_cuda = thin_prism_cuda;

        // Save settings
        ctx.N = N;
        ctx.K_sh = K;
        ctx.channels = channels;
        ctx.sh_degree = sh_degree;
        ctx.image_width = image_width;
        ctx.image_height = image_height;
        ctx.tile_size = tile_size;
        ctx.tile_width = num_tiles_x;
        ctx.tile_height = num_tiles_y;
        ctx.eps2d = eps2d;
        ctx.near_plane = near_plane;
        ctx.far_plane = far_plane;
        ctx.radius_clip = radius_clip;
        ctx.scaling_modifier = scaling_modifier;
        ctx.calc_compensations = calc_compensations;
        ctx.render_mode = render_mode;
        ctx.camera_model = camera_model;
        ctx.frame_id = frame_id;
        ctx.render_tile_x_offset = tile_x_offset;
        ctx.render_tile_y_offset = tile_y_offset;
        ctx.render_tile_width = tile_width;
        ctx.render_tile_height = tile_height;

        return std::pair{render_output, ctx};
    }

    void gsplat_rasterize_backward(
        const GsplatRasterizeContext& ctx,
        const core::Tensor& grad_image,
        const core::Tensor& grad_alpha,
        core::SplatData& gaussian_model,
        AdamOptimizer& optimizer) {

        // Get arena for temporary allocations
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        auto arena_allocator = arena.get_allocator(ctx.frame_id);

        const uint32_t N = ctx.N;
        const uint32_t K = ctx.K_sh;
        const uint32_t H = ctx.image_height;
        const uint32_t W = ctx.image_width;
        const uint32_t channels = ctx.channels;

        // Calculate sizes for arena allocation
        auto align = [](size_t size, size_t alignment = 128) {
            return (size + alignment - 1) & ~(alignment - 1);
        };

        size_t v_render_colors_size = align(H * W * channels * sizeof(float));
        size_t v_render_alphas_size = align(H * W * sizeof(float));
        size_t v_means_size = align(N * 3 * sizeof(float));
        size_t v_quats_size = align(N * 4 * sizeof(float));
        size_t v_scales_size = align(N * 3 * sizeof(float));
        size_t v_opacities_size = align(N * sizeof(float));
        size_t v_sh_coeffs_size = align(N * K * 3 * sizeof(float));

        size_t total_bwd_size = v_render_colors_size + v_render_alphas_size +
                                v_means_size + v_quats_size + v_scales_size +
                                v_opacities_size + v_sh_coeffs_size;

        char* bwd_blob = arena_allocator(total_bwd_size);

        // Carve out backward buffers
        char* bwd_ptr = bwd_blob;
        auto* v_render_colors_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_render_colors_size;
        auto* v_render_alphas_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_render_alphas_size;
        auto* v_means_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_means_size;
        auto* v_quats_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_quats_size;
        auto* v_scales_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_scales_size;
        auto* v_opacities_ptr = reinterpret_cast<float*>(bwd_ptr);
        bwd_ptr += v_opacities_size;
        auto* v_sh_coeffs_ptr = reinterpret_cast<float*>(bwd_ptr);

        // Zero the gradient buffers
        cudaMemsetAsync(v_means_ptr, 0, N * 3 * sizeof(float), nullptr);
        cudaMemsetAsync(v_quats_ptr, 0, N * 4 * sizeof(float), nullptr);
        cudaMemsetAsync(v_scales_ptr, 0, N * 3 * sizeof(float), nullptr);
        cudaMemsetAsync(v_opacities_ptr, 0, N * sizeof(float), nullptr);
        cudaMemsetAsync(v_sh_coeffs_ptr, 0, N * K * 3 * sizeof(float), nullptr);

        // Prepare grad_render_colors [1, H, W, channels] - permute from CHW to HWC using custom kernel
        // This avoids memory pool allocation from tensor permute().contiguous()
        if (grad_image.is_valid() && grad_image.numel() > 0) {
            // grad_image is [C, H, W], need [H, W, C]
            kernels::launch_permute_chw_to_hwc(
                grad_image.ptr<float>(),
                v_render_colors_ptr,
                static_cast<int>(channels), static_cast<int>(H), static_cast<int>(W),
                nullptr);
        } else {
            cudaMemsetAsync(v_render_colors_ptr, 0, H * W * channels * sizeof(float), nullptr);
        }

        // Prepare grad_render_alphas [H, W] - squeeze from [1, H, W] using custom kernel
        // This avoids memory pool allocation from tensor permute().contiguous()
        if (grad_alpha.is_valid() && grad_alpha.numel() > 0) {
            // grad_alpha is [1, H, W], need [H, W] - same memory layout
            kernels::launch_squeeze_1hw_to_hw(
                grad_alpha.ptr<float>(),
                v_render_alphas_ptr,
                static_cast<int>(H), static_cast<int>(W),
                nullptr);
        } else {
            cudaMemsetAsync(v_render_alphas_ptr, 0, H * W * sizeof(float), nullptr);
        }

        UnscentedTransformParameters ut_params;

        // Get background pointer
        const float* bg_ptr = nullptr;
        if (ctx.bg_color.is_valid() && ctx.bg_color.numel() > 0) {
            bg_ptr = ctx.bg_color.ptr<float>();
        }

        GSPLAT_DEBUG_SYNC("BEFORE gsplat backward");

        // Call backward with raw pointers
        gsplat_lfs::rasterize_from_world_with_sh_bwd(
            ctx.means.ptr<float>(),
            ctx.quats.ptr<float>(),
            ctx.scales.ptr<float>(),
            ctx.opacities.ptr<float>(),
            ctx.sh_coeffs.ptr<float>(),
            ctx.sh_degree,
            bg_ptr,
            nullptr, // masks
            N,
            1, // C
            K,
            ctx.image_width,
            ctx.image_height,
            ctx.tile_size,
            ctx.viewmat_ptr,
            nullptr, // viewmats1
            ctx.K_ptr,
            ctx.camera_model,
            ctx.eps2d,
            ctx.near_plane,
            ctx.far_plane,
            ctx.radius_clip,
            ctx.scaling_modifier,
            ctx.calc_compensations,
            static_cast<int>(ctx.render_mode),
            ut_params,
            ShutterType::GLOBAL,
            ctx.radial_ptr,
            ctx.tangential_ptr,
            ctx.thin_prism_ptr,
            ctx.render_alphas_ptr,
            ctx.last_ids_ptr,
            ctx.tile_offsets_ptr,
            ctx.flatten_ids_ptr,
            ctx.n_isects,
            ctx.colors_ptr,
            ctx.radii_ptr,
            ctx.means2d_ptr,
            ctx.depths_ptr,
            ctx.compensations_ptr,
            v_render_colors_ptr,
            v_render_alphas_ptr,
            v_means_ptr,
            v_quats_ptr,
            v_scales_ptr,
            v_opacities_ptr,
            v_sh_coeffs_ptr,
            nullptr // stream
        );

        GSPLAT_DEBUG_SYNC("AFTER gsplat backward");

        // ============ Chain rule for activation functions ============
        // gsplat backward returns gradients w.r.t. activated parameters
        // We need to chain rule back to raw parameters
        // Use custom CUDA kernels to avoid tensor allocations

        // Scales: exp(raw) -> v_scales_raw = v_scales * exp(raw_scales) = v_scales * scales
        // In-place: v_scales_ptr *= scales
        kernels::launch_exp_backward(v_scales_ptr, ctx.scales.ptr<float>(), N, nullptr);

        GSPLAT_DEBUG_SYNC("AFTER exp_backward");

        // Opacities: sigmoid(raw) -> v_opacities_raw = v_opacities * sigmoid * (1 - sigmoid)
        // In-place: v_opacities_ptr *= sigmoid * (1 - sigmoid)
        kernels::launch_sigmoid_backward(v_opacities_ptr, ctx.opacities.ptr<float>(), N, nullptr);

        GSPLAT_DEBUG_SYNC("AFTER sigmoid_backward");

        // Quaternions: normalize(raw) -> need Jacobian of normalization
        // v_raw = (v_activated - q_norm * dot(q_norm, v_activated)) / ||q_raw||
        // In-place modification of v_quats_ptr
        auto raw_quats = gaussian_model.rotation_raw();
        kernels::launch_quat_normalize_backward(
            v_quats_ptr,
            ctx.quats.ptr<float>(),
            raw_quats.ptr<float>(),
            N,
            nullptr);

        GSPLAT_DEBUG_SYNC("AFTER quat_normalize_backward");

        // ============ Accumulate gradients into optimizer using CUDA kernels ============
        // This avoids any tensor operations that might allocate from memory pool

        // Means: [N, 3] -> [N, 3]
        kernels::launch_grad_accumulate(
            optimizer.get_grad(ParamType::Means).ptr<float>(),
            v_means_ptr,
            N * 3,
            nullptr);

        // Scales: [N, 3] -> [N, 3]
        kernels::launch_grad_accumulate(
            optimizer.get_grad(ParamType::Scaling).ptr<float>(),
            v_scales_ptr,
            N * 3,
            nullptr);

        // Rotations: [N, 4] -> [N, 4]
        kernels::launch_grad_accumulate(
            optimizer.get_grad(ParamType::Rotation).ptr<float>(),
            v_quats_ptr,
            N * 4,
            nullptr);

        // Opacities: [N] -> [N, 1] (same memory layout)
        kernels::launch_grad_accumulate_unsqueeze(
            optimizer.get_grad(ParamType::Opacity).ptr<float>(),
            v_opacities_ptr,
            N,
            nullptr);

        // SH coefficients: [N, K, 3] -> sh0 [N, 1, 3] + shN [N, K_dst, 3]
        // K is active SH coeffs, K_dst is the full buffer width (max_sh_degree^2 - 1)
        float* dst_shN = nullptr;
        int64_t K_dst = 0;
        if (K > 1) {
            auto shN_grad = optimizer.get_grad(ParamType::ShN);
            if (shN_grad.is_valid() && shN_grad.numel() > 0 && shN_grad.ndim() >= 2) {
                dst_shN = shN_grad.ptr<float>();
                K_dst = static_cast<int64_t>(shN_grad.shape()[1]); // [N, K_dst, 3]
            }
        }

        GSPLAT_DEBUG_SYNC("BEFORE SH kernel");

        kernels::launch_grad_accumulate_sh(
            optimizer.get_grad(ParamType::Sh0).ptr<float>(),
            dst_shN,
            v_sh_coeffs_ptr,
            N,
            K,     // K_src: active SH coefficients
            K_dst, // K_dst: destination buffer width
            nullptr);

        GSPLAT_DEBUG_SYNC("AFTER SH kernel");

        // Update densification info if available (shape is [2, N])
        const bool update_densification_info =
            gaussian_model._densification_info.ndim() == 2 &&
            gaussian_model._densification_info.shape()[1] >= N;

        if (update_densification_info) {
            // Compute ||grad_means[i]||_2 and add to densification_info[i]
            // No tensor allocations needed
            kernels::launch_grad_norm_accumulate(
                gaussian_model._densification_info.ptr<float>(),
                v_means_ptr,
                N,
                nullptr);

            GSPLAT_DEBUG_SYNC("AFTER grad_norm_accumulate");
        }

        // Free internally allocated buffers from forward
        if (ctx.isect_ids_ptr != nullptr) {
            cudaFree(ctx.isect_ids_ptr);
        }
        if (ctx.flatten_ids_ptr != nullptr) {
            cudaFree(ctx.flatten_ids_ptr);
        }

        GSPLAT_DEBUG_SYNC("AFTER cudaFree");

        // End arena frame to release memory from forward pass
        arena.end_frame(ctx.frame_id);

        GSPLAT_DEBUG_SYNC("AFTER end_frame");
    }

} // namespace lfs::training
