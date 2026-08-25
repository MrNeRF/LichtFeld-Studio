/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gsplat_rasterizer.hpp"
#include "core/crash_handler.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant_kernels.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "gsplat/Ops.h"
#include "training/kernels/grad_alpha.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace lfs::training {

    namespace {
        struct PinnedDeviceFloats {
            float* pinned = nullptr;
            size_t pinned_cap = 0;
            size_t bump = 0;

            void release() noexcept {
                if (pinned) {
                    (void)cudaFreeHost(pinned);
                    pinned = nullptr;
                    pinned_cap = 0;
                    bump = 0;
                }
            }

            ~PinnedDeviceFloats() { release(); }

            void begin_frame() {
                bump = 0;
                // K(9) + radial(6) + tangential(2) + thin-prism(4)
                ensure_pinned(32);
            }

            void ensure_pinned(size_t n) {
                if (n <= pinned_cap && pinned) {
                    return;
                }
                float* next = nullptr;
                LFS_CUDA_CHECK(cudaMallocHost(&next, n * sizeof(float)));
                if (pinned) {
                    if (bump > 0) {
                        std::memcpy(next, pinned, bump * sizeof(float));
                    }
                    (void)cudaFreeHost(pinned);
                }
                pinned = next;
                pinned_cap = n;
            }

            void copy_to(core::Tensor& dest, const float* src, size_t n, cudaStream_t stream) {
                if (n == 0) {
                    dest = {};
                    return;
                }
                ensure_pinned(bump + n);
                float* const slot = pinned + bump;
                bump += n;
                std::memcpy(slot, src, n * sizeof(float));
                if (!dest.is_valid() || dest.numel() < n) {
                    dest = core::Tensor::empty(
                        {n}, core::Device::CUDA, core::DataType::Float32);
                }
                if (dest.stream() != stream) {
                    dest.set_stream(stream);
                }
                LFS_CUDA_CHECK(cudaMemcpyAsync(
                    dest.ptr<float>(), slot, n * sizeof(float),
                    cudaMemcpyHostToDevice, stream));
            }
        };

        struct GsplatThreadLocalCaches {
            PinnedDeviceFloats staging;
            core::Tensor K;
            core::Tensor radial;
            core::Tensor tangential;
            core::Tensor thin_prism;
            core::Tensor image_chw;
            core::Tensor alpha_chw;
            core::Tensor depth_chw;
            core::Tensor shN_dequant;
        };

        thread_local GsplatThreadLocalCaches gsplat_thread_caches;
    } // namespace

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
        bool use_gut,
        const core::Tensor& bg_image) {

        // Begin arena frame for memory allocation
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        uint64_t frame_id = arena.begin_frame(core::getCurrentCUDAStream());
        auto arena_allocator = arena.get_allocator(frame_id);
        try {

            // Full image dimensions
            const uint32_t full_image_height = static_cast<uint32_t>(viewpoint_camera.image_height());
            const uint32_t full_image_width = static_cast<uint32_t>(viewpoint_camera.image_width());

            // Render dimensions (0 = full image)
            const uint32_t image_width = (tile_width > 0) ? static_cast<uint32_t>(tile_width) : full_image_width;
            const uint32_t image_height = (tile_height > 0) ? static_cast<uint32_t>(tile_height) : full_image_height;

            const float* viewmat_ptr = viewpoint_camera.world_view_transform_ptr();

            // Convert from lfs::core::CameraModelType (enum class) to global CameraModelType (plain enum) for CUDA kernels
            const ::CameraModelType camera_model = static_cast<::CameraModelType>(
                static_cast<int>(viewpoint_camera.camera_model_type()));

            // Build K directly from intrinsics to avoid extra CUDA->CPU->CUDA roundtrips.
            const auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();
            float k00 = fx;
            float k11 = fy;
            float k02 = cx - static_cast<float>(tile_x_offset);
            float k12 = cy - static_cast<float>(tile_y_offset);

            // For equirectangular cameras in tile mode, encode tile info in K matrix.
            // The CUDA kernels read these values as:
            //   K[0][0] (focal_length.x) = full_image_width
            //   K[1][1] (focal_length.y) = full_image_height
            //   K[0][2] (principal_point.x) = tile_x_offset
            //   K[1][2] (principal_point.y) = tile_y_offset
            if (camera_model == CameraModelType::EQUIRECTANGULAR) {
                k00 = static_cast<float>(full_image_width);
                k11 = static_cast<float>(full_image_height);
                k02 = static_cast<float>(tile_x_offset);
                k12 = static_cast<float>(tile_y_offset);
            }

            core::Tensor K_tensor;

            // Raw parameters: kernels apply sigmoid/exp/normalize internally.
            auto ensure_contiguous = [](core::Tensor t) -> core::Tensor {
                return t.is_contiguous() ? t : t.contiguous();
            };
            auto means = ensure_contiguous(gaussian_model.get_means());
            auto opacities = ensure_contiguous(gaussian_model.opacity_raw());
            auto scales = ensure_contiguous(gaussian_model.scaling_raw());
            auto quats = ensure_contiguous(gaussian_model.rotation_raw());
            auto sh0 = ensure_contiguous(gaussian_model.sh0());
            auto shN = ensure_contiguous(gaussian_model.shN());
            const uint32_t sh_degree = static_cast<uint32_t>(gaussian_model.get_active_sh_degree());

            // Squeeze opacities if needed
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
                if (!opacities.is_contiguous()) {
                    opacities = opacities.contiguous();
                }
            }

            core::pin_operands({&means, &opacities, &scales, &quats, &sh0, &shN});

            // Current-stream-first (the caller's guard), tensor stream as
            // fallback — matches the lib-wide rule and the begin_frame stream,
            // so a metrics-thread render lands its kernels and consumers on the
            // same stream as the arena frame.
            const cudaStream_t fwd_stream = core::getCurrentCUDAStream()
                                                ? core::getCurrentCUDAStream()
                                                : means.stream();

            const std::array<float, 9> K_host = {
                k00, 0.0f, k02,
                0.0f, k11, k12,
                0.0f, 0.0f, 1.0f};
            gsplat_thread_caches.staging.begin_frame();
            gsplat_thread_caches.staging.copy_to(
                gsplat_thread_caches.K, K_host.data(), 9, fwd_stream);
            K_tensor = gsplat_thread_caches.K;

            // Get raw pointers
            const float* means_ptr = means.ptr<float>();
            const float* opacities_ptr = opacities.ptr<float>();
            const float* scales_ptr = scales.ptr<float>();
            const float* quats_ptr = quats.ptr<float>();
            const float* sh0_ptr = sh0.ptr<float>();
            // gsplat is float-native — dequant q16 shN into a temporary float4
            // buffer (does not mutate resident storage). FastGS path decodes in-registers.
            core::Tensor shN_dequant_temp;
            const float* shN_ptr = nullptr;
            if (sh_degree > 0 && shN.is_valid() && shN.numel() > 0) {
                if (gaussian_model.shN_value_quantized() &&
                    gaussian_model.shN_value_bounds().is_valid()) {
                    const auto n_prims = static_cast<std::size_t>(gaussian_model.size());
                    const auto rest =
                        static_cast<std::uint32_t>(gaussian_model.max_sh_coeffs_rest());
                    const auto n_floats = core::sh_swizzled_float_count(n_prims, rest);
                    auto& dequant = gsplat_thread_caches.shN_dequant;
                    if (!dequant.is_valid() || dequant.numel() < n_floats) {
                        dequant = core::Tensor::empty(
                            core::TensorShape({n_floats}), core::Device::CUDA,
                            core::DataType::Float32);
                    }
                    if (dequant.stream() != fwd_stream) {
                        dequant.set_stream(fwd_stream);
                    }
                    shN_dequant_temp = dequant;
                    const auto q16 = lfs::core::resolve_q16_bind_ptrs(gaussian_model);
                    lfs::core::sh_value_quant::decode_shN_u16_to_float4(
                        reinterpret_cast<const std::uint16_t*>(q16.codes),
                        q16.bounds,
                        shN_dequant_temp.ptr<float>(),
                        n_prims,
                        rest,
                        fwd_stream);
                    shN_ptr = shN_dequant_temp.ptr<float>();
                } else {
                    if (shN.dtype() != core::DataType::Float32) {
                        LOG_ERROR(
                            "gsplat SH-rest requires Float32 or valid q16 codes+bounds; "
                            "refusing to reinterpret non-Float32 storage");
                        throw std::runtime_error(
                            "gsplat_rasterize_forward: unsupported SH-rest storage; "
                            "expected Float32 or valid q16 codes+bounds");
                    }
                    shN_ptr = shN.ptr<float>();
                }
            }

            // Background color and image pointers
            // bg_color and bg_image are mutually exclusive - use one or the other
            const bool use_bg_image = bg_image.is_valid() && !bg_image.is_empty();
            const float* bg_color_ptr = nullptr;
            const float* bg_image_ptr = nullptr;

            if (use_bg_image) {
                // Use per-pixel background image - passed directly to gsplat kernel
                core::pin_operands({&bg_image});
                bg_image_ptr = bg_image.ptr<float>();
            } else if (bg_color.is_valid() && bg_color.numel() > 0) {
                core::pin_operands({&bg_color});
                bg_color_ptr = bg_color.ptr<float>();
            }

            // Settings
            constexpr float eps2d = 0.3f;
            constexpr float near_plane = 0.01f;
            constexpr float far_plane = 10000.0f;
            constexpr float radius_clip = 0.0f;
            constexpr uint32_t tile_size = 16;
            const bool calc_compensations = antialiased;
            const float* K_ptr = K_tensor.ptr<float>();

            // Distortion coefficients. CPU tensors are uploaded through the
            // persistent pinned staging (distinct slots from K). Device tensors
            // are reused as-is.
            const core::Tensor radial_dist = viewpoint_camera.radial_distortion();
            const core::Tensor tangential_dist = viewpoint_camera.tangential_distortion();
            core::Tensor radial_cuda, tangential_cuda, thin_prism_cuda;
            const float* radial_ptr = nullptr;
            const float* tangential_ptr = nullptr;
            const float* thin_prism_ptr = nullptr;

            auto upload_dist = [&](const core::Tensor& src, size_t n, core::Tensor& dest) {
                if (!src.is_valid() || src.numel() == 0 || n == 0) {
                    dest = {};
                    return;
                }
                const size_t copy_n = std::min(n, static_cast<size_t>(src.numel()));
                if (src.device() == core::Device::CUDA && src.is_contiguous() &&
                    src.numel() == copy_n) {
                    dest = src;
                    if (dest.stream() != fwd_stream) {
                        dest.set_stream(fwd_stream);
                    }
                    return;
                }
                core::Tensor host = src;
                if (!host.is_contiguous()) {
                    host = host.contiguous();
                }
                if (host.device() != core::Device::CPU) {
                    host = host.cpu();
                }
                gsplat_thread_caches.staging.copy_to(dest, host.ptr<float>(), copy_n, fwd_stream);
            };

            switch (camera_model) {
            case CameraModelType::THIN_PRISM_FISHEYE:
                if (radial_dist.is_valid() && radial_dist.numel() == 4) {
                    upload_dist(radial_dist, 4, gsplat_thread_caches.radial);
                    radial_cuda = gsplat_thread_caches.radial;
                }
                if (tangential_dist.is_valid() && tangential_dist.numel() == 4) {
                    upload_dist(tangential_dist, 4, gsplat_thread_caches.thin_prism);
                    thin_prism_cuda = gsplat_thread_caches.thin_prism;
                }
                break;
            case CameraModelType::FISHEYE:
                if (radial_dist.is_valid() && radial_dist.numel() >= 4) {
                    upload_dist(radial_dist.numel() == 4 ? radial_dist : radial_dist.slice(0, 0, 4),
                                4, gsplat_thread_caches.radial);
                    radial_cuda = gsplat_thread_caches.radial;
                }
                break;
            case CameraModelType::PINHOLE: {
                if (radial_dist.is_valid() && radial_dist.numel() > 0) {
                    const size_t n_rad = std::min(radial_dist.numel(), size_t(6));
                    upload_dist(radial_dist.numel() == n_rad ? radial_dist : radial_dist.slice(0, 0, n_rad),
                                n_rad, gsplat_thread_caches.radial);
                    radial_cuda = gsplat_thread_caches.radial;
                }
                if (tangential_dist.is_valid() && tangential_dist.numel() >= 2) {
                    upload_dist(tangential_dist.numel() == 2 ? tangential_dist : tangential_dist.slice(0, 0, 2),
                                2, gsplat_thread_caches.tangential);
                    tangential_cuda = gsplat_thread_caches.tangential;
                }
                break;
            }
            default:
                break;
            }
            if (radial_cuda.is_valid() && radial_cuda.numel() > 0) {
                radial_ptr = radial_cuda.ptr<float>();
            }
            if (tangential_cuda.is_valid() && tangential_cuda.numel() > 0) {
                tangential_ptr = tangential_cuda.ptr<float>();
            }
            if (thin_prism_cuda.is_valid() && thin_prism_cuda.numel() > 0) {
                thin_prism_ptr = thin_prism_cuda.ptr<float>();
            }

            UnscentedTransformParameters ut_params;

            // Calculate buffer dimensions
            const uint32_t N = static_cast<uint32_t>(means.shape()[0]);
            const uint32_t C = 1;                                   // Single camera
            const uint32_t K = (sh_degree + 1u) * (sh_degree + 1u); // active SH coefficients including sh0
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
            const size_t dirs_size = (sh_degree > 0) ? align(C * N * 3 * sizeof(float)) : 0;
            const size_t conics_size = align(C * N * 3 * sizeof(float));
            const size_t compensations_size = calc_compensations ? align(C * N * sizeof(float)) : 0;
            const size_t tiles_per_gauss_size = align(C * N * sizeof(int32_t));
            const size_t tile_offsets_size =
                align((C * num_tiles_y * num_tiles_x + 1u) * sizeof(int32_t));
            const size_t colors_size = align(C * N * channels * sizeof(float));
            const size_t last_ids_size = align(C * H * W * sizeof(int32_t));

            const size_t total_size = radii_size + means2d_size + depths_size + dirs_size +
                                      conics_size + compensations_size + tiles_per_gauss_size +
                                      tile_offsets_size + colors_size + last_ids_size;

            // Allocate from arena
            char* blob = arena_allocator(total_size);
            if (blob == nullptr) {
                throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::ResourceExhausted,
                    .domain = lfs::ErrorDomain::CUDA,
                    .user_message = "Ran out of GPU memory while rendering during training.",
                    .detail = std::format("gsplat forward arena allocation failed ({} bytes)", total_size),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }

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
            auto* last_ids_ptr_out = reinterpret_cast<int32_t*>(ptr);

            auto& cached_image_chw = gsplat_thread_caches.image_chw;
            auto& cached_alpha_chw = gsplat_thread_caches.alpha_chw;
            auto& cached_depth_chw = gsplat_thread_caches.depth_chw;
            const core::TensorShape image_shape = {
                static_cast<size_t>(channels), static_cast<size_t>(H), static_cast<size_t>(W)};
            if (!cached_image_chw.is_valid() || cached_image_chw.shape() != image_shape) {
                cached_image_chw = core::Tensor::empty(image_shape, core::Device::CUDA, core::DataType::Float32);
            }
            if (cached_image_chw.stream() != fwd_stream) {
                cached_image_chw.set_stream(fwd_stream);
            }
            const core::TensorShape alpha_shape = {1UL, static_cast<size_t>(H), static_cast<size_t>(W)};
            if (!cached_alpha_chw.is_valid() || cached_alpha_chw.shape() != alpha_shape) {
                cached_alpha_chw = core::Tensor::empty(alpha_shape, core::Device::CUDA, core::DataType::Float32);
            }
            if (cached_alpha_chw.stream() != fwd_stream) {
                cached_alpha_chw.set_stream(fwd_stream);
            }
            float* const render_colors_ptr_out = cached_image_chw.ptr<float>();
            float* const render_alphas_ptr_out = cached_alpha_chw.ptr<float>();

            // Setup result struct
            gsplat_lfs::RasterizeWithSHResult result{
                .render_colors = render_colors_ptr_out,
                .render_alphas = render_alphas_ptr_out,
                .radii = radii_ptr_out,
                .means2d = means2d_ptr_out,
                .depths = depths_ptr_out,
                .colors = colors_ptr_out,
                .dirs = (sh_degree > 0) ? dirs_ptr_out : nullptr,
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
                sh0_ptr,
                shN_ptr,
                sh_degree,
                bg_color_ptr,
                bg_image_ptr, // per-pixel background image
                nullptr,      // masks
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
                fwd_stream);
            // isect_ids / flatten_ids are borrowed from TLS high-water cache —
            // never transfer ownership or free on error paths.

            RenderOutput render_output;
            render_output.alpha = cached_alpha_chw;

            switch (render_mode) {
            case GsplatRenderMode::RGB:
                render_output.image = cached_image_chw;
                break;
            case GsplatRenderMode::D:
                render_output.depth = cached_image_chw;
                break;
            case GsplatRenderMode::ED: {
                const core::TensorShape depth_shape = {1UL, static_cast<size_t>(H), static_cast<size_t>(W)};
                if (!cached_depth_chw.is_valid() || cached_depth_chw.shape() != depth_shape ||
                    cached_depth_chw.stream() != fwd_stream) {
                    cached_depth_chw = core::Tensor::empty(depth_shape, core::Device::CUDA, core::DataType::Float32);
                    if (cached_depth_chw.stream() != fwd_stream)
                        cached_depth_chw.set_stream(fwd_stream);
                }
                render_output.depth = cached_image_chw.div(cached_alpha_chw.clamp_min(1e-10f));
                break;
            }
            case GsplatRenderMode::RGB_D:
                render_output.image = cached_image_chw.slice(0, 0, 3);
                render_output.depth = cached_image_chw.slice(0, 3, 4);
                break;
            case GsplatRenderMode::RGB_ED: {
                render_output.image = cached_image_chw.slice(0, 0, 3);
                auto accum_depth = cached_image_chw.slice(0, 3, 4);
                render_output.depth = accum_depth.div(cached_alpha_chw.clamp_min(1e-10f));
                break;
            }
            }

            // NOTE: Background image blending is now handled inside gsplat kernel directly
            // No post-blending needed - bg_image participates in compositing

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
            ctx.dirs_ptr = (sh_degree > 0) ? dirs_ptr_out : nullptr;
            ctx.tile_offsets_ptr = tile_offsets_ptr_out;
            ctx.last_ids_ptr = last_ids_ptr_out;
            ctx.compensations_ptr = compensations_ptr_out;

            // Borrowed TLS high-water pointers (valid through backward; do not free)
            ctx.isect_ids_ptr = result.isect_ids;
            ctx.flatten_ids_ptr = result.flatten_ids;
            ctx.n_isects = result.n_isects;
            ctx.n_sort = result.n_sort;

            // Save input tensors for backward (these are references, not copies)
            ctx.means = means;
            ctx.quats = quats;
            ctx.scales = scales;
            ctx.opacities = opacities;
            ctx.sh0 = sh0;
            // backward calls ctx.shN.ptr<float>(). When resident shN is q16
            // (Float16 bit-patterns), the forward dequant temp is the float4 buffer
            // the kernels actually read — must save that, not the raw codes.
            ctx.shN = shN_dequant_temp.is_valid() ? shN_dequant_temp : shN;

            // Store camera pointers
            ctx.viewmat_ptr = viewmat_ptr;
            ctx.K_ptr = K_ptr;
            ctx.K_tensor = K_tensor;
            ctx.bg_color = bg_color;
            ctx.bg_image = bg_image; // Save bg_image for backward pass

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
            ctx.stream = fwd_stream;
            ctx.render_tile_x_offset = tile_x_offset;
            ctx.render_tile_y_offset = tile_y_offset;
            ctx.render_tile_width = tile_width;
            ctx.render_tile_height = tile_height;

            return std::pair{render_output, ctx};
        } catch (...) {
            // Isect buffers are TLS high-water — leave them; only unwind arena.
            // End on the same stream begin_frame used (same guard → same value),
            // not the streamless device-sync path, so the arena frame chain stays
            // intact for the next frame instead of falling back to a full sync.
            arena.end_frame(frame_id, core::getCurrentCUDAStream());
            throw;
        }
    }

    void gsplat_rasterize_backward(
        const GsplatRasterizeContext& ctx,
        const core::Tensor& grad_image,
        const core::Tensor& grad_alpha,
        core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const core::Tensor& pixel_error_map) {

        // Get arena for temporary allocations
        auto& arena = core::GlobalArenaManager::instance().get_arena();
        auto arena_allocator = arena.get_allocator(ctx.frame_id);
        // Run the backward work + arena frame release on the exact stream the
        // forward began the frame on (ctx.stream), so begin_frame and end_frame
        // chain on the same stream rather than relying on the caller's guard
        // matching. Falls back to the current/tensor stream only if unset.
        const cudaStream_t stream = ctx.stream
                                        ? ctx.stream
                                        : (core::getCurrentCUDAStream()
                                               ? core::getCurrentCUDAStream()
                                               : ctx.means.stream());
        try {

            const uint32_t N = ctx.N;
            const uint32_t K = ctx.K_sh;
            const uint32_t H = ctx.image_height;
            const uint32_t W = ctx.image_width;
            const uint32_t channels = ctx.channels;

            // Calculate sizes for arena allocation
            auto align = [](size_t size, size_t alignment = 128) {
                return (size + alignment - 1) & ~(alignment - 1);
            };

            const bool have_color_grad = grad_image.is_valid() && grad_image.numel() > 0;
            const bool have_alpha_grad = grad_alpha.is_valid() && grad_alpha.numel() > 0;
            size_t v_render_colors_size = have_color_grad ? 0 : align(H * W * channels * sizeof(float));
            size_t v_render_alphas_size = have_alpha_grad ? 0 : align(H * W * sizeof(float));
            size_t v_means_size = align(N * 3 * sizeof(float));
            size_t v_quats_size = align(N * 4 * sizeof(float));
            size_t v_scales_size = align(N * 3 * sizeof(float));
            size_t v_opacities_size = align(N * sizeof(float));
            size_t v_sh_coeffs_size = align(N * K * 3 * sizeof(float));

            size_t total_bwd_size = v_render_colors_size + v_render_alphas_size +
                                    v_means_size + v_quats_size + v_scales_size +
                                    v_opacities_size + v_sh_coeffs_size;

            char* bwd_blob = arena_allocator(total_bwd_size);
            if (bwd_blob == nullptr) {
                throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::ResourceExhausted,
                    .domain = lfs::ErrorDomain::CUDA,
                    .user_message = "Ran out of GPU memory during training backward.",
                    .detail = std::format("gsplat backward arena allocation failed ({} bytes)", total_bwd_size),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }

            // Carve out backward buffers
            char* bwd_ptr = bwd_blob;
            float* v_render_colors_ptr = nullptr;
            float* v_render_alphas_ptr = nullptr;
            if (!have_color_grad) {
                v_render_colors_ptr = reinterpret_cast<float*>(bwd_ptr);
                bwd_ptr += v_render_colors_size;
            } else {
                v_render_colors_ptr = const_cast<float*>(grad_image.ptr<float>());
            }
            if (!have_alpha_grad) {
                v_render_alphas_ptr = reinterpret_cast<float*>(bwd_ptr);
                bwd_ptr += v_render_alphas_size;
            } else {
                v_render_alphas_ptr = const_cast<float*>(grad_alpha.ptr<float>());
            }
            auto* v_means_ptr = reinterpret_cast<float*>(bwd_ptr);
            bwd_ptr += v_means_size;
            auto* v_quats_ptr = reinterpret_cast<float*>(bwd_ptr);
            bwd_ptr += v_quats_size;
            auto* v_scales_ptr = reinterpret_cast<float*>(bwd_ptr);
            bwd_ptr += v_scales_size;
            auto* v_opacities_ptr = reinterpret_cast<float*>(bwd_ptr);
            bwd_ptr += v_opacities_size;
            auto* v_sh_coeffs_ptr = reinterpret_cast<float*>(bwd_ptr);

            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(bwd_blob, 0, total_bwd_size, stream),
                "gsplat backward gradient arena clear");

            UnscentedTransformParameters ut_params;

            // Get background color and image pointers (same as forward)
            const bool use_bg_image = ctx.bg_image.is_valid() && !ctx.bg_image.is_empty();
            const float* bg_color_ptr = nullptr;
            const float* bg_image_ptr = nullptr;

            if (use_bg_image) {
                // Use per-pixel background image
                bg_image_ptr = ctx.bg_image.ptr<float>();
            } else if (ctx.bg_color.is_valid() && ctx.bg_color.numel() > 0) {
                bg_color_ptr = ctx.bg_color.ptr<float>();
            }

            // Pixel-error densification input ([H, W] or [1, H, W])
            const bool update_densification_info =
                gaussian_model._densification_info.ndim() == 2 &&
                gaussian_model._densification_info.shape()[1] >= N;
            core::Tensor error_map_2d;
            if (update_densification_info && pixel_error_map.is_valid() && pixel_error_map.numel() > 0) {
                error_map_2d = pixel_error_map;
                if (error_map_2d.ndim() == 3 && error_map_2d.shape()[0] == 1) {
                    error_map_2d = error_map_2d.squeeze(0);
                }
                assert(error_map_2d.ndim() == 2 &&
                       static_cast<uint32_t>(error_map_2d.shape()[0]) == H &&
                       static_cast<uint32_t>(error_map_2d.shape()[1]) == W &&
                       "pixel_error_map must have shape [H, W] or [1, H, W]");
                if (error_map_2d.device() != core::Device::CUDA) {
                    error_map_2d = error_map_2d.cuda();
                }
                if (!error_map_2d.is_contiguous()) {
                    error_map_2d = error_map_2d.contiguous();
                }
            }
            float* const densification_info_ptr = update_densification_info
                                                      ? gaussian_model._densification_info.ptr<float>()
                                                      : nullptr;
            const float* const pixel_error_map_ptr = (update_densification_info && error_map_2d.is_valid())
                                                         ? error_map_2d.ptr<float>()
                                                         : nullptr;

            // Call backward with raw pointers
            gsplat_lfs::rasterize_from_world_with_sh_bwd(
                ctx.means.ptr<float>(),
                ctx.quats.ptr<float>(),
                ctx.scales.ptr<float>(),
                ctx.opacities.ptr<float>(),
                ctx.sh0.ptr<float>(),
                (ctx.sh_degree > 0 && ctx.shN.is_valid() && ctx.shN.numel() > 0) ? ctx.shN.ptr<float>() : nullptr,
                ctx.sh_degree,
                bg_color_ptr,
                bg_image_ptr, // per-pixel background image
                nullptr,      // masks
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
                ctx.n_sort > 0 ? static_cast<uint32_t>(ctx.n_sort)
                               : static_cast<uint32_t>(ctx.n_isects),
                ctx.colors_ptr,
                ctx.dirs_ptr,
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
                densification_info_ptr,
                pixel_error_map_ptr,
                stream);

            // ============ Accumulate gradients into optimizer using CUDA kernels ============
            // This avoids any tensor operations that might allocate from memory pool

            // Means: [N, 3] -> [N, 3]
            auto& means_grad = optimizer.get_grad(ParamType::Means);
            means_grad.set_stream(stream);
            kernels::launch_grad_accumulate(
                means_grad.ptr<float>(),
                v_means_ptr,
                N * 3,
                stream);

            // Scales: [N, 3] -> [N, 3]
            auto& scaling_grad = optimizer.get_grad(ParamType::Scaling);
            scaling_grad.set_stream(stream);
            kernels::launch_grad_accumulate(
                scaling_grad.ptr<float>(),
                v_scales_ptr,
                N * 3,
                stream);

            // Rotations: [N, 4] -> [N, 4]
            auto& rotation_grad = optimizer.get_grad(ParamType::Rotation);
            rotation_grad.set_stream(stream);
            kernels::launch_grad_accumulate(
                rotation_grad.ptr<float>(),
                v_quats_ptr,
                N * 4,
                stream);

            // Opacities: [N] -> [N, 1] (same memory layout)
            auto& opacity_grad = optimizer.get_grad(ParamType::Opacity);
            opacity_grad.set_stream(stream);
            kernels::launch_grad_accumulate_unsqueeze(
                opacity_grad.ptr<float>(),
                v_opacities_ptr,
                N,
                stream);

            // SH coefficients: [N, K, 3] -> sh0 [N, 1, 3] + swizzled shN.
            float* dst_shN = nullptr;
            if (K > 1) {
                auto& shN_grad = optimizer.get_grad(ParamType::ShN);
                if (shN_grad.is_valid() && shN_grad.numel() > 0) {
                    shN_grad.set_stream(stream);
                    dst_shN = shN_grad.ptr<float>();
                }
            }

            auto& sh0_grad = optimizer.get_grad(ParamType::Sh0);
            sh0_grad.set_stream(stream);
            kernels::launch_grad_accumulate_sh_swizzled(
                sh0_grad.ptr<float>(),
                dst_shN,
                v_sh_coeffs_ptr,
                N,
                K,
                static_cast<std::uint32_t>(gaussian_model.max_sh_coeffs_rest()),
                stream);

            // Accumulate gradient norms when pixel-error map is not provided
            if (update_densification_info && pixel_error_map_ptr == nullptr) {
                gaussian_model._densification_info.set_stream(stream);
                kernels::launch_grad_norm_accumulate(
                    gaussian_model._densification_info.ptr<float>(),
                    v_means_ptr,
                    N,
                    stream);
            }

            // Isect/flatten ids stay in the TLS high-water cache for the next
            // forward (no per-step cudaFree). Arena still ends with the frame.
            arena.end_frame(ctx.frame_id, stream);
        } catch (...) {
            arena.end_frame(ctx.frame_id, stream);
            throw;
        }
    }

    bool release_gsplat_rasterizer_thread_local_caches() noexcept {
        gsplat_thread_caches.staging.release();
        gsplat_thread_caches.K = {};
        gsplat_thread_caches.radial = {};
        gsplat_thread_caches.tangential = {};
        gsplat_thread_caches.thin_prism = {};
        gsplat_thread_caches.image_chw = {};
        gsplat_thread_caches.alpha_chw = {};
        gsplat_thread_caches.depth_chw = {};
        gsplat_thread_caches.shN_dequant = {};
        return !gsplat_thread_caches.K.is_valid() &&
               !gsplat_thread_caches.image_chw.is_valid() &&
               !gsplat_thread_caches.alpha_chw.is_valid() &&
               !gsplat_thread_caches.depth_chw.is_valid();
    }

    namespace {
        // main-thread TLS gsplat caches released before pool shutdown.
        const bool g_gsplat_tls_release_hook_registered = [] {
            lfs::core::register_gpu_pre_shutdown_hook([]() noexcept {
                (void)release_gsplat_rasterizer_thread_local_caches();
                (void)gsplat_lfs::release_intersect_thread_local_cache();
            });
            return true;
        }();
    } // namespace

} // namespace lfs::training
