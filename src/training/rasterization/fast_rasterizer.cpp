/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "training/kernels/grad_alpha.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>

namespace lfs::training {

    /**
     * @brief Dumps all rasterizer input data when a crash occurs for debugging.
     *
     * Creates a directory in the CURRENT WORKING DIRECTORY with the format:
     *   crash_dump_YYYYMMDD_HHMMSS_MMM/
     *
     * Where YYYYMMDD_HHMMSS is the timestamp and MMM is milliseconds.
     *
     * The directory contains:
     *   - means.tensor         : float32 [N, 3] - Gaussian positions
     *   - raw_scales.tensor    : float32 [N, 3] - Raw scale parameters (pre-activation)
     *   - raw_rotations.tensor : float32 [N, 4] - Raw rotation quaternions (pre-normalization)
     *   - raw_opacities.tensor : float32 [N, 1] - Raw opacity values (pre-sigmoid)
     *   - sh0.tensor           : float32 [N, 3] - DC spherical harmonic coefficients
     *   - shN.tensor           : float32 [N, K, 3] - Higher-order SH coefficients (K = total_bases_sh_rest)
     *   - w2c.tensor           : float32 [1, 4, 4] - World-to-camera transformation matrix
     *   - cam_position.tensor  : float32 [3] - Camera position in world coordinates
     *   - params.json          : JSON file with scalar parameters and tensor shapes
     *
     * Tensor file format (.tensor):
     *   - Header: magic (4B) + version (4B) + dtype (1B) + device (1B) + rank (2B) + numel (8B)
     *   - Shape: rank * uint64 dimension values
     *   - Data: raw float32 values (always saved from CPU, regardless of original device)
     *
     * To reload tensors in code:
     *   auto tensor = lfs::core::load_tensor("crash_dump_.../means.tensor");
     *
     * @param error_msg The exception message that triggered the crash
     * @param means Gaussian positions tensor [N, 3]
     * @param raw_scales Raw scale parameters [N, 3]
     * @param raw_rotations Raw rotation quaternions [N, 4]
     * @param raw_opacities Raw opacity values [N, 1]
     * @param sh0 DC spherical harmonic coefficients [N, 3]
     * @param shN Higher-order SH coefficients [N, K, 3]
     * @param w2c World-to-camera transform [1, 4, 4]
     * @param cam_position Camera position [3]
     * @param n_primitives Number of Gaussians
     * @param active_sh_bases Number of active SH bases: (sh_degree+1)^2
     * @param total_bases_sh_rest Total higher-order SH bases (K dimension of shN)
     * @param width Render width in pixels
     * @param height Render height in pixels
     * @param fx Focal length x
     * @param fy Focal length y
     * @param cx Principal point x (adjusted for tile offset)
     * @param cy Principal point y (adjusted for tile offset)
     * @param near_plane Near clipping plane
     * @param far_plane Far clipping plane
     */
    static void dump_crash_data(
        const std::string& error_msg,
        const core::Tensor& means,
        const core::Tensor& raw_scales,
        const core::Tensor& raw_rotations,
        const core::Tensor& raw_opacities,
        const core::Tensor& sh0,
        const core::Tensor& shN,
        const core::Tensor& w2c,
        const core::Tensor& cam_position,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float near_plane,
        float far_plane) {

        // Create crash dump directory with timestamp in CURRENT WORKING DIRECTORY
        // Example: ./crash_dump_20251211_143052_847/
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", std::localtime(&time_t));

        // Directory path is relative to cwd, e.g. "./crash_dump_20251211_143052_847"
        std::string dump_dir = std::string("crash_dump_") + time_buf + "_" + std::to_string(ms.count());
        std::filesystem::create_directories(dump_dir);

        // Log absolute path for easier debugging
        auto abs_path = std::filesystem::absolute(dump_dir);

        LOG_ERROR("Rasterizer crash! Dumping data to: {}", abs_path.string());
        LOG_ERROR("Error: {}", error_msg);

        try {
            // Dump tensors as binary .tensor files
            // Each file contains: header + shape dims + raw float32 data
            // Tensors are copied to CPU before saving if they're on CUDA
            if (means.is_valid())
                core::save_tensor(means, dump_dir + "/means.tensor"); // [N, 3]
            if (raw_scales.is_valid())
                core::save_tensor(raw_scales, dump_dir + "/raw_scales.tensor"); // [N, 3]
            if (raw_rotations.is_valid())
                core::save_tensor(raw_rotations, dump_dir + "/raw_rotations.tensor"); // [N, 4]
            if (raw_opacities.is_valid())
                core::save_tensor(raw_opacities, dump_dir + "/raw_opacities.tensor"); // [N, 1]
            if (sh0.is_valid())
                core::save_tensor(sh0, dump_dir + "/sh0.tensor"); // [N, 3]
            if (shN.is_valid())
                core::save_tensor(shN, dump_dir + "/shN.tensor"); // [N, K, 3]
            if (w2c.is_valid())
                core::save_tensor(w2c, dump_dir + "/w2c.tensor"); // [1, 4, 4]
            if (cam_position.is_valid())
                core::save_tensor(cam_position, dump_dir + "/cam_position.tensor"); // [3]

            // Dump scalar parameters to params.json
            // This is a human-readable JSON file containing:
            // - error: The exception message
            // - n_primitives: Number of Gaussians (N)
            // - active_sh_bases: (sh_degree+1)^2, e.g., 1 for degree 0, 4 for degree 1
            // - total_bases_sh_rest: K dimension of shN tensor
            // - width, height: Render dimensions in pixels
            // - fx, fy, cx, cy: Camera intrinsics
            // - near_plane, far_plane: Clipping planes
            // - *_shape: Shape of each tensor for verification
            std::ofstream params_file(dump_dir + "/params.json");
            if (params_file) {
                params_file << "{\n";
                params_file << "  \"error\": \"" << error_msg << "\",\n";
                params_file << "  \"n_primitives\": " << n_primitives << ",\n";
                params_file << "  \"active_sh_bases\": " << active_sh_bases << ",\n";
                params_file << "  \"total_bases_sh_rest\": " << total_bases_sh_rest << ",\n";
                params_file << "  \"width\": " << width << ",\n";
                params_file << "  \"height\": " << height << ",\n";
                params_file << "  \"fx\": " << fx << ",\n";
                params_file << "  \"fy\": " << fy << ",\n";
                params_file << "  \"cx\": " << cx << ",\n";
                params_file << "  \"cy\": " << cy << ",\n";
                params_file << "  \"near_plane\": " << near_plane << ",\n";
                params_file << "  \"far_plane\": " << far_plane << ",\n";
                params_file << "  \"means_shape\": [" << means.shape()[0];
                for (size_t i = 1; i < means.ndim(); ++i)
                    params_file << ", " << means.shape()[i];
                params_file << "],\n";
                params_file << "  \"raw_scales_shape\": [" << raw_scales.shape()[0];
                for (size_t i = 1; i < raw_scales.ndim(); ++i)
                    params_file << ", " << raw_scales.shape()[i];
                params_file << "],\n";
                params_file << "  \"raw_rotations_shape\": [" << raw_rotations.shape()[0];
                for (size_t i = 1; i < raw_rotations.ndim(); ++i)
                    params_file << ", " << raw_rotations.shape()[i];
                params_file << "],\n";
                params_file << "  \"raw_opacities_shape\": [" << raw_opacities.shape()[0];
                for (size_t i = 1; i < raw_opacities.ndim(); ++i)
                    params_file << ", " << raw_opacities.shape()[i];
                params_file << "],\n";
                params_file << "  \"sh0_shape\": [" << sh0.shape()[0];
                for (size_t i = 1; i < sh0.ndim(); ++i)
                    params_file << ", " << sh0.shape()[i];
                params_file << "],\n";
                params_file << "  \"shN_shape\": [" << shN.shape()[0];
                for (size_t i = 1; i < shN.ndim(); ++i)
                    params_file << ", " << shN.shape()[i];
                params_file << "]\n";
                params_file << "}\n";
            }

            LOG_ERROR("Crash dump complete: {}", abs_path.string());
        } catch (const std::exception& dump_error) {
            LOG_ERROR("Failed to create crash dump: {}", dump_error.what());
        }
    }

    std::expected<std::pair<RenderOutput, FastRasterizeContext>, std::string> fast_rasterize_forward(
        core::Camera& viewpoint_camera,
        core::SplatData& gaussian_model,
        core::Tensor& bg_color,
        int tile_x_offset,
        int tile_y_offset,
        int tile_width,
        int tile_height,
        bool mip_filter) {
        // Get camera parameters
        const int full_width = viewpoint_camera.image_width();
        const int full_height = viewpoint_camera.image_height();

        // Determine tile dimensions (tile_width/height=0 means render full image)
        const int width = (tile_width > 0) ? tile_width : full_width;
        const int height = (tile_height > 0) ? tile_height : full_height;

        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Adjust camera center point for tile rendering
        // When rendering a tile at offset, the principal point shifts
        const float cx_adjusted = cx - static_cast<float>(tile_x_offset);
        const float cy_adjusted = cy - static_cast<float>(tile_y_offset);

        // Get Gaussian parameters
        auto& means = gaussian_model.means();
        auto& raw_opacities = gaussian_model.opacity_raw();
        auto& raw_scales = gaussian_model.scaling_raw();
        auto& raw_rotations = gaussian_model.rotation_raw();
        auto& sh0 = gaussian_model.sh0();
        auto& shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        // Get direct GPU pointers (tensors are already contiguous on CUDA)
        const float* w2c_ptr = viewpoint_camera.world_view_transform_ptr();
        const float* cam_position_ptr = viewpoint_camera.cam_position_ptr();

        const int n_primitives = static_cast<int>(means.shape()[0]);
        const int total_bases_sh_rest = (shN.is_valid() && shN.ndim() >= 2)
                                            ? static_cast<int>(shN.shape()[1])
                                            : 0;

        // Pre-allocate output tensors (reused across iterations)
        thread_local core::Tensor image;
        thread_local core::Tensor alpha;
        thread_local core::Tensor output_image;
        thread_local int last_width = -1;
        thread_local int last_height = -1;

        // Only reallocate if dimensions changed
        if (last_width != width || last_height != height) {
            image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)});
            alpha = core::Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            output_image = core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)}, core::Device::CUDA);
            last_width = width;
            last_height = height;
        }

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        // Use adjusted cx/cy for tile rendering
        fast_lfs::rasterization::ForwardContext forward_ctx;
        try {
            forward_ctx = fast_lfs::rasterization::forward_raw(
                means.ptr<float>(),
                raw_scales.ptr<float>(),
                raw_rotations.ptr<float>(),
                raw_opacities.ptr<float>(),
                sh0.ptr<float>(),
                shN.ptr<float>(),
                w2c_ptr,
                cam_position_ptr,
                image.ptr<float>(),
                alpha.ptr<float>(),
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                fx,
                fy,
                cx_adjusted, // Use adjusted cx for tile offset
                cy_adjusted, // Use adjusted cy for tile offset
                near_plane,
                far_plane,
                mip_filter);
        } catch (const std::exception& e) {
            // Dump all input data for debugging
            dump_crash_data(
                e.what(),
                means,
                raw_scales,
                raw_rotations,
                raw_opacities,
                sh0,
                shN,
                viewpoint_camera.world_view_transform(),
                viewpoint_camera.cam_position(),
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                fx,
                fy,
                cx_adjusted,
                cy_adjusted,
                near_plane,
                far_plane);
            throw; // Re-throw after dumping
        } catch (...) {
            // Handle non-std::exception crashes
            dump_crash_data(
                "Unknown exception (not std::exception)",
                means,
                raw_scales,
                raw_rotations,
                raw_opacities,
                sh0,
                shN,
                viewpoint_camera.world_view_transform(),
                viewpoint_camera.cam_position(),
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                fx,
                fy,
                cx_adjusted,
                cy_adjusted,
                near_plane,
                far_plane);
            throw; // Re-throw after dumping
        }

        // Check if forward failed due to OOM
        if (!forward_ctx.success) {
            return std::unexpected(std::string(forward_ctx.error_message));
        }

        // Prepare render output
        RenderOutput render_output;
        // output = image + (1 - alpha) * bg_color
        // (output_image is pre-allocated above)

        kernels::launch_fused_background_blend(
            image.ptr<float>(),
            alpha.ptr<float>(),
            bg_color.ptr<float>(),
            output_image.ptr<float>(),
            height,
            width,
            nullptr // default stream
        );

        render_output.image = output_image;
        render_output.alpha = alpha;
        render_output.width = width;
        render_output.height = height;

        // Prepare context for backward
        FastRasterizeContext ctx;
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color; // Save bg_color for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.raw_opacities = raw_opacities;
        ctx.shN = shN;

        // Store camera pointers directly (tensors are managed by camera, already contiguous)
        ctx.w2c_ptr = w2c_ptr;
        ctx.cam_position_ptr = cam_position_ptr;

        // Store forward context (contains buffer pointers, frame_id, etc.)
        ctx.forward_ctx = forward_ctx;

        ctx.active_sh_bases = active_sh_bases;
        ctx.total_bases_sh_rest = total_bases_sh_rest;
        ctx.width = width;
        ctx.height = height;
        ctx.focal_x = fx;
        ctx.focal_y = fy;
        ctx.center_x = cx_adjusted; // Store adjusted cx for backward
        ctx.center_y = cy_adjusted; // Store adjusted cy for backward
        ctx.near_plane = near_plane;
        ctx.far_plane = far_plane;
        ctx.mip_filter = mip_filter;

        // Store tile information
        ctx.tile_x_offset = tile_x_offset;
        ctx.tile_y_offset = tile_y_offset;
        ctx.tile_width = tile_width;
        ctx.tile_height = tile_height;

        return std::pair{render_output, ctx};
    }

    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const core::Tensor& grad_image,
        core::SplatData& gaussian_model,
        AdamOptimizer& optimizer,
        const core::Tensor& grad_alpha_extra) {

        // Compute grad_alpha from background blending: output = image + (1 - alpha) * bg
        int H, W;
        bool is_chw_layout;

        if (grad_image.shape()[0] == 3) {
            is_chw_layout = true;
            H = static_cast<int>(grad_image.shape()[1]);
            W = static_cast<int>(grad_image.shape()[2]);
        } else if (grad_image.shape()[2] == 3) {
            is_chw_layout = false;
            H = static_cast<int>(grad_image.shape()[0]);
            W = static_cast<int>(grad_image.shape()[1]);
        } else {
            throw std::runtime_error("Unexpected grad_image shape");
        }

        auto grad_alpha = core::Tensor::empty({static_cast<size_t>(H), static_cast<size_t>(W)}, core::Device::CUDA);

        kernels::launch_fused_grad_alpha(
            grad_image.ptr<float>(),
            ctx.bg_color.ptr<float>(),
            grad_alpha.ptr<float>(),
            H, W,
            is_chw_layout,
            nullptr);

        if (grad_alpha_extra.is_valid() && grad_alpha_extra.numel() > 0) {
            grad_alpha = grad_alpha + grad_alpha_extra;
        }

        const int n_primitives = static_cast<int>(ctx.means.shape()[0]);
        // densification_info has shape [2, N]
        const bool update_densification_info = gaussian_model._densification_info.ndim() == 2 &&
                                               gaussian_model._densification_info.shape()[1] >= static_cast<size_t>(n_primitives);

        // Get gradient pointers from optimizer
        auto backward_result = fast_lfs::rasterization::backward_raw(
            update_densification_info ? gaussian_model._densification_info.ptr<float>() : nullptr,
            grad_image.ptr<float>(),
            grad_alpha.ptr<float>(),
            ctx.image.ptr<float>(),
            ctx.alpha.ptr<float>(),
            ctx.means.ptr<float>(),
            ctx.raw_scales.ptr<float>(),
            ctx.raw_rotations.ptr<float>(),
            ctx.raw_opacities.ptr<float>(),
            ctx.shN.ptr<float>(),
            ctx.w2c_ptr,
            ctx.cam_position_ptr,
            ctx.forward_ctx,
            optimizer.get_grad(ParamType::Means).ptr<float>(),
            optimizer.get_grad(ParamType::Scaling).ptr<float>(),
            optimizer.get_grad(ParamType::Rotation).ptr<float>(),
            optimizer.get_grad(ParamType::Opacity).ptr<float>(),
            optimizer.get_grad(ParamType::Sh0).ptr<float>(),
            optimizer.get_grad(ParamType::ShN).ptr<float>(),
            nullptr,
            n_primitives,
            ctx.active_sh_bases,
            ctx.total_bases_sh_rest,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y,
            ctx.mip_filter);

        if (!backward_result.success) {
            throw std::runtime_error(std::string("Backward failed: ") + backward_result.error_message);
        }
    }
} // namespace lfs::training
