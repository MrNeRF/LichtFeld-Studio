/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/sh_layout.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/ops/fast_cuda.hpp"
#include "lfs/training/perf_bench.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "lfs/training/vram_ledger.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace lfs::training {

    // Forward pass context - holds intermediate buffers needed for backward
    struct CudaFastFrame {
        CudaFastFrame() = default;
        ~CudaFastFrame() {
            release_forward_context();
        }

        CudaFastFrame(const CudaFastFrame&) = delete;
        CudaFastFrame& operator=(const CudaFastFrame&) = delete;

        CudaFastFrame(CudaFastFrame&& other) noexcept {
            move_from(std::move(other));
        }

        CudaFastFrame& operator=(CudaFastFrame&& other) noexcept {
            if (this != &other) {
                release_forward_context();
                move_from(std::move(other));
            }
            return *this;
        }

        lfs::core::Tensor image;
        lfs::core::Tensor alpha;
        lfs::core::Tensor bg_color; // Saved for alpha gradient computation

        // Gaussian parameters (saved to avoid re-fetching in backward)
        lfs::core::Tensor means;
        lfs::core::Tensor raw_scales;
        lfs::core::Tensor raw_rotations;
        lfs::core::Tensor raw_opacities;
        lfs::core::Tensor shN;

        const float* w2c_ptr = nullptr;
        const float* cam_position_ptr = nullptr;

        // Forward context (contains buffer pointers, frame_id, etc.)
        fast_lfs::rasterization::ForwardContext forward_ctx = {};
        // Last queue using the frame, retained beyond execution-scope lifetime.
        cudaStream_t completion_stream = nullptr;

        int active_sh_bases = 0;
        int width = 0;
        int height = 0;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
        bool mip_filter = false;

        // Background image for per-pixel blending (optional, empty = use bg_color)
        lfs::core::Tensor bg_image;
        // Q16 bounds handle captured with the forward. Empty unless that forward was Q16.
        lfs::core::Tensor sh_value_bounds;

        void set_forward_context(fast_lfs::rasterization::ForwardContext ctx) noexcept {
            release_forward_context();
            forward_ctx = ctx;
            completion_stream = ctx.stream;
            owns_forward_context_ = ctx.success;
        }

        void release_forward_context() noexcept {
            if (!owns_forward_context_) {
                return;
            }
            owns_forward_context_ = false;
            fast_lfs::rasterization::release_forward_context(forward_ctx, completion_stream);
            forward_ctx = {};
            completion_stream = nullptr;
        }

        void mark_forward_context_released() noexcept {
            owns_forward_context_ = false;
            forward_ctx = {};
            completion_stream = nullptr;
        }

    private:
        bool owns_forward_context_ = false;

        void move_from(CudaFastFrame&& other) noexcept {
            image = std::move(other.image);
            alpha = std::move(other.alpha);
            bg_color = std::move(other.bg_color);
            means = std::move(other.means);
            raw_scales = std::move(other.raw_scales);
            raw_rotations = std::move(other.raw_rotations);
            raw_opacities = std::move(other.raw_opacities);
            shN = std::move(other.shN);
            w2c_ptr = std::exchange(other.w2c_ptr, nullptr);
            cam_position_ptr = std::exchange(other.cam_position_ptr, nullptr);
            forward_ctx = std::exchange(other.forward_ctx, {});
            completion_stream = std::exchange(other.completion_stream, nullptr);
            active_sh_bases = std::exchange(other.active_sh_bases, 0);
            width = std::exchange(other.width, 0);
            height = std::exchange(other.height, 0);
            focal_x = std::exchange(other.focal_x, 0.0f);
            focal_y = std::exchange(other.focal_y, 0.0f);
            center_x = std::exchange(other.center_x, 0.0f);
            center_y = std::exchange(other.center_y, 0.0f);
            mip_filter = std::exchange(other.mip_filter, false);
            bg_image = std::move(other.bg_image);
            sh_value_bounds = std::move(other.sh_value_bounds);
            owns_forward_context_ = std::exchange(other.owns_forward_context_, false);
        }
    };

    namespace {
        struct FastCaches {
            core::Tensor image;
            core::Tensor alpha;
            core::Tensor depth;
            core::Tensor normal;
            int width = -1;
            int height = -1;
        };

        struct CudaFastState : lfs::gpu_ops::BackendState {
            FastCaches caches;
            CudaFastFrame frame;
            std::string message;
        };

        [[nodiscard]] CudaFastState& state_of(lfs::gpu_ops::FastSaved& saved) {
            if (!saved.backend) {
                throw std::logic_error("fast raster op called without a created state");
            }
            return static_cast<CudaFastState&>(*saved.backend);
        }

        [[nodiscard]] const CudaFastState* state_of(const lfs::gpu_ops::FastSaved& saved) {
            return static_cast<const CudaFastState*>(saved.backend.get());
        }

        [[nodiscard]] int checked_dim_to_int(size_t value, const char* name) {
            if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error(std::string(name) + " exceeds int range");
            }
            return static_cast<int>(value);
        }

        [[nodiscard]] bool has_background_image(const core::Tensor& bg_image) {
            return bg_image.is_valid() && !bg_image.is_empty();
        }

    } // namespace

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
     *   - shN.tensor           : float32 [swizzled_floats] - vksplat swizzled higher-order SH
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
     * @param shN Higher-order SH coefficients in vksplat swizzled layout
     * @param w2c World-to-camera transform [1, 4, 4]
     * @param cam_position Camera position [3]
     * @param n_primitives Number of Gaussians
     * @param active_sh_bases Number of active SH bases: (sh_degree+1)^2
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
        auto time_t_val = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

        std::tm tm{};
#ifdef _WIN32
        localtime_s(&tm, &time_t_val);
#else
        localtime_r(&time_t_val, &tm);
#endif
        char time_buf[64];
        std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", &tm);

        // Directory path is relative to cwd, e.g. "./crash_dump_20251211_143052_847"
        std::string dump_dir = std::string("crash_dump_") + time_buf + "_" + std::to_string(ms.count());
        std::filesystem::create_directories(dump_dir);

        // Log absolute path for easier debugging
        auto abs_path = std::filesystem::absolute(dump_dir);

        LOG_ERROR("Rasterizer crash! Dumping data to: {}", lfs::core::path_to_utf8(abs_path));
        LOG_ERROR("Error: {}", error_msg);

        try {
            // Dump tensors as binary .tensor files
            // Each file contains: header + shape dims + raw float32 data
            // Tensors are copied to CPU before saving if they're on GPU
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
                core::save_tensor(shN, dump_dir + "/shN.tensor"); // swizzled shN
            if (w2c.is_valid())
                core::save_tensor(w2c, dump_dir + "/w2c.tensor"); // [1, 4, 4]
            if (cam_position.is_valid())
                core::save_tensor(cam_position, dump_dir + "/cam_position.tensor"); // [3]

            // Dump scalar parameters to params.json
            // This is a human-readable JSON file containing:
            // - error: The exception message
            // - n_primitives: Number of Gaussians (N)
            // - active_sh_bases: (sh_degree+1)^2, e.g., 1 for degree 0, 4 for degree 1
            // - shN_layout: storage layout of the dumped higher-order SH tensor
            // - width, height: Render dimensions in pixels
            // - fx, fy, cx, cy: Camera intrinsics
            // - near_plane, far_plane: Clipping planes
            // - *_shape: Shape of each tensor for verification
            std::ofstream params_file;
            if (lfs::core::open_file_for_write(std::filesystem::path(dump_dir) / "params.json",
                                               std::ios::out | std::ios::binary, params_file)) {
                params_file << "{\n";
                params_file << "  \"error\": \"" << error_msg << "\",\n";
                params_file << "  \"n_primitives\": " << n_primitives << ",\n";
                params_file << "  \"active_sh_bases\": " << active_sh_bases << ",\n";
                params_file << "  \"shN_layout\": \"swizzled-sh-reorder-32\",\n";
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
                params_file << "],\n";
                // shN is stored in compact vksplat float4-packed swizzled layout
                // (ceil(N/32) * active_slots * 32 * 4 floats). Crash-dump consumers should
                // deswizzle via shAt(p, k) (returns a float4-slot index; multiply by 4 for the
                // float offset) before interpreting as canonical [N, K, 3].
                params_file << "  \"shN_layout\": \"swizzled-sh-reorder-32\"\n";
                params_file << "}\n";
            }

            LOG_ERROR("Crash dump complete: {}", lfs::core::path_to_utf8(abs_path));
        } catch (const std::exception& dump_error) {
            LOG_ERROR("Failed to create crash dump: {}", dump_error.what());
        }
    }

    [[nodiscard]] int degree_from_layout_bases(const uint32_t layout_bases) {
        if (layout_bases >= 16) {
            return 3;
        }
        if (layout_bases >= 9) {
            return 2;
        }
        if (layout_bases >= 4) {
            return 1;
        }
        return 0;
    }

    [[nodiscard]] lfs::gpu_ops::RasterResult raster_fail(
        std::string& slot, const lfs::gpu_ops::RasterResult::Code code, std::string text) {
        slot = std::move(text);
        return {.code = code, .has_work = false, .message = slot};
    }

    lfs::gpu_ops::RasterResult fast_ops_forward(
        lfs::gpu_ops::FastSaved& saved,
        const lfs::gpu_ops::SplatInputs& splats,
        const lfs::gpu_ops::Tensor& view,
        const lfs::gpu_ops::Tensor& camera_position,
        const lfs::gpu_ops::Tensor& bg_color,
        const lfs::gpu_ops::Tensor& bg_image,
        const lfs::gpu_ops::FastParams& params,
        const lfs::gpu_ops::RenderOutputs& outputs,
        lfs::gpu_ops::Tensor& max_screen_share) {
        auto& state = state_of(saved);
        FastCaches& caches = state.caches;
        const auto& means = splats.means;
        const auto& raw_scales = splats.raw_scales;
        const auto& raw_rotations = splats.raw_rotations;
        const auto& raw_opacities = splats.raw_opacities;
        const auto& sh0 = splats.sh0;
        const auto& shN = splats.shN;
        const auto& sh_bounds = splats.sh_value_bounds;

        const int full_width = params.full_image.w;
        const int full_height = params.full_image.h;
        const int width = (params.tile_w > 0) ? params.tile_w : full_width;
        const int height = (params.tile_h > 0) ? params.tile_h : full_height;
        const float fx = params.intrinsics.fx;
        const float fy = params.intrinsics.fy;
        const float cx_adjusted = params.intrinsics.cx - static_cast<float>(params.tile_x);
        const float cy_adjusted = params.intrinsics.cy - static_cast<float>(params.tile_y);
        const int active_sh_bases = static_cast<int>(params.sh.active_bases);
        const int sh_layout_bases = static_cast<int>(params.sh.layout_bases);
        const bool mip_filter = params.mip_filter;
        const bool render_normal = params.render_normal;
        const bool render_depth = params.render_depth;
        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        const int n_primitives = checked_dim_to_int(means.shape()[0], "n_primitives");
        if (n_primitives == 0) {
            return raster_fail(state.message, lfs::gpu_ops::RasterResult::Code::Failed,
                               "n_primitives is 0 - model has no gaussians");
        }

        for (const auto* input : std::initializer_list<const core::Tensor*>{
                 &means, &raw_scales, &raw_rotations, &raw_opacities, &sh0, &shN,
                 &bg_color, &bg_image, &view, &camera_position}) {
            if (input->is_valid())
                input->sync_to_stream(lfs::core::getCurrentCUDAStream());
        }
        const float* w2c_ptr = view.ptr<float>();
        const float* cam_position_ptr = camera_position.ptr<float>();
        // Pre-allocate output tensors (reused across iterations)
        auto& image = caches.image;
        auto& alpha = caches.alpha;
        auto& depth = caches.depth;
        auto& normal = caches.normal;
        auto& last_width = caches.width;
        auto& last_height = caches.height;

        const cudaStream_t raster_stream = lfs::core::getCurrentCUDAStream();

        // Reallocate when either the shape or owning stream changes. Calling
        // Tensor::set_stream on a cache backed by a destroyed stream would try
        // to bridge from that dead handle before re-homing it.
        if (!image.is_valid() || !alpha.is_valid() ||
            last_width != width || last_height != height ||
            image.stream() != raster_stream || alpha.stream() != raster_stream) {
            image = core::Tensor::empty_exact({3, static_cast<size_t>(height), static_cast<size_t>(width)});
            alpha = core::Tensor::empty_exact({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            depth = core::Tensor();
            normal = core::Tensor();
            if (image.stream() != raster_stream)
                image.set_stream(raster_stream);
            if (alpha.stream() != raster_stream)
                alpha.set_stream(raster_stream);
            last_width = width;
            last_height = height;
        }
        if (!render_depth) {
            depth = core::Tensor();
        } else if (!depth.is_valid() || depth.stream() != raster_stream) {
            depth = core::Tensor::empty_exact({1, static_cast<size_t>(height), static_cast<size_t>(width)});
            if (depth.stream() != raster_stream)
                depth.set_stream(raster_stream);
        }
        if (render_normal &&
            (!normal.is_valid() ||
             normal.shape() != core::TensorShape({3, static_cast<size_t>(height), static_cast<size_t>(width)}) ||
             normal.stream() != raster_stream)) {
            normal = core::Tensor::empty_exact({3, static_cast<size_t>(height), static_cast<size_t>(width)});
            if (normal.stream() != raster_stream)
                normal.set_stream(raster_stream);
        }

        if (max_screen_share.is_valid())
            max_screen_share.set_stream(raster_stream);

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        // Use adjusted cx/cy for tile rendering
        fast_lfs::rasterization::ForwardContext forward_ctx;
        try {
            const float* bg_image_ptr =
                has_background_image(bg_image) ? bg_image.ptr<float>() : nullptr;
            const float* bg_color_ptr =
                bg_image_ptr != nullptr
                    ? nullptr
                    : (bg_color.is_valid() && bg_color.numel() >= 3 ? bg_color.ptr<float>() : nullptr);
            // q16: Float16 codes + bounds. IEEE f16 float4-swizzle: Float16 without
            // bounds (exportable GUI). fp32: Float32 float4-swizzle.
            // Generation-checked fetch: never pass a baked exportable pointer that
            // survived a capacity grow.
            const bool shN_q16 = params.sh.storage == lfs::gpu_ops::ShStorage::Q16;
            const bool shN_f16 = params.sh.storage == lfs::gpu_ops::ShStorage::IeeeFloat16;
            const float* shN_ptr = nullptr;
            const float* shN_bounds_ptr = nullptr;
            unsigned shN_n_cells = 0u;
            if (shN_q16) {
                shN_ptr = static_cast<const float*>(lfs::core::resolve_exportable_device_ptr(shN));
                if (sh_bounds.is_valid() && sh_bounds.numel() > 0) {
                    shN_bounds_ptr = static_cast<const float*>(
                        lfs::core::resolve_exportable_device_ptr(sh_bounds));
                }
                shN_n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(
                    lfs::core::sh_rest_coefficients_for_degree(
                        degree_from_layout_bases(params.sh.layout_bases)));
            } else if (shN_f16) {
                shN_ptr = static_cast<const float*>(lfs::core::resolve_exportable_device_ptr(shN));
            } else {
                shN_ptr = shN.ptr<float>();
            }
            const unsigned shN_bits = (shN_q16 || shN_f16) ? 16u : 0u;
            forward_ctx = fast_lfs::rasterization::forward_raw(
                means.ptr<float>(),
                raw_scales.ptr<float>(),
                raw_rotations.ptr<float>(),
                raw_opacities.ptr<float>(),
                sh0.ptr<float>(),
                shN_ptr,
                w2c_ptr,
                cam_position_ptr,
                image.ptr<float>(),
                alpha.ptr<float>(),
                render_depth ? depth.ptr<float>() : nullptr,
                render_normal ? normal.ptr<float>() : nullptr,
                bg_color_ptr,
                bg_image_ptr,
                n_primitives,
                active_sh_bases,
                sh_layout_bases,
                width,
                height,
                fx,
                fy,
                cx_adjusted, // Use adjusted cx for tile offset
                cy_adjusted, // Use adjusted cy for tile offset
                near_plane,
                far_plane,
                mip_filter,
                raster_stream,
                shN_bounds_ptr,
                shN_n_cells,
                shN_bits,
                (max_screen_share.is_valid() &&
                 max_screen_share.ndim() == 1 &&
                 max_screen_share.numel() >= static_cast<size_t>(n_primitives))
                    ? max_screen_share.ptr<float>()
                    : nullptr);
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
                view,
                camera_position,
                n_primitives,
                active_sh_bases,
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
                view,
                camera_position,
                n_primitives,
                active_sh_bases,
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

        if (!forward_ctx.success) {
            const std::string message = forward_ctx.error_message
                                            ? forward_ctx.error_message
                                            : "unknown forward failure";
            // instance-count overflow is a bad frame, not a run-killer.
            // FailedPrecondition → trainer skips the step and continues.
            if (forward_ctx.instance_count_overflow) {
                return raster_fail(state.message, lfs::gpu_ops::RasterResult::Code::InstanceOverflow,
                                   message);
            }
            return raster_fail(
                state.message,
                forward_ctx.resource_exhausted ? lfs::gpu_ops::RasterResult::Code::ResourceExhausted
                                               : lfs::gpu_ops::RasterResult::Code::Failed,
                message);
        }

        // Take ownership before any post-forward tensor work so exceptions cannot leave
        // the arena frame active.
        CudaFastFrame ctx;
        ctx.set_forward_context(forward_ctx);

        // Prepare context for backward
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color; // Save bg_color for alpha gradient
        ctx.bg_image = bg_image; // Save bg_image for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.raw_opacities = raw_opacities;
        ctx.shN = shN;
        ctx.sh_value_bounds = sh_bounds;

        // Store camera pointers directly (tensors are managed by camera, already contiguous)
        ctx.w2c_ptr = w2c_ptr;
        ctx.cam_position_ptr = cam_position_ptr;

        ctx.active_sh_bases = active_sh_bases;
        ctx.width = width;
        ctx.height = height;
        ctx.focal_x = fx;
        ctx.focal_y = fy;
        ctx.center_x = cx_adjusted; // Store adjusted cx for backward
        ctx.center_y = cy_adjusted; // Store adjusted cy for backward
        ctx.mip_filter = mip_filter;

        outputs.image = image;
        outputs.alpha = alpha;
        outputs.depth = render_depth ? depth : core::Tensor{};
        outputs.normal = render_normal ? normal : core::Tensor{};
        state.frame = std::move(ctx);
        return {
            .code = lfs::gpu_ops::RasterResult::Code::Success,
            .has_work = state.frame.forward_ctx.n_instances > 0,
            .message = {},
        };
    }

    void fast_ops_backward(
        lfs::gpu_ops::FastSaved& saved,
        const lfs::gpu_ops::RenderGradients& gradients,
        lfs::gpu_ops::Tensor& densification,
        const lfs::gpu_ops::Tensor& error_map,
        const lfs::gpu_ops::Tensor& edge_map,
        lfs::gpu_ops::Tensor& edge_scores,
        const lfs::gpu_ops::BackwardAdam& adam,
        const DensificationType densification_type) {
        auto& state = state_of(saved);
        const auto fused_adam = fast_adam_settings(adam);
        const float* edge_weight = edge_map.is_valid() && edge_map.numel() > 0 ? edge_map.ptr<float>() : nullptr;
        float* edge_score = edge_scores.is_valid() && edge_scores.numel() > 0 ? edge_scores.ptr<float>() : nullptr;
        auto& caches = state.caches;
        auto& ctx = state.frame;
        const auto& grad_image = gradients.image;
        const auto& grad_alpha_extra = gradients.alpha;
        const auto& grad_depth = gradients.depth;
        const auto& grad_normal = gradients.normal;

        if (grad_image.ndim() != 3 || grad_image.shape()[0] != 3) {
            throw std::runtime_error("FastGS backward expects a [3, H, W] image gradient");
        }
        const int H = checked_dim_to_int(grad_image.shape()[1], "grad_image height");
        const int W = checked_dim_to_int(grad_image.shape()[2], "grad_image width");
        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        ctx.completion_stream = stream;
        for (const auto* input : std::initializer_list<const core::Tensor*>{&grad_image, &grad_alpha_extra, &grad_depth, &grad_normal,
                                                                            &ctx.bg_image, &ctx.bg_color, &ctx.image, &ctx.alpha, &error_map}) {
            if (input->is_valid())
                input->sync_to_stream(stream);
        }

        // The blend backward derives the background alpha gradient per pixel
        // from grad_image and the background; no [H, W] map is materialized.
        fast_lfs::rasterization::BackgroundAlphaGradient background_grad;
        if (has_background_image(ctx.bg_image)) {
            core::pin_operands({&grad_image, &ctx.bg_image});
            background_grad.bg_image = ctx.bg_image.ptr<float>();
        } else if (ctx.bg_color.is_valid() && ctx.bg_color.numel() >= 3) {
            core::pin_operands({&grad_image, &ctx.bg_color});
            background_grad.bg_color = ctx.bg_color.ptr<float>();
        }
        core::Tensor grad_alpha_extra_2d;
        if (grad_alpha_extra.is_valid() && grad_alpha_extra.numel() > 0) {
            grad_alpha_extra_2d = (grad_alpha_extra.ndim() == 3 && grad_alpha_extra.shape()[0] == 1)
                                      ? grad_alpha_extra.squeeze(0)
                                      : grad_alpha_extra;
            if (!(grad_alpha_extra_2d.ndim() == 2 &&
                  checked_dim_to_int(grad_alpha_extra_2d.shape()[0], "grad_alpha_extra height") == H &&
                  checked_dim_to_int(grad_alpha_extra_2d.shape()[1], "grad_alpha_extra width") == W &&
                  grad_alpha_extra_2d.dtype() == core::DataType::Float32)) {
                throw std::runtime_error("grad_alpha_extra must have shape [H, W] or [1, H, W]");
            }
            grad_alpha_extra_2d = grad_alpha_extra_2d.contiguous();
            background_grad.grad_alpha_extra = grad_alpha_extra_2d.ptr<float>();
        }

        core::Tensor grad_depth_2d;
        const float* grad_depth_ptr = nullptr;
        if (grad_depth.is_valid() && grad_depth.numel() > 0) {
            grad_depth_2d = grad_depth;
            if (grad_depth_2d.ndim() == 3 && grad_depth_2d.shape()[0] == 1) {
                grad_depth_2d = grad_depth_2d.squeeze(0);
            }
            if (!(grad_depth_2d.ndim() == 2 &&
                  checked_dim_to_int(grad_depth_2d.shape()[0], "grad_depth height") == H &&
                  checked_dim_to_int(grad_depth_2d.shape()[1], "grad_depth width") == W)) {
                throw std::runtime_error("grad_depth must have shape [H, W] or [1, H, W]");
            }
            if (grad_depth_2d.device() != core::Device::GPU) {
                grad_depth_2d = grad_depth_2d.gpu();
            }
            if (!grad_depth_2d.is_contiguous()) {
                grad_depth_2d = grad_depth_2d.contiguous();
            }
            grad_depth_ptr = grad_depth_2d.ptr<float>();
        }

        core::Tensor grad_normal_chw;
        const float* grad_normal_ptr = nullptr;
        if (grad_normal.is_valid() && grad_normal.numel() > 0) {
            grad_normal_chw = grad_normal;
            if (!(grad_normal_chw.ndim() == 3 &&
                  grad_normal_chw.shape()[0] == 3 &&
                  checked_dim_to_int(grad_normal_chw.shape()[1], "grad_normal height") == H &&
                  checked_dim_to_int(grad_normal_chw.shape()[2], "grad_normal width") == W)) {
                throw std::runtime_error("grad_normal must have shape [3, H, W]");
            }
            if (grad_normal_chw.device() != core::Device::GPU) {
                grad_normal_chw = grad_normal_chw.gpu();
            }
            if (!grad_normal_chw.is_contiguous()) {
                grad_normal_chw = grad_normal_chw.contiguous();
            }
            grad_normal_ptr = grad_normal_chw.ptr<float>();
        }

        const int n_primitives = checked_dim_to_int(ctx.means.shape()[0], "n_primitives");
        // densification has shape [2, N]
        const bool update_densification = densification.ndim() == 2 &&
                                          densification.shape()[1] >= static_cast<size_t>(n_primitives);
        const bool use_pixel_error_densification = update_densification &&
                                                   error_map.is_valid() &&
                                                   error_map.numel() > 0;

        core::Tensor error_map_2d;
        if (use_pixel_error_densification) {
            error_map_2d = error_map;
            if (error_map_2d.ndim() == 3 && error_map_2d.shape()[0] == 1) {
                error_map_2d = error_map_2d.squeeze(0);
            }
            assert(error_map_2d.ndim() == 2 &&
                   checked_dim_to_int(error_map_2d.shape()[0], "error_map height") == H &&
                   checked_dim_to_int(error_map_2d.shape()[1], "error_map width") == W &&
                   "error_map must have shape [H, W] or [1, H, W]");
            if (error_map_2d.device() != core::Device::GPU) {
                error_map_2d = error_map_2d.gpu();
            }
            if (!error_map_2d.is_contiguous()) {
                error_map_2d = error_map_2d.contiguous();
            }
        }

        // blend_backward_cu does not read the image buffer
        // ((void)image in the kernel) — it reconstructs transmittance from
        // blended image in ctx (one-image VRAM already resident for the
        // loss path); do not allocate a separate pre-blend cache.

        if (!fused_adam.enabled) {
            throw std::runtime_error("FastGS fused Adam state is not available");
        }

        // the backward binds shN-rest exactly like the forward
        // generation-checked resolve plus EXPLICIT representation params
        // (bounds / n_cells / bits). The fused Adam settings' sh_value_* copy is
        // enablement-gated (null through SH warmup) and gates only the update
        // path; the first ACTIVE_SH_BASES>1 backward lands on the degree-up
        // iteration, which at default cadence (interval 1000 == warmup) decoded
        // q16 u16 codes as fp32 float4-swizzle → ~3x region overread → Warp MMU
        // fault on the exportable block's committed-page edge (GUI); silent
        // garbage gradients on plain arenas (headless).
        const float* bwd_shN_ptr = nullptr;
        const float* bwd_shN_bounds_ptr = nullptr;
        unsigned bwd_shN_n_cells = 0u;
        unsigned bwd_shN_bits = 0u;
        const bool shN_q16 = ctx.sh_value_bounds.is_valid() && ctx.sh_value_bounds.numel() > 0;
        const bool shN_f16 = !shN_q16 && ctx.shN.is_valid() &&
                             ctx.shN.dtype() == core::DataType::Float16;
        if (shN_q16) {
            bwd_shN_ptr = static_cast<const float*>(lfs::core::resolve_exportable_device_ptr(ctx.shN));
            bwd_shN_bounds_ptr = static_cast<const float*>(
                lfs::core::resolve_exportable_device_ptr(ctx.sh_value_bounds));
            bwd_shN_n_cells = lfs::core::sh_value_quant::n_value_cells_per_prim(
                lfs::core::sh_rest_coefficients_for_degree(
                    degree_from_layout_bases(static_cast<uint32_t>(ctx.forward_ctx.sh_layout_bases))));
            bwd_shN_bits = 16u;
        } else if (shN_f16) {
            bwd_shN_ptr = static_cast<const float*>(lfs::core::resolve_exportable_device_ptr(ctx.shN));
            bwd_shN_bits = 16u;
        } else if (ctx.shN.is_valid()) {
            bwd_shN_ptr = ctx.shN.ptr<float>();
        }
        if (update_densification)
            densification.set_stream(stream);
        auto backward_result = fast_lfs::rasterization::backward_raw(
            update_densification ? densification.ptr<float>() : nullptr,
            use_pixel_error_densification ? error_map_2d.ptr<float>() : nullptr,
            grad_image.ptr<float>(),
            background_grad,
            grad_depth_ptr,
            grad_normal_ptr,
            ctx.image.ptr<float>(),
            ctx.alpha.ptr<float>(),
            ctx.means.ptr<float>(),
            ctx.raw_scales.ptr<float>(),
            ctx.raw_rotations.ptr<float>(),
            ctx.raw_opacities.ptr<float>(),
            bwd_shN_ptr,
            ctx.w2c_ptr,
            ctx.cam_position_ptr,
            ctx.forward_ctx,
            nullptr,
            n_primitives,
            ctx.forward_ctx.n_visible,
            ctx.active_sh_bases,
            ctx.forward_ctx.sh_layout_bases,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y,
            ctx.mip_filter,
            densification_type,
            &fused_adam,
            bwd_shN_bounds_ptr,
            bwd_shN_n_cells,
            bwd_shN_bits,
            fused_adam.mean_step_far_mask,
            fused_adam.mean_step_far_mask_n,
            edge_weight,
            edge_score);

        ctx.mark_forward_context_released();
        state.frame = {};

        if (!backward_result.success) {
            throw std::runtime_error(std::string("Backward failed: ") + backward_result.error_message);
        }
    }

    fast_lfs::rasterization::FusedAdamParam fast_adam_group(const lfs::gpu_ops::BackwardAdamParam& src) {
        fast_lfs::rasterization::FusedAdamParam dst;
        if (!src.enabled) {
            return dst;
        }
        dst.param = static_cast<float*>(lfs::core::resolve_exportable_device_ptr(src.parameter));
        if (src.value_bits == 16 && src.sh_value_bounds.is_valid() && src.sh_value_bounds.numel() > 0) {
            dst.sh_value_bounds = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(src.sh_value_bounds));
            dst.sh_value_bits = 16;
            dst.sh_value_n_cells = src.value_cells;
        } else if (src.value_bits == 16) {
            dst.sh_value_bits = 16;
        }
        if (src.packed_moments.is_valid() && src.packed_moments.numel() > 0) {
            dst.joint_packed = src.packed_moments.ptr<uint8_t>();
        }
        if (src.joint_bounds.is_valid() && src.joint_bounds.numel() > 0) {
            dst.joint_bounds = src.joint_bounds.ptr<float>();
        }
        dst.joint_bits = src.joint_bits;
        dst.n_primitives = src.primitives;
        if (src.frozen_mask.is_valid() && src.frozen_mask.numel() > 0) {
            dst.frozen_mask = src.frozen_mask.ptr<bool>();
            dst.frozen_mask_size = static_cast<int>(src.frozen_mask.numel());
        }
        dst.frozen_lr_scale = src.frozen_lr_scale;
        if (src.crop_damping_mask.is_valid() && src.crop_damping_mask.numel() > 0) {
            dst.crop_damping_mask = src.crop_damping_mask.ptr<bool>();
            dst.crop_damping_mask_size = static_cast<int>(src.crop_damping_mask.numel());
        }
        dst.cropbox_lr_scale = src.cropbox_lr_scale;
        dst.n_elements = src.elements;
        dst.n_attributes = src.attributes;
        dst.step_size = src.step_size;
        dst.bias_correction2_sqrt_rcp = src.bc2_sqrt_rcp;
        dst.enabled = true;
        if (src.screen_share.is_valid() && src.screen_share.numel() > 0 &&
            src.screen_share_limit > 0.f && src.screen_share_limit < 1.f) {
            dst.screen_share_max = src.screen_share.ptr<float>();
            dst.screen_share_n = static_cast<int>(src.screen_share.numel());
            dst.screen_share_limit = src.screen_share_limit;
            dst.screen_share_penalty = src.screen_share_penalty;
        }
        return dst;
    }

    fast_lfs::rasterization::FusedAdamSettings fast_adam_settings(const lfs::gpu_ops::BackwardAdam& adam) {
        fast_lfs::rasterization::FusedAdamSettings fused;
        fused.beta1 = adam.beta1;
        fused.beta2 = adam.beta2;
        fused.eps = adam.eps;
        fused.scale_reg_weight = adam.scale_reg_weight;
        fused.flatten_reg_weight = adam.flatten_reg_weight;
        fused.opacity_reg_weight = adam.opacity_reg_weight;
        if (adam.scale_reg_weight > 0.f && adam.scale_reg_loss.is_valid() && adam.scale_reg_loss.numel() > 0) {
            fused.scale_reg_loss_out = adam.scale_reg_loss.ptr<float>();
        }
        if (adam.opacity_reg_weight > 0.f && adam.opacity_reg_loss.is_valid() &&
            adam.opacity_reg_loss.numel() > 0) {
            fused.opacity_reg_loss_out = adam.opacity_reg_loss.ptr<float>();
        }
        if (adam.sparsity_sigmoid.is_valid() && adam.sparsity_sigmoid.numel() > 0) {
            fused.sparsity_opa_sigmoid = adam.sparsity_sigmoid.ptr<float>();
            fused.sparsity_z = adam.sparsity_z.is_valid() ? adam.sparsity_z.ptr<float>() : nullptr;
            fused.sparsity_u = adam.sparsity_u.is_valid() ? adam.sparsity_u.ptr<float>() : nullptr;
            fused.sparsity_n = static_cast<int>(adam.sparsity_sigmoid.numel());
            fused.sparsity_rho = adam.sparsity_rho;
            fused.sparsity_grad_loss = adam.sparsity_grad_loss;
        }
        using lfs::gpu_ops::AdamSlot;
        fused.means = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::Means)]);
        fused.scaling = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::Scaling)]);
        fused.rotation = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::Rotation)]);
        fused.opacity = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::Opacity)]);
        fused.sh0 = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::Sh0)]);
        fused.shN = fast_adam_group(adam.groups[static_cast<std::size_t>(AdamSlot::ShN)]);
        fused.per_splat_mean_step = adam.per_splat_mean_step;
        fused.mean_step_median_extent = adam.median_extent;
        fused.mean_step_r_min = adam.r_min;
        fused.mean_step_r_max = adam.r_max;
        if (adam.far_mask.is_valid() && adam.far_mask.numel() > 0 && fused.means.n_primitives > 0) {
            fused.mean_step_far_mask = adam.far_mask.ptr<bool>();
            fused.mean_step_far_mask_n = std::min(
                static_cast<int>(adam.far_mask.numel()), fused.means.n_primitives);
        }
        fused.enabled = fused.means.enabled || fused.scaling.enabled || fused.rotation.enabled ||
                        fused.opacity.enabled || fused.sh0.enabled || fused.shN.enabled;
        return fused;
    }

    void fast_ops_release(lfs::gpu_ops::FastSaved& saved) noexcept {
        if (!saved.backend) {
            return;
        }
        state_of(saved).frame = {};
    }

    lfs::gpu_ops::State fast_ops_create() {
        return std::make_unique<CudaFastState>();
    }

    [[nodiscard]] lfs::Error fast_raster_error(const lfs::gpu_ops::RasterResult& result) {
        using Code = lfs::gpu_ops::RasterResult::Code;
        const std::string detail{result.message};
        if (detail.find("n_primitives is 0") != std::string::npos) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "FastGS cannot render an empty Gaussian model.",
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        if (result.code == Code::InstanceOverflow) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::FailedPrecondition,
                .domain = lfs::ErrorDomain::CUDA,
                .user_message =
                    "Pathological splat extents produced more tile instances "
                    "than the 32-bit FastGS path can represent; skipping step.",
                .detail = detail,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const bool exhausted = result.code == Code::ResourceExhausted;
        return lfs::make_error(lfs::ErrorInit{
            .code = exhausted ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::CUDA,
            .user_message = exhausted ? "Ran out of GPU memory while rendering during training."
                                      : "FastGS forward rasterization failed.",
            .detail = detail,
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    lfs::gpu_ops::ShStorage fast_sh_storage(const core::SplatData& model) {
        if (model.shN_value_quantized()) {
            return lfs::gpu_ops::ShStorage::Q16;
        }
        if (model.shN_ieee_f16()) {
            return lfs::gpu_ops::ShStorage::IeeeFloat16;
        }
        return lfs::gpu_ops::ShStorage::Float32;
    }

    lfs::gpu_ops::RasterResult fast_render(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        core::Camera& camera,
        core::SplatData& model,
        core::Tensor& bg_color,
        const int tile_x,
        const int tile_y,
        const int tile_width,
        const int tile_height,
        const bool mip_filter,
        const core::Tensor& bg_image,
        const bool render_normal,
        const bool render_depth,
        RenderOutput& output) {
        const int sh_degree = model.get_active_sh_degree();
        const int max_sh_degree = model.get_max_sh_degree();
        auto [fx, fy, cx, cy] = camera.get_intrinsics();
        core::Tensor no_bounds;
        const bool q16 = model.shN_value_quantized();
        const lfs::gpu_ops::FastParams params{
            .full_image = {camera.image_height(), camera.image_width()},
            .intrinsics = {fx, fy, cx, cy},
            .tile_x = tile_x,
            .tile_y = tile_y,
            .tile_w = tile_width,
            .tile_h = tile_height,
            .sh = {
                .storage = fast_sh_storage(model),
                .active_bases = static_cast<uint32_t>((sh_degree + 1) * (sh_degree + 1)),
                .layout_bases = static_cast<uint32_t>((max_sh_degree + 1) * (max_sh_degree + 1)),
            },
            .mip_filter = mip_filter,
            .render_normal = render_normal,
            .render_depth = render_depth,
        };
        const lfs::gpu_ops::SplatInputs splats{
            .means = model.means(),
            .raw_scales = model.scaling_raw(),
            .raw_rotations = model.rotation_raw(),
            .raw_opacities = model.opacity_raw(),
            .sh0 = model.sh0(),
            .shN = model.shN(),
            .sh_value_bounds = q16 ? model.shN_value_bounds() : no_bounds,
        };
        const lfs::gpu_ops::RenderOutputs rendered{
            .image = output.image,
            .alpha = output.alpha,
            .depth = output.depth,
            .normal = output.normal,
        };
        const auto result = ops.forward(
            saved, splats, camera.world_view_transform(), camera.cam_position(),
            bg_color, bg_image, params, rendered, model._max_screen_share);
        if (result.code == lfs::gpu_ops::RasterResult::Code::Success) {
            output.width = tile_width > 0 ? tile_width : camera.image_width();
            output.height = tile_height > 0 ? tile_height : camera.image_height();
        }
        return result;
    }

    RenderOutput fast_infer(
        const lfs::gpu_ops::FastRasterOps& ops,
        lfs::gpu_ops::FastSaved& saved,
        core::Camera& camera,
        core::SplatData& model,
        core::Tensor& bg_color,
        const bool mip_filter,
        const core::Tensor& bg_image,
        const bool render_normal) {
        RenderOutput output;
        const auto result = fast_render(
            ops, saved, camera, model, bg_color, 0, 0, 0, 0, mip_filter, bg_image, render_normal, true, output);
        if (result.code != lfs::gpu_ops::RasterResult::Code::Success) {
            throw lfs::Exception(fast_raster_error(result));
        }
        ops.release(saved);
        return output;
    }

    void fast_record_vram(
        const lfs::gpu_ops::FastSaved& saved,
        const core::Tensor& image,
        const core::Tensor& alpha,
        const bool run_gaussian_backward,
        const size_t num_primitives) {
        const auto* state = state_of(saved);
        if (state == nullptr) {
            return;
        }
        const auto& ctx = state->frame.forward_ctx;
        constexpr std::string_view scope = "rasterizer.fastgs";
        record_vram_current(scope, "forward.per_primitive_buffers", ctx.per_primitive_buffers_size);
        record_vram_current(scope, "forward.per_tile_buffers", ctx.per_tile_buffers_size);
        record_vram_current(scope, "forward.sorted_indices_live", ctx.sorted_primitive_indices_size);
        record_vram_current(scope, "forward.sort_workspace_arena", ctx.per_instance_sort_total_size, false,
                            lfs::diagnostics::VramAllocationMethod::Arena);
        record_rasterizer_arena_disclosure(scope);
        const size_t raster_arena_live = ctx.per_primitive_buffers_size + ctx.per_tile_buffers_size +
                                         ctx.per_instance_sort_total_size;
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        profiler.setGauge("vram.audit.fastgs_raster_live.required_bytes", static_cast<double>(raster_arena_live));
        profiler.setGauge("vram.audit.fastgs_raster_live.allocated_bytes", static_cast<double>(raster_arena_live));
        if (PerfBenchCollector::enabled()) {
            PerfBenchCollector::instance().set_fastgs_raster_live_bytes(raster_arena_live, 0);
        }
        record_vram_current(scope, "forward.sort_scratch_transient", 0, true);
        record_vram_current(scope, "forward.sort_total_transient", 0, true);
        record_vram_current(scope, "backward.grad_mean2d_helper", num_primitives * 2 * sizeof(float));
        record_vram_current(scope, "backward.grad_conic_helper", num_primitives * 3 * sizeof(float));
        record_vram_current(scope, "backward.fused_grad_opacity_helper",
                            run_gaussian_backward && ctx.grad_opacity_helper ? num_primitives * sizeof(float) : 0,
                            true);
        record_vram_current(scope, "backward.fused_grad_color_helper",
                            run_gaussian_backward && ctx.grad_color_helper ? num_primitives * 3 * sizeof(float) : 0,
                            true);
        record_vram_tensor(scope, "output.image", image);
        record_vram_tensor(scope, "output.alpha", alpha);
        record_vram_tensor(scope, "saved.bg_color", state->frame.bg_color);
    }

    void fast_release_caches(lfs::gpu_ops::FastSaved& saved) noexcept {
        if (!saved.backend) {
            return;
        }
        state_of(saved).caches = {};
    }

    const lfs::gpu_ops::FastRasterOps& cuda_fast_ops() {
        static const lfs::gpu_ops::FastRasterOps ops{
            .create = fast_ops_create,
            .forward = fast_ops_forward,
            .backward = fast_ops_backward,
            .release = fast_ops_release,
            .warmup = fast_lfs::rasterization::warmup_kernels,
        };
        return ops;
    }

    CudaFastFrameView cuda_fast_frame_view(const lfs::gpu_ops::FastSaved& saved) noexcept {
        CudaFastFrameView view;
        const auto* state = state_of(saved);
        if (state == nullptr) {
            return view;
        }
        const auto& frame = state->frame;
        view.n_instances = frame.forward_ctx.n_instances;
        view.n_visible = frame.forward_ctx.n_visible;
        view.per_tile_buffers_size = frame.forward_ctx.per_tile_buffers_size;
        view.per_instance_sort_total_size = frame.forward_ctx.per_instance_sort_total_size;
        view.frame_id = frame.forward_ctx.frame_id;
        view.completion_stream = frame.completion_stream;
        return view;
    }

} // namespace lfs::training
