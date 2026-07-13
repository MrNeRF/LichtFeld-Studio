/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/mesh2splat.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace lfs::core {
    namespace param {
        // Mask mode for attention mask behavior during training
        enum class MaskMode {
            None,             // No masking applied
            Segment,          // Soft penalty to enforce alpha→0 in masked areas
            Ignore,           // Completely ignore masked regions in loss
            SegmentAndIgnore, // 3-band mask (0-255): value<128 ignore, 128<=value<=250 segment, value>250 keep
            AlphaConsistent   // Enforce exact alpha values from mask
        };

        // Background mode for training - only one can be active at a time
        enum class BackgroundMode {
            SolidColor, // Use bg_color RGB values
            Modulation, // Sinusoidal background modulation
            Image,      // Use custom background image
            Random      // Random per-pixel colors each iteration
        };

        inline constexpr std::string_view kStrategyMCMC = "mcmc";
        inline constexpr std::string_view kStrategyMRNF = "mrnf";
        inline constexpr std::string_view kStrategyMNRFLegacy = "mnrf";
        inline constexpr std::string_view kStrategyLFSLegacy = "lfs";
        inline constexpr std::string_view kStrategyIGSPlus = "igs+";

        [[nodiscard]] inline std::filesystem::path default_dataset_output_path(
            const std::filesystem::path& dataset_path) {
            auto base_path = dataset_path;
            auto ext = dataset_path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (ext == ".json") {
                base_path = dataset_path.parent_path();
            }
            return base_path / "output";
        }

        [[nodiscard]] inline constexpr std::string_view canonical_strategy_name(const std::string_view strategy) noexcept {
            if (strategy == kStrategyMCMC)
                return kStrategyMCMC;
            if (strategy == kStrategyMRNF || strategy == kStrategyMNRFLegacy || strategy == kStrategyLFSLegacy)
                return kStrategyMRNF;
            if (strategy == kStrategyIGSPlus)
                return kStrategyIGSPlus;
            return {};
        }

        [[nodiscard]] inline constexpr bool is_valid_strategy_name(const std::string_view strategy) noexcept {
            return !canonical_strategy_name(strategy).empty();
        }

        [[nodiscard]] inline constexpr bool is_mrnf_strategy(const std::string_view strategy) noexcept {
            return canonical_strategy_name(strategy) == kStrategyMRNF;
        }

        [[nodiscard]] inline constexpr bool strategy_names_match(
            const std::string_view lhs,
            const std::string_view rhs) noexcept {
            const auto lhs_canonical = canonical_strategy_name(lhs);
            const auto rhs_canonical = canonical_strategy_name(rhs);
            if (!lhs_canonical.empty() && !rhs_canonical.empty())
                return lhs_canonical == rhs_canonical;
            return lhs == rhs;
        }

        struct LFS_CORE_API OptimizationParameters {
            size_t iterations = 30'000;
            size_t sh_degree_interval = 1'000;
            float means_lr = 0.000016f;
            float means_lr_end = 0.00000016f;
            float shs_lr = 0.0025f;
            float opacity_lr = 0.025f;
            float scaling_lr = 0.005f;
            float scaling_lr_end = 0.005f;
            float rotation_lr = 0.001f;
            float lambda_dssim = 0.2f;
            float min_opacity = 0.005f;
            size_t refine_every = 100;
            size_t start_refine = 500;
            size_t stop_refine = 25'000;
            float grad_threshold = 0.0002f;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
            float init_opacity = 0.5f;
            float init_scaling = 0.1f;
            int max_cap = 1000000;

            std::vector<size_t> eval_steps = {7'000, 30'000};  // Steps to evaluate the model
            std::vector<size_t> save_steps = {7'000, 30'000};  // Steps to save the model
            bool bg_modulation = false;                        // Enable sinusoidal background modulation
            bool enable_eval = false;                          // Only evaluate when explicitly enabled
            bool enable_save_eval_images = true;               // Save during evaluation images
            bool headless = false;                             // Disable visualization during training
            bool auto_train = false;                           // Start training immediately on startup
            bool no_splash = false;                            // Skip splash screen on startup
            bool debug_python = false;                         // Start debugpy listener for plugin debugging
            int debug_python_port = 5678;                      // Port for debugpy listener
            std::string strategy = std::string(kStrategyMRNF); // Optimization strategy: mcmc, mrnf, igs+.

            // Mask parameters
            MaskMode mask_mode = MaskMode::None;      // Attention mask mode
            bool invert_masks = false;                // Invert mask values (swap object/background)
            float mask_threshold = 0.5f;              // Threshold: >= threshold → 1.0, < threshold → keep original
            float mask_opacity_penalty_weight = 1.0f; // Opacity penalty weight for segment mode
            float mask_opacity_penalty_power = 2.0f;  // Penalty falloff (1=linear, 2=quadratic)
            bool use_alpha_as_mask = true;            // Auto-use alpha channel from RGBA images as mask

            // Depth supervision
            bool use_depth_loss = false;         // Use dataset depth maps when available
            float depth_loss_weight = 2.0f;      // Depth supervision weight (decays over training)
            std::string depth_loss_mode = "ssi"; // ssi (auto prior), ssi-disparity, or ssi-depth

            // Normal supervision
            bool use_normal_loss = false;            // Use dataset normal maps when available
            float normal_loss_weight = 0.05f;        // Prior normal supervision weight
            float normal_consistency_weight = 0.05f; // Depth-normal consistency weight
            float normal_flatten_weight = 1.0f;      // L1 on the smallest scale axis while normal supervision is active
            std::string normal_loss_space = "auto";  // auto, camera-opencv, camera-opengl, or world

            // Mip filter (anti-aliasing)
            bool mip_filter = false;

            // Background settings for training
            BackgroundMode bg_mode = BackgroundMode::SolidColor; // Which background mode to use
            std::array<float, 3> bg_color = {0.0f, 0.0f, 0.0f};  // RGB background color [0-1]
            std::filesystem::path bg_image_path = {};            // Custom background image path

            // Bilateral grid parameters
            bool use_bilateral_grid = false;
            int bilateral_grid_X = 16;
            int bilateral_grid_Y = 16;
            int bilateral_grid_W = 8;
            float bilateral_grid_lr = 2e-3f;
            float tv_loss_weight = 10.f;

            // PPISP (Physically-Plausible ISP) parameters
            bool use_ppisp = false;
            float ppisp_lr = 2e-3f;
            float ppisp_reg_weight = 0.001f;
            int ppisp_warmup_steps = 500;
            bool ppisp_freeze_from_sidecar = false;
            std::filesystem::path ppisp_sidecar_path = {};
            bool ppisp_use_controller = false;
            bool ppisp_freeze_gaussians_on_distill = true;
            int ppisp_controller_activation_step = -1; // Negative values use the last-5000-steps default schedule
            float ppisp_controller_lr = 2e-3f;

            // Shared densification thresholds and reset controls
            float prune_opacity = 0.005f;
            float grow_scale3d = 0.01f;
            float grow_scale2d = 0.05f;
            float prune_scale3d = 0.1f;
            float prune_scale2d = 0.15f;
            size_t reset_every = 3'000;
            size_t pause_refine_after_reset = 0;
            bool revised_opacity = false;
            bool gut = false;
            bool undistort = false;
            float steps_scaler = 1.f; // Scales training step counts; values <= 0 disable scaling

            // MRNF strategy specific parameters
            float growth_grad_threshold = 0.003f;
            float grow_fraction = 0.07f;
            size_t grow_until_iter = 15000;
            float opacity_decay = 0.004f;
            float scale_decay = 0.002f;
            float means_noise_weight = 50.0f;
            float bounds_percentile = 0.8f;
            bool use_error_map = true;
            bool use_edge_map = true;

            // Random initialization parameters
            bool random = false;        // Use random initialization instead of SfM
            int init_num_pts = 100'000; // Number of random points to initialize
            float init_extent = 3.0f;   // Extent of random point cloud

            // Sparsity optimization parameters
            bool enable_sparsity = false;
            int sparsify_steps = 15000;
            float init_rho = 0.0005f;
            float prune_ratio = 0.6f;

            std::string config_file = "";

            void scale_steps(float ratio);
            void apply_step_scaling();
            void remove_step_scaling();
            [[nodiscard]] int resolved_total_iterations() const;
            [[nodiscard]] int resolved_ppisp_controller_activation_step(int total_iterations) const;

            nlohmann::json to_json() const;
            static OptimizationParameters from_json(const nlohmann::json& j);

            [[nodiscard]] std::string validate() const;

            // Factory methods for strategy presets
            static OptimizationParameters mcmc_defaults();
            static OptimizationParameters mrnf_defaults();
            static OptimizationParameters igs_plus_defaults();
        };

        struct LFS_CORE_API LoadingParams {
            bool use_cpu_memory = true;
            float min_cpu_free_memory_ratio = 0.1f; // make sure at least 10% RAM is free
            float min_cpu_free_GB = 1.0f;           // min GB we want to be free
            bool use_fs_cache = true;
            bool print_cache_status = true;
            int print_status_freq_num = 500; // every print_status_freq_num calls for load print cache status
            bool use_16bit_color = false;
            nlohmann::json to_json() const;
            static LoadingParams from_json(const nlohmann::json& j);
        };

        struct LFS_CORE_API DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "";
            std::string output_name = "";
            std::string images = "images";
            int resize_factor = -1;
            int test_every = 8;
            std::vector<std::string> timelapse_images = {};
            int timelapse_every = 50;
            int max_width = 3840;
            int min_track_length = 0;
            LoadingParams loading_params;

            // Mask loading parameters (copied from optimization params)
            bool invert_masks = false;
            float mask_threshold = 0.5f;

            // Not serialized — UI-controlled per import.
            std::string centralize_dataset = "off";

            nlohmann::json to_json() const;
            static DatasetConfig from_json(const nlohmann::json& j);
        };

        struct LFS_CORE_API ServerConfig {
            int tcp_server_connection_port = -1;    // Set the TCP connection port when tcp connection is in use for server requests, -1 for auto
            int tcp_broadcast_connection_port = -1; // Set the TCP connection port when tcp connection is in use for broadcasting, -1 for auto
            bool tcp_connection = false;            // Use TCP connection for signals and events

            nlohmann::json to_json() const;
            static ServerConfig from_json(const nlohmann::json& j);
        };

        // Headless camera-path -> video render mode (see --render-camera-path)
        struct LFS_CORE_API RenderPathConfig {
            std::filesystem::path camera_path; // sequencer::Timeline JSON keyframe path
            std::filesystem::path load_path;   // trained scene: .ply/.sog/.spz or .resume checkpoint
            std::filesystem::path output_path; // destination .mp4
            int width = 1920;
            int height = 1080;
            int fps = 30;
            int crf = 18;
        };

        struct LFS_CORE_API TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;
            ServerConfig server;

            // Viewer mode: splat files to load (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)
            std::vector<std::filesystem::path> view_paths;

            // COLMAP sparse folder for camera-only import (no images required)
            std::optional<std::filesystem::path> import_cameras_path = std::nullopt;

            // Optional splat file for initialization (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)
            std::optional<std::string> init_path = std::nullopt;

            // Optional trained splats to append to the training model before optimizer initialization
            std::vector<std::filesystem::path> add_splat_paths;
            std::vector<bool> add_splat_freeze;
            bool exclude_frozen_add_splats_from_export = false;

            // Checkpoint to resume training from
            std::optional<std::filesystem::path> resume_checkpoint = std::nullopt;

            // Headless camera-path -> video render mode (see --render-camera-path)
            std::optional<RenderPathConfig> render_path = std::nullopt;

            // Python scripts to execute for custom training callbacks
            std::vector<std::filesystem::path> python_scripts;

            // True when --bg-color was provided on the command line.
            bool cli_bg_color_set = false;

            std::vector<int> disabled_camera_uids;

            [[nodiscard]] std::string validate() const;
        };

        // Output format for conversion tool
        enum class OutputFormat { PLY,
                                  SOG,
                                  SPZ,
                                  HTML,
                                  USD,
                                  USDA,
                                  USDC,
                                  RAD };

        // PLY -> RAD only: per-bucket LOD tree builder for the out-of-core
        // converter. BHATT is the quality-validated default; OCTREE trades
        // unvalidated quality for a much faster parallel build.
        enum class LodBuilder { BHATT,
                                OCTREE };

        enum class RadExportMode { Stream,
                                   NonStream };

        // Parameters for the convert command
        struct LFS_CORE_API ConvertParameters {
            std::filesystem::path input_path;
            std::filesystem::path output_path; // Empty = derive from input
            OutputFormat format = OutputFormat::PLY;
            int sh_degree = 3; // 0-3, -1 = keep original
            int sog_iterations = 10;
            // PLY -> RAD only: replicate the source across an AxB ground-plane
            // grid instead of pre-tiling the input file.
            std::uint32_t tiles_x = 1;
            std::uint32_t tiles_y = 1;
            LodBuilder lod_builder = LodBuilder::BHATT;
            RadExportMode rad_export_mode = RadExportMode::Stream;
            bool overwrite = false; // Skip overwrite prompts
        };

        // Parameters for the mesh2splat command
        struct LFS_CORE_API Mesh2SplatParameters {
            std::filesystem::path input_path;
            std::filesystem::path output_path; // Empty = derive from input
            OutputFormat format = OutputFormat::PLY;
            std::vector<OutputFormat> formats{OutputFormat::PLY};
            Mesh2SplatOptions options;
            int sog_iterations = 10;
            bool overwrite = false;
        };

        enum class PreprocessOutputMode { Depth,
                                          Normals,
                                          Both };

        struct LFS_CORE_API PreprocessParameters {
            std::filesystem::path dataset_path;
            std::string images_folder = "images";
            std::filesystem::path model_path;
            PreprocessOutputMode mode = PreprocessOutputMode::Both;
            int max_side = 518;
            std::int64_t num_tokens = 1800;
            int threads = 0;
            int png_compression = 1;
            int bit_depth = 16;
            bool force_cpu = false;
            bool overwrite = false;
            bool no_download = false;
            bool download_only = false;
        };

        // Modern C++23 functions returning expected values
        LFS_CORE_API std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path);

        // Save training parameters to JSON
        LFS_CORE_API std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);

    } // namespace param
} // namespace lfs::core
