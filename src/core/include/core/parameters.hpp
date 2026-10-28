/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/image_sharpen.hpp"
#include "core/mesh2splat.hpp"
#include "core/path_utils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
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

        enum class PPISPHoldoutAppearance { Mean,
                                            Nearest };

        [[nodiscard]] inline constexpr std::optional<PPISPHoldoutAppearance> ppisp_holdout_appearance_from_string(std::string_view value) noexcept {
            if (value == "mean")
                return PPISPHoldoutAppearance::Mean;
            if (value == "nearest")
                return PPISPHoldoutAppearance::Nearest;
            return std::nullopt;
        }

        enum class EvalSpace {
            Distorted,
            Undistorted,
        };

        [[nodiscard]] inline constexpr std::string_view eval_space_name(
            const EvalSpace space) noexcept {
            switch (space) {
            case EvalSpace::Distorted:
                return "distorted";
            case EvalSpace::Undistorted:
                return "undistorted";
            }
            return "distorted";
        }

        [[nodiscard]] inline constexpr std::optional<EvalSpace> eval_space_from_string(
            const std::string_view value) noexcept {
            if (value == "distorted")
                return EvalSpace::Distorted;
            if (value == "undistorted")
                return EvalSpace::Undistorted;
            return std::nullopt;
        }

        // Grid the render is quantized to before metrics; Auto follows each reference image's encoding.
        enum class EvalBitDepth {
            Auto,
            Eight,
            Sixteen,
            Float,
        };

        // Pixel sets scored with an evaluation mask: the mask, every valid pixel, the valid pixels outside the mask.
        enum class EvalPixelSet : uint8_t {
            Mask,
            Unmasked,
            Inverted,
        };
        inline constexpr std::array kEvalPixelSets{EvalPixelSet::Mask, EvalPixelSet::Unmasked, EvalPixelSet::Inverted};

        // Mesh2Splat allocates resolution^2 * 6 output splats up front: 2048 is already a 2.4 GB buffer.
        inline constexpr int kMaxInitMeshResolution = 2048;

        // The set's name in eval_mask_report.
        [[nodiscard]] inline constexpr std::string_view eval_pixel_set_name(const EvalPixelSet set) noexcept {
            switch (set) {
            case EvalPixelSet::Mask:
                return "mask";
            case EvalPixelSet::Unmasked:
                return "unmasked";
            case EvalPixelSet::Inverted:
                return "inverted";
            }
            return "mask";
        }

        [[nodiscard]] inline constexpr std::string_view eval_bit_depth_name(const EvalBitDepth depth) noexcept {
            switch (depth) {
            case EvalBitDepth::Auto:
                return "auto";
            case EvalBitDepth::Eight:
                return "8";
            case EvalBitDepth::Sixteen:
                return "16";
            case EvalBitDepth::Float:
                return "float";
            }
            return "auto";
        }

        [[nodiscard]] inline constexpr std::optional<EvalBitDepth> eval_bit_depth_from_string(
            const std::string_view value) noexcept {
            if (value == "auto")
                return EvalBitDepth::Auto;
            if (value == "8")
                return EvalBitDepth::Eight;
            if (value == "16")
                return EvalBitDepth::Sixteen;
            if (value == "float")
                return EvalBitDepth::Float;
            return std::nullopt;
        }

        enum class DensifyErrorMap {
            Ssim,   // full SSIM (luminance × contrast × structure)
            SsimCs, // contrast × structure only (luminance excluded)
        };

        [[nodiscard]] inline constexpr std::string_view densify_error_map_name(
            const DensifyErrorMap mode) noexcept {
            switch (mode) {
            case DensifyErrorMap::Ssim:
                return "ssim";
            case DensifyErrorMap::SsimCs:
                return "ssim_cs";
            }
            return "ssim_cs";
        }

        [[nodiscard]] inline constexpr std::optional<DensifyErrorMap> densify_error_map_from_string(
            const std::string_view value) noexcept {
            if (value == "ssim")
                return DensifyErrorMap::Ssim;
            if (value == "ssim_cs")
                return DensifyErrorMap::SsimCs;
            return std::nullopt;
        }

        enum class NormalLossSpace {
            Auto,
            CameraOpenCV,
            CameraOpenGL,
            World,
        };

        [[nodiscard]] inline constexpr std::string_view normal_loss_space_name(
            const NormalLossSpace space) noexcept {
            switch (space) {
            case NormalLossSpace::Auto:
                return "auto";
            case NormalLossSpace::CameraOpenCV:
                return "camera-opencv";
            case NormalLossSpace::CameraOpenGL:
                return "camera-opengl";
            case NormalLossSpace::World:
                return "world";
            }
            return "auto";
        }

        [[nodiscard]] inline constexpr std::optional<NormalLossSpace> normal_loss_space_from_string(
            const std::string_view value) noexcept {
            if (value == "auto")
                return NormalLossSpace::Auto;
            if (value == "camera-opencv")
                return NormalLossSpace::CameraOpenCV;
            if (value == "camera-opengl")
                return NormalLossSpace::CameraOpenGL;
            if (value == "world")
                return NormalLossSpace::World;
            return std::nullopt;
        }

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
            auto ext = lfs::core::path_to_utf8(dataset_path.extension());
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
            float late_lr_anneal = 1.0f;
            float scaling_lr_end = 0.005f;
            float rotation_lr = 0.001f;
            // With an active crop box: Adam steps and refine scores of splats that no camera sees
            // inside, in front of or behind the box. Strategy noise, decay and resets are unscaled.
            float cropbox_lr_scale = 0.1f;
            float cropbox_loss_weight = 0.1f;
            // With an active crop box: share of new splats placed inside it (0..1). 0 = auto: growth
            // follows the ROI-weighted error with no fixed split.
            float cropbox_growth_inside = 0.0f;
            // With an active crop box: the splats outside it are removed when training completes and left out
            // of every export. A stopped run keeps them so it can resume.
            bool crop_to_cropbox = false;
            float lambda_dssim = 0.2f;
            // SteadySplats Eq. 7: weighted RGB squared distance to each pixel's expected render color.
            float color_variance_weight = 0.0f;
            size_t color_variance_warmup_iters = 0;
            float min_opacity = 0.005f;
            size_t refine_every = 100;
            size_t start_refine = 500;
            size_t stop_refine = 25'000;
            // 0 disables. Applied until stop_refine, at the densify-safe mutation point.
            size_t morton_reorder_interval = 5'000;
            int sh_degree = 3;
            float opacity_reg = 0.01f;
            float scale_reg = 0.01f;
            float scale_reg_decay_power = -1.0f;
            float erank_reg = 0.0f;
            float dc_reg = 0.0f;
            float sh_rest_reg = 0.0f;
            float thin_structure_weight = 0.0f;
            float gradient_loss_weight = 0.0f;
            bool opacity_decay_rendered_only = false;
            float densify_structure_weight = 0.0f;
            float init_opacity = 0.5f;
            float init_scaling = 0.1f;
            int max_cap = 1000000;

            std::vector<size_t> eval_steps = {7'000, 30'000};  // Steps to evaluate the model
            std::vector<size_t> save_steps = {7'000, 30'000};  // Steps at which to save the project (project.licht)
            bool bg_modulation = false;                        // Enable sinusoidal background modulation
            bool enable_eval = false;                          // Only evaluate when explicitly enabled
            bool eval_all = false;                             // Train on every image and evaluate all of them
            bool eval_flip = false;                            // Also compute FLIP and save its error maps
            EvalSpace eval_space = EvalSpace::Distorted;       // Reference image space used for evaluation
            EvalBitDepth eval_bit_depth = EvalBitDepth::Auto;  // Grid the render is quantized to for evaluation
            std::string eval_mask = "";                        // Mesh path, bbox:..., cropbox, masks:<folder>, depth:near,far, points:radius,close or points:<file>; empty disables
            std::string eval_mask_report = "";                 // Pixel sets scored with the mask, comma list of mask, unmasked and inverted; the first is the headline. Empty: the mask
            float eval_mask_opacity = 0.85f;                   // Rendered opacity a pixel needs to count as covered by a splat mask
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
            bool depth_auto_generate = true;     // Generate missing/mismatched maps from images/ with MoGe-2

            // Normal supervision
            bool use_normal_loss = false;             // Use dataset normal maps when available
            bool normal_auto_generate = true;         // Generate missing/mismatched maps from images/ with MoGe-2
            float normal_loss_weight = 0.005f;        // Prior normal supervision weight (sweep: free geometry; raise to ~0.1 when geometry is the product)
            float normal_consistency_weight = 0.001f; // Depth-normal consistency weight
            float normal_flatten_weight = 0.0f;       // L1 on the smallest scale axis; off by default - it suppresses densification under the benchmark refine regime
            float normal_start_fraction = 0.08f;      // Start normal supervision at floor(fraction * total iterations)
            float normal_end_fraction = 1.0f;         // Active while iter < end_fraction * total; 1.0 means until the end
            NormalLossSpace normal_loss_space = NormalLossSpace::Auto;

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

            // Combined per-photo exposure + residual grid (replaces standalone grid/PPISP)
            bool use_exposure_correction = false;
            int exposure_correction_grid_start_iter = 1000;

            [[nodiscard]] bool bilateral_grid_active() const {
                return use_bilateral_grid || use_exposure_correction;
            }
            [[nodiscard]] bool ppisp_active() const {
                return use_ppisp || use_exposure_correction;
            }

            // PPISP (Physically-Plausible ISP) parameters
            bool use_ppisp = false;
            bool ppisp_exposure_from_exif = true;
            float ppisp_lr = 2e-3f;
            float ppisp_reg_weight = 0.001f;
            int ppisp_warmup_steps = 500;
            bool ppisp_freeze_from_sidecar = false;
            std::filesystem::path ppisp_sidecar_path = {};
            PPISPHoldoutAppearance ppisp_holdout_appearance = PPISPHoldoutAppearance::Nearest;
            bool ppisp_use_controller = false;
            bool ppisp_freeze_gaussians_on_distill = true;
            int ppisp_controller_activation_step = -1; // Negative values use the last-5000-steps default schedule
            float ppisp_controller_lr = 2e-3f;

            // Shared densification thresholds and reset controls
            float prune_opacity = 0.005f;
            size_t reset_every = 3'000;
            bool gut = false;
            bool undistort = false;
            float steps_scaler = 1.f;       // Scales training step counts; values <= 0 disable scaling
            float image_count_scaler = 1.f; // Share of steps_scaler applied by the GUI image-count auto-scale

            // MRNF strategy specific parameters
            float growth_grad_threshold = 0.003f;
            float grow_fraction = 0.07f;
            size_t grow_until_iter = 15000;
            // At the cap, frees the splats whose removal changes the image least so error-guided
            // growth keeps working until grow_until_iter.
            bool growth_exchange = false;
            float opacity_decay = 0.004f;
            float scale_decay = 0.002f;
            float means_noise_weight = 50.0f;
            float bounds_percentile = 0.8f;
            bool use_error_map = true;
            DensifyErrorMap densify_error_map = DensifyErrorMap::SsimCs;
            // Dimensionless on-screen share cap. <=0 or >=1 disables the cap.
            float max_screen_share = 0.3f;
            float screen_share_penalty = 1.0f;
            bool use_edge_map = true;

            // Random initialization parameters
            bool random = false;        // Use random initialization instead of SfM
            int init_num_pts = 100'000; // Number of random points to initialize
            float init_extent = 3.0f;   // Extent of random point cloud

            // Mesh2Splat raster resolution when the init file is a mesh
            int init_mesh_resolution = Mesh2SplatOptions{}.resolution_target;

            // POPSpa post-training sparsification
            bool enable_sparsity = false;
            int popspa_target_count = 256000;
            int popspa_first_prune_count = 0;
            int popspa_sparsify_steps = 5000;
            int popspa_refine_steps = 5000;
            int popspa_projection_interval = 50;
            int popspa_seed = 42;
            float popspa_rho = 0.0005f;
            float popspa_erank_weight = 0.01f;
            float popspa_thin_scale_weight = 1.0f;
            float popspa_erank_epsilon = 1e-7f;
            float popspa_means_lr = 1.6e-6f;
            float popspa_scales_lr = 0.005f;
            float popspa_opacities_lr = 0.05f;
            float popspa_quaternions_lr = 0.001f;
            float popspa_sh0_lr = 0.0025f;
            float popspa_shn_lr = 0.000125f;
            float popspa_ssim_weight = 0.2f;

            // Perf / profiling instruments (CLI-only; no env toggles)
            bool perf_bench = false;     // --perf-bench → write perf_bench.json
            int perf_bench_warmup = 200; // --perf-bench-warmup=N
            int profile_start_iter = -1; // --profile-window=START:STOP
            int profile_stop_iter = -1;  // one-past-last profiled iteration

            std::string config_file = "";

            void scale_steps(float ratio);
            void apply_step_scaling();
            void remove_step_scaling();
            [[nodiscard]] int resolved_total_iterations() const;
            [[nodiscard]] float color_variance_weight_at(int iteration) const;
            [[nodiscard]] bool normal_supervision_active(int iter) const;
            [[nodiscard]] float scale_reg_at(int iter) const;
            void resolve_mrnf_capacity_defaults();
            // Every test_every-th image is withheld from training for evaluation.
            [[nodiscard]] bool holds_out_eval_images() const { return enable_eval && !eval_all; }
            [[nodiscard]] int resolved_ppisp_controller_activation_step(int total_iterations) const;

            nlohmann::json to_json() const;
            static OptimizationParameters from_json(const nlohmann::json& j);

            [[nodiscard]] std::string validate() const;

            // Factory methods for strategy presets
            static OptimizationParameters mcmc_defaults();
            static OptimizationParameters mrnf_defaults();
            static OptimizationParameters igs_plus_defaults();
            static OptimizationParameters defaults_for_strategy(std::string_view strategy);
        };

        // eval_mask is a mesh path, "bbox:x0,y0,z0,x1,y1,z1" (a world-space box), "cropbox"
        // (the training model's crop box when training starts), "masks:<folder>" (one mask per image),
        // "depth:near,far" (rendered pixels whose expected depth lies in the range), "points:radius,close"
        // (the initial point cloud splatted with a pixel radius and closed by `close` pixels) or "points:<file>"
        // (the positions of a splat or point cloud PLY, with the default radius and closing).
        [[nodiscard]] LFS_CORE_API bool is_eval_mask_box(std::string_view spec);
        [[nodiscard]] LFS_CORE_API bool is_eval_mask_cropbox(std::string_view spec);
        [[nodiscard]] LFS_CORE_API bool is_eval_mask_folder(std::string_view spec);
        [[nodiscard]] LFS_CORE_API std::string_view eval_mask_folder(std::string_view spec);
        [[nodiscard]] LFS_CORE_API bool is_eval_mask_depth(std::string_view spec);
        [[nodiscard]] LFS_CORE_API std::optional<std::array<float, 2>> parse_eval_mask_depth(std::string_view spec);
        [[nodiscard]] LFS_CORE_API bool is_eval_mask_points(std::string_view spec);
        [[nodiscard]] LFS_CORE_API std::optional<std::array<int, 2>> parse_eval_mask_points(std::string_view spec);
        // The PLY of a points:<file> spec, whose positions replace the initial point cloud.
        [[nodiscard]] LFS_CORE_API std::optional<std::string_view> eval_mask_points_file(std::string_view spec);
        // The PLY of a splat:<file> spec, whose rendered coverage selects the evaluated pixels.
        [[nodiscard]] LFS_CORE_API std::optional<std::string_view> eval_mask_splat_file(std::string_view spec);
        [[nodiscard]] LFS_CORE_API std::optional<std::array<float, 6>> parse_eval_mask_box(std::string_view spec);
        // Three finite numbers rx,ry,rz, as --cropbox-rotation takes them.
        [[nodiscard]] LFS_CORE_API std::optional<std::array<float, 3>> parse_cropbox_rotation(std::string_view spec);
        [[nodiscard]] LFS_CORE_API std::string normalize_eval_mask(std::string_view spec);
        // The sets of an eval_mask_report list in order, the first being the headline set; empty for an empty list.
        [[nodiscard]] LFS_CORE_API lfs::Result<std::vector<EvalPixelSet>> parse_eval_pixel_sets(std::string_view list);

        struct LFS_CORE_API LoadingParams {
            bool use_cpu_memory = true;
            float min_cpu_free_memory_ratio = 0.1f; // make sure at least 10% RAM is free
            float min_cpu_free_GB = 1.0f;           // min GB we want to be free
            bool print_cache_status = true;
            int print_status_freq_num = 500; // every print_status_freq_num calls for load print cache status
            bool use_16bit_color = false;
            // Training images only; evaluation and the GT view load the photos unchanged.
            SharpenParams sharpen;
            nlohmann::json to_json() const;
            static LoadingParams from_json(const nlohmann::json& j);
        };

        struct LFS_CORE_API DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "";
            // True when an output path was explicitly supplied by CLI or runtime API.
            // Runtime-only: to_json, from_json, and PRMS dataset_json omit it.
            bool output_path_explicit = false;
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

            // Dataset import option; persisted in training configs and project PRMS.
            std::string centralize_dataset = "off";

            nlohmann::json to_json() const;
            static DatasetConfig from_json(const nlohmann::json& j);
            [[nodiscard]] std::string validate() const;

            // output_name without a splat or project extension, so "scene.ply" names scene.ply, not scene.ply.ply.
            [[nodiscard]] std::string output_stem() const;
            // output_path/<output_stem>.licht, or output_path/project.licht without an output name.
            [[nodiscard]] std::filesystem::path project_file() const;
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
            bool include_provenance = true; // always written to the format's metadata slot; caller chooses full vs minimal, writers fall back to minimal
        };

        struct TrainingParameters;
        enum class OutputFormat;

        // Process-local presence map for --resume. Keys dumped here are the
        // only ones re-applied after a project/checkpoint restore; omitted
        // keys leave the restored values untouched. Not serialized.
        struct LFS_CORE_API ExplicitTrainingOverrides {
            std::string optimization_json;
            std::string dataset_json;

            [[nodiscard]] bool empty() const noexcept {
                return optimization_json.empty() && dataset_json.empty();
            }
            [[nodiscard]] bool has_optimization_key(std::string_view key) const;
            [[nodiscard]] bool has_dataset_key(std::string_view key) const;
        };

        LFS_CORE_API void merge_explicit_json_overlay(std::string& dst_json, std::string_view src_json);
        LFS_CORE_API void apply_explicit_training_overrides(
            TrainingParameters& target,
            const ExplicitTrainingOverrides& overrides);

        struct LFS_CORE_API TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;
            ServerConfig server;

            // Process-local startup policy. These flags deliberately do not
            // belong to OptimizationParameters: saved training configurations
            // must not make a later normal launch enter safe mode.
            bool safe_mode = false;
            bool no_download = false;
            bool reset_preferences = false;
            bool reset_layout = false;
            bool reset_all_settings = false;
            std::optional<int> mcp_port = std::nullopt;

            // Viewer mode: splat files to load (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)
            std::vector<std::filesystem::path> view_paths;

            // COLMAP sparse folder for camera-only import (no images required)
            std::optional<std::filesystem::path> import_cameras_path = std::nullopt;

            // Optional splat file for initialization (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)
            std::optional<std::string> init_path = std::nullopt;

            // Region of interest: an enabled crop box on the training model, min x, y, z then max x, y, z in the
            // dataset's coordinates. --cropbox-loss-weight and --cropbox-lr-scale set how much the rest still counts.
            std::optional<std::array<float, 6>> cropbox = std::nullopt;
            // Turns the --cropbox box about its centre by X, Y, Z degrees about the dataset's axes, as Rx * Ry * Rz.
            // Unset keeps the box axis-aligned with its corners stored as given.
            std::optional<std::array<float, 3>> cropbox_rotation = std::nullopt;

            // Optional trained splats to append to the training model before optimizer initialization
            std::vector<std::filesystem::path> add_splat_paths;
            std::vector<bool> add_splat_freeze;
            bool add_splats_applied = false;
            float freeze_lr_scale = 0.0f;
            bool exclude_frozen_add_splats_from_export = false;
            bool include_provenance = true; // always written to the format's metadata slot; caller chooses full vs minimal, writers fall back to minimal

            // Checkpoint to resume training from
            std::optional<std::filesystem::path> resume_checkpoint = std::nullopt;

            // Project to open as the GUI lifecycle document.
            std::optional<std::filesystem::path> project_path = std::nullopt;

            // Embedded CKPT project to resume training from. The display model
            // is hydrated first; full trainer state must be loaded before
            // train() is allowed to start.
            std::optional<std::filesystem::path> resume_project = std::nullopt;

            // Untrained .licht passed to --data-path. Its REFS dataset folder,
            // or the embedded dataset copy, becomes the training data source.
            std::optional<std::filesystem::path> dataset_project = std::nullopt;

            // Headless/integration-test trigger for the production training
            // snapshot path. Empty unless the user passed --save-project-path;
            // the trainer then falls back to the bound project destination.
            std::optional<size_t> save_project_at_iteration = std::nullopt;
            std::filesystem::path save_project_path;

            // Headless camera-path -> video render mode (see --render-camera-path)
            std::optional<RenderPathConfig> render_path = std::nullopt;

            // eval subcommand: score the model without training, saving or exporting.
            bool evaluate_only = false;

            // Python scripts to execute for custom training callbacks
            std::vector<std::filesystem::path> python_scripts;

            // Additional final-splat exports written next to project.licht after
            // training completes. Empty = only the .licht project is written.
            std::vector<OutputFormat> export_formats;
            int sog_iterations = 10;
            int lod_levels = 4;
            float lod_ratio = 0.5f;
            int lod_chunk_count = 512;
            float lod_chunk_extent = 16.0f;
            int lod_chunk_min = 8;

            // True when --bg-color was provided on the command line.
            bool cli_bg_color_set = false;
            // True when -i/--iter was provided. Resume adapters use this to
            // distinguish an explicit continuation target from the default.
            bool cli_iterations_set = false;
            // True when the command line set any absolute step value; the GUI
            // image-count auto-scale then leaves that strategy untouched.
            bool cli_step_values_set = false;

            ExplicitTrainingOverrides overrides;

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
                                  RAD,
                                  SSOG,
                                  GLB };

        // Inline so that tools compiled without lfs_core, such as the standalone path tests, can use it.
        [[nodiscard]] constexpr std::string_view output_format_extension(
            const OutputFormat format) noexcept {
            switch (format) {
            case OutputFormat::PLY: return ".ply";
            case OutputFormat::SOG: return ".sog";
            case OutputFormat::SSOG: return ".ssog";
            case OutputFormat::SPZ: return ".spz";
            case OutputFormat::GLB: return ".glb";
            case OutputFormat::HTML: return ".html";
            case OutputFormat::USD: return ".usd";
            case OutputFormat::USDA: return ".usda";
            case OutputFormat::USDC: return ".usdc";
            case OutputFormat::RAD: return ".rad";
            }
            return ".ply";
        }

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
            int lod_levels = 4;
            float lod_ratio = 0.5f;
            int lod_chunk_count = 512;
            float lod_chunk_extent = 16.0f;
            int lod_chunk_min = 8;
            int spz_version = 4; // SPZ container version: 4 (zstd) or 3 (legacy gzip)
            // PLY -> RAD only: replicate the source across an AxB ground-plane
            // grid instead of pre-tiling the input file.
            std::uint32_t tiles_x = 1;
            std::uint32_t tiles_y = 1;
            LodBuilder lod_builder = LodBuilder::BHATT;
            RadExportMode rad_export_mode = RadExportMode::Stream;
            bool overwrite = false;         // Skip overwrite prompts
            bool include_provenance = true; // always written to the format's metadata slot; caller chooses full vs minimal, writers fall back to minimal
        };

        // Parameters for the mesh2splat command
        struct LFS_CORE_API Mesh2SplatParameters {
            std::filesystem::path input_path;
            std::filesystem::path output_path; // Empty = derive from input
            OutputFormat format = OutputFormat::PLY;
            std::vector<OutputFormat> formats{OutputFormat::PLY};
            Mesh2SplatOptions options;
            int sog_iterations = 10;
            int lod_levels = 4;
            float lod_ratio = 0.5f;
            int lod_chunk_count = 512;
            float lod_chunk_extent = 16.0f;
            int lod_chunk_min = 8;
            int spz_version = 4; // SPZ container version: 4 (zstd) or 3 (legacy gzip)
            bool overwrite = false;
            bool include_provenance = true; // always written to the format's metadata slot; caller chooses full vs minimal, writers fall back to minimal
        };

        // Parameters for the compare command
        struct LFS_CORE_API CompareParameters {
            std::filesystem::path reference_path; // An image, a folder of images or a COLMAP dataset folder
            std::filesystem::path test_path;      // The same kind as reference_path
            std::filesystem::path output_path;
            std::filesystem::path colmap_path; // Cameras of a reference folder that is no COLMAP dataset
            // The eval_* options and enable_save_eval_images as in training; undistort says that test images
            // without cameras of their own were rendered with the undistorted reference cameras.
            OptimizationParameters evaluation{.enable_save_eval_images = false};
            bool no_download = false;
        };

        enum class PreprocessOutputMode { Depth,
                                          Normals,
                                          Both };

        // Native in-tree MoGe-2 is the only runtime. The CLI still accepts
        // --inference-backend native (and auto as an alias).
        enum class InferenceBackend { Native };

        struct LFS_CORE_API PreprocessParameters {
            std::filesystem::path dataset_path;
            std::string images_folder = "images";
            std::filesystem::path model_path;
            PreprocessOutputMode mode = PreprocessOutputMode::Both;
            InferenceBackend inference_backend = InferenceBackend::Native;
            int max_side = 518;
            std::int64_t num_tokens = 1800;
            int threads = 0;
            int png_compression = 6;
            int bit_depth = 16;
            bool overwrite = false;
            bool no_download = false;
            bool download_only = false;
            std::vector<std::filesystem::path> image_paths; // Empty = scan images_folder
            std::string depth_folder = "depth";
            std::string normals_folder = "normals";
        };

        // Parameters for an individual batch training job
        struct LFS_CORE_API BatchJobItem {
            std::filesystem::path dataset_path;
            std::string name;                                 // Friendly name; defaults to stem of dataset_path
            std::filesystem::path output_path;                // Explicit output directory
            std::vector<std::string> extra_args;              // Extra CLI args for this specific job
            std::optional<std::filesystem::path> config_file; // Optional per-job JSON config
        };

        // Parameters for the batch-train command
        struct LFS_CORE_API BatchTrainingParameters {
            std::vector<std::filesystem::path> input_paths; // Folders to scan, dataset paths, or manifest JSONs
            std::filesystem::path output_dir;               // Base output directory
            std::filesystem::path config_file;              // Path to batch manifest JSON
            std::vector<BatchJobItem> jobs;                 // Explicit jobs
            std::vector<std::string> shared_args;           // Command-line training flags passed to all jobs
            bool recursive = false;                         // Recursively scan directories for datasets
            bool continue_on_failure = true;                // Keep running subsequent jobs on failure
            bool dry_run = false;                           // Inspect resolved jobs without running
            int timeout_minutes = 0;                        // Timeout per job in minutes (0 = disabled)
            std::filesystem::path summary_file;             // Path for batch_summary.json
            std::filesystem::path report_file;              // Path for batch_report.txt
            std::string log_level = "info";
        };

        struct VideoMaskParameters {
            std::filesystem::path input_path, output_path, model_path;
            std::vector<std::array<float, 3>> points;
            std::optional<std::array<float, 4>> box;
            std::array<double, 4> circle{0.5, 0.5, 0.0, 0.0};
            int seed_frame = 0;
            double fps = 1.0, start_time = 0.0, end_time = -1.0;
            float fisheye_fov = 200, view_fov = 160;
            int dilation = 12;
            bool recenter = false, require_operator = false;
            bool overwrite = false, no_download = false, download_only = false;
        };

        // Modern C++23 functions returning expected values
        LFS_CORE_API std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path);
        LFS_CORE_API std::expected<OptimizationParameters, std::string> read_optim_params_from_json(
            const std::filesystem::path& path,
            ExplicitTrainingOverrides& captured_overrides);
        LFS_CORE_API std::expected<TrainingParameters, lfs::Error> read_training_parameters_from_json(
            const std::filesystem::path& path,
            const TrainingParameters& defaults = {});
        LFS_CORE_API std::expected<BatchTrainingParameters, lfs::Error> read_batch_training_parameters_from_json(
            const std::filesystem::path& path,
            const BatchTrainingParameters& defaults = {});

        // Save training parameters to JSON
        LFS_CORE_API std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);

    } // namespace param
} // namespace lfs::core
