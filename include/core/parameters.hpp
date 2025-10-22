/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace gs {
    namespace param {
        struct OptimizationParameters {
            size_t iterations = 30'000;
            size_t sh_degree_interval = 1'000;
            float means_lr = 0.00016f;
            float shs_lr = 0.0025f;
            float opacity_lr = 0.05f;
            float scaling_lr = 0.005f;
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
            int num_workers = 16;
            int max_cap = 1000000;
            std::vector<size_t> eval_steps = {7'000, 30'000}; // Steps to evaluate the model
            std::vector<size_t> save_steps = {7'000, 30'000}; // Steps to save the model
            bool skip_intermediate_saving = false;            // Skip saving intermediate results and only save final output
            bool bg_modulation = false;                       // Enable sinusoidal background modulation
            bool enable_eval = false;                         // Only evaluate when explicitly enabled
            bool rc = false;                                  // Workaround for reality captures - doesn't properly convert COLMAP camera model
            bool enable_save_eval_images = true;              // Save during evaluation images
            bool headless = false;                            // Disable visualization during training
            std::string render_mode = "RGB";                  // Render mode: RGB, D, ED, RGB_D, RGB_ED
            std::string strategy = "mcmc";                    // Optimization strategy: mcmc, default.
            bool preload_to_ram = false;                      // If true, the entire dataset will be loaded into RAM at startup
            std::string pose_optimization = "none";           // Pose optimization type: none, direct, mlp

            // Bilateral grid parameters
            bool use_bilateral_grid = false;
            int bilateral_grid_X = 16;
            int bilateral_grid_Y = 16;
            int bilateral_grid_W = 8;
            float bilateral_grid_lr = 2e-3f;
            float tv_loss_weight = 10.f;

            // Default strategy specific parameters
            float prune_opacity = 0.005f;
            float grow_scale3d = 0.01f;
            float grow_scale2d = 0.05f;
            float prune_scale3d = 0.1f;
            float prune_scale2d = 0.15f;
            size_t reset_every = 3'000;
            size_t pause_refine_after_reset = 0;
            bool revised_opacity = false;
            bool gut = false;
            float steps_scaler = 0.f;  // If < 0, step size scaling is disabled
            bool antialiasing = false; // Enable antialiasing in rendering

            // Random initialization parameters
            bool random = false;        // Use random initialization instead of SfM
            int init_num_pts = 100'000; // Number of random points to initialize
            float init_extent = 3.0f;   // Extent of random point cloud

            // SOG format parameters
            bool save_sog = false;   // Save in SOG format alongside PLY
            int sog_iterations = 10; // K-means iterations for SOG compression

            // Sparsity optimization parameters
            bool enable_sparsity = false;
            int sparsify_steps = 15000;
            float init_rho = 0.0005f;
            float prune_ratio = 0.6f;

            std::string config_file = "";

            nlohmann::json to_json() const;
            static OptimizationParameters from_json(const nlohmann::json& j);
        };

        struct LoadingParams {
            bool use_cpu_memory = true;
            float min_cpu_free_memory_ratio = 0.1f; // make sure at least 10% RAM is free
            std::size_t min_cpu_free_GB = 1;        // min GB we want to be free
            bool use_fs_cache = true;
            bool print_cache_status = true;
            int print_status_freq_num = 500; // every print_status_freq_num calls for load print cache status

            nlohmann::json to_json() const;
            static LoadingParams from_json(const nlohmann::json& j);
        };

        struct DatasetConfig {
            std::filesystem::path data_path = "";
            std::filesystem::path output_path = "";
            std::filesystem::path project_path = ""; // if path is relative it will be saved to output_path/project_name.ls
            std::string images = "images";
            int resize_factor = -1;
            int test_every = 8;
            std::vector<std::string> timelapse_images = {};
            int timelapse_every = 50;
            int max_width = 3840;
            LoadingParams loading_params;

            nlohmann::json to_json() const;
            static DatasetConfig from_json(const nlohmann::json& j);
        };

        struct TrainingParameters {
            DatasetConfig dataset;
            OptimizationParameters optimization;

            // Viewer mode specific
            std::filesystem::path ply_path = "";

            // Optional PLY splat file for initialization
            std::optional<std::string> init_ply = std::nullopt;
        };

        // Modern C++23 functions returning expected values
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(std::filesystem::path& path);

        // Save training parameters to JSON
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path);

        std::expected<LoadingParams, std::string> read_loading_params_from_json(std::filesystem::path& path);
    } // namespace param
} // namespace gs