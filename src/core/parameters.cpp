/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include <chrono>
#include <ctime>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace lfs::core {
    namespace param {
        namespace {

            /**
             * @brief Read and parse a JSON configuration file
             * @param path Path to the JSON file
             * @return Expected JSON object or error message
             */
            std::expected<nlohmann::json, std::string> read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Configuration file does not exist: {}", path.string()));
                }

                std::ifstream file(path);
                if (!file.is_open()) {
                    return std::unexpected(std::format("Could not open configuration file: {}", path.string()));
                }

                try {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    return nlohmann::json::parse(buffer.str());
                } catch (const nlohmann::json::parse_error& e) {
                    return std::unexpected(std::format("JSON parsing error in {}: {}", path.string(), e.what()));
                }
            }

            /**
             * @brief Verify optimization parameters between JSON and struct defaults
             *
             * This function performs a comprehensive verification of optimization parameters:
             * 1. Checks if all struct parameters exist in JSON
             * 2. Verifies values match between JSON and struct defaults
             * 3. Identifies unknown parameters in JSON (with warnings)
             * 4. Provides detailed error reporting
             *
             * @param defaults The default parameters from the struct
             * @param json The JSON configuration to verify
             * @param strict If true, unknown parameters are treated as errors
             * @return bool True if verification passed, false otherwise
             */
            bool verify_optimization_parameters(const OptimizationParameters& defaults,
                                                const nlohmann::json& json,
                                                bool strict = false) {
                bool all_match = true;
                std::vector<std::string> missing_in_json;
                std::vector<std::string> unknown_params;
                std::vector<std::string> mismatched_values;

                // Define all expected parameters and their types
                struct ParamInfo {
                    std::string name;
                    std::variant<int, size_t, float, int64_t, std::string, bool> value;
                    std::string description; // Added for better documentation
                };

                const std::vector<ParamInfo> expected_params = {
                    {"iterations", defaults.iterations, "Total number of training iterations"},
                    {"means_lr", defaults.means_lr, "Initial learning rate for position updates"},
                    {"shs_lr", defaults.shs_lr, "Learning rate for spherical harmonics updates"},
                    {"opacity_lr", defaults.opacity_lr, "Learning rate for opacity updates"},
                    {"scaling_lr", defaults.scaling_lr, "Learning rate for scaling updates"},
                    {"rotation_lr", defaults.rotation_lr, "Learning rate for rotation updates"},
                    {"lambda_dssim", defaults.lambda_dssim, "DSSIM loss weight"},
                    {"min_opacity", defaults.min_opacity, "Minimum opacity threshold"},
                    {"refine_every", defaults.refine_every, "Interval between densification steps"},
                    {"start_refine", defaults.start_refine, "Starting iteration for densification"},
                    {"stop_refine", defaults.stop_refine, "Ending iteration for densification"},
                    {"grad_threshold", defaults.grad_threshold, "Gradient threshold for densification"},
                    {"opacity_reg", defaults.opacity_reg, "Opacity L1 regularization weight"},
                    {"scale_reg", defaults.scale_reg, "Scale L1 regularization weight"},
                    {"init_opacity", defaults.init_opacity, "Initial opacity value for new Gaussians"},
                    {"init_scaling", defaults.init_scaling, "Initial scaling value for new Gaussians"},
                    {"sh_degree", defaults.sh_degree, "Spherical harmonics degree"},
                    {"num_workers", defaults.num_workers, "Number of image loader threads"},
                    {"max_cap", defaults.max_cap, "Maximum number of Gaussians for MCMC strategy"},
                    {"strategy", defaults.strategy, "Optimization strategy: mcmc, default"},
                    {"enable_eval", defaults.enable_eval, "Enable evaluation during training"},
                    {"enable_save_eval_images", defaults.enable_save_eval_images, "Save images during evaluation"},
                    {"use_bilateral_grid", defaults.use_bilateral_grid, "Enable bilateral grid for appearance modeling"},
                    {"bilateral_grid_X", defaults.bilateral_grid_X, "Bilateral grid X dimension"},
                    {"bilateral_grid_Y", defaults.bilateral_grid_Y, "Bilateral grid Y dimension"},
                    {"bilateral_grid_W", defaults.bilateral_grid_W, "Bilateral grid W dimension"},
                    {"bilateral_grid_lr", defaults.bilateral_grid_lr, "Learning rate for bilateral grid"},
                    {"tv_loss_weight", defaults.tv_loss_weight, "Weight for total variation loss"},
                    {"prune_opacity", defaults.prune_opacity, "Opacity pruning threshold"},
                    {"grow_scale3d", defaults.grow_scale3d, "3D scale threshold for duplication"},
                    {"grow_scale2d", defaults.grow_scale2d, "2D scale threshold for splitting"},
                    {"prune_scale3d", defaults.prune_scale3d, "3D scale threshold for pruning"},
                    {"prune_scale2d", defaults.prune_scale2d, "2D scale threshold for pruning"},
                    {"reset_every", defaults.reset_every, "Reset opacity every this many iterations"},
                    {"pause_refine_after_reset", defaults.pause_refine_after_reset, "Pause refinement after reset for N iterations"},
                    {"revised_opacity", defaults.revised_opacity, "Use revised opacity heuristic"},
                    {"steps_scaler", defaults.steps_scaler, "Scales the training steps and values"},
                    {"sh_degree_interval", defaults.sh_degree_interval, "Interval for increasing SH degree"},
                    {"random", defaults.random, "Use random initialization instead of SfM"},
                    {"init_num_pts", defaults.init_num_pts, "Number of random initialization points"},
                    {"init_extent", defaults.init_extent, "Extent of random initialization"},
                    {"tile_mode", defaults.tile_mode, "Tile mode for memory-efficient training (1=1 tile, 2=2 tiles, 4=4 tiles)"},
                    {"enable_sparsity", defaults.enable_sparsity, "Enable sparsity optimization"},
                    {"sparsify_steps", defaults.sparsify_steps, "Number of steps for sparsification"},
                    {"init_rho", defaults.init_rho, "Initial ADMM penalty parameter"},
                    {"prune_ratio", defaults.prune_ratio, "Final pruning ratio for sparsity"},
                    {"bg_modulation", defaults.bg_modulation, "Enable sinusoidal background modulation"},
                    {"gut", defaults.gut, "Enable GUT mode"},
                    {"mask_mode", std::string("none"), "Mask mode: none, segment, ignore, alpha_consistent"},
                    {"invert_masks", defaults.invert_masks, "Invert mask values"},
                    {"mask_opacity_penalty_weight", defaults.mask_opacity_penalty_weight, "Opacity penalty weight for segment mode"},
                    {"mask_opacity_penalty_power", defaults.mask_opacity_penalty_power, "Penalty falloff power"},
                    {"mask_threshold", defaults.mask_threshold, "Mask threshold value"}};

                // Check all expected parameters
                for (const auto& param : expected_params) {
                    if (!json.contains(param.name)) {
                        // Skip eval_steps and save_steps as they are handled separately
                        if (param.name != "eval_steps" && param.name != "save_steps") {
                            missing_in_json.push_back(param.name);
                            all_match = false;
                        }
                        continue;
                    }

                    // Compare values based on type
                    std::visit([&](const auto& default_val) {
                        using T = std::decay_t<decltype(default_val)>;
                        try {
                            if (json[param.name].get<T>() != default_val) {
                                mismatched_values.push_back(param.name);
                                all_match = false;
                            }
                        } catch (...) {
                            // Type mismatch
                            mismatched_values.push_back(param.name + " (type mismatch)");
                            all_match = false;
                        }
                    },
                               param.value);
                }

                // Check for any unknown parameters in JSON
                for (const auto& [key, value] : json.items()) {
                    bool found = false;
                    for (const auto& param : expected_params) {
                        if (key == param.name) {
                            found = true;
                            break;
                        }
                    }
                    if (key == "eval_steps" || key == "save_steps") {
                        found = true;
                    }
                    if (!found) {
                        unknown_params.push_back(key);
                        if (strict) {
                            all_match = false;
                        }
                    }
                }

                // Log mismatches
                if (!mismatched_values.empty()) {
                    for (const auto& param : mismatched_values) {
                        for (const auto& p : expected_params) {
                            if (p.name == param) {
                                std::string default_val;
                                std::visit([&](const auto& val) {
                                    default_val = std::format("{}", val);
                                },
                                           p.value);
                                LOG_DEBUG("Parameter mismatch: {} (JSON={}, default={})",
                                          param, json[param].dump(), default_val);
                                break;
                            }
                        }
                    }
                }

                if (!unknown_params.empty()) {
                    for (const auto& param : unknown_params) {
                        LOG_DEBUG("Unknown parameter in JSON: {}", param);
                    }
                }

                return all_match;
            }
        } // namespace

        nlohmann::json OptimizationParameters::to_json() const {

            nlohmann::json opt_json;
            opt_json["iterations"] = iterations;
            opt_json["means_lr"] = means_lr;
            opt_json["shs_lr"] = shs_lr;
            opt_json["opacity_lr"] = opacity_lr;
            opt_json["scaling_lr"] = scaling_lr;
            opt_json["rotation_lr"] = rotation_lr;
            opt_json["lambda_dssim"] = lambda_dssim;
            opt_json["min_opacity"] = min_opacity;
            opt_json["refine_every"] = refine_every;
            opt_json["start_refine"] = start_refine;
            opt_json["stop_refine"] = stop_refine;
            opt_json["grad_threshold"] = grad_threshold;
            opt_json["sh_degree"] = sh_degree;
            opt_json["opacity_reg"] = opacity_reg;
            opt_json["scale_reg"] = scale_reg;
            opt_json["init_opacity"] = init_opacity;
            opt_json["init_scaling"] = init_scaling;
            opt_json["num_workers"] = num_workers;
            opt_json["max_cap"] = max_cap;
            opt_json["eval_steps"] = eval_steps;
            opt_json["save_steps"] = save_steps;
            opt_json["enable_eval"] = enable_eval;
            opt_json["enable_save_eval_images"] = enable_save_eval_images;
            opt_json["strategy"] = strategy;
            opt_json["use_bilateral_grid"] = use_bilateral_grid;
            opt_json["bilateral_grid_X"] = bilateral_grid_X;
            opt_json["bilateral_grid_Y"] = bilateral_grid_Y;
            opt_json["bilateral_grid_W"] = bilateral_grid_W;
            opt_json["bilateral_grid_lr"] = bilateral_grid_lr;
            opt_json["tv_loss_weight"] = tv_loss_weight;
            opt_json["prune_opacity"] = prune_opacity;
            opt_json["grow_scale3d"] = grow_scale3d;
            opt_json["grow_scale2d"] = grow_scale2d;
            opt_json["prune_scale3d"] = prune_scale3d;
            opt_json["prune_scale2d"] = prune_scale2d;
            opt_json["reset_every"] = reset_every;
            opt_json["pause_refine_after_reset"] = pause_refine_after_reset;
            opt_json["revised_opacity"] = revised_opacity;
            opt_json["gut"] = gut;
            opt_json["steps_scaler"] = steps_scaler;
            opt_json["sh_degree_interval"] = sh_degree_interval;
            opt_json["random"] = random;
            opt_json["init_num_pts"] = init_num_pts;
            opt_json["init_extent"] = init_extent;
            opt_json["tile_mode"] = tile_mode;
            opt_json["enable_sparsity"] = enable_sparsity;
            opt_json["sparsify_steps"] = sparsify_steps;
            opt_json["init_rho"] = init_rho;
            opt_json["prune_ratio"] = prune_ratio;
            opt_json["bg_modulation"] = bg_modulation;

            // Mask parameters
            static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent"};
            opt_json["mask_mode"] = MASK_MODE_NAMES[static_cast<int>(mask_mode)];
            opt_json["invert_masks"] = invert_masks;
            opt_json["mask_opacity_penalty_weight"] = mask_opacity_penalty_weight;
            opt_json["mask_opacity_penalty_power"] = mask_opacity_penalty_power;
            opt_json["mask_threshold"] = mask_threshold;

            return opt_json;
        }

        OptimizationParameters OptimizationParameters::from_json(const nlohmann::json& json) {

            OptimizationParameters params;
            params.iterations = json["iterations"];
            params.means_lr = json["means_lr"];
            params.shs_lr = json["shs_lr"];
            params.opacity_lr = json["opacity_lr"];
            params.scaling_lr = json["scaling_lr"];
            params.rotation_lr = json["rotation_lr"];
            params.lambda_dssim = json["lambda_dssim"];
            params.min_opacity = json["min_opacity"];
            params.refine_every = json["refine_every"];
            params.start_refine = json["start_refine"];
            params.stop_refine = json["stop_refine"];
            params.grad_threshold = json["grad_threshold"];
            params.sh_degree = json["sh_degree"];

            if (json.contains("opacity_reg")) {
                params.opacity_reg = json["opacity_reg"];
            }
            if (json.contains("scale_reg")) {
                params.scale_reg = json["scale_reg"];
            }
            if (json.contains("init_opacity")) {
                params.init_opacity = json["init_opacity"];
            }
            if (json.contains("init_scaling")) {
                params.init_scaling = json["init_scaling"];
            }
            if (json.contains("num_workers")) {
                params.num_workers = json["num_workers"];
            }
            if (json.contains("max_cap")) {
                params.max_cap = json["max_cap"];
            }

            if (json.contains("strategy")) {
                std::string strategy = json["strategy"];
                if (strategy == "mcmc" || strategy == "default") {
                    params.strategy = strategy;
                } else {
                    LOG_WARN("Invalid strategy '{}' in JSON, using default", strategy);
                }
            }

            if (json.contains("eval_steps")) {
                params.eval_steps.clear();
                for (const auto& step : json["eval_steps"]) {
                    params.eval_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("save_steps")) {
                params.save_steps.clear();
                for (const auto& step : json["save_steps"]) {
                    params.save_steps.push_back(step.get<size_t>());
                }
            }

            if (json.contains("enable_eval")) {
                params.enable_eval = json["enable_eval"];
            }
            if (json.contains("enable_save_eval_images")) {
                params.enable_save_eval_images = json["enable_save_eval_images"];
            }
            if (json.contains("use_bilateral_grid")) {
                params.use_bilateral_grid = json["use_bilateral_grid"];
            }
            if (json.contains("bilateral_grid_X")) {
                params.bilateral_grid_X = json["bilateral_grid_X"];
            }
            if (json.contains("bilateral_grid_Y")) {
                params.bilateral_grid_Y = json["bilateral_grid_Y"];
            }
            if (json.contains("bilateral_grid_W")) {
                params.bilateral_grid_W = json["bilateral_grid_W"];
            }
            if (json.contains("bilateral_grid_lr")) {
                params.bilateral_grid_lr = json["bilateral_grid_lr"];
            }
            if (json.contains("tv_loss_weight")) {
                params.tv_loss_weight = json["tv_loss_weight"];
            }
            if (json.contains("prune_opacity")) {
                params.prune_opacity = json["prune_opacity"];
            }
            if (json.contains("grow_scale3d")) {
                params.grow_scale3d = json["grow_scale3d"];
            }
            if (json.contains("grow_scale2d")) {
                params.grow_scale2d = json["grow_scale2d"];
            }
            if (json.contains("prune_scale3d")) {
                params.prune_scale3d = json["prune_scale3d"];
            }
            if (json.contains("prune_scale2d")) {
                params.prune_scale2d = json["prune_scale2d"];
            }
            if (json.contains("reset_every")) {
                params.reset_every = json["reset_every"];
            }
            if (json.contains("pause_refine_after_reset")) {
                params.pause_refine_after_reset = json["pause_refine_after_reset"];
            }
            if (json.contains("revised_opacity")) {
                params.revised_opacity = json["revised_opacity"];
            }
            if (json.contains("steps_scaler")) {
                params.steps_scaler = json["steps_scaler"];
            }
            if (json.contains("sh_degree_interval")) {
                params.sh_degree_interval = json["sh_degree_interval"];
            }
            if (json.contains("random")) {
                params.random = json["random"];
            }
            if (json.contains("init_num_pts")) {
                params.init_num_pts = json["init_num_pts"];
            }
            if (json.contains("init_extent")) {
                params.init_extent = json["init_extent"];
            }
            if (json.contains("tile_mode")) {
                params.tile_mode = json["tile_mode"];
            }
            if (json.contains("enable_sparsity")) {
                params.enable_sparsity = json["enable_sparsity"];
            }
            if (json.contains("sparsify_steps")) {
                params.sparsify_steps = json["sparsify_steps"];
            }
            if (json.contains("init_rho")) {
                params.init_rho = json["init_rho"];
            }
            if (json.contains("prune_ratio")) {
                params.prune_ratio = json["prune_ratio"];
            }
            if (json.contains("bg_modulation")) {
                params.bg_modulation = json["bg_modulation"];
            }
            if (json.contains("gut")) {
                params.gut = json["gut"];
            }

            // Mask parameters
            if (json.contains("mask_mode")) {
                std::string mode = json["mask_mode"];
                if (mode == "none") {
                    params.mask_mode = MaskMode::None;
                } else if (mode == "segment") {
                    params.mask_mode = MaskMode::Segment;
                } else if (mode == "ignore") {
                    params.mask_mode = MaskMode::Ignore;
                } else if (mode == "alpha_consistent") {
                    params.mask_mode = MaskMode::AlphaConsistent;
                }
            }
            if (json.contains("invert_masks")) {
                params.invert_masks = json["invert_masks"];
            }
            if (json.contains("mask_opacity_penalty_weight")) {
                params.mask_opacity_penalty_weight = json["mask_opacity_penalty_weight"];
            }
            if (json.contains("mask_opacity_penalty_power")) {
                params.mask_opacity_penalty_power = json["mask_opacity_penalty_power"];
            }
            if (json.contains("mask_threshold")) {
                params.mask_threshold = json["mask_threshold"];
            }

            return params;
        }

        /**
         * @brief Read optimization parameters from JSON file
         * @param[in] json file to load
         * @return Expected OptimizationParameters or error message
         */
        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path) {
            auto json_result = read_json_file(path);

            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            auto json = *json_result;

            // Create default parameters for verification
            OptimizationParameters defaults;

            // Verify parameters before reading
            verify_optimization_parameters(defaults, json);

            try {
                OptimizationParameters params = OptimizationParameters::from_json(json);

                return params;

            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error parsing optimization parameters: {}", e.what()));
            }
        }

        /**
         * @brief Save full training parameters (dataset + optimization) to JSON
         * @param params The full training parameters
         * @param output_path Path to the output directory
         * @return Expected void or error message
         */
        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {

            try {
                nlohmann::json json;

                // Dataset configuration
                json["dataset"] = params.dataset.to_json();

                // Optimization configuration
                nlohmann::json opt_json = params.optimization.to_json();

                json["optimization"] = opt_json;

                // Add timestamp
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                json["timestamp"] = ss.str();

                // Use path directly if .json, otherwise append default filename
                const std::filesystem::path filepath = (output_path.extension() == ".json")
                                                           ? output_path
                                                           : output_path / "training_config.json";
                std::ofstream file(filepath);
                if (!file.is_open()) {
                    return std::unexpected(std::format("Could not open file for writing: {}", filepath.string()));
                }

                file << json.dump(4);
                file.close();

                LOG_INFO("Saved training config to: {}", filepath.string());
                return {};

            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error saving training parameters: {}", e.what()));
            }
        }

        LoadingParams LoadingParams::from_json(const nlohmann::json& j) {

            LoadingParams params;
            if (j.contains("use_cpu_memory")) {
                params.use_cpu_memory = j["use_cpu_memory"];
            }
            if (j.contains("min_cpu_free_memory_ratio")) {
                params.min_cpu_free_memory_ratio = j["min_cpu_free_memory_ratio"];
            }
            if (j.contains("min_cpu_free_GB")) {
                params.min_cpu_free_GB = j["min_cpu_free_GB"];
            }
            if (j.contains("use_fs_cache")) {
                params.use_fs_cache = j["use_fs_cache"];
            }
            if (j.contains("print_cache_status")) {
                params.print_cache_status = j["print_cache_status"];
            }
            if (j.contains("print_status_freq_num")) {
                params.print_status_freq_num = j["print_status_freq_num"];
            }

            return params;
        }

        nlohmann::json LoadingParams::to_json() const {
            nlohmann::json loading_json;
            loading_json["use_cpu_memory"] = use_cpu_memory;
            loading_json["min_cpu_free_memory_ratio"] = min_cpu_free_memory_ratio;
            loading_json["min_cpu_free_GB"] = min_cpu_free_GB;
            loading_json["use_fs_cache"] = use_fs_cache;
            loading_json["print_cache_status"] = print_cache_status;
            loading_json["print_status_freq_num"] = print_status_freq_num;

            return loading_json;
        }

        nlohmann::json DatasetConfig::to_json() const {
            nlohmann::json json;

            json["data_path"] = data_path.string();
            json["output_folder"] = output_path.string();
            json["images"] = images;
            json["resize_factor"] = resize_factor;
            json["test_every"] = test_every;
            json["max_width"] = max_width;
            json["loading_params"] = loading_params.to_json();
            json["invert_masks"] = invert_masks;
            json["mask_threshold"] = mask_threshold;

            return json;
        }

        DatasetConfig DatasetConfig::from_json(const nlohmann::json& j) {
            DatasetConfig dataset;

            dataset.data_path = j["data_path"].get<std::string>();
            dataset.images = j["images"].get<std::string>();
            dataset.resize_factor = j["resize_factor"].get<int>();
            dataset.max_width = j["max_width"].get<int>();
            dataset.test_every = j["test_every"].get<int>();
            dataset.output_path = j["output_folder"].get<std::string>();

            if (j.contains("loading_params")) {
                dataset.loading_params = LoadingParams::from_json(j["loading_params"]);
            }
            if (j.contains("invert_masks")) {
                dataset.invert_masks = j["invert_masks"].get<bool>();
            }
            if (j.contains("mask_threshold")) {
                dataset.mask_threshold = j["mask_threshold"].get<float>();
            }

            return dataset;
        }

        std::expected<LoadingParams, std::string> read_loading_params_from_json(const std::filesystem::path& path) {
            auto json_result = read_json_file(path);

            if (!json_result) {
                return std::unexpected(json_result.error());
            }
            LoadingParams loading_params;
            try {
                loading_params = LoadingParams::from_json(*json_result);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error reading loading parameters: {}", e.what()));
            }
            return loading_params;
        }

        std::filesystem::path get_parameter_file_path(const std::string& filename) {
            static constexpr const char* PARAM_DIR = "parameter";

            // Primary: Use runtime-detected resource directory
            auto runtime_path = lfs::core::getResourceBaseDir() / PARAM_DIR / filename;
            if (std::filesystem::exists(runtime_path)) {
                return runtime_path;
            }

            // Fallback: Search up from executable directory
#ifdef _WIN32
            char exe_buf[MAX_PATH];
            GetModuleFileNameA(nullptr, exe_buf, MAX_PATH);
            auto search_dir = std::filesystem::path(exe_buf).parent_path();
#else
            auto search_dir = std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
            while (!search_dir.empty()) {
                if (const auto path = search_dir / PARAM_DIR / filename; std::filesystem::exists(path)) {
                    return path;
                }
                const auto parent = search_dir.parent_path();
                if (parent == search_dir)
                    break;
                search_dir = parent;
            }
            return runtime_path; // Return runtime path even if not found (for error message)
        }

    } // namespace param
} // namespace lfs::core