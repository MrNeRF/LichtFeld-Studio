/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <chrono>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

namespace lfs::core {
    namespace param {
        namespace {
            std::expected<nlohmann::json, std::string> read_json_file(const std::filesystem::path& path) {
                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Config file not found: {}", path_to_utf8(path)));
                }

                std::ifstream file;
                if (!open_file_for_read(path, file)) {
                    return std::unexpected(std::format("Cannot open config: {}", path_to_utf8(path)));
                }

                try {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    return nlohmann::json::parse(buffer.str());
                } catch (const nlohmann::json::parse_error& e) {
                    return std::unexpected(std::format("JSON parse error in {}: {}", path_to_utf8(path), e.what()));
                }
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
            opt_json["max_cap"] = max_cap;
            opt_json["eval_steps"] = eval_steps;
            opt_json["save_steps"] = save_steps;
            opt_json["enable_eval"] = enable_eval;
            opt_json["enable_save_eval_images"] = enable_save_eval_images;
            opt_json["strategy"] = strategy;
            opt_json["mip_filter"] = mip_filter;
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

        OptimizationParameters OptimizationParameters::mcmc_defaults() {
            return {};
        }

        OptimizationParameters OptimizationParameters::default_strategy_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = "default";
            p.opacity_lr = 0.025f;
            p.stop_refine = 15'000;
            p.opacity_reg = 0.0f;
            p.scale_reg = 0.0f;
            p.init_opacity = 0.1f;
            p.max_cap = 6'000'000;
            p.tv_loss_weight = 5.0f;
            return p;
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
            if (json.contains("mip_filter")) {
                params.mip_filter = json["mip_filter"];
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

        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path) {
            auto json_result = read_json_file(path);
            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            const auto& json = *json_result;
            // Support both flat and nested {"optimization": {...}} formats
            const auto& opt_json = json.contains("optimization") ? json["optimization"] : json;

            try {
                return OptimizationParameters::from_json(opt_json);
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error parsing optimization parameters: {}", e.what()));
            }
        }

        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {
            try {
                nlohmann::json json;
                json["dataset"] = params.dataset.to_json();
                json["optimization"] = params.optimization.to_json();

                const auto now = std::chrono::system_clock::now();
                const auto time_t = std::chrono::system_clock::to_time_t(now);
                std::stringstream ss;
                ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                json["timestamp"] = ss.str();

                const std::filesystem::path filepath = (output_path.extension() == ".json")
                                                           ? output_path
                                                           : output_path / "training_config.json";
                std::ofstream file;
                if (!open_file_for_write(filepath, file)) {
                    return std::unexpected(std::format("Cannot write: {}", path_to_utf8(filepath)));
                }

                file << json.dump(4);
                LOG_INFO("Saved config: {}", path_to_utf8(filepath));
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

            json["data_path"] = path_to_utf8(data_path);
            json["output_folder"] = path_to_utf8(output_path);
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

            // Use utf8_to_path for proper Unicode handling since JSON is UTF-8 encoded
            dataset.data_path = utf8_to_path(j["data_path"].get<std::string>());
            dataset.images = j["images"].get<std::string>();
            dataset.resize_factor = j["resize_factor"].get<int>();
            dataset.max_width = j["max_width"].get<int>();
            dataset.test_every = j["test_every"].get<int>();
            dataset.output_path = utf8_to_path(j["output_folder"].get<std::string>());

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

    } // namespace param
} // namespace lfs::core
