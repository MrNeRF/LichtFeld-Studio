/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/logger.hpp"
#include "core/optimization_properties.hpp"
#include "core/path_utils.hpp"
#include "core/property_registry.hpp"
#include <any>
#include <chrono>
#include <cmath>

#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>

namespace lfs::core {
    namespace param {
        namespace {
            using prop::PropertyMeta;
            using prop::PropertyObjectRef;
            using prop::PropertyRegistry;
            using prop::PropType;

            [[nodiscard]] std::string_view optimization_json_key(const PropertyMeta& meta) {
                return meta.json_key.empty() ? std::string_view(meta.id) : std::string_view(meta.json_key);
            }

            [[nodiscard]] prop::PropertyGroup optimization_property_snapshot() {
                ensure_optimization_properties_registered();
                auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
                if (!group)
                    throw std::runtime_error("Optimization property registry is unavailable");
                return std::move(*group);
            }

            void write_registered_optimization_properties(
                nlohmann::json& json,
                const OptimizationParameters& params) {
                const auto group = optimization_property_snapshot();
                const auto ref = PropertyObjectRef::cpp(
                    const_cast<OptimizationParameters*>(&params));

                for (const auto& meta : group.properties) {
                    if (!meta.getter)
                        throw std::runtime_error("Optimization property has no getter: " + meta.id);

                    const std::string key(optimization_json_key(meta));
                    const auto value = meta.getter(ref);
                    switch (meta.type) {
                    case PropType::Bool:
                        json[key] = std::any_cast<bool>(value);
                        break;
                    case PropType::Int:
                        json[key] = std::any_cast<int>(value);
                        break;
                    case PropType::Float:
                        json[key] = std::any_cast<float>(value);
                        break;
                    case PropType::String:
                        json[key] = std::any_cast<std::string>(value);
                        break;
                    case PropType::SizeT:
                        json[key] = std::any_cast<size_t>(value);
                        break;
                    case PropType::Enum: {
                        const int enum_value = std::any_cast<int>(value);
                        const auto item = std::ranges::find_if(meta.enum_items, [enum_value](const auto& candidate) {
                            return candidate.value == enum_value;
                        });
                        if (item == meta.enum_items.end())
                            throw std::runtime_error("Optimization enum has an invalid value: " + meta.id);
                        json[key] = item->wire_value.empty() ? item->identifier : item->wire_value;
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported registered optimization JSON type: " + meta.id);
                    }
                }
            }

            void read_registered_optimization_properties(
                const nlohmann::json& json,
                OptimizationParameters& params) {
                const auto group = optimization_property_snapshot();
                auto ref = PropertyObjectRef::cpp(&params);

                for (const auto& meta : group.properties) {
                    const std::string key(optimization_json_key(meta));
                    if (!meta.json_required && !json.contains(key))
                        continue;

                    const auto& value = json.at(key);
                    if (meta.id == "strategy")
                        continue;
                    if (!meta.setter)
                        throw std::runtime_error("Optimization property has no setter: " + meta.id);

                    switch (meta.type) {
                    case PropType::Bool:
                        meta.setter(ref, std::any(value.get<bool>()));
                        break;
                    case PropType::Int:
                        meta.setter(ref, std::any(value.get<int>()));
                        break;
                    case PropType::Float:
                        meta.setter(ref, std::any(value.get<float>()));
                        break;
                    case PropType::String:
                        meta.setter(ref, std::any(value.get<std::string>()));
                        break;
                    case PropType::SizeT:
                        meta.setter(ref, std::any(value.get<size_t>()));
                        break;
                    case PropType::Enum: {
                        const auto wire_value = value.get<std::string>();
                        const auto item = std::ranges::find_if(meta.enum_items, [&wire_value](const auto& candidate) {
                            const auto& candidate_wire = candidate.wire_value.empty()
                                                             ? candidate.identifier
                                                             : candidate.wire_value;
                            return candidate_wire == wire_value;
                        });
                        if (item != meta.enum_items.end())
                            meta.setter(ref, std::any(item->value));
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported registered optimization JSON type: " + meta.id);
                    }
                }
            }

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

        void OptimizationParameters::scale_steps(const float ratio) {
            const auto apply = [ratio](const size_t v) {
                return static_cast<size_t>(std::lround(static_cast<float>(v) * ratio));
            };
            iterations = apply(iterations);
            start_refine = apply(start_refine);
            stop_refine = apply(stop_refine);
            reset_every = apply(reset_every);
            refine_every = apply(refine_every);
            sh_degree_interval = apply(sh_degree_interval);
            grow_until_iter = apply(grow_until_iter);

            for (auto* steps : {&eval_steps, &save_steps}) {
                std::set<size_t> unique;
                for (const auto s : *steps) {
                    if (const size_t scaled = apply(s); scaled > 0)
                        unique.insert(scaled);
                }
                steps->assign(unique.begin(), unique.end());
            }
        }

        void OptimizationParameters::apply_step_scaling() {
            if (steps_scaler <= 0.f || steps_scaler == 1.f)
                return;
            LOG_INFO("Scaling training steps by factor: {}", steps_scaler);
            scale_steps(steps_scaler);
        }

        void OptimizationParameters::remove_step_scaling() {
            if (steps_scaler <= 0.f || steps_scaler == 1.f)
                return;
            scale_steps(1.0f / steps_scaler);
        }

        int OptimizationParameters::resolved_total_iterations() const {
            const int base_iters = static_cast<int>(iterations);
            const int sparse_tail = enable_sparsity ? std::max(0, sparsify_steps) : 0;
            return base_iters + sparse_tail;
        }

        int OptimizationParameters::resolved_ppisp_controller_activation_step(const int total_iterations) const {
            if (ppisp_controller_activation_step >= 0)
                return ppisp_controller_activation_step;

            const float clamped_scaler = std::max(steps_scaler, 1.0f);
            const int tail_iters = static_cast<int>(std::lround(5000.0f * clamped_scaler));
            return std::max(0, total_iterations - tail_iters);
        }

        nlohmann::json OptimizationParameters::to_json() const {
            nlohmann::json opt_json;
            write_registered_optimization_properties(opt_json, *this);

            const auto canonical_strategy = canonical_strategy_name(strategy);
            opt_json["strategy"] = canonical_strategy.empty() ? strategy : std::string(canonical_strategy);

            // Residue not represented by scalar registry properties.
            opt_json["eval_steps"] = eval_steps;
            opt_json["save_steps"] = save_steps;
            opt_json["enable_save_eval_images"] = enable_save_eval_images;
            opt_json["ppisp_sidecar_path"] = path_to_utf8(ppisp_sidecar_path);
            opt_json["bg_color"] = {bg_color[0], bg_color[1], bg_color[2]};
            if (!bg_image_path.empty())
                opt_json["bg_image_path"] = path_to_utf8(bg_image_path);

            return opt_json;
        }

        std::string OptimizationParameters::validate() const {
            const auto invalid_nonnegative = [](const float value, const std::string_view name) -> std::string {
                if (!std::isfinite(value) || value < 0.0f)
                    return std::format("{} must be finite and nonnegative (got {})", name, value);
                return {};
            };
            const auto invalid_probability = [](const float value, const std::string_view name) -> std::string {
                if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
                    return std::format("{} must be finite and within [0, 1] (got {})", name, value);
                return {};
            };
            constexpr size_t MAX_ITERATION_VALUE = static_cast<size_t>(std::numeric_limits<int>::max());

            if (!is_valid_strategy_name(strategy))
                return std::format("strategy must be one of mcmc, mrnf, or igs+ (got '{}')", strategy);
            if (iterations == 0 || iterations > MAX_ITERATION_VALUE)
                return std::format("iterations must be within [1, {}] (got {})", MAX_ITERATION_VALUE, iterations);
            if (refine_every == 0 || refine_every > MAX_ITERATION_VALUE)
                return std::format("refine_every must be within [1, {}] (got {})", MAX_ITERATION_VALUE, refine_every);
            if (reset_every == 0 || reset_every > MAX_ITERATION_VALUE)
                return std::format("reset_every must be within [1, {}] (got {})", MAX_ITERATION_VALUE, reset_every);
            if (sh_degree_interval == 0 || sh_degree_interval > MAX_ITERATION_VALUE)
                return std::format("sh_degree_interval must be within [1, {}] (got {})", MAX_ITERATION_VALUE, sh_degree_interval);
            if (start_refine > stop_refine)
                return std::format("start_refine must not exceed stop_refine ({} > {})", start_refine, stop_refine);
            if (start_refine > MAX_ITERATION_VALUE || stop_refine > MAX_ITERATION_VALUE ||
                grow_until_iter > MAX_ITERATION_VALUE)
                return "refinement iteration fields must fit in a signed int";
            if (max_cap < 0)
                return std::format("max_cap must be nonnegative (got {})", max_cap);
            if (sh_degree < 0 || sh_degree > 3)
                return std::format("sh_degree must be within [0, 3] (got {})", sh_degree);
            if (sparsify_steps < 0)
                return std::format("sparsify_steps must be nonnegative (got {})", sparsify_steps);
            if (enable_sparsity &&
                static_cast<uint64_t>(iterations) + static_cast<uint64_t>(sparsify_steps) >
                    static_cast<uint64_t>(std::numeric_limits<int>::max()))
                return "iterations plus sparsify_steps must fit in a signed int";
            if (init_num_pts <= 0)
                return std::format("init_num_pts must be positive (got {})", init_num_pts);
            if (!std::isfinite(init_extent) || init_extent <= 0.0f)
                return std::format("init_extent must be finite and positive (got {})", init_extent);
            if (!std::isfinite(init_scaling) || init_scaling <= 0.0f)
                return std::format("init_scaling must be finite and positive (got {})", init_scaling);
            if (!std::isfinite(init_opacity) || init_opacity <= 0.0f || init_opacity >= 1.0f)
                return std::format("init_opacity must be finite and within (0, 1) (got {})", init_opacity);
            if (!std::isfinite(mask_opacity_penalty_power) || mask_opacity_penalty_power <= 0.0f)
                return std::format("mask_opacity_penalty_power must be finite and positive (got {})", mask_opacity_penalty_power);
            if (!std::isfinite(steps_scaler))
                return std::format("steps_scaler must be finite (got {})", steps_scaler);
            if (ppisp_warmup_steps < 0)
                return std::format("ppisp_warmup_steps must be nonnegative (got {})", ppisp_warmup_steps);
            if (debug_python && (debug_python_port <= 0 || debug_python_port > 65535))
                return std::format("debug_python_port must be within [1, 65535] (got {})", debug_python_port);

            const std::array nonnegative_fields{
                std::pair{"means_lr", means_lr},
                std::pair{"means_lr_end", means_lr_end},
                std::pair{"shs_lr", shs_lr},
                std::pair{"opacity_lr", opacity_lr},
                std::pair{"scaling_lr", scaling_lr},
                std::pair{"scaling_lr_end", scaling_lr_end},
                std::pair{"rotation_lr", rotation_lr},
                std::pair{"opacity_reg", opacity_reg},
                std::pair{"scale_reg", scale_reg},
                std::pair{"mask_opacity_penalty_weight", mask_opacity_penalty_weight},
                std::pair{"depth_loss_weight", depth_loss_weight},
                std::pair{"bilateral_grid_lr", bilateral_grid_lr},
                std::pair{"tv_loss_weight", tv_loss_weight},
                std::pair{"ppisp_lr", ppisp_lr},
                std::pair{"ppisp_reg_weight", ppisp_reg_weight},
                std::pair{"ppisp_controller_lr", ppisp_controller_lr},
                std::pair{"growth_grad_threshold", growth_grad_threshold},
                std::pair{"means_noise_weight", means_noise_weight},
                std::pair{"init_rho", init_rho},
            };
            for (const auto& [name, value] : nonnegative_fields) {
                if (auto error = invalid_nonnegative(value, name); !error.empty())
                    return error;
            }

            const std::array probability_fields{
                std::pair{"lambda_dssim", lambda_dssim},
                std::pair{"min_opacity", min_opacity},
                std::pair{"cropbox_lr_scale", cropbox_lr_scale},
                std::pair{"cropbox_loss_weight", cropbox_loss_weight},
                std::pair{"mask_threshold", mask_threshold},
                std::pair{"prune_opacity", prune_opacity},
                std::pair{"grow_fraction", grow_fraction},
                std::pair{"opacity_decay", opacity_decay},
                std::pair{"scale_decay", scale_decay},
                std::pair{"bounds_percentile", bounds_percentile},
                std::pair{"prune_ratio", prune_ratio},
            };
            for (const auto& [name, value] : probability_fields) {
                if (auto error = invalid_probability(value, name); !error.empty())
                    return error;
            }
            for (size_t i = 0; i < bg_color.size(); ++i) {
                if (auto error = invalid_probability(bg_color[i], std::format("bg_color[{}]", i)); !error.empty())
                    return error;
            }

            if (bilateral_grid_X <= 0 || bilateral_grid_Y <= 0 || bilateral_grid_W <= 0)
                return std::format("bilateral grid dimensions must be positive (got {}x{}x{})",
                                   bilateral_grid_X, bilateral_grid_Y, bilateral_grid_W);
            const uint64_t bilateral_xy = static_cast<uint64_t>(bilateral_grid_X) * bilateral_grid_Y;
            if (bilateral_xy > static_cast<uint64_t>(std::numeric_limits<int>::max()) /
                                   static_cast<uint64_t>(bilateral_grid_W))
                return std::format("bilateral grid dimensions are too large ({}x{}x{})",
                                   bilateral_grid_X, bilateral_grid_Y, bilateral_grid_W);
            if (gut && canonical_strategy_name(strategy) == kStrategyIGSPlus)
                return "GUT and igs+ strategy cannot be used together";
            if (ppisp_freeze_from_sidecar && !use_ppisp)
                return "PPISP sidecar freeze requires PPISP enabled";
            if (depth_loss_mode != "ssi" && depth_loss_mode != "ssi-disparity" && depth_loss_mode != "ssi-depth")
                return "depth_loss_mode must be 'ssi', 'ssi-disparity', or 'ssi-depth'";
            if (normal_loss_space != NormalLossSpace::Auto &&
                normal_loss_space != NormalLossSpace::CameraOpenCV &&
                normal_loss_space != NormalLossSpace::CameraOpenGL &&
                normal_loss_space != NormalLossSpace::World)
                return "normal_loss_space must be 'auto', 'camera-opencv', 'camera-opengl', or 'world'";
            return {};
        }

        std::string TrainingParameters::validate() const {
            if (auto error = optimization.validate(); !error.empty()) {
                return error;
            }
            if (auto error = dataset.validate(); !error.empty()) {
                return error;
            }
            const auto valid_port = [](const int port) { return port == -1 || (port > 0 && port <= 65535); };
            if (!valid_port(server.tcp_server_connection_port))
                return std::format("tcp_server_connection_port must be -1 or within [1, 65535] (got {})",
                                   server.tcp_server_connection_port);
            if (!valid_port(server.tcp_broadcast_connection_port))
                return std::format("tcp_broadcast_connection_port must be -1 or within [1, 65535] (got {})",
                                   server.tcp_broadcast_connection_port);
            if (render_path) {
                if (render_path->width <= 0 || render_path->height <= 0 ||
                    (render_path->width % 2) != 0 || (render_path->height % 2) != 0)
                    return std::format("render dimensions must be positive and even (got {}x{})",
                                       render_path->width, render_path->height);
                if (render_path->width > std::numeric_limits<int>::max() / render_path->height)
                    return std::format("render pixel count exceeds signed-int limits (got {}x{})",
                                       render_path->width, render_path->height);
                if (render_path->fps <= 0)
                    return std::format("render fps must be positive (got {})", render_path->fps);
                if (render_path->crf < 0 || render_path->crf > 51)
                    return std::format("render crf must be within [0, 51] (got {})", render_path->crf);
            }
            if (!std::isfinite(freeze_lr_scale) || freeze_lr_scale < 0.0f || freeze_lr_scale > 1.0f) {
                return std::format("freeze_lr_scale must be within [0, 1] (got {})", freeze_lr_scale);
            }
            if (!add_splat_paths.empty()) {
                if (resume_checkpoint.has_value()) {
                    return "--add-splat cannot be used together with --resume";
                }
                if (!add_splat_freeze.empty() && add_splat_freeze.size() != add_splat_paths.size()) {
                    return "--add-splat freeze metadata is inconsistent";
                }
                for (const auto& path : add_splat_paths) {
                    if (path.empty()) {
                        return "--add-splat path cannot be empty";
                    }
                    if (!std::filesystem::exists(path)) {
                        return std::format("Added splat does not exist: '{}'",
                                           lfs::core::path_to_utf8(path));
                    }
                }
            }
            if (optimization.ppisp_freeze_from_sidecar && !resume_checkpoint.has_value()) {
                if (optimization.ppisp_sidecar_path.empty()) {
                    return "PPISP sidecar freeze requires a sidecar path";
                }
                if (!std::filesystem::exists(optimization.ppisp_sidecar_path)) {
                    return std::format("PPISP sidecar does not exist: '{}'",
                                       lfs::core::path_to_utf8(optimization.ppisp_sidecar_path));
                }
            }
            return {};
        }

        std::string DatasetConfig::validate() const {
            if (resize_factor != -1 && resize_factor < 1)
                return std::format("resize_factor must be -1 or positive (got {})", resize_factor);
            if (test_every <= 0)
                return std::format("test_every must be positive (got {})", test_every);
            if (timelapse_every <= 0)
                return std::format("timelapse_every must be positive (got {})", timelapse_every);
            if (max_width < 0)
                return std::format("max_width must be nonnegative (got {})", max_width);
            if (min_track_length < 0)
                return std::format("min_track_length must be nonnegative (got {})", min_track_length);
            if (!std::isfinite(mask_threshold) || mask_threshold < 0.0f || mask_threshold > 1.0f)
                return std::format("dataset mask_threshold must be finite and within [0, 1] (got {})", mask_threshold);
            if (!std::isfinite(loading_params.min_cpu_free_memory_ratio) ||
                loading_params.min_cpu_free_memory_ratio < 0.0f ||
                loading_params.min_cpu_free_memory_ratio > 1.0f)
                return std::format("min_cpu_free_memory_ratio must be finite and within [0, 1] (got {})",
                                   loading_params.min_cpu_free_memory_ratio);
            if (!std::isfinite(loading_params.min_cpu_free_GB) || loading_params.min_cpu_free_GB < 0.0f)
                return std::format("min_cpu_free_GB must be finite and nonnegative (got {})",
                                   loading_params.min_cpu_free_GB);
            if (loading_params.print_status_freq_num <= 0)
                return std::format("print_status_freq_num must be positive (got {})",
                                   loading_params.print_status_freq_num);
            return {};
        }

        OptimizationParameters OptimizationParameters::mcmc_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = std::string(kStrategyMCMC);
            return p;
        }

        OptimizationParameters OptimizationParameters::mrnf_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = std::string(kStrategyMRNF);
            p.refine_every = 200;
            p.start_refine = 0;
            p.stop_refine = 28'500;
            p.max_cap = 5'000'000;
            p.min_opacity = 1.0f / 255.0f;
            p.means_lr = 2e-5f;
            p.means_lr_end = 2e-7f;
            p.opacity_lr = 0.012f;
            p.scaling_lr = 7e-3f;
            p.scaling_lr_end = 5e-3f;
            p.rotation_lr = 2e-3f;
            p.shs_lr = 2e-3f;
            p.lambda_dssim = 0.2f;
            p.opacity_reg = 0.0f;
            p.scale_reg = 0.0f;
            p.use_error_map = true;
            p.use_edge_map = true;
            return p;
        }

        OptimizationParameters OptimizationParameters::igs_plus_defaults() {
            auto p = OptimizationParameters{};
            p.strategy = "igs+";
            p.means_lr = 0.000016f;
            p.shs_lr = 0.005f;
            p.scaling_lr = 0.02f;
            p.rotation_lr = 0.0015f;
            p.stop_refine = 15'000;
            p.refine_every = 500;
            p.opacity_reg = 0.0f;
            p.scale_reg = 0.0f;
            p.init_opacity = 0.1f;
            p.init_scaling = 0.1f;
            p.max_cap = 4'000'000;
            p.tv_loss_weight = 5.0f;
            return p;
        }

        OptimizationParameters OptimizationParameters::defaults_for_strategy(const std::string_view strategy) {
            const auto canonical_strategy = canonical_strategy_name(strategy);
            if (canonical_strategy == kStrategyMCMC)
                return mcmc_defaults();
            if (canonical_strategy == kStrategyIGSPlus)
                return igs_plus_defaults();
            return mrnf_defaults();
        }

        OptimizationParameters OptimizationParameters::from_json(const nlohmann::json& json) {
            OptimizationParameters params;
            if (json.contains("strategy")) {
                const auto strategy = json.at("strategy").get<std::string>();
                if (const auto canonical = canonical_strategy_name(strategy); !canonical.empty()) {
                    params = defaults_for_strategy(canonical);
                } else {
                    LOG_WARN("Invalid strategy '{}' in JSON, using default", strategy);
                }
            }
            read_registered_optimization_properties(json, params);

            // Residue not represented by scalar registry properties.
            if (json.contains("eval_steps")) {
                params.eval_steps.clear();
                for (const auto& step : json.at("eval_steps"))
                    params.eval_steps.push_back(step.get<size_t>());
            }
            if (json.contains("save_steps")) {
                params.save_steps.clear();
                for (const auto& step : json.at("save_steps"))
                    params.save_steps.push_back(step.get<size_t>());
            }
            if (json.contains("enable_save_eval_images"))
                params.enable_save_eval_images = json.at("enable_save_eval_images");
            if (json.contains("ppisp_sidecar_path")) {
                params.ppisp_sidecar_path =
                    utf8_to_path(json.at("ppisp_sidecar_path").get<std::string>());
            }
            if (json.contains("bg_color") && json.at("bg_color").is_array() &&
                json.at("bg_color").size() == 3) {
                params.bg_color = {
                    json.at("bg_color")[0],
                    json.at("bg_color")[1],
                    json.at("bg_color")[2]};
            }
            if (json.contains("bg_image_path")) {
                params.bg_image_path =
                    utf8_to_path(json.at("bg_image_path").get<std::string>());
            }

            if (json.contains("depth_loss_mode") &&
                (params.depth_loss_mode == "pearson" ||
                 params.depth_loss_mode == "adaptive-warped-l1")) {
                LOG_WARN(
                    "Migrating legacy depth loss mode '{}' to 'ssi'; the current depth "
                    "pipeline auto-detects whether the prior stores depth or disparity",
                    params.depth_loss_mode);
                params.depth_loss_mode = "ssi";
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
                auto params = OptimizationParameters::from_json(opt_json);
                if (auto error = params.validate(); !error.empty())
                    return std::unexpected("Invalid optimization parameters: " + error);
                return params;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error parsing optimization parameters: {}", e.what()));
            }
        }

        std::expected<void, std::string> save_training_parameters_to_json(
            const TrainingParameters& params,
            const std::filesystem::path& output_path) {
            try {
                auto opt_copy = params.optimization;
                opt_copy.remove_step_scaling();

                nlohmann::json json;
                json["dataset"] = params.dataset.to_json();
                json["server"] = params.server.to_json();
                json["optimization"] = opt_copy.to_json();

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
            if (j.contains("print_cache_status")) {
                params.print_cache_status = j["print_cache_status"];
            }
            if (j.contains("print_status_freq_num")) {
                params.print_status_freq_num = j["print_status_freq_num"];
            }
            if (j.contains("use_16bit_color")) {
                params.use_16bit_color = j["use_16bit_color"];
            } else if (j.contains("use_8bit_color")) {
                params.use_16bit_color = !j["use_8bit_color"].get<bool>();
            }

            return params;
        }

        nlohmann::json LoadingParams::to_json() const {
            nlohmann::json loading_json;
            loading_json["use_cpu_memory"] = use_cpu_memory;
            loading_json["min_cpu_free_memory_ratio"] = min_cpu_free_memory_ratio;
            loading_json["min_cpu_free_GB"] = min_cpu_free_GB;
            loading_json["print_cache_status"] = print_cache_status;
            loading_json["print_status_freq_num"] = print_status_freq_num;
            loading_json["use_16bit_color"] = use_16bit_color;

            return loading_json;
        }

        nlohmann::json ServerConfig::to_json() const {
            nlohmann::json json;
            json["tcp_server_connection_port"] = tcp_server_connection_port;
            json["tcp_broadcast_connection_port"] = tcp_broadcast_connection_port;
            json["tcp_connection"] = tcp_connection;

            return json;
        }

        ServerConfig ServerConfig::from_json(const nlohmann::json& j) {
            ServerConfig server;

            if (j.contains("tcp_server_connection_port")) {
                server.tcp_server_connection_port = j["tcp_server_connection_port"].get<int>();
            }
            if (j.contains("tcp_broadcast_connection_port")) {
                server.tcp_broadcast_connection_port = j["tcp_broadcast_connection_port"].get<int>();
            }
            if (j.contains("tcp_connection")) {
                server.tcp_connection = j["tcp_connection"].get<bool>();
            }

            return server;
        }

        nlohmann::json DatasetConfig::to_json() const {
            nlohmann::json json;

            json["data_path"] = path_to_utf8(data_path);
            json["output_folder"] = path_to_utf8(output_path);
            json["images"] = images;
            json["resize_factor"] = resize_factor;
            json["test_every"] = test_every;
            json["max_width"] = max_width;
            json["min_track_length"] = min_track_length;
            json["loading_params"] = loading_params.to_json();
            json["invert_masks"] = invert_masks;
            json["mask_threshold"] = mask_threshold;
            if (!output_name.empty())
                json["output_name"] = output_name;

            return json;
        }

        DatasetConfig DatasetConfig::from_json(const nlohmann::json& j) {
            DatasetConfig dataset;

            // Use utf8_to_path for proper Unicode handling since JSON is UTF-8 encoded
            dataset.data_path = utf8_to_path(j["data_path"].get<std::string>());
            dataset.images = j["images"].get<std::string>();
            dataset.resize_factor = j["resize_factor"].get<int>();
            dataset.max_width = j["max_width"].get<int>();
            if (j.contains("min_track_length")) {
                dataset.min_track_length = j["min_track_length"].get<int>();
            }
            dataset.test_every = j["test_every"].get<int>();
            dataset.output_path = utf8_to_path(j["output_folder"].get<std::string>());

            if (j.contains("output_name")) {
                dataset.output_name = j["output_name"].get<std::string>();
            }
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
