/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/logger.hpp"
#include "core/optimization_properties.hpp"
#include "core/path_utils.hpp"
#include "core/project_path.hpp"
#include "core/property_registry.hpp"
#include <any>
#include <cassert>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <ctime>

#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace lfs::core {
    namespace param {
        std::string_view output_format_extension(
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

        namespace {
            using prop::PropertyMeta;
            using prop::PropertyObjectRef;
            using prop::PropertyRegistry;
            using prop::PropType;

            [[nodiscard]] float mrnf_shs_lr_for_capacity(const int max_cap) {
                constexpr double kReferenceCapacity = 1'000'000.0;
                const double cap = std::max(static_cast<double>(max_cap), kReferenceCapacity);
                return static_cast<float>(0.005 * std::sqrt(kReferenceCapacity / cap));
            }

            [[nodiscard]] float mrnf_grow_fraction_for_capacity(const int max_cap) {
                constexpr double kReferenceCapacity = 1'000'000.0;
                constexpr double kSaturationCapacity = 5'000'000.0;
                constexpr double kAtReference = 0.0758;
                constexpr double kAtSaturation = 0.12;
                const double cap = std::clamp(static_cast<double>(max_cap),
                                              kReferenceCapacity, kSaturationCapacity);
                const double t = std::log(cap / kReferenceCapacity) /
                                 std::log(kSaturationCapacity / kReferenceCapacity);
                return static_cast<float>(kAtReference + t * (kAtSaturation - kAtReference));
            }

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

            [[nodiscard]] std::optional<std::string> validate_registered_optimization_enums(
                const nlohmann::json& json) {
                if (!json.is_object())
                    return std::nullopt;

                const auto group = optimization_property_snapshot();
                for (const auto& meta : group.properties) {
                    if (meta.type != PropType::Enum)
                        continue;

                    const std::string key(optimization_json_key(meta));
                    if (!json.contains(key))
                        continue;

                    std::string accepted_values;
                    for (const auto& item : meta.enum_items) {
                        if (!accepted_values.empty())
                            accepted_values += ", ";
                        accepted_values += item.wire_value.empty() ? item.identifier : item.wire_value;
                    }

                    const auto& value = json.at(key);
                    if (!value.is_string()) {
                        return std::format(
                            "Invalid value {} for optimization field '{}'; accepted values: {}",
                            value.dump(), meta.id, accepted_values);
                    }

                    const auto wire_value = value.get<std::string>();
                    const auto item = std::ranges::find_if(meta.enum_items, [&wire_value](const auto& candidate) {
                        const auto& candidate_wire = candidate.wire_value.empty()
                                                         ? candidate.identifier
                                                         : candidate.wire_value;
                        return candidate_wire == wire_value;
                    });
                    if (item == meta.enum_items.end()) {
                        return std::format(
                            "Invalid value '{}' for optimization field '{}'; accepted values: {}",
                            wire_value, meta.id, accepted_values);
                    }
                }
                return std::nullopt;
            }

            [[nodiscard]] bool is_json_integer(const nlohmann::json& value) {
                return value.is_number_integer() || value.is_number_unsigned();
            }

            [[nodiscard]] bool is_json_int32(const nlohmann::json& value) {
                return is_json_integer(value) &&
                       value.get<long double>() >= std::numeric_limits<int>::min() &&
                       value.get<long double>() <= std::numeric_limits<int>::max();
            }

            [[nodiscard]] bool is_json_size_t(const nlohmann::json& value) {
                return is_json_integer(value) && value.get<long double>() >= 0.0L &&
                       value.get<long double>() <= std::numeric_limits<size_t>::max();
            }

            [[nodiscard]] std::string_view optimization_json_type_name(const PropType type) {
                switch (type) {
                case PropType::Bool: return "boolean";
                case PropType::Int: return "integer";
                case PropType::Float: return "number";
                case PropType::String:
                case PropType::Enum: return "string";
                case PropType::SizeT: return "non-negative integer";
                case PropType::Vec3:
                case PropType::Color3: return "array of 3 numbers";
                case PropType::IntVector: return "array of integers";
                case PropType::FloatVector: return "array of numbers";
                default: return "valid JSON value";
                }
            }

            [[nodiscard]] bool optimization_json_type_matches(
                const PropType type,
                const nlohmann::json& value) {
                switch (type) {
                case PropType::Bool: return value.is_boolean();
                case PropType::Int: return is_json_int32(value);
                case PropType::Float: return value.is_number();
                case PropType::String:
                case PropType::Enum: return value.is_string();
                case PropType::SizeT: return is_json_size_t(value);
                case PropType::Vec3:
                case PropType::Color3:
                    return value.is_array() && value.size() == 3 &&
                           std::ranges::all_of(value, [](const auto& component) { return component.is_number(); });
                case PropType::IntVector:
                    return value.is_array() && std::ranges::all_of(value, is_json_integer);
                case PropType::FloatVector:
                    return value.is_array() && std::ranges::all_of(value, [](const auto& item) { return item.is_number(); });
                default: return true;
                }
            }

            [[nodiscard]] std::optional<std::string> validate_optimization_json_types(
                const nlohmann::json& json) {
                if (!json.is_object())
                    return std::nullopt;
                const auto check = [&json](const std::string_view key, const std::string_view expected, auto predicate)
                    -> std::optional<std::string> {
                    if (json.contains(key) && !predicate(json.at(key)))
                        return std::format("Invalid type for optimization.{}; expected {}", key, expected);
                    return std::nullopt;
                };
                if (auto error = check("strategy", "string", [](const auto& value) { return value.is_string(); }))
                    return error;
                if (auto error = check("enable_save_eval_images", "boolean", [](const auto& value) { return value.is_boolean(); }))
                    return error;
                if (auto error = check("ppisp_sidecar_path", "string", [](const auto& value) { return value.is_string(); }))
                    return error;
                if (auto error = check("bg_image_path", "string", [](const auto& value) { return value.is_string(); }))
                    return error;

                const auto group = optimization_property_snapshot();
                for (const auto& meta : group.properties) {
                    const std::string key(optimization_json_key(meta));
                    if (!json.contains(key))
                        continue;
                    const auto& value = json.at(key);
                    if (!optimization_json_type_matches(meta.type, value)) {
                        return std::format(
                            "Invalid type for optimization.{}; expected {}",
                            meta.id, optimization_json_type_name(meta.type));
                    }
                }

                for (const auto& key : {"eval_steps", "save_steps"}) {
                    if (!json.contains(key))
                        continue;
                    const auto& value = json.at(key);
                    if (!value.is_array() || !std::ranges::all_of(value, is_json_size_t)) {
                        return std::format(
                            "Invalid type for optimization.{}; expected array of non-negative integers",
                            key);
                    }
                }
                if (json.contains("bg_color")) {
                    const auto& value = json.at("bg_color");
                    if (!value.is_array() || value.size() != 3 ||
                        !std::ranges::all_of(value, [](const auto& component) { return component.is_number(); })) {
                        return "Invalid type for optimization.bg_color; expected array of 3 numbers";
                    }
                }
                return std::nullopt;
            }

            [[nodiscard]] std::optional<std::string> validate_dataset_json_types(const nlohmann::json& json) {
                if (!json.is_object())
                    return std::nullopt;
                const auto check = [&json](const std::string_view key, const std::string_view expected, auto predicate)
                    -> std::optional<std::string> {
                    if (json.contains(key) && !predicate(json.at(key)))
                        return std::format("Invalid type for dataset.{}; expected {}", key, expected);
                    return std::nullopt;
                };
                if (auto error = check("data_path", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("output_folder", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("output_path", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("images", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("resize_factor", "integer", is_json_int32))
                    return error;
                if (auto error = check("max_width", "integer", is_json_int32))
                    return error;
                if (auto error = check("min_track_length", "integer", is_json_int32))
                    return error;
                if (auto error = check("test_every", "integer", is_json_int32))
                    return error;
                if (auto error = check("timelapse_images", "array of strings", [](const auto& v) { return v.is_array() && std::ranges::all_of(v, [](const auto& item) { return item.is_string(); }); }))
                    return error;
                if (auto error = check("timelapse_every", "integer", is_json_int32))
                    return error;
                if (auto error = check("output_name", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("invert_masks", "boolean", [](const auto& v) { return v.is_boolean(); }))
                    return error;
                if (auto error = check("mask_threshold", "number", [](const auto& v) { return v.is_number(); }))
                    return error;
                if (auto error = check("centralize_dataset", "string", [](const auto& v) { return v.is_string(); }))
                    return error;
                if (auto error = check("loading_params", "object", [](const auto& v) { return v.is_object(); }))
                    return error;
                if (json.contains("loading_params")) {
                    const auto& loading = json.at("loading_params");
                    for (const auto key : {"use_cpu_memory", "print_cache_status", "use_16bit_color", "use_8bit_color"}) {
                        if (loading.contains(key) && !loading.at(key).is_boolean())
                            return std::format("Invalid type for dataset.loading_params.{}; expected boolean", key);
                    }
                    for (const auto key : {"min_cpu_free_memory_ratio", "min_cpu_free_GB"}) {
                        if (loading.contains(key) && !loading.at(key).is_number())
                            return std::format("Invalid type for dataset.loading_params.{}; expected number", key);
                    }
                    if (loading.contains("print_status_freq_num") && !is_json_int32(loading.at("print_status_freq_num")))
                        return "Invalid type for dataset.loading_params.print_status_freq_num; expected integer";
                }
                return std::nullopt;
            }

            [[nodiscard]] std::optional<std::string> validate_server_json_types(const nlohmann::json& json) {
                if (!json.is_object())
                    return std::nullopt;
                if (json.contains("tcp_server_connection_port") &&
                    !is_json_int32(json.at("tcp_server_connection_port")))
                    return "Invalid type for server.tcp_server_connection_port; expected integer";
                if (json.contains("tcp_broadcast_connection_port") &&
                    !is_json_int32(json.at("tcp_broadcast_connection_port")))
                    return "Invalid type for server.tcp_broadcast_connection_port; expected integer";
                if (json.contains("tcp_connection") && !json.at("tcp_connection").is_boolean())
                    return "Invalid type for server.tcp_connection; expected boolean";
                return std::nullopt;
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
                OptimizationParameters& params,
                const bool skip_missing = false) {
                const auto group = optimization_property_snapshot();
                auto ref = PropertyObjectRef::cpp(&params);

                for (const auto& meta : group.properties) {
                    const std::string key(optimization_json_key(meta));
                    if (!json.contains(key) && (skip_missing || !meta.json_required))
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
                        if (item == meta.enum_items.end()) {
                            LOG_WARN("Unknown enum value '{}' for optimization field '{}'; keeping current value",
                                     wire_value, meta.id);
                            break;
                        }
                        meta.setter(ref, std::any(item->value));
                        break;
                    }
                    default:
                        throw std::runtime_error("Unsupported registered optimization JSON type: " + meta.id);
                    }
                }
            }

            nlohmann::json parse_overlay_object(const std::string_view text) {
                if (text.empty())
                    return nlohmann::json::object();
                auto parsed = nlohmann::json::parse(text);
                if (!parsed.is_object())
                    return nlohmann::json::object();
                return parsed;
            }

            bool overlay_has_key(const std::string_view text, const std::string_view key) {
                if (text.empty() || key.empty())
                    return false;
                return parse_overlay_object(text).contains(key);
            }

            // Older builds keep both keys as unknown data while rewriting steps_scaler,
            // so the image share is trusted only next to the total it was written with.
            std::optional<float> stored_image_count_scaler(const nlohmann::json& json, const float steps_scaler) {
                if (!json.contains("image_count_scaler") || !json.contains("image_count_scaler_total"))
                    return std::nullopt;
                if (json.at("image_count_scaler_total").get<float>() != steps_scaler)
                    return std::nullopt;
                return json.at("image_count_scaler").get<float>();
            }

            void apply_optimization_json_overlay(
                OptimizationParameters& params,
                const nlohmann::json& json,
                const bool skip_missing = false) {
                std::optional<RasterBackendId> backend;
                if (json.contains("raster_backend")) {
                    const auto name = json.at("raster_backend").get<std::string>();
                    backend = parse_training_backend(name);
                    if (!backend)
                        throw std::invalid_argument("Unknown training raster_backend: " + name);
                    if (json.contains("gut")) {
                        const bool gut = json.at("gut").get<bool>();
                        const auto boolean_backend = gut
                                                         ? RasterBackendId::ThreeDGUT
                                                         : RasterBackendId::ThreeDGS;
                        if (*backend != boolean_backend) {
                            throw std::invalid_argument(std::format(
                                "Conflicting raster_backend '{}' and gut={}; set both consistently or specify only one",
                                name, gut));
                        }
                    }
                }
                if (json.contains("strategy")) {
                    const auto strategy = json.at("strategy").get<std::string>();
                    if (const auto canonical = canonical_strategy_name(strategy); !canonical.empty()) {
                        params.strategy = std::string(canonical);
                    } else {
                        LOG_WARN("Invalid strategy '{}' in JSON, using default", strategy);
                    }
                }
                if (const auto removed = json.find("background_improvements");
                    removed != json.end() && removed->is_boolean() && removed->get<bool>()) {
                    LOG_WARN("Ignoring background_improvements: the option was removed and MRNF trains with its default profile");
                }
                if (const auto removed = json.find("hard_clip_stop_iter");
                    removed != json.end() && removed->is_number_integer() && removed->get<int>() != 0) {
                    LOG_WARN("Ignoring hard_clip_stop_iter: MRNF no longer hard-clips splats by screen share");
                }
                if (const auto removed = json.find("oversize_split_fraction");
                    removed != json.end() && removed->is_number() && removed->get<float>() > 0.0f) {
                    LOG_WARN("Ignoring oversize_split_fraction: MRNF no longer reserves growth for oversized splats");
                }
                read_registered_optimization_properties(json, params, skip_missing);
                if (const auto image_count_scaler = stored_image_count_scaler(json, params.steps_scaler))
                    params.image_count_scaler = *image_count_scaler;

                if (backend)
                    params.set_raster_backend(*backend);

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
            }

            void apply_dataset_json_overlay(DatasetConfig& dataset, const nlohmann::json& j) {
                if (j.contains("data_path"))
                    dataset.data_path = utf8_to_path(j["data_path"].get<std::string>());
                if (j.contains("output_folder"))
                    dataset.output_path = utf8_to_path(j["output_folder"].get<std::string>());
                if (j.contains("output_path"))
                    dataset.output_path = utf8_to_path(j["output_path"].get<std::string>());
                if (j.contains("images"))
                    dataset.images = j["images"].get<std::string>();
                if (j.contains("resize_factor"))
                    dataset.resize_factor = j["resize_factor"].get<int>();
                if (j.contains("max_width"))
                    dataset.max_width = j["max_width"].get<int>();
                if (j.contains("min_track_length"))
                    dataset.min_track_length = j["min_track_length"].get<int>();
                if (j.contains("test_every"))
                    dataset.test_every = j["test_every"].get<int>();
                if (j.contains("timelapse_images"))
                    dataset.timelapse_images = j["timelapse_images"].get<std::vector<std::string>>();
                if (j.contains("timelapse_every"))
                    dataset.timelapse_every = j["timelapse_every"].get<int>();
                if (j.contains("output_name"))
                    dataset.output_name = j["output_name"].get<std::string>();
                if (j.contains("invert_masks"))
                    dataset.invert_masks = j["invert_masks"].get<bool>();
                if (j.contains("mask_threshold"))
                    dataset.mask_threshold = j["mask_threshold"].get<float>();
                if (j.contains("centralize_dataset"))
                    dataset.centralize_dataset = j["centralize_dataset"].get<std::string>();
                if (j.contains("loading_params") && j["loading_params"].is_object()) {
                    const auto& loading = j["loading_params"];
                    auto merged = dataset.loading_params.to_json();
                    for (auto it = loading.begin(); it != loading.end(); ++it)
                        merged[it.key()] = it.value();
                    dataset.loading_params = LoadingParams::from_json(merged);
                }
            }

            [[nodiscard]] lfs::Error config_import_error(std::string detail, const std::filesystem::path& path) {
                lfs::SmallFields fields;
                fields.add("path", path_to_utf8(path));
                return lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::InvalidArgument,
                    .domain = lfs::ErrorDomain::IO,
                    .user_message = "The config file could not be imported.",
                    .detail = std::move(detail),
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .fields = std::move(fields),
                });
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
            morton_reorder_interval = apply(morton_reorder_interval);
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

        bool OptimizationParameters::normal_supervision_active(const int iter) const {
            if (!use_normal_loss)
                return false;
            const float total_f = static_cast<float>(std::max(1, resolved_total_iterations()));
            const int start_iter = static_cast<int>(normal_start_fraction * total_f);
            if (iter < start_iter)
                return false;
            // 1.0 keeps supervision on through the inclusive last iteration.
            return normal_end_fraction >= 1.0f ||
                   static_cast<float>(iter) < normal_end_fraction * total_f;
        }

        float OptimizationParameters::scale_reg_at(const int iter) const {
            if (scale_reg_decay_power < 0.0f)
                return scale_reg;
            const float p = std::max(scale_reg_decay_power, 0.0f);
            const float t = std::clamp(static_cast<float>(iter) /
                                           static_cast<float>(std::max<size_t>(iterations, 1)),
                                       0.0f, 1.0f);
            return scale_reg * (p + 1.0f) * std::pow(1.0f - t, p);
        }

        void OptimizationParameters::resolve_mrnf_capacity_defaults() {
            if (canonical_strategy_name(strategy) != kStrategyMRNF)
                return;
            if (grow_fraction < 0.0f)
                grow_fraction = mrnf_grow_fraction_for_capacity(max_cap);
            if (shs_lr < 0.0f)
                shs_lr = mrnf_shs_lr_for_capacity(max_cap);
        }

        int OptimizationParameters::resolved_ppisp_controller_activation_step(const int total_iterations) const {
            if (ppisp_controller_activation_step >= 0)
                return ppisp_controller_activation_step;

            const float clamped_scaler = std::max(steps_scaler, 1.0f);
            const int tail_iters = static_cast<int>(std::lround(5000.0f * clamped_scaler));
            return std::max(0, total_iterations - tail_iters);
        }

        bool is_eval_mask_box(const std::string_view spec) {
            return spec.starts_with("bbox:");
        }

        bool is_eval_mask_cropbox(const std::string_view spec) {
            return spec == "cropbox";
        }

        bool is_eval_mask_folder(const std::string_view spec) {
            return spec.starts_with("masks:");
        }

        std::string_view eval_mask_folder(const std::string_view spec) {
            assert(is_eval_mask_folder(spec));
            return spec.substr(6);
        }

        namespace {
            template <size_t N>
            std::optional<std::array<float, N>> parse_float_list(std::string_view rest) {
                std::array<float, N> values{};
                for (size_t i = 0; i < N; ++i) {
                    const auto comma = rest.find(',');
                    if ((comma == std::string_view::npos) != (i + 1 == N))
                        return std::nullopt;
                    auto token = rest.substr(0, comma);
                    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.front())))
                        token.remove_prefix(1);
                    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back())))
                        token.remove_suffix(1);
                    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), values[i]);
                    if (token.empty() || error != std::errc{} || end != token.data() + token.size() ||
                        !std::isfinite(values[i]))
                        return std::nullopt;
                    if (comma != std::string_view::npos)
                        rest.remove_prefix(comma + 1);
                }
                return values;
            }
        } // namespace

        std::optional<std::array<float, 6>> parse_eval_mask_box(const std::string_view spec) {
            if (!is_eval_mask_box(spec))
                return std::nullopt;
            const auto box = parse_float_list<6>(spec.substr(5));
            if (!box)
                return std::nullopt;
            for (size_t axis = 0; axis < 3; ++axis) {
                if (!((*box)[axis] < (*box)[axis + 3]))
                    return std::nullopt;
            }
            return box;
        }

        bool is_eval_mask_depth(const std::string_view spec) {
            return spec.starts_with("depth:");
        }

        bool is_eval_mask_points(const std::string_view spec) {
            return spec == "points" || spec.starts_with("points:");
        }

        std::optional<std::string_view> eval_mask_splat_file(const std::string_view spec) {
            if (!spec.starts_with("splat:") || spec.size() == 6)
                return std::nullopt;
            return spec.substr(6);
        }

        std::optional<std::string_view> eval_mask_points_file(const std::string_view spec) {
            if (!spec.starts_with("points:") || parse_float_list<2>(spec.substr(7)))
                return std::nullopt;
            return spec.substr(7);
        }

        std::optional<std::array<int, 2>> parse_eval_mask_points(const std::string_view spec) {
            if (!is_eval_mask_points(spec))
                return std::nullopt;
            if (spec == "points" || eval_mask_points_file(spec))
                return std::array<int, 2>{2, 3};
            const auto values = parse_float_list<2>(spec.substr(7));
            if (!values)
                return std::nullopt;
            const std::array<int, 2> limits{32, 64};
            std::array<int, 2> result{};
            for (size_t i = 0; i < 2; ++i) {
                const float value = (*values)[i];
                if (value != std::floor(value) || value < 0.0f || value > static_cast<float>(limits[i]))
                    return std::nullopt;
                result[i] = static_cast<int>(value);
            }
            return result;
        }

        std::optional<std::array<float, 2>> parse_eval_mask_depth(const std::string_view spec) {
            if (!is_eval_mask_depth(spec))
                return std::nullopt;
            const auto range = parse_float_list<2>(spec.substr(6));
            if (!range || !((*range)[0] >= 0.0f) || !((*range)[0] < (*range)[1]))
                return std::nullopt;
            return range;
        }

        std::string normalize_eval_mask(const std::string_view spec) {
            if (spec.empty())
                return {};
            if (const auto box = parse_eval_mask_box(spec))
                return std::format("bbox:{},{},{},{},{},{}", (*box)[0], (*box)[1], (*box)[2], (*box)[3], (*box)[4], (*box)[5]);
            if (const auto range = parse_eval_mask_depth(spec))
                return std::format("depth:{},{}", (*range)[0], (*range)[1]);
            const auto normalize_path = [](const std::string_view path) {
                std::error_code error;
                auto absolute = std::filesystem::absolute(utf8_to_path(std::string(path)), error);
                if (error)
                    return std::string(path);
                auto canonical = std::filesystem::weakly_canonical(absolute, error);
                return path_to_utf8((error ? absolute : canonical).lexically_normal());
            };
            if (const auto file = eval_mask_splat_file(spec))
                return "splat:" + normalize_path(*file);
            if (const auto file = eval_mask_points_file(spec))
                return "points:" + normalize_path(*file);
            if (const auto points = parse_eval_mask_points(spec))
                return std::format("points:{},{}", (*points)[0], (*points)[1]);
            if (is_eval_mask_box(spec) || is_eval_mask_depth(spec) || is_eval_mask_points(spec) ||
                is_eval_mask_cropbox(spec))
                return std::string(spec);
            if (is_eval_mask_folder(spec))
                return "masks:" + normalize_path(eval_mask_folder(spec));
            return normalize_path(spec);
        }

        nlohmann::json OptimizationParameters::to_json() const {
            nlohmann::json opt_json;
            write_registered_optimization_properties(opt_json, *this);
            opt_json["raster_backend"] = training_backend_descriptor(raster_backend()).wire_name;

            const auto canonical_strategy = canonical_strategy_name(strategy);
            opt_json["strategy"] = canonical_strategy.empty() ? strategy : std::string(canonical_strategy);

            // Residue not represented by scalar registry properties.
            opt_json["image_count_scaler"] = image_count_scaler;
            opt_json["image_count_scaler_total"] = steps_scaler;
            opt_json["eval_steps"] = eval_steps;
            opt_json["save_steps"] = save_steps;
            opt_json["enable_save_eval_images"] = enable_save_eval_images;
            opt_json["ppisp_sidecar_path"] = path_to_utf8(ppisp_sidecar_path);
            opt_json["bg_color"] = {bg_color[0], bg_color[1], bg_color[2]};
            if (!bg_image_path.empty())
                opt_json["bg_image_path"] = path_to_utf8(bg_image_path);
            if (!eval_mask.empty())
                opt_json["eval_mask"] = normalize_eval_mask(eval_mask);

            return opt_json;
        }

        TrainingBackendConflictDescriptor training_backend_conflict_descriptor(
            const TrainingBackendConflict conflict) {
            constexpr std::string_view backend = "3DGUT";
            constexpr std::string_view fallback_backend = "3DGS";
            switch (conflict) {
            case TrainingBackendConflict::IGSPlus:
                return {"igs_plus", backend, "IGS+", fallback_backend};
            case TrainingBackendConflict::MipFilter:
                return {"mip_filter", backend, "Mip Filter", fallback_backend};
            case TrainingBackendConflict::DepthSupervision:
                return {"depth_supervision", backend, "Depth Loss", fallback_backend};
            case TrainingBackendConflict::NormalSupervision:
                return {"normal_supervision", backend, "Normal Loss", fallback_backend};
            case TrainingBackendConflict::None:
                return {};
            }
            return {};
        }

        TrainingBackendConflict OptimizationParameters::backend_conflict() const {
            const auto& capabilities = training_backend_descriptor(raster_backend()).capabilities;
            if (is_training_feature_unsupported(capabilities.igs_plus) &&
                canonical_strategy_name(strategy) == kStrategyIGSPlus)
                return TrainingBackendConflict::IGSPlus;
            if (is_training_feature_unsupported(capabilities.mip_filter) && mip_filter)
                return TrainingBackendConflict::MipFilter;
            if (is_training_feature_unsupported(capabilities.depth_supervision) && use_depth_loss)
                return TrainingBackendConflict::DepthSupervision;
            if (is_training_feature_unsupported(capabilities.normal_supervision) && use_normal_loss)
                return TrainingBackendConflict::NormalSupervision;
            return TrainingBackendConflict::None;
        }

        std::string OptimizationParameters::backend_conflict_message() const {
            const auto descriptor = training_backend_conflict_descriptor(backend_conflict());
            if (descriptor.id.empty()) {
                return {};
            }
            return std::format(
                "{} cannot be used with {}. Change this setting or select {}.",
                descriptor.backend_name,
                descriptor.feature_name,
                descriptor.fallback_backend_name);
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
            if (eval_mask_invert && eval_mask.empty())
                return "eval_mask_invert requires eval_mask";
            if (!(eval_mask_opacity > 0.0f && eval_mask_opacity <= 1.0f))
                return std::format("eval_mask_opacity must be greater than 0 and at most 1 (got {})", eval_mask_opacity);
            if (!eval_mask.empty() && !enable_eval)
                return "eval_mask requires evaluation to be enabled";
            // The mask file is checked where it is read, so settings stored in a project stay valid
            // when the file moves.
            if (is_eval_mask_box(eval_mask)) {
                if (!parse_eval_mask_box(eval_mask))
                    return std::format("eval_mask box must be bbox:x0,y0,z0,x1,y1,z1 with each minimum below its maximum (got '{}')", eval_mask);
            } else if (const auto file = eval_mask_splat_file(eval_mask)) {
                if (!utf8_to_path(std::string(*file)).is_absolute())
                    return std::format("eval_mask splat file must be an absolute path (got '{}')", eval_mask);
            } else if (const auto file = eval_mask_points_file(eval_mask)) {
                if (!utf8_to_path(std::string(*file)).is_absolute())
                    return std::format("eval_mask points file must be an absolute path (got '{}')", eval_mask);
            } else if (is_eval_mask_points(eval_mask)) {
                if (!parse_eval_mask_points(eval_mask))
                    return std::format("eval_mask points must be points or points:radius,close with whole radius 0..32 and close 0..64 (got '{}')", eval_mask);
            } else if (is_eval_mask_depth(eval_mask)) {
                if (!parse_eval_mask_depth(eval_mask))
                    return std::format("eval_mask depth range must be depth:near,far with 0 <= near < far (got '{}')", eval_mask);
            } else if (is_eval_mask_folder(eval_mask)) {
                if (!utf8_to_path(std::string(eval_mask_folder(eval_mask))).is_absolute())
                    return std::format("eval_mask folder must be an absolute path: {}", eval_mask);
            } else if (!eval_mask.empty() && !is_eval_mask_cropbox(eval_mask) && !utf8_to_path(eval_mask).is_absolute()) {
                return "eval_mask must be an absolute path";
            }
            if (iterations == 0 || iterations > MAX_ITERATION_VALUE)
                return std::format("iterations must be within [1, {}] (got {})", MAX_ITERATION_VALUE, iterations);
            if (refine_every == 0 || refine_every > MAX_ITERATION_VALUE)
                return std::format("refine_every must be within [1, {}] (got {})", MAX_ITERATION_VALUE, refine_every);
            if (morton_reorder_interval > MAX_ITERATION_VALUE)
                return std::format("morton_reorder_interval must be within [0, {}] (got {})",
                                   MAX_ITERATION_VALUE, morton_reorder_interval);
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
            if (!std::isfinite(image_count_scaler) || image_count_scaler <= 0.f)
                return std::format("image_count_scaler must be finite and positive (got {})", image_count_scaler);
            if (!std::isfinite(steps_scaler))
                return std::format("steps_scaler must be finite (got {})", steps_scaler);
            if (!std::isfinite(max_screen_share))
                return std::format("max_screen_share must be finite (got {})", max_screen_share);
            if (ppisp_holdout_appearance != PPISPHoldoutAppearance::Mean && ppisp_holdout_appearance != PPISPHoldoutAppearance::Nearest)
                return "ppisp_holdout_appearance must be mean or nearest";
            if (!std::isfinite(densify_structure_weight) || densify_structure_weight < 0.0f || densify_structure_weight > 4.0f)
                return std::format("densify_structure_weight must be finite and within [0, 4] (got {})", densify_structure_weight);
            if (!std::isfinite(gradient_loss_weight) || gradient_loss_weight < 0.0f || gradient_loss_weight > 8.0f)
                return std::format("gradient_loss_weight must be finite and within [0, 8] (got {})", gradient_loss_weight);
            if (!std::isfinite(thin_structure_weight) || thin_structure_weight < 0.0f || thin_structure_weight > 4.0f)
                return std::format("thin_structure_weight must be finite and within [0, 4] (got {})", thin_structure_weight);
            if (!std::isfinite(late_lr_anneal) || late_lr_anneal <= 0.0f || late_lr_anneal > 1.0f)
                return std::format("late_lr_anneal must be finite and within (0, 1] (got {})", late_lr_anneal);
            if (!std::isfinite(scale_reg_decay_power) || scale_reg_decay_power < -1.0f)
                return std::format("scale_reg_decay_power must be finite and at least -1 (got {})", scale_reg_decay_power);
            if (perf_bench_warmup < 0)
                return std::format("perf_bench_warmup must be nonnegative (got {})", perf_bench_warmup);
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
                std::pair{"erank_reg", erank_reg},
                std::pair{"dc_reg", dc_reg},
                std::pair{"sh_rest_reg", sh_rest_reg},
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
                std::pair{"screen_share_penalty", screen_share_penalty},
            };
            for (const auto& [name, value] : nonnegative_fields) {
                const bool automatic_mrnf_value = is_mrnf_strategy(strategy) && value == -1.0f &&
                                                  (std::string_view{name} == "shs_lr");
                if (auto error = invalid_nonnegative(value, name);
                    !error.empty() && !automatic_mrnf_value)
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
                std::pair{"normal_start_fraction", normal_start_fraction},
                std::pair{"normal_end_fraction", normal_end_fraction},
            };
            for (const auto& [name, value] : probability_fields) {
                const bool automatic_mrnf_value = is_mrnf_strategy(strategy) && value == -1.0f &&
                                                  (std::string_view{name} == "grow_fraction");
                if (auto error = invalid_probability(value, name);
                    !error.empty() && !automatic_mrnf_value)
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
            const auto conflict = backend_conflict();
            if (conflict != TrainingBackendConflict::None) {
                return backend_conflict_message();
            }
            if (use_exposure_correction &&
                (use_bilateral_grid || use_ppisp || ppisp_use_controller || ppisp_freeze_from_sidecar)) {
                return "use_exposure_correction cannot be combined with use_bilateral_grid, "
                       "use_ppisp, ppisp_use_controller, or ppisp_freeze_from_sidecar; "
                       "exposure correction replaces the standalone bilateral grid and PPISP options";
            }
            if (ppisp_freeze_from_sidecar && !use_ppisp)
                return "PPISP sidecar freeze requires PPISP enabled";
            if (depth_loss_mode != "ssi" && depth_loss_mode != "ssi-disparity" && depth_loss_mode != "ssi-depth")
                return "depth_loss_mode must be 'ssi', 'ssi-disparity', or 'ssi-depth'";
            if (normal_loss_space != NormalLossSpace::Auto &&
                normal_loss_space != NormalLossSpace::CameraOpenCV &&
                normal_loss_space != NormalLossSpace::CameraOpenGL &&
                normal_loss_space != NormalLossSpace::World)
                return "normal_loss_space must be 'auto', 'camera-opencv', 'camera-opengl', or 'world'";
            if (eval_space != EvalSpace::Distorted && eval_space != EvalSpace::Undistorted)
                return "eval_space must be 'distorted' or 'undistorted'";
            if (eval_bit_depth != EvalBitDepth::Auto && eval_bit_depth != EvalBitDepth::Eight &&
                eval_bit_depth != EvalBitDepth::Sixteen && eval_bit_depth != EvalBitDepth::Float)
                return "eval_bit_depth must be 'auto', '8', '16' or 'float'";
            if (normal_start_fraction > normal_end_fraction)
                return std::format(
                    "normal_start_fraction must not exceed normal_end_fraction ({} > {})",
                    normal_start_fraction, normal_end_fraction);
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
            if (mcp_port && (*mcp_port < 1 || *mcp_port > 65535))
                return std::format("mcp_port must be within [1, 65535] (got {})", *mcp_port);
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
                if (!add_splats_applied &&
                    (resume_checkpoint.has_value() ||
                     resume_project.has_value() ||
                     project_path.has_value())) {
                    return "--add-splat cannot be used together with --resume";
                }
                if (!add_splat_freeze.empty() && add_splat_freeze.size() != add_splat_paths.size()) {
                    return "--add-splat freeze metadata is inconsistent";
                }
                for (const auto& path : add_splat_paths) {
                    if (path.empty()) {
                        return "--add-splat path cannot be empty";
                    }
                    if (!add_splats_applied && !std::filesystem::exists(path)) {
                        return std::format("Added splat does not exist: '{}'",
                                           lfs::core::path_to_utf8(path));
                    }
                }
            }
            if (resume_checkpoint && resume_project) {
                return "Only one resume source may be active";
            }
            if (project_path &&
                (resume_checkpoint || resume_project)) {
                return "A project path and --resume are mutually exclusive";
            }
            if (dataset_project &&
                (resume_checkpoint || resume_project)) {
                return "--data-path project.licht and --resume are mutually exclusive";
            }
            if (project_path) {
                auto extension = project_path->extension().string();
                std::ranges::transform(
                    extension, extension.begin(),
                    [](const unsigned char character) {
                        return static_cast<char>(
                            std::tolower(character));
                    });
                if (extension != ".licht") {
                    return "The project path must reference a .licht file";
                }
                if (!lfs::core::project::isPublishedLichtPath(*project_path)) {
                    return lfs::core::project::unpublishedLichtUserMessage(
                        *project_path);
                }
            }
            if (resume_project) {
                auto resume_extension = resume_project->extension().string();
                std::ranges::transform(
                    resume_extension, resume_extension.begin(),
                    [](const unsigned char character) {
                        return static_cast<char>(
                            std::tolower(character));
                    });
                if (resume_extension != ".licht") {
                    return "The resume project must reference a .licht file";
                }
                if (!lfs::core::project::isPublishedLichtPath(*resume_project)) {
                    return lfs::core::project::unpublishedLichtUserMessage(
                        *resume_project);
                }
            }
            if (save_project_at_iteration && *save_project_at_iteration == 0) {
                return "--save-project-at-iter must be positive";
            }
            if (save_project_at_iteration &&
                *save_project_at_iteration >
                    optimization.iterations) {
                return "--save-project-at-iter cannot exceed the training iteration limit";
            }
            if (!save_project_at_iteration &&
                !save_project_path.empty()) {
                return "--save-project-path requires --save-project-at-iter";
            }
            if (!save_project_path.empty() &&
                save_project_path.extension() != ".licht") {
                return "--save-project-path must end in .licht";
            }
            if (optimization.ppisp_freeze_from_sidecar &&
                !resume_checkpoint.has_value() && !resume_project.has_value()) {
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
            // Exports go to output_path / output_name, so the name must stay inside it.
            if (const auto name = lfs::core::utf8_to_path(output_name);
                output_name == "." || name.has_root_path() || name.has_root_name() ||
                std::ranges::any_of(name, [](const auto& part) { return part == ".."; })) {
                return "output-name must stay inside the output path (no absolute paths or '..')";
            }
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
            p.use_exposure_correction = true;
            p.refine_every = 200;
            p.start_refine = 0;
            p.stop_refine = 28'500;
            p.max_cap = 5'000'000;
            p.grow_fraction = -1.0f;
            p.shs_lr = -1.0f;
            p.thin_structure_weight = 0.5f;
            p.gradient_loss_weight = 1.8f;
            p.opacity_decay_rendered_only = true;
            p.densify_structure_weight = 1.0f;
            p.min_opacity = 1.0f / 255.0f;
            p.means_lr_end = 2e-7f;
            p.opacity_lr = 0.012f;
            p.scaling_lr_end = 5e-3f;
            p.late_lr_anneal = 0.3f;
            p.lambda_dssim = 0.22f;
            p.growth_grad_threshold = 0.00309693f;
            p.max_screen_share = 0.586511f;
            p.screen_share_penalty = 0.847085f;
            p.means_lr = 2.17871e-5f;
            p.refine_every = 163;
            p.scaling_lr = 0.00828016f;
            p.rotation_lr = 0.0015f;
            p.opacity_reg = 0.003f;
            p.scale_reg = 0.01f;
            p.scale_reg_decay_power = 0.4f;
            p.erank_reg = 0.001f;
            p.dc_reg = 0.001f;
            p.sh_rest_reg = 0.001f;
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
            apply_optimization_json_overlay(params, json, false);
            params.eval_mask = normalize_eval_mask(params.eval_mask);
            // Legacy GUI saves recorded the image factor in steps_scaler.
            if (!stored_image_count_scaler(json, params.steps_scaler))
                params.image_count_scaler = params.steps_scaler > 0.f ? params.steps_scaler : 1.f;
            return params;
        }

        bool ExplicitTrainingOverrides::has_optimization_key(const std::string_view key) const {
            return overlay_has_key(optimization_json, key);
        }

        bool ExplicitTrainingOverrides::has_dataset_key(const std::string_view key) const {
            return overlay_has_key(dataset_json, key);
        }

        void merge_explicit_json_overlay(std::string& dst_json, const std::string_view src_json) {
            if (src_json.empty())
                return;
            auto src = parse_overlay_object(src_json);
            if (src.empty())
                return;
            auto dst = parse_overlay_object(dst_json);
            for (auto it = src.begin(); it != src.end(); ++it)
                dst[it.key()] = it.value();
            dst_json = dst.dump();
        }

        void apply_explicit_training_overrides(
            TrainingParameters& target,
            const ExplicitTrainingOverrides& overrides) {
            if (!overrides.optimization_json.empty())
                apply_optimization_json_overlay(
                    target.optimization,
                    parse_overlay_object(overrides.optimization_json),
                    true);
            if (!overrides.dataset_json.empty())
                apply_dataset_json_overlay(
                    target.dataset, parse_overlay_object(overrides.dataset_json));
        }

        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(
            const std::filesystem::path& path,
            ExplicitTrainingOverrides& captured_overrides) {
            auto json_result = read_json_file(path);
            if (!json_result) {
                return std::unexpected(json_result.error());
            }

            const auto& json = *json_result;
            const auto& opt_json = json.contains("optimization") ? json["optimization"] : json;

            try {
                if (!opt_json.is_object())
                    return std::unexpected("Optimization parameters must be a JSON object");
                if (const auto error = validate_optimization_json_types(opt_json))
                    return std::unexpected("Error parsing optimization parameters: " + *error);
                if (const auto error = validate_registered_optimization_enums(opt_json))
                    return std::unexpected("Error parsing optimization parameters: " + *error);
                if (json.contains("dataset")) {
                    if (!json["dataset"].is_object())
                        return std::unexpected("Dataset parameters must be a JSON object");
                    if (const auto error = validate_dataset_json_types(json["dataset"]))
                        return std::unexpected(*error);
                }
                if (json.contains("server")) {
                    if (!json["server"].is_object())
                        return std::unexpected("Server parameters must be a JSON object");
                    if (const auto error = validate_server_json_types(json["server"]))
                        return std::unexpected(*error);
                }

                auto params = OptimizationParameters::from_json(opt_json);
                if (auto error = params.validate(); !error.empty())
                    return std::unexpected("Invalid optimization parameters: " + error);
                if (opt_json.is_object() && !opt_json.empty())
                    captured_overrides.optimization_json = opt_json.dump();
                if (json.contains("dataset") && json["dataset"].is_object() &&
                    !json["dataset"].empty()) {
                    captured_overrides.dataset_json = json["dataset"].dump();
                }
                return params;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("Error parsing optimization parameters: {}", e.what()));
            }
        }

        std::expected<OptimizationParameters, std::string> read_optim_params_from_json(const std::filesystem::path& path) {
            ExplicitTrainingOverrides unused;
            return read_optim_params_from_json(path, unused);
        }

        std::expected<TrainingParameters, lfs::Error> read_training_parameters_from_json(
            const std::filesystem::path& path,
            const TrainingParameters& defaults) {
            auto json_result = read_json_file(path);
            if (!json_result) {
                return std::unexpected(config_import_error(std::move(json_result.error()), path));
            }

            const auto& json = *json_result;
            const auto& opt_json = json.contains("optimization") ? json["optimization"] : json;
            if (!opt_json.is_object()) {
                return std::unexpected(config_import_error("Optimization parameters must be a JSON object", path));
            }

            try {
                if (const auto error = validate_optimization_json_types(opt_json)) {
                    return std::unexpected(config_import_error(*error, path));
                }
                if (const auto error = validate_registered_optimization_enums(opt_json)) {
                    return std::unexpected(config_import_error(*error, path));
                }

                TrainingParameters params = defaults;
                params.optimization = OptimizationParameters::mrnf_defaults();
                if (opt_json.contains("strategy")) {
                    const auto strategy = opt_json.at("strategy").get<std::string>();
                    const auto canonical = canonical_strategy_name(strategy);
                    if (!canonical.empty()) {
                        params.optimization = OptimizationParameters::defaults_for_strategy(canonical);
                    }
                }
                apply_optimization_json_overlay(params.optimization, opt_json, true);

                if (json.contains("dataset")) {
                    if (!json["dataset"].is_object()) {
                        return std::unexpected(config_import_error("Dataset parameters must be a JSON object", path));
                    }
                    if (const auto error = validate_dataset_json_types(json["dataset"])) {
                        return std::unexpected(config_import_error(*error, path));
                    }
                    apply_dataset_json_overlay(params.dataset, json["dataset"]);
                }
                if (json.contains("server")) {
                    if (!json["server"].is_object()) {
                        return std::unexpected(config_import_error("Server parameters must be a JSON object", path));
                    }
                    const auto& server_json = json["server"];
                    if (server_json.contains("tcp_server_connection_port")) {
                        if (!is_json_integer(server_json["tcp_server_connection_port"]) ||
                            server_json["tcp_server_connection_port"].get<long double>() < std::numeric_limits<int>::min() ||
                            server_json["tcp_server_connection_port"].get<long double>() > std::numeric_limits<int>::max()) {
                            return std::unexpected(config_import_error(
                                "Invalid type for server field 'tcp_server_connection_port'; expected integer", path));
                        }
                        params.server.tcp_server_connection_port =
                            server_json["tcp_server_connection_port"].get<int>();
                    }
                    if (server_json.contains("tcp_broadcast_connection_port")) {
                        if (!is_json_integer(server_json["tcp_broadcast_connection_port"]) ||
                            server_json["tcp_broadcast_connection_port"].get<long double>() < std::numeric_limits<int>::min() ||
                            server_json["tcp_broadcast_connection_port"].get<long double>() > std::numeric_limits<int>::max()) {
                            return std::unexpected(config_import_error(
                                "Invalid type for server field 'tcp_broadcast_connection_port'; expected integer", path));
                        }
                        params.server.tcp_broadcast_connection_port =
                            server_json["tcp_broadcast_connection_port"].get<int>();
                    }
                    if (server_json.contains("tcp_connection")) {
                        if (!server_json["tcp_connection"].is_boolean()) {
                            return std::unexpected(config_import_error(
                                "Invalid type for server field 'tcp_connection'; expected boolean", path));
                        }
                        params.server.tcp_connection = server_json["tcp_connection"].get<bool>();
                    }
                }

                if (const auto error = params.optimization.validate(); !error.empty()) {
                    return std::unexpected(config_import_error("Invalid optimization parameters: " + error, path));
                }
                if (const auto error = params.dataset.validate(); !error.empty()) {
                    return std::unexpected(config_import_error("Invalid dataset parameters: " + error, path));
                }
                return params;
            } catch (const std::exception& e) {
                return std::unexpected(config_import_error(std::format("Error parsing training parameters: {}", e.what()), path));
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
                const auto time_t_val = std::chrono::system_clock::to_time_t(now);
                std::tm tm{};
#ifdef _WIN32
                localtime_s(&tm, &time_t_val);
#else
                localtime_r(&time_t_val, &tm);
#endif
                std::stringstream ss;
                ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
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
            json["timelapse_images"] = timelapse_images;
            json["timelapse_every"] = timelapse_every;
            json["max_width"] = max_width;
            json["min_track_length"] = min_track_length;
            json["loading_params"] = loading_params.to_json();
            json["invert_masks"] = invert_masks;
            json["mask_threshold"] = mask_threshold;
            json["centralize_dataset"] = centralize_dataset;
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
            if (j.contains("timelapse_images")) {
                dataset.timelapse_images =
                    j["timelapse_images"]
                        .get<std::vector<std::string>>();
            }
            if (j.contains("timelapse_every")) {
                dataset.timelapse_every =
                    j["timelapse_every"].get<int>();
            }
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
            if (j.contains("centralize_dataset")) {
                dataset.centralize_dataset = j["centralize_dataset"].get<std::string>();
            }

            return dataset;
        }

    } // namespace param
} // namespace lfs::core
