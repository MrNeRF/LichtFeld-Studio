/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_params.hpp"

#include "control/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/parameter_manager.hpp"
#include "visualizer/training/training_manager.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <set>

namespace lfs::python {

    using namespace lfs::core::param;
    using namespace lfs::core::prop;
    using lfs::training::CommandCenter;

    void register_optimization_properties() {
        PropertyGroupBuilder<OptimizationParameters>("optimization", "Optimization")
            // Training control
            .size_prop(&OptimizationParameters::iterations,
                       "iterations", "Max Iterations", 30000, 1, 1000000,
                       "Maximum number of training iterations")
            .int_prop(&OptimizationParameters::sh_degree,
                      "sh_degree", "SH Degree", 3, 0, 3,
                      "Spherical harmonics degree (0-3)")
            .size_prop(&OptimizationParameters::sh_degree_interval,
                       "sh_degree_interval", "SH Interval", 1000, 100, 10000,
                       "Iterations between SH degree increases")
            .int_prop(&OptimizationParameters::max_cap,
                      "max_cap", "Max Gaussians", 1000000, 1000, 10000000,
                      "Maximum number of gaussians")

            // Learning rates
            .float_prop(&OptimizationParameters::means_lr,
                        "means_lr", "Position LR", 0.000016f, 0.0f, 0.001f,
                        "Learning rate for gaussian positions")
            .flags(PROP_LIVE_UPDATE)
            .float_prop(&OptimizationParameters::shs_lr,
                        "shs_lr", "SH LR", 0.0025f, 0.0f, 0.1f,
                        "Learning rate for spherical harmonics")
            .flags(PROP_LIVE_UPDATE)
            .float_prop(&OptimizationParameters::opacity_lr,
                        "opacity_lr", "Opacity LR", 0.05f, 0.0f, 1.0f,
                        "Learning rate for opacity")
            .flags(PROP_LIVE_UPDATE)
            .float_prop(&OptimizationParameters::scaling_lr,
                        "scaling_lr", "Scale LR", 0.005f, 0.0f, 0.1f,
                        "Learning rate for gaussian scales")
            .flags(PROP_LIVE_UPDATE)
            .float_prop(&OptimizationParameters::rotation_lr,
                        "rotation_lr", "Rotation LR", 0.001f, 0.0f, 0.1f,
                        "Learning rate for rotations")
            .flags(PROP_LIVE_UPDATE)

            // Loss parameters
            .float_prop(&OptimizationParameters::lambda_dssim,
                        "lambda_dssim", "DSSIM Weight", 0.2f, 0.0f, 1.0f,
                        "Weight for structural similarity loss")
            .float_prop(&OptimizationParameters::opacity_reg,
                        "opacity_reg", "Opacity Reg", 0.01f, 0.0f, 1.0f,
                        "Opacity regularization weight")
            .float_prop(&OptimizationParameters::scale_reg,
                        "scale_reg", "Scale Reg", 0.01f, 0.0f, 1.0f,
                        "Scale regularization weight")

            // Refinement
            .size_prop(&OptimizationParameters::refine_every,
                       "refine_every", "Refine Every", 100, 1, 1000,
                       "Interval for adaptive density control")
            .size_prop(&OptimizationParameters::start_refine,
                       "start_refine", "Start Refine", 500, 0, 10000,
                       "Iteration to start refinement")
            .size_prop(&OptimizationParameters::stop_refine,
                       "stop_refine", "Stop Refine", 25000, 0, 100000,
                       "Iteration to stop refinement")
            .float_prop(&OptimizationParameters::grad_threshold,
                        "grad_threshold", "Grad Threshold", 0.0002f, 0.0f, 0.01f,
                        "Gradient threshold for densification")
            .float_prop(&OptimizationParameters::min_opacity,
                        "min_opacity", "Min Opacity", 0.005f, 0.0f, 0.1f,
                        "Minimum opacity for pruning")
            .float_prop(&OptimizationParameters::init_opacity,
                        "init_opacity", "Init Opacity", 0.5f, 0.0f, 1.0f,
                        "Initial opacity for new gaussians")
            .float_prop(&OptimizationParameters::init_scaling,
                        "init_scaling", "Init Scale", 0.1f, 0.0f, 1.0f,
                        "Initial scale for new gaussians")

            // Mask parameters
            .enum_prop(&OptimizationParameters::mask_mode,
                       "mask_mode", "Mask Mode", MaskMode::None,
                       {{"None", MaskMode::None},
                        {"Segment", MaskMode::Segment},
                        {"Ignore", MaskMode::Ignore},
                        {"AlphaConsistent", MaskMode::AlphaConsistent}},
                       "Attention mask behavior during training")
            .bool_prop(&OptimizationParameters::invert_masks,
                       "invert_masks", "Invert Masks", false,
                       "Swap object and background in masks")
            .float_prop(&OptimizationParameters::mask_threshold,
                        "mask_threshold", "Mask Threshold", 0.5f, 0.0f, 1.0f,
                        "Threshold for mask binarization")
            .float_prop(&OptimizationParameters::mask_opacity_penalty_weight,
                        "mask_opacity_penalty_weight", "Penalty Weight", 1.0f, 0.0f, 10.0f,
                        "Opacity penalty weight for segment mode")
            .float_prop(&OptimizationParameters::mask_opacity_penalty_power,
                        "mask_opacity_penalty_power", "Penalty Power", 2.0f, 0.5f, 4.0f,
                        "Power for opacity penalty in segment mode")
            .bool_prop(&OptimizationParameters::use_alpha_as_mask,
                       "use_alpha_as_mask", "Use Alpha as Mask", true,
                       "Use alpha channel from RGBA images as mask source")

            // Bilateral grid
            .bool_prop(&OptimizationParameters::use_bilateral_grid,
                       "use_bilateral_grid", "Bilateral Grid", false,
                       "Enable bilateral grid color correction")
            .flags(PROP_NEEDS_RESTART)
            .int_prop(&OptimizationParameters::bilateral_grid_X,
                      "bilateral_grid_x", "Grid X", 16, 4, 64,
                      "Bilateral grid X resolution")
            .int_prop(&OptimizationParameters::bilateral_grid_Y,
                      "bilateral_grid_y", "Grid Y", 16, 4, 64,
                      "Bilateral grid Y resolution")
            .int_prop(&OptimizationParameters::bilateral_grid_W,
                      "bilateral_grid_w", "Grid W", 8, 2, 32,
                      "Bilateral grid intensity bins")
            .float_prop(&OptimizationParameters::bilateral_grid_lr,
                        "bilateral_grid_lr", "Grid LR", 0.002f, 0.0f, 0.1f,
                        "Bilateral grid learning rate")
            .float_prop(&OptimizationParameters::tv_loss_weight,
                        "tv_loss_weight", "TV Loss Weight", 10.0f, 0.0f, 100.0f,
                        "Total variation loss weight")

            // Strategy
            .string_prop(&OptimizationParameters::strategy,
                         "strategy", "Strategy", "mcmc",
                         "Optimization strategy: mcmc or adc")
            .flags(PROP_NEEDS_RESTART)

            // ADC strategy parameters
            .float_prop(&OptimizationParameters::prune_opacity,
                        "prune_opacity", "Prune Opacity", 0.005f, 0.0f, 0.1f,
                        "Opacity threshold for pruning (ADC)")
            .float_prop(&OptimizationParameters::grow_scale3d,
                        "grow_scale3d", "Grow Scale 3D", 0.01f, 0.0f, 0.1f,
                        "3D scale threshold for growing (ADC)")
            .float_prop(&OptimizationParameters::grow_scale2d,
                        "grow_scale2d", "Grow Scale 2D", 0.05f, 0.0f, 0.2f,
                        "2D scale threshold for growing (ADC)")
            .size_prop(&OptimizationParameters::reset_every,
                       "reset_every", "Reset Every", 3000, 100, 10000,
                       "Iteration interval for opacity reset (ADC)")
            .float_prop(&OptimizationParameters::prune_scale3d,
                        "prune_scale3d", "Prune Scale 3D", 0.1f, 0.0f, 1.0f,
                        "3D scale threshold for pruning (ADC)")
            .float_prop(&OptimizationParameters::prune_scale2d,
                        "prune_scale2d", "Prune Scale 2D", 0.15f, 0.0f, 0.5f,
                        "2D scale threshold for pruning (ADC)")
            .size_prop(&OptimizationParameters::pause_refine_after_reset,
                       "pause_refine_after_reset", "Pause After Reset", 0, 0, 1000,
                       "Iterations to pause refinement after opacity reset")
            .bool_prop(&OptimizationParameters::revised_opacity,
                       "revised_opacity", "Revised Opacity", false,
                       "Use revised opacity calculation for ADC")

            // Flags
            .bool_prop(&OptimizationParameters::mip_filter,
                       "mip_filter", "Mip Filter", false,
                       "Enable mip filtering (anti-aliasing)")
            .bool_prop(&OptimizationParameters::use_ppisp,
                       "ppisp", "PPISP", true,
                       "Per-pixel image signal processing")
            .bool_prop(&OptimizationParameters::bg_modulation,
                       "bg_modulation", "BG Modulation", false,
                       "Enable sinusoidal background modulation")
            .bool_prop(&OptimizationParameters::headless,
                       "headless", "Headless", false,
                       "Run without visualization")
            .flags(PROP_READONLY)
            .bool_prop(&OptimizationParameters::enable_eval,
                       "enable_eval", "Enable Eval", false,
                       "Run evaluation at specified steps")

            // Random initialization
            .bool_prop(&OptimizationParameters::random,
                       "random", "Random Init", false,
                       "Use random initialization instead of SfM")
            .flags(PROP_NEEDS_RESTART)
            .int_prop(&OptimizationParameters::init_num_pts,
                      "init_num_pts", "Init Points", 100000, 1000, 1000000,
                      "Number of random points to initialize")
            .float_prop(&OptimizationParameters::init_extent,
                        "init_extent", "Init Extent", 3.0f, 0.1f, 10.0f,
                        "Extent of random point cloud")

            // Sparsity
            .bool_prop(&OptimizationParameters::enable_sparsity,
                       "enable_sparsity", "Enable Sparsity", false,
                       "Enable sparsity optimization")
            .int_prop(&OptimizationParameters::sparsify_steps,
                      "sparsify_steps", "Sparsify Steps", 15000, 1000, 50000,
                      "Iteration to run sparsification")
            .float_prop(&OptimizationParameters::prune_ratio,
                        "prune_ratio", "Prune Ratio", 0.6f, 0.0f, 1.0f,
                        "Target pruning ratio for sparsification")
            .float_prop(&OptimizationParameters::init_rho,
                        "init_rho", "Init Rho", 0.001f, 0.0f, 0.01f,
                        "Initial rho for sparsity optimization")
            .int_prop(&OptimizationParameters::tile_mode,
                      "tile_mode", "Tile Mode", 1, 1, 4,
                      "Tile mode (1, 2, or 4)")
            .float_prop(&OptimizationParameters::steps_scaler,
                        "steps_scaler", "Steps Scaler", 1.0f, 0.0f, 10.0f,
                        "Scale training step counts")
            .bool_prop(&OptimizationParameters::gut,
                       "gut", "GUT", false,
                       "Gaussian Unscented Transform")
            .bool_prop(&OptimizationParameters::undistort,
                       "undistort", "Undistort", false,
                       "Undistort images on-the-fly before training")
            .flags(PROP_NEEDS_RESTART)
            .enum_prop(&OptimizationParameters::bg_mode,
                       "bg_mode", "Background Mode", BackgroundMode::SolidColor,
                       {{"SolidColor", BackgroundMode::SolidColor},
                        {"Modulation", BackgroundMode::Modulation},
                        {"Image", BackgroundMode::Image},
                        {"Random", BackgroundMode::Random}},
                       "Background mode")
            .build();
    }

    void register_dataset_properties() {
        PropertyGroup group;
        group.id = "dataset";
        group.name = "Dataset";

        auto add_string = [&](const std::string& id, const std::string& name, const std::string& default_val,
                              const std::string& desc, bool readonly, std::function<std::string(const DatasetConfig&)> getter,
                              std::function<void(DatasetConfig&, const std::string&)> setter = nullptr) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::String;
            meta.default_string = default_val;
            if (readonly) {
                meta.flags = PROP_READONLY;
            }
            meta.getter = [getter](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return getter(*static_cast<const DatasetConfig*>(ref.ptr));
            };
            if (setter) {
                meta.setter = [setter](PropertyObjectRef& ref, const std::any& val) {
                    assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                    setter(*static_cast<DatasetConfig*>(ref.ptr), std::any_cast<std::string>(val));
                };
            }
            group.properties.push_back(std::move(meta));
        };

        auto add_int = [&](const std::string& id, const std::string& name, int default_val, int min_val, int max_val,
                           const std::string& desc, bool readonly, std::function<int(const DatasetConfig&)> getter,
                           std::function<void(DatasetConfig&, int)> setter = nullptr) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Int;
            meta.default_value = default_val;
            meta.min_value = min_val;
            meta.max_value = max_val;
            meta.soft_min = min_val;
            meta.soft_max = max_val;
            meta.step = 1.0;
            if (readonly) {
                meta.flags = PROP_READONLY;
            }
            meta.getter = [getter](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return getter(*static_cast<const DatasetConfig*>(ref.ptr));
            };
            if (setter) {
                meta.setter = [setter](PropertyObjectRef& ref, const std::any& val) {
                    assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                    setter(*static_cast<DatasetConfig*>(ref.ptr), std::any_cast<int>(val));
                };
            }
            group.properties.push_back(std::move(meta));
        };

        auto add_bool = [&](const std::string& id, const std::string& name, bool default_val, const std::string& desc,
                            bool readonly, std::function<bool(const DatasetConfig&)> getter,
                            std::function<void(DatasetConfig&, bool)> setter = nullptr) {
            PropertyMeta meta;
            meta.id = id;
            meta.name = name;
            meta.description = desc;
            meta.type = PropType::Bool;
            meta.default_value = default_val ? 1.0 : 0.0;
            if (readonly) {
                meta.flags = PROP_READONLY;
            }
            meta.getter = [getter](const PropertyObjectRef& ref) -> std::any {
                assert(ref.is_cpp() && "Cannot call C++ property getter with Python object");
                return getter(*static_cast<const DatasetConfig*>(ref.ptr));
            };
            if (setter) {
                meta.setter = [setter](PropertyObjectRef& ref, const std::any& val) {
                    assert(ref.is_cpp() && "Cannot call C++ property setter with Python object");
                    setter(*static_cast<DatasetConfig*>(ref.ptr), std::any_cast<bool>(val));
                };
            }
            group.properties.push_back(std::move(meta));
        };

        add_string(
            "data_path", "Data Path", "", "Path to training data", true,
            [](const DatasetConfig& c) { return c.data_path.string(); });

        add_string(
            "output_path", "Output Path", "", "Path for output files", true,
            [](const DatasetConfig& c) { return c.output_path.string(); });

        add_string(
            "images", "Images Folder", "images", "Subfolder containing images", true,
            [](const DatasetConfig& c) { return c.images; });

        add_int(
            "resize_factor", "Resize Factor", -1, -1, 8, "Image resize factor (-1 = auto)", false,
            [](const DatasetConfig& c) { return c.resize_factor; },
            [](DatasetConfig& c, int v) { c.resize_factor = v; });

        add_int(
            "test_every", "Test Every", 8, 1, 100, "Use every Nth image for testing", true,
            [](const DatasetConfig& c) { return c.test_every; });

        add_int(
            "max_width", "Max Width", 3840, 640, 4096, "Maximum image width", false,
            [](const DatasetConfig& c) { return c.max_width; },
            [](DatasetConfig& c, int v) { c.max_width = v; });

        add_bool(
            "use_cpu_cache", "CPU Cache", true, "Cache images in CPU memory", false,
            [](const DatasetConfig& c) { return c.loading_params.use_cpu_memory; },
            [](DatasetConfig& c, bool v) { c.loading_params.use_cpu_memory = v; });

        add_bool(
            "use_fs_cache", "FS Cache", true, "Use filesystem cache for images", false,
            [](const DatasetConfig& c) { return c.loading_params.use_fs_cache; },
            [](DatasetConfig& c, bool v) { c.loading_params.use_fs_cache = v; });

        PropertyRegistry::instance().register_group(std::move(group));
    }

    namespace {
        core::param::OptimizationParameters& get_default_params() {
            static core::param::OptimizationParameters default_params{};
            return default_params;
        }
    } // namespace

    bool PyOptimizationParams::has_params() const {
        auto* pm = get_parameter_manager();
        return pm != nullptr;
    }

    core::param::OptimizationParameters& PyOptimizationParams::params() {
        auto* pm = get_parameter_manager();
        if (!pm) {
            return get_default_params();
        }
        return pm->getActiveParams();
    }

    const core::param::OptimizationParameters& PyOptimizationParams::params() const {
        auto* pm = get_parameter_manager();
        if (!pm) {
            return get_default_params();
        }
        return pm->getActiveParams();
    }

    nb::object PyOptimizationParams::get(const std::string& prop_id) const {
        auto meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        const auto& p = params();
        auto ref = PropertyObjectRef::cpp(const_cast<OptimizationParameters*>(&p));
        std::any value = meta->getter(ref);

        switch (meta->type) {
        case PropType::Bool:
            return nb::cast(std::any_cast<bool>(value));
        case PropType::Int:
            return nb::cast(std::any_cast<int>(value));
        case PropType::Float:
            return nb::cast(std::any_cast<float>(value));
        case PropType::String:
            return nb::cast(std::any_cast<std::string>(value));
        case PropType::SizeT:
            return nb::cast(std::any_cast<size_t>(value));
        case PropType::Enum:
            return nb::cast(std::any_cast<int>(value));
        default:
            throw std::runtime_error("Unsupported property type");
        }
    }

    void PyOptimizationParams::set(const std::string& prop_id, nb::object value) {
        auto meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        if (meta->is_readonly()) {
            throw std::runtime_error("Property is read-only: " + prop_id);
        }

        auto& p = params();
        auto ref = PropertyObjectRef::cpp(&p);
        std::any old_value = meta->getter(ref);
        std::any new_value;

        switch (meta->type) {
        case PropType::Bool:
            new_value = nb::cast<bool>(value);
            break;
        case PropType::Int:
            new_value = nb::cast<int>(value);
            break;
        case PropType::Float:
            new_value = nb::cast<float>(value);
            break;
        case PropType::String:
            new_value = nb::cast<std::string>(value);
            break;
        case PropType::SizeT:
            new_value = static_cast<size_t>(nb::cast<int64_t>(value));
            break;
        case PropType::Enum:
            new_value = nb::cast<int>(value);
            break;
        default:
            throw std::runtime_error("Unsupported property type");
        }

        meta->setter(ref, new_value);
        PropertyRegistry::instance().notify("optimization", prop_id, old_value, new_value);
    }

    nb::dict PyOptimizationParams::prop_info(const std::string& prop_id) const {
        auto meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        nb::dict info;
        info["id"] = meta->id;
        info["name"] = meta->name;
        info["description"] = meta->description;
        info["group"] = meta->group;
        info["readonly"] = meta->is_readonly();
        info["live_update"] = meta->is_live_update();
        info["needs_restart"] = meta->needs_restart();

        switch (meta->type) {
        case PropType::Float:
            info["type"] = "float";
            info["min"] = meta->min_value;
            info["max"] = meta->max_value;
            info["default"] = meta->default_value;
            break;
        case PropType::Int:
            info["type"] = "int";
            info["min"] = static_cast<int>(meta->min_value);
            info["max"] = static_cast<int>(meta->max_value);
            info["default"] = static_cast<int>(meta->default_value);
            break;
        case PropType::SizeT:
            info["type"] = "int";
            info["min"] = static_cast<int64_t>(meta->min_value);
            info["max"] = static_cast<int64_t>(meta->max_value);
            info["default"] = static_cast<int64_t>(meta->default_value);
            break;
        case PropType::Bool:
            info["type"] = "bool";
            info["default"] = meta->default_value > 0.5;
            break;
        case PropType::String:
            info["type"] = "string";
            info["default"] = meta->default_string;
            break;
        case PropType::Enum:
            info["type"] = "enum";
            info["default"] = meta->default_enum;
            {
                nb::list items;
                for (const auto& ei : meta->enum_items) {
                    nb::dict item;
                    item["name"] = ei.name;
                    item["value"] = ei.value;
                    items.append(item);
                }
                info["items"] = items;
            }
            break;
        default:
            info["type"] = "unknown";
            break;
        }

        return info;
    }

    void PyOptimizationParams::reset(const std::string& prop_id) {
        auto meta = PropertyRegistry::instance().get_property("optimization", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        std::any default_val;
        switch (meta->type) {
        case PropType::Float:
            default_val = static_cast<float>(meta->default_value);
            break;
        case PropType::Int:
            default_val = static_cast<int>(meta->default_value);
            break;
        case PropType::SizeT:
            default_val = static_cast<size_t>(meta->default_value);
            break;
        case PropType::Bool:
            default_val = meta->default_value > 0.5;
            break;
        case PropType::String:
            default_val = meta->default_string;
            break;
        case PropType::Enum:
            default_val = meta->default_enum;
            break;
        default:
            throw std::runtime_error("Unsupported property type for reset");
        }

        auto& p = params();
        auto ref = PropertyObjectRef::cpp(&p);
        std::any old_value = meta->getter(ref);
        meta->setter(ref, default_val);
        PropertyRegistry::instance().notify("optimization", prop_id, old_value, default_val);
    }

    nb::list PyOptimizationParams::properties() const {
        auto* group = PropertyRegistry::instance().get_group("optimization");
        if (!group) {
            return nb::list();
        }

        nb::list result;
        for (const auto& prop : group->properties) {
            nb::dict item;
            item["id"] = prop.id;
            item["name"] = prop.name;
            item["group"] = prop.group;
            item["value"] = get(prop.id);
            result.append(item);
        }
        return result;
    }

    nb::dict PyOptimizationParams::get_all_properties() const {
        nb::dict result;
        const auto* group = PropertyRegistry::instance().get_group("optimization");
        if (!group) {
            return result;
        }

        nb::module_ props_module = nb::module_::import_("lfs_plugins.props");

        for (const auto& meta : group->properties) {
            nb::object prop_obj;

            switch (meta.type) {
            case PropType::Float: {
                nb::object cls = props_module.attr("FloatProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<float>(meta.default_value),
                    nb::arg("min") = static_cast<float>(meta.min_value),
                    nb::arg("max") = static_cast<float>(meta.max_value),
                    nb::arg("step") = static_cast<float>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Int: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(meta.default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value),
                    nb::arg("max") = static_cast<int>(meta.max_value),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::SizeT: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(meta.default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value),
                    nb::arg("max") = static_cast<int>(meta.max_value),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Bool: {
                nb::object cls = props_module.attr("BoolProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_value != 0.0,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::String: {
                nb::object cls = props_module.attr("StringProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_string,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Enum: {
                nb::object cls = props_module.attr("EnumProperty");
                nb::list items;
                std::string default_id;
                for (size_t i = 0; i < meta.enum_items.size(); ++i) {
                    const auto& item = meta.enum_items[i];
                    items.append(nb::make_tuple(item.identifier, item.name, ""));
                    if (static_cast<int>(i) == meta.default_enum) {
                        default_id = item.identifier;
                    }
                }
                prop_obj = cls(
                    nb::arg("items") = items,
                    nb::arg("default") = default_id,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            default:
                continue;
            }

            result[meta.id.c_str()] = prop_obj;
        }

        return result;
    }

    bool PyDatasetConfig::has_params() const {
        return get_trainer_manager() != nullptr;
    }

    bool PyDatasetConfig::can_edit() const {
        const auto* tm = get_trainer_manager();
        if (!tm)
            return false;
        return tm->getState() == lfs::vis::TrainingState::Ready && tm->getCurrentIteration() == 0;
    }

    core::param::DatasetConfig& PyDatasetConfig::params() {
        auto* tm = get_trainer_manager();
        if (!tm) {
            throw std::runtime_error("TrainerManager not available");
        }
        return tm->getEditableDatasetParams();
    }

    const core::param::DatasetConfig& PyDatasetConfig::params() const {
        const auto* tm = get_trainer_manager();
        if (!tm) {
            throw std::runtime_error("TrainerManager not available");
        }
        if (can_edit()) {
            return tm->getEditableDatasetParams();
        }
        if (tm->hasTrainer()) {
            if (const auto* trainer = tm->getTrainer()) {
                return trainer->getParams().dataset;
            }
        }
        return tm->getEditableDatasetParams();
    }

    nb::object PyDatasetConfig::get(const std::string& prop_id) const {
        auto meta = PropertyRegistry::instance().get_property("dataset", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        const auto& p = params();
        auto ref = PropertyObjectRef::cpp(const_cast<DatasetConfig*>(&p));
        std::any value = meta->getter(ref);

        switch (meta->type) {
        case PropType::Bool:
            return nb::cast(std::any_cast<bool>(value));
        case PropType::Int:
            return nb::cast(std::any_cast<int>(value));
        case PropType::Float:
            return nb::cast(std::any_cast<float>(value));
        case PropType::String:
            return nb::cast(std::any_cast<std::string>(value));
        case PropType::SizeT:
            return nb::cast(std::any_cast<size_t>(value));
        default:
            throw std::runtime_error("Unsupported property type");
        }
    }

    void PyDatasetConfig::set(const std::string& prop_id, nb::object value) {
        auto meta = PropertyRegistry::instance().get_property("dataset", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        if (meta->is_readonly()) {
            throw std::runtime_error("Property is read-only: " + prop_id);
        }

        if (!can_edit()) {
            throw std::runtime_error("Cannot edit dataset params during training");
        }

        auto& p = params();
        auto ref = PropertyObjectRef::cpp(&p);
        std::any old_value = meta->getter(ref);
        std::any new_value;

        switch (meta->type) {
        case PropType::Bool:
            new_value = nb::cast<bool>(value);
            break;
        case PropType::Int:
            new_value = nb::cast<int>(value);
            break;
        case PropType::Float:
            new_value = nb::cast<float>(value);
            break;
        case PropType::String:
            new_value = nb::cast<std::string>(value);
            break;
        case PropType::SizeT:
            new_value = static_cast<size_t>(nb::cast<int64_t>(value));
            break;
        default:
            throw std::runtime_error("Unsupported property type");
        }

        meta->setter(ref, new_value);
        PropertyRegistry::instance().notify("dataset", prop_id, old_value, new_value);
    }

    nb::dict PyDatasetConfig::prop_info(const std::string& prop_id) const {
        auto meta = PropertyRegistry::instance().get_property("dataset", prop_id);
        if (!meta) {
            throw std::runtime_error("Unknown property: " + prop_id);
        }

        nb::dict info;
        info["id"] = meta->id;
        info["name"] = meta->name;
        info["description"] = meta->description;
        info["group"] = meta->group;
        info["readonly"] = meta->is_readonly();

        switch (meta->type) {
        case PropType::Float:
            info["type"] = "float";
            info["min"] = meta->min_value;
            info["max"] = meta->max_value;
            info["default"] = meta->default_value;
            break;
        case PropType::Int:
            info["type"] = "int";
            info["min"] = static_cast<int>(meta->min_value);
            info["max"] = static_cast<int>(meta->max_value);
            info["default"] = static_cast<int>(meta->default_value);
            break;
        case PropType::SizeT:
            info["type"] = "int";
            info["min"] = static_cast<int64_t>(meta->min_value);
            info["max"] = static_cast<int64_t>(meta->max_value);
            info["default"] = static_cast<int64_t>(meta->default_value);
            break;
        case PropType::Bool:
            info["type"] = "bool";
            info["default"] = meta->default_value > 0.5;
            break;
        case PropType::String:
            info["type"] = "string";
            info["default"] = meta->default_string;
            break;
        default:
            info["type"] = "unknown";
            break;
        }

        return info;
    }

    nb::list PyDatasetConfig::properties() const {
        auto* group = PropertyRegistry::instance().get_group("dataset");
        if (!group) {
            return nb::list();
        }

        nb::list result;
        for (const auto& prop : group->properties) {
            nb::dict item;
            item["id"] = prop.id;
            item["name"] = prop.name;
            item["group"] = prop.group;
            item["value"] = get(prop.id);
            result.append(item);
        }
        return result;
    }

    nb::dict PyDatasetConfig::get_all_properties() const {
        nb::dict result;
        const auto* group = PropertyRegistry::instance().get_group("dataset");
        if (!group) {
            return result;
        }

        nb::module_ props_module = nb::module_::import_("lfs_plugins.props");

        for (const auto& meta : group->properties) {
            nb::object prop_obj;

            switch (meta.type) {
            case PropType::Float: {
                nb::object cls = props_module.attr("FloatProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<float>(meta.default_value),
                    nb::arg("min") = static_cast<float>(meta.min_value),
                    nb::arg("max") = static_cast<float>(meta.max_value),
                    nb::arg("step") = static_cast<float>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Int: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(meta.default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value),
                    nb::arg("max") = static_cast<int>(meta.max_value),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::SizeT: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(meta.default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value),
                    nb::arg("max") = static_cast<int>(meta.max_value),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Bool: {
                nb::object cls = props_module.attr("BoolProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_value != 0.0,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::String: {
                nb::object cls = props_module.attr("StringProperty");
                prop_obj = cls(
                    nb::arg("default") = meta.default_string,
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            default:
                continue;
            }

            result[meta.id.c_str()] = prop_obj;
        }

        return result;
    }

    void register_params(nb::module_& m) {
        register_optimization_properties();
        register_dataset_properties();

        nb::enum_<MaskMode>(m, "MaskMode")
            .value("NONE", MaskMode::None)
            .value("SEGMENT", MaskMode::Segment)
            .value("IGNORE", MaskMode::Ignore)
            .value("ALPHA_CONSISTENT", MaskMode::AlphaConsistent);

        nb::enum_<BackgroundMode>(m, "BackgroundMode")
            .value("SOLID_COLOR", BackgroundMode::SolidColor)
            .value("MODULATION", BackgroundMode::Modulation)
            .value("IMAGE", BackgroundMode::Image)
            .value("RANDOM", BackgroundMode::Random);

        nb::class_<PyOptimizationParams>(m, "OptimizationParams")
            .def(nb::init<>())
            .def_prop_ro(
                "__property_group__", [](PyOptimizationParams&) { return "optimization"; }, "Property group identifier")
            .def("get", &PyOptimizationParams::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyOptimizationParams::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("__getattr__", &PyOptimizationParams::get, nb::arg("name"), "Get property value by attribute name")
            .def("prop_info", &PyOptimizationParams::prop_info, nb::arg("prop_id"),
                 "Get metadata for a property")
            .def("reset", &PyOptimizationParams::reset, nb::arg("prop_id"),
                 "Reset property to default value")
            .def("properties", &PyOptimizationParams::properties,
                 "List all properties with their current values")
            .def("get_all_properties", &PyOptimizationParams::get_all_properties,
                 "Get all property descriptors as Python Property objects")
            .def("has_params", &PyOptimizationParams::has_params,
                 "Check if ParameterManager is available")
            .def_prop_rw(
                "iterations",
                [](PyOptimizationParams& self) { return self.params().iterations; },
                [](PyOptimizationParams& self, size_t v) { self.params().iterations = v; },
                "Maximum training iterations")
            .def_prop_rw(
                "means_lr",
                [](PyOptimizationParams& self) { return self.params().means_lr; },
                [](PyOptimizationParams& self, float v) { self.params().means_lr = v; },
                "Learning rate for gaussian positions")
            .def_prop_rw(
                "shs_lr",
                [](PyOptimizationParams& self) { return self.params().shs_lr; },
                [](PyOptimizationParams& self, float v) { self.params().shs_lr = v; },
                "Learning rate for spherical harmonics")
            .def_prop_rw(
                "opacity_lr",
                [](PyOptimizationParams& self) { return self.params().opacity_lr; },
                [](PyOptimizationParams& self, float v) { self.params().opacity_lr = v; },
                "Learning rate for opacity")
            .def_prop_rw(
                "scaling_lr",
                [](PyOptimizationParams& self) { return self.params().scaling_lr; },
                [](PyOptimizationParams& self, float v) { self.params().scaling_lr = v; },
                "Learning rate for gaussian scales")
            .def_prop_rw(
                "rotation_lr",
                [](PyOptimizationParams& self) { return self.params().rotation_lr; },
                [](PyOptimizationParams& self, float v) { self.params().rotation_lr = v; },
                "Learning rate for rotations")
            .def_prop_rw(
                "lambda_dssim",
                [](PyOptimizationParams& self) { return self.params().lambda_dssim; },
                [](PyOptimizationParams& self, float v) { self.params().lambda_dssim = v; },
                "Weight for structural similarity loss")
            .def_prop_rw(
                "sh_degree",
                [](PyOptimizationParams& self) { return self.params().sh_degree; },
                [](PyOptimizationParams& self, int v) { self.params().sh_degree = v; },
                "Spherical harmonics degree (0-3)")
            .def_prop_rw(
                "max_cap",
                [](PyOptimizationParams& self) { return self.params().max_cap; },
                [](PyOptimizationParams& self, int v) { self.params().max_cap = v; },
                "Maximum number of gaussians")
            .def_prop_ro(
                "strategy", [](PyOptimizationParams& self) { return self.params().strategy; },
                "Active optimization strategy name")
            .def(
                "set_strategy",
                [](PyOptimizationParams& /*self*/, const std::string& strategy) {
                    if (strategy != "mcmc" && strategy != "adc") {
                        throw std::invalid_argument("Strategy must be 'mcmc' or 'adc'");
                    }
                    auto* pm = get_parameter_manager();
                    if (pm) {
                        pm->setActiveStrategy(strategy);
                    }
                },
                nb::arg("strategy"),
                "Set active strategy ('mcmc' or 'adc')")
            .def_prop_ro(
                "headless", [](PyOptimizationParams& self) { return self.params().headless; },
                "Whether running without visualization")
            .def_prop_rw(
                "tile_mode",
                [](PyOptimizationParams& self) { return self.params().tile_mode; },
                [](PyOptimizationParams& self, int v) { self.params().tile_mode = v; },
                "Tile mode (1, 2, or 4)")
            .def_prop_rw(
                "steps_scaler",
                [](PyOptimizationParams& self) { return self.params().steps_scaler; },
                [](PyOptimizationParams& self, float v) { self.params().steps_scaler = v; },
                "Scale factor for training step counts")
            .def(
                "apply_step_scaling",
                [](PyOptimizationParams& self, float new_scaler) {
                    auto& opt = self.params();
                    new_scaler = std::max(0.0f, new_scaler);
                    const float prev = opt.steps_scaler;
                    opt.steps_scaler = new_scaler;
                    if (new_scaler <= 0.0f)
                        return;

                    const float ratio = (prev > 0.0f) ? (new_scaler / prev) : new_scaler;
                    const auto scale = [ratio](const size_t v) {
                        return static_cast<size_t>(std::lround(static_cast<float>(v) * ratio));
                    };
                    opt.iterations = scale(opt.iterations);
                    opt.start_refine = scale(opt.start_refine);
                    opt.reset_every = scale(opt.reset_every);
                    opt.stop_refine = scale(opt.stop_refine);
                    opt.refine_every = scale(opt.refine_every);
                    opt.sh_degree_interval = scale(opt.sh_degree_interval);

                    auto scale_vec = [ratio](std::vector<size_t>& steps) {
                        std::set<size_t> unique;
                        for (const auto& s : steps) {
                            size_t scaled = static_cast<size_t>(std::lround(static_cast<float>(s) * ratio));
                            if (scaled > 0)
                                unique.insert(scaled);
                        }
                        steps.assign(unique.begin(), unique.end());
                    };
                    scale_vec(opt.eval_steps);
                    scale_vec(opt.save_steps);
                },
                nb::arg("new_scaler"),
                "Set steps_scaler and scale all step-related parameters by the ratio")
            .def_prop_rw(
                "gut",
                [](PyOptimizationParams& self) { return self.params().gut; },
                [](PyOptimizationParams& self, bool v) { self.params().gut = v; },
                "Enable Gaussian Unscented Transform")
            .def_prop_rw(
                "use_bilateral_grid",
                [](PyOptimizationParams& self) { return self.params().use_bilateral_grid; },
                [](PyOptimizationParams& self, bool v) { self.params().use_bilateral_grid = v; },
                "Enable bilateral grid color correction")
            .def_prop_rw(
                "enable_sparsity",
                [](PyOptimizationParams& self) { return self.params().enable_sparsity; },
                [](PyOptimizationParams& self, bool v) { self.params().enable_sparsity = v; },
                "Enable sparsity optimization")
            .def_prop_rw(
                "mip_filter",
                [](PyOptimizationParams& self) { return self.params().mip_filter; },
                [](PyOptimizationParams& self, bool v) { self.params().mip_filter = v; },
                "Enable mip filtering (anti-aliasing)")
            .def_prop_rw(
                "ppisp",
                [](PyOptimizationParams& self) { return self.params().use_ppisp; },
                [](PyOptimizationParams& self, bool v) { self.params().use_ppisp = v; },
                "Enable per-pixel image signal processing")
            .def_prop_rw(
                "bg_mode",
                [](PyOptimizationParams& self) { return self.params().bg_mode; },
                [](PyOptimizationParams& self, BackgroundMode v) { self.params().bg_mode = v; },
                "Background rendering mode")
            .def_prop_rw(
                "bg_color",
                [](PyOptimizationParams& self) {
                    auto& c = self.params().bg_color;
                    return std::make_tuple(c[0], c[1], c[2]);
                },
                [](PyOptimizationParams& self, std::tuple<float, float, float> v) {
                    self.params().bg_color = {std::get<0>(v), std::get<1>(v), std::get<2>(v)};
                },
                "Background color as (r, g, b) tuple")
            .def_prop_rw(
                "bg_image_path",
                [](PyOptimizationParams& self) { return self.params().bg_image_path.string(); },
                [](PyOptimizationParams& self, const std::string& v) { self.params().bg_image_path = v; },
                "Path to background image")
            .def_prop_rw(
                "random",
                [](PyOptimizationParams& self) { return self.params().random; },
                [](PyOptimizationParams& self, bool v) { self.params().random = v; },
                "Use random initialization instead of SfM")
            .def_prop_rw(
                "mask_mode",
                [](PyOptimizationParams& self) { return self.params().mask_mode; },
                [](PyOptimizationParams& self, MaskMode v) { self.params().mask_mode = v; },
                "Attention mask behavior during training")
            .def_prop_rw(
                "invert_masks",
                [](PyOptimizationParams& self) { return self.params().invert_masks; },
                [](PyOptimizationParams& self, bool v) { self.params().invert_masks = v; },
                "Swap object and background in masks")
            .def_prop_rw(
                "use_alpha_as_mask",
                [](PyOptimizationParams& self) { return self.params().use_alpha_as_mask; },
                [](PyOptimizationParams& self, bool v) { self.params().use_alpha_as_mask = v; },
                "Use alpha channel from RGBA images as mask source")
            .def_prop_rw(
                "undistort",
                [](PyOptimizationParams& self) { return self.params().undistort; },
                [](PyOptimizationParams& self, bool v) { self.params().undistort = v; },
                "Undistort images on-the-fly before training")
            .def_prop_ro(
                "save_steps",
                [](PyOptimizationParams& self) -> std::vector<size_t> {
                    return self.params().save_steps;
                },
                "List of iterations at which to save checkpoints")
            .def(
                "add_save_step",
                [](PyOptimizationParams& self, size_t step) {
                    auto& steps = self.params().save_steps;
                    if (std::find(steps.begin(), steps.end(), step) == steps.end()) {
                        steps.push_back(step);
                        std::sort(steps.begin(), steps.end());
                    }
                },
                nb::arg("step"),
                "Add a save step (ignored if duplicate)")
            .def(
                "remove_save_step",
                [](PyOptimizationParams& self, size_t step) {
                    auto& steps = self.params().save_steps;
                    steps.erase(std::remove(steps.begin(), steps.end(), step), steps.end());
                },
                nb::arg("step"),
                "Remove a save step")
            .def(
                "clear_save_steps",
                [](PyOptimizationParams& self) {
                    self.params().save_steps.clear();
                },
                "Clear all save steps");

        m.def(
            "optimization_params", []() { return PyOptimizationParams{}; },
            "Get the optimization parameters object");

        nb::class_<PyDatasetConfig>(m, "DatasetParams")
            .def(nb::init<>())
            .def_prop_ro(
                "__property_group__", [](PyDatasetConfig&) { return "dataset"; }, "Property group identifier")
            .def("get", &PyDatasetConfig::get, nb::arg("name"), "Get property value by name")
            .def("set", &PyDatasetConfig::set, nb::arg("name"), nb::arg("value"), "Set property value by name")
            .def("prop_info", &PyDatasetConfig::prop_info, nb::arg("prop_id"), "Get metadata for a property")
            .def("properties", &PyDatasetConfig::properties, "List all properties with their current values")
            .def("get_all_properties", &PyDatasetConfig::get_all_properties,
                 "Get all property descriptors as Python Property objects")
            .def("has_params", &PyDatasetConfig::has_params,
                 "Check if TrainerManager is available")
            .def("can_edit", &PyDatasetConfig::can_edit,
                 "Check if dataset params can be edited (before training starts)")
            .def_prop_ro(
                "data_path", [](const PyDatasetConfig& self) { return self.params().data_path.string(); },
                "Path to training data directory")
            .def_prop_ro(
                "output_path", [](const PyDatasetConfig& self) { return self.params().output_path.string(); },
                "Path for output files")
            .def_prop_ro(
                "images", [](const PyDatasetConfig& self) { return self.params().images; },
                "Subfolder name containing images")
            .def_prop_ro(
                "test_every", [](const PyDatasetConfig& self) { return self.params().test_every; },
                "Use every Nth image for testing")
            .def_prop_rw(
                "resize_factor",
                [](const PyDatasetConfig& self) { return self.params().resize_factor; },
                [](PyDatasetConfig& self, int v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    self.params().resize_factor = v;
                },
                "Image resize factor (-1 = auto)")
            .def_prop_rw(
                "max_width",
                [](const PyDatasetConfig& self) { return self.params().max_width; },
                [](PyDatasetConfig& self, int v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    if (v <= 0 || v > 4096)
                        throw std::invalid_argument("max_width must be between 1 and 4096");
                    self.params().max_width = v;
                },
                "Maximum image width in pixels")
            .def_prop_rw(
                "use_cpu_cache",
                [](const PyDatasetConfig& self) { return self.params().loading_params.use_cpu_memory; },
                [](PyDatasetConfig& self, bool v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    self.params().loading_params.use_cpu_memory = v;
                },
                "Cache images in CPU memory")
            .def_prop_rw(
                "use_fs_cache",
                [](const PyDatasetConfig& self) { return self.params().loading_params.use_fs_cache; },
                [](PyDatasetConfig& self, bool v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    self.params().loading_params.use_fs_cache = v;
                },
                "Use filesystem cache for images");

        m.def(
            "dataset_params", []() { return PyDatasetConfig{}; },
            "Get the dataset parameters object");

        // Property change callback
        m.def(
            "on_property_change",
            [](const std::string& property_path, nb::callable callback) {
                // Parse property_path like "optimization.means_lr"
                auto dot_pos = property_path.find('.');
                if (dot_pos == std::string::npos) {
                    throw std::runtime_error("Invalid property path. Use 'group.property' format");
                }
                std::string group_id = property_path.substr(0, dot_pos);
                std::string prop_id = property_path.substr(dot_pos + 1);

                // Wrap Python callback
                nb::object cb_obj = nb::cast<nb::object>(callback);
                auto cpp_callback = [cb_obj](const std::string& /*group*/,
                                             const std::string& /*prop*/,
                                             const std::any& old_val,
                                             const std::any& new_val) {
                    nb::gil_scoped_acquire gil;
                    try {
                        // Convert std::any to Python objects
                        nb::object py_old, py_new;
                        if (old_val.type() == typeid(float)) {
                            py_old = nb::cast(std::any_cast<float>(old_val));
                            py_new = nb::cast(std::any_cast<float>(new_val));
                        } else if (old_val.type() == typeid(int)) {
                            py_old = nb::cast(std::any_cast<int>(old_val));
                            py_new = nb::cast(std::any_cast<int>(new_val));
                        } else if (old_val.type() == typeid(bool)) {
                            py_old = nb::cast(std::any_cast<bool>(old_val));
                            py_new = nb::cast(std::any_cast<bool>(new_val));
                        } else if (old_val.type() == typeid(size_t)) {
                            py_old = nb::cast(std::any_cast<size_t>(old_val));
                            py_new = nb::cast(std::any_cast<size_t>(new_val));
                        } else if (old_val.type() == typeid(std::string)) {
                            py_old = nb::cast(std::any_cast<std::string>(old_val));
                            py_new = nb::cast(std::any_cast<std::string>(new_val));
                        } else {
                            py_old = nb::none();
                            py_new = nb::none();
                        }
                        cb_obj(py_old, py_new);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Property change callback error: {}", e.what());
                    }
                };

                size_t sub_id = PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                return sub_id;
            },
            nb::arg("property_path"), nb::arg("callback"),
            "Register a callback for property changes. Returns subscription ID.\n"
            "Usage: lf.on_property_change('optimization.means_lr', lambda old, new: print(f'{old} -> {new}'))");

        m.def(
            "unsubscribe_property_change",
            [](size_t subscription_id) {
                PropertyRegistry::instance().unsubscribe(subscription_id);
            },
            nb::arg("subscription_id"),
            "Unsubscribe from property change notifications");

        // Decorator-style callback registration
        m.def(
            "property_callback",
            [](const std::string& property_path) {
                return nb::cpp_function([property_path](nb::object func) {
                    auto dot_pos = property_path.find('.');
                    if (dot_pos == std::string::npos) {
                        throw std::runtime_error("Invalid property path. Use 'group.property' format");
                    }
                    std::string group_id = property_path.substr(0, dot_pos);
                    std::string prop_id = property_path.substr(dot_pos + 1);

                    nb::object cb_obj = func;
                    auto cpp_callback = [cb_obj](const std::string&, const std::string&,
                                                 const std::any& old_val, const std::any& new_val) {
                        nb::gil_scoped_acquire gil;
                        try {
                            nb::object py_old, py_new;
                            if (old_val.type() == typeid(float)) {
                                py_old = nb::cast(std::any_cast<float>(old_val));
                                py_new = nb::cast(std::any_cast<float>(new_val));
                            } else if (old_val.type() == typeid(int)) {
                                py_old = nb::cast(std::any_cast<int>(old_val));
                                py_new = nb::cast(std::any_cast<int>(new_val));
                            } else if (old_val.type() == typeid(bool)) {
                                py_old = nb::cast(std::any_cast<bool>(old_val));
                                py_new = nb::cast(std::any_cast<bool>(new_val));
                            } else if (old_val.type() == typeid(size_t)) {
                                py_old = nb::cast(std::any_cast<size_t>(old_val));
                                py_new = nb::cast(std::any_cast<size_t>(new_val));
                            } else if (old_val.type() == typeid(std::string)) {
                                py_old = nb::cast(std::any_cast<std::string>(old_val));
                                py_new = nb::cast(std::any_cast<std::string>(new_val));
                            } else {
                                py_old = nb::none();
                                py_new = nb::none();
                            }
                            cb_obj(py_old, py_new);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Property change callback error: {}", e.what());
                        }
                    };

                    PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                    return func;
                });
            },
            nb::arg("property_path"),
            "Decorator for property change handlers.\n"
            "Usage: @lf.property_callback('optimization.means_lr')\n"
            "       def on_lr_change(old_val, new_val): ...");
    }

} // namespace lfs::python
