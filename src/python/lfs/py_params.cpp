/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_params.hpp"

#include "control/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/logger.hpp"
#include "core/optimization_properties.hpp"
#include "core/path_utils.hpp"
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
#include <mutex>
#include <set>
#include <stdexcept>

namespace lfs::python {

    using namespace lfs::core::param;
    using namespace lfs::core::prop;
    using lfs::training::CommandCenter;

    std::any resolve_optimization_default(
        const PropertyMeta& meta,
        const OptimizationParameters& source) {
        if (!meta.getter)
            throw std::runtime_error("Optimization property has no getter: " + meta.id);
        auto ref = PropertyObjectRef::cpp(const_cast<OptimizationParameters*>(&source));
        return meta.getter(ref);
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
            meta.default_value = default_val;
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
            meta.default_value = static_cast<int64_t>(default_val);
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
            meta.default_value = default_val;
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
            [](const DatasetConfig& c) { return lfs::core::path_to_utf8(c.data_path); });

        add_string(
            "output_path", "Output Path", "", "Path for output files", true,
            [](const DatasetConfig& c) { return lfs::core::path_to_utf8(c.output_path); });

        add_string(
            "images", "Images Folder", "images", "Subfolder containing images", true,
            [](const DatasetConfig& c) { return c.images; });

        add_int(
            "resize_factor", "Resize Factor", -1, -1, 8, "Image resize factor (-1 = auto)", false,
            [](const DatasetConfig& c) { return c.resize_factor; },
            [](DatasetConfig& c, int v) { c.resize_factor = v; });

        add_int(
            "test_every", "Test Every", 8, 1, 10000, "Use every Nth image for testing", true,
            [](const DatasetConfig& c) { return c.test_every; });

        add_int(
            "max_width", "Max Width", 3840, 0, 65535, "Maximum image width; 0 disables the cap", false,
            [](const DatasetConfig& c) { return c.max_width; },
            [](DatasetConfig& c, int v) { c.max_width = v; });

        add_int(
            "min_track_length", "Minimum Track Length", 0, 0, 65535,
            "Minimum COLMAP sparse point track length; 0 disables filtering", false,
            [](const DatasetConfig& c) { return c.min_track_length; },
            [](DatasetConfig& c, int v) { c.min_track_length = v; });

        add_bool(
            "use_cpu_cache", "CPU Cache", true, "Cache images in CPU memory", false,
            [](const DatasetConfig& c) { return c.loading_params.use_cpu_memory; },
            [](DatasetConfig& c, bool v) { c.loading_params.use_cpu_memory = v; });

        add_bool(
            "use_16bit_color", "16-bit Color", false, "Train with 16-bit color images (HDR); caches losslessly as JPEG 2000", false,
            [](const DatasetConfig& c) { return c.loading_params.use_16bit_color; },
            [](DatasetConfig& c, bool v) { c.loading_params.use_16bit_color = v; });

        PropertyRegistry::instance().register_group(std::move(group));
    }

    namespace {
        std::mutex python_property_subscriptions_mutex;
        std::set<size_t> python_property_subscriptions;

        void track_python_property_subscription(const size_t id) {
            std::lock_guard lock(python_property_subscriptions_mutex);
            python_property_subscriptions.insert(id);
        }

        void forget_python_property_subscription(const size_t id) {
            std::lock_guard lock(python_property_subscriptions_mutex);
            python_property_subscriptions.erase(id);
        }

        void clear_python_property_subscriptions() {
            std::set<size_t> subscriptions;
            {
                std::lock_guard lock(python_property_subscriptions_mutex);
                subscriptions.swap(python_property_subscriptions);
            }
            for (const size_t id : subscriptions) {
                PropertyRegistry::instance().unsubscribe(id);
            }
        }

        core::param::OptimizationParameters& get_default_params() {
            static core::param::OptimizationParameters default_params =
                core::param::OptimizationParameters::mrnf_defaults();
            return default_params;
        }

        core::param::OptimizationParameters copy_optimization_default_source() {
            if (auto* pm = get_parameter_manager())
                return pm->copySessionParams();
            return core::param::OptimizationParameters::defaults_for_strategy(get_default_params().strategy);
        }

        template <typename F>
        void modify_params(F&& fn) {
            auto* pm = get_parameter_manager();
            if (!pm) {
                fn(get_default_params());
                return;
            }
            pm->modifyActiveParams(std::forward<F>(fn));
        }

        void set_active_strategy(const std::string_view strategy) {
            const auto canonical_strategy = core::param::canonical_strategy_name(strategy);
            if (canonical_strategy.empty())
                throw std::invalid_argument("Strategy must be 'mcmc', 'mrnf', or 'igs+'");

            if (auto* pm = get_parameter_manager()) {
                pm->modifyActiveParams([&](auto&) { pm->setActiveStrategy(canonical_strategy); });
            } else {
                get_default_params() = core::param::OptimizationParameters::defaults_for_strategy(canonical_strategy);
            }
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

        if (prop_id == "strategy") {
            const std::any old_value = params().strategy;
            set_active_strategy(nb::cast<std::string>(value));
            const std::any new_value = params().strategy;
            PropertyRegistry::instance().notify("optimization", prop_id, old_value, new_value);
            return;
        }

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

        std::any old_value;
        modify_params([&](auto& p) {
            auto ref = PropertyObjectRef::cpp(&p);
            old_value = meta->getter(ref);
            meta->setter(ref, new_value);
        });
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
        info["advanced"] = meta->is_advanced();
        info["locale_key"] = meta->ui_locale_key;
        info["tooltip_key"] = meta->ui_tooltip_key;
        info["step"] = meta->step;
        if (meta->ui_precision) {
            info["precision"] = *meta->ui_precision;
        }

        const auto default_source = copy_optimization_default_source();
        const std::any default_value = resolve_optimization_default(*meta, default_source);

        switch (meta->type) {
        case PropType::Float:
            info["type"] = "float";
            info["min"] = meta->min_value.value();
            info["max"] = meta->max_value.value();
            info["default"] = std::any_cast<float>(default_value);
            break;
        case PropType::Int:
            info["type"] = "int";
            info["min"] = static_cast<int>(meta->min_value.value());
            info["max"] = static_cast<int>(meta->max_value.value());
            info["default"] = std::any_cast<int>(default_value);
            break;
        case PropType::SizeT:
            info["type"] = "int";
            info["min"] = static_cast<int64_t>(meta->min_value.value());
            info["max"] = static_cast<int64_t>(meta->max_value.value());
            info["default"] = static_cast<int64_t>(std::any_cast<size_t>(default_value));
            break;
        case PropType::Bool:
            info["type"] = "bool";
            info["default"] = std::any_cast<bool>(default_value);
            break;
        case PropType::String:
            info["type"] = "string";
            info["default"] = std::any_cast<std::string>(default_value);
            break;
        case PropType::Enum:
            info["type"] = "enum";
            info["default"] = std::any_cast<int>(default_value);
            {
                nb::list items;
                for (const auto& ei : meta->enum_items) {
                    nb::dict item;
                    item["name"] = ei.name;
                    item["identifier"] = ei.identifier;
                    item["value"] = ei.value;
                    item["locale_key"] = ei.locale_key;
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
        if (meta->is_readonly()) {
            throw std::runtime_error("Property is read-only: " + prop_id);
        }

        const auto default_source = copy_optimization_default_source();
        const std::any default_val = resolve_optimization_default(*meta, default_source);
        std::any old_value;
        modify_params([&](auto& p) {
            auto ref = PropertyObjectRef::cpp(&p);
            old_value = meta->getter(ref);
            meta->setter(ref, default_val);
        });
        PropertyRegistry::instance().notify("optimization", prop_id, old_value, default_val);
    }

    nb::list PyOptimizationParams::properties() const {
        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
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
        const auto group = PropertyRegistry::instance().get_group_snapshot("optimization");
        if (!group) {
            return result;
        }

        nb::module_ props_module = nb::module_::import_("lfs_plugins.props");
        const auto default_source = copy_optimization_default_source();

        for (const auto& meta : group->properties) {
            nb::object prop_obj;
            const std::any default_value = resolve_optimization_default(meta, default_source);

            switch (meta.type) {
            case PropType::Float: {
                nb::object cls = props_module.attr("FloatProperty");
                prop_obj = cls(
                    nb::arg("default") = std::any_cast<float>(default_value),
                    nb::arg("min") = static_cast<float>(meta.min_value.value()),
                    nb::arg("max") = static_cast<float>(meta.max_value.value()),
                    nb::arg("step") = static_cast<float>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Int: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = std::any_cast<int>(default_value),
                    nb::arg("min") = static_cast<int>(meta.min_value.value()),
                    nb::arg("max") = static_cast<int>(meta.max_value.value()),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::SizeT: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(std::any_cast<size_t>(default_value)),
                    nb::arg("min") = static_cast<int>(meta.min_value.value()),
                    nb::arg("max") = static_cast<int>(meta.max_value.value()),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Bool: {
                nb::object cls = props_module.attr("BoolProperty");
                prop_obj = cls(
                    nb::arg("default") = std::any_cast<bool>(default_value),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::String: {
                nb::object cls = props_module.attr("StringProperty");
                prop_obj = cls(
                    nb::arg("default") = std::any_cast<std::string>(default_value),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Enum: {
                nb::object cls = props_module.attr("EnumProperty");
                nb::list items;
                std::string default_id;
                const int default_enum = std::any_cast<int>(default_value);
                for (const auto& item : meta.enum_items) {
                    items.append(nb::make_tuple(item.identifier, item.name, ""));
                    if (item.value == default_enum) {
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

    core::param::DatasetConfig PyDatasetConfig::params() const {
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
            info["min"] = meta->min_value.value();
            info["max"] = meta->max_value.value();
            info["default"] = std::get<double>(meta->default_value.value());
            break;
        case PropType::Int:
            info["type"] = "int";
            info["min"] = static_cast<int>(meta->min_value.value());
            info["max"] = static_cast<int>(meta->max_value.value());
            info["default"] = static_cast<int>(std::get<int64_t>(meta->default_value.value()));
            break;
        case PropType::SizeT:
            info["type"] = "int";
            info["min"] = static_cast<int64_t>(meta->min_value.value());
            info["max"] = static_cast<int64_t>(meta->max_value.value());
            info["default"] = std::get<int64_t>(meta->default_value.value());
            break;
        case PropType::Bool:
            info["type"] = "bool";
            info["default"] = std::get<bool>(meta->default_value.value());
            break;
        case PropType::String:
            info["type"] = "string";
            info["default"] = std::get<std::string>(meta->default_value.value());
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
                    nb::arg("default") = static_cast<float>(std::get<double>(meta.default_value.value())),
                    nb::arg("min") = static_cast<float>(meta.min_value.value()),
                    nb::arg("max") = static_cast<float>(meta.max_value.value()),
                    nb::arg("step") = static_cast<float>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Int: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(std::get<int64_t>(meta.default_value.value())),
                    nb::arg("min") = static_cast<int>(meta.min_value.value()),
                    nb::arg("max") = static_cast<int>(meta.max_value.value()),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::SizeT: {
                nb::object cls = props_module.attr("IntProperty");
                prop_obj = cls(
                    nb::arg("default") = static_cast<int>(std::get<int64_t>(meta.default_value.value())),
                    nb::arg("min") = static_cast<int>(meta.min_value.value()),
                    nb::arg("max") = static_cast<int>(meta.max_value.value()),
                    nb::arg("step") = static_cast<int>(meta.step),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::Bool: {
                nb::object cls = props_module.attr("BoolProperty");
                prop_obj = cls(
                    nb::arg("default") = std::get<bool>(meta.default_value.value()),
                    nb::arg("name") = meta.name,
                    nb::arg("description") = meta.description);
                break;
            }
            case PropType::String: {
                nb::object cls = props_module.attr("StringProperty");
                prop_obj = cls(
                    nb::arg("default") = std::get<std::string>(meta.default_value.value()),
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
        core::param::ensure_optimization_properties_registered();
        register_dataset_properties();

        nb::enum_<MaskMode>(m, "MaskMode")
            .value("NONE", MaskMode::None)
            .value("SEGMENT", MaskMode::Segment)
            .value("IGNORE", MaskMode::Ignore)
            .value("SEGMENT_AND_IGNORE", MaskMode::SegmentAndIgnore)
            .value("ALPHA_CONSISTENT", MaskMode::AlphaConsistent);

        nb::enum_<NormalLossSpace>(m, "NormalLossSpace")
            .value("AUTO", NormalLossSpace::Auto)
            .value("CAMERA_OPENCV", NormalLossSpace::CameraOpenCV)
            .value("CAMERA_OPENGL", NormalLossSpace::CameraOpenGL)
            .value("WORLD", NormalLossSpace::World);

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
            .def(
                "validate", [](PyOptimizationParams& self) { return self.params().validate(); },
                "Validate parameter consistency, returns empty string if valid")
            .def_prop_rw(
                "iterations",
                [](PyOptimizationParams& self) { return self.params().iterations; },
                [](PyOptimizationParams&, size_t v) { modify_params([v](auto& p) { p.iterations = v; }); },
                "Maximum training iterations")
            .def_prop_rw(
                "means_lr",
                [](PyOptimizationParams& self) { return self.params().means_lr; },
                [](PyOptimizationParams& self, float v) { self.set("means_lr", nb::cast(v)); },
                "Learning rate for gaussian positions")
            .def_prop_rw(
                "means_lr_end",
                [](PyOptimizationParams& self) { return self.params().means_lr_end; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.means_lr_end = v; }); },
                "Target end learning rate for gaussian positions")
            .def_prop_rw(
                "shs_lr",
                [](PyOptimizationParams& self) { return self.params().shs_lr; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.shs_lr = v; }); },
                "Learning rate for spherical harmonics")
            .def_prop_rw(
                "opacity_lr",
                [](PyOptimizationParams& self) { return self.params().opacity_lr; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.opacity_lr = v; }); },
                "Learning rate for opacity")
            .def_prop_rw(
                "scaling_lr",
                [](PyOptimizationParams& self) { return self.params().scaling_lr; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.scaling_lr = v; }); },
                "Learning rate for gaussian scales")
            .def_prop_rw(
                "scaling_lr_end",
                [](PyOptimizationParams& self) { return self.params().scaling_lr_end; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.scaling_lr_end = v; }); },
                "Target end learning rate for gaussian scales")
            .def_prop_rw(
                "rotation_lr",
                [](PyOptimizationParams& self) { return self.params().rotation_lr; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.rotation_lr = v; }); },
                "Learning rate for rotations")
            .def_prop_rw(
                "cropbox_lr_scale",
                [](PyOptimizationParams& self) { return self.params().cropbox_lr_scale; },
                [](PyOptimizationParams& self, float v) { self.set("cropbox_lr_scale", nb::cast(v)); },
                "Scales Adam steps and refinement signals for rejected splats; strategy noise, decay, and resets remain active")
            .def_prop_rw(
                "cropbox_loss_weight",
                [](PyOptimizationParams& self) { return self.params().cropbox_loss_weight; },
                [](PyOptimizationParams& self, float v) { self.set("cropbox_loss_weight", nb::cast(v)); },
                "Scales pixel losses for camera rays outside the active crop box")
            .def_prop_rw(
                "lambda_dssim",
                [](PyOptimizationParams& self) { return self.params().lambda_dssim; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.lambda_dssim = v; }); },
                "Weight for structural similarity loss")
            .def_prop_rw(
                "sh_degree",
                [](PyOptimizationParams& self) { return self.params().sh_degree; },
                [](PyOptimizationParams&, int v) { modify_params([v](auto& p) { p.sh_degree = v; }); },
                "Spherical harmonics degree (0-3)")
            .def_prop_rw(
                "max_cap",
                [](PyOptimizationParams& self) { return self.params().max_cap; },
                [](PyOptimizationParams&, int v) { modify_params([v](auto& p) { p.max_cap = v; }); },
                "Maximum number of gaussians")
            .def_prop_ro(
                "strategy", [](PyOptimizationParams& self) { return self.params().strategy; },
                "Active optimization strategy name")
            .def(
                "set_strategy",
                [](PyOptimizationParams& /*self*/, const std::string& strategy) { set_active_strategy(strategy); },
                nb::arg("strategy"),
                "Set active strategy ('mcmc', 'mrnf', or 'igs+')")
            .def_prop_ro(
                "headless", [](PyOptimizationParams& self) { return self.params().headless; },
                "Whether running without visualization")
            .def_prop_rw(
                "enable_eval",
                [](PyOptimizationParams& self) { return self.params().enable_eval; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.enable_eval = v; }); },
                "Enable evaluation during training")
            .def_prop_rw(
                "steps_scaler",
                [](PyOptimizationParams& self) { return self.params().steps_scaler; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.steps_scaler = v; }); },
                "Scale factor for training step counts")
            .def(
                "apply_step_scaling",
                [](PyOptimizationParams&, float new_scaler) {
                    modify_params([new_scaler](auto& opt) {
                        const float clamped = std::max(0.0f, new_scaler);
                        const float prev = opt.steps_scaler;
                        opt.steps_scaler = clamped;
                        if (clamped <= 0.0f)
                            return;

                        const float ratio = (prev > 0.0f) ? (clamped / prev) : clamped;
                        opt.scale_steps(ratio);
                    });
                },
                nb::arg("new_scaler"),
                "Set steps_scaler and scale all step-related parameters by the ratio")
            .def(
                "auto_scale_steps",
                [](PyOptimizationParams&, const size_t image_count) {
                    if (auto* pm = get_parameter_manager())
                        pm->autoScaleSteps(image_count);
                },
                nb::arg("image_count"),
                "Auto-scale steps for all strategies based on image count")
            .def_prop_rw(
                "gut",
                [](PyOptimizationParams& self) { return self.params().gut; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.gut = v; }); },
                "Enable Gaussian Unscented Transform")
            .def_prop_rw(
                "use_bilateral_grid",
                [](PyOptimizationParams& self) { return self.params().use_bilateral_grid; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.use_bilateral_grid = v; }); },
                "Enable bilateral grid color correction")
            .def_prop_rw(
                "enable_sparsity",
                [](PyOptimizationParams& self) { return self.params().enable_sparsity; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.enable_sparsity = v; }); },
                "Enable sparsity optimization")
            .def_prop_rw(
                "mip_filter",
                [](PyOptimizationParams& self) { return self.params().mip_filter; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.mip_filter = v; }); },
                "Enable mip filtering (anti-aliasing)")
            .def_prop_rw(
                "ppisp",
                [](PyOptimizationParams& self) { return self.params().use_ppisp; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.use_ppisp = v; }); },
                "Enable per-pixel image signal processing")
            .def_prop_rw(
                "ppisp_use_controller",
                [](PyOptimizationParams& self) { return self.params().ppisp_use_controller; },
                [](PyOptimizationParams& self, bool v) { self.params().ppisp_use_controller = v; },
                "Enable PPISP controller for novel view synthesis")
            .def_prop_rw(
                "ppisp_freeze_from_sidecar",
                [](PyOptimizationParams& self) { return self.params().ppisp_freeze_from_sidecar; },
                [](PyOptimizationParams& self, bool v) { self.params().ppisp_freeze_from_sidecar = v; },
                "Freeze PPISP learning and reuse a PPISP sidecar during training")
            .def_prop_rw(
                "ppisp_sidecar_path",
                [](PyOptimizationParams& self) {
                    return lfs::core::path_to_utf8(self.params().ppisp_sidecar_path);
                },
                [](PyOptimizationParams& self, const std::string& v) {
                    self.params().ppisp_sidecar_path = lfs::core::utf8_to_path(v);
                },
                "Path to a PPISP sidecar used for frozen PPISP training")
            .def_prop_rw(
                "ppisp_controller_activation_step",
                [](PyOptimizationParams& self) { return self.params().ppisp_controller_activation_step; },
                [](PyOptimizationParams& self, int v) { self.params().ppisp_controller_activation_step = v; },
                "Iteration to start controller distillation (negative = default schedule)")
            .def_prop_rw(
                "ppisp_controller_lr",
                [](PyOptimizationParams& self) { return self.params().ppisp_controller_lr; },
                [](PyOptimizationParams& self, float v) { self.params().ppisp_controller_lr = v; },
                "Learning rate for PPISP controller")
            .def_prop_rw(
                "ppisp_freeze_gaussians",
                [](PyOptimizationParams& self) { return self.params().ppisp_freeze_gaussians_on_distill; },
                [](PyOptimizationParams& self, bool v) { self.params().ppisp_freeze_gaussians_on_distill = v; },
                "Freeze Gaussians during controller distillation")
            .def_prop_rw(
                "bg_mode",
                [](PyOptimizationParams& self) { return self.params().bg_mode; },
                [](PyOptimizationParams&, BackgroundMode v) { modify_params([v](auto& p) { p.bg_mode = v; }); },
                "Background rendering mode")
            .def_prop_rw(
                "bg_color",
                [](PyOptimizationParams& self) {
                    auto& c = self.params().bg_color;
                    return std::make_tuple(c[0], c[1], c[2]);
                },
                [](PyOptimizationParams&, std::tuple<float, float, float> v) {
                    modify_params([v](auto& p) { p.bg_color = {std::get<0>(v), std::get<1>(v), std::get<2>(v)}; });
                },
                "Background color as (r, g, b) tuple")
            .def_prop_rw(
                "bg_image_path",
                [](PyOptimizationParams& self) { return lfs::core::path_to_utf8(self.params().bg_image_path); },
                [](PyOptimizationParams&, const std::string& v) {
                    modify_params([&v](auto& p) { p.bg_image_path = lfs::core::utf8_to_path(v); });
                },
                "Path to background image")
            .def_prop_rw(
                "random",
                [](PyOptimizationParams& self) { return self.params().random; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.random = v; }); },
                "Use random initialization instead of SfM")
            .def_prop_rw(
                "mask_mode",
                [](PyOptimizationParams& self) { return self.params().mask_mode; },
                [](PyOptimizationParams&, MaskMode v) { modify_params([v](auto& p) { p.mask_mode = v; }); },
                "Attention mask behavior during training")
            .def_prop_rw(
                "invert_masks",
                [](PyOptimizationParams& self) { return self.params().invert_masks; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.invert_masks = v; }); },
                "Swap object and background in masks")
            .def_prop_rw(
                "use_alpha_as_mask",
                [](PyOptimizationParams& self) { return self.params().use_alpha_as_mask; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.use_alpha_as_mask = v; }); },
                "Use alpha channel from RGBA images as mask source")
            .def_prop_rw(
                "use_depth_loss",
                [](PyOptimizationParams& self) { return self.params().use_depth_loss; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.use_depth_loss = v; }); },
                "Load depth maps and use depth-map supervision during training")
            .def_prop_rw(
                "depth_loss_weight",
                [](PyOptimizationParams& self) { return self.params().depth_loss_weight; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.depth_loss_weight = std::max(0.0f, v); }); },
                "Weight for depth-map supervision")
            .def_prop_rw(
                "depth_loss_mode",
                [](PyOptimizationParams& self) { return self.params().depth_loss_mode; },
                [](PyOptimizationParams&, const std::string& v) { modify_params([v](auto& p) { p.depth_loss_mode = v; }); },
                "Depth prior convention: 'ssi' (auto-detect), 'ssi-disparity', or 'ssi-depth'")
            .def_prop_rw(
                "use_normal_loss",
                [](PyOptimizationParams& self) { return self.params().use_normal_loss; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.use_normal_loss = v; }); },
                "Load normal maps and use normal-map supervision during training")
            .def_prop_rw(
                "normal_loss_weight",
                [](PyOptimizationParams& self) { return self.params().normal_loss_weight; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.normal_loss_weight = std::max(0.0f, v); }); },
                "Weight for prior normal supervision")
            .def_prop_rw(
                "normal_consistency_weight",
                [](PyOptimizationParams& self) { return self.params().normal_consistency_weight; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.normal_consistency_weight = std::max(0.0f, v); }); },
                "Weight for depth-normal consistency supervision")
            .def_prop_rw(
                "normal_flatten_weight",
                [](PyOptimizationParams& self) { return self.params().normal_flatten_weight; },
                [](PyOptimizationParams&, float v) { modify_params([v](auto& p) { p.normal_flatten_weight = std::max(0.0f, v); }); },
                "Min-axis scale flattening weight while normal supervision is active")
            .def_prop_rw(
                "normal_loss_space",
                [](PyOptimizationParams& self) { return std::string(normal_loss_space_name(self.params().normal_loss_space)); },
                [](PyOptimizationParams&, const std::string& v) {
                    const auto parsed = normal_loss_space_from_string(v);
                    if (!parsed) {
                        throw std::invalid_argument(
                            "normal_loss_space must be 'auto', 'camera-opencv', 'camera-opengl', or 'world'");
                    }
                    modify_params([value = *parsed](auto& p) { p.normal_loss_space = value; });
                },
                "Normal prior coordinate space: 'auto', 'camera-opencv', 'camera-opengl', or 'world'")
            .def_prop_rw(
                "undistort",
                [](PyOptimizationParams& self) { return self.params().undistort; },
                [](PyOptimizationParams&, bool v) { modify_params([v](auto& p) { p.undistort = v; }); },
                "Undistort images on-the-fly before training")
            .def_prop_ro(
                "save_steps",
                [](PyOptimizationParams& self) -> std::vector<size_t> {
                    return self.params().save_steps;
                },
                "List of iterations at which to save the project (project.licht)")
            .def(
                "add_save_step",
                [](PyOptimizationParams&, size_t step) {
                    modify_params([step](auto& p) {
                        if (std::find(p.save_steps.begin(), p.save_steps.end(), step) == p.save_steps.end()) {
                            p.save_steps.push_back(step);
                            std::sort(p.save_steps.begin(), p.save_steps.end());
                        }
                    });
                },
                nb::arg("step"),
                "Add a save step (ignored if duplicate)")
            .def(
                "remove_save_step",
                [](PyOptimizationParams&, size_t step) {
                    modify_params([step](auto& p) {
                        p.save_steps.erase(std::remove(p.save_steps.begin(), p.save_steps.end(), step), p.save_steps.end());
                    });
                },
                nb::arg("step"),
                "Remove a save step")
            .def(
                "clear_save_steps",
                [](PyOptimizationParams&) {
                    modify_params([](auto& p) { p.save_steps.clear(); });
                },
                "Clear all save steps")
            .def_prop_ro(
                "eval_steps",
                [](PyOptimizationParams& self) -> std::vector<size_t> {
                    return self.params().eval_steps;
                },
                "List of iterations at which to run evaluation")
            .def(
                "add_eval_step",
                [](PyOptimizationParams&, size_t step) {
                    modify_params([step](auto& p) {
                        if (std::find(p.eval_steps.begin(), p.eval_steps.end(), step) == p.eval_steps.end()) {
                            p.eval_steps.push_back(step);
                            std::sort(p.eval_steps.begin(), p.eval_steps.end());
                        }
                    });
                },
                nb::arg("step"),
                "Add an eval step (ignored if duplicate)")
            .def(
                "remove_eval_step",
                [](PyOptimizationParams&, size_t step) {
                    modify_params([step](auto& p) {
                        p.eval_steps.erase(std::remove(p.eval_steps.begin(), p.eval_steps.end(), step), p.eval_steps.end());
                    });
                },
                nb::arg("step"),
                "Remove an eval step")
            .def(
                "clear_eval_steps",
                [](PyOptimizationParams&) {
                    modify_params([](auto& p) { p.eval_steps.clear(); });
                },
                "Clear all eval steps");

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
                "data_path", [](const PyDatasetConfig& self) { return lfs::core::path_to_utf8(self.params().data_path); },
                "Path to training data directory")
            .def_prop_ro(
                "output_path", [](const PyDatasetConfig& self) { return lfs::core::path_to_utf8(self.params().output_path); },
                "Path for output files")
            .def_prop_ro(
                "images", [](const PyDatasetConfig& self) { return self.params().images; },
                "Subfolder name containing images")
            .def_prop_rw(
                "test_every",
                [](const PyDatasetConfig& self) { return self.params().test_every; },
                [](PyDatasetConfig& self, int v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    if (v < 1)
                        throw std::invalid_argument("test_every must be at least 1");
                    self.params().test_every = v;
                },
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
                    if (v < 0)
                        throw std::invalid_argument("max_width must be non-negative; 0 disables the cap");
                    self.params().max_width = v;
                },
                "Maximum image width in pixels; 0 disables the cap")
            .def_prop_rw(
                "min_track_length",
                [](const PyDatasetConfig& self) { return self.params().min_track_length; },
                [](PyDatasetConfig& self, int v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    if (v < 0)
                        throw std::invalid_argument("min_track_length must be non-negative; 0 disables filtering");
                    self.params().min_track_length = v;
                },
                "Minimum COLMAP sparse point track length; 0 disables filtering")
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
                "use_16bit_color",
                [](const PyDatasetConfig& self) { return self.params().loading_params.use_16bit_color; },
                [](PyDatasetConfig& self, bool v) {
                    if (!self.can_edit())
                        throw std::runtime_error("Cannot edit dataset params during training");
                    self.params().loading_params.use_16bit_color = v;
                },
                "Train with 16-bit color images (HDR); caches losslessly as JPEG 2000")
            .def_prop_ro(
                "centralize_dataset",
                [](const PyDatasetConfig& self) { return self.params().centralize_dataset; },
                "Dataset centralization mode used for the last load: 'off', 'by_pointcloud', 'by_cameras'");

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

                const size_t sub_id = PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                track_python_property_subscription(sub_id);
                return sub_id;
            },
            nb::arg("property_path"), nb::arg("callback"),
            "Register a callback for property changes. Returns subscription ID.\n"
            "Usage: lf.on_property_change('optimization.means_lr', lambda old, new: print(f'{old} -> {new}'))");

        m.def(
            "unsubscribe_property_change",
            [](size_t subscription_id) {
                PropertyRegistry::instance().unsubscribe(subscription_id);
                forget_python_property_subscription(subscription_id);
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

                    const size_t sub_id = PropertyRegistry::instance().subscribe(group_id, prop_id, cpp_callback);
                    track_python_property_subscription(sub_id);
                    return func;
                });
            },
            nb::arg("property_path"),
            "Decorator for property change handlers.\n"
            "Usage: @lf.property_callback('optimization.means_lr')\n"
            "       def on_lr_change(old_val, new_val): ...");

        m.def("_clear_property_callbacks", &clear_python_property_subscriptions);
        nb::module_::import_("atexit").attr("register")(m.attr("_clear_property_callbacks"));
    }

} // namespace lfs::python
