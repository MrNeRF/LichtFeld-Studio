/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_prop_registry.hpp"
#include "core/logger.hpp"
#include "core/property_registry.hpp"

#include <cassert>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {

        core::prop::PropType infer_prop_type(nb::object descriptor) {
            // Check the Python descriptor class name to determine type
            if (!descriptor.is_valid()) {
                return core::prop::PropType::Float;
            }

            nb::object cls = descriptor.attr("__class__");
            std::string cls_name = nb::cast<std::string>(cls.attr("__name__"));

            if (cls_name == "FloatProperty") {
                return core::prop::PropType::Float;
            } else if (cls_name == "IntProperty") {
                return core::prop::PropType::Int;
            } else if (cls_name == "BoolProperty") {
                return core::prop::PropType::Bool;
            } else if (cls_name == "StringProperty") {
                return core::prop::PropType::String;
            } else if (cls_name == "EnumProperty") {
                return core::prop::PropType::Enum;
            } else if (cls_name == "FloatVectorProperty") {
                // Check subtype first for colors
                if (nb::hasattr(descriptor, "subtype")) {
                    std::string subtype = nb::cast<std::string>(descriptor.attr("subtype"));
                    if (subtype == "COLOR" || subtype == "COLOR_GAMMA") {
                        int size = 3;
                        if (nb::hasattr(descriptor, "size")) {
                            size = nb::cast<int>(descriptor.attr("size"));
                        }
                        return (size == 4) ? core::prop::PropType::Color4 : core::prop::PropType::Color3;
                    }
                }
                // Fall back to Vec types based on size
                if (nb::hasattr(descriptor, "size")) {
                    int size = nb::cast<int>(descriptor.attr("size"));
                    if (size == 2)
                        return core::prop::PropType::Vec2;
                    if (size == 4)
                        return core::prop::PropType::Vec4;
                }
                return core::prop::PropType::Vec3;
            } else if (cls_name == "TensorProperty") {
                return core::prop::PropType::Tensor;
            }

            return core::prop::PropType::Float;
        }

        core::prop::PropUIHint infer_ui_hint(nb::object descriptor) {
            if (!descriptor.is_valid()) {
                return core::prop::PropUIHint::Default;
            }

            // Check for subtype hint
            if (nb::hasattr(descriptor, "subtype")) {
                std::string subtype = nb::cast<std::string>(descriptor.attr("subtype"));
                if (subtype == "SLIDER") {
                    return core::prop::PropUIHint::Slider;
                }
            }

            return core::prop::PropUIHint::Default;
        }

    } // namespace

    core::prop::PropertyMeta python_property_to_meta(nb::object descriptor, const std::string& prop_id) {
        core::prop::PropertyMeta meta;
        meta.id = prop_id;
        meta.source = core::prop::PropSource::PYTHON;
        meta.type = infer_prop_type(descriptor);
        meta.ui_hint = infer_ui_hint(descriptor);

        // Extract common attributes
        if (nb::hasattr(descriptor, "name")) {
            meta.name = nb::cast<std::string>(descriptor.attr("name"));
        } else {
            meta.name = prop_id;
        }

        if (nb::hasattr(descriptor, "description")) {
            meta.description = nb::cast<std::string>(descriptor.attr("description"));
        }

        // Extract numeric constraints
        if (nb::hasattr(descriptor, "min")) {
            nb::object min_val = descriptor.attr("min");
            if (!min_val.is_none()) {
                meta.min_value = nb::cast<double>(min_val);
            }
        }
        if (nb::hasattr(descriptor, "max")) {
            nb::object max_val = descriptor.attr("max");
            if (!max_val.is_none()) {
                meta.max_value = nb::cast<double>(max_val);
            }
        }
        if (nb::hasattr(descriptor, "soft_min")) {
            nb::object soft_min = descriptor.attr("soft_min");
            if (!soft_min.is_none()) {
                meta.soft_min = nb::cast<double>(soft_min);
            }
        }
        if (nb::hasattr(descriptor, "soft_max")) {
            nb::object soft_max = descriptor.attr("soft_max");
            if (!soft_max.is_none()) {
                meta.soft_max = nb::cast<double>(soft_max);
            }
        }
        if (nb::hasattr(descriptor, "step")) {
            nb::object step = descriptor.attr("step");
            if (!step.is_none()) {
                meta.step = nb::cast<double>(step);
            }
        }

        // Extract default value
        if (nb::hasattr(descriptor, "default")) {
            nb::object default_val = descriptor.attr("default");
            if (!default_val.is_none()) {
                switch (meta.type) {
                case core::prop::PropType::Float:
                    meta.default_value = nb::cast<double>(default_val);
                    break;
                case core::prop::PropType::Int:
                    meta.default_value = static_cast<double>(nb::cast<int>(default_val));
                    break;
                case core::prop::PropType::Bool:
                    meta.default_value = nb::cast<bool>(default_val) ? 1.0 : 0.0;
                    break;
                case core::prop::PropType::String:
                    meta.default_string = nb::cast<std::string>(default_val);
                    break;
                case core::prop::PropType::Vec2:
                case core::prop::PropType::Vec3:
                case core::prop::PropType::Vec4:
                case core::prop::PropType::Color3:
                case core::prop::PropType::Color4:
                    if (nb::isinstance<nb::tuple>(default_val)) {
                        nb::tuple t = nb::cast<nb::tuple>(default_val);
                        size_t sz = t.size();
                        if (sz >= 2) {
                            meta.default_vec2[0] = nb::cast<double>(t[0]);
                            meta.default_vec2[1] = nb::cast<double>(t[1]);
                        }
                        if (sz >= 3) {
                            meta.default_vec3[0] = nb::cast<double>(t[0]);
                            meta.default_vec3[1] = nb::cast<double>(t[1]);
                            meta.default_vec3[2] = nb::cast<double>(t[2]);
                        }
                        if (sz >= 4) {
                            meta.default_vec4[0] = nb::cast<double>(t[0]);
                            meta.default_vec4[1] = nb::cast<double>(t[1]);
                            meta.default_vec4[2] = nb::cast<double>(t[2]);
                            meta.default_vec4[3] = nb::cast<double>(t[3]);
                        }
                    }
                    break;
                default:
                    break;
                }
            }
        }

        // Extract enum items
        if (meta.type == core::prop::PropType::Enum && nb::hasattr(descriptor, "items")) {
            nb::object items = descriptor.attr("items");
            if (nb::isinstance<nb::tuple>(items) || nb::isinstance<nb::list>(items)) {
                int idx = 0;
                for (auto item : items) {
                    if (nb::isinstance<nb::tuple>(item)) {
                        nb::tuple t = nb::cast<nb::tuple>(item);
                        if (t.size() >= 2) {
                            core::prop::EnumItem ei;
                            ei.identifier = nb::cast<std::string>(t[0]);
                            ei.name = nb::cast<std::string>(t[1]);
                            ei.value = idx;
                            meta.enum_items.push_back(std::move(ei));
                        }
                    }
                    ++idx;
                }
            }
        }

        // Python property getter/setter lambdas
        // These access Python object attributes directly
        core::prop::PropType prop_type = meta.type;
        std::string id_copy = prop_id;

        meta.getter = [id_copy, prop_type](const core::prop::PropertyObjectRef& ref) -> std::any {
            assert(ref.is_python() && "Cannot call Python property getter with C++ object");
            if (!ref.ptr)
                return std::any{};
            const auto* py_obj = static_cast<const nb::object*>(ref.ptr);
            if (!py_obj->is_valid())
                return std::any{};
            try {
                nb::object val = py_obj->attr(id_copy.c_str());
                switch (prop_type) {
                case core::prop::PropType::Float:
                    return nb::cast<float>(val);
                case core::prop::PropType::Int:
                    return nb::cast<int>(val);
                case core::prop::PropType::Bool:
                    return nb::cast<bool>(val);
                case core::prop::PropType::String:
                    return nb::cast<std::string>(val);
                case core::prop::PropType::Vec2: {
                    nb::tuple t = nb::cast<nb::tuple>(val);
                    std::array<float, 2> arr{};
                    for (size_t i = 0; i < 2 && i < t.size(); ++i) {
                        arr[i] = nb::cast<float>(t[i]);
                    }
                    return arr;
                }
                case core::prop::PropType::Vec3:
                case core::prop::PropType::Color3: {
                    nb::tuple t = nb::cast<nb::tuple>(val);
                    std::array<float, 3> arr{};
                    for (size_t i = 0; i < 3 && i < t.size(); ++i) {
                        arr[i] = nb::cast<float>(t[i]);
                    }
                    return arr;
                }
                case core::prop::PropType::Vec4:
                case core::prop::PropType::Color4: {
                    nb::tuple t = nb::cast<nb::tuple>(val);
                    std::array<float, 4> arr{};
                    for (size_t i = 0; i < 4 && i < t.size(); ++i) {
                        arr[i] = nb::cast<float>(t[i]);
                    }
                    return arr;
                }
                case core::prop::PropType::Tensor:
                    return val;
                default:
                    return std::any{};
                }
            } catch (...) {
                LOG_WARN("Property getter failed for '{}'", id_copy);
                return std::any{};
            }
        };

        meta.setter = [id_copy, prop_type](core::prop::PropertyObjectRef& ref, const std::any& value) {
            assert(ref.is_python() && "Cannot call Python property setter with C++ object");
            if (!ref.ptr)
                return;
            auto* py_obj = static_cast<nb::object*>(ref.ptr);
            if (!py_obj->is_valid())
                return;
            try {
                switch (prop_type) {
                case core::prop::PropType::Float:
                    py_obj->attr(id_copy.c_str()) = nb::cast(std::any_cast<float>(value));
                    break;
                case core::prop::PropType::Int:
                    py_obj->attr(id_copy.c_str()) = nb::cast(std::any_cast<int>(value));
                    break;
                case core::prop::PropType::Bool:
                    py_obj->attr(id_copy.c_str()) = nb::cast(std::any_cast<bool>(value));
                    break;
                case core::prop::PropType::String:
                    py_obj->attr(id_copy.c_str()) = nb::cast(std::any_cast<std::string>(value));
                    break;
                case core::prop::PropType::Vec2: {
                    auto arr = std::any_cast<std::array<float, 2>>(value);
                    py_obj->attr(id_copy.c_str()) = nb::make_tuple(arr[0], arr[1]);
                    break;
                }
                case core::prop::PropType::Vec3:
                case core::prop::PropType::Color3: {
                    auto arr = std::any_cast<std::array<float, 3>>(value);
                    py_obj->attr(id_copy.c_str()) = nb::make_tuple(arr[0], arr[1], arr[2]);
                    break;
                }
                case core::prop::PropType::Vec4:
                case core::prop::PropType::Color4: {
                    auto arr = std::any_cast<std::array<float, 4>>(value);
                    py_obj->attr(id_copy.c_str()) = nb::make_tuple(arr[0], arr[1], arr[2], arr[3]);
                    break;
                }
                case core::prop::PropType::Tensor:
                    py_obj->attr(id_copy.c_str()) = std::any_cast<nb::object>(value);
                    break;
                default:
                    break;
                }
            } catch (...) {
                LOG_WARN("Property setter failed for '{}'", id_copy);
            }
        };

        return meta;
    }

    void register_python_property_group(const std::string& group_id, const std::string& group_name,
                                        nb::object property_group_class) {
        if (!property_group_class.is_valid()) {
            LOG_WARN("Property group '{}' has invalid class", group_id);
            return;
        }

        if (!nb::hasattr(property_group_class, "_get_property_descriptors")) {
            LOG_WARN("Property group '{}' missing _get_property_descriptors", group_id);
            return;
        }

        nb::dict descriptors;
        try {
            nb::object result = property_group_class.attr("_get_property_descriptors")();
            if (!nb::isinstance<nb::dict>(result)) {
                return;
            }
            descriptors = nb::cast<nb::dict>(result);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to get property descriptors for '{}': {}", group_id, e.what());
            return;
        }

        if (descriptors.size() == 0) {
            return;
        }

        core::prop::PropertyGroup group;
        group.id = group_id;
        group.name = group_name;

        for (auto [key, value] : descriptors) {
            try {
                std::string prop_id = nb::cast<std::string>(key);
                auto meta = python_property_to_meta(nb::cast<nb::object>(value), prop_id);
                group.properties.push_back(std::move(meta));
            } catch (const std::exception& e) {
                LOG_WARN("Failed to convert property '{}': {}", group_id, e.what());
            }
        }

        if (!group.properties.empty()) {
            const size_t prop_count = group.properties.size();
            core::prop::PropertyRegistry::instance().register_group(std::move(group));
            LOG_INFO("Registered Python property group '{}' with {} properties", group_id, prop_count);
        }
    }

    void unregister_python_property_group(const std::string& group_id) {
        core::prop::PropertyRegistry::instance().unregister_group(group_id);
        LOG_INFO("Unregistered Python property group '{}'", group_id);
    }

} // namespace lfs::python
