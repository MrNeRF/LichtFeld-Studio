/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/property_registry.hpp"
#include "py_prop_traits.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    template <typename T>
    class PyProp {
    public:
        PyProp(T* obj, const std::string& group_id) : obj_(obj),
                                                      group_id_(group_id) {}

        nb::object getattr(const std::string& name) const {
            auto meta = core::prop::PropertyRegistry::instance().get_property(group_id_, name);
            if (!meta) {
                throw nb::attribute_error(("Unknown property: " + name).c_str());
            }
            auto ref = core::prop::PropertyObjectRef::cpp(const_cast<T*>(obj_));
            return any_to_python(meta->getter(ref), meta->type);
        }

        void setattr(const std::string& name, nb::object value) {
            auto meta = core::prop::PropertyRegistry::instance().get_property(group_id_, name);
            if (!meta) {
                throw nb::attribute_error(("Unknown property: " + name).c_str());
            }
            if (meta->is_readonly()) {
                throw nb::attribute_error(("Property is read-only: " + name).c_str());
            }

            auto ref = core::prop::PropertyObjectRef::cpp(obj_);
            const std::any old_val = meta->getter(ref);
            const std::any new_val = python_to_any(value, meta->type);
            meta->setter(ref, new_val);

            if (meta->on_update) {
                meta->on_update(ref, old_val, new_val);
            }

            core::prop::PropertyRegistry::instance().notify(group_id_, name, old_val, new_val);
        }

        nb::list dir() const {
            const auto* group = core::prop::PropertyRegistry::instance().get_group(group_id_);
            nb::list result;
            if (group) {
                for (const auto& prop : group->properties) {
                    result.append(prop.id);
                }
            }
            return result;
        }

        nb::dict prop_info(const std::string& name) const {
            auto meta = core::prop::PropertyRegistry::instance().get_property(group_id_, name);
            if (!meta) {
                return nb::dict();
            }

            nb::dict info;
            info["id"] = meta->id;
            info["name"] = meta->name;
            info["description"] = meta->description;
            info["type"] = prop_type_string(meta->type);
            info["readonly"] = meta->is_readonly();
            info["animatable"] = meta->is_animatable();
            info["flags"] = meta->flags;
            info["operator_arg"] = meta->has_flag(core::prop::PROP_OPERATOR_ARG);

            if (meta->min_value) {
                info["min"] = *meta->min_value;
            }
            if (meta->max_value) {
                info["max"] = *meta->max_value;
            }
            if (meta->default_value) {
                info["default"] = prop_default_to_python(*meta->default_value);
            }

            return info;
        }

        T* obj() { return obj_; }
        const T* obj() const { return obj_; }
        const std::string& group_id() const { return group_id_; }

    private:
        T* obj_;
        std::string group_id_;
    };

} // namespace lfs::python
