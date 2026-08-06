/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"

namespace lfs::python {

    PyOperatorProperties::PyOperatorProperties(const std::string& operator_id)
        : operator_id_(operator_id),
          properties_(nb::dict()) {
    }

    void PyOperatorProperties::set_property(const std::string& name, nb::object value) {
        properties_[nb::str(name.c_str())] = std::move(value);
    }

    nb::object PyOperatorProperties::get_property(const std::string& name) const {
        auto key = nb::str(name.c_str());
        if (properties_.contains(key)) {
            return properties_[key];
        }
        return nb::none();
    }

    nb::dict PyOperatorProperties::get_properties() const {
        return properties_;
    }

} // namespace lfs::python
