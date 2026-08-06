/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/property_system.hpp"

#include <nanobind/nanobind.h>

#include <string>

namespace nb = nanobind;

namespace lfs::python {

    // Register a Python PropertyGroup (or operator class) with the C++ PropertyRegistry
    // group_id format: "operator.<idname>" for operators, "<type>.<name>" for other classes
    void register_python_property_group(
        const std::string& group_id,
        const std::string& group_name,
        nb::object property_group_class);

    // Unregister a Python property group
    void unregister_python_property_group(const std::string& group_id);

    nb::dict property_group_info(const std::string& group_id);

} // namespace lfs::python
