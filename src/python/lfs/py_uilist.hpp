/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_ui.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace nb = nanobind;

namespace lfs::python {

    struct PyUIListInfo {
        std::string id;
        nb::object list_class;
        nb::object list_instance;
    };

    class PyUIListRegistry {
    public:
        static PyUIListRegistry& instance();

        void register_uilist(nb::object list_class);
        void unregister_uilist(const std::string& id);
        void unregister_all();

        [[nodiscard]] nb::object get_uilist(const std::string& id);
        [[nodiscard]] std::vector<std::string> get_uilist_ids() const;

    private:
        PyUIListRegistry() = default;
        PyUIListRegistry(const PyUIListRegistry&) = delete;
        PyUIListRegistry& operator=(const PyUIListRegistry&) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, PyUIListInfo> uilists_;
    };

    class PyUILayoutTemplates {
    public:
        static std::tuple<bool, int> template_list(
            PyUILayout& layout,
            const std::string& listtype_name,
            const std::string& list_id,
            nb::object data,
            const std::string& propname,
            int active_index,
            int rows = 5);

        static bool template_tree(
            PyUILayout& layout,
            const std::string& label,
            nb::object draw_callback,
            bool default_open = false);

        static std::tuple<bool, std::string> template_id(
            PyUILayout& layout,
            const std::string& label,
            const std::vector<std::string>& items,
            const std::string& current_id);
    };

    void register_uilist(nb::module_& m);
    void add_template_methods_to_uilayout(nb::class_<PyUILayout>& layout_class);

} // namespace lfs::python
