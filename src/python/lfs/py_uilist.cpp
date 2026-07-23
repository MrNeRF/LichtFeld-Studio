/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_uilist.hpp"
#include "core/logger.hpp"

namespace lfs::python {

    PyUIListRegistry& PyUIListRegistry::instance() {
        static PyUIListRegistry registry;
        return registry;
    }

    void PyUIListRegistry::register_uilist(nb::object list_class) {
        std::lock_guard lock(mutex_);

        std::string id;
        if (nb::hasattr(list_class, "list_id")) {
            id = nb::cast<std::string>(list_class.attr("list_id"));
        } else if (nb::hasattr(list_class, "__name__")) {
            id = nb::cast<std::string>(list_class.attr("__name__"));
        } else {
            LOG_ERROR("UIList class missing list_id or __name__ attribute");
            return;
        }

        uilists_[id] = {id, list_class, nb::object()};
    }

    void PyUIListRegistry::unregister_uilist(const std::string& id) {
        std::lock_guard lock(mutex_);
        uilists_.erase(id);
    }

    void PyUIListRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        uilists_.clear();
    }

    PyUIListInfo* PyUIListRegistry::ensure_instance(PyUIListInfo& uilist) {
        if (!uilist.list_instance.is_valid() || uilist.list_instance.is_none()) {
            nb::gil_scoped_acquire gil;
            try {
                uilist.list_instance = uilist.list_class();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to instantiate UIList {}: {}", uilist.id, e.what());
                return nullptr;
            }
        }
        return &uilist;
    }

    PyUIListInfo* PyUIListRegistry::get_uilist(const std::string& id) {
        std::lock_guard lock(mutex_);
        auto it = uilists_.find(id);
        if (it == uilists_.end())
            return nullptr;
        return ensure_instance(it->second);
    }

    std::vector<std::string> PyUIListRegistry::get_uilist_ids() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(uilists_.size());
        for (const auto& [id, _] : uilists_) {
            ids.push_back(id);
        }
        return ids;
    }

    std::tuple<bool, int> PyUILayoutTemplates::template_list(
        PyUILayout& layout,
        const std::string&,
        const std::string&,
        nb::object,
        const std::string&,
        const int active_index,
        int) {
        if (layout.isDrawHook()) {
            layout.warnUnsupportedInDrawHook("template_list");
            return {false, active_index};
        }
        throw nb::type_error(
            "UILayout.template_list is unsupported; use the live "
            "RmlUILayout.template_list API from a Python panel draw callback");
    }

    bool PyUILayoutTemplates::template_tree(
        PyUILayout& layout,
        const std::string&,
        nb::object,
        bool) {
        if (layout.isDrawHook()) {
            layout.warnUnsupportedInDrawHook("template_tree");
            return false;
        }
        throw nb::type_error(
            "UILayout.template_tree is unsupported; use the live "
            "RmlUILayout.tree_node/tree_pop API from a Python panel draw callback");
    }

    std::tuple<bool, std::string> PyUILayoutTemplates::template_id(
        PyUILayout& layout,
        const std::string&,
        const std::vector<std::string>&,
        const std::string& current_id) {
        if (layout.isDrawHook()) {
            layout.warnUnsupportedInDrawHook("template_id");
            return {false, current_id};
        }
        throw nb::type_error(
            "UILayout.template_id is unsupported; use the live "
            "RmlUILayout.combo API from a Python panel draw callback");
    }

    void register_uilist(nb::module_& m) {
        m.def(
            "register_uilist", [](nb::object list_class) {
                PyUIListRegistry::instance().register_uilist(list_class);
            },
            nb::arg("list_class"), "Register a UIList class for custom list rendering");

        m.def(
            "unregister_uilist", [](const std::string& id) {
                PyUIListRegistry::instance().unregister_uilist(id);
            },
            nb::arg("id"), "Unregister a UIList by ID");

        m.def(
            "unregister_all_uilists", []() { PyUIListRegistry::instance().unregister_all(); },
            "Unregister all UIList classes");
        m.def(
            "get_uilist_ids", []() { return PyUIListRegistry::instance().get_uilist_ids(); },
            "Get all registered UIList IDs");
    }

    void add_template_methods_to_uilayout(nb::class_<PyUILayout>& layout_class) {
        layout_class
            .def(
                "template_list",
                [](PyUILayout& self, const std::string& listtype_name, const std::string& list_id,
                   nb::object data, const std::string& propname, int active_index, int rows) {
                    return PyUILayoutTemplates::template_list(self, listtype_name, list_id,
                                                              data, propname, active_index, rows);
                },
                nb::arg("listtype_name"), nb::arg("list_id"), nb::arg("data"),
                nb::arg("propname"), nb::arg("active_index"), nb::arg("rows") = 5,
                "Unsupported on UILayout; use RmlUILayout.template_list.")
            .def(
                "template_tree",
                [](PyUILayout& self, const std::string& label, nb::object draw_callback, bool default_open) {
                    return PyUILayoutTemplates::template_tree(self, label, draw_callback, default_open);
                },
                nb::arg("label"), nb::arg("draw_callback"), nb::arg("default_open") = false,
                "Unsupported on UILayout; use RmlUILayout.tree_node/tree_pop.")
            .def(
                "template_id",
                [](PyUILayout& self, const std::string& label, const std::vector<std::string>& items,
                   const std::string& current_id) {
                    return PyUILayoutTemplates::template_id(self, label, items, current_id);
                },
                nb::arg("label"), nb::arg("items"), nb::arg("current_id"),
                "Unsupported on UILayout; use RmlUILayout.combo.");
    }

} // namespace lfs::python
