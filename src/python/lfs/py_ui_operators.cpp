/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "py_ui.hpp"
#include "visualizer/operator/operator_registry.hpp"

namespace lfs::python {

    void register_operator_return_value(nb::module_& m) {
        nb::class_<PyOperatorReturnValue>(m, "OperatorReturnValue")
            .def_ro("status", &PyOperatorReturnValue::status, "Result status string")
            .def_prop_ro("_data", &PyOperatorReturnValue::get_all_data, "Raw return data dictionary")
            .def_prop_ro("finished", &PyOperatorReturnValue::finished, "Whether operator completed successfully")
            .def_prop_ro("cancelled", &PyOperatorReturnValue::cancelled, "Whether operator was cancelled")
            .def_prop_ro("running_modal", &PyOperatorReturnValue::running_modal, "Whether operator is running as modal")
            .def_prop_ro("pass_through", &PyOperatorReturnValue::pass_through, "Whether event should pass through")
            .def("__getattr__", &PyOperatorReturnValue::getattr, "Access return data by attribute name")
            .def("__bool__", &PyOperatorReturnValue::finished, "True if operator finished successfully");
    }

    void register_ui_operators(nb::module_& m) {

        nb::class_<PyOperatorProperties>(m, "OperatorProperties")
            .def(nb::init<const std::string&>(), nb::arg("operator_id"))
            .def("__setattr__", &PyOperatorProperties::set_property, "Set an operator property value")
            .def("__getattr__", &PyOperatorProperties::get_property, "Get an operator property value")
            .def_prop_ro("properties", &PyOperatorProperties::get_properties, "All properties as a dictionary")
            .def_prop_ro("operator_id", &PyOperatorProperties::get_operator_id, "Operator identifier string");

        m.def(
            "unregister_operator", [](const std::string& id) {
                vis::op::operators().unregisterOperator(id);
            },
            nb::arg("id"), "Unregister an operator");

        m.def(
            "unregister_all_operators", []() {
                vis::op::operators().unregisterAllPython();
            },
            "Unregister all Python operators");

        m.def(
            "execute_operator", [](const std::string& id) {
                return vis::op::operators().invoke(id).status == vis::op::OperatorResult::FINISHED;
            },
            nb::arg("id"), "Execute an operator by id");

        m.def(
            "poll_operator", [](const std::string& id) {
                return vis::op::operators().poll(id);
            },
            nb::arg("id"), "Check if an operator can run");

        m.def(
            "get_operator_ids", []() {
                auto ops = vis::op::operators().getAllOperators();
                std::vector<std::string> ids;
                ids.reserve(ops.size());
                for (const auto* desc : ops) {
                    ids.push_back(desc->id());
                }
                return ids;
            },
            "Get list of registered operator ids");

        auto ops_module = m.def_submodule("ops", "Operator invocation");

        ops_module.def(
            "invoke", [](const std::string& id, nb::kwargs kwargs) -> PyOperatorReturnValue {
                auto& registry = vis::op::operators();

                nb::object instance = get_python_operator_instance(id);
                std::vector<std::string> set_kwargs;

                if (kwargs && nb::len(kwargs) > 0) {
                    if (instance.is_valid() && !instance.is_none()) {
                        for (auto [key, value] : kwargs) {
                            std::string key_str = nb::cast<std::string>(key);
                            nb::setattr(instance, key_str.c_str(), value);
                            set_kwargs.push_back(key_str);
                        }
                    }
                }

                auto result = registry.invoke(id);

                std::string status_str;
                switch (result.status) {
                case vis::op::OperatorResult::FINISHED:
                    status_str = "FINISHED";
                    break;
                case vis::op::OperatorResult::CANCELLED:
                    status_str = "CANCELLED";
                    break;
                case vis::op::OperatorResult::RUNNING_MODAL:
                    status_str = "RUNNING_MODAL";
                    break;
                case vis::op::OperatorResult::PASS_THROUGH:
                    status_str = "PASS_THROUGH";
                    break;
                }

                nb::dict data;
                if (instance.is_valid() && !instance.is_none() && nb::hasattr(instance, "_return_data")) {
                    nb::object return_data = instance.attr("_return_data");
                    if (nb::isinstance<nb::dict>(return_data)) {
                        data = nb::cast<nb::dict>(return_data);
                    }
                    nb::delattr(instance, "_return_data");
                }

                for (const auto& key : set_kwargs) {
                    if (nb::hasattr(instance, key.c_str())) {
                        nb::delattr(instance, key.c_str());
                    }
                }

                return PyOperatorReturnValue(status_str, data);
            },
            "Invoke an operator by id with optional properties");

        ops_module.def(
            "poll", [](const std::string& id) {
                return vis::op::operators().poll(id);
            },
            nb::arg("id"), "Check if operator can run");

        ops_module.def(
            "cancel_modal", []() {
                vis::op::operators().cancelModalOperator();
            },
            "Cancel any active modal operator");

        ops_module.def(
            "has_modal", []() {
                return vis::op::operators().hasModalOperator();
            },
            "Check if a modal operator is running");
    }

} // namespace lfs::python
