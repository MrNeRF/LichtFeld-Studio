/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_operator.hpp"
#include "py_ui.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/operator/operator.hpp"
#include "visualizer/operator/operator_id.hpp"
#include "visualizer/operator/operator_registry.hpp"
#include <glm/glm.hpp>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    void register_operators(nb::module_& m) {
        auto ops = m.def_submodule("ops", "Operator system");

        nb::enum_<vis::op::OperatorResult>(ops, "OperatorResult")
            .value("FINISHED", vis::op::OperatorResult::FINISHED)
            .value("CANCELLED", vis::op::OperatorResult::CANCELLED)
            .value("RUNNING_MODAL", vis::op::OperatorResult::RUNNING_MODAL)
            .value("PASS_THROUGH", vis::op::OperatorResult::PASS_THROUGH);

        nb::enum_<vis::op::BuiltinOp>(ops, "BuiltinOp")
            .value("SelectionStroke", vis::op::BuiltinOp::SelectionStroke)
            .value("BrushStroke", vis::op::BuiltinOp::BrushStroke)
            .value("TransformSet", vis::op::BuiltinOp::TransformSet)
            .value("TransformTranslate", vis::op::BuiltinOp::TransformTranslate)
            .value("TransformRotate", vis::op::BuiltinOp::TransformRotate)
            .value("TransformScale", vis::op::BuiltinOp::TransformScale)
            .value("TransformApplyBatch", vis::op::BuiltinOp::TransformApplyBatch)
            .value("AlignPickPoint", vis::op::BuiltinOp::AlignPickPoint)
            .value("Undo", vis::op::BuiltinOp::Undo)
            .value("Redo", vis::op::BuiltinOp::Redo)
            .value("Delete", vis::op::BuiltinOp::Delete)
            .value("SelectionClear", vis::op::BuiltinOp::SelectionClear)
            .value("SceneSelectNode", vis::op::BuiltinOp::SceneSelectNode)
            .value("CropBoxAdd", vis::op::BuiltinOp::CropBoxAdd)
            .value("CropBoxSet", vis::op::BuiltinOp::CropBoxSet)
            .value("CropBoxFit", vis::op::BuiltinOp::CropBoxFit)
            .value("CropBoxReset", vis::op::BuiltinOp::CropBoxReset)
            .value("EllipsoidAdd", vis::op::BuiltinOp::EllipsoidAdd)
            .value("EllipsoidSet", vis::op::BuiltinOp::EllipsoidSet)
            .value("EllipsoidFit", vis::op::BuiltinOp::EllipsoidFit)
            .value("EllipsoidReset", vis::op::BuiltinOp::EllipsoidReset);

        nb::enum_<vis::op::BuiltinTool>(ops, "BuiltinTool")
            .value("Select", vis::op::BuiltinTool::Select)
            .value("Translate", vis::op::BuiltinTool::Translate)
            .value("Rotate", vis::op::BuiltinTool::Rotate)
            .value("Scale", vis::op::BuiltinTool::Scale)
            .value("Mirror", vis::op::BuiltinTool::Mirror)
            .value("Brush", vis::op::BuiltinTool::Brush)
            .value("Align", vis::op::BuiltinTool::Align);

        nb::enum_<vis::op::OperatorFlags>(ops, "OperatorFlags")
            .value("NONE", vis::op::OperatorFlags::NONE)
            .value("REGISTER", vis::op::OperatorFlags::REGISTER)
            .value("UNDO", vis::op::OperatorFlags::UNDO)
            .value("UNDO_GROUPED", vis::op::OperatorFlags::UNDO_GROUPED)
            .value("INTERNAL", vis::op::OperatorFlags::INTERNAL)
            .value("MODAL", vis::op::OperatorFlags::MODAL)
            .value("BLOCKING", vis::op::OperatorFlags::BLOCKING)
            .def("__or__", [](vis::op::OperatorFlags a, vis::op::OperatorFlags b) { return a | b; })
            .def("__and__", [](vis::op::OperatorFlags a, vis::op::OperatorFlags b) { return a & b; });

        nb::class_<vis::op::OperatorDescriptor>(ops, "OperatorDescriptor")
            .def(nb::init<>())
            .def_prop_ro("id", &vis::op::OperatorDescriptor::id, "Unique operator identifier")
            .def_rw("label", &vis::op::OperatorDescriptor::label, "Display label")
            .def_rw("description", &vis::op::OperatorDescriptor::description, "Tooltip description")
            .def_rw("icon", &vis::op::OperatorDescriptor::icon, "Icon name")
            .def_rw("shortcut", &vis::op::OperatorDescriptor::shortcut, "Keyboard shortcut string")
            .def_rw("flags", &vis::op::OperatorDescriptor::flags, "Operator behavior flags");

        ops.def(
            "invoke",
            [](const std::string& id, nb::kwargs kwargs) -> PyOperatorReturnValue {
                auto& registry = vis::op::operators();

                nb::object instance = get_python_operator_instance(id);
                if (kwargs && nb::len(kwargs) > 0) {
                    if (instance.is_valid() && !instance.is_none()) {
                        for (auto [key, value] : kwargs) {
                            std::string key_str = nb::cast<std::string>(key);
                            nb::setattr(instance, key_str.c_str(), value);
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

                return PyOperatorReturnValue(status_str, std::move(data));
            },
            "Invoke an operator by id with optional kwargs");

        ops.def(
            "poll", [](const std::string& id) { return vis::op::operators().poll(id); }, nb::arg("id"),
            "Check if an operator can run");

        ops.def(
            "get_all", []() {
                std::vector<std::string> ids;
                for (const auto* desc : vis::op::operators().getAllOperators()) {
                    ids.push_back(desc->id());
                }
                return ids;
            },
            "Get all registered operator IDs");

        ops.def(
            "get_descriptor",
            [](const std::string& id) -> std::optional<vis::op::OperatorDescriptor> {
                const auto* desc = vis::op::operators().getDescriptor(id);
                return desc ? std::optional(*desc) : std::nullopt;
            },
            nb::arg("id"),
            "Get operator descriptor by ID (None if not found)");

        ops.def(
            "undo", [] { vis::op::undoHistory().undo(); },
            "Undo the last operation");
        ops.def(
            "redo", [] { vis::op::undoHistory().redo(); },
            "Redo the last undone operation");
        ops.def(
            "can_undo", [] { return vis::op::undoHistory().canUndo(); },
            "Check if undo is available");
        ops.def(
            "can_redo", [] { return vis::op::undoHistory().canRedo(); },
            "Check if redo is available");
        ops.def(
            "has_modal", [] { return vis::op::operators().hasModalOperator(); },
            "Check if a modal operator is running");
        ops.def(
            "cancel_modal", [] { vis::op::operators().cancelModalOperator(); },
            "Cancel the active modal operator");
    }

} // namespace lfs::python
