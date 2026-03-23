/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_pipeline.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/operation/operation.hpp"
#include "visualizer/operation/ops/edit_ops.hpp"
#include "visualizer/operation/ops/select_ops.hpp"
#include "visualizer/operation/ops/transform_ops.hpp"
#include "visualizer/operation/pipeline.hpp"

#include <glm/glm.hpp>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace lfs::python {

    namespace {
        vis::op::OperatorProperties dict_to_props(const nb::dict& properties) {
            vis::op::OperatorProperties props;
            for (const auto& [key, value] : properties) {
                const auto key_str = nb::cast<std::string>(key);
                if (nb::isinstance<nb::float_>(value)) {
                    props.set(key_str, nb::cast<float>(value));
                } else if (nb::isinstance<nb::int_>(value)) {
                    props.set(key_str, nb::cast<int>(value));
                } else if (nb::isinstance<nb::bool_>(value)) {
                    props.set(key_str, nb::cast<bool>(value));
                } else if (nb::isinstance<nb::str>(value)) {
                    props.set(key_str, nb::cast<std::string>(value));
                } else if (nb::isinstance<nb::tuple>(value) || nb::isinstance<nb::list>(value)) {
                    const nb::sequence seq = nb::cast<nb::sequence>(value);
                    const auto len = nb::len(seq);
                    if (len == 3 && nb::isinstance<nb::float_>(seq[0])) {
                        props.set(key_str, glm::vec3(nb::cast<float>(seq[0]),
                                                     nb::cast<float>(seq[1]),
                                                     nb::cast<float>(seq[2])));
                    } else if (len == 16 && nb::isinstance<nb::float_>(seq[0])) {
                        glm::mat4 m;
                        for (int i = 0; i < 16; ++i) {
                            (&m[0][0])[i] = nb::cast<float>(seq[i]);
                        }
                        props.set(key_str, m);
                    }
                }
            }
            return props;
        }

        class PyStage {
        public:
            vis::op::OperationFactory factory;
            vis::op::OperatorProperties props;

            PyStage(vis::op::OperationFactory f, vis::op::OperatorProperties p = {})
                : factory(std::move(f)),
                  props(std::move(p)) {}
        };

        class PyPipeline {
        public:
            vis::op::Pipeline pipeline;

            PyPipeline() = default;
            explicit PyPipeline(std::string name)
                : pipeline(std::move(name)) {}

            PyPipeline& add(const PyStage& stage) {
                pipeline.add(stage.factory, stage.props);
                return *this;
            }

            PyPipeline& operator_or(const PyStage& stage) {
                return add(stage);
            }

            nb::dict execute() {
                auto* scene = vis::services().sceneOrNull();
                if (!scene) {
                    nb::dict result;
                    result["ok"] = false;
                    result["error"] = "No scene available";
                    return result;
                }

                auto op_result = pipeline.execute(*scene);

                nb::dict result;
                result["ok"] = op_result.ok();
                result["error"] = op_result.error;
                return result;
            }

            bool poll() const {
                auto* scene = vis::services().sceneOrNull();
                if (!scene) {
                    return false;
                }
                return pipeline.poll(*scene);
            }
        };

        template <typename OpClass>
        PyStage make_stage(const nb::kwargs& kwargs) {
            return PyStage(
                [] { return std::make_unique<OpClass>(); },
                kwargs ? dict_to_props(nb::dict(kwargs)) : vis::op::OperatorProperties{});
        }

    } // namespace

    void register_pipeline(nb::module_& m) {
        auto pipe = m.def_submodule("pipeline", "Compositional operations system");

        nb::class_<PyStage>(pipe, "Stage")
            .def(
                "__or__", [](const PyStage& a, const PyStage& b) {
                    PyPipeline p;
                    p.add(a);
                    p.add(b);
                    return p;
                },
                "Chain two stages into a pipeline")
            .def("execute", [](const PyStage& s) {
                    PyPipeline p;
                    p.add(s);
                    return p.execute(); }, "Execute this stage immediately");

        nb::class_<PyPipeline>(pipe, "Pipeline")
            .def(nb::init<>())
            .def(nb::init<std::string>(), nb::arg("name"), "Create a named pipeline")
            .def("add", &PyPipeline::add, nb::rv_policy::reference, "Append a stage to the pipeline")
            .def("__or__", &PyPipeline::operator_or, nb::rv_policy::reference, "Append a stage via pipe operator")
            .def("execute", &PyPipeline::execute, "Execute all stages and return result dict")
            .def("poll", &PyPipeline::poll, "Check if all stages can execute");

        auto select = pipe.def_submodule("select", "Selection operations");
        select.def(
            "all", [](nb::kwargs kwargs) { return make_stage<vis::op::SelectAll>(kwargs); }, "Create select-all stage");
        select.def(
            "none", [](nb::kwargs kwargs) { return make_stage<vis::op::SelectNone>(kwargs); }, "Create deselect-all stage");
        select.def(
            "invert", [](nb::kwargs kwargs) { return make_stage<vis::op::SelectInvert>(kwargs); }, "Create invert-selection stage");
        select.def(
            "grow", [](nb::kwargs kwargs) { return make_stage<vis::op::SelectGrow>(kwargs); }, "Create grow-selection stage");
        select.def(
            "shrink", [](nb::kwargs kwargs) { return make_stage<vis::op::SelectShrink>(kwargs); }, "Create shrink-selection stage");

        auto edit = pipe.def_submodule("edit", "Edit operations");
        edit.def(
            "delete_", [](nb::kwargs kwargs) { return make_stage<vis::op::EditDelete>(kwargs); }, "Create delete stage");
        edit.def(
            "duplicate", [](nb::kwargs kwargs) { return make_stage<vis::op::EditDuplicate>(kwargs); }, "Create duplicate stage");

        auto transform = pipe.def_submodule("transform", "Transform operations");
        transform.def(
            "translate", [](nb::kwargs kwargs) { return make_stage<vis::op::TransformTranslate>(kwargs); }, "Create translation stage");
        transform.def(
            "rotate", [](nb::kwargs kwargs) { return make_stage<vis::op::TransformRotate>(kwargs); }, "Create rotation stage");
        transform.def(
            "scale", [](nb::kwargs kwargs) { return make_stage<vis::op::TransformScale>(kwargs); }, "Create scale stage");
        transform.def(
            "set", [](nb::kwargs kwargs) { return make_stage<vis::op::TransformSet>(kwargs); }, "Create set-transform stage");
    }

} // namespace lfs::python
