/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_scene.hpp"
#include "py_ui.hpp"
#include "python/python_runtime.hpp"

#include "control/command_api.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/scene.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/training/training_manager.hpp"

namespace lfs::python {

    bool PyAppContext::has_scene() const {
        auto* sm = get_scene_manager();
        return sm && sm->getScene().getNodeCount() > 0;
    }

    uint64_t PyAppContext::scene_generation() const { return get_scene_generation(); }

    bool PyAppContext::has_trainer() const {
        const auto* tm = get_trainer_manager();
        return tm && tm->hasTrainer();
    }

    bool PyAppContext::is_training() const {
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().is_running : false;
    }

    bool PyAppContext::is_paused() const {
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().is_paused : false;
    }

    int PyAppContext::iteration() const {
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().iteration : 0;
    }

    int PyAppContext::max_iterations() const {
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().max_iterations : 0;
    }

    float PyAppContext::loss() const {
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().loss : 0.0f;
    }

    bool PyAppContext::has_selection() const {
        auto* sm = get_scene_manager();
        return sm && sm->hasSelectedNode();
    }

    size_t PyAppContext::num_gaussians() const {
        if (auto* scene = get_application_scene()) {
            return scene->getTotalGaussianCount();
        }
        const auto* cc = lfs::event::command_center();
        return cc ? cc->snapshot().num_gaussians : 0;
    }

    int PyAppContext::selection_submode() const { return get_selection_submode(); }
    int PyAppContext::pivot_mode() const { return get_pivot_mode(); }
    int PyAppContext::transform_space() const { return get_transform_space(); }

    std::tuple<float, float, float, float> PyAppContext::viewport_bounds() const {
        float x, y, w, h;
        get_viewport_bounds(x, y, w, h);
        return {x, y, w, h};
    }

    bool PyAppContext::viewport_valid() const { return has_viewport_bounds(); }

    nb::object PyAppContext::scene() const {
        auto* sm = get_scene_manager();
        if (!sm || sm->getScene().getNodeCount() == 0) {
            return nb::none();
        }
        return nb::cast(PyScene(&sm->getScene()));
    }

    nb::object PyAppContext::selected_objects() const {
        auto* sm = get_scene_manager();
        if (!sm) {
            return nb::list();
        }

        const auto names = sm->getSelectedNodeNames();
        auto& scene_ref = sm->getScene();
        nb::list result;

        for (const auto& name : names) {
            if (auto* node = scene_ref.getMutableNode(name)) {
                result.append(nb::cast(PySceneNode(node, &scene_ref)));
            }
        }
        return result;
    }

    nb::object PyAppContext::active_object() const {
        auto* sm = get_scene_manager();
        if (!sm || !sm->hasSelectedNode()) {
            return nb::none();
        }

        const auto name = sm->getSelectedNodeName();
        if (name.empty()) {
            return nb::none();
        }

        auto& scene_ref = sm->getScene();
        if (auto* node = scene_ref.getMutableNode(name)) {
            return nb::cast(PySceneNode(node, &scene_ref));
        }
        return nb::none();
    }

    nb::object PyAppContext::selected_gaussians() const {
        auto* scene = get_application_scene();
        if (!scene || !scene->hasSelection()) {
            return nb::none();
        }

        auto mask = scene->getSelectionMask();
        if (!mask) {
            return nb::none();
        }
        return nb::cast(PyTensor(*mask, false));
    }

    PyAppContext get_app_context() { return PyAppContext{}; }

    void register_ui_context(nb::module_& m) {
        nb::class_<PyAppContext>(m, "AppContext")
            .def(nb::init<>())
            .def_prop_ro("has_scene", &PyAppContext::has_scene)
            .def_prop_ro("scene_generation", &PyAppContext::scene_generation)
            .def_prop_ro("has_trainer", &PyAppContext::has_trainer)
            .def_prop_ro("is_training", &PyAppContext::is_training)
            .def_prop_ro("is_paused", &PyAppContext::is_paused)
            .def_prop_ro("iteration", &PyAppContext::iteration)
            .def_prop_ro("max_iterations", &PyAppContext::max_iterations)
            .def_prop_ro("loss", &PyAppContext::loss)
            .def_prop_ro("has_selection", &PyAppContext::has_selection)
            .def_prop_ro("num_gaussians", &PyAppContext::num_gaussians)
            .def_prop_ro("selection_submode", &PyAppContext::selection_submode)
            .def_prop_ro("pivot_mode", &PyAppContext::pivot_mode)
            .def_prop_ro("transform_space", &PyAppContext::transform_space)
            .def_prop_ro("viewport_bounds", &PyAppContext::viewport_bounds)
            .def_prop_ro("viewport_valid", &PyAppContext::viewport_valid)
            .def_prop_ro("scene", &PyAppContext::scene)
            .def_prop_ro("selected_objects", &PyAppContext::selected_objects)
            .def_prop_ro("active_object", &PyAppContext::active_object)
            .def_prop_ro("selected_gaussians", &PyAppContext::selected_gaussians,
                         "Gaussian selection mask tensor");

        m.def("context", &get_app_context, "Get the current application context");
    }

} // namespace lfs::python
