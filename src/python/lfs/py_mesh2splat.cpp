/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_mesh2splat.hpp"

#include "core/mesh2splat.hpp"
#include "core/scene.hpp"
#include "python/python_runtime.hpp"
#include "visualizer/ipc/view_context.hpp"

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <glm/glm.hpp>

namespace nb = nanobind;

namespace lfs::python {

    void register_mesh2splat(nb::module_& m) {
        m.def(
            "mesh_to_splat",
            [](const std::string& mesh_name,
               float sigma,
               float quality,
               int max_resolution,
               std::optional<std::tuple<float, float, float>> light_dir,
               float light_intensity,
               float ambient) {
                auto* scene = get_application_scene();
                if (!scene)
                    throw std::runtime_error("No scene available");

                const auto* node = scene->getNode(mesh_name);
                if (!node || node->type != core::NodeType::MESH || !node->mesh)
                    throw std::runtime_error("No mesh node named '" + mesh_name + "'");

                if (invoke_mesh2splat_active())
                    throw std::runtime_error("A mesh-to-splat conversion is already running");

                constexpr int kMinRes = core::Mesh2SplatOptions::kMinResolution;

                core::Mesh2SplatOptions opts;
                opts.resolution_target = kMinRes + static_cast<int>(quality * (max_resolution - kMinRes));
                opts.sigma = sigma;
                opts.light_intensity = light_intensity;
                opts.ambient = ambient;

                if (light_dir) {
                    auto [x, y, z] = *light_dir;
                    opts.light_dir = glm::normalize(glm::vec3(x, y, z));
                } else {
                    auto view = vis::get_current_view_info();
                    if (view) {
                        glm::vec3 cam(view->translation[0], view->translation[1], view->translation[2]);
                        float len = glm::length(cam);
                        if (len > 1e-6f)
                            opts.light_dir = cam / len;
                    }
                }

                invoke_mesh2splat_start(node->mesh, mesh_name, opts);
            },
            nb::arg("mesh_name"),
            nb::arg("sigma") = 0.65f,
            nb::arg("quality") = 0.5f,
            nb::arg("max_resolution") = 1024,
            nb::arg("light_dir") = nb::none(),
            nb::arg("light_intensity") = 0.7f,
            nb::arg("ambient") = 0.4f,
            "Convert a mesh node to gaussian splats. Runs asynchronously on the GL thread.");

        m.def(
            "is_mesh2splat_active",
            []() { return invoke_mesh2splat_active(); },
            "Check if a mesh-to-splat conversion is currently running");

        m.def(
            "get_mesh2splat_progress",
            []() { return invoke_mesh2splat_progress(); },
            "Get mesh-to-splat conversion progress (0.0 to 1.0)");

        m.def(
            "get_mesh2splat_error",
            []() { return invoke_mesh2splat_error(); },
            "Get error message from last mesh-to-splat conversion (empty on success)");
    }

} // namespace lfs::python
