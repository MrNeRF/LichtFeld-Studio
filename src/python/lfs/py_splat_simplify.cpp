/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_splat_simplify.hpp"

#include "core/scene.hpp"
#include "python/python_runtime.hpp"

#include <nanobind/stl/string.h>

namespace nb = nanobind;

namespace lfs::python {

    void register_splat_simplify(nb::module_& m) {
        m.def(
            "simplify_splats",
            [](const std::string& source_name,
               double ratio,
               int knn_k,
               double merge_cap,
               float opacity_prune_threshold) {
                auto* scene = get_application_scene();
                if (!scene)
                    throw std::runtime_error("No scene available");

                const auto* node = scene->getNode(source_name);
                if (!node || node->type != core::NodeType::SPLAT || !node->model)
                    throw std::runtime_error("No splat node named '" + source_name + "'");

                if (invoke_splat_simplify_active())
                    throw std::runtime_error("A splat simplification is already running");

                core::SplatSimplifyOptions opts;
                opts.ratio = ratio;
                opts.knn_k = knn_k;
                opts.merge_cap = merge_cap;
                opts.opacity_prune_threshold = opacity_prune_threshold;
                invoke_splat_simplify_start(source_name, opts);
            },
            nb::arg("source_name"),
            nb::arg("ratio") = 0.1,
            nb::arg("knn_k") = 16,
            nb::arg("merge_cap") = 0.5,
            nb::arg("opacity_prune_threshold") = 0.1f,
            "Simplify a splat node asynchronously and create a new output node.");

        m.def(
            "cancel_splat_simplify",
            []() { invoke_splat_simplify_cancel(); },
            "Cancel the active splat simplification job");

        m.def(
            "is_splat_simplify_active",
            []() { return invoke_splat_simplify_active(); },
            "Check if a splat simplification job is currently running");

        m.def(
            "get_splat_simplify_progress",
            []() { return invoke_splat_simplify_progress(); },
            "Get splat simplification progress (0.0 to 1.0)");

        m.def(
            "get_splat_simplify_stage",
            []() { return invoke_splat_simplify_stage(); },
            "Get splat simplification stage text");

        m.def(
            "get_splat_simplify_error",
            []() { return invoke_splat_simplify_error(); },
            "Get the last splat simplification error (empty on success)");
    }

} // namespace lfs::python
