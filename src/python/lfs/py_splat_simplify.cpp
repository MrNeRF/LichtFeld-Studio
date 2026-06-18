/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_splat_simplify.hpp"

#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_simplify.hpp"
#include "core/splat_simplify_history.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"
#include "python/python_runtime.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace {

    struct PyProgressCallback {
        nb::object callback;

        [[nodiscard]] bool valid() const {
            return callback.is_valid() && !callback.is_none();
        }

        [[nodiscard]] bool operator()(const float progress, const std::string& stage) const {
            if (!valid())
                return true;
            nb::gil_scoped_acquire gil;
            try {
                nb::object result = callback(progress, stage);
                if (nb::isinstance<nb::bool_>(result))
                    return nb::cast<bool>(result);
                return true;
            } catch (const std::exception& e) {
                LOG_ERROR("Python splat simplify progress callback error: {}", e.what());
                return false;
            }
        }
    };

} // namespace

namespace lfs::python {

    namespace {

        class PySplatSimplifyMergeTree {
        public:
            explicit PySplatSimplifyMergeTree(std::shared_ptr<core::SplatSimplifyMergeTree> owner)
                : owner_(std::move(owner)) {
                if (!owner_)
                    throw std::runtime_error("Invalid SplatSimplifyMergeTree owner");
            }

            PyTensor source_means() const { return PyTensor(owner_->source_means, false); }
            PyTensor source_sh0() const { return PyTensor(owner_->source_sh0, false); }
            PyTensor source_shN() const { return PyTensor(owner_->source_shN, false); }
            PyTensor source_scaling() const { return PyTensor(owner_->source_scaling, false); }
            PyTensor source_rotation() const { return PyTensor(owner_->source_rotation, false); }
            PyTensor source_opacity() const { return PyTensor(owner_->source_opacity, false); }

            int source_active_sh_degree() const { return owner_->source_active_sh_degree; }
            int source_max_sh_degree() const { return owner_->source_max_sh_degree; }
            float source_scene_scale() const { return owner_->source_scene_scale; }

            int target_count() const { return owner_->target_count; }
            int post_prune_count() const { return owner_->post_prune_count; }
            double requested_ratio() const { return owner_->requested_ratio; }
            float requested_lod_base() const { return owner_->requested_lod_base; }
            float requested_opacity_prune_threshold() const { return owner_->requested_opacity_prune_threshold; }

            const std::vector<int32_t>& final_roots() const { return owner_->final_roots; }
            const std::vector<int32_t>& pruned_leaf_ids() const { return owner_->pruned_leaf_ids; }
            const std::vector<int32_t>& merge_left() const { return owner_->merge_left; }
            const std::vector<int32_t>& merge_right() const { return owner_->merge_right; }
            const std::vector<int32_t>& merge_pass() const { return owner_->merge_pass; }

            int leaf_count() const { return owner_->leaf_count(); }
            int merge_count() const { return owner_->merge_count(); }

        private:
            std::shared_ptr<core::SplatSimplifyMergeTree> owner_;
        };

        class PySplatSimplifyResult {
        public:
            PySplatSimplifyResult(std::shared_ptr<core::SplatData> splat_data,
                                  std::shared_ptr<core::SplatSimplifyMergeTree> merge_tree)
                : splat_data_(std::move(splat_data)), merge_tree_(std::move(merge_tree)) {
                if (!splat_data_ || !merge_tree_)
                    throw std::runtime_error("Invalid SplatSimplifyResult payload");
            }

            PySplatData splat_data() const { return PySplatData(splat_data_); }
            PySplatSimplifyMergeTree merge_tree() const { return PySplatSimplifyMergeTree(merge_tree_); }

        private:
            std::shared_ptr<core::SplatData> splat_data_;
            std::shared_ptr<core::SplatSimplifyMergeTree> merge_tree_;
        };

    } // namespace

    void register_splat_simplify(nb::module_& m) {
        nb::class_<PySplatSimplifyMergeTree>(m, "SplatSimplifyMergeTree")
            .def_prop_ro("source_means", &PySplatSimplifyMergeTree::source_means,
                         "Filtered source means tensor [N, 3]")
            .def_prop_ro("source_sh0", &PySplatSimplifyMergeTree::source_sh0,
                         "Filtered source SH0 tensor [N, 1, 3]")
            .def_prop_ro("source_shN", &PySplatSimplifyMergeTree::source_shN,
                         "Filtered source higher-order SH tensor [N, K, 3]")
            .def_prop_ro("source_scaling", &PySplatSimplifyMergeTree::source_scaling,
                         "Filtered source scaling tensor [N, 3] in log-space")
            .def_prop_ro("source_rotation", &PySplatSimplifyMergeTree::source_rotation,
                         "Filtered source rotation tensor [N, 4]")
            .def_prop_ro("source_opacity", &PySplatSimplifyMergeTree::source_opacity,
                         "Filtered source opacity tensor [N, 1] in logit-space")
            .def_prop_ro("source_active_sh_degree", &PySplatSimplifyMergeTree::source_active_sh_degree,
                         "Active SH degree of the filtered source splat")
            .def_prop_ro("source_max_sh_degree", &PySplatSimplifyMergeTree::source_max_sh_degree,
                         "Maximum SH degree of the filtered source splat")
            .def_prop_ro("source_scene_scale", &PySplatSimplifyMergeTree::source_scene_scale,
                         "Scene scale of the filtered source splat")
            .def_prop_ro("target_count", &PySplatSimplifyMergeTree::target_count,
                         "Requested target count before pruning")
            .def_prop_ro("post_prune_count", &PySplatSimplifyMergeTree::post_prune_count,
                         "Count remaining after opacity pruning")
            .def_prop_ro("requested_ratio", &PySplatSimplifyMergeTree::requested_ratio,
                         "Requested simplify ratio")
            .def_prop_ro("requested_lod_base", &PySplatSimplifyMergeTree::requested_lod_base,
                         "Requested LOD base factor")
            .def_prop_ro("requested_opacity_prune_threshold",
                         &PySplatSimplifyMergeTree::requested_opacity_prune_threshold,
                         "Requested opacity prune threshold")
            .def_prop_ro("final_roots", &PySplatSimplifyMergeTree::final_roots,
                         "Tree node ids that survive into the simplified output")
            .def_prop_ro("pruned_leaf_ids", &PySplatSimplifyMergeTree::pruned_leaf_ids,
                         "Leaf ids removed during the initial opacity prune")
            .def_prop_ro("merge_left", &PySplatSimplifyMergeTree::merge_left,
                         "Left child id for each merge node")
            .def_prop_ro("merge_right", &PySplatSimplifyMergeTree::merge_right,
                         "Right child id for each merge node")
            .def_prop_ro("merge_pass", &PySplatSimplifyMergeTree::merge_pass,
                         "Zero-based simplify pass index for each merge node")
            .def("leaf_count", &PySplatSimplifyMergeTree::leaf_count,
                 "Number of source leaves in the tree")
            .def("merge_count", &PySplatSimplifyMergeTree::merge_count,
                 "Number of merge nodes in the tree");

        nb::class_<PySplatSimplifyResult>(m, "SplatSimplifyResult")
            .def_prop_ro("splat_data", &PySplatSimplifyResult::splat_data,
                         "Simplified output splat data")
            .def_prop_ro("merge_tree", &PySplatSimplifyResult::merge_tree,
                         "Merge tree describing how the output was formed");

        m.def(
            "simplify_splats",
            [](const std::string& source_name,
               double ratio,
               float lod_base,
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
                opts.lod_base = lod_base;
                opts.opacity_prune_threshold = opacity_prune_threshold;
                invoke_splat_simplify_start(source_name, opts);
            },
            nb::arg("source_name"),
            nb::arg("ratio") = 0.1,
            nb::arg("lod_base") = 2.0f,
            nb::arg("opacity_prune_threshold") = 0.1f,
            "Simplify a splat node asynchronously and create a new output node.");

        m.def(
            "simplify_splat_data_with_history",
            [](const PySplatData& source,
               double ratio,
               float lod_base,
               float opacity_prune_threshold,
               nb::object progress) {
                core::SplatSimplifyOptions opts;
                opts.ratio = ratio;
                opts.lod_base = lod_base;
                opts.opacity_prune_threshold = opacity_prune_threshold;

                PyProgressCallback py_progress{std::move(progress)};
                core::SplatSimplifyProgressCallback progress_cb;
                if (py_progress.valid()) {
                    progress_cb = [py_progress](const float value, const std::string& stage) {
                        return py_progress(value, stage);
                    };
                }

                auto result = std::expected<core::SplatSimplifyResult, std::string>(
                    std::unexpected(std::string("unknown error")));
                {
                    nb::gil_scoped_release release;
                    result = core::simplify_splats_with_history(*source.data(), opts, std::move(progress_cb));
                }
                if (!result)
                    throw std::runtime_error(result.error());

                auto output = std::shared_ptr<core::SplatData>(std::move(result->splat));
                auto merge_tree = std::make_shared<core::SplatSimplifyMergeTree>(std::move(result->merge_tree));
                return PySplatSimplifyResult(std::move(output), std::move(merge_tree));
            },
            nb::arg("source"),
            nb::arg("ratio") = 0.1,
            nb::arg("lod_base") = 2.0f,
            nb::arg("opacity_prune_threshold") = 0.1f,
            nb::arg("progress") = nb::none(),
            "Synchronously simplify SplatData and return both the simplified output and its merge tree.");

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
