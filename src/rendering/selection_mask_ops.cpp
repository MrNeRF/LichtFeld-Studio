/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "selection_tensor_utils.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_filters.hpp"
#include "core/tensor_labels.hpp"
#include "rendering/render_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::GpuBackend;
        using lfs::core::GpuBackendScope;
        using lfs::core::Tensor;

        using detail::groupInput;
        using detail::matchBackend;
        using detail::requireGpuBackend;

        [[nodiscard]] bool nodeMaskRestrictsSelection(const std::vector<bool>& mask) {
            return std::any_of(mask.begin(), mask.end(), [](const bool enabled) { return !enabled; });
        }

        [[nodiscard]] Tensor asBool(const Tensor& tensor) {
            if (tensor.dtype() == DataType::Bool) {
                return tensor;
            }
            return tensor != 0;
        }

        [[nodiscard]] Tensor meansNx3(const Tensor& means, const std::size_t n, const GpuBackend backend) {
            Tensor contig = matchBackend(means, backend);
            if (contig.dtype() != DataType::Float32) {
                contig = contig.to(DataType::Float32);
            }
            contig = contig.contiguous();
            if (contig.ndim() == 2 && contig.size(1) == 3 && contig.size(0) == n) {
                return contig;
            }
            if (contig.numel() == n * 3) {
                return contig.reshape(lfs::core::TensorShape({n, std::size_t{3}}));
            }
            throw std::runtime_error("selection mask ops expect means with shape [N, 3]");
        }

        void validate_locked_group_mask(const core::Tensor& mask, const core::GpuBackend backend) {
            if (mask.is_valid() && (mask.dtype() != core::DataType::Bool || mask.numel() != 256 ||
                                    !mask.is_contiguous() || core::gpu_backend_of(mask) != backend))
                throw std::invalid_argument(
                    "Selection group locks must be 256 dense boolean flags on the selection backend");
        }
    } // namespace

    namespace {
        void updateGroupLabels(const Tensor& selection, const Tensor* indices, const Tensor& existing_mask,
                               Tensor& output_mask, uint8_t group_id, const Tensor& locked_groups, bool add_mode,
                               const Tensor* transform_indices, const std::vector<bool>& valid_nodes,
                               bool replace_mode) {
            const auto backend = requireGpuBackend(selection, "selection group updates require a GPU mask");
            const GpuBackendScope scope(backend);
            validate_locked_group_mask(locked_groups, backend);
            const auto n = selection.numel();
            const auto m = indices ? output_mask.numel() : n;
            const auto selected = asBool(matchBackend(selection, backend).flatten());
            const auto old = existing_mask.is_valid() && existing_mask.numel() == m
                                 ? groupInput(existing_mask, backend, DataType::UInt8)
                                 : Tensor{};
            const auto ids = indices ? groupInput(*indices, backend, DataType::Int32) : Tensor{};
            Tensor categories, allowed;
            if (nodeMaskRestrictsSelection(valid_nodes) && transform_indices && transform_indices->is_valid() &&
                transform_indices->numel() == n) {
                categories = groupInput(*transform_indices, backend, DataType::Int32);
                allowed = Tensor::from_vector(valid_nodes, {valid_nodes.size()}, Device::GPU);
            }
            if (!output_mask.is_valid() || output_mask.numel() != m || output_mask.dtype() != DataType::UInt8 ||
                lfs::core::gpu_backend_of(output_mask) != backend)
                output_mask = Tensor::empty({m}, Device::GPU, DataType::UInt8);
            const auto mode = replace_mode ? lfs::core::LabelUpdateMode::Replace
                              : add_mode   ? lfs::core::LabelUpdateMode::Add
                                           : lfs::core::LabelUpdateMode::Remove;
            lfs::core::update_labels(output_mask, selected,
                                     {.label = group_id,
                                      .mode = mode,
                                      .existing = old.is_valid() ? &old : nullptr,
                                      .locked = locked_groups.is_valid() ? &locked_groups : nullptr,
                                      .indices = indices ? &ids : nullptr,
                                      .categories = categories.is_valid() ? &categories : nullptr,
                                      .allowed = allowed.is_valid() ? &allowed : nullptr});
        }
    } // namespace

    void apply_selection_group_tensor_mask(const Tensor& cumulative_selection, const Tensor& existing_mask,
                                           Tensor& output_mask, uint8_t group_id, const Tensor& locked_groups,
                                           bool add_mode, const Tensor* transform_indices,
                                           const std::vector<bool>& valid_nodes, bool replace_mode) {
        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0)
            return;
        updateGroupLabels(cumulative_selection, nullptr, existing_mask, output_mask, group_id, locked_groups, add_mode,
                          transform_indices, valid_nodes, replace_mode);
    }

    void apply_selection_group_indexed_tensor_mask(const Tensor& visible_selection, const Tensor& visible_indices,
                                                   const Tensor& existing_mask, Tensor& output_mask, uint8_t group_id,
                                                   const Tensor& locked_groups, bool add_mode,
                                                   const Tensor* transform_indices,
                                                   const std::vector<bool>& valid_nodes, bool replace_mode) {
        if (!visible_selection.is_valid() || !visible_indices.is_valid() || !output_mask.is_valid() ||
            visible_selection.size(0) == 0 || visible_indices.numel() != visible_selection.numel())
            return;
        updateGroupLabels(visible_selection, &visible_indices, existing_mask, output_mask, group_id, locked_groups,
                          add_mode, transform_indices, valid_nodes, replace_mode);
    }

    void filter_selection_by_node_mask(Tensor& selection, const Tensor& indices, const std::vector<bool>& valid) {
        if (!selection.is_valid() || !indices.is_valid() || valid.empty() || !nodeMaskRestrictsSelection(valid) ||
            selection.numel() == 0 || indices.numel() != selection.numel())
            return;
        const auto backend = requireGpuBackend(selection, "node filtering requires a GPU mask");
        const GpuBackendScope scope(backend);
        const auto ids = groupInput(indices, backend, DataType::Int32);
        const auto allowed = Tensor::from_vector(valid, {valid.size()}, Device::GPU);
        lfs::core::filter_points(selection, nullptr, {.indices = &ids, .allowed = &allowed});
    }

    namespace {
        void preparePointFilter(lfs::core::PointFilter& filter, Tensor& matrices, Tensor& ids, const Tensor* transforms,
                                const Tensor* indices, size_t n, GpuBackend backend) {
            if (transforms && transforms->is_valid() && transforms->numel()) {
                matrices = groupInput(*transforms, backend, DataType::Float32);
                filter.transforms = &matrices;
            }
            if (indices && indices->is_valid() && indices->numel() == n) {
                ids = groupInput(*indices, backend, DataType::Int32);
                filter.indices = &ids;
            }
        }
    } // namespace

    void filter_selection_by_crop(Tensor& selection, const Tensor& means, const Tensor* box_transform,
                                  const Tensor* box_min, const Tensor* box_max, bool box_inverse,
                                  const Tensor* ellipsoid_transform, const Tensor* ellipsoid_radii,
                                  bool ellipsoid_inverse, const Tensor* transforms, const Tensor* indices) {
        if (!selection.is_valid() || !means.is_valid() || selection.numel() == 0 || means.size(0) != selection.numel())
            return;
        const auto complete = [](const Tensor* t, size_t n) { return t && t->is_valid() && t->numel() >= n; };
        const bool box = complete(box_transform, 16) && complete(box_min, 3) && complete(box_max, 3);
        const bool ellipsoid = complete(ellipsoid_transform, 16) && complete(ellipsoid_radii, 3);
        if (!box && !ellipsoid)
            return;
        const auto backend = requireGpuBackend(selection, "crop filtering requires a GPU mask");
        const GpuBackendScope scope(backend);
        const auto points = meansNx3(means, selection.numel(), backend);
        Tensor matrices, ids, bt, bmin, bmax, et, radii;
        lfs::core::PointFilter filter{.box_inverse = box_inverse, .ellipsoid_inverse = ellipsoid_inverse};
        preparePointFilter(filter, matrices, ids, transforms, indices, selection.numel(), backend);
        const auto shape = [&](const Tensor& t, size_t n) {
            return groupInput(t, backend, DataType::Float32).slice(0, 0, n);
        };
        if (box) {
            bt = shape(*box_transform, 16);
            bmin = shape(*box_min, 3);
            bmax = shape(*box_max, 3);
            filter.box_transform = &bt;
            filter.box_min = &bmin;
            filter.box_max = &bmax;
        }
        if (ellipsoid) {
            et = shape(*ellipsoid_transform, 16);
            radii = shape(*ellipsoid_radii, 3);
            filter.ellipsoid_transform = &et;
            filter.ellipsoid_radii = &radii;
        }
        lfs::core::filter_points(selection, &points, filter);
    }

    void filter_selection_by_screen_window(Tensor& selection, const Tensor& means, const std::array<float, 9>& rotation,
                                           const std::array<float, 3>& translation, core::PointProjectionModel model,
                                           int width, int height, float fx, float fy, float cx, float cy, float ortho,
                                           float near_depth, float far_depth, float scale_x, float scale_y,
                                           float offset_x, float offset_y, const Tensor* transforms,
                                           const Tensor* indices) {
        if (!selection.is_valid() || !means.is_valid() || width <= 0 || height <= 0 || selection.numel() == 0 ||
            means.size(0) != selection.numel())
            return;
        const auto backend = requireGpuBackend(selection, "screen filtering requires a GPU mask");
        const GpuBackendScope scope(backend);
        const auto points = meansNx3(means, selection.numel(), backend);
        const float sanitized_ortho = std::isfinite(ortho) && ortho > 1.e-5f ? ortho : DEFAULT_ORTHO_SCALE;
        const lfs::core::PointFilterWindow window{.projection = {.rotation = rotation,
                                                                 .translation = translation,
                                                                 .focal_x = fx,
                                                                 .focal_y = fy,
                                                                 .center_x = cx,
                                                                 .center_y = cy,
                                                                 .ortho_scale = sanitized_ortho,
                                                                 .width = width,
                                                                 .height = height,
                                                                 .model = model},
                                                  .near_depth = near_depth,
                                                  .far_depth = far_depth,
                                                  .scale_x = scale_x,
                                                  .scale_y = scale_y,
                                                  .offset_x = offset_x,
                                                  .offset_y = offset_y};
        lfs::core::PointFilter filter{.window = &window};
        Tensor matrices, ids;
        preparePointFilter(filter, matrices, ids, transforms, indices, selection.numel(), backend);
        lfs::core::filter_points(selection, &points, filter);
    }

    void merge_selection_mask_or(Tensor& accumulated_mask, const Tensor& delta_mask) {
        if (!accumulated_mask.is_valid() || !delta_mask.is_valid() || accumulated_mask.numel() == 0 ||
            accumulated_mask.numel() != delta_mask.numel()) {
            return;
        }
        if (accumulated_mask.device() == Device::CPU) {
            accumulated_mask.copy_from(accumulated_mask | delta_mask);
            return;
        }
        const GpuBackend backend = requireGpuBackend(accumulated_mask, "merge_selection_mask_or requires a GPU mask");
        GpuBackendScope scope(backend);
        const Tensor delta = matchBackend(delta_mask, backend);
        if (accumulated_mask.dtype() == DataType::Bool && delta.dtype() == DataType::Bool) {
            accumulated_mask.masked_fill_(asBool(delta), 1.0f);
            return;
        }
        accumulated_mask.copy_from(accumulated_mask | delta);
    }

} // namespace lfs::rendering
