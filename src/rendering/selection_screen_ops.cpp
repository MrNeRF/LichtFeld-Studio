/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"
#include "selection_tensor_utils.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_spatial.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::GpuBackend;
        using lfs::core::GpuBackendScope;
        using lfs::core::Tensor;

        constexpr float kInvalidScreenFill = kInvalidScreenPositionThreshold * 100000.0f;

        struct ScreenAxes {
            Tensor x;
            Tensor y;
            Tensor valid;
        };

        ScreenAxes screen_axes(const Tensor& screen_positions) {
            ScreenAxes axes{
                .x = screen_positions.slice(1, 0, 1),
                .y = screen_positions.slice(1, 1, 2),
            };
            axes.valid = (axes.x >= kInvalidScreenPositionThreshold)
                             .logical_and(axes.y >= kInvalidScreenPositionThreshold);
            return axes;
        }

        using detail::groupInput;
        using detail::matchBackend;
        using detail::requireGpuBackend;

        void markScreenRegion(const Tensor& screen, Tensor& selection, lfs::core::PointRegion2D region,
                              const Tensor* geometry = nullptr) {
            if (!screen.is_valid() || screen.size(0) == 0 || !selection.is_valid() || selection.numel() == 0)
                return;
            region.minimum_coordinate = kInvalidScreenPositionThreshold;
            if (selection.device() == Device::CPU) {
                const auto positions = screen.cpu();
                const auto vertices = geometry ? geometry->cpu() : Tensor{};
                lfs::core::mark_points_2d(selection, positions, region, geometry ? &vertices : nullptr);
                return;
            }
            const auto backend = requireGpuBackend(selection, "screen selection requires a GPU mask");
            GpuBackendScope scope(backend);
            const auto positions = matchBackend(screen, backend);
            const auto vertices = geometry ? matchBackend(*geometry, backend) : Tensor{};
            lfs::core::mark_points_2d(selection, positions, region, geometry ? &vertices : nullptr);
        }

    } // namespace

    void rect_select_tensor(
        const Tensor& screen_positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        Tensor& selection_out) {
        markScreenRegion(screen_positions, selection_out,
                         {.kind = lfs::core::PointRegion2DKind::Rectangle, .x0 = x0, .y0 = y0, .x1 = x1, .y1 = y1});
    }

    void brush_select_tensor(const Tensor& screen_positions, const float mouse_x, const float mouse_y,
                             const float radius, Tensor& selection_out) {
        markScreenRegion(screen_positions, selection_out,
                         {.kind = lfs::core::PointRegion2DKind::Disk, .x0 = mouse_x, .y0 = mouse_y, .radius = radius});
    }

    void polygon_select_tensor(const Tensor& screen_positions, const Tensor& polygon_vertices, Tensor& selection_out) {
        if (!polygon_vertices.is_valid() || polygon_vertices.size(0) < 3)
            return;
        markScreenRegion(screen_positions, selection_out,
                         {.kind = lfs::core::PointRegion2DKind::Polygon}, &polygon_vertices);
    }

    int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        const float x,
        const float y,
        const float radius) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return -1;
        }
        if (screen_positions.device() != Device::GPU ||
            screen_positions.dtype() != DataType::Float32 ||
            screen_positions.ndim() != 2 ||
            screen_positions.size(1) != 2) {
            throw std::runtime_error("pick_projected_gaussian_tensor expects a GPU Float32 [N, 2] tensor");
        }
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor finite = axes.valid.logical_and(axes.x.isfinite()).logical_and(axes.y.isfinite());
        const Tensor dx = axes.x - x;
        const Tensor dy = axes.y - y;
        const Tensor dist_sq = (dx * dx + dy * dy)
                                   .masked_fill(finite.logical_not(), std::numeric_limits<float>::infinity())
                                   .flatten();
        const float best = dist_sq.min_scalar();
        if (!(best <= radius * radius)) {
            return -1;
        }
        // Equal distances pick the largest index.
        const auto candidates = (dist_sq == best).nonzero().flatten().to_vector_int();
        return *std::max_element(candidates.begin(), candidates.end());
    }

    void set_selection_element(Tensor& selection, const int index, const bool value) {
        if (!selection.is_valid() || index < 0 ||
            static_cast<size_t>(index) >= selection.numel()) {
            return;
        }
        if (selection.dtype() == DataType::Bool) {
            selection.slice(0, index, index + 1).fill_(value ? 1.f : 0.f, selection.stream());
            return;
        }
        const GpuBackendScope scope(lfs::core::gpu_backend_of(selection).value_or(GpuBackend::CUDA));
        const auto indices = Tensor::from_vector(std::vector<int>{index}, {1}, selection.device());
        selection.index_fill_(0, indices, value ? 1.f : 0.f);
    }

    void brush_select_disks_tensor(
        const Tensor& screen_positions,
        const std::vector<float>& disk_xy,
        const float radius,
        Tensor& selection_out) {
        if (disk_xy.size() < 2 || !screen_positions.is_valid() || screen_positions.size(0) == 0 ||
            !selection_out.is_valid() || selection_out.numel() == 0) {
            return;
        }
        const std::size_t disk_count = disk_xy.size() / 2;
        if (disk_count == 1) {
            brush_select_tensor(screen_positions, disk_xy[0], disk_xy[1], radius, selection_out);
            return;
        }
        const auto centers = Tensor::from_vector(
            std::vector<float>(disk_xy.begin(), disk_xy.begin() + disk_count * 2), {disk_count, 2}, Device::CPU);
        markScreenRegion(screen_positions, selection_out,
                         {.kind = lfs::core::PointRegion2DKind::Disks, .radius = radius}, &centers);
    }

    Tensor project_screen_positions_tensor(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const float center_x,
        const float center_y,
        const core::PointProjectionModel camera_model,
        const float ortho_scale,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices,
        const std::vector<bool>& node_visibility_mask) {
        if (!means.is_valid() || means.size(0) == 0) {
            return {};
        }
        if (means.device() != Device::GPU || means.dtype() != DataType::Float32 ||
            means.ndim() != 2 || means.size(1) != 3) {
            throw std::runtime_error("project_screen_positions_tensor expects a GPU Float32 [N, 3] means tensor");
        }
        if (width <= 0 || height <= 0) {
            return {};
        }
        const auto n = means.size(0);
        if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("screen position count exceeds int range");
        }
        const auto backend = requireGpuBackend(means, "project_screen_positions_tensor requires GPU means");
        GpuBackendScope scope(backend);
        Tensor matrices, indices, visibility;
        if (model_transforms && model_transforms->is_valid() && model_transforms->numel() > 0) {
            matrices = groupInput(*model_transforms, backend, DataType::Float32);
            if (matrices.numel() % 16 != 0) {
                throw std::runtime_error(
                    "model_transforms tensor must contain a multiple of 16 float values (N x 4 x 4).");
            }
        }
        if (transform_indices && transform_indices->is_valid() && transform_indices->numel() >= n) {
            indices = groupInput(*transform_indices, backend, DataType::Int32);
            if (indices.numel() > n) {
                indices = indices.slice(0, 0, static_cast<int>(n));
            }
            if (!node_visibility_mask.empty()) {
                visibility = Tensor::from_vector(node_visibility_mask, {node_visibility_mask.size()}, Device::GPU);
            }
        }
        using lfs::core::PointProjectionModel;
        const lfs::core::PointProjection projection{
            .rotation = view_rotation_rows,
            .translation = translation,
            .focal_x = pixel_focal_x,
            .focal_y = pixel_focal_y,
            .center_x = center_x,
            .center_y = center_y,
            .ortho_scale = ortho_scale,
            .width = width,
            .height = height,
            .model = camera_model,
            .invalid_value = kInvalidScreenFill,
        };
        return lfs::core::project_points(means, projection, &matrices, &indices, &visibility);
    }

} // namespace lfs::rendering
