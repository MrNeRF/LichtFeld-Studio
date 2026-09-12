/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

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
    } // namespace

    void rect_select_tensor(
        const Tensor& screen_positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor inside = axes.valid
                                  .logical_and(axes.x >= x0)
                                  .logical_and(axes.x <= x1)
                                  .logical_and(axes.y >= y0)
                                  .logical_and(axes.y <= y1)
                                  .flatten();
        selection_out.masked_fill_(inside, 1.0f);
    }

    int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        const float x,
        const float y,
        const float radius) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return -1;
        }
        if (screen_positions.device() != Device::CUDA ||
            screen_positions.dtype() != DataType::Float32 ||
            screen_positions.ndim() != 2 ||
            screen_positions.size(1) != 2) {
            throw std::runtime_error("pick_projected_gaussian_tensor expects a CUDA Float32 [N, 2] tensor");
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
        // Equal distances pick the largest index, as the kernel did.
        const auto candidates = (dist_sq == best).nonzero().flatten().to_vector_int();
        return *std::max_element(candidates.begin(), candidates.end());
    }

    void set_selection_element(Tensor& selection, const int index, const bool value) {
        if (!selection.is_valid() || index < 0 ||
            static_cast<size_t>(index) >= selection.numel()) {
            return;
        }
        if (lfs::core::gpu_backend_of(selection) == lfs::core::GpuBackend::Vulkan) {
            Tensor idx = Tensor::from_vector(std::vector<int>{index}, {1}, selection.device());
            selection.index_fill_(0, idx, value ? 1.0f : 0.0f);
            return;
        }
        set_selection_element(selection.ptr<bool>(), index, value);
    }

} // namespace lfs::rendering
