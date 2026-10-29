/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "io/video/video_frame_geometry.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace lfs::io::video {

    // Fits an HWC Float32 RGB camera image into width x height on the image's device and
    // backend. Destination pixel centers map through `mapping` (videoFrameMapping); bilinear
    // taps clamp to the image edge and samples outside the camera image are black.
    [[nodiscard]] inline core::Tensor reframeVideoFrame(const core::Tensor& rgb_hwc, const int width,
                                                        const int height, const VideoFrameMapping& mapping) {
        struct Axis {
            std::vector<int> first, second;
            std::vector<float> keep, take, inside;
        };
        const auto axis = [](const int size, const int source, const float scale, const float offset) {
            Axis result;
            for (int i = 0; i < size; ++i) {
                const float position = (static_cast<float>(i) + 0.5f) * scale + offset;
                const float clamped = std::clamp(position - 0.5f, 0.0f, static_cast<float>(source - 1));
                const int first = static_cast<int>(clamped);
                const float weight = clamped - static_cast<float>(first);
                result.first.push_back(first);
                result.second.push_back(std::min(first + 1, source - 1));
                result.keep.push_back(1.0f - weight);
                result.take.push_back(weight);
                result.inside.push_back(position >= 0.0f && position < static_cast<float>(source) ? 1.0f : 0.0f);
            }
            return result;
        };
        const auto rows = axis(height, static_cast<int>(rgb_hwc.size(0)), mapping.scale_y, mapping.offset_y);
        const auto columns = axis(width, static_cast<int>(rgb_hwc.size(1)), mapping.scale_x, mapping.offset_x);

        std::optional<core::GpuBackendScope> backend;
        if (const auto source_backend = core::gpu_backend_of(rgb_hwc))
            backend.emplace(*source_backend);
        const auto device = rgb_hwc.device();
        const auto indices = [device](const std::vector<int>& values) {
            return core::Tensor::from_vector(values, core::TensorShape{values.size()}, device);
        };
        const auto row_weights = [device](const std::vector<float>& values) {
            return core::Tensor::from_vector(values, core::TensorShape{values.size(), 1, 1}, device);
        };
        const auto column_weights = [device](const std::vector<float>& values) {
            return core::Tensor::from_vector(values, core::TensorShape{1, values.size(), 1}, device);
        };
        // Indices are clamped above, so gathers need no host-side bounds check.
        constexpr auto clamp = core::BoundaryMode::Clamp;
        const auto x0 = indices(columns.first);
        const auto x1 = indices(columns.second);
        const auto keep_x = column_weights(columns.keep);
        const auto take_x = column_weights(columns.take);
        const auto blend_columns = [&](const core::Tensor& image_rows) {
            return image_rows.index_select(1, x0, clamp)
                .mul(keep_x)
                .add(image_rows.index_select(1, x1, clamp).mul(take_x));
        };
        const auto source = rgb_hwc.contiguous();
        const auto top = blend_columns(source.index_select(0, indices(rows.first), clamp));
        const auto bottom = blend_columns(source.index_select(0, indices(rows.second), clamp));
        return top.mul(row_weights(rows.keep))
            .add(bottom.mul(row_weights(rows.take)))
            .mul(row_weights(rows.inside))
            .mul(column_weights(columns.inside))
            .contiguous();
    }

    // Fits a camera image rendered at videoCameraRenderSize(width, height, ...) into width x height.
    [[nodiscard]] inline core::Tensor reframeVideoFrame(const core::Tensor& rgb_hwc, const int width,
                                                        const int height, const VideoFrameFit fit) {
        return reframeVideoFrame(rgb_hwc, width, height,
                                 videoFrameMapping(static_cast<int>(rgb_hwc.size(1)), static_cast<int>(rgb_hwc.size(0)),
                                                   width, height, fit));
    }

} // namespace lfs::io::video
