/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "display_tensors.hpp"

#include "rendering/image_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr float kLoQuantile = 0.02f;
        constexpr float kHiQuantile = 0.98f;
        constexpr float kFarDepthLimit = 1.0e9f;

        struct PaletteStop {
            float edge;
            glm::vec3 from;
            glm::vec3 to;
        };

        // Segment i covers [edge_{i-1}, edge_i) with a smoothstep mix of its two colors.
        constexpr std::array<PaletteStop, 5> kPalette{{
            {0.20f, {0.050f, 0.040f, 0.150f}, {0.060f, 0.195f, 0.500f}},
            {0.43f, {0.060f, 0.195f, 0.500f}, {0.000f, 0.500f, 0.650f}},
            {0.67f, {0.000f, 0.500f, 0.650f}, {0.360f, 0.735f, 0.410f}},
            {0.86f, {0.360f, 0.735f, 0.410f}, {0.965f, 0.820f, 0.300f}},
            {1.00f, {0.965f, 0.820f, 0.300f}, {0.985f, 0.430f, 0.125f}},
        }};

        Tensor on_gpu(const Tensor& tensor) {
            return tensor.device() == Device::CUDA ? tensor.contiguous() : tensor.to(Device::CUDA).contiguous();
        }

        Tensor smoothstep(const Tensor& x, const float edge0, const float edge1) {
            const Tensor t = (x - edge0).div(edge1 - edge0).clamp(0.0f, 1.0f);
            return t * t * (t * -2.0f + 3.0f);
        }

        // glm::mix(from, to, s) = from * (1 - s) + to * s, per channel.
        std::array<Tensor, 3> mix(const glm::vec3& from, const glm::vec3& to, const Tensor& s) {
            const Tensor one_minus = s * -1.0f + 1.0f;
            return {one_minus * from.r + s * to.r,
                    one_minus * from.g + s * to.g,
                    one_minus * from.b + s * to.b};
        }

        std::array<Tensor, 3> paletteColors(const Tensor& near_t) {
            float lower = 0.0f;
            std::array<Tensor, 3> color;
            for (size_t i = 0; i < kPalette.size(); ++i) {
                const PaletteStop& stop = kPalette[i];
                std::array<Tensor, 3> segment = mix(stop.from, stop.to, smoothstep(near_t, lower, stop.edge));
                if (i == 0) {
                    color = std::move(segment);
                } else {
                    const Tensor below = near_t < lower;
                    for (size_t c = 0; c < 3; ++c) {
                        color[c] = Tensor::where(below, color[c], segment[c]);
                    }
                }
                lower = stop.edge;
            }
            return color;
        }

        std::pair<float, float> robustDepthRange(const Tensor& depth, const Tensor& valid) {
            const Tensor values = depth.masked_select(valid);
            const size_t count = values.numel();
            if (count < 2) {
                return {0.0f, 0.0f};
            }
            const Tensor sorted = values.sort(0, false).first;
            const auto quantile = [&](const float q) {
                const auto n = static_cast<size_t>(q * static_cast<float>(count - 1));
                return sorted.slice(0, n, n + 1).to_vector()[0];
            };
            return {quantile(kLoQuantile), quantile(kHiQuantile)};
        }

        std::shared_ptr<Tensor> assembleChw(const std::array<Tensor, 3>& channels, const int height, const int width) {
            std::vector<Tensor> planes;
            planes.reserve(3);
            for (const Tensor& channel : channels) {
                planes.push_back(channel.reshape({1, height, width}));
            }
            return std::make_shared<Tensor>(Tensor::cat(planes, 0).cpu().contiguous());
        }
    } // namespace

    glm::vec3 depthPaletteForDisplay(float near_t) {
        near_t = std::clamp(near_t, 0.0f, 1.0f);
        float lower = 0.0f;
        for (const PaletteStop& stop : kPalette) {
            if (near_t < stop.edge || &stop == &kPalette.back()) {
                return glm::mix(stop.from, stop.to, glm::smoothstep(lower, stop.edge, near_t));
            }
            lower = stop.edge;
        }
        return kPalette.back().to;
    }

    std::shared_ptr<Tensor> makeDepthDisplayTensor(
        const Tensor& depth,
        const lfs::rendering::DepthVisualizationMode depth_visualization_mode,
        const glm::vec3& background_color) {
        if (!depth.is_valid() || depth.ndim() != 2) {
            return {};
        }
        const int height = static_cast<int>(depth.size(0));
        const int width = static_cast<int>(depth.size(1));
        if (width <= 0 || height <= 0) {
            return {};
        }
        const Tensor d = on_gpu(depth).to(DataType::Float32);
        const Tensor valid = d.isfinite().logical_and(d > 0.0f).logical_and(d < kFarDepthLimit);
        const auto [range_lo, range_hi] = robustDepthRange(d, valid);
        const float range_span = range_hi - range_lo;
        const bool background_only = !(range_span > 1.0e-6f);

        std::array<Tensor, 3> color;
        if (background_only) {
            const Tensor plane = Tensor::full({static_cast<size_t>(height), static_cast<size_t>(width)}, 0.0f, Device::CPU);
            color = {plane + background_color.r, plane + background_color.g, plane + background_color.b};
            return assembleChw(color, height, width);
        }
        const Tensor near_t = (d - range_lo).div(range_span).clamp(0.0f, 1.0f) * -1.0f + 1.0f;
        if (depth_visualization_mode == lfs::rendering::DepthVisualizationMode::Grayscale) {
            color = {near_t, near_t, near_t};
        } else {
            color = paletteColors(near_t);
        }
        const Tensor missing = valid.logical_not();
        const std::array<float, 3> background{background_color.r, background_color.g, background_color.b};
        for (size_t c = 0; c < 3; ++c) {
            color[c] = color[c].masked_fill(missing, background[c]);
        }
        return assembleChw(color, height, width);
    }

    std::shared_ptr<Tensor> makeNormalDisplayTensor(const Tensor& normal) {
        if (!normal.is_valid() || normal.ndim() != 3) {
            return {};
        }
        const auto layout = lfs::rendering::detectImageLayout(normal);
        if (layout == lfs::rendering::ImageLayout::Unknown ||
            lfs::rendering::imageChannels(normal, layout) < 3) {
            return {};
        }
        Tensor n = on_gpu(normal).to(DataType::Float32);
        if (layout == lfs::rendering::ImageLayout::HWC) {
            n = n.permute({2, 0, 1}).contiguous();
        }
        n = n.slice(0, 0, 3).contiguous();
        const int height = static_cast<int>(n.size(1));
        const int width = static_cast<int>(n.size(2));
        if (width <= 0 || height <= 0) {
            return {};
        }
        const Tensor length = (n * n).sum(0).sqrt();
        const Tensor degenerate = length.isfinite().logical_and(length > 1.0e-6f).logical_not().unsqueeze(0);
        const Tensor color = n.div(length.unsqueeze(0)).mul(0.5f).add(0.5f).clamp(0.0f, 1.0f).masked_fill(degenerate, 0.5f);
        return std::make_shared<Tensor>(color.cpu().contiguous());
    }

} // namespace lfs::vis
