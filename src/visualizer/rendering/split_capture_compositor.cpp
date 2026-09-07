/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_capture_compositor.hpp"

#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "rendering_types.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

namespace lfs::vis {

    std::shared_ptr<lfs::core::Tensor> composeSplitCaptureCpu(
        const VulkanSplitViewParams& params,
        const glm::ivec2& output_size) {
        const auto load_panel = [](const std::shared_ptr<const lfs::core::Tensor>& image)
            -> std::optional<std::tuple<lfs::core::Tensor, int, int, int>> {
            if (!image || !image->is_valid() || image->ndim() != 3) {
                return std::nullopt;
            }
            const auto layout = lfs::rendering::detectImageLayout(*image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return std::nullopt;
            }
            lfs::core::Tensor tensor = *image;
            if (tensor.dtype() == lfs::core::DataType::UInt8) {
                tensor = tensor.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (tensor.dtype() != lfs::core::DataType::Float32) {
                tensor = tensor.to(lfs::core::DataType::Float32);
            }
            if (layout == lfs::rendering::ImageLayout::HWC) {
                tensor = tensor.permute({2, 0, 1}).contiguous();
            }
            tensor = tensor.cpu().contiguous();
            const int width = static_cast<int>(
                layout == lfs::rendering::ImageLayout::HWC ? image->size(1) : image->size(2));
            const int height = static_cast<int>(
                layout == lfs::rendering::ImageLayout::HWC ? image->size(0) : image->size(1));
            const int channels = static_cast<int>(
                layout == lfs::rendering::ImageLayout::HWC ? image->size(2) : image->size(0));
            if (width <= 0 || height <= 0 || channels < 3) {
                return std::nullopt;
            }
            return std::make_tuple(std::move(tensor), width, height, channels);
        };

        auto left_data = load_panel(params.left.image);
        auto right_data = load_panel(params.right.image);
        if (!left_data || !right_data || output_size.x <= 0 || output_size.y <= 0) {
            return {};
        }

        // CPU readbacks are tightly packed; padded GPU-pool UV limits do not
        // apply to these tensors.
        VulkanSplitViewPanel left_panel = params.left;
        VulkanSplitViewPanel right_panel = params.right;
        const auto apply_tight_uv = [](VulkanSplitViewPanel& panel, const int width, const int height) {
            panel.uv_scale = {1.0f, 1.0f};
            panel.uv_clamp_max = {
                (static_cast<float>(width) - 0.5f) / static_cast<float>(width),
                (static_cast<float>(height) - 0.5f) / static_cast<float>(height)};
        };
        apply_tight_uv(left_panel, std::get<1>(*left_data), std::get<2>(*left_data));
        apply_tight_uv(right_panel, std::get<1>(*right_data), std::get<2>(*right_data));

        const int width = output_size.x;
        const int height = output_size.y;
        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        std::vector<float> output(3 * pixel_count, 0.0f);
        for (std::size_t index = 0; index < pixel_count; ++index) {
            output[index] = params.background.r;
            output[pixel_count + index] = params.background.g;
            output[2 * pixel_count + index] = params.background.b;
        }

        const auto sample = [](const lfs::core::Tensor& tensor,
                               const int image_width,
                               const int image_height,
                               const float u,
                               const float v,
                               const int channel) {
            const float sample_x = std::clamp(u, 0.0f, 1.0f) * image_width - 0.5f;
            const float sample_y = std::clamp(v, 0.0f, 1.0f) * image_height - 0.5f;
            const int x0_unclamped = static_cast<int>(std::floor(sample_x));
            const int y0_unclamped = static_cast<int>(std::floor(sample_y));
            const int x0 = std::clamp(x0_unclamped, 0, image_width - 1);
            const int y0 = std::clamp(y0_unclamped, 0, image_height - 1);
            const int x1 = std::clamp(x0_unclamped + 1, 0, image_width - 1);
            const int y1 = std::clamp(y0_unclamped + 1, 0, image_height - 1);
            const float tx = sample_x - static_cast<float>(x0_unclamped);
            const float ty = sample_y - static_cast<float>(y0_unclamped);
            const float* data = tensor.ptr<float>();
            const auto at = [=](const int x, const int y) {
                return data[(static_cast<std::size_t>(channel) * image_height + y) *
                                image_width +
                            x];
            };
            return std::lerp(std::lerp(at(x0, y0), at(x1, y0), tx),
                             std::lerp(at(x0, y1), at(x1, y1), tx), ty);
        };

        const int rect_x = std::clamp(params.content_rect.x, 0, width);
        const int rect_y = std::clamp(params.content_rect.y, 0, height);
        const int rect_w = std::clamp(params.content_rect.z, 0, width - rect_x);
        const int rect_h = std::clamp(params.content_rect.w, 0, height - rect_y);
        if (rect_w <= 0 || rect_h <= 0) {
            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {3, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        const int divider = rect_x + splitViewDividerPixel(rect_w, params.split_position);
        const float split_x = static_cast<float>(rect_x) +
                              std::clamp(params.split_position, 0.0f, 1.0f) * rect_w;
        const float center_y = static_cast<float>(rect_y) + rect_h * 0.5f;
        constexpr glm::vec3 kDividerColor(0.29f, 0.33f, 0.42f);
        constexpr float kMinBarWidthPx = 4.0f;
        constexpr float kHandleHeightPx = 80.0f;
        constexpr float kHandleWidthPx = 24.0f;
        constexpr float kCornerRadiusPx = 6.0f;
        constexpr float kGripSpacingPx = 10.0f;
        constexpr float kGripWidthPx = 2.0f;
        constexpr float kGripLengthPx = 12.0f;
        constexpr int kGripLineCount = 2;

        const auto write = [&](const std::size_t index, const glm::vec3 color) {
            output[index] = color.r;
            output[pixel_count + index] = color.g;
            output[2 * pixel_count + index] = color.b;
        };

        for (int y = rect_y; y < rect_y + rect_h; ++y) {
            const float v = splitViewPixelCenterUv(y, rect_y, rect_h);
            for (int x = rect_x; x < rect_x + rect_w; ++x) {
                const float u = splitViewPixelCenterUv(x, rect_x, rect_w);
                const bool use_left = x < divider;
                const auto& panel = use_left ? left_panel : right_panel;
                const auto& panel_data = use_left ? *left_data : *right_data;
                const int panel_width = std::get<1>(panel_data);
                const int panel_height = std::get<2>(panel_data);
                const lfs::core::Tensor& panel_tensor = std::get<0>(panel_data);
                const std::size_t output_index = static_cast<std::size_t>(y) * width + x;

                if (params.exact_texel_sampling) {
                    const int local_x = x - rect_x;
                    const int local_y = y - rect_y;
                    const int texel_x = std::clamp(local_x, 0, panel_width - 1);
                    const int texel_y = std::clamp(
                        panel.flip_y ? rect_h - 1 - local_y : local_y,
                        0,
                        panel_height - 1);
                    const float* data = panel_tensor.ptr<float>();
                    const auto fetch = [&](const int channel) {
                        return data[(static_cast<std::size_t>(channel) * panel_height + texel_y) *
                                        panel_width +
                                    texel_x];
                    };
                    write(output_index, {fetch(0), fetch(1), fetch(2)});
                } else {
                    float panel_u = u;
                    if (panel.normalize_x_to_panel) {
                        const float span = std::max(
                            panel.end_position - panel.start_position, 1.0e-6f);
                        panel_u = (u - panel.start_position) / span;
                    }
                    const float panel_v = panel.flip_y ? 1.0f - v : v;
                    const glm::vec2 clamp_max = glm::clamp(
                        panel.uv_clamp_max, glm::vec2(0.0f), glm::vec2(1.0f));
                    const glm::vec2 texture_uv = glm::min(
                        (glm::vec2(panel_u, panel_v) * panel.texcoord_scale +
                         panel.texcoord_offset) *
                            panel.uv_scale,
                        clamp_max);
                    const auto sample_color = [&](const glm::vec2 uv) {
                        return glm::vec3{
                            sample(panel_tensor, panel_width, panel_height, uv.x, uv.y, 0),
                            sample(panel_tensor, panel_width, panel_height, uv.x, uv.y, 1),
                            sample(panel_tensor, panel_width, panel_height, uv.x, uv.y, 2)};
                    };
                    glm::vec3 color = sample_color(texture_uv);
                    if (panel.spatial_filter) {
                        constexpr float kStrength = 0.18f;
                        const float texel_x = 1.0f / std::max(panel_width, 1);
                        const float texel_y = 1.0f / std::max(panel_height, 1);
                        const auto clamp_uv = [&](const glm::vec2 uv) {
                            return glm::clamp(uv, glm::vec2(0.0f), clamp_max);
                        };
                        const glm::vec3 left = sample_color(clamp_uv(texture_uv - glm::vec2(texel_x, 0.0f)));
                        const glm::vec3 right = sample_color(clamp_uv(texture_uv + glm::vec2(texel_x, 0.0f)));
                        const glm::vec3 up = sample_color(clamp_uv(texture_uv - glm::vec2(0.0f, texel_y)));
                        const glm::vec3 down = sample_color(clamp_uv(texture_uv + glm::vec2(0.0f, texel_y)));
                        const glm::vec3 sharpened = color * (1.0f + 4.0f * kStrength) -
                                                    (left + right + up + down) * kStrength;
                        color = glm::clamp(
                            sharpened,
                            glm::min(color, glm::min(glm::min(left, right), glm::min(up, down))),
                            glm::max(color, glm::max(glm::max(left, right), glm::max(up, down))));
                    }
                    write(output_index, color);
                }

                const float distance_from_split =
                    std::abs(static_cast<float>(x) + 0.5f - split_x);
                if (distance_from_split < kMinBarWidthPx * 0.5f) {
                    glm::vec3 color = kDividerColor;
                    const float distance_from_center =
                        std::abs(static_cast<float>(y) + 0.5f - center_y);
                    const float handle_height = std::min(kHandleHeightPx, static_cast<float>(rect_h));
                    const float handle_width = std::min(kHandleWidthPx, static_cast<float>(rect_w));
                    if (distance_from_center < handle_height * 0.5f &&
                        distance_from_split < handle_width * 0.5f) {
                        const float corner_radius = std::min(
                            kCornerRadiusPx, std::min(handle_width, handle_height) * 0.5f);
                        const glm::vec2 corner_distance =
                            glm::vec2(distance_from_split, distance_from_center) -
                            (glm::vec2(handle_width, handle_height) * 0.5f -
                             glm::vec2(corner_radius));
                        if (corner_distance.x <= 0.0f || corner_distance.y <= 0.0f ||
                            glm::length(corner_distance) <= corner_radius) {
                            color = kDividerColor * 0.8f;
                            const float local_y = static_cast<float>(y) + 0.5f - center_y;
                            for (int line = -kGripLineCount; line <= kGripLineCount; ++line) {
                                if (std::abs(local_y - line * kGripSpacingPx) < kGripWidthPx &&
                                    distance_from_split < kGripLengthPx * 0.5f) {
                                    color = glm::vec3(0.9f);
                                    break;
                                }
                            }
                        }
                    }
                    write(output_index, color);
                }
            }
        }

        auto tensor = lfs::core::Tensor::from_vector(
            output,
            {3, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

} // namespace lfs::vis
