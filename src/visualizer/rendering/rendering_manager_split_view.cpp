/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering_manager_split_view.hpp"

#include "core/camera.hpp"
#include "core/tensor.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/rendering.hpp"
#include "viewport_request_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace lfs::vis {
    glm::ivec2 gtComparisonPreviewSize(const lfs::core::Camera& camera,
                                       const glm::ivec2 viewport_size) {
        int base_width = camera.camera_width();
        int base_height = camera.camera_height();
        if (base_width <= 0 || base_height <= 0) {
            base_width = camera.image_width();
            base_height = camera.image_height();
        }
        if (camera.camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
            camera.is_undistort_precomputed()) {
            const auto& undistort = camera.undistort_params();
            base_width = undistort.dst_width;
            base_height = undistort.dst_height;
        }

        base_width = std::max(base_width, 1);
        base_height = std::max(base_height, 1);
        const int bound_width = std::max(viewport_size.x, 1);
        const int bound_height = std::max(viewport_size.y, 1);
        const bool width_limited = static_cast<std::int64_t>(base_width) * bound_height >=
                                   static_cast<std::int64_t>(base_height) * bound_width;
        const int max_dimension = std::max(width_limited ? bound_width : bound_height, 1);
        if (base_width <= max_dimension && base_height <= max_dimension)
            return {base_width, base_height};
        if (base_width >= base_height) {
            return {max_dimension,
                    std::max(static_cast<int>(static_cast<std::int64_t>(max_dimension) * base_height /
                                              base_width),
                             1)};
        }
        return {std::max(static_cast<int>(static_cast<std::int64_t>(max_dimension) * base_width /
                                          base_height),
                         1),
                max_dimension};
    }

    std::shared_ptr<lfs::core::Tensor> makeGTComparePlaceholderTensor(
        glm::ivec2 size, const glm::vec3 tint) {
        size.x = std::max(size.x, 1);
        size.y = std::max(size.y, 1);
        const std::size_t pixel_count =
            static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y);
        std::vector<float> output(3 * pixel_count, 0.0f);
        for (int y = 0; y < size.y; ++y) {
            for (int x = 0; x < size.x; ++x) {
                const bool stripe = ((x / 18) + (y / 18)) % 2 == 0;
                const glm::vec3 color = glm::clamp(
                    tint + glm::vec3(stripe ? 0.045f : 0.0f), glm::vec3(0.0f), glm::vec3(1.0f));
                const std::size_t idx = static_cast<std::size_t>(y) * size.x + x;
                output[idx] = color.r;
                output[pixel_count + idx] = color.g;
                output[2 * pixel_count + idx] = color.b;
            }
        }
        auto tensor = lfs::core::Tensor::from_vector(
            output,
            {std::size_t{3}, static_cast<std::size_t>(size.y), static_cast<std::size_t>(size.x)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    std::shared_ptr<lfs::core::Tensor> makeNormalDisplayFromDepthTensor(
        const lfs::core::Tensor& depth,
        const lfs::rendering::CameraIntrinsics& intrinsics) {
        if (!depth.is_valid() || depth.ndim() != 2 ||
            intrinsics.focal_x <= 0.0f || intrinsics.focal_y <= 0.0f) {
            return {};
        }
        auto depth_cpu = depth.cpu().contiguous();
        const int height = static_cast<int>(depth_cpu.size(0));
        const int width = static_cast<int>(depth_cpu.size(1));
        if (width <= 2 || height <= 2)
            return {};

        constexpr float kMinExpectedDepth = 1.0e-6f;
        constexpr float kMaxRelativeDepthJump = 0.05f;
        constexpr float kMinCrossNormSq = 1.0e-24f;
        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        std::vector<float> output(3 * pixel_count, 0.5f);
        const float* const source = depth_cpu.ptr<float>();
        if (!source)
            return {};
        const auto valid_depth = [](const float value) {
            return std::isfinite(value) && value >= kMinExpectedDepth && value < 1.0e9f;
        };
        const auto ray = [&](const int x, const int y) {
            return glm::vec3((static_cast<float>(x) + 0.5f - intrinsics.center_x) / intrinsics.focal_x,
                             (static_cast<float>(y) + 0.5f - intrinsics.center_y) / intrinsics.focal_y,
                             1.0f);
        };
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * width + x;
                const float center = source[index];
                if (!valid_depth(center))
                    continue;
                const float dxp = source[index + 1];
                const float dxm = source[index - 1];
                const float dyp = source[index + width];
                const float dym = source[index - width];
                const float jump_limit = kMaxRelativeDepthJump * center;
                if (!valid_depth(dxp) || !valid_depth(dxm) || !valid_depth(dyp) || !valid_depth(dym) ||
                    std::abs(dxp - center) > jump_limit || std::abs(dxm - center) > jump_limit ||
                    std::abs(dyp - center) > jump_limit || std::abs(dym - center) > jump_limit) {
                    continue;
                }
                const glm::vec3 tx = dxp * ray(x + 1, y) - dxm * ray(x - 1, y);
                const glm::vec3 ty = dyp * ray(x, y + 1) - dym * ray(x, y - 1);
                glm::vec3 normal = glm::cross(tx, ty);
                const float norm_sq = glm::dot(normal, normal);
                if (!std::isfinite(norm_sq) || norm_sq < kMinCrossNormSq)
                    continue;
                if (glm::dot(normal, ray(x, y)) > 0.0f)
                    normal = -normal;
                const glm::vec3 color = glm::clamp(glm::normalize(normal) * 0.5f + glm::vec3(0.5f),
                                                   glm::vec3(0.0f), glm::vec3(1.0f));
                output[index] = color.r;
                output[pixel_count + index] = color.g;
                output[2 * pixel_count + index] = color.b;
            }
        }
        auto tensor = lfs::core::Tensor::from_vector(
            output, {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    std::vector<ViewportInteractionPanel> buildSplitViewInteractionPanels(
        const Viewport& viewport, const RenderSettings& settings,
        const glm::vec2 screen_viewport_pos, const glm::vec2 screen_viewport_size) {
        std::vector<ViewportInteractionPanel> panels;
        if (screen_viewport_size.x <= 0.0f || screen_viewport_size.y <= 0.0f)
            return panels;

        const int full_screen_height = std::max(static_cast<int>(std::lround(screen_viewport_size.y)), 1);
        panels.push_back({
            .panel = SplitViewPanelId::Left,
            .viewport_data =
                {.rotation = viewport.getRotationMatrix(),
                 .translation = viewport.getTranslation(),
                 .size = {std::max(static_cast<int>(std::lround(screen_viewport_size.x)), 1),
                          full_screen_height},
                 .focal_length_mm = settings.focal_length_mm,
                 .orthographic = settings.orthographic,
                 .ortho_scale = settings.ortho_scale},
            .viewport_pos = screen_viewport_pos,
            .viewport_size = screen_viewport_size,
        });
        return panels;
    }

    lfs::rendering::ViewportRenderRequest buildPlyComparisonRenderRequest(
        const FrameContext& context, const glm::ivec2 panel_size,
        const Viewport& source_viewport, const SplitViewPanelId panel_id,
        const glm::ivec2 full_size) {
        const auto layouts = makePlyComparisonPanelLayouts(full_size.x, context.settings.split_position);
        const auto& layout = layouts[splitViewPanelIndex(panel_id)];
        return buildViewportRenderRequest(
            context, panel_size, &source_viewport, panel_id,
            {layout.panel.x, 0}, full_size);
    }

    SplitCompositeContentRect resolveSplitCompositeContentRect(
        const glm::ivec2 output_size, const bool letterbox, const glm::ivec2 content_size) {
        SplitCompositeContentRect rect{.width = output_size.x, .height = output_size.y};
        if (!letterbox || content_size.x <= 0 || content_size.y <= 0 ||
            output_size.x <= 0 || output_size.y <= 0) {
            return rect;
        }

        const float content_aspect =
            static_cast<float>(content_size.x) / static_cast<float>(content_size.y);
        const float output_aspect =
            static_cast<float>(output_size.x) / static_cast<float>(output_size.y);
        if (content_aspect > output_aspect) {
            rect.width = output_size.x;
            rect.height = std::max(
                static_cast<int>(std::lround(static_cast<float>(output_size.x) / content_aspect)), 1);
            rect.y = std::max((output_size.y - rect.height) / 2, 0);
        } else {
            rect.height = output_size.y;
            rect.width = std::max(
                static_cast<int>(std::lround(static_cast<float>(output_size.y) * content_aspect)), 1);
            rect.x = std::max((output_size.x - rect.width) / 2, 0);
        }
        rect.width = std::clamp(rect.width, 1, output_size.x);
        rect.height = std::clamp(rect.height, 1, output_size.y);
        return rect;
    }

    lfs::rendering::FrameMetadata makeSplitMetadata(
        const lfs::rendering::FrameMetadata& left,
        const lfs::rendering::FrameMetadata& right,
        const float split_position) {
        return {
            .viewer_backend = left.viewer_backend == lfs::rendering::ViewerBackend::Cuda
                                  ? right.viewer_backend.value_or(lfs::rendering::ViewerBackend::Cuda)
                                  : (left.viewer_backend ? left.viewer_backend : right.viewer_backend),
            .depth_panels =
                {lfs::rendering::FramePanelMetadata{
                     .depth = left.depth_panel_count > 0 ? left.depth_panels[0].depth : nullptr,
                     .start_position = 0.0f,
                     .end_position = split_position},
                 lfs::rendering::FramePanelMetadata{
                     .depth = right.depth_panel_count > 0 ? right.depth_panels[0].depth : nullptr,
                     .start_position = split_position,
                     .end_position = 1.0f}},
            .depth_panel_count = 2,
            .valid = true,
            .far_plane = left.valid ? left.far_plane : right.far_plane,
            .orthographic = left.valid ? left.orthographic : right.orthographic,
        };
    }

    SplitViewInfo makeGTSplitViewInfo(const GTComparisonMode mode,
                                      const std::string& image_name) {
        const char* mode_label = "GT Compare";
        const char* left_name = "Ground Truth";
        const char* right_name = "Rendered";
        if (mode == GTComparisonMode::Depth) {
            mode_label = "GT Depth Compare";
            left_name = "GT Depth";
            right_name = "Rendered Depth";
        } else if (mode == GTComparisonMode::Normal) {
            mode_label = "GT Normal Compare";
            left_name = "GT Normal";
            right_name = "Rendered Normal";
        } else if (mode == GTComparisonMode::Loss) {
            mode_label = "GT Loss";
            left_name = "Loss";
            right_name = "";
        }
        return {.enabled = true,
                .mode_label = mode_label,
                .detail_label = image_name,
                .left_name = left_name,
                .right_name = right_name};
    }

    std::shared_ptr<lfs::core::Tensor> composeSplitViewCpuImage(
        const SplitViewCpuDesc& params, const glm::ivec2 output_size) {
        const auto load_panel = [](const std::shared_ptr<const lfs::core::Tensor>& image)
            -> std::optional<std::tuple<lfs::core::Tensor, int, int, int>> {
            if (!image || !image->is_valid() || image->ndim() != 3)
                return std::nullopt;
            const auto layout = lfs::rendering::detectImageLayout(*image);
            if (layout == lfs::rendering::ImageLayout::Unknown)
                return std::nullopt;
            lfs::core::Tensor tensor = *image;
            if (tensor.dtype() == lfs::core::DataType::UInt8)
                tensor = tensor.to(lfs::core::DataType::Float32) / 255.0f;
            else if (tensor.dtype() != lfs::core::DataType::Float32)
                tensor = tensor.to(lfs::core::DataType::Float32);
            if (layout == lfs::rendering::ImageLayout::HWC)
                tensor = tensor.permute({2, 0, 1}).contiguous();
            tensor = tensor.cpu().contiguous();
            const int width = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(1)
                                                   : image->size(2));
            const int height = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                    ? image->size(0)
                                                    : image->size(1));
            const int channels = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                      ? image->size(2)
                                                      : image->size(0));
            return std::make_tuple(std::move(tensor), width, height, channels);
        };
        auto left_data = load_panel(params.left.image);
        auto right_data = load_panel(params.right.image);
        if (!left_data || !right_data)
            return {};

        SplitViewCpuPanelDesc left_panel = params.left;
        SplitViewCpuPanelDesc right_panel = params.right;
        const auto apply_tight_cpu_uv = [](SplitViewCpuPanelDesc& panel, const int width, const int height) {
            const int texture_width = std::max(width, 1);
            const int texture_height = std::max(height, 1);
            panel.uv_scale = {1.0f, 1.0f};
            panel.uv_clamp_max = {
                (static_cast<float>(texture_width) - 0.5f) / static_cast<float>(texture_width),
                (static_cast<float>(texture_height) - 0.5f) / static_cast<float>(texture_height)};
        };
        apply_tight_cpu_uv(left_panel, std::get<1>(*left_data), std::get<2>(*left_data));
        apply_tight_cpu_uv(right_panel, std::get<1>(*right_data), std::get<2>(*right_data));

        const int width = output_size.x;
        const int height = output_size.y;
        if (width <= 0 || height <= 0)
            return {};
        const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
        std::vector<float> output(3 * pixel_count, 0.0f);
        for (std::size_t i = 0; i < pixel_count; ++i) {
            output[i] = params.background.r;
            output[pixel_count + i] = params.background.g;
            output[2 * pixel_count + i] = params.background.b;
        }

        const auto sample = [](const lfs::core::Tensor& tensor, const int sample_width,
                               const int sample_height, const float u, const float v,
                               const int channel) {
            const float sample_x = std::clamp(u, 0.0f, 1.0f) * static_cast<float>(sample_width) - 0.5f;
            const float sample_y = std::clamp(v, 0.0f, 1.0f) * static_cast<float>(sample_height) - 0.5f;
            const int x0_unclamped = static_cast<int>(std::floor(sample_x));
            const int y0_unclamped = static_cast<int>(std::floor(sample_y));
            const int x0 = std::clamp(x0_unclamped, 0, sample_width - 1);
            const int y0 = std::clamp(y0_unclamped, 0, sample_height - 1);
            const int x1 = std::clamp(x0_unclamped + 1, 0, sample_width - 1);
            const int y1 = std::clamp(y0_unclamped + 1, 0, sample_height - 1);
            const float tx = sample_x - static_cast<float>(x0_unclamped);
            const float ty = sample_y - static_cast<float>(y0_unclamped);
            const float* data = tensor.ptr<float>();
            const auto at = [data, sample_width, sample_height, channel](const int x, const int y) {
                return data[(static_cast<std::size_t>(channel) * sample_height + y) * sample_width + x];
            };
            return std::lerp(std::lerp(at(x0, y0), at(x1, y0), tx),
                             std::lerp(at(x0, y1), at(x1, y1), tx), ty);
        };

        const int rect_x = params.content_rect.x;
        const int rect_y = params.content_rect.y;
        const int rect_width = std::max(params.content_rect.z, 1);
        const int rect_height = std::max(params.content_rect.w, 1);
        const int divider = rect_x + splitViewDividerPixel(rect_width, params.split_position);
        const float split_x = static_cast<float>(rect_x) +
                              std::clamp(params.split_position, 0.0f, 1.0f) *
                                  static_cast<float>(rect_width);
        const float center_y = static_cast<float>(rect_y) + static_cast<float>(rect_height) * 0.5f;

        constexpr glm::vec3 kDividerColor(0.29f, 0.33f, 0.42f);
        constexpr float kMinBarWidthPx = 4.0f;
        constexpr float kHandleHeightPx = 80.0f;
        constexpr float kHandleWidthPx = 24.0f;
        constexpr float kCornerRadiusPx = 6.0f;
        constexpr float kGripSpacingPx = 10.0f;
        constexpr float kGripWidthPx = 2.0f;
        constexpr float kGripLengthPx = 12.0f;
        constexpr int kGripLineCount = 2;

        const auto write = [&](const std::size_t idx, const glm::vec3& color) {
            output[idx] = color.r;
            output[pixel_count + idx] = color.g;
            output[2 * pixel_count + idx] = color.b;
        };
        const auto sample_panel_color = [&](const SplitViewCpuPanelDesc& panel,
                                            const auto& data, const float u, const float v) {
            float panel_u = u;
            if (panel.normalize_x_to_panel) {
                const float span = std::max(panel.end_position - panel.start_position, 1e-6f);
                panel_u = (u - panel.start_position) / span;
            }
            const float panel_v = panel.flip_y ? 1.0f - v : v;
            const glm::vec2 clamp_max = glm::clamp(panel.uv_clamp_max,
                                                   glm::vec2(0.0f), glm::vec2(1.0f));
            const glm::vec2 texture_uv = glm::min(
                (glm::vec2(panel_u, panel_v) * panel.texcoord_scale + panel.texcoord_offset) *
                    panel.uv_scale,
                clamp_max);
            const auto sample_color = [&](const glm::vec2 sample_uv) {
                return glm::vec3{
                    sample(std::get<0>(data), std::get<1>(data), std::get<2>(data), sample_uv.x, sample_uv.y, 0),
                    sample(std::get<0>(data), std::get<1>(data), std::get<2>(data), sample_uv.x, sample_uv.y, 1),
                    sample(std::get<0>(data), std::get<1>(data), std::get<2>(data), sample_uv.x, sample_uv.y, 2)};
            };
            glm::vec3 color = sample_color(texture_uv);
            if (panel.spatial_filter) {
                constexpr float kSpatialSharpenStrength = 0.18f;
                const float texel_x = 1.0f / static_cast<float>(std::max(std::get<1>(data), 1));
                const float texel_y = 1.0f / static_cast<float>(std::max(std::get<2>(data), 1));
                const auto clamp_uv = [&clamp_max](const glm::vec2 uv) {
                    return glm::clamp(uv, glm::vec2(0.0f), clamp_max);
                };
                const glm::vec3 left = sample_color(clamp_uv(texture_uv - glm::vec2(texel_x, 0.0f)));
                const glm::vec3 right = sample_color(clamp_uv(texture_uv + glm::vec2(texel_x, 0.0f)));
                const glm::vec3 up = sample_color(clamp_uv(texture_uv - glm::vec2(0.0f, texel_y)));
                const glm::vec3 down = sample_color(clamp_uv(texture_uv + glm::vec2(0.0f, texel_y)));
                const glm::vec3 sharpened = color * (1.0f + 4.0f * kSpatialSharpenStrength) -
                                            (left + right + up + down) * kSpatialSharpenStrength;
                color = glm::clamp(sharpened,
                                   glm::min(color, glm::min(glm::min(left, right), glm::min(up, down))),
                                   glm::max(color, glm::max(glm::max(left, right), glm::max(up, down))));
            }
            return color;
        };

        for (int y = rect_y; y < rect_y + rect_height; ++y) {
            const float v = splitViewPixelCenterUv(y, rect_y, rect_height);
            for (int x = rect_x; x < rect_x + rect_width; ++x) {
                const float u = splitViewPixelCenterUv(x, rect_x, rect_width);
                const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                glm::vec3 color;
                if (params.loss_visualization) {
                    color = gtLossHeatmapColor(sample_panel_color(left_panel, *left_data, u, v),
                                               sample_panel_color(right_panel, *right_data, u, v));
                } else if (x < divider) {
                    color = sample_panel_color(left_panel, *left_data, u, v);
                } else {
                    color = sample_panel_color(right_panel, *right_data, u, v);
                }
                write(idx, color);

                const float dist_from_split = std::abs(static_cast<float>(x) + 0.5f - split_x);
                if (!params.loss_visualization && dist_from_split < kMinBarWidthPx * 0.5f) {
                    glm::vec3 divider_color = kDividerColor;
                    const float dist_from_center = std::abs(static_cast<float>(y) + 0.5f - center_y);
                    const float handle_height = std::min(kHandleHeightPx, static_cast<float>(rect_height));
                    const float handle_width = std::min(kHandleWidthPx, static_cast<float>(rect_width));
                    if (dist_from_center < handle_height * 0.5f &&
                        dist_from_split < handle_width * 0.5f) {
                        const float corner_radius =
                            std::min(kCornerRadiusPx, std::min(handle_width, handle_height) * 0.5f);
                        const glm::vec2 corner_dist =
                            glm::vec2(dist_from_split, dist_from_center) -
                            (glm::vec2(handle_width, handle_height) * 0.5f - glm::vec2(corner_radius));
                        if (corner_dist.x <= 0.0f || corner_dist.y <= 0.0f ||
                            glm::length(corner_dist) <= corner_radius) {
                            divider_color = kDividerColor * 0.8f;
                            const float local_y = static_cast<float>(y) + 0.5f - center_y;
                            for (int i = -kGripLineCount; i <= kGripLineCount; ++i) {
                                const float line_y = static_cast<float>(i) * kGripSpacingPx;
                                if (std::abs(local_y - line_y) < kGripWidthPx &&
                                    dist_from_split < kGripLengthPx * 0.5f) {
                                    divider_color = glm::vec3(0.9f);
                                    break;
                                }
                            }
                        }
                    }
                    write(idx, divider_color);
                }
            }
        }

        auto tensor = lfs::core::Tensor::from_vector(
            output,
            {static_cast<std::size_t>(3), static_cast<std::size_t>(height),
             static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(tensor.permute({1, 2, 0}).contiguous());
    }
} // namespace lfs::vis
