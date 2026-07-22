/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/logger.hpp"
#include "core/memory_pressure.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "model_renderability.hpp"
#include "point_cloud_vulkan_renderer.hpp"
#include "rendering/image_layout.hpp"
#include "rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "training/trainer.hpp"
#include "training/training_manager.hpp"
#include "viewport_appearance_correction.hpp"
#include "viewport_region_utils.hpp"
#include "viewport_request_builder.hpp"
#include "vksplat_viewport_renderer.hpp"
#include "vulkan_external_tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {

    namespace {
        constexpr bool kEnableLodTransitionWeights = true;
        constexpr double kGpuLodRenderCapacityOverhead = 1.20;
        constexpr float kInteractiveResizeRenderScale = 0.33f;
        constexpr auto kTrainingOutputResizeStableDelay = std::chrono::milliseconds(500);
        constexpr int kMaxInteractiveVksplatSettlePasses = 8;

        struct LodObjectFrame {
            glm::mat4 object_to_view{1.0f};
            float object_scale = 1.0f;
        };

        [[nodiscard]] LodObjectFrame makeLodObjectFrame(
            const lfs::rendering::FrameView& frame_view,
            const lfs::rendering::GaussianSceneState& scene) {
            glm::mat4 object_to_world(1.0f);
            if (scene.model_transforms && !scene.model_transforms->empty()) {
                object_to_world = scene.model_transforms->front();
            }

            const float sx = glm::length(glm::vec3(object_to_world[0]));
            const float sy = glm::length(glm::vec3(object_to_world[1]));
            const float sz = glm::length(glm::vec3(object_to_world[2]));
            const float object_scale = std::max({sx, sy, sz, 1.0f});

            return {.object_to_view = frame_view.getViewMatrix() * object_to_world,
                    .object_scale = object_scale};
        }

        [[nodiscard]] lfs::rendering::GaussianLodGpuTraversalState makeLodGpuTraversalState(
            const LodObjectFrame& lod_frame,
            const SparkLodController::LodParameters& params,
            const std::size_t node_count) {
            const glm::mat4 view_to_object = glm::inverse(lod_frame.object_to_view);
            glm::vec3 forward = -glm::vec3(view_to_object[2]);
            const float forward_length = glm::length(forward);
            if (forward_length > 1.0e-6f) {
                forward /= forward_length;
            } else {
                forward = {0.0f, 0.0f, -1.0f};
            }

            lfs::rendering::GaussianLodGpuTraversalState state;
            state.enabled = true;
            state.node_count = node_count;
            state.pixel_scale_limit = params.pixel_scale_limit;
            state.object_scale = params.object_scale;
            state.behind_camera_penalty = params.behind_camera_penalty;
            state.cone_foveation = params.cone_foveation;
            state.cone_inner_degrees = params.cone_inner_degrees;
            state.cone_outer_degrees = params.cone_outer_degrees;
            state.outside_view_foveation = params.outside_view_foveation;
            state.viewport_half_tan_x = params.viewport_half_tan_x;
            state.viewport_half_tan_y = params.viewport_half_tan_y;
            state.ortho_half_width = params.ortho_half_width;
            state.ortho_half_height = params.ortho_half_height;
            state.view_origin = glm::vec3(view_to_object[3]);
            state.view_forward = forward;
            state.object_to_view = lod_frame.object_to_view;
            state.viewport_foveation = params.viewport_foveation;
            state.orthographic = params.orthographic;
            return state;
        }

        [[nodiscard]] std::optional<std::shared_lock<std::shared_mutex>> acquireLiveModelRenderLock(
            const SceneManager* const scene_manager) {
            std::optional<std::shared_lock<std::shared_mutex>> lock;
            if (const auto* tm = scene_manager ? scene_manager->getTrainerManager() : nullptr) {
                if (const auto* trainer = tm->getTrainer()) {
                    lock.emplace(trainer->getRenderMutex());
                }
            }
            return lock;
        }

        [[nodiscard]] bool isRetryableSharedScratchUnavailable(const std::string_view error) {
            return error.find("shared scratch") != std::string_view::npos &&
                   (error.find("busy") != std::string_view::npos ||
                    error.find("capacity insufficient") != std::string_view::npos);
        }

        [[nodiscard]] bool isRetryableVksplatOutputResizeWait(const std::string_view error) {
            return error.find("VkSplat output resize wait failed") != std::string_view::npos &&
                   error.find("VK_ERROR_DEVICE_LOST") == std::string_view::npos &&
                   (error.find("VK_TIMEOUT") != std::string_view::npos ||
                    error.find("vkWaitForFences") != std::string_view::npos);
        }

        [[nodiscard]] DirtyMask vksplatOutputResizeRetryDirty(const DirtyMask frame_dirty) {
            return frame_dirty != 0 ? frame_dirty : DirtyFlag::VIEWPORT | DirtyFlag::CAMERA;
        }

        [[nodiscard]] DirtyMask vksplatSharedScratchRetryDirty(const DirtyMask frame_dirty) {
            return frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS;
        }

        [[nodiscard]] std::pair<float, float> robustDepthDisplayRange(const float* src, std::size_t count) {
            std::vector<float> valid;
            valid.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const float d = src[i];
                if (std::isfinite(d) && d > 0.0f && d < 1.0e9f) {
                    valid.push_back(d);
                }
            }
            if (valid.size() < 2) {
                return {0.0f, 0.0f};
            }

            constexpr float kLoQuantile = 0.02f;
            constexpr float kHiQuantile = 0.98f;
            const auto quantile = [&](float q) {
                const auto n = static_cast<std::size_t>(q * static_cast<float>(valid.size() - 1));
                std::nth_element(valid.begin(), valid.begin() + n, valid.end());
                return valid[n];
            };
            return {quantile(kLoQuantile), quantile(kHiQuantile)};
        }

        [[nodiscard]] glm::vec3 depthPaletteForDisplay(float near_t) {
            near_t = std::clamp(near_t, 0.0f, 1.0f);
            const glm::vec3 far_0(0.050f, 0.040f, 0.150f);
            const glm::vec3 far_1(0.060f, 0.195f, 0.500f);
            const glm::vec3 mid_0(0.000f, 0.500f, 0.650f);
            const glm::vec3 mid_1(0.360f, 0.735f, 0.410f);
            const glm::vec3 near_0(0.965f, 0.820f, 0.300f);
            const glm::vec3 near_1(0.985f, 0.430f, 0.125f);

            if (near_t < 0.20f) {
                return glm::mix(far_0, far_1, glm::smoothstep(0.00f, 0.20f, near_t));
            }
            if (near_t < 0.43f) {
                return glm::mix(far_1, mid_0, glm::smoothstep(0.20f, 0.43f, near_t));
            }
            if (near_t < 0.67f) {
                return glm::mix(mid_0, mid_1, glm::smoothstep(0.43f, 0.67f, near_t));
            }
            if (near_t < 0.86f) {
                return glm::mix(mid_1, near_0, glm::smoothstep(0.67f, 0.86f, near_t));
            }
            return glm::mix(near_0, near_1, glm::smoothstep(0.86f, 1.00f, near_t));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeDepthDisplayTensor(
            const lfs::core::Tensor& depth,
            const RenderSettings& settings) {
            if (!depth.is_valid() || depth.ndim() != 2) {
                return {};
            }

            auto depth_cpu = depth.cpu().contiguous();
            const int height = static_cast<int>(depth_cpu.size(0));
            const int width = static_cast<int>(depth_cpu.size(1));
            if (width <= 0 || height <= 0) {
                return {};
            }

            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            std::vector<float> output(3 * pixel_count, 0.0f);
            const float* const src = depth_cpu.ptr<float>();
            if (!src) {
                return {};
            }

            const auto [range_lo, range_hi] = robustDepthDisplayRange(src, pixel_count);
            const float range_span = range_hi - range_lo;

            const bool grayscale =
                settings.depth_visualization_mode == lfs::rendering::DepthVisualizationMode::Grayscale;
            for (std::size_t idx = 0; idx < pixel_count; ++idx) {
                const float d = src[idx];
                glm::vec3 color = settings.background_color;
                if (std::isfinite(d) && d > 0.0f && d < 1.0e9f && range_span > 1.0e-6f) {
                    const float depth_t = std::clamp((d - range_lo) / range_span, 0.0f, 1.0f);
                    const float near_t = 1.0f - depth_t;
                    color = grayscale ? glm::vec3(near_t) : depthPaletteForDisplay(near_t);
                }
                output[idx] = color.r;
                output[pixel_count + idx] = color.g;
                output[2 * pixel_count + idx] = color.b;
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeNormalDisplayTensor(
            const lfs::core::Tensor& normal) {
            if (!normal.is_valid() || normal.ndim() != 3) {
                return {};
            }
            const auto layout = lfs::rendering::detectImageLayout(normal);
            if (layout == lfs::rendering::ImageLayout::Unknown ||
                lfs::rendering::imageChannels(normal, layout) < 3) {
                return {};
            }

            auto normal_cpu = normal.cpu().contiguous();
            if (layout == lfs::rendering::ImageLayout::HWC) {
                normal_cpu = normal_cpu.permute({2, 0, 1}).contiguous();
            }

            const int height = static_cast<int>(normal_cpu.size(1));
            const int width = static_cast<int>(normal_cpu.size(2));
            if (width <= 0 || height <= 0) {
                return {};
            }

            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            std::vector<float> output(3 * pixel_count, 0.5f);
            const float* const src = normal_cpu.ptr<float>();
            if (!src) {
                return {};
            }

            for (std::size_t idx = 0; idx < pixel_count; ++idx) {
                glm::vec3 n(src[idx], src[pixel_count + idx], src[2 * pixel_count + idx]);
                const float len = glm::length(n);
                if (std::isfinite(len) && len > 1.0e-6f) {
                    n /= len;
                    const glm::vec3 color = glm::clamp(n * 0.5f + glm::vec3(0.5f),
                                                       glm::vec3(0.0f),
                                                       glm::vec3(1.0f));
                    output[idx] = color.r;
                    output[pixel_count + idx] = color.g;
                    output[2 * pixel_count + idx] = color.b;
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeNormalDisplayFromDepthTensor(
            const lfs::core::Tensor& depth,
            const lfs::rendering::CameraIntrinsics& intrinsics) {
            if (!depth.is_valid() || depth.ndim() != 2 ||
                intrinsics.focal_x <= 0.0f || intrinsics.focal_y <= 0.0f) {
                return {};
            }

            auto depth_cpu = depth.cpu().contiguous();
            const int height = static_cast<int>(depth_cpu.size(0));
            const int width = static_cast<int>(depth_cpu.size(1));
            if (width <= 2 || height <= 2) {
                return {};
            }

            constexpr float kMinExpectedDepth = 1.0e-6f;
            constexpr float kMaxRelativeDepthJump = 0.05f;
            constexpr float kMinCrossNormSq = 1.0e-24f;
            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            std::vector<float> output(3 * pixel_count, 0.5f);
            const float* const src = depth_cpu.ptr<float>();
            if (!src) {
                return {};
            }

            const auto valid_depth = [](const float d) {
                return std::isfinite(d) && d >= kMinExpectedDepth && d < 1.0e9f;
            };
            const auto ray = [&](const int x, const int y) {
                return glm::vec3(
                    (static_cast<float>(x) + 0.5f - intrinsics.center_x) / intrinsics.focal_x,
                    (static_cast<float>(y) + 0.5f - intrinsics.center_y) / intrinsics.focal_y,
                    1.0f);
            };

            for (int y = 1; y < height - 1; ++y) {
                for (int x = 1; x < width - 1; ++x) {
                    const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                    const float center = src[idx];
                    if (!valid_depth(center)) {
                        continue;
                    }

                    const float dxp = src[idx + 1];
                    const float dxm = src[idx - 1];
                    const float dyp = src[idx + width];
                    const float dym = src[idx - width];
                    const float jump_limit = kMaxRelativeDepthJump * center;
                    if (!valid_depth(dxp) || !valid_depth(dxm) ||
                        !valid_depth(dyp) || !valid_depth(dym) ||
                        std::abs(dxp - center) > jump_limit ||
                        std::abs(dxm - center) > jump_limit ||
                        std::abs(dyp - center) > jump_limit ||
                        std::abs(dym - center) > jump_limit) {
                        continue;
                    }

                    const glm::vec3 tx = dxp * ray(x + 1, y) - dxm * ray(x - 1, y);
                    const glm::vec3 ty = dyp * ray(x, y + 1) - dym * ray(x, y - 1);
                    glm::vec3 n = glm::cross(tx, ty);
                    const float norm_sq = glm::dot(n, n);
                    if (!std::isfinite(norm_sq) || norm_sq < kMinCrossNormSq) {
                        continue;
                    }

                    if (glm::dot(n, ray(x, y)) > 0.0f) {
                        n = -n;
                    }
                    n = glm::normalize(n);
                    const glm::vec3 color = glm::clamp(n * 0.5f + glm::vec3(0.5f),
                                                       glm::vec3(0.0f),
                                                       glm::vec3(1.0f));
                    output[idx] = color.r;
                    output[pixel_count + idx] = color.g;
                    output[2 * pixel_count + idx] = color.b;
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {std::size_t{3}, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] glm::ivec2 gtComparisonPreviewSize(
            const lfs::core::Camera& camera,
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
            const double scale = std::min(
                1.0,
                std::min(static_cast<double>(bound_width) / static_cast<double>(base_width),
                         static_cast<double>(bound_height) / static_cast<double>(base_height)));
            return {
                std::max(static_cast<int>(std::lround(static_cast<double>(base_width) * scale)), 1),
                std::max(static_cast<int>(std::lround(static_cast<double>(base_height) * scale)), 1)};
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> resizeChwDisplayTensor(
            const std::shared_ptr<lfs::core::Tensor>& image,
            const glm::ivec2 target_size) {
            if (!image || !image->is_valid() || image->ndim() != 3 ||
                target_size.x <= 0 || target_size.y <= 0) {
                return {};
            }
            const auto layout = lfs::rendering::detectImageLayout(*image);
            if (layout == lfs::rendering::ImageLayout::Unknown) {
                return {};
            }

            lfs::core::Tensor src = *image;
            if (src.dtype() == lfs::core::DataType::UInt8) {
                src = src.to(lfs::core::DataType::Float32) / 255.0f;
            } else if (src.dtype() != lfs::core::DataType::Float32) {
                src = src.to(lfs::core::DataType::Float32);
            }
            if (layout == lfs::rendering::ImageLayout::HWC) {
                src = src.permute({2, 0, 1}).contiguous();
            }
            src = src.cpu().contiguous();

            const int src_channels = static_cast<int>(src.size(0));
            const int src_height = static_cast<int>(src.size(1));
            const int src_width = static_cast<int>(src.size(2));
            if (src_channels <= 0 || src_width <= 0 || src_height <= 0) {
                return {};
            }
            if (src_width == target_size.x && src_height == target_size.y &&
                src_channels >= 3) {
                return std::make_shared<lfs::core::Tensor>(std::move(src));
            }

            const int dst_width = target_size.x;
            const int dst_height = target_size.y;
            const std::size_t dst_pixel_count =
                static_cast<std::size_t>(dst_width) * static_cast<std::size_t>(dst_height);
            std::vector<float> output(3 * dst_pixel_count, 0.0f);
            const float* const src_data = src.ptr<float>();
            if (!src_data) {
                return {};
            }

            const auto sample = [&](const int channel, const int x, const int y) {
                const int c = std::clamp(channel, 0, src_channels - 1);
                return src_data[(static_cast<std::size_t>(c) * src_height + y) * src_width + x];
            };

            const float scale_x = static_cast<float>(src_width) / static_cast<float>(dst_width);
            const float scale_y = static_cast<float>(src_height) / static_cast<float>(dst_height);
            for (int y = 0; y < dst_height; ++y) {
                float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                int y0 = static_cast<int>(std::floor(src_y));
                float wy = src_y - static_cast<float>(y0);
                if (y0 < 0) {
                    y0 = 0;
                    wy = 0.0f;
                }
                int y1 = y0 + 1;
                if (y1 >= src_height) {
                    y1 = y0 = src_height - 1;
                    wy = 0.0f;
                }

                for (int x = 0; x < dst_width; ++x) {
                    float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    int x0 = static_cast<int>(std::floor(src_x));
                    float wx = src_x - static_cast<float>(x0);
                    if (x0 < 0) {
                        x0 = 0;
                        wx = 0.0f;
                    }
                    int x1 = x0 + 1;
                    if (x1 >= src_width) {
                        x1 = x0 = src_width - 1;
                        wx = 0.0f;
                    }

                    const std::size_t dst_idx = static_cast<std::size_t>(y) * dst_width + x;
                    for (int c = 0; c < 3; ++c) {
                        const float top =
                            glm::mix(sample(c, x0, y0), sample(c, x1, y0), wx);
                        const float bottom =
                            glm::mix(sample(c, x0, y1), sample(c, x1, y1), wx);
                        output[static_cast<std::size_t>(c) * dst_pixel_count + dst_idx] =
                            glm::mix(top, bottom, wy);
                    }
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {std::size_t{3},
                 static_cast<std::size_t>(dst_height),
                 static_cast<std::size_t>(dst_width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeGTComparePlaceholderTensor(
            glm::ivec2 size,
            const glm::vec3 tint) {
            size.x = std::max(size.x, 1);
            size.y = std::max(size.y, 1);
            const std::size_t pixel_count =
                static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y);
            std::vector<float> output(3 * pixel_count, 0.0f);
            for (int y = 0; y < size.y; ++y) {
                for (int x = 0; x < size.x; ++x) {
                    const bool stripe = ((x / 18) + (y / 18)) % 2 == 0;
                    const glm::vec3 color = glm::clamp(
                        tint + glm::vec3(stripe ? 0.045f : 0.0f),
                        glm::vec3(0.0f),
                        glm::vec3(1.0f));
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

        [[nodiscard]] std::optional<std::pair<size_t, size_t>>
        plyComparisonPairForOffset(const size_t node_count, const size_t offset) {
            if (node_count < 2) {
                return std::nullopt;
            }

            size_t remaining = offset % ((node_count * (node_count - 1)) / 2);
            for (size_t left = 0; left + 1 < node_count; ++left) {
                const size_t row_count = node_count - left - 1;
                if (remaining < row_count) {
                    return std::pair<size_t, size_t>{left, left + 1 + remaining};
                }
                remaining -= row_count;
            }

            return std::nullopt;
        }

        [[nodiscard]] std::vector<ViewportInteractionPanel> buildVulkanInteractionPanels(
            const Viewport& primary_viewport,
            const SplitViewService& split_view_service,
            const RenderSettings& settings,
            const glm::vec2& screen_viewport_pos,
            const glm::vec2& screen_viewport_size) {
            std::vector<ViewportInteractionPanel> panels;
            if (screen_viewport_size.x <= 0.0f || screen_viewport_size.y <= 0.0f) {
                return panels;
            }

            const int full_screen_width = std::max(static_cast<int>(std::lround(screen_viewport_size.x)), 1);
            const int full_screen_height = std::max(static_cast<int>(std::lround(screen_viewport_size.y)), 1);
            const auto make_panel = [&](const SplitViewPanelId panel_id,
                                        const Viewport* const viewport,
                                        const float offset_x,
                                        const float width) {
                panels.push_back({
                    .panel = panel_id,
                    .viewport_data =
                        {.rotation = viewport->getRotationMatrix(),
                         .translation = viewport->getTranslation(),
                         .size = {
                             std::max(static_cast<int>(std::lround(width)), 1),
                             full_screen_height,
                         },
                         .focal_length_mm = settings.focal_length_mm,
                         .orthographic = settings.orthographic,
                         .ortho_scale = settings.ortho_scale},
                    .viewport_pos = {screen_viewport_pos.x + offset_x, screen_viewport_pos.y},
                    .viewport_size = {width, screen_viewport_size.y},
                });
            };

            const auto layouts = split_view_service.panelLayouts(settings, full_screen_width);
            if (!layouts || full_screen_width <= 1) {
                make_panel(SplitViewPanelId::Left, &primary_viewport, 0.0f, screen_viewport_size.x);
                return panels;
            }

            panels.reserve(layouts->size());
            make_panel(SplitViewPanelId::Left,
                       &primary_viewport,
                       static_cast<float>((*layouts)[0].x),
                       static_cast<float>((*layouts)[0].width));
            make_panel(SplitViewPanelId::Right,
                       &split_view_service.secondaryViewport(),
                       static_cast<float>((*layouts)[1].x),
                       static_cast<float>((*layouts)[1].width));
            return panels;
        }

        // CPU split-view composite kept ONLY for capture/screenshot — runs lazily on
        // demand when captureViewportImage() is called, never per-frame. Mirrors the
        // pixel-for-pixel result of the Vulkan split_view.frag shader so screenshots
        // match what the user sees on screen.
        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> composeSplitViewCpu(
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
                lfs::core::Tensor t = *image;
                if (t.dtype() == lfs::core::DataType::UInt8) {
                    t = t.to(lfs::core::DataType::Float32) / 255.0f;
                } else if (t.dtype() != lfs::core::DataType::Float32) {
                    t = t.to(lfs::core::DataType::Float32);
                }
                if (layout == lfs::rendering::ImageLayout::HWC) {
                    t = t.permute({2, 0, 1}).contiguous();
                }
                t = t.cpu().contiguous();
                const int w = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(1)
                                                   : image->size(2));
                const int h = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(0)
                                                   : image->size(1));
                const int c = static_cast<int>(layout == lfs::rendering::ImageLayout::HWC
                                                   ? image->size(2)
                                                   : image->size(0));
                return std::make_tuple(std::move(t), w, h, c);
            };
            auto left_data = load_panel(params.left.image);
            auto right_data = load_panel(params.right.image);
            if (!left_data || !right_data) {
                return {};
            }

            const int width = output_size.x;
            const int height = output_size.y;
            if (width <= 0 || height <= 0) {
                return {};
            }
            const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
            std::vector<float> output(3 * pixel_count, 0.0f);
            for (std::size_t i = 0; i < pixel_count; ++i) {
                output[i] = params.background.r;
                output[pixel_count + i] = params.background.g;
                output[2 * pixel_count + i] = params.background.b;
            }

            const auto sample = [](const lfs::core::Tensor& tensor, int w, int h,
                                   float u, float v, int channel) {
                const int x = std::clamp(static_cast<int>(std::lround(std::clamp(u, 0.0f, 1.0f) *
                                                                      static_cast<float>(w - 1))),
                                         0, w - 1);
                const int y = std::clamp(static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) *
                                                                      static_cast<float>(h - 1))),
                                         0, h - 1);
                const float* data = tensor.ptr<float>();
                return data[(static_cast<std::size_t>(channel) * h + y) * w + x];
            };

            const int rect_x = params.content_rect.x;
            const int rect_y = params.content_rect.y;
            const int rect_w = std::max(params.content_rect.z, 1);
            const int rect_h = std::max(params.content_rect.w, 1);
            const int divider = rect_x + splitViewDividerPixel(rect_w, params.split_position);
            const float split_x = static_cast<float>(rect_x) +
                                  std::clamp(params.split_position, 0.0f, 1.0f) *
                                      static_cast<float>(rect_w);
            const float center_y = static_cast<float>(rect_y) + static_cast<float>(rect_h) * 0.5f;

            constexpr glm::vec3 kDividerColor(0.29f, 0.33f, 0.42f);
            constexpr float kMinBarWidthPx = 4.0f;
            constexpr float kHandleHeightPx = 80.0f;
            constexpr float kHandleWidthPx = 24.0f;
            constexpr float kCornerRadiusPx = 6.0f;
            constexpr float kGripSpacingPx = 10.0f;
            constexpr float kGripWidthPx = 2.0f;
            constexpr float kGripLengthPx = 12.0f;
            constexpr int kGripLineCount = 2;

            const auto write = [&](std::size_t idx, const glm::vec3& c) {
                output[idx] = c.r;
                output[pixel_count + idx] = c.g;
                output[2 * pixel_count + idx] = c.b;
            };

            for (int y = rect_y; y < rect_y + rect_h; ++y) {
                const float v = rect_h > 1
                                    ? static_cast<float>(y - rect_y) / static_cast<float>(rect_h - 1)
                                    : 0.0f;
                for (int x = rect_x; x < rect_x + rect_w; ++x) {
                    const float u = rect_w > 1
                                        ? static_cast<float>(x - rect_x) / static_cast<float>(rect_w - 1)
                                        : 0.0f;
                    const bool use_left = x < divider;
                    const auto& panel = use_left ? params.left : params.right;
                    const auto& data = use_left ? *left_data : *right_data;
                    float panel_u = u;
                    if (panel.normalize_x_to_panel) {
                        const float span = std::max(panel.end_position - panel.start_position, 1e-6f);
                        panel_u = (u - panel.start_position) / span;
                    }
                    const float panel_v = panel.flip_y ? 1.0f - v : v;
                    const std::size_t idx = static_cast<std::size_t>(y) * width + x;
                    write(idx,
                          {sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 0),
                           sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 1),
                           sample(std::get<0>(data), std::get<1>(data), std::get<2>(data),
                                  panel_u, panel_v, 2)});

                    const float dist_from_split = std::abs(static_cast<float>(x) + 0.5f - split_x);
                    if (dist_from_split < kMinBarWidthPx * 0.5f) {
                        glm::vec3 color = kDividerColor;
                        const float dist_from_center =
                            std::abs(static_cast<float>(y) + 0.5f - center_y);
                        const float handle_h = std::min(kHandleHeightPx, static_cast<float>(rect_h));
                        const float handle_w = std::min(kHandleWidthPx, static_cast<float>(rect_w));
                        if (dist_from_center < handle_h * 0.5f &&
                            dist_from_split < handle_w * 0.5f) {
                            const float corner_radius =
                                std::min(kCornerRadiusPx, std::min(handle_w, handle_h) * 0.5f);
                            const glm::vec2 corner_dist =
                                glm::vec2(dist_from_split, dist_from_center) -
                                (glm::vec2(handle_w, handle_h) * 0.5f - glm::vec2(corner_radius));
                            if (corner_dist.x <= 0.0f || corner_dist.y <= 0.0f ||
                                glm::length(corner_dist) <= corner_radius) {
                                color = kDividerColor * 0.8f;
                                const float local_y = static_cast<float>(y) + 0.5f - center_y;
                                for (int i = -kGripLineCount; i <= kGripLineCount; ++i) {
                                    const float line_y = static_cast<float>(i) * kGripSpacingPx;
                                    if (std::abs(local_y - line_y) < kGripWidthPx &&
                                        dist_from_split < kGripLengthPx * 0.5f) {
                                        color = glm::vec3(0.9f);
                                        break;
                                    }
                                }
                            }
                        }
                        write(idx, color);
                    }
                }
            }

            auto tensor = lfs::core::Tensor::from_vector(
                output,
                {static_cast<std::size_t>(3),
                 static_cast<std::size_t>(height),
                 static_cast<std::size_t>(width)},
                lfs::core::Device::CPU);
            return std::make_shared<lfs::core::Tensor>(std::move(tensor));
        }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeViewportCaptureImageHwc(
            lfs::core::Tensor image) {
            if (!image.is_valid() || image.ndim() != 3) {
                return std::make_shared<lfs::core::Tensor>(std::move(image));
            }

            const auto layout = lfs::rendering::detectImageLayout(image);
            if (layout == lfs::rendering::ImageLayout::CHW) {
                image = image.permute({1, 2, 0});
            }
            return std::make_shared<lfs::core::Tensor>(image.cpu().contiguous());
        }

        struct SplitCompositeContentRect {
            int x = 0;
            int y = 0;
            int width = 0;
            int height = 0;
        };

        [[nodiscard]] SplitCompositeContentRect resolveSplitCompositeContentRect(
            const glm::ivec2 output_size,
            const bool letterbox,
            const glm::ivec2 content_size) {
            SplitCompositeContentRect rect{
                .x = 0,
                .y = 0,
                .width = output_size.x,
                .height = output_size.y};
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
                    static_cast<int>(std::lround(static_cast<float>(output_size.x) / content_aspect)),
                    1);
                rect.x = 0;
                rect.y = std::max((output_size.y - rect.height) / 2, 0);
            } else {
                rect.height = output_size.y;
                rect.width = std::max(
                    static_cast<int>(std::lround(static_cast<float>(output_size.y) * content_aspect)),
                    1);
                rect.x = std::max((output_size.x - rect.width) / 2, 0);
                rect.y = 0;
            }
            rect.width = std::clamp(rect.width, 1, output_size.x);
            rect.height = std::clamp(rect.height, 1, output_size.y);
            return rect;
        }

        [[nodiscard]] lfs::rendering::FrameMetadata makeSplitMetadata(
            const lfs::rendering::FrameMetadata& left,
            const lfs::rendering::FrameMetadata& right,
            const float split_position) {
            lfs::rendering::FrameMetadata metadata{
                .depth_panels =
                    {lfs::rendering::FramePanelMetadata{
                         .depth = left.depth_panel_count > 0 ? left.depth_panels[0].depth : nullptr,
                         .start_position = 0.0f,
                         .end_position = split_position,
                     },
                     lfs::rendering::FramePanelMetadata{
                         .depth = right.depth_panel_count > 0 ? right.depth_panels[0].depth : nullptr,
                         .start_position = split_position,
                         .end_position = 1.0f,
                     }},
                .depth_panel_count = 2,
                .valid = true,
                .far_plane = left.valid ? left.far_plane : right.far_plane,
                .orthographic = left.valid ? left.orthographic : right.orthographic};
            return metadata;
        }

        // Per-frame mesh-pass payload (without depth blit). The Vulkan splat path
        // publishes this every frame so the viewport pass sees the current camera;
        // otherwise the mesh stays anchored to whichever frame last set it.
        [[nodiscard]] RenderingManager::VulkanMeshFrame populateMeshFrame(
            const FrameContext& frame_ctx,
            const RenderSettings& settings,
            const VulkanSplitViewParams& split_view_params) {
            RenderingManager::VulkanMeshFrame frame;
            const auto vp_data = frame_ctx.makeViewportData();
            frame.view_projection = vp_data.getProjectionMatrix() * vp_data.getViewMatrix();
            frame.camera_position = vp_data.translation;
            frame.items.reserve(frame_ctx.scene_state.meshes.size());

            const bool any_selected_mesh = std::any_of(
                frame_ctx.scene_state.meshes.begin(),
                frame_ctx.scene_state.meshes.end(),
                [](const auto& mesh) { return mesh.is_selected; });
            const bool any_selected_node = std::any_of(
                frame_ctx.scene_state.selected_node_mask.begin(),
                frame_ctx.scene_state.selected_node_mask.end(),
                [](const bool selected) { return selected; });
            const bool dim_non_emphasized =
                settings.desaturate_unselected && (any_selected_mesh || any_selected_node);

            const glm::vec3 headlight_dir = glm::length(vp_data.translation) > 1e-6f
                                                ? glm::normalize(vp_data.translation)
                                                : settings.mesh_light_dir;

            for (const auto& mesh : frame_ctx.scene_state.meshes) {
                if (!mesh.mesh) {
                    continue;
                }
                lfs::vis::VulkanMeshDrawItem item{};
                item.mesh = mesh.mesh;
                item.model = mesh.transform;
                item.light_dir = headlight_dir;
                item.light_intensity = settings.mesh_light_intensity;
                item.ambient = settings.mesh_ambient;
                item.backface_culling = settings.mesh_backface_culling;
                item.is_emphasized = mesh.is_selected;
                item.dim_non_emphasized = dim_non_emphasized;
                item.flash_intensity = frame_ctx.selection_flash_intensity;
                item.wireframe_overlay = settings.mesh_wireframe;
                item.wireframe_color = settings.mesh_wireframe_color;
                item.wireframe_width = settings.mesh_wireframe_width;
                item.shadow_enabled = settings.mesh_shadow_enabled;
                item.shadow_map_resolution = settings.mesh_shadow_resolution;
                frame.items.push_back(item);
            }

            const auto frame_view = frame_ctx.makeFrameView();
            frame.environment.enabled = environmentBackgroundEnabled(settings);
            frame.environment.map_path = settings.environment_map_path;
            frame.environment.camera_to_world = vp_data.rotation;
            frame.environment.viewport_size =
                glm::vec2(static_cast<float>(frame_view.size.x), static_cast<float>(frame_view.size.y));
            if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
                const auto& intr = *frame_view.intrinsics_override;
                frame.environment.intrinsics =
                    glm::vec4(intr.focal_x, intr.focal_y, intr.center_x, intr.center_y);
            } else {
                const auto [fx, fy] =
                    lfs::rendering::computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
                frame.environment.intrinsics =
                    glm::vec4(fx, fy, frame_view.size.x * 0.5f, frame_view.size.y * 0.5f);
            }
            frame.environment.exposure = settings.environment_exposure;
            frame.environment.rotation_radians = glm::radians(settings.environment_rotation_degrees);
            frame.environment.equirectangular_view = settings.equirectangular;

            frame.split_view = split_view_params;

            return frame;
        }
    } // namespace

    std::expected<lfs::core::Tensor, std::string> RenderingManager::buildVksplatSelectionMask(
        SceneManager& scene_manager,
        const lfs::rendering::FrameView& frame_view,
        const bool equirectangular,
        const VksplatSelectionMaskShape shape,
        const std::vector<glm::vec4>& primitives,
        const std::vector<glm::vec2>& polygon_vertices,
        std::uint32_t* const picked_ring_id_out) {
        LOG_TIMER("RenderingManager::buildVksplatSelectionMask");
        const auto settings = getSettings();
        if (!lfs::rendering::isVkSplatBackend(settings.raster_backend)) {
            return std::unexpected("VkSplat selection query is available only when a VkSplat backend is active");
        }
        if (!last_vulkan_context_) {
            return std::unexpected("VkSplat selection query requires an active Vulkan context");
        }
        if (!last_vulkan_context_->externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat selection query requires CUDA/Vulkan external-memory interop");
        }
        // Point-cloud mode renders with a separate graphics pipeline, but selection
        // still needs the same projected-center mask. Let it use the GPU query
        // instead of falling back to SelectionService's CPU screen-position pass.
        const bool polygon_mode = (shape == VksplatSelectionMaskShape::Polygon);
        if (polygon_mode) {
            if (polygon_vertices.size() < 3) {
                return std::unexpected("VkSplat polygon selection requires at least 3 vertices");
            }
        } else if (primitives.empty()) {
            return std::unexpected("VkSplat selection query requires at least one primitive");
        }

        auto render_lock = acquireLiveModelRenderLock(&scene_manager);
        auto scene_state = scene_manager.buildRenderState();
        const lfs::core::SplatData* const model = scene_state.combined_model;
        if (!hasRenderableGaussians(model)) {
            return std::unexpected("VkSplat selection query found no renderable Gaussian model");
        }

        if (!vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
        }

        const auto map_shape = [](const VksplatSelectionMaskShape value) {
            switch (value) {
            case VksplatSelectionMaskShape::Brush:
                return VksplatViewportRenderer::SelectionMaskShape::Brush;
            case VksplatSelectionMaskShape::Rectangle:
                return VksplatViewportRenderer::SelectionMaskShape::Rectangle;
            case VksplatSelectionMaskShape::Polygon:
                return VksplatViewportRenderer::SelectionMaskShape::Polygon;
            case VksplatSelectionMaskShape::Ring:
                return VksplatViewportRenderer::SelectionMaskShape::Ring;
            }
            return VksplatViewportRenderer::SelectionMaskShape::Brush;
        };

        const bool is_training = scene_manager.hasDataset() &&
                                 scene_manager.getTrainerManager() &&
                                 scene_manager.getTrainerManager()->isRunning();

        VksplatViewportRenderer::SelectionMaskRequest request{
            .frame_view = frame_view,
            .scene =
                {.model_transforms = &scene_state.model_transforms,
                 .transform_indices = scene_state.transform_indices,
                 .node_visibility_mask = scene_state.node_visibility_mask},
            .shape = map_shape(shape),
            .primitives = primitives,
            .polygon_vertices = polygon_vertices,
            .gut = lfs::rendering::isGutBackend(settings.raster_backend),
            .equirectangular = equirectangular,
            .mip_filter = settings.mip_filter,
            .ring_width = settings.ring_width,
            .synchronize_input_upload = is_training,
            .picked_ring_id_out = picked_ring_id_out,
        };

        const bool force_input_upload =
            (dirty_mask_.load(std::memory_order_relaxed) & DirtyFlag::SPLATS) != 0;
        return vksplat_viewport_renderer_->buildSelectionMask(
            *last_vulkan_context_, *model, request, force_input_upload);
    }

    RenderingManager::VulkanFrameResult RenderingManager::renderVulkanFrame(const RenderContext& context) {
        LOG_TIMER("renderVulkanFrame");
        const RenderSettings frame_settings = [this] {
            std::lock_guard lock(settings_mutex_);
            return settings_;
        }();
        SceneManager* const scene_manager = context.scene_manager;
        auto* const trainer_manager = scene_manager ? scene_manager->getTrainerManager() : nullptr;
        const bool is_training = scene_manager && scene_manager->hasDataset() &&
                                 trainer_manager && trainer_manager->isRunning();
        if (!is_training && vksplat_viewport_renderer_) {
            // Clear a callback capturing the previous trainer before any cached or
            // minimized-frame early return can leave it reachable by an auxiliary submit.
            vksplat_viewport_renderer_->setLiveSubmitCallback({});
        }
        if (context.vulkan_context) {
            last_vulkan_context_ = context.vulkan_context;
        }

        const auto framebuffer_region =
            resolveFramebufferViewportRegion(context.viewport, context.logical_screen_size, context.viewport_region);
        glm::ivec2 current_size = context.viewport.frameBufferSize;
        if (context.viewport_region) {
            current_size = framebuffer_region.size;
        }
        if (current_size.x <= 0 || current_size.y <= 0) {
            return {.image = vulkan_viewport_image_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        }
        initialized_ = true;

        std::optional<lfs::core::CUDAStreamGuard> frame_stream_guard;
        const auto cached_frame_result = [this]() -> VulkanFrameResult {
            if (vulkan_external_viewport_image_ != VK_NULL_HANDLE) {
                return {.image = {},
                        .external_image = vulkan_external_viewport_image_,
                        .external_image_view = vulkan_external_viewport_image_view_,
                        .external_image_layout = vulkan_external_viewport_image_layout_,
                        .external_image_generation = vulkan_external_viewport_image_generation_,
                        .image_generation = vulkan_viewport_image_generation_,
                        .size = vulkan_viewport_image_size_,
                        .flip_y = vulkan_viewport_image_flip_y_};
            }
            if (!vulkan_viewport_image_) {
                return {.image = {},
                        .image_generation = split_view_image_generation_,
                        .size = vulkan_viewport_image_size_,
                        .flip_y = vulkan_viewport_image_flip_y_};
            }
            return {.image = vulkan_viewport_image_,
                    .image_generation = vulkan_viewport_image_generation_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
        };
        const auto update_cached_split_position = [this, &frame_settings](const bool require_position_change) -> bool {
            if (!split_view_service_.isActive(frame_settings)) {
                return false;
            }

            const float split_position = std::clamp(frame_settings.split_position, 0.0f, 1.0f);
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            if (!vulkan_mesh_frame_.split_view.enabled) {
                return false;
            }

            auto& split = vulkan_mesh_frame_.split_view;
            const bool split_position_changed =
                split.split_position != split_position ||
                split.left.end_position != split_position ||
                split.right.start_position != split_position ||
                (vulkan_mesh_frame_.panels.size() == 2 &&
                 (vulkan_mesh_frame_.panels[0].end_position != split_position ||
                  vulkan_mesh_frame_.panels[1].start_position != split_position));
            if (!split_position_changed) {
                return !require_position_change;
            }

            split.split_position = split_position;
            split.left.start_position = 0.0f;
            split.left.end_position = split_position;
            split.right.start_position = split_position;
            split.right.end_position = 1.0f;

            if (vulkan_mesh_frame_.panels.size() == 2) {
                vulkan_mesh_frame_.panels[0].start_position = 0.0f;
                vulkan_mesh_frame_.panels[0].end_position = split_position;
                vulkan_mesh_frame_.panels[1].start_position = split_position;
                vulkan_mesh_frame_.panels[1].end_position = 1.0f;
            }
            viewport_artifact_service_.invalidateCapturedImage();
            return true;
        };
        const auto has_cached_split_view_output = [this]() -> bool {
            if (vulkan_viewport_image_size_.x <= 0 || vulkan_viewport_image_size_.y <= 0) {
                return false;
            }
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            return vulkan_mesh_frame_.split_view.enabled;
        };

        const auto ensure_auxiliary_rendering_engine =
            [this]() -> std::expected<lfs::rendering::RenderingEngine*, std::string> {
            if (!engine_) {
                engine_ = lfs::rendering::RenderingEngine::create();
            }
            if (!engine_->isInitialized()) {
                if (auto init_result = engine_->initialize(); !init_result) {
                    return std::unexpected(init_result.error());
                }
            }
            return engine_.get();
        };

        glm::vec2 screen_viewport_pos(0.0f, 0.0f);
        glm::vec2 screen_viewport_size(
            static_cast<float>(context.viewport.windowSize.x),
            static_cast<float>(context.viewport.windowSize.y));
        if (context.viewport_region) {
            screen_viewport_pos = {context.viewport_region->x, context.viewport_region->y};
            screen_viewport_size = {context.viewport_region->width, context.viewport_region->height};
        }
        const auto interaction_panels = buildVulkanInteractionPanels(
            context.viewport,
            split_view_service_,
            frame_settings,
            screen_viewport_pos,
            screen_viewport_size);
        viewport_interaction_context_.updatePickContext(interaction_panels);

        const auto resize_result = frame_lifecycle_service_.handleViewportResize(current_size);
        if (resize_result.dirty) {
            markDirty(resize_result.dirty);
        }
        const bool resize_deferring = frame_lifecycle_service_.isResizeDeferring();
        float scale = std::clamp(frame_settings.render_scale, 0.25f, 1.0f);
        if (resize_result.render_interactive_frame) {
            scale = std::min(scale, kInteractiveResizeRenderScale);
        }
        // Under an active VRAM pressure lease, halve the viewer render resolution
        // to shrink per-frame output allocation. Restored automatically once the
        // coordinator observes sustained headroom. Does not affect training.
        if (lfs::core::MemoryPressureCoordinator::instance().pressure_active()) {
            scale = std::clamp(scale * 0.5f, 0.25f, 1.0f);
        }
        glm::ivec2 render_size(
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.x) * scale)), 1),
            std::max(static_cast<int>(std::lround(static_cast<float>(current_size.y) * scale)), 1));
        const bool vksplat_render_size_changed = render_size != last_vksplat_render_size_;

        const DirtyMask pending_dirty = dirty_mask_.load(std::memory_order_relaxed);
        const bool only_split_position_pending =
            (pending_dirty & ~DirtyFlag::SPLIT_POSITION) == 0;
        if ((pending_dirty & DirtyFlag::SPLIT_POSITION) != 0 &&
            vulkan_viewport_image_size_ == render_size &&
            has_cached_split_view_output() &&
            update_cached_split_position(!only_split_position_pending)) {
            dirty_mask_.fetch_and(~DirtyFlag::SPLIT_POSITION, std::memory_order_relaxed);
            LOG_PERF("renderVulkanFrame: split-position early cache HIT (returning cached image)");
            if (!resize_deferring) {
                releaseResizeTrainingPause();
            }
            return cached_frame_result();
        }

        if (resize_deferring && is_training) {
            requestResizeTrainingPause(trainer_manager);
        }
        const auto release_resize_pause_if_idle = [this, resize_deferring]() {
            if (!resize_deferring) {
                releaseResizeTrainingPause();
            }
        };

        auto render_lock = acquireLiveModelRenderLock(scene_manager);

        const lfs::core::SplatData* const model = scene_manager ? scene_manager->getModelForRendering() : nullptr;
        SceneRenderState scene_state;
        if (scene_manager) {
            LOG_TIMER("renderVulkanFrame.buildRenderState");
            scene_state = scene_manager->buildRenderState();
        }
        const bool has_renderable_model = hasRenderableGaussians(model);
        const bool has_visible_gaussian_model =
            has_renderable_model && scene_state.visible_splat_count > 0;
        const bool has_point_cloud =
            scene_state.point_cloud != nullptr && scene_state.point_cloud->size() > 0;
        const bool has_meshes = std::any_of(scene_state.meshes.begin(),
                                            scene_state.meshes.end(),
                                            [](const auto& mesh) { return mesh.mesh != nullptr; });
        const bool has_environment = environmentBackgroundEnabled(frame_settings);
        const bool has_render_content = has_visible_gaussian_model || has_point_cloud || has_meshes || has_environment;
        const size_t model_ptr = reinterpret_cast<size_t>(model);

        if (const auto model_change = frame_lifecycle_service_.handleModelChange(model_ptr, viewport_artifact_service_);
            model_change.changed) {
            gt_comparison_image_cache_ = {};
            clearVulkanViewportImageState();
            last_logged_vksplat_render_error_.clear();
            if (vksplat_viewport_renderer_) {
                if (is_training && lfs::rendering::isVkSplatBackend(frame_settings.raster_backend)) {
                    LOG_DEBUG("Preserving VkSplat renderer across training model change");
                } else {
                    // The trainer must drop the fence handle before reset destroys
                    // the CUDA import it points at.
                    if (trainer_manager) {
                        if (auto* trainer = trainer_manager->getTrainer()) {
                            trainer->setViewerReleaseFence(nullptr);
                        }
                    }
                    // reset() destroys render_stream_; drop it from the TLS current
                    // stream first so the rest of the frame doesn't enqueue work on a
                    // stale handle. Re-installed after the handshake re-init below.
                    frame_stream_guard.reset();
                    vksplat_viewport_renderer_->reset();
                }
            }
            viewport_artifact_service_.clearViewportOutput();
            markDirty(DirtyFlag::ALL);
        }

        const bool synchronize_vksplat_input_upload = is_training;
        if (const DirtyMask training_dirty = frame_lifecycle_service_.handleTrainingRefresh(
                is_training,
                framerate_controller_.getSettings().training_frame_refresh_time_sec);
            training_dirty) {
            markDirty(training_dirty);
        }

        const bool has_cached_gpu_only_frame = [&]() {
            if (vulkan_viewport_image_size_.x <= 0 || vulkan_viewport_image_size_.y <= 0) {
                return false;
            }
            std::lock_guard lock(vulkan_mesh_frame_mutex_);
            return vulkan_mesh_frame_.split_view.enabled ||
                   !vulkan_mesh_frame_.items.empty() ||
                   vulkan_mesh_frame_.environment.enabled;
        }();
        const bool has_cached_viewport_output =
            vulkan_viewport_image_ != nullptr ||
            vulkan_external_viewport_image_ != VK_NULL_HANDLE ||
            has_cached_gpu_only_frame;
        LOG_PERF("renderVulkanFrame.resize deferring={} render={} completed={} render_scale={:.2f} render_size={}x{} current_size={}x{}",
                 resize_deferring,
                 resize_result.render_interactive_frame,
                 resize_result.completed,
                 scale,
                 render_size.x,
                 render_size.y,
                 current_size.x,
                 current_size.y);

        if (const DirtyMask required_dirty = frame_lifecycle_service_.requiredDirtyMask(
                has_cached_viewport_output,
                has_render_content,
                frame_settings.split_view_mode);
            required_dirty) {
            dirty_mask_.fetch_or(required_dirty, std::memory_order_relaxed);
        }

        DirtyMask frame_dirty = dirty_mask_.exchange(0);
        if (lod_controller_ && lod_controller_->hasReadyResults()) {
            frame_dirty |= DirtyFlag::CAMERA;
        }
        if (lod_controller_ && lod_controller_->transitionActive()) {
            frame_dirty |= DirtyFlag::CAMERA;
        }
        const bool camera_pose_changed =
            camera_pose_dirty_.exchange(false, std::memory_order_acq_rel);
        const bool vksplat_split_layout_changed =
            (frame_dirty & (DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLIT_POSITION)) != 0;
        const bool vksplat_view_changed =
            camera_pose_changed || vksplat_render_size_changed || vksplat_split_layout_changed;
        if (vksplat_view_changed) {
            vksplat_camera_settle_passes_remaining_ = kMaxInteractiveVksplatSettlePasses;
        }
        const bool settle_vksplat_capacity =
            vksplat_camera_settle_passes_remaining_ > 0;
        constexpr DirtyMask projection_dirty =
            DirtyFlag::CAMERA | DirtyFlag::SPLATS | DirtyFlag::VIEWPORT | DirtyFlag::SPLIT_VIEW;
        if ((frame_dirty & projection_dirty) != 0) {
            ++viewport_projection_generation_;
            if (viewport_projection_generation_ == 0) {
                ++viewport_projection_generation_;
            }
        }
        LOG_PERF("renderVulkanFrame.frame_dirty=0x{:x} model={} pc={} mesh={} env={}",
                 frame_dirty, has_renderable_model, has_point_cloud, has_meshes, has_environment);
        if (!has_render_content) {
            clearVulkanViewportImageState();
            last_logged_vksplat_render_error_.clear();
            viewport_artifact_service_.clearViewportOutput();
            clearVulkanMeshFrame();
            render_lock.reset();
            release_resize_pause_if_idle();
            return {};
        }

        const DirtyMask split_deferred_dirty = frame_dirty & ~DirtyFlag::SPLIT_POSITION;
        if ((frame_dirty & DirtyFlag::SPLIT_POSITION) != 0 &&
            has_cached_viewport_output &&
            update_cached_split_position(split_deferred_dirty != 0)) {
            const DirtyMask deferred_dirty = split_deferred_dirty;
            if (deferred_dirty != 0) {
                dirty_mask_.fetch_or(deferred_dirty, std::memory_order_relaxed);
            }
            LOG_PERF("renderVulkanFrame: split-position cache HIT (returning cached image, deferred_dirty=0x{:x})",
                     deferred_dirty);
            render_lock.reset();
            release_resize_pause_if_idle();
            return cached_frame_result();
        }

        if (resize_deferring &&
            has_cached_viewport_output &&
            !resize_result.render_interactive_frame) {
            update_cached_split_position(false);
            constexpr DirtyMask resize_defer_consumed_dirty =
                DirtyFlag::CAMERA | DirtyFlag::VIEWPORT | DirtyFlag::OVERLAY;
            const DirtyMask deferred_dirty = frame_dirty & ~resize_defer_consumed_dirty;
            if (deferred_dirty != 0) {
                dirty_mask_.fetch_or(deferred_dirty, std::memory_order_relaxed);
            }
            LOG_PERF("renderVulkanFrame: resize defer (returning cached image)");
            render_lock.reset();
            return cached_frame_result();
        }

        const bool vksplat_viewport_resize =
            is_training &&
            context.vulkan_context != nullptr &&
            vksplat_viewport_renderer_ != nullptr &&
            vksplat_viewport_renderer_->nextOutputImagesNeedResize(
                render_size,
                VksplatViewportRenderer::OutputSlot::Main) &&
            lfs::rendering::isVkSplatBackend(frame_settings.raster_backend);
        if (vksplat_viewport_resize) {
            const auto* const trainer = trainer_manager ? trainer_manager->getTrainer() : nullptr;
            if (!trainer || !trainer->is_paused()) {
                requestResizeTrainingPause(trainer_manager);
                dirty_mask_.fetch_or(vksplatOutputResizeRetryDirty(frame_dirty),
                                     std::memory_order_relaxed);
                render_lock.reset();
                return cached_frame_result();
            }
            if (has_cached_viewport_output &&
                frame_lifecycle_service_.resizeRecentlyChanged(kTrainingOutputResizeStableDelay)) {
                dirty_mask_.fetch_or(vksplatOutputResizeRetryDirty(frame_dirty),
                                     std::memory_order_relaxed);
                render_lock.reset();
                return cached_frame_result();
            }
            LOG_DEBUG("Training paused for VkSplat output resize to {}x{}", render_size.x, render_size.y);
        }
        const auto release_resize_pause_on_return = [this, resize_deferring]() {
            if (!resize_deferring) {
                releaseResizeTrainingPause();
            }
        };
        struct ResizePauseReleaseOnReturn {
            const decltype(release_resize_pause_on_return)& release;
            bool active = false;
            ~ResizePauseReleaseOnReturn() {
                if (active) {
                    release();
                }
            }
        } resize_pause_release_on_return{release_resize_pause_on_return,
                                         resize_training_pause_active_ && !resize_deferring};

        if (!vksplat_viewport_resize && frame_dirty == 0 && has_cached_viewport_output) {
            LOG_PERF("renderVulkanFrame: cache HIT (returning cached image)");
            render_lock.reset();
            return cached_frame_result();
        }

        // Only a frame that will perform renderer work may join the trainer's
        // read epoch. Cached frames must not enqueue a legacy-stream wait or
        // publish a reader edge that serializes the next training step.
        if (is_training && context.vulkan_context &&
            lfs::rendering::isVkSplatBackend(frame_settings.raster_backend)) {
            if (!vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
            }
            if (const auto ok = vksplat_viewport_renderer_->ensureHandshakeReady(*context.vulkan_context); !ok) {
                LOG_WARN("VkSplat handshake pre-init skipped: {}", ok.error());
            }
        }
        // ensureHandshakeReady() may recreate render_stream_. Install the current
        // stream before any frame preparation can enqueue CUDA work.
        frame_stream_guard.reset();
        if (vksplat_viewport_renderer_ && vksplat_viewport_renderer_->renderStream()) {
            frame_stream_guard.emplace(vksplat_viewport_renderer_->renderStream());
        }
        lfs::training::Trainer* live_trainer = nullptr;
        if (is_training && trainer_manager && vksplat_viewport_renderer_ &&
            vksplat_viewport_renderer_->renderStream() &&
            vksplat_viewport_renderer_->renderCompleteFence()) {
            // A live stream without its release fence is a partial initialization;
            // render() will fail too, so never install a handshake missing the
            // trainer's reverse dependency.
            live_trainer = trainer_manager->getTrainer();
        }
        // Held shared until all CPU/GPU frame preparation and readback paths exit.
        std::optional<std::shared_lock<std::shared_mutex>> model_read_lock;
        if (live_trainer) {
            model_read_lock.emplace(live_trainer->getModelAccessMutex());
        }
        if (live_trainer) {
            live_trainer->setViewerReleaseFence(vksplat_viewport_renderer_->renderCompleteFence());
            live_trainer->beginModelRead(vksplat_viewport_renderer_->renderStream());
            lfs::training::Trainer* const trainer = live_trainer;
            vksplat_viewport_renderer_->setLiveSubmitCallback(
                [trainer](const std::uint64_t value) { trainer->publishViewerBorrow(value); });
        } else if (vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_->setLiveSubmitCallback({});
        }
        // Prompt callbacks publish after each submit. This scope-exit edge also
        // covers preparation errors after CUDA model reads but before a submit.
        struct ViewerBorrowPublisher {
            lfs::training::Trainer* trainer;
            VksplatViewportRenderer* renderer;
            ~ViewerBorrowPublisher() {
                if (trainer && renderer) {
                    try {
                        trainer->endModelRead(renderer->renderStream());
                        trainer->publishViewerBorrow(renderer->renderCompleteValue());
                    } catch (const std::exception& e) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during frame teardown: {}",
                                  e.what());
                    } catch (...) {
                        LOG_ERROR("ViewerBorrowPublisher: endModelRead/publishViewerBorrow failed "
                                  "during frame teardown with an unknown error");
                    }
                }
            }
        } viewer_borrow_publisher{live_trainer, vksplat_viewport_renderer_.get()};

        framerate_controller_.beginFrame();

        const FrameContext frame_ctx{
            .viewport = context.viewport,
            .viewport_region = context.viewport_region,
            .render_lock_held = render_lock.has_value(),
            .scene_manager = scene_manager,
            .model = model,
            .scene_state = std::move(scene_state),
            .settings = frame_settings,
            .render_size = render_size,
            .viewport_pos = {0, 0},
            .frame_dirty = frame_dirty,
            .training_active = is_training,
            .cursor_preview = viewport_overlay_service_.cursorPreview(),
            .gizmo = viewport_overlay_service_.makeFrameGizmoState(),
            .hovered_camera_id = camera_interaction_service_.hoveredCameraId(),
            .current_camera_id = camera_interaction_service_.currentCameraId(),
            .hovered_gaussian_id = viewport_overlay_service_.hoveredGaussianId(),
            .selection_flash_intensity = getSelectionFlashIntensity(),
            .view_panels = {}};

        std::shared_ptr<lfs::core::Tensor> rendered_image;
        lfs::rendering::FrameMetadata rendered_metadata{};
        std::string render_error;
        bool rendered_image_contains_ground_truth = false;
        glm::ivec2 rendered_gt_content_size{0, 0};
        std::optional<SplitViewInfo> rendered_split_info;
        VulkanSplitViewParams pending_split_view{};
        const auto release_inactive_split_outputs = [&] {
            if (vksplat_viewport_renderer_ && !split_view_service_.isActive(frame_settings)) {
                vksplat_viewport_renderer_->releaseSplitOutputResources();
            }
        };

        const auto populate_independent_split_mesh_panels =
            [&](VulkanMeshFrame& frame) {
                if (!pending_split_view.enabled ||
                    !splitViewUsesIndependentPanels(frame_settings.split_view_mode)) {
                    return;
                }
                const auto layouts = split_view_service_.panelLayouts(frame_settings, render_size.x);
                if (!layouts || render_size.y <= 0) {
                    return;
                }
                frame.panels.clear();
                const auto append_panel =
                    [&](const Viewport& viewport, const std::size_t index) {
                        const auto& layout = (*layouts)[index];
                        const glm::ivec2 panel_size{std::max(layout.width, 1), render_size.y};
                        const auto panel_view = frame_ctx.makeViewportData(viewport, panel_size);
                        frame.panels.push_back(lfs::vis::VulkanMeshViewportPanel{
                            .start_position = layout.start_position,
                            .end_position = layout.end_position,
                            .view_projection = panel_view.getProjectionMatrix() * panel_view.getViewMatrix(),
                            .camera_position = panel_view.translation});
                    };
                append_panel(context.viewport, 0);
                append_panel(split_view_service_.secondaryViewport(), 1);
            };

        struct RenderedPanel {
            std::shared_ptr<lfs::core::Tensor> image;
            lfs::rendering::FrameMetadata metadata;
            VkImageView external_image_view = VK_NULL_HANDLE;
            std::uint64_t external_image_generation = 0;
            bool flip_y = false;
        };

        const auto ensure_cuda_viewport_image =
            [](std::shared_ptr<lfs::core::Tensor> image,
               const std::string_view label) -> std::shared_ptr<lfs::core::Tensor> {
            if (!image || !image->is_valid()) {
                return {};
            }
            if (image->device() == lfs::core::Device::CUDA) {
                return image;
            }
            auto cuda_image = image->cuda();
            if (!cuda_image.is_valid() || cuda_image.device() != lfs::core::Device::CUDA) {
                LOG_WARN("{} produced a non-CUDA tensor; falling back to the external image path", label);
                return {};
            }
            return std::make_shared<lfs::core::Tensor>(std::move(cuda_image));
        };

        VkSemaphore latest_vksplat_completion_semaphore = VK_NULL_HANDLE;
        std::uint64_t latest_vksplat_completion_value = 0;
        bool vksplat_inputs_forced_this_frame = false;
        const auto note_lod_page_generation = [&](const std::uint64_t generation) {
            if (generation == 0 || generation == lod_controller_page_map_generation_) {
                return;
            }
            LOG_DEBUG("LOD page map generation advanced: renderer={} controller={}",
                      generation,
                      lod_controller_page_map_generation_);
            notifyAsyncLodResultsReady();
        };
        bool vksplat_capacity_settle_checked = false;
        const auto note_vksplat_render_progress =
            [&](const VksplatViewportRenderer::RenderResult& result) {
                last_vksplat_render_size_ = render_size;
                if (result.lod_streaming_active) {
                    requestRenderFollowUp();
                }
                if (!settle_vksplat_capacity || vksplat_capacity_settle_checked) {
                    return;
                }
                vksplat_capacity_settle_checked = true;
                if (vksplat_view_changed) {
                    requestRenderFollowUp();
                    return;
                }
                if (result.capacity_readback_settled) {
                    vksplat_camera_settle_passes_remaining_ = 0;
                    return;
                }
                if (--vksplat_camera_settle_passes_remaining_ > 0) {
                    requestRenderFollowUp();
                }
            };
        const auto render_panel_image =
            [&](const Viewport& source_viewport,
                const glm::ivec2 panel_size,
                const std::optional<SplitViewPanelId> panel_id,
                const std::optional<std::vector<bool>>& node_visibility_override,
                const lfs::core::SplatData* model_override = nullptr,
                const std::vector<glm::mat4>* model_transforms_override = nullptr,
                const std::optional<VksplatViewportRenderer::OutputSlot> vksplat_output_slot = std::nullopt,
                const lfs::rendering::ViewportRenderRequest* request_override = nullptr)
            -> std::expected<RenderedPanel, std::string> {
            const lfs::core::SplatData* const panel_model = model_override ? model_override : model;
            const bool panel_has_visible_gaussian_model =
                hasRenderableGaussians(panel_model) &&
                (model_override != nullptr || panel_model != model || has_visible_gaussian_model);
            if (panel_size.x <= 0 || panel_size.y <= 0) {
                return std::unexpected("Invalid split-view panel size");
            }

            if ((frame_settings.point_cloud_mode || !panel_has_visible_gaussian_model) && has_point_cloud && !panel_model) {
                const std::vector<glm::mat4> point_cloud_transforms = {frame_ctx.scene_state.point_cloud_transform};
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene =
                        {.model_transforms = &point_cloud_transforms,
                         .transform_indices = nullptr,
                         .node_visibility_mask = {}},
                    .filters = state.filters,
                    .overlay = state.overlay,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(frame_settings)};
                auto auxiliary_engine = ensure_auxiliary_rendering_engine();
                if (!auxiliary_engine) {
                    return std::unexpected(auxiliary_engine.error());
                }
                auto result = (*auxiliary_engine)->renderPointCloudImage(*frame_ctx.scene_state.point_cloud, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Raw point-cloud panel render returned no image"
                                                  : result.error());
                }
                const bool flip_y = !result->metadata.flip_y;
                return RenderedPanel{.image = std::move(result->image),
                                     .metadata = std::move(result->metadata),
                                     .flip_y = flip_y};
            }

            if (!panel_has_visible_gaussian_model) {
                return std::unexpected("No renderable model for split-view panel");
            }

            if (frame_settings.point_cloud_mode) {
                const auto state = buildSplitViewPointCloudPanelRenderState(frame_ctx, panel_size, &source_viewport);
                std::vector<glm::mat4> transforms_storage;
                auto scene = state.scene;
                if (model_transforms_override) {
                    scene.model_transforms = model_transforms_override;
                } else if (!scene.model_transforms) {
                    transforms_storage = {glm::mat4(1.0f)};
                    scene.model_transforms = &transforms_storage;
                }
                if (node_visibility_override) {
                    scene.node_visibility_mask = *node_visibility_override;
                }
                lfs::rendering::PointCloudRenderRequest request{
                    .frame_view = state.frame_view,
                    .render = state.render,
                    .scene = scene,
                    .filters = state.filters,
                    .overlay = state.overlay,
                    .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(frame_settings)};
                auto auxiliary_engine = ensure_auxiliary_rendering_engine();
                if (!auxiliary_engine) {
                    return std::unexpected(auxiliary_engine.error());
                }
                auto result = (*auxiliary_engine)->renderPointCloudImage(*panel_model, request);
                if (!result || !result->image) {
                    return std::unexpected(result ? "Point-cloud panel render returned no image"
                                                  : result.error());
                }
                const bool flip_y = !result->metadata.flip_y;
                return RenderedPanel{.image = std::move(result->image),
                                     .metadata = std::move(result->metadata),
                                     .flip_y = flip_y};
            }

            auto request = request_override
                               ? *request_override
                               : buildViewportRenderRequest(frame_ctx, panel_size, &source_viewport, panel_id);
            std::vector<std::uint32_t> lod_touched_chunks;
            if (lod_controller_ && lod_controller_->hasTree()) {
                lod_controller_->advanceTransition();
                const bool lod_transition_active = lod_controller_->transitionActive();
                if (lod_transition_active) {
                    notifyAsyncLodResultsReady();
                }
                const auto& selected = frame_settings.lod_enabled
                                           ? lod_controller_->selectedIndices()
                                           : lod_controller_->fullQualityIndices();
                if (!selected.empty()) {
                    request.lod_indices = selected.data();
                    if (kEnableLodTransitionWeights &&
                        frame_settings.lod_enabled &&
                        lod_transition_active) {
                        const auto& weights = lod_controller_->selectedWeights();
                        if (weights.size() == selected.size()) {
                            request.lod_weights = weights.data();
                        }
                    }
                    if (lod_controller_->pageMappingActive()) {
                        const auto& logical = frame_settings.lod_enabled
                                                  ? lod_controller_->selectedLogicalIndices()
                                                  : lod_controller_->fullQualityLogicalIndices();
                        if (logical.size() == selected.size()) {
                            request.lod_logical_indices = logical.data();
                        }
                    }
                    if (frame_settings.lod_debug_colors) {
                        const auto& levels = frame_settings.lod_enabled
                                                 ? lod_controller_->selectedLevels()
                                                 : lod_controller_->fullQualityLevels();
                        if (levels.size() == selected.size()) {
                            request.lod_levels = levels.data();
                        }
                    }
                    request.lod_count = selected.size();
                    request.lod_selection_hash = lod_controller_->selectionHash();
                    request.lod_generation = lod_controller_->statsGeneration();
                    lod_touched_chunks = lod_controller_->touchedChunks();
                    request.lod_touched_chunks = lod_touched_chunks.data();
                    request.lod_touched_chunk_count = lod_touched_chunks.size();
                    request.lod_debug_mode = frame_settings.lod_debug_colors;
                }
            }
            std::vector<glm::mat4> transforms_storage;
            if (model_transforms_override) {
                request.scene.model_transforms = model_transforms_override;
            } else if (!request.scene.model_transforms) {
                transforms_storage = {glm::mat4(1.0f)};
                request.scene.model_transforms = &transforms_storage;
            }
            if (node_visibility_override) {
                request.scene.node_visibility_mask = *node_visibility_override;
            }
            request.raster_backend =
                lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);

            const bool vksplat_panel_supported =
                vksplat_output_slot.has_value() &&
                context.vulkan_context != nullptr &&
                lfs::rendering::isVkSplatBackend(request.raster_backend);
            if (vksplat_panel_supported) {
                if (!vksplat_viewport_renderer_) {
                    vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
                }
                const bool force_input_upload =
                    (frame_dirty & DirtyFlag::SPLATS) != 0 && !vksplat_inputs_forced_this_frame;
                LOG_TIMER("vksplat.split_panel.render");
                auto result = vksplat_viewport_renderer_->render(
                    *context.vulkan_context,
                    *panel_model,
                    request,
                    force_input_upload,
                    *vksplat_output_slot,
                    synchronize_vksplat_input_upload);
                if (result) {
                    if (force_input_upload) {
                        vksplat_inputs_forced_this_frame = true;
                    }
                    note_lod_page_generation(result->lod_page_generation);
                    note_vksplat_render_progress(*result);
                    latest_vksplat_completion_semaphore = result->completion_semaphore;
                    latest_vksplat_completion_value = result->completion_value;
                    lfs::rendering::FrameMetadata metadata{};
                    metadata.valid = true;
                    metadata.flip_y = result->flip_y;
                    return RenderedPanel{.image = nullptr,
                                         .metadata = std::move(metadata),
                                         .external_image_view = result->image_view,
                                         .external_image_generation = result->generation,
                                         .flip_y = result->flip_y};
                }
                return std::unexpected(result.error());
            }

            if (!context.vulkan_context) {
                return std::unexpected("VkSplat split-view panel requires an active Vulkan context");
            }
            return std::unexpected("VkSplat split-view panel did not receive an output slot");
        };

        const auto make_split_panel =
            [](const RenderedPanel& panel,
               const float start,
               const float end,
               const bool normalize_x_to_panel) {
                return VulkanSplitViewPanel{
                    .image = panel.image,
                    .start_position = start,
                    .end_position = end,
                    .normalize_x_to_panel = normalize_x_to_panel,
                    .flip_y = panel.flip_y,
                    .external_image_view = panel.external_image_view,
                    .external_image_generation = panel.external_image_generation,
                };
            };
        const auto make_placeholder_panel =
            [](const glm::ivec2 size, const glm::vec3 tint) {
                lfs::rendering::FrameMetadata metadata{};
                metadata.valid = true;
                metadata.flip_y = false;
                return RenderedPanel{
                    .image = makeGTComparePlaceholderTensor(size, tint),
                    .metadata = std::move(metadata),
                    .flip_y = false,
                };
            };

        if (splitViewUsesGTComparison(frame_settings.split_view_mode) && scene_manager &&
            (has_visible_gaussian_model || has_point_cloud)) {
            std::shared_ptr<lfs::core::Camera> camera;
            const auto cameras = scene_manager->getScene().getAllCameras();
            if (frame_ctx.current_camera_id >= 0) {
                for (const auto& candidate : cameras) {
                    if (candidate && candidate->uid() == frame_ctx.current_camera_id) {
                        camera = candidate;
                        break;
                    }
                }
            }
            if (!camera && !cameras.empty()) {
                for (const auto& candidate : cameras) {
                    if (candidate) {
                        camera = candidate;
                        break;
                    }
                }
            }

            if (camera) {
                try {
                    const GTComparisonMode gt_mode = frame_settings.gt_comparison_mode;
                    const glm::ivec2 preview_gt_size = gtComparisonPreviewSize(*camera, render_size);
                    const int preview_max_dimension = std::max(preview_gt_size.x, preview_gt_size.y);
                    std::shared_ptr<lfs::core::Tensor> gt_image;
                    std::string gt_error;

                    if (gt_mode == GTComparisonMode::RGB) {
                        if (camera->image_path().empty()) {
                            gt_error = "RGB GT comparison requires a source image";
                        } else {
                            const bool undistort_requested =
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed();
                            const bool cache_hit =
                                gt_comparison_image_cache_.image &&
                                gt_comparison_image_cache_.camera_uid == camera->uid() &&
                                gt_comparison_image_cache_.image_size == preview_gt_size &&
                                gt_comparison_image_cache_.undistort_requested == undistort_requested &&
                                gt_comparison_image_cache_.image_path == camera->image_path();
                            if (cache_hit) {
                                gt_image = gt_comparison_image_cache_.image;
                            } else {
                                // GT comparison loads a viewport-scaled preview. Do not publish that
                                // transient size on the shared camera; training uses image dimensions
                                // as its raster target and may be running concurrently.
                                auto gt_tensor = camera->load_and_get_image(
                                    -1, preview_max_dimension, false, false);
                                if (gt_tensor.is_valid() && gt_tensor.ndim() == 3) {
                                    const auto gt_layout = lfs::rendering::detectImageLayout(gt_tensor);
                                    if (gt_layout != lfs::rendering::ImageLayout::Unknown) {
                                        const bool undistort_gt =
                                            gt_layout == lfs::rendering::ImageLayout::CHW &&
                                            undistort_requested;
                                        if (undistort_gt) {
                                            const auto scaled = lfs::core::scale_undistort_params(
                                                camera->undistort_params(),
                                                lfs::rendering::imageWidth(gt_tensor, gt_layout),
                                                lfs::rendering::imageHeight(gt_tensor, gt_layout));
                                            gt_tensor = lfs::core::undistort_image(gt_tensor, scaled, nullptr);
                                        }
                                        // Static GT display images must be decoupled from the CUDA pool
                                        // while training can recycle device buffers mid-frame.
                                        gt_tensor = gt_tensor.cpu();
                                        gt_tensor = lfs::rendering::flipImageVertical(gt_tensor, gt_layout);
                                        gt_image = std::make_shared<lfs::core::Tensor>(std::move(gt_tensor));
                                        gt_image = resizeChwDisplayTensor(gt_image, preview_gt_size);
                                        gt_comparison_image_cache_ = {
                                            .camera_uid = camera->uid(),
                                            .undistort_requested = undistort_requested,
                                            .image_path = camera->image_path(),
                                            .image = gt_image,
                                            .image_size = preview_gt_size};
                                    }
                                }
                            }
                            if (!gt_image) {
                                gt_error = "RGB GT comparison could not load the source image";
                            }
                        }
                    } else if (gt_mode == GTComparisonMode::Depth) {
                        if (!camera->has_depth()) {
                            gt_error = "Depth GT comparison requires a depth map for the selected camera";
                        } else {
                            auto depth = camera->load_and_get_depth(-1, preview_max_dimension);
                            if (depth.is_valid() && depth.ndim() == 2 &&
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed() &&
                                !camera->is_undistort_prepared()) {
                                const auto scaled = lfs::core::scale_undistort_params(
                                    camera->undistort_params(),
                                    static_cast<int>(depth.shape()[1]),
                                    static_cast<int>(depth.shape()[0]));
                                depth = lfs::core::undistort_mask(depth, scaled, nullptr);
                            }
                            gt_image = makeDepthDisplayTensor(depth, frame_settings);
                            if (gt_image) {
                                gt_image = resizeChwDisplayTensor(gt_image, preview_gt_size);
                            }
                            if (gt_image) {
                                auto flipped = lfs::rendering::flipImageVertical(
                                    *gt_image,
                                    lfs::rendering::ImageLayout::CHW);
                                gt_image = std::make_shared<lfs::core::Tensor>(std::move(flipped));
                            } else {
                                gt_error = "Depth GT comparison could not load the camera depth map";
                            }
                        }
                    } else {
                        if (!camera->has_normal()) {
                            gt_error = "Normal GT comparison requires a normal map for the selected camera";
                        } else {
                            auto normal = camera->load_and_get_normal(
                                -1,
                                preview_max_dimension,
                                lfs::core::Camera::NormalPriorDecode{});
                            if (normal.is_valid() && normal.ndim() == 3 &&
                                camera->camera_model_type() != lfs::core::CameraModelType::EQUIRECTANGULAR &&
                                camera->is_undistort_precomputed() &&
                                !camera->is_undistort_prepared()) {
                                const auto normal_layout = lfs::rendering::detectImageLayout(normal);
                                if (normal_layout != lfs::rendering::ImageLayout::Unknown) {
                                    const auto scaled = lfs::core::scale_undistort_params(
                                        camera->undistort_params(),
                                        lfs::rendering::imageWidth(normal, normal_layout),
                                        lfs::rendering::imageHeight(normal, normal_layout));
                                    normal = lfs::core::undistort_image(normal, scaled, nullptr);
                                }
                            }
                            gt_image = makeNormalDisplayTensor(normal);
                            if (gt_image) {
                                gt_image = resizeChwDisplayTensor(gt_image, preview_gt_size);
                            }
                            if (gt_image) {
                                auto flipped = lfs::rendering::flipImageVertical(
                                    *gt_image,
                                    lfs::rendering::ImageLayout::CHW);
                                gt_image = std::make_shared<lfs::core::Tensor>(std::move(flipped));
                            } else {
                                gt_error = "Normal GT comparison could not load the camera normal map";
                            }
                        }
                    }

                    if (!gt_image || !gt_image->is_valid()) {
                        LOG_DEBUG("{}",
                                  gt_error.empty() ? "GT comparison could not load ground-truth data" : gt_error);
                        gt_image = makeGTComparePlaceholderTensor(preview_gt_size, glm::vec3(0.16f, 0.035f, 0.050f));
                    }

                    if (gt_image && gt_image->is_valid()) {
                        const auto gt_layout = lfs::rendering::detectImageLayout(*gt_image);
                        if (gt_layout == lfs::rendering::ImageLayout::Unknown) {
                            render_error = "GT comparison produced an unsupported ground-truth image layout";
                        } else {
                            const glm::ivec2 gt_size{
                                lfs::rendering::imageWidth(*gt_image, gt_layout),
                                lfs::rendering::imageHeight(*gt_image, gt_layout)};

                            auto request = buildViewportRenderRequest(frame_ctx, gt_size);
                            const glm::mat4 scene_transform =
                                detail::currentSceneTransform(scene_manager, camera->uid());
                            const auto render_camera =
                                detail::buildGTRenderCamera(*camera, gt_size, scene_transform);
                            if (render_camera) {
                                request.frame_view.rotation = render_camera->rotation;
                                request.frame_view.translation = render_camera->translation;
                                request.frame_view.intrinsics_override = render_camera->intrinsics;
                                request.frame_view.orthographic = false;
                                request.frame_view.ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
                                request.equirectangular = render_camera->equirectangular;
                            }

                            RenderedPanel compare_panel{};
                            std::string compare_error;

                            if (gt_mode != GTComparisonMode::RGB) {
                                if (!has_visible_gaussian_model || frame_settings.point_cloud_mode) {
                                    compare_error =
                                        "Normal/depth GT comparison requires a visible Gaussian model in splat mode";
                                } else if (!context.vulkan_context) {
                                    compare_error = "Normal/depth GT comparison requires an active Vulkan context";
                                } else if (!render_camera) {
                                    compare_error = "Normal/depth GT comparison could not build the GT render camera";
                                } else if (gt_mode == GTComparisonMode::Normal &&
                                           !request.frame_view.intrinsics_override.has_value()) {
                                    compare_error = "Normal GT comparison requires pinhole camera intrinsics";
                                } else {
                                    request.depth_view = false;
                                    if (!vksplat_viewport_renderer_) {
                                        vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
                                    }
                                    vksplat_viewport_renderer_->setDepthCaptureMode(true, true);
                                    struct DepthCaptureModeGuard {
                                        VksplatViewportRenderer* renderer = nullptr;
                                        ~DepthCaptureModeGuard() {
                                            if (renderer) {
                                                renderer->setDepthCaptureMode(false);
                                            }
                                        }
                                    } depth_capture_guard{vksplat_viewport_renderer_.get()};

                                    auto rendered = render_panel_image(
                                        context.viewport,
                                        gt_size,
                                        std::nullopt,
                                        std::nullopt,
                                        nullptr,
                                        nullptr,
                                        VksplatViewportRenderer::OutputSlot::Preview,
                                        &request);
                                    if (!rendered) {
                                        compare_error = rendered.error();
                                    } else {
                                        auto raw_depth = vksplat_viewport_renderer_->readOutputDepthImage(
                                            *context.vulkan_context,
                                            VksplatViewportRenderer::OutputSlot::Preview);
                                        if (!raw_depth || !*raw_depth) {
                                            compare_error = raw_depth ? "VkSplat GT comparison depth readback returned no data"
                                                                      : raw_depth.error();
                                        } else {
                                            auto display = gt_mode == GTComparisonMode::Depth
                                                               ? makeDepthDisplayTensor(**raw_depth, frame_settings)
                                                               : makeNormalDisplayFromDepthTensor(
                                                                     **raw_depth,
                                                                     *request.frame_view.intrinsics_override);
                                            if (!display || !display->is_valid()) {
                                                compare_error =
                                                    gt_mode == GTComparisonMode::Depth
                                                        ? "Depth GT comparison could not visualize rendered depth"
                                                        : "Normal GT comparison could not derive rendered normals";
                                            } else {
                                                compare_panel = std::move(*rendered);
                                                compare_panel.image = std::move(display);
                                                compare_panel.external_image_view = VK_NULL_HANDLE;
                                                compare_panel.external_image_generation = 0;
                                            }
                                        }
                                    }
                                }
                            } else {
                                const bool use_point_cloud_compare =
                                    frame_settings.point_cloud_mode || !has_visible_gaussian_model;
                                if (use_point_cloud_compare && has_visible_gaussian_model) {
                                    auto point_request = buildPointCloudRenderRequest(
                                        frame_ctx, gt_size, frame_ctx.scene_state.model_transforms);
                                    point_request.frame_view = request.frame_view;
                                    if (auto auxiliary_engine = ensure_auxiliary_rendering_engine(); auxiliary_engine) {
                                        auto rendered = (*auxiliary_engine)->renderPointCloudImage(*model, point_request);
                                        if (rendered && rendered->image) {
                                            const bool flip_y = !rendered->metadata.flip_y;
                                            compare_panel = RenderedPanel{.image = std::move(rendered->image),
                                                                          .metadata = std::move(rendered->metadata),
                                                                          .flip_y = flip_y};
                                        } else {
                                            compare_error = rendered ? "Point-cloud GT comparison render returned no image"
                                                                     : rendered.error();
                                        }
                                    } else {
                                        compare_error = auxiliary_engine.error();
                                    }
                                } else if (use_point_cloud_compare && has_point_cloud) {
                                    const std::vector<glm::mat4> point_cloud_transforms = {
                                        frame_ctx.scene_state.point_cloud_transform};
                                    auto point_request = buildPointCloudRenderRequest(
                                        frame_ctx, gt_size, point_cloud_transforms);
                                    point_request.frame_view = request.frame_view;
                                    if (auto auxiliary_engine = ensure_auxiliary_rendering_engine(); auxiliary_engine) {
                                        auto rendered = (*auxiliary_engine)->renderPointCloudImage(*frame_ctx.scene_state.point_cloud, point_request);
                                        if (rendered && rendered->image) {
                                            const bool flip_y = !rendered->metadata.flip_y;
                                            compare_panel = RenderedPanel{.image = std::move(rendered->image),
                                                                          .metadata = std::move(rendered->metadata),
                                                                          .flip_y = flip_y};
                                        } else {
                                            compare_error = rendered ? "Raw point-cloud GT comparison render returned no image"
                                                                     : rendered.error();
                                        }
                                    } else {
                                        compare_error = auxiliary_engine.error();
                                    }
                                } else if (has_visible_gaussian_model) {
                                    auto rendered = render_panel_image(
                                        context.viewport,
                                        gt_size,
                                        std::nullopt,
                                        std::nullopt,
                                        nullptr,
                                        nullptr,
                                        VksplatViewportRenderer::OutputSlot::SplitRight,
                                        &request);
                                    if (rendered) {
                                        compare_panel = std::move(*rendered);
                                    } else {
                                        compare_error = rendered.error();
                                    }
                                } else {
                                    compare_error = "GT comparison requires a renderable Gaussian model or point cloud";
                                }
                            }

                            if (!compare_panel.image && compare_panel.external_image_view == VK_NULL_HANDLE &&
                                !compare_error.empty() &&
                                !(gt_mode == GTComparisonMode::RGB &&
                                  isRetryableSharedScratchUnavailable(compare_error))) {
                                LOG_WARN("GT comparison using placeholder rendered panel: {}", compare_error);
                                compare_panel = make_placeholder_panel(gt_size, glm::vec3(0.035f, 0.045f, 0.070f));
                            }

                            if (compare_panel.image || compare_panel.external_image_view != VK_NULL_HANDLE) {
                                if (gt_mode == GTComparisonMode::RGB) {
                                    bool compare_panel_ppisp_applied = false;
                                    if (!compare_panel.image &&
                                        compare_panel.external_image_view != VK_NULL_HANDLE &&
                                        frame_settings.apply_appearance_correction &&
                                        vksplat_viewport_renderer_ &&
                                        context.vulkan_context) {
                                        auto image = vksplat_viewport_renderer_->readOutputImage(
                                            *context.vulkan_context,
                                            VksplatViewportRenderer::OutputSlot::SplitRight);
                                        if (image && *image) {
                                            compare_panel.image = applyViewportAppearanceCorrection(
                                                std::move(*image),
                                                scene_manager,
                                                frame_settings,
                                                camera->uid());
                                            compare_panel.image = ensure_cuda_viewport_image(
                                                std::move(compare_panel.image),
                                                "VkSplat GT comparison PPISP correction");
                                            if (compare_panel.image && compare_panel.image->is_valid()) {
                                                compare_panel.flip_y = compare_panel.metadata.flip_y;
                                                compare_panel.external_image_view = VK_NULL_HANDLE;
                                                compare_panel.external_image_generation = 0;
                                                compare_panel_ppisp_applied = true;
                                            }
                                        } else {
                                            LOG_WARN("VkSplat GT comparison PPISP readback failed: {}",
                                                     image ? "missing image payload" : image.error());
                                        }
                                    }
                                    if (compare_panel.image && !compare_panel_ppisp_applied) {
                                        compare_panel.image = applyViewportAppearanceCorrection(
                                            std::move(compare_panel.image),
                                            scene_manager,
                                            frame_settings,
                                            camera->uid());
                                    }
                                }

                                if (gt_image) {
                                    if (auto cuda_gt = ensure_cuda_viewport_image(
                                            gt_image,
                                            "GT comparison ground-truth display")) {
                                        gt_image = std::move(cuda_gt);
                                    }
                                }
                                if (compare_panel.image) {
                                    if (auto cuda_compare = ensure_cuda_viewport_image(
                                            compare_panel.image,
                                            "GT comparison rendered display")) {
                                        compare_panel.image = std::move(cuda_compare);
                                    }
                                }

                                const SplitCompositeContentRect rect =
                                    resolveSplitCompositeContentRect(render_size, true, gt_size);
                                pending_split_view.enabled = true;
                                pending_split_view.split_position = frame_settings.split_position;
                                pending_split_view.background = frame_settings.background_color;
                                pending_split_view.content_rect = {rect.x, rect.y, rect.width, rect.height};
                                pending_split_view.left = {std::move(gt_image), 0.0f, frame_settings.split_position, false, true};
                                pending_split_view.right =
                                    make_split_panel(compare_panel, frame_settings.split_position, 1.0f, false);
                                rendered_metadata = compare_panel.metadata;
                                rendered_image_contains_ground_truth = true;
                                rendered_gt_content_size = gt_size;
                                const char* mode_label = "GT Compare";
                                const char* left_name = "Ground Truth";
                                const char* right_name = "Rendered";
                                if (gt_mode == GTComparisonMode::Depth) {
                                    mode_label = "GT Depth Compare";
                                    left_name = "GT Depth";
                                    right_name = "Rendered Depth";
                                } else if (gt_mode == GTComparisonMode::Normal) {
                                    mode_label = "GT Normal Compare";
                                    left_name = "GT Normal";
                                    right_name = "Rendered Normal";
                                }
                                rendered_split_info = SplitViewInfo{
                                    .enabled = true,
                                    .mode_label = mode_label,
                                    .detail_label = camera->image_name(),
                                    .left_name = left_name,
                                    .right_name = right_name};
                            } else {
                                render_error = compare_error;
                            }
                        }
                    }
                } catch (const std::exception& e) {
                    render_error = std::format("GT comparison failed: {}", e.what());
                }
            }
        } else if (splitViewUsesIndependentPanels(frame_settings.split_view_mode)) {
            if (const auto layouts = split_view_service_.panelLayouts(frame_settings, render_size.x);
                layouts && render_size.x > 1) {
                auto left = render_panel_image(
                    context.viewport,
                    {std::max((*layouts)[0].width, 1), render_size.y},
                    SplitViewPanelId::Left,
                    std::nullopt,
                    nullptr,
                    nullptr,
                    VksplatViewportRenderer::OutputSlot::SplitLeft);
                auto right = render_panel_image(
                    split_view_service_.secondaryViewport(),
                    {std::max((*layouts)[1].width, 1), render_size.y},
                    SplitViewPanelId::Right,
                    std::nullopt,
                    nullptr,
                    nullptr,
                    VksplatViewportRenderer::OutputSlot::SplitRight);
                if (left && right) {
                    pending_split_view.enabled = true;
                    pending_split_view.split_position = frame_settings.split_position;
                    pending_split_view.background = frame_settings.background_color;
                    pending_split_view.content_rect = {0, 0, render_size.x, render_size.y};
                    pending_split_view.left = make_split_panel(
                        *left, (*layouts)[0].start_position, (*layouts)[0].end_position, true);
                    pending_split_view.right = make_split_panel(
                        *right, (*layouts)[1].start_position, (*layouts)[1].end_position, true);
                    rendered_metadata = makeSplitMetadata(left->metadata, right->metadata, frame_settings.split_position);
                    rendered_split_info = SplitViewInfo{
                        .enabled = true,
                        .mode_label = "Split View",
                        .detail_label = "Primary | Secondary",
                        .left_name = "Primary View",
                        .right_name = "Secondary View"};
                } else {
                    render_error = left ? right.error() : left.error();
                }
            }
        } else if (splitViewUsesPLYComparison(frame_settings.split_view_mode) && scene_manager && has_visible_gaussian_model) {
            const auto visible_nodes = scene_manager->getScene().getVisibleSplatNodeSlots();
            const auto pair = plyComparisonPairForOffset(visible_nodes.size(), frame_settings.split_view_offset);
            if (pair) {
                const auto& left_node = visible_nodes[pair->first];
                const auto& right_node = visible_nodes[pair->second];
                const size_t slot_count = std::max(frame_ctx.scene_state.model_transforms.size(),
                                                   frame_ctx.scene_state.node_visibility_mask.size());
                if (!left_node.node || !right_node.node ||
                    left_node.slot_index >= slot_count ||
                    right_node.slot_index >= slot_count) {
                    render_error = "PLY comparison render slots are unavailable";
                } else {
                    std::vector<bool> left_mask(slot_count, false);
                    std::vector<bool> right_mask(slot_count, false);
                    left_mask[left_node.slot_index] = true;
                    right_mask[right_node.slot_index] = true;

                    auto left = render_panel_image(
                        context.viewport,
                        render_size,
                        std::nullopt,
                        std::optional<std::vector<bool>>(left_mask),
                        nullptr,
                        nullptr,
                        VksplatViewportRenderer::OutputSlot::SplitLeft);
                    auto right = render_panel_image(
                        context.viewport,
                        render_size,
                        std::nullopt,
                        std::optional<std::vector<bool>>(right_mask),
                        nullptr,
                        nullptr,
                        VksplatViewportRenderer::OutputSlot::SplitRight);
                    if (left && right) {
                        pending_split_view.enabled = true;
                        pending_split_view.split_position = frame_settings.split_position;
                        pending_split_view.background = frame_settings.background_color;
                        pending_split_view.content_rect = {0, 0, render_size.x, render_size.y};
                        pending_split_view.left = make_split_panel(*left, 0.0f, frame_settings.split_position, false);
                        pending_split_view.right = make_split_panel(*right, frame_settings.split_position, 1.0f, false);
                        rendered_metadata = makeSplitMetadata(left->metadata, right->metadata, frame_settings.split_position);
                        rendered_split_info = SplitViewInfo{
                            .enabled = true,
                            .mode_label = "Split View",
                            .detail_label = std::format("{} | {}",
                                                        left_node.node->name,
                                                        right_node.node->name),
                            .left_name = left_node.node->name,
                            .right_name = right_node.node->name};
                    } else {
                        render_error = left ? right.error() : left.error();
                    }
                }
            }
        }

        // Split-view panels borrow the trainer's shared rasterizer arena while
        // training. When the trainer holds it the panel render reports a retryable
        // "shared scratch busy" error; collapsing to the full-frame fallback would
        // drop the ground-truth panel for as long as the arena stays contended.
        // Keep the last good split frame instead, mirroring the single-view path,
        // and retry next frame once the arena frees.
        if (split_view_service_.isActive(frame_settings) && !pending_split_view.enabled &&
            synchronize_vksplat_input_upload && has_cached_viewport_output &&
            isRetryableSharedScratchUnavailable(render_error)) {
            dirty_mask_.fetch_or(frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS,
                                 std::memory_order_relaxed);
            render_lock.reset();
            LOG_DEBUG("Split-view shared scratch unavailable ({}); returning cached split image",
                      render_error);
            return cached_frame_result();
        }

        const bool render_point_cloud = frame_settings.point_cloud_mode || !has_visible_gaussian_model;

        if (rendered_image || pending_split_view.enabled) {
            // Split-view paths populate pending_split_view directly; skip the
            // full-viewport fallback that would set rendered_image to a wrong-
            // sized tensor and squash the left panel through the scene interop.
        } else if (render_point_cloud &&
                   ((frame_settings.point_cloud_mode && has_visible_gaussian_model) || has_point_cloud)) {
            // Brush edits mutate sh0 in place — same tensor pointer but new
            // contents. Invalidate the derived-colors cache so the next frame
            // re-derives + re-uploads.
            if ((frame_dirty & DirtyFlag::SPLATS) != 0) {
                point_cloud_colors_cache_key_ = nullptr;
                point_cloud_colors_cache_size_ = 0;
                point_cloud_colors_cache_ = lfs::core::Tensor{};
                ++point_cloud_data_revision_;
            }
            std::vector<glm::mat4> point_cloud_transforms_storage;
            const std::vector<glm::mat4>* transforms_for_request = nullptr;
            if (frame_settings.point_cloud_mode && has_visible_gaussian_model) {
                transforms_for_request = &frame_ctx.scene_state.model_transforms;
            } else {
                point_cloud_transforms_storage = {frame_ctx.scene_state.point_cloud_transform};
                transforms_for_request = &point_cloud_transforms_storage;
            }
            auto pc_request = buildPointCloudRenderRequest(
                frame_ctx, render_size, *transforms_for_request);
            if ((frame_dirty & DirtyFlag::SELECTION) != 0) {
                ++point_cloud_preview_selection_revision_;
            }

            // Vulkan-native path: skip CUDA staging, drive an external VkImage
            // straight through the same plumbing the VkSplat backend uses.
            auto try_vulkan = [&]() -> std::optional<VulkanFrameResult> {
                if (!context.vulkan_context) {
                    return std::nullopt;
                }
                if (!point_cloud_vulkan_renderer_) {
                    point_cloud_vulkan_renderer_ = std::make_unique<PointCloudVulkanRenderer>();
                }

                lfs::core::Tensor splat_positions;
                const lfs::core::Tensor* positions_ptr = nullptr;
                const lfs::core::Tensor* colors_ptr = nullptr;
                if (frame_settings.point_cloud_mode && has_visible_gaussian_model) {
                    constexpr float SH_C0 = 0.28209479177387814f;
                    const auto& sh0 = model->sh0_raw();
                    const void* sh0_key = sh0.is_valid() ? sh0.ptr<float>() : nullptr;
                    const std::size_t sh0_count =
                        sh0.is_valid() ? static_cast<std::size_t>(sh0.size(0)) : 0;
                    if (sh0_key != point_cloud_colors_cache_key_ ||
                        sh0_count != point_cloud_colors_cache_size_ ||
                        !point_cloud_colors_cache_.is_valid()) {
                        try {
                            point_cloud_colors_cache_ =
                                (sh0.slice(1, 0, 1).squeeze(1) * SH_C0 + 0.5f).clamp(0.0f, 1.0f);
                        } catch (const std::exception& e) {
                            LOG_ERROR("Point cloud color derivation failed: {}", e.what());
                            return std::nullopt;
                        }
                        point_cloud_colors_cache_key_ = sh0_key;
                        point_cloud_colors_cache_size_ = sh0_count;
                    }
                    splat_positions = model->get_means();
                    positions_ptr = &splat_positions;
                    colors_ptr = &point_cloud_colors_cache_;
                } else {
                    positions_ptr = &frame_ctx.scene_state.point_cloud->means;
                    colors_ptr = &frame_ctx.scene_state.point_cloud->colors;
                }

                const auto vp_data = frame_ctx.makeViewportData();
                const glm::mat4 view = vp_data.getViewMatrix();
                const glm::mat4 projection = lfs::rendering::createProjectionMatrix(
                    pc_request.frame_view.size,
                    lfs::rendering::focalLengthToVFov(pc_request.frame_view.focal_length_mm),
                    pc_request.frame_view.orthographic,
                    pc_request.frame_view.ortho_scale,
                    pc_request.frame_view.near_plane,
                    pc_request.frame_view.far_plane);
                // glm::perspective/ortho emit OpenGL-NDC (Y up); Vulkan NDC has
                // Y down. Apply the same clip-space flip the mesh pass does so
                // the rendered image isn't upside-down vs. the screen quad's
                // top-left UV origin.
                glm::mat4 clip_y_flip(1.0f);
                clip_y_flip[1][1] = -1.0f;
                const glm::mat4 view_proj = clip_y_flip * projection * view;
                const float focal_y = lfs::core::fov2focal(
                    lfs::rendering::focalLengthToVFovRad(pc_request.frame_view.focal_length_mm),
                    pc_request.frame_view.size.y);

                PointCloudVulkanRenderer::RenderRequest vk_req{};
                vk_req.positions = positions_ptr;
                vk_req.colors = colors_ptr;
                vk_req.positions_revision = point_cloud_data_revision_;
                vk_req.colors_revision = point_cloud_data_revision_;
                vk_req.model_transforms = pc_request.scene.model_transforms;
                vk_req.transform_indices = pc_request.scene.transform_indices.get();
                vk_req.node_visibility_mask = &pc_request.scene.node_visibility_mask;
                if (frame_settings.point_cloud_mode && has_visible_gaussian_model &&
                    model->has_deleted_mask()) {
                    vk_req.deleted_mask = &model->deleted();
                    vk_req.deleted_mask_revision = point_cloud_data_revision_;
                }
                vk_req.selection_mask = pc_request.overlay.selection_mask.get();
                vk_req.preview_selection_mask = pc_request.overlay.transient_mask.mask;
                vk_req.selection_colors = &pc_request.overlay.selection_colors;
                vk_req.preview_selection_additive = pc_request.overlay.transient_mask.additive;
                vk_req.selection_revision = point_cloud_preview_selection_revision_;
                vk_req.preview_selection_revision = point_cloud_preview_selection_revision_;
                if (pc_request.filters.crop_box.has_value()) {
                    PointCloudVulkanRenderer::CropBox crop{};
                    crop.to_local = pc_request.filters.crop_box->transform;
                    crop.min = pc_request.filters.crop_box->min;
                    crop.max = pc_request.filters.crop_box->max;
                    crop.inverse = pc_request.filters.crop_inverse;
                    crop.desaturate = pc_request.filters.crop_desaturate;
                    vk_req.crop = crop;
                }
                vk_req.view = view;
                vk_req.view_projection = view_proj;
                vk_req.size = pc_request.frame_view.size;
                vk_req.background_color = pc_request.frame_view.background_color;
                vk_req.transparent_background = pc_request.transparent_background;
                vk_req.orthographic = pc_request.frame_view.orthographic;
                vk_req.ortho_scale = pc_request.frame_view.ortho_scale;
                vk_req.focal_y = focal_y;
                vk_req.voxel_size = pc_request.render.voxel_size;
                vk_req.scaling_modifier = pc_request.render.scaling_modifier;
                vk_req.depth_view = frame_settings.depth_view;
                vk_req.depth_view_min = frame_settings.depth_view_min;
                vk_req.depth_view_max = frame_settings.depth_view_max;
                vk_req.depth_visualization_mode = frame_settings.depth_visualization_mode;

                LOG_TIMER("renderVulkanFrame.point_cloud_vulkan");
                auto render_result = point_cloud_vulkan_renderer_->render(
                    *context.vulkan_context, vk_req);
                if (!render_result) {
                    LOG_ERROR("Point cloud Vulkan render failed: {}", render_result.error());
                    return std::nullopt;
                }

                render_lock.reset();
                clearVulkanViewportImageState(render_result->size, render_result->flip_y);
                vulkan_external_viewport_image_ = render_result->image;
                vulkan_external_viewport_image_view_ = render_result->image_view;
                vulkan_external_viewport_image_layout_ = render_result->image_layout;
                vulkan_external_viewport_image_generation_ = render_result->generation;

                lfs::rendering::FrameMetadata metadata{};
                metadata.valid = true;
                metadata.flip_y = render_result->flip_y;
                viewport_artifact_service_.clearViewportOutput();
                viewport_artifact_service_.setLazyCapture(
                    [this]() -> std::shared_ptr<lfs::core::Tensor> {
                        if (!point_cloud_vulkan_renderer_ || !last_vulkan_context_) {
                            return {};
                        }
                        auto image = point_cloud_vulkan_renderer_->readOutputImage(
                            *last_vulkan_context_,
                            PointCloudVulkanRenderer::OutputSlot::Main);
                        if (!image) {
                            LOG_ERROR("Failed to capture point-cloud Vulkan viewport image: {}",
                                      image.error());
                            return {};
                        }
                        return std::move(*image);
                    },
                    metadata,
                    render_result->size);

                if (resize_result.completed) {
                    frame_lifecycle_service_.noteResizeCompleted();
                    lfs::core::Tensor::trim_memory_pool();
                }
                queueCameraMetricsRefreshIfStale(scene_manager);
                viewport_interaction_context_.scene_manager = scene_manager;
                split_view_service_.updateInfo(FrameResources{});

                if (!frame_ctx.scene_state.meshes.empty() ||
                    environmentBackgroundEnabled(frame_settings)) {
                    auto mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);
                    populate_independent_split_mesh_panels(mesh_frame);
                    if (render_result->depth_image_view != VK_NULL_HANDLE) {
                        // Hardware depth attachment stores Vulkan-native NDC z; the
                        // depth-blit pass can use it directly without near/far conversion.
                        mesh_frame.depth_blit.external_image_view = render_result->depth_image_view;
                        mesh_frame.depth_blit.external_image_generation = render_result->depth_generation;
                        mesh_frame.depth_blit.depth_is_ndc = true;
                        mesh_frame.depth_blit.flip_y = render_result->flip_y;
                    }
                    setVulkanMeshFrame(std::move(mesh_frame));
                } else {
                    clearVulkanMeshFrame();
                }

                return VulkanFrameResult{
                    .image = {},
                    .external_image = vulkan_external_viewport_image_,
                    .external_image_view = vulkan_external_viewport_image_view_,
                    .external_image_layout = vulkan_external_viewport_image_layout_,
                    .external_image_generation = vulkan_external_viewport_image_generation_,
                    .size = vulkan_viewport_image_size_,
                    .flip_y = vulkan_viewport_image_flip_y_};
            };
            if (auto vk_result = try_vulkan(); vk_result) {
                return *vk_result;
            }

            if (!context.vulkan_context) {
                render_error = "Point-cloud viewer rendering requires an active Vulkan context";
            } else {
                render_error = "Point-cloud Vulkan render failed";
            }
        } else if (has_visible_gaussian_model) {
            auto request = buildViewportRenderRequest(frame_ctx, render_size);
            request.raster_backend =
                lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
            request.gut = lfs::rendering::isGutBackend(request.raster_backend);
            std::vector<std::uint32_t> lod_touched_chunks;
            if (lfs::rendering::isVkSplatBackend(request.raster_backend) &&
                context.vulkan_context &&
                !vksplat_viewport_renderer_) {
                vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
            }

            const bool has_lod_tree = model && model->lod_tree && model->lod_tree->has_tree();
            if (has_lod_tree) {
                // Debug colors stay on the GPU path: the selector emits
                // per-node levels alongside indices.
                const bool prefer_gpu_lod =
                    frame_settings.lod_enabled &&
                    lfs::rendering::isVkSplatBackend(request.raster_backend);
                const auto create_lod_controller = [this]() {
                    auto controller = std::make_unique<SparkLodController>();
                    controller->setReadyCallback([this] {
                        notifyAsyncLodResultsReady();
                    });
                    return controller;
                };
                if (!lod_controller_) {
                    lod_controller_ = create_lod_controller();
                }
                if (lod_controller_model_ != model) {
                    lod_controller_.reset();
                    lod_controller_ = create_lod_controller();
                    lod_controller_->attach(*model);
                    lod_controller_model_ = model;
                    lod_controller_needs_sync_traversal_ = true;
                    lod_controller_page_map_generation_ = 0;
                }
                // Spark-style quality scaler: the rendered cut targets
                // LOD Budget x Render Scale splats.
                const std::size_t effective_lod_budget = std::max<std::size_t>(
                    1,
                    static_cast<std::size_t>(
                        std::llround(static_cast<double>(frame_settings.lod_max_splats) *
                                     std::max(frame_settings.lod_render_scale, 0.1f))));
                if (vksplat_viewport_renderer_) {
                    // Bounded page pool only matters while a LoD cut is rendered;
                    // with LoD off the full-quality reference needs every page.
                    std::size_t pool_budget_splats = 0;
                    if (frame_settings.lod_enabled) {
                        constexpr std::size_t kAutoPoolFactor = 4;
                        const std::size_t floor_splats =
                            2 * effective_lod_budget + lfs::core::SplatLodTree::kChunkSplats;
                        pool_budget_splats =
                            frame_settings.lod_page_pool_splats > 0
                                ? frame_settings.lod_page_pool_splats
                                : kAutoPoolFactor * effective_lod_budget;
                        if (pool_budget_splats < floor_splats) {
                            static std::size_t last_warned_budget = 0;
                            if (last_warned_budget != pool_budget_splats) {
                                last_warned_budget = pool_budget_splats;
                                LOG_WARN("LOD page pool budget {} below working-set floor {}; clamping",
                                         pool_budget_splats,
                                         floor_splats);
                            }
                            pool_budget_splats = floor_splats;
                        }
                    }
                    vksplat_viewport_renderer_->setLodPagePoolBudget(pool_budget_splats);
                    vksplat_viewport_renderer_->setLodPoolVramFraction(frame_settings.lod_pool_vram_fraction);
                    vksplat_viewport_renderer_->setLodFadeFrames(
                        static_cast<std::uint32_t>(std::max(frame_settings.lod_fade_frames, 0)));
                    if (auto page_snapshot = vksplat_viewport_renderer_->ensureLodPageCacheSnapshot(*model);
                        page_snapshot &&
                        page_snapshot->generation != lod_controller_page_map_generation_) {
                        lod_controller_->applyPageMaps(page_snapshot->page_to_chunk,
                                                       page_snapshot->chunk_to_page,
                                                       !prefer_gpu_lod);
                        lod_controller_page_map_generation_ = page_snapshot->generation;
                        notifyAsyncLodResultsReady();
                    }
                }

                std::optional<lfs::rendering::GaussianLodGpuTraversalState> lod_gpu_traversal;
                if (frame_settings.lod_enabled) {
                    SparkLodController::LodParameters params;
                    params.max_splats = effective_lod_budget;
                    params.lod_render_scale = frame_settings.lod_render_scale;
                    params.behind_camera_penalty = frame_settings.lod_behind_camera_penalty;
                    params.cone_foveation = frame_settings.lod_cone_foveation;
                    params.cone_inner_degrees = frame_settings.lod_cone_inner_degrees;
                    params.cone_outer_degrees = frame_settings.lod_cone_outer_degrees;
                    const LodObjectFrame lod_frame = makeLodObjectFrame(request.frame_view, request.scene);
                    params.object_scale = lod_frame.object_scale;

                    // Compute pixel_scale_limit dynamically from camera FOV and viewport size,
                    // matching Spark's runtime computation.
                    {
                        const auto& fv = request.frame_view;
                        if (fv.orthographic) {
                            params.pixel_scale_limit = fv.ortho_scale / static_cast<float>(fv.size.y);
                            if (fv.ortho_scale > 0.0f) {
                                params.ortho_half_width =
                                    static_cast<float>(fv.size.x) / (2.0f * fv.ortho_scale);
                                params.ortho_half_height =
                                    static_cast<float>(fv.size.y) / (2.0f * fv.ortho_scale);
                            }
                        } else {
                            float vfov = lfs::rendering::focalLengthToVFov(fv.focal_length_mm);
                            float half_tan_fov = std::tan(glm::radians(vfov) * 0.5f);
                            params.pixel_scale_limit = (2.0f * half_tan_fov) / static_cast<float>(fv.size.y);
                            params.viewport_half_tan_y = half_tan_fov;
                            params.viewport_half_tan_x =
                                half_tan_fov * (static_cast<float>(fv.size.x) / static_cast<float>(fv.size.y));
                        }
                        // Spark multiplies each node's pixel scale by lod_scale
                        // (bigger scale = finer cut); dividing the stop limit is
                        // equivalent.
                        params.pixel_scale_limit /= std::max(params.lod_render_scale, 0.1f);
                        params.orthographic = fv.orthographic;
                    }

                    if (lod_controller_needs_sync_traversal_) {
                        // One-time sync traversal even in GPU mode: gives the
                        // renderer a valid static fallback cut for failure-mode
                        // frames before the GPU selector has produced output.
                        LOG_TIMER("lod_controller.update_sync");
                        lod_controller_->update(lod_frame.object_to_view, params);
                        lod_controller_needs_sync_traversal_ = false;
                    } else if (!prefer_gpu_lod) {
                        {
                            LOG_TIMER("lod_controller.swap_async_results");
                            lod_controller_->swapAsyncResults(true, true, true);
                        }
                        LOG_TIMER("lod_controller.update_async_request");
                        lod_controller_->updateAsync(lod_frame.object_to_view, params);
                    }
                    if (prefer_gpu_lod) {
                        lod_gpu_traversal = makeLodGpuTraversalState(
                            lod_frame,
                            params,
                            model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                        if (lod_gpu_traversal->node_count > 0) {
                            const auto budget_capacity = static_cast<std::size_t>(
                                std::ceil(static_cast<double>(effective_lod_budget) *
                                          kGpuLodRenderCapacityOverhead));
                            // Out-of-core RAD models keep only a coarse prefix in
                            // model->size(); the GPU cut selects logical tree nodes.
                            const std::size_t capacity_limit = std::max<std::size_t>(
                                model->size(),
                                model->lod_tree ? model->lod_tree->total_nodes() : 0u);
                            lod_gpu_traversal->output_capacity =
                                std::clamp<std::size_t>(budget_capacity, 1u, capacity_limit);
                            request.lod_gpu_traversal = *lod_gpu_traversal;
                        }
                    }
                } else {
                    lod_controller_->activateFullQualityReference();
                }

                lod_controller_->advanceTransition();
                const bool lod_transition_active = lod_controller_->transitionActive();
                if (lod_transition_active) {
                    notifyAsyncLodResultsReady();
                }
                const auto& selected = frame_settings.lod_enabled
                                           ? lod_controller_->selectedIndices()
                                           : lod_controller_->fullQualityIndices();
                if (!selected.empty()) {
                    request.lod_indices = selected.data();
                    if (kEnableLodTransitionWeights &&
                        frame_settings.lod_enabled &&
                        lod_transition_active) {
                        const auto& weights = lod_controller_->selectedWeights();
                        if (weights.size() == selected.size()) {
                            request.lod_weights = weights.data();
                        }
                    }
                    if (lod_controller_->pageMappingActive()) {
                        const auto& logical = frame_settings.lod_enabled
                                                  ? lod_controller_->selectedLogicalIndices()
                                                  : lod_controller_->fullQualityLogicalIndices();
                        if (logical.size() == selected.size()) {
                            request.lod_logical_indices = logical.data();
                        }
                    }
                    if (frame_settings.lod_debug_colors) {
                        const auto& levels = frame_settings.lod_enabled
                                                 ? lod_controller_->selectedLevels()
                                                 : lod_controller_->fullQualityLevels();
                        if (levels.size() == selected.size()) {
                            request.lod_levels = levels.data();
                        }
                    }
                    request.lod_count = selected.size();
                    request.lod_selection_hash = lod_controller_->selectionHash();
                    request.lod_generation = lod_controller_->statsGeneration();
                    if (!prefer_gpu_lod) {
                        // GPU mode derives prefetch priorities from the
                        // selector's chunk-touch readback instead. Passing
                        // lod_gpu_traversal here would activate GPU selection
                        // in the renderer and override the CPU cut (breaking
                        // debug colors), so the CPU path sends indices only.
                        lod_touched_chunks = lod_controller_->touchedChunks();
                        request.lod_touched_chunks = lod_touched_chunks.data();
                        request.lod_touched_chunk_count = lod_touched_chunks.size();
                    }
                }
                request.lod_debug_mode = frame_settings.lod_debug_colors;
            } else {
                lod_controller_.reset();
                lod_controller_model_ = nullptr;
                lod_controller_needs_sync_traversal_ = false;
                lod_controller_page_map_generation_ = 0;
            }

            if (lfs::rendering::isVkSplatBackend(request.raster_backend)) {
                if (!context.vulkan_context) {
                    render_error = "VkSplat backend requires an active Vulkan context";
                } else {
                    if (!vksplat_viewport_renderer_) {
                        vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
                    }
                    const auto publish_vksplat_result = [&](const VksplatViewportRenderer::RenderResult& render_result) -> VulkanFrameResult {
                        render_lock.reset();
                        note_lod_page_generation(render_result.lod_page_generation);
                        note_vksplat_render_progress(render_result);
                        const bool transparent_viewer_compositing =
                            environmentBackgroundUsesTransparentViewerCompositing(frame_settings);
                        lfs::rendering::FrameMetadata metadata{};
                        metadata.valid = true;
                        metadata.flip_y = render_result.flip_y;
                        metadata.color_has_alpha = transparent_viewer_compositing;

                        const auto publish_mesh_frame_for_vksplat = [&]() {
                            // VkSplat returns before the shared mesh-frame setup below.
                            // Republish here so overlays and mesh/environment passes see
                            // the current camera and splat depth view.
                            const bool publish_mesh_frame =
                                !frame_ctx.scene_state.meshes.empty() ||
                                environmentBackgroundEnabled(frame_settings) ||
                                render_result.depth_image_view != VK_NULL_HANDLE;
                            if (publish_mesh_frame) {
                                auto mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);
                                populate_independent_split_mesh_panels(mesh_frame);
                                if (render_result.depth_image_view != VK_NULL_HANDLE) {
                                    mesh_frame.depth_blit.external_image_view = render_result.depth_image_view;
                                    mesh_frame.depth_blit.external_image_generation = render_result.depth_generation;
                                    mesh_frame.depth_blit.depth_is_ndc = false;
                                    mesh_frame.depth_blit.flip_y = render_result.flip_y;
                                    mesh_frame.depth_blit.near_plane = request.frame_view.near_plane > 0.0f
                                                                           ? request.frame_view.near_plane
                                                                           : 0.1f;
                                    mesh_frame.depth_blit.far_plane = request.frame_view.far_plane > 0.0f
                                                                          ? request.frame_view.far_plane
                                                                          : 1000.0f;
                                }
                                setVulkanMeshFrame(std::move(mesh_frame));
                            } else {
                                clearVulkanMeshFrame();
                            }
                        };

                        const lfs::rendering::FrameView capture_frame_view = request.frame_view;
                        const auto make_capture_composite_request =
                            [&]() -> lfs::rendering::VideoCompositeFrameRequest {
                            return {
                                .viewport =
                                    {.rotation = capture_frame_view.rotation,
                                     .translation = capture_frame_view.translation,
                                     .size = capture_frame_view.size,
                                     .focal_length_mm = capture_frame_view.focal_length_mm,
                                     .orthographic = capture_frame_view.orthographic,
                                     .ortho_scale = capture_frame_view.ortho_scale},
                                .frame_view = capture_frame_view,
                                .background_color = frame_settings.background_color,
                                .environment =
                                    {.enabled = transparent_viewer_compositing,
                                     .map_path = transparent_viewer_compositing
                                                     ? std::filesystem::path(frame_settings.environment_map_path)
                                                     : std::filesystem::path{},
                                     .exposure = frame_settings.environment_exposure,
                                     .rotation_degrees = frame_settings.environment_rotation_degrees,
                                     .equirectangular = frame_settings.equirectangular},
                                .meshes = {},
                            };
                        };

                        if (frame_settings.apply_appearance_correction) {
                            auto image = transparent_viewer_compositing
                                             ? vksplat_viewport_renderer_->readOutputImageRgba(
                                                   *context.vulkan_context,
                                                   VksplatViewportRenderer::OutputSlot::Main)
                                             : vksplat_viewport_renderer_->readOutputImage(
                                                   *context.vulkan_context,
                                                   VksplatViewportRenderer::OutputSlot::Main);
                            if (image && *image) {
                                auto corrected_image = applyViewportAppearanceCorrection(
                                    std::move(*image),
                                    scene_manager,
                                    frame_settings,
                                    frame_ctx.current_camera_id);
                                corrected_image = ensure_cuda_viewport_image(
                                    std::move(corrected_image),
                                    "VkSplat viewport PPISP correction");
                                if (corrected_image && corrected_image->is_valid()) {
                                    vulkan_viewport_image_ = corrected_image;
                                    ++vulkan_viewport_image_generation_;
                                    if (vulkan_viewport_image_generation_ == 0) {
                                        ++vulkan_viewport_image_generation_;
                                    }
                                    vulkan_external_viewport_image_ = VK_NULL_HANDLE;
                                    vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
                                    vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
                                    vulkan_external_viewport_image_generation_ = 0;
                                    vulkan_viewport_image_size_ = render_result.size;
                                    vulkan_viewport_image_flip_y_ = render_result.flip_y;
                                    vulkan_gt_comparison_content_size_ = {0, 0};
                                    viewport_artifact_service_.updateFromImageOutput(
                                        corrected_image, metadata, render_result.size, true);
                                    if (transparent_viewer_compositing) {
                                        const auto capture_composite_request = make_capture_composite_request();
                                        viewport_artifact_service_.setLazyCaptureForCurrentOutput(
                                            [this,
                                             capture_composite_request,
                                             metadata,
                                             render_size = render_result.size,
                                             corrected_capture_image = corrected_image]()
                                                -> std::shared_ptr<lfs::core::Tensor> {
                                                auto* const engine = getRenderingEngine();
                                                if (!engine) {
                                                    LOG_ERROR("Failed to composite VkSplat viewport capture: rendering engine unavailable");
                                                    return {};
                                                }
                                                auto materialized = engine->materializeGpuFrame(
                                                    corrected_capture_image,
                                                    metadata,
                                                    render_size);
                                                if (!materialized || !materialized->valid()) {
                                                    LOG_ERROR("Failed to materialize corrected VkSplat viewport capture for environment composite: {}",
                                                              materialized ? "invalid frame" : materialized.error());
                                                    return corrected_capture_image;
                                                }
                                                auto composited = engine->renderVideoCompositeFrame(
                                                    *materialized,
                                                    capture_composite_request);
                                                if (!composited) {
                                                    LOG_ERROR("Failed to composite environment into corrected VkSplat viewport capture: {}",
                                                              composited.error());
                                                    return corrected_capture_image;
                                                }
                                                return makeViewportCaptureImageHwc(std::move(*composited));
                                            },
                                            metadata,
                                            render_result.size);
                                    }

                                    if (resize_result.completed) {
                                        frame_lifecycle_service_.noteResizeCompleted();
                                        lfs::core::Tensor::trim_memory_pool();
                                    }
                                    queueCameraMetricsRefreshIfStale(scene_manager);
                                    viewport_interaction_context_.scene_manager = scene_manager;
                                    split_view_service_.updateInfo(FrameResources{});
                                    publish_mesh_frame_for_vksplat();
                                    release_inactive_split_outputs();

                                    return {.image = vulkan_viewport_image_,
                                            .image_generation = vulkan_viewport_image_generation_,
                                            .size = vulkan_viewport_image_size_,
                                            .flip_y = vulkan_viewport_image_flip_y_};
                                }
                                LOG_WARN("VkSplat PPISP correction produced no valid viewport image; falling back to uncorrected external image");
                            } else {
                                LOG_WARN("VkSplat PPISP readback failed: {}",
                                         image ? "missing image payload" : image.error());
                            }
                        }

                        clearVulkanViewportImageState(render_result.size, render_result.flip_y);
                        vulkan_external_viewport_image_ = render_result.image;
                        vulkan_external_viewport_image_view_ = render_result.image_view;
                        vulkan_external_viewport_image_layout_ = render_result.image_layout;
                        vulkan_external_viewport_image_generation_ = render_result.generation;
                        const auto capture_composite_request = make_capture_composite_request();
                        viewport_artifact_service_.setLazyCapture(
                            [this, transparent_viewer_compositing, capture_composite_request, metadata, render_size = render_result.size]()
                                -> std::shared_ptr<lfs::core::Tensor> {
                                if (!vksplat_viewport_renderer_ || !last_vulkan_context_) {
                                    return {};
                                }
                                auto image = transparent_viewer_compositing
                                                 ? vksplat_viewport_renderer_->readOutputImageRgba(
                                                       *last_vulkan_context_,
                                                       VksplatViewportRenderer::OutputSlot::Main)
                                                 : vksplat_viewport_renderer_->readOutputImage(
                                                       *last_vulkan_context_,
                                                       VksplatViewportRenderer::OutputSlot::Main);
                                if (!image) {
                                    LOG_ERROR("Failed to capture VkSplat viewport image: {}", image.error());
                                    return {};
                                }
                                if (transparent_viewer_compositing) {
                                    auto* const engine = getRenderingEngine();
                                    if (!engine) {
                                        LOG_ERROR("Failed to composite VkSplat viewport capture: rendering engine unavailable");
                                        return {};
                                    }
                                    auto frame_image = std::move(*image);
                                    auto materialized = engine->materializeGpuFrame(
                                        frame_image,
                                        metadata,
                                        render_size);
                                    if (!materialized || !materialized->valid()) {
                                        LOG_ERROR("Failed to materialize VkSplat viewport capture for environment composite: {}",
                                                  materialized ? "invalid frame" : materialized.error());
                                        return frame_image;
                                    }
                                    auto composited = engine->renderVideoCompositeFrame(
                                        *materialized,
                                        capture_composite_request);
                                    if (!composited) {
                                        LOG_ERROR("Failed to composite environment into VkSplat viewport capture: {}",
                                                  composited.error());
                                        return frame_image;
                                    }
                                    return makeViewportCaptureImageHwc(std::move(*composited));
                                }
                                return std::move(*image);
                            },
                            metadata,
                            render_result.size);

                        if (resize_result.completed) {
                            frame_lifecycle_service_.noteResizeCompleted();
                            lfs::core::Tensor::trim_memory_pool();
                        }
                        queueCameraMetricsRefreshIfStale(scene_manager);
                        viewport_interaction_context_.scene_manager = scene_manager;
                        split_view_service_.updateInfo(FrameResources{});

                        publish_mesh_frame_for_vksplat();
                        release_inactive_split_outputs();

                        return {.image = {},
                                .external_image = vulkan_external_viewport_image_,
                                .external_image_view = vulkan_external_viewport_image_view_,
                                .external_image_layout = vulkan_external_viewport_image_layout_,
                                .external_image_generation = vulkan_external_viewport_image_generation_,
                                .completion_semaphore = render_result.completion_semaphore,
                                .completion_value = render_result.completion_value,
                                .size = vulkan_viewport_image_size_,
                                .flip_y = vulkan_viewport_image_flip_y_};
                    };

                    const DirtyMask non_overlay_dirty = frame_dirty & ~DirtyFlag::SELECTION;
                    const bool can_rerender_selection_overlay =
                        (frame_dirty & DirtyFlag::SELECTION) != 0 &&
                        (frame_dirty == DirtyFlag::SELECTION ||
                         (is_training && (non_overlay_dirty & ~DirtyFlag::SPLATS) == 0)) &&
                        vulkan_external_viewport_image_ != VK_NULL_HANDLE &&
                        vulkan_viewport_image_size_ == render_size &&
                        !split_view_service_.isActive(frame_settings);
                    if (can_rerender_selection_overlay) {
                        LOG_TIMER("vksplat.selection_overlay");
                        std::expected<VksplatViewportRenderer::RenderResult, std::string> overlay_result =
                            std::unexpected("VkSplat selection overlay was not executed");
                        try {
                            overlay_result = vksplat_viewport_renderer_->rerenderSelectionOverlay(
                                *context.vulkan_context,
                                *model,
                                request,
                                VksplatViewportRenderer::OutputSlot::Main,
                                synchronize_vksplat_input_upload);
                        } catch (const std::exception& e) {
                            overlay_result = std::unexpected(
                                std::format("VkSplat selection overlay threw: {}", e.what()));
                            lfs::core::Tensor::trim_memory_pool();
                        }
                        if (overlay_result) {
                            if (is_training && non_overlay_dirty != 0) {
                                dirty_mask_.fetch_or(non_overlay_dirty, std::memory_order_relaxed);
                            }
                            return publish_vksplat_result(*overlay_result);
                        }
                        LOG_DEBUG("VkSplat selection overlay fast path unavailable: {}",
                                  overlay_result.error());
                    }

                    const bool force_input_upload = (frame_dirty & DirtyFlag::SPLATS) != 0;
                    LOG_TIMER("vksplat.render");
                    std::expected<VksplatViewportRenderer::RenderResult, std::string> render_result =
                        std::unexpected("VkSplat render was not executed");
                    try {
                        render_result = vksplat_viewport_renderer_->render(
                            *context.vulkan_context,
                            *model,
                            request,
                            force_input_upload,
                            VksplatViewportRenderer::OutputSlot::Main,
                            synchronize_vksplat_input_upload);
                    } catch (const std::exception& e) {
                        render_result = std::unexpected(std::format("VkSplat render threw: {}", e.what()));
                        lfs::core::Tensor::trim_memory_pool();
                    }
                    if (render_result) {
                        last_logged_vksplat_render_error_.clear();
                        return publish_vksplat_result(*render_result);
                    }
                    const bool shared_scratch_retryable =
                        isRetryableSharedScratchUnavailable(render_result.error());
                    const bool output_resize_wait_retryable =
                        isRetryableVksplatOutputResizeWait(render_result.error());
                    if (synchronize_vksplat_input_upload &&
                        has_cached_viewport_output &&
                        (shared_scratch_retryable || output_resize_wait_retryable)) {
                        const DirtyMask retry_dirty =
                            output_resize_wait_retryable
                                ? vksplatOutputResizeRetryDirty(frame_dirty)
                                : vksplatSharedScratchRetryDirty(frame_dirty);
                        dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
                        const bool cached_size_matches = vulkan_viewport_image_size_ == render_size;
                        if (vksplat_viewport_resize || !cached_size_matches) {
                            LOG_DEBUG("{} ({}); skipping cached viewport image, retry_dirty=0x{:x}, vksplat_resize={}, cached_size={}x{}, render_size={}x{}",
                                      output_resize_wait_retryable
                                          ? "VkSplat output resize wait is still pending"
                                          : "VkSplat shared scratch unavailable",
                                      render_result.error(),
                                      retry_dirty,
                                      vksplat_viewport_resize,
                                      vulkan_viewport_image_size_.x,
                                      vulkan_viewport_image_size_.y,
                                      render_size.x,
                                      render_size.y);
                            render_lock.reset();
                            return {};
                        }
                        LOG_DEBUG("{} ({}); returning cached viewport image, retry_dirty=0x{:x}",
                                  output_resize_wait_retryable
                                      ? "VkSplat output resize wait is still pending"
                                      : "VkSplat shared scratch unavailable",
                                  render_result.error(),
                                  retry_dirty);
                        render_lock.reset();
                        return cached_frame_result();
                    }
                    render_error = render_result.error();
                }
            } else {
                render_error = "Gaussian viewer rendering requires a VkSplat backend";
            }
        }

        if (rendered_image && !rendered_image_contains_ground_truth) {
            rendered_image = applyViewportAppearanceCorrection(
                std::move(rendered_image),
                scene_manager,
                frame_settings,
                frame_ctx.current_camera_id);
        }

        if (frame_ctx.scene_state.meshes.empty()) {
            clearVulkanMeshFrame();
        }

        if ((rendered_image || render_error.empty() || pending_split_view.enabled) &&
            (environmentBackgroundEnabled(frame_settings) || !frame_ctx.scene_state.meshes.empty() ||
             pending_split_view.enabled)) {
            VulkanMeshFrame gpu_mesh_frame = populateMeshFrame(frame_ctx, frame_settings, pending_split_view);
            populate_independent_split_mesh_panels(gpu_mesh_frame);

            // Splat depth -> mesh-pass z-test source. Only meaningful when the
            // active render path produced a tensor-backed depth output.
            if (rendered_image && rendered_metadata.depth_panel_count > 0 &&
                rendered_metadata.depth_panels[0].depth &&
                rendered_metadata.depth_panels[0].depth->is_valid()) {
                gpu_mesh_frame.depth_blit.depth = rendered_metadata.depth_panels[0].depth;
                gpu_mesh_frame.depth_blit.depth_is_ndc = rendered_metadata.depth_is_ndc;
                // Depth and color tensors share storage orientation; the viewport
                // pass already flips the screen quad's UVs for the color image,
                // so the depth-blit pass inherits that flip and needs no extra one.
                gpu_mesh_frame.depth_blit.flip_y = false;
                gpu_mesh_frame.depth_blit.near_plane = rendered_metadata.near_plane > 0.0f
                                                           ? rendered_metadata.near_plane
                                                           : 0.1f;
                gpu_mesh_frame.depth_blit.far_plane = rendered_metadata.far_plane > 0.0f
                                                          ? rendered_metadata.far_plane
                                                          : 1000.0f;
            }

            setVulkanMeshFrame(std::move(gpu_mesh_frame));
        }
        render_lock.reset();

        const bool has_gpu_only_pass =
            !frame_ctx.scene_state.meshes.empty() ||
            environmentBackgroundEnabled(frame_settings) ||
            pending_split_view.enabled;

        if (!rendered_image && has_gpu_only_pass) {
            clearVulkanViewportImageState(render_size, false);
            vulkan_gt_comparison_content_size_ =
                rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
            FrameResources split_info_resources;
            if (rendered_split_info) {
                split_info_resources.split_view_executed = true;
                split_info_resources.split_info = *rendered_split_info;
            }
            split_view_service_.updateInfo(split_info_resources);

            if (pending_split_view.enabled) {
                viewport_artifact_service_.setLazyCapture(
                    [this, params = pending_split_view, render_size]()
                        -> std::shared_ptr<lfs::core::Tensor> {
                        VulkanSplitViewParams capture_params = params;
                        {
                            std::lock_guard lock(vulkan_mesh_frame_mutex_);
                            if (vulkan_mesh_frame_.split_view.enabled) {
                                capture_params = vulkan_mesh_frame_.split_view;
                            }
                        }
                        const auto read_panel = [this](const VksplatViewportRenderer::OutputSlot slot)
                            -> std::shared_ptr<lfs::core::Tensor> {
                            if (!vksplat_viewport_renderer_ || !last_vulkan_context_) {
                                return {};
                            }
                            auto image = vksplat_viewport_renderer_->readOutputImage(
                                *last_vulkan_context_,
                                slot);
                            if (!image) {
                                LOG_ERROR("Failed to capture VkSplat split-view panel: {}", image.error());
                                return {};
                            }
                            return std::move(*image);
                        };
                        if (!capture_params.left.image &&
                            capture_params.left.external_image_view != VK_NULL_HANDLE) {
                            capture_params.left.image =
                                read_panel(VksplatViewportRenderer::OutputSlot::SplitLeft);
                        }
                        if (!capture_params.right.image &&
                            capture_params.right.external_image_view != VK_NULL_HANDLE) {
                            capture_params.right.image =
                                read_panel(VksplatViewportRenderer::OutputSlot::SplitRight);
                        }
                        if (!capture_params.left.image || !capture_params.right.image) {
                            return {};
                        }
                        return composeSplitViewCpu(capture_params, render_size);
                    },
                    rendered_metadata,
                    render_size);
            } else {
                viewport_artifact_service_.clearViewportOutput();
            }

            // Split-view: hand both panels to gui_manager so the existing scene
            // image interop covers the left panel and a parallel slot covers the right.
            // Both arrive in the viewport pass as external Vulkan image views; the
            // CPU staging path stays as a fallback for any frame where interop fails.
            VulkanFrameResult result{};
            result.image_generation = ++split_view_image_generation_;
            if (result.image_generation == 0) {
                result.image_generation = ++split_view_image_generation_;
            }
            const bool split_uses_external_image =
                pending_split_view.enabled &&
                (pending_split_view.left.external_image_view != VK_NULL_HANDLE ||
                 pending_split_view.right.external_image_view != VK_NULL_HANDLE);
            if (split_uses_external_image) {
                result.completion_semaphore = latest_vksplat_completion_semaphore;
                result.completion_value = latest_vksplat_completion_value;
            }
            result.size = vulkan_viewport_image_size_;
            result.flip_y = vulkan_viewport_image_flip_y_;

            const auto tensor_size = [](const lfs::core::Tensor& t) -> glm::ivec2 {
                const auto layout = lfs::rendering::detectImageLayout(t);
                if (layout == lfs::rendering::ImageLayout::Unknown) {
                    return {0, 0};
                }
                return {lfs::rendering::imageWidth(t, layout),
                        lfs::rendering::imageHeight(t, layout)};
            };

            if (pending_split_view.enabled) {
                if (auto left_tensor = pending_split_view.left.image) {
                    const auto sz = tensor_size(*left_tensor);
                    result.image = std::move(left_tensor);
                    result.size = sz;
                    result.flip_y = pending_split_view.left.flip_y;
                }
                if (auto right_tensor = pending_split_view.right.image) {
                    const auto sz = tensor_size(*right_tensor);
                    result.split_right_image = std::move(right_tensor);
                    result.split_right_size = sz;
                    result.split_right_flip_y = pending_split_view.right.flip_y;
                }
            }
            if (!pending_split_view.enabled) {
                release_inactive_split_outputs();
            }
            return result;
        }

        if (!rendered_image) {
            const bool shared_scratch_retryable =
                synchronize_vksplat_input_upload &&
                isRetryableSharedScratchUnavailable(render_error);
            const bool output_resize_wait_retryable =
                isRetryableVksplatOutputResizeWait(render_error);
            if (shared_scratch_retryable || output_resize_wait_retryable) {
                const DirtyMask retry_dirty =
                    output_resize_wait_retryable
                        ? vksplatOutputResizeRetryDirty(frame_dirty)
                        : vksplatSharedScratchRetryDirty(frame_dirty);
                dirty_mask_.fetch_or(retry_dirty,
                                     std::memory_order_relaxed);
                render_lock.reset();
                const bool cached_size_matches = vulkan_viewport_image_size_ == render_size;
                if (has_cached_viewport_output && !vksplat_viewport_resize && cached_size_matches) {
                    LOG_DEBUG("{} ({}); returning cached viewport image, retry_dirty=0x{:x}",
                              output_resize_wait_retryable
                                  ? "VkSplat output resize wait is still pending"
                                  : "VkSplat shared scratch unavailable",
                              render_error,
                              retry_dirty);
                    return cached_frame_result();
                }

                LOG_DEBUG("{} ({}); skipping viewport frame, retry_dirty=0x{:x}, cached_output={}, vksplat_resize={}, cached_size={}x{}, render_size={}x{}",
                          output_resize_wait_retryable
                              ? "VkSplat output resize wait is still pending"
                              : "VkSplat shared scratch unavailable",
                          render_error,
                          retry_dirty,
                          has_cached_viewport_output,
                          vksplat_viewport_resize,
                          vulkan_viewport_image_size_.x,
                          vulkan_viewport_image_size_.y,
                          render_size.x,
                          render_size.y);
                return {};
            }

            if (has_visible_gaussian_model &&
                lfs::rendering::isVkSplatBackend(frame_settings.raster_backend)) {
                const std::string degraded_error =
                    render_error.empty() ? "missing image payload" : render_error;
                if (last_logged_vksplat_render_error_ != degraded_error) {
                    last_logged_vksplat_render_error_ = degraded_error;
                    LOG_ERROR("VkSplat entered degraded mode; retaining the last good viewport image: {}",
                              degraded_error);
                }

                // A failed attempt never publishes its candidate completion
                // value. Retry next frame, but keep presenting the previous
                // image when its dimensions still match.
                const DirtyMask retry_dirty = frame_dirty != 0 ? frame_dirty : DirtyFlag::SPLATS;
                dirty_mask_.fetch_or(retry_dirty, std::memory_order_relaxed);
                render_lock.reset();
                const bool cached_size_matches = vulkan_viewport_image_size_ == render_size;
                if (has_cached_viewport_output && !vksplat_viewport_resize && cached_size_matches) {
                    return cached_frame_result();
                }
                return {};
            }

            LOG_ERROR("Failed to render Vulkan viewport image: {}",
                      render_error.empty() ? "missing image payload" : render_error);
            clearVulkanViewportImageState();
            return {};
        }

        auto viewport_image = std::move(rendered_image);
        vulkan_viewport_image_ = viewport_image;
        ++vulkan_viewport_image_generation_;
        if (vulkan_viewport_image_generation_ == 0)
            ++vulkan_viewport_image_generation_;
        vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        vulkan_external_viewport_image_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        vulkan_external_viewport_image_generation_ = 0;
        vulkan_viewport_image_size_ = render_size;
        vulkan_viewport_image_flip_y_ = !rendered_metadata.flip_y;
        vulkan_gt_comparison_content_size_ =
            rendered_image_contains_ground_truth ? rendered_gt_content_size : glm::ivec2{0, 0};
        viewport_artifact_service_.updateFromImageOutput(
            std::move(viewport_image), rendered_metadata, render_size, true);
        release_inactive_split_outputs();

        if (resize_result.completed) {
            frame_lifecycle_service_.noteResizeCompleted();
            lfs::core::Tensor::trim_memory_pool();
        }

        queueCameraMetricsRefreshIfStale(scene_manager);
        viewport_interaction_context_.scene_manager = scene_manager;
        FrameResources split_info_resources;
        if (rendered_split_info) {
            split_info_resources.split_view_executed = true;
            split_info_resources.split_info = std::move(*rendered_split_info);
        }
        split_view_service_.updateInfo(split_info_resources);

        return {.image = vulkan_viewport_image_,
                .image_generation = vulkan_viewport_image_generation_,
                .size = vulkan_viewport_image_size_,
                .flip_y = vulkan_viewport_image_flip_y_};
    }

    std::expected<void, std::string> RenderingManager::ensureVksplatTrainingSharedScratchReady(
        VulkanContext& context,
        const lfs::core::SplatData& model,
        glm::ivec2 viewport_size) {
        const RenderSettings settings = getSettings();
        if (!lfs::rendering::isVkSplatBackend(settings.raster_backend) || model.size() <= 0) {
            return {};
        }
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = getRenderedSize();
        }
        if (viewport_size.x <= 0 || viewport_size.y <= 0) {
            viewport_size = {1280, 720};
        }
        if (!vksplat_viewport_renderer_) {
            vksplat_viewport_renderer_ = std::make_unique<VksplatViewportRenderer>();
        }
        if (auto ok = vksplat_viewport_renderer_->ensureHandshakeReady(context); !ok) {
            return std::unexpected(ok.error());
        }
        return vksplat_viewport_renderer_->ensureTrainingSharedScratchReady(
            context,
            static_cast<std::size_t>(model.size()),
            viewport_size);
    }

    lfs::io::SplatTensorAllocator RenderingManager::makeSplatTensorAllocator() const {
        if (!last_vulkan_context_ || !last_vulkan_context_->externalMemoryInteropEnabled()) {
            return {};
        }
        return [context = last_vulkan_context_](lfs::core::TensorShape shape,
                                                const size_t capacity,
                                                const lfs::core::DataType dtype,
                                                const std::string_view name) -> lfs::core::Tensor {
            const std::string debug_name{name};
            auto tensor = makeVulkanExternalTensor(
                *context,
                std::move(shape),
                dtype,
                capacity,
                debug_name.c_str(),
                nullptr,
                false);
            if (!tensor) {
                throw lfs::core::TensorError(std::format(
                    "Vulkan-external thumbnail splat tensor allocation failed for '{}': {}",
                    debug_name,
                    tensor.error()));
            }
            tensor->set_name(debug_name);
            return std::move(*tensor);
        };
    }

} // namespace lfs::vis
