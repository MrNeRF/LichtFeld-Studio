/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_artifact_service.hpp"
#include "core/cuda_debug.hpp"
#include "rendering/rendering.hpp"
#include <cmath>
#include <cuda_runtime.h>

namespace lfs::vis {

    namespace {

        [[nodiscard]] float linearizeDepthSample(const float depth_sample,
                                                 const float near_plane,
                                                 const float far_plane,
                                                 const bool orthographic,
                                                 const bool depth_is_ndc) {
            if (!depth_is_ndc) {
                return depth_sample < 1e9f ? depth_sample : -1.0f;
            }

            constexpr float DEPTH_BG_THRESHOLD = 0.9999f;
            if (depth_sample >= DEPTH_BG_THRESHOLD) {
                return -1.0f;
            }

            if (orthographic) {
                return near_plane + depth_sample * (far_plane - near_plane);
            }

            const float z_ndc = depth_sample * 2.0f - 1.0f;
            const float A = (far_plane + near_plane) / (far_plane - near_plane);
            const float B = (2.0f * far_plane * near_plane) / (far_plane - near_plane);
            return B / (A - z_ndc);
        }

    } // namespace

    ViewportArtifactService::~ViewportArtifactService() = default;

    bool ViewportArtifactService::hasGpuFrame() const {
        return gpu_frame_ && gpu_frame_->valid();
    }

    bool ViewportArtifactService::hasViewportOutput() const {
        return hasGpuFrame() || (captured_image_ && captured_image_->is_valid());
    }

    bool ViewportArtifactService::hasOutputArtifacts() const {
        for (size_t i = 0; i < metadata_.depth_panel_count && i < metadata_.depth_panels.size(); ++i) {
            if (metadata_.depth_panels[i].depth && metadata_.depth_panels[i].depth->is_valid()) {
                return true;
            }
        }
        return hasGpuFrame() || rendered_size_.x > 0 || rendered_size_.y > 0;
    }

    std::shared_ptr<lfs::core::Tensor> ViewportArtifactService::getCapturedImageIfCurrent() const {
        if (captured_image_ && captured_artifact_generation_ == artifact_generation_) {
            return captured_image_;
        }
        return {};
    }

    void ViewportArtifactService::invalidateCapture() {
        captured_image_.reset();
        captured_artifact_generation_ = 0;
        ++artifact_generation_;
        if (artifact_generation_ == 0) {
            artifact_generation_ = 1;
        }
    }

    void ViewportArtifactService::clearViewportOutput() {
        metadata_ = {};
        gpu_frame_.reset();
        rendered_size_ = {0, 0};
        lazy_capture_ = {};
        invalidateCapture();
    }

    void ViewportArtifactService::updateFromFrameResources(const FrameResources& resources,
                                                           const bool viewport_output_updated) {
        metadata_ = resources.cached_metadata;
        gpu_frame_ = resources.cached_gpu_frame;
        rendered_size_ = resources.cached_result_size;
        if (viewport_output_updated) {
            invalidateCapture();
        }
    }

    void ViewportArtifactService::updateFromImageOutput(std::shared_ptr<lfs::core::Tensor> image,
                                                        const lfs::rendering::FrameMetadata& metadata,
                                                        const glm::ivec2& rendered_size,
                                                        const bool viewport_output_updated) {
        metadata_ = makeCachedRenderMetadata(metadata);
        gpu_frame_.reset();
        rendered_size_ = rendered_size;
        lazy_capture_ = {};
        if (viewport_output_updated) {
            invalidateCapture();
        }
        storeCapturedImage(std::move(image));
    }

    void ViewportArtifactService::storeCapturedImage(std::shared_ptr<lfs::core::Tensor> image) {
        captured_image_ = std::move(image);
        captured_artifact_generation_ = captured_image_ ? artifact_generation_ : 0;
    }

    void ViewportArtifactService::setLazyCapture(LazyCaptureFn fn,
                                                 const lfs::rendering::FrameMetadata& metadata,
                                                 const glm::ivec2& rendered_size) {
        metadata_ = makeCachedRenderMetadata(metadata);
        gpu_frame_.reset();
        rendered_size_ = rendered_size;
        invalidateCapture();
        lazy_capture_ = std::move(fn);
    }

    std::shared_ptr<lfs::core::Tensor> ViewportArtifactService::resolveLazyCapture() {
        if (!lazy_capture_) {
            return {};
        }
        if (captured_image_ && captured_artifact_generation_ == artifact_generation_) {
            return captured_image_;
        }
        auto image = lazy_capture_();
        storeCapturedImage(image);
        return image;
    }

    float ViewportArtifactService::sampleLinearDepthAt(
        const int x,
        const int y,
        const glm::ivec2& fallback_viewport_size,
        const lfs::rendering::RenderingEngine* const engine,
        const std::optional<SplitViewPanelId> panel) const {
        int viewport_width = rendered_size_.x;
        int viewport_height = rendered_size_.y;
        if (viewport_width <= 0 || viewport_height <= 0) {
            viewport_width = fallback_viewport_size.x;
            viewport_height = fallback_viewport_size.y;
            if (viewport_width <= 0 || viewport_height <= 0) {
                return -1.0f;
            }
        }

        float splat_depth = -1.0f;

        const float active_near_plane =
            (gpu_frame_ && gpu_frame_->valid()) ? gpu_frame_->near_plane
                                                : (metadata_.valid ? metadata_.near_plane
                                                                   : lfs::rendering::DEFAULT_NEAR_PLANE);
        const float active_far_plane =
            (gpu_frame_ && gpu_frame_->valid()) ? gpu_frame_->far_plane
                                                : (metadata_.valid ? metadata_.far_plane
                                                                   : lfs::rendering::DEFAULT_FAR_PLANE);
        const bool active_orthographic =
            (gpu_frame_ && gpu_frame_->valid()) ? gpu_frame_->orthographic
                                                : metadata_.orthographic;

        if (metadata_.valid) {
            const lfs::core::Tensor* depth_ptr = nullptr;
            int panel_local_x = x;
            int panel_viewport_width = viewport_width;

            if (metadata_.depth_panel_count > 0) {
                size_t panel_index = 0;
                int panel_start_x = 0;
                int panel_end_x = viewport_width;

                if (panel && metadata_.depth_panel_count > 1) {
                    panel_index = (*panel == SplitViewPanelId::Right) ? 1 : 0;
                } else if (!panel && metadata_.depth_panel_count > 1) {
                    for (size_t i = 0; i < metadata_.depth_panel_count && i < metadata_.depth_panels.size(); ++i) {
                        const auto& depth_panel = metadata_.depth_panels[i];
                        const int candidate_start =
                            static_cast<int>(std::lround(static_cast<float>(viewport_width) * depth_panel.start_position));
                        const int candidate_end =
                            static_cast<int>(std::lround(static_cast<float>(viewport_width) * depth_panel.end_position));
                        if (x >= candidate_start && (x < candidate_end || i + 1 == metadata_.depth_panel_count)) {
                            panel_index = i;
                            panel_start_x = candidate_start;
                            panel_end_x = candidate_end;
                            break;
                        }
                    }
                }

                if (panel_index < metadata_.depth_panel_count && panel_index < metadata_.depth_panels.size()) {
                    const auto& depth_panel = metadata_.depth_panels[panel_index];
                    panel_start_x =
                        static_cast<int>(std::lround(static_cast<float>(viewport_width) * depth_panel.start_position));
                    panel_end_x =
                        static_cast<int>(std::lround(static_cast<float>(viewport_width) * depth_panel.end_position));
                    panel_viewport_width = std::max(panel_end_x - panel_start_x, 1);
                    if (!panel) {
                        panel_local_x -= panel_start_x;
                    }
                    if (depth_panel.depth && depth_panel.depth->is_valid()) {
                        depth_ptr = depth_panel.depth.get();
                    }
                }
            }

            if (depth_ptr && depth_ptr->ndim() == 3) {
                const int depth_height = static_cast<int>(depth_ptr->size(1));
                const int depth_width = static_cast<int>(depth_ptr->size(2));

                int scaled_x = x;
                int scaled_y = y;
                if (depth_width != panel_viewport_width || depth_height != viewport_height) {
                    scaled_x = static_cast<int>(static_cast<float>(panel_local_x) * depth_width / panel_viewport_width);
                    scaled_y = static_cast<int>(static_cast<float>(y) * depth_height / viewport_height);
                } else {
                    scaled_x = panel_local_x;
                }

                // The CUDA rasterizer outputs depth with a bottom-left origin. Tools operate
                // in window coordinates with a top-left origin, so flip Y before sampling.
                scaled_y = (depth_height - 1) - scaled_y;

                if (scaled_x >= 0 && scaled_x < depth_width && scaled_y >= 0 && scaled_y < depth_height) {
                    float d;
                    const float* gpu_ptr = depth_ptr->ptr<float>() + scaled_y * depth_width + scaled_x;
                    CHECK_CUDA(cudaMemcpy(&d, gpu_ptr, sizeof(float), cudaMemcpyDeviceToHost));
                    splat_depth = linearizeDepthSample(
                        d, active_near_plane, active_far_plane, active_orthographic, metadata_.depth_is_ndc);
                }
            }
        }

        if (splat_depth > 0.0f) {
            return splat_depth;
        }
        (void)engine;
        return -1.0f;
    }

} // namespace lfs::vis
