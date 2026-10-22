/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_viewport_renderer.hpp"
#include <cstring>
#include <glm/gtc/type_ptr.hpp>

namespace lfs::vis {
    namespace {
        VulkanGSRendererUniforms cameraKey(VulkanGSRendererUniforms uniforms) {
            uniforms.sort_capacity = 0;
            uniforms.wave_base = 0;
            uniforms.depth_wave = 0;
            uniforms.shN_address = 0;
            uniforms.stochastic_frame = 0;
            uniforms.stochastic_history_valid = 0;
            return uniforms;
        }

        glm::mat4 viewWithoutSetback(const VulkanGSRendererUniforms& camera) {
            auto view = glm::make_mat4(camera.world_view_transform);
            view[3][2] -= camera.depth_offset;
            return view;
        }
    } // namespace

    VksplatViewportRenderer::StochasticFrame VksplatViewportRenderer::prepareStochasticFrame(
        VulkanGSRendererUniforms& uniforms, const lfs::core::SplatData& model,
        const OutputSlot slot, const bool content_changed) const {
        const auto& history = stochastic_histories_[static_cast<std::size_t>(slot)];
        StochasticFrame frame{.camera = cameraKey(uniforms),
                              .model = &model,
                              .generation = stochastic_history_generation_,
                              .enabled = uniforms.stochastic_mode != 0u};
        if (!frame.enabled)
            return frame;
        const bool compatible = history.valid && !content_changed && history.model == &model &&
                                history.generation == frame.generation &&
                                history.camera.image_width == uniforms.image_width &&
                                history.camera.image_height == uniforms.image_height &&
                                history.camera.num_splats == uniforms.num_splats &&
                                history.camera.stochastic_mode == uniforms.stochastic_mode;
        const bool same_camera = compatible &&
                                 std::memcmp(&history.camera, &frame.camera, sizeof(frame.camera)) == 0;
        frame.samples = same_camera ? history.samples : 0u;
        frame.history_valid = compatible && (same_camera || uniforms.stochastic_mode == 2u);
        frame.output_color = frame.history_valid ? history.last_color ^ 1u : 0u;
        uniforms.stochastic_frame = stochastic_sample_serial_ + 1u;
        uniforms.stochastic_history_valid = frame.history_valid ? 1u : 0u;
        return frame;
    }

    const _VulkanBuffer* VksplatViewportRenderer::resolveStochasticFrame(
        const VulkanGSRendererUniforms& uniforms, const StochasticFrame& frame,
        const OutputSlot slot, const glm::vec3& background) {
        if (!frame.enabled)
            return nullptr;
        auto& history = stochastic_histories_[static_cast<std::size_t>(slot)];
        if (uniforms.stochastic_mode == 2u)
            renderer_.executeStochasticReuse(uniforms, buffers_, history.buffers);

        const auto& previous = history.camera;
        VulkanGSStochasticResolveUniforms resolve{};
        resolve.width = uniforms.image_width;
        resolve.height = uniforms.image_height;
        resolve.mode = uniforms.stochastic_mode;
        resolve.history_valid = frame.history_valid ? 1u : 0u;
        resolve.sample_count = frame.samples;
        resolve.orthographic = uniforms.camera_model == 1u;
        resolve.previous_orthographic = previous.camera_model == 1u;
        resolve.temporal_projection_valid = uniforms.camera_model <= 1u && previous.camera_model <= 1u;
        const float intrinsics[] = {uniforms.fx, uniforms.fy, uniforms.cx, uniforms.cy};
        const float previous_intrinsics[] = {previous.fx, previous.fy, previous.cx, previous.cy};
        std::memcpy(resolve.intrinsics, intrinsics, sizeof(intrinsics));
        std::memcpy(resolve.previous_intrinsics, previous_intrinsics, sizeof(previous_intrinsics));
        if (frame.history_valid) {
            const auto transform = viewWithoutSetback(previous) * glm::inverse(viewWithoutSetback(uniforms));
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 4; ++col)
                    resolve.view_to_previous[row][col] = transform[col][row];
        }
        resolve.origins[0] = static_cast<float>(uniforms.render_origin_x);
        resolve.origins[1] = static_cast<float>(uniforms.render_origin_y);
        resolve.origins[2] = static_cast<float>(previous.render_origin_x);
        resolve.origins[3] = static_cast<float>(previous.render_origin_y);
        resolve.background[0] = background.r;
        resolve.background[1] = background.g;
        resolve.background[2] = background.b;
        return &renderer_.executeStochasticResolve(resolve, buffers_, history.buffers, frame.output_color);
    }

    void VksplatViewportRenderer::commitStochasticFrame(const StochasticFrame& frame, const OutputSlot slot) {
        auto& history = stochastic_histories_[static_cast<std::size_t>(slot)];
        history.camera = frame.camera;
        history.model = frame.model;
        history.generation = frame.generation;
        history.samples = frame.samples + 1u;
        history.last_color = frame.output_color;
        history.valid = frame.enabled;
        if (frame.enabled)
            ++stochastic_sample_serial_;
    }

    void VksplatViewportRenderer::releaseStochasticBuffers() {
        for (auto& history : stochastic_histories_) {
            renderer_.destroyBuffer(history.buffers.ids_depths.deviceBuffer);
            renderer_.destroyBuffer(history.buffers.occlusion.deviceBuffer);
            renderer_.destroyBuffer(history.buffers.outgoing_weights.deviceBuffer);
            for (auto& color : history.buffers.color)
                renderer_.destroyBuffer(color.deviceBuffer);
            history = {};
        }
    }
} // namespace lfs::vis
