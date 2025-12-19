/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor_fwd.hpp"
#include "framebuffer_factory.hpp"
#include "gl_resources.hpp"
#include "shader.hpp"
#include "shader_manager.hpp"
#include <memory>

namespace lfs::rendering {
    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    // Depth parameters for screen quad rendering
    struct DepthParams {
        float near_plane = 0.1f;
        float far_plane = 100000.0f;
        bool orthographic = false;
        bool has_depth = false;
        bool depth_is_ndc = false;  // True if depth is already NDC (0-1), skip conversion
        GLuint external_depth_texture = 0;  // If set, use this texture instead of framebuffer's depth
    };

    class ScreenQuadRenderer {
    protected:
        VAO quadVAO_;
        VBO quadVBO_;
        DepthParams depth_params_;

    public:
        std::shared_ptr<FrameBuffer> framebuffer;

        explicit ScreenQuadRenderer(FrameBufferMode mode = FrameBufferMode::CPU);
        virtual ~ScreenQuadRenderer() = default;

        // Updated to return Result for consistency
        virtual Result<void> render(std::shared_ptr<Shader> shader) const;
        Result<void> render(ManagedShader& shader) const;

        virtual Result<void> uploadData(const unsigned char* image, int width_, int height_);
        Result<void> uploadFromCUDA(const Tensor& cuda_image, int width, int height);
        Result<void> uploadDepth(const float* depth_data, int width, int height);
        Result<void> uploadDepthFromCUDA(const Tensor& cuda_depth, int width, int height);

        // Set depth parameters for proper depth conversion in shader
        void setDepthParams(const DepthParams& params) { depth_params_ = params; }
        const DepthParams& getDepthParams() const { return depth_params_; }

        bool isInteropEnabled() const;

    protected:
        virtual GLuint getTextureID() const;
        GLuint getDepthTextureID() const;
    };
} // namespace lfs::rendering