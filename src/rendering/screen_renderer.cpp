/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "screen_renderer.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#ifdef CUDA_GL_INTEROP_ENABLED
#include "cuda_gl_interop.hpp"
#endif

namespace lfs::rendering {

    ScreenQuadRenderer::ScreenQuadRenderer(FrameBufferMode mode) {
        LOG_TIMER_TRACE("ScreenQuadRenderer::ScreenQuadRenderer");
        LOG_DEBUG("Creating ScreenQuadRenderer with mode: {}", mode == FrameBufferMode::CUDA_INTEROP ? "CUDA_INTEROP" : "CPU");

        framebuffer = createFrameBuffer(mode);

        float quadVertices[] = {
            // positions   // texCoords
            -1.0f, 1.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f,
            1.0f, -1.0f, 1.0f, 0.0f,

            -1.0f, 1.0f, 0.0f, 1.0f,
            1.0f, -1.0f, 1.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 1.0f};

        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            throw std::runtime_error(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            throw std::runtime_error(vbo_result.error());
        }
        quadVBO_ = std::move(*vbo_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        std::span<const float> vertices_span(quadVertices, sizeof(quadVertices) / sizeof(float));

        builder.attachVBO(quadVBO_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 4 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0})
            .setAttribute({.index = 1,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 4 * sizeof(float),
                           .offset = (void*)(2 * sizeof(float)),
                           .divisor = 0});

        quadVAO_ = builder.build();
        LOG_DEBUG("ScreenQuadRenderer initialized successfully");
    }

    Result<void> ScreenQuadRenderer::render(std::shared_ptr<Shader> shader) const {
        if (!shader) {
            LOG_ERROR("Shader is null");
            return std::unexpected("Shader is null");
        }

        LOG_TIMER_TRACE("ScreenQuadRenderer::render");

        shader->bind();

        VAOBinder vao_bind(quadVAO_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, getTextureID());

        try {
            shader->set_uniform("screenTexture", 0);
        } catch (const std::exception& e) {
            shader->unbind();
            LOG_ERROR("Failed to set uniform: {}", e.what());
            return std::unexpected(std::format("Failed to set uniform: {}", e.what()));
        }

        glDrawArrays(GL_TRIANGLES, 0, 6);

        shader->unbind();
        return {};
    }

    Result<void> ScreenQuadRenderer::render(ManagedShader& shader) const {
        LOG_TIMER_TRACE("ScreenQuadRenderer::render");

        ShaderScope s(shader);

        VAOBinder vao_bind(quadVAO_);

        // Bind color texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, getTextureID());
        if (auto result = shader.set("screenTexture", 0); !result) {
            return result;
        }

        // Bind depth texture and set depth parameters
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, getDepthTextureID());
        shader.set("depthTexture", 1);  // Ignore error - optional uniform
        shader.set("has_depth", depth_params_.has_depth);
        shader.set("near_plane", depth_params_.near_plane);
        shader.set("far_plane", depth_params_.far_plane);
        shader.set("orthographic", depth_params_.orthographic);
        shader.set("depth_is_ndc", depth_params_.depth_is_ndc);

        // Enable depth writing when we have depth data
        GLboolean prev_depth_mask;
        GLboolean prev_depth_test;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &prev_depth_mask);
        prev_depth_test = glIsEnabled(GL_DEPTH_TEST);

        if (depth_params_.has_depth) {
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_ALWAYS);  // Always write depth from CUDA render
        }

        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Restore depth state
        glDepthMask(prev_depth_mask);
        if (!prev_depth_test) {
            glDisable(GL_DEPTH_TEST);
        }
        glDepthFunc(GL_LESS);

        return {};
    }

    Result<void> ScreenQuadRenderer::uploadData(const unsigned char* image, int width_, int height_) {
        if (!framebuffer) {
            LOG_ERROR("Framebuffer not initialized");
            return std::unexpected("Framebuffer not initialized");
        }

        LOG_TRACE("Uploading image data: {}x{}", width_, height_);
        framebuffer->uploadImage(image, width_, height_);
        return {};
    }

    Result<void> ScreenQuadRenderer::uploadFromCUDA(const Tensor& cuda_image, int width, int height) {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            LOG_TRACE("Using CUDA interop for upload");
            return interop_fb->uploadFromCUDA(cuda_image);
        }
#endif
        // Fallback to CPU upload
        LOG_TRACE("Using CPU fallback for CUDA image upload");
        auto cpu_image = cuda_image;
        if (cpu_image.dtype() != lfs::core::DataType::UInt8) {
            cpu_image = (cpu_image.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
        }
        cpu_image = cpu_image.cpu().contiguous();
        return uploadData(cpu_image.ptr<unsigned char>(), width, height);
    }

    bool ScreenQuadRenderer::isInteropEnabled() const {
#ifdef CUDA_GL_INTEROP_ENABLED
        return std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer) != nullptr;
#else
        return false;
#endif
    }

    Result<void> ScreenQuadRenderer::uploadDepth(const float* depth_data, int width, int height) {
        if (!framebuffer) {
            LOG_ERROR("Framebuffer not initialized");
            return std::unexpected("Framebuffer not initialized");
        }

        LOG_TRACE("Uploading depth data: {}x{}", width, height);
        framebuffer->uploadDepth(depth_data, width, height);
        return {};
    }

    Result<void> ScreenQuadRenderer::uploadDepthFromCUDA(const Tensor& cuda_depth, int width, int height) {
        if (!framebuffer) {
            LOG_ERROR("Framebuffer not initialized");
            return std::unexpected("Framebuffer not initialized");
        }

        LOG_TRACE("Uploading depth from CUDA: {}x{}", width, height);

#ifdef CUDA_GL_INTEROP_ENABLED
        // Try interop path for direct CUDA→GL transfer
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            LOG_TRACE("Using CUDA-GL interop for depth upload");
            return interop_fb->uploadDepthFromCUDA(cuda_depth);
        }
#endif

        // Fallback: Copy depth tensor to CPU and upload
        auto depth_cpu = cuda_depth.cpu().contiguous();

        // Handle [1, H, W] shape
        if (depth_cpu.ndim() == 3 && depth_cpu.size(0) == 1) {
            depth_cpu = depth_cpu.squeeze(0);
        }

        if (depth_cpu.size(0) != static_cast<size_t>(height) ||
            depth_cpu.size(1) != static_cast<size_t>(width)) {
            LOG_ERROR("Depth tensor size mismatch: expected {}x{}, got {}x{}",
                      height, width, depth_cpu.size(0), depth_cpu.size(1));
            return std::unexpected("Depth tensor size mismatch");
        }

        framebuffer->uploadDepth(depth_cpu.ptr<float>(), width, height);
        return {};
    }

    GLuint ScreenQuadRenderer::getTextureID() const {
#ifdef CUDA_GL_INTEROP_ENABLED
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            return interop_fb->getInteropTexture();
        }
#endif
        return framebuffer->getFrameTexture();
    }

    GLuint ScreenQuadRenderer::getDepthTextureID() const {
        // Use external depth texture if provided (zero-copy from FBO)
        if (depth_params_.external_depth_texture != 0) {
            return depth_params_.external_depth_texture;
        }
#ifdef CUDA_GL_INTEROP_ENABLED
        // Use interop depth texture if available (direct CUDA→GL)
        if (auto interop_fb = std::dynamic_pointer_cast<InteropFrameBuffer>(framebuffer)) {
            return interop_fb->getDepthInteropTexture();
        }
#endif
        return framebuffer ? framebuffer->getDepthTexture() : 0;
    }

} // namespace lfs::rendering