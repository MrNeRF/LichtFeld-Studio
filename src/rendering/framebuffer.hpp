/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// clang-format off
// CRITICAL: GLAD must be included before other OpenGL usage
#include <glad/glad.h>
// clang-format on

#include "core/logger.hpp"
#include <expected>
#include <stdexcept>
#include <string>

namespace lfs::rendering {

    // Consistent error handling
    template <typename T>
    using Result = std::expected<T, std::string>;

    class FrameBuffer {

    private:
        GLuint fbo;
        GLuint texture;      // color texture
        GLuint depthTexture; // depth texture

    protected: // Changed from private to protected so derived classes can access
        int width = 1;
        int height = 1;

    public:
        FrameBuffer() {
            LOG_DEBUG("Creating FrameBuffer");
            auto result = init(width, height);
            if (!result) {
                LOG_ERROR("Failed to initialize FrameBuffer: {}", result.error());
                throw std::runtime_error(result.error());
            }
        }

        virtual ~FrameBuffer() { // Made virtual for proper inheritance
            LOG_TRACE("Destroying FrameBuffer");
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &texture);
            glDeleteTextures(1, &depthTexture);
        }

        Result<void> init(int w, int h) {
            LOG_TIMER_TRACE("FrameBuffer::init");
            LOG_DEBUG("Initializing FrameBuffer with size {}x{}", w, h);

            width = w;
            height = h;

            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            // --- Color texture ---
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, texture, 0);

            // Depth texture (R32F for CUDA depth storage, not FBO-attached)
            glGenTextures(1, &depthTexture);
            glBindTexture(GL_TEXTURE_2D, depthTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                LOG_ERROR("Framebuffer is not complete");
                return std::unexpected("Framebuffer is not complete");
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            LOG_DEBUG("FrameBuffer initialized successfully");
            return {};
        }

        virtual void resize(int newWidth, int newHeight) {
            LOG_TRACE("Resizing FrameBuffer from {}x{} to {}x{}", width, height, newWidth, newHeight);

            width = newWidth;
            height = newHeight;

            // IMPORTANT: Bind the framebuffer before resizing its attachments
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            // Resize color texture
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            // Re-attach color texture to framebuffer
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, texture, 0);

            // Resize depth texture
            glBindTexture(GL_TEXTURE_2D, depthTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            // Check framebuffer completeness after resize
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                LOG_ERROR("Framebuffer incomplete after resize to {}x{}", width, height);
            }

            // Unbind framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        void uploadImage(const unsigned char* data, int width_, int height_) {
            if (width != width_ || height != height_) {
                resize(width_, height_);
            }

            LOG_TIMER_TRACE("FrameBuffer::uploadImage");
            LOG_TRACE("Uploading image data: {}x{}", width_, height_);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGB, GL_UNSIGNED_BYTE, data);
        }

        void uploadDepth(const float* depth_data, int width_, int height_) {
            if (width != width_ || height != height_) {
                resize(width_, height_);
            }

            LOG_TIMER_TRACE("FrameBuffer::uploadDepth");
            LOG_TRACE("Uploading depth data: {}x{}", width_, height_);

            glBindTexture(GL_TEXTURE_2D, depthTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RED, GL_FLOAT, depth_data);
        }

        void uploadImageAndDepth(const unsigned char* rgb_data,
                                 const float* depth_data,
                                 int new_width,
                                 int new_height) {
            if (width != new_width || height != new_height) {
                resize(new_width, new_height);
            }

            LOG_TIMER_TRACE("FrameBuffer::uploadImageAndDepth");
            LOG_TRACE("Uploading image and depth data: {}x{}", new_width, new_height);

            glBindTexture(GL_TEXTURE_2D, texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RGB, GL_UNSIGNED_BYTE, rgb_data);
            glBindTexture(GL_TEXTURE_2D, depthTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                            GL_RED, GL_FLOAT, depth_data);
        }

        void bind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        }

        void unbind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        GLuint getFrameTexture() const { return texture; }
        GLuint getDepthTexture() const { return depthTexture; }
        int getWidth() const { return width; }
        int getHeight() const { return height; }
    };
} // namespace lfs::rendering