/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framebuffer.hpp"
#include "gl_resources.hpp"
#include <exception>
#include <format>
#include <glm/vec2.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lfs::rendering {

    class HighPrecisionRenderTarget {
    public:
        Result<void> ensureSize(const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            if (!fbo_) {
                GLuint fbo_id = 0;
                glGenFramebuffers(1, &fbo_id);
                fbo_ = FBO(fbo_id);
                if (!fbo_) {
                    return std::unexpected("Failed to create framebuffer");
                }
            }

            if (!color_texture_) {
                GLuint color_texture_id = 0;
                glGenTextures(1, &color_texture_id);
                color_texture_ = Texture(color_texture_id);
                if (!color_texture_) {
                    return std::unexpected("Failed to create color texture");
                }
            }

            if (!depth_texture_) {
                GLuint depth_texture_id = 0;
                glGenTextures(1, &depth_texture_id);
                depth_texture_ = Texture(depth_texture_id);
                if (!depth_texture_) {
                    return std::unexpected("Failed to create depth texture");
                }
            }

            glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());

            glBindTexture(GL_TEXTURE_2D, color_texture_.get());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, size.x, size.y, 0, GL_RGB, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_.get(), 0);

            glBindTexture(GL_TEXTURE_2D, depth_texture_.get());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, size.x, size.y, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_.get(), 0);

            const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                return std::unexpected(std::format("Framebuffer incomplete: 0x{:x}", status));
            }

            glBindTexture(GL_TEXTURE_2D, 0);
            width_ = size.x;
            height_ = size.y;
            return {};
        }

        [[nodiscard]] GLuint framebuffer() const { return fbo_.get(); }
        [[nodiscard]] GLuint colorTexture() const { return color_texture_.get(); }
        [[nodiscard]] GLuint depthTexture() const { return depth_texture_.get(); }
        [[nodiscard]] int width() const { return width_; }
        [[nodiscard]] int height() const { return height_; }

    private:
        FBO fbo_;
        Texture color_texture_;
        Texture depth_texture_;
        int width_ = 0;
        int height_ = 0;
    };

    class DisplayRenderTarget {
    public:
        Result<void> ensureSize(const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            if (!fbo_) {
                GLuint fbo_id = 0;
                glGenFramebuffers(1, &fbo_id);
                fbo_ = FBO(fbo_id);
                if (!fbo_) {
                    return std::unexpected("Failed to create framebuffer");
                }
            }

            if (!color_texture_) {
                GLuint color_texture_id = 0;
                glGenTextures(1, &color_texture_id);
                color_texture_ = Texture(color_texture_id);
                if (!color_texture_) {
                    return std::unexpected("Failed to create color texture");
                }
            }

            if (!depth_texture_) {
                GLuint depth_texture_id = 0;
                glGenTextures(1, &depth_texture_id);
                depth_texture_ = Texture(depth_texture_id);
                if (!depth_texture_) {
                    return std::unexpected("Failed to create depth texture");
                }
            }

            glBindFramebuffer(GL_FRAMEBUFFER, fbo_.get());

            glBindTexture(GL_TEXTURE_2D, color_texture_.get());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_.get(), 0);

            glBindTexture(GL_TEXTURE_2D, depth_texture_.get());
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, size.x, size.y, 0,
                         GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture_.get(), 0);

            const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
            if (status != GL_FRAMEBUFFER_COMPLETE) {
                return std::unexpected(std::format("Framebuffer incomplete: 0x{:x}", status));
            }

            glBindTexture(GL_TEXTURE_2D, 0);
            width_ = size.x;
            height_ = size.y;
            return {};
        }

        [[nodiscard]] GLuint framebuffer() const { return fbo_.get(); }
        [[nodiscard]] GLuint colorTexture() const { return color_texture_.get(); }
        [[nodiscard]] GLuint depthTexture() const { return depth_texture_.get(); }
        [[nodiscard]] int width() const { return width_; }
        [[nodiscard]] int height() const { return height_; }

    private:
        FBO fbo_;
        Texture color_texture_;
        Texture depth_texture_;
        int width_ = 0;
        int height_ = 0;
    };

    class RenderTargetPool {
    public:
        Result<std::shared_ptr<FrameBuffer>> acquire(std::string_view key, const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            auto& target = targets_[std::string(key)];
            if (!target) {
                try {
                    target = std::make_shared<FrameBuffer>();
                } catch (const std::exception& e) {
                    return std::unexpected(std::string("Failed to create render target: ") + e.what());
                }
            }

            if (target->getWidth() != size.x || target->getHeight() != size.y) {
                target->resize(size.x, size.y);
            }

            return target;
        }

        Result<std::shared_ptr<HighPrecisionRenderTarget>> acquireHighPrecision(
            std::string_view key,
            const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            auto& target = high_precision_targets_[std::string(key)];
            if (!target) {
                target = std::make_shared<HighPrecisionRenderTarget>();
            }

            if (auto result = target->ensureSize(size); !result) {
                return std::unexpected(result.error());
            }

            return target;
        }

        Result<std::shared_ptr<DisplayRenderTarget>> acquireDisplay(
            std::string_view key,
            const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            auto& target = display_targets_[std::string(key)];
            if (!target) {
                target = std::make_shared<DisplayRenderTarget>();
            }

            if (auto result = target->ensureSize(size); !result) {
                return std::unexpected(result.error());
            }

            return target;
        }

        void clear() {
            targets_.clear();
            display_targets_.clear();
            high_precision_targets_.clear();
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<FrameBuffer>> targets_;
        std::unordered_map<std::string, std::shared_ptr<DisplayRenderTarget>> display_targets_;
        std::unordered_map<std::string, std::shared_ptr<HighPrecisionRenderTarget>> high_precision_targets_;
    };

} // namespace lfs::rendering
