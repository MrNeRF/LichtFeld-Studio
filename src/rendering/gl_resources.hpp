/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <expected>
#include <filesystem>
#include <glad/glad.h>
#include <source_location>
#include <span>
#include <utility>

namespace lfs::rendering {

    // Consistent error handling
    template <typename T>
    using Result = std::expected<T, std::string>;

    // RAII wrappers for OpenGL resources
    template <typename Deleter>
    class GLResource {
        GLuint id_ = 0;

    public:
        GLResource() = default;
        explicit GLResource(GLuint id) : id_(id) {}
        ~GLResource() {
            if (id_) {
                Deleter deleter;
                deleter(1, &id_);
                LOG_TRACE("Deleted GL resource: {}", id_);
            }
        }

        // Move only
        GLResource(GLResource&& other) noexcept : id_(std::exchange(other.id_, 0)) {}
        GLResource& operator=(GLResource&& other) noexcept {
            if (this != &other) {
                if (id_) {
                    Deleter deleter;
                    deleter(1, &id_);
                    LOG_TRACE("Deleted GL resource during move: {}", id_);
                }
                id_ = std::exchange(other.id_, 0);
            }
            return *this;
        }

        GLResource(const GLResource&) = delete;
        GLResource& operator=(const GLResource&) = delete;

        GLuint get() const { return id_; }
        GLuint* ptr() { return &id_; }
        operator GLuint() const { return id_; }
        explicit operator bool() const { return id_ != 0; }
    };

    // Deleter functors
    struct VAODeleter {
        void operator()(GLsizei n, const GLuint* arrays) const {
            glDeleteVertexArrays(n, arrays);
        }
    };

    struct BufferDeleter {
        void operator()(GLsizei n, const GLuint* buffers) const {
            glDeleteBuffers(n, buffers);
        }
    };

    struct TextureDeleter {
        void operator()(GLsizei n, const GLuint* textures) const {
            glDeleteTextures(n, textures);
        }
    };

    struct FramebufferDeleter {
        void operator()(GLsizei n, const GLuint* framebuffers) const {
            glDeleteFramebuffers(n, framebuffers);
        }
    };

    struct RenderbufferDeleter {
        void operator()(GLsizei n, const GLuint* renderbuffers) const {
            glDeleteRenderbuffers(n, renderbuffers);
        }
    };

    // Specific resource types
    using VAO = GLResource<VAODeleter>;
    using VBO = GLResource<BufferDeleter>;
    using EBO = GLResource<BufferDeleter>;
    using Texture = GLResource<TextureDeleter>;
    using FBO = GLResource<FramebufferDeleter>;
    using RBO = GLResource<RenderbufferDeleter>;

    // Factory functions with error handling
    inline Result<VAO> create_vao() {
        GLuint id;
        glGenVertexArrays(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            LOG_ERROR("Failed to create VAO");
            return std::unexpected("Failed to create VAO");
        }
        LOG_TRACE("Created VAO: {}", id);
        return VAO(id);
    }

    inline Result<VBO> create_vbo() {
        GLuint id;
        glGenBuffers(1, &id);
        if (glGetError() != GL_NO_ERROR || id == 0) {
            LOG_ERROR("Failed to create VBO");
            return std::unexpected("Failed to create VBO");
        }
        LOG_TRACE("Created VBO: {}", id);
        return VBO(id);
    }

    // Scoped binders
    template <GLenum Target>
    class BufferBinder {
        GLint prev_;
        static constexpr GLenum query = []() constexpr -> GLenum {
            if (Target == GL_ARRAY_BUFFER)
                return GL_ARRAY_BUFFER_BINDING;
            if (Target == GL_ELEMENT_ARRAY_BUFFER)
                return GL_ELEMENT_ARRAY_BUFFER_BINDING;
            if (Target == GL_UNIFORM_BUFFER)
                return GL_UNIFORM_BUFFER_BINDING;
            if (Target == GL_TEXTURE_BUFFER)
                return GL_TEXTURE_BUFFER_BINDING;
            if (Target == GL_COPY_READ_BUFFER)
                return GL_COPY_READ_BUFFER_BINDING;
            if (Target == GL_COPY_WRITE_BUFFER)
                return GL_COPY_WRITE_BUFFER_BINDING;
            if (Target == GL_PIXEL_PACK_BUFFER)
                return GL_PIXEL_PACK_BUFFER_BINDING;
            if (Target == GL_PIXEL_UNPACK_BUFFER)
                return GL_PIXEL_UNPACK_BUFFER_BINDING;
            if (Target == GL_TRANSFORM_FEEDBACK_BUFFER)
                return GL_TRANSFORM_FEEDBACK_BUFFER_BINDING;
            return 0;
        }();

    public:
        explicit BufferBinder(GLuint vbo) {
            glGetIntegerv(query, &prev_);
            glBindBuffer(Target, vbo);
            LOG_TRACE("Bound buffer {} to target 0x{:x}", vbo, Target);
        }
        ~BufferBinder() {
            glBindBuffer(Target, prev_);
            LOG_TRACE("Restored buffer {} to target 0x{:x}", prev_, Target);
        }
    };

    class VAOBinder {
        GLint prev_;

    public:
        explicit VAOBinder(GLuint vao) {
            glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_);
            glBindVertexArray(vao);
            LOG_TRACE("Bound VAO: {}", vao);
        }
        ~VAOBinder() {
            glBindVertexArray(prev_);
            LOG_TRACE("Restored VAO: {}", prev_);
        }
    };

    // Helper for vertex attribute setup
    struct VertexAttribute {
        GLuint index;
        GLint size;
        GLenum type;
        GLboolean normalized = GL_FALSE;
        GLsizei stride = 0;
        const void* offset = nullptr;
        GLuint divisor = 0; // For instancing

        void apply() const {
            glEnableVertexAttribArray(index);
            glVertexAttribPointer(index, size, type, normalized, stride, offset);
            if (divisor > 0)
                glVertexAttribDivisor(index, divisor);
            LOG_TRACE("Applied vertex attribute: index={}, size={}, stride={}, divisor={}",
                      index, size, stride, divisor);
        }
    };

    // Convenience function for buffer data upload
    template <typename T>
    void upload_buffer(GLenum target, const T* data, std::size_t count, GLenum usage = GL_STATIC_DRAW) {
        glBufferData(target, count * sizeof(T), data, usage);
        LOG_TRACE("Uploaded {} bytes to buffer (target=0x{:x}, usage=0x{:x})",
                  count * sizeof(T), target, usage);
    }

    // Overload for dynamic spans
    template <typename T>
    void upload_buffer(GLenum target, std::span<T> data, GLenum usage = GL_STATIC_DRAW) {
        glBufferData(target, data.size_bytes(), data.data(), usage);
        LOG_TRACE("Uploaded {} bytes to buffer (target=0x{:x}, usage=0x{:x})",
                  data.size_bytes(), target, usage);
    }

    // VAO Builder for proper EBO handling
    class VAOBuilder {
        VAO vao_;
        bool built_ = false;
        GLuint current_vbo_ = 0;

    public:
        explicit VAOBuilder(VAO&& vao) : vao_(std::move(vao)) {
            glBindVertexArray(vao_);
            LOG_TRACE("VAOBuilder: Started building VAO {}", vao_.get());
        }

        ~VAOBuilder() {
            if (!built_) {
                // If not built, unbind and clean up
                glBindVertexArray(0);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                LOG_WARN("VAOBuilder: Destroyed without building");
            }
        }

        // Move only
        VAOBuilder(VAOBuilder&& other) noexcept
            : vao_(std::move(other.vao_)),
              built_(other.built_),
              current_vbo_(other.current_vbo_) {
            other.built_ = true; // Prevent other from unbinding
        }

        VAOBuilder(const VAOBuilder&) = delete;
        VAOBuilder& operator=(const VAOBuilder&) = delete;
        VAOBuilder& operator=(VAOBuilder&&) = delete;

        // Attach and fill a VBO with data
        VAOBuilder& attachVBO(const VBO& vbo, std::span<const float> data, GLenum usage = GL_STATIC_DRAW) {
            current_vbo_ = vbo.get();
            glBindBuffer(GL_ARRAY_BUFFER, current_vbo_);
            glBufferData(GL_ARRAY_BUFFER, data.size_bytes(), data.data(), usage);
            LOG_TRACE("VAOBuilder: Attached VBO {} with {} bytes", current_vbo_, data.size_bytes());
            return *this;
        }

        // Attach VBO without data (for instance buffers that will be filled later)
        VAOBuilder& attachVBO(const VBO& vbo) {
            current_vbo_ = vbo.get();
            glBindBuffer(GL_ARRAY_BUFFER, current_vbo_);
            LOG_TRACE("VAOBuilder: Attached VBO {} without data", current_vbo_);
            return *this;
        }

        // Set attribute for the currently bound VBO
        VAOBuilder& setAttribute(const VertexAttribute& attr) {
            attr.apply();
            return *this;
        }

        // Set attribute with explicit VBO binding
        VAOBuilder& setAttribute(const VBO& vbo, const VertexAttribute& attr) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo.get());
            attr.apply();
            current_vbo_ = vbo.get();
            LOG_TRACE("VAOBuilder: Set attribute for VBO {}", vbo.get());
            return *this;
        }

        // Special handling for EBO - it stays bound to the VAO
        VAOBuilder& attachEBO(const EBO& ebo, std::span<const unsigned int> indices, GLenum usage = GL_STATIC_DRAW) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.get());
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size_bytes(), indices.data(), usage);
            // EBO binding is now part of VAO state - no unbind!
            LOG_TRACE("VAOBuilder: Attached EBO {} with {} indices", ebo.get(), indices.size());
            return *this;
        }

        // Attach EBO without data
        VAOBuilder& attachEBO(const EBO& ebo) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo.get());
            LOG_TRACE("VAOBuilder: Attached EBO {} without data", ebo.get());
            return *this;
        }

        // Build and return the VAO
        VAO build() {
            built_ = true;
            glBindVertexArray(0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            // Note: We do NOT unbind GL_ELEMENT_ARRAY_BUFFER here - it's part of VAO state
            LOG_DEBUG("VAOBuilder: Built VAO {}", vao_.get());
            return std::move(vao_);
        }

        // Get the VAO without consuming (for special cases)
        GLuint get() const { return vao_.get(); }
    };

} // namespace lfs::rendering