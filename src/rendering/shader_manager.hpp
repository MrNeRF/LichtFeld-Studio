/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "shader.hpp"
#include <expected>
#include <format>
#include <memory>
#include <source_location>
#include <string>

namespace lfs::rendering {

    template <typename T>
    using Result = std::expected<T, std::string>;

    struct ShaderError {
        std::string message;
        std::source_location location;

        std::string what() const;
    };

    template <typename T>
    using ShaderResult = std::expected<T, ShaderError>;

    class ManagedShader {
        std::shared_ptr<Shader> shader_;
        std::string name_;

        friend class ShaderScope; // Allow ShaderScope to access private members

    public:
        ManagedShader() = default;
        ManagedShader(std::shared_ptr<Shader> shader, std::string_view name);

        Result<void> bind();
        Result<void> unbind();

        template <typename T>
        Result<void> set(std::string_view uniform, const T& value,
                         std::source_location loc = std::source_location::current()) {
            if (!shader_) {
                LOG_ERROR("Null shader '{}' when setting '{}'", name_, uniform);
                return std::unexpected(std::format("Shader '{}' is not initialized", name_));
            }

            try {
                shader_->set_uniform(std::string(uniform), value);
                return {};
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to set '{}' on shader '{}': {}", uniform, name_, e.what());
                return std::unexpected(std::format("Shader '{}': Failed to set '{}': {}",
                                                   name_, uniform, e.what()));
            } catch (...) {
                LOG_ERROR("Unknown exception setting '{}' on shader '{}'", uniform, name_);
                return std::unexpected(std::format("Shader '{}': Unknown error setting '{}'",
                                                   name_, uniform));
            }
        }

        Result<void> setInt(std::string_view uniform, int value,
                            std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Result<void> setFloat(std::string_view uniform, float value,
                              std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Result<void> setVec2(std::string_view uniform, const glm::vec2& value,
                             std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Result<void> setVec3(std::string_view uniform, const glm::vec3& value,
                             std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Result<void> setVec4(std::string_view uniform, const glm::vec4& value,
                             std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Result<void> setMat4(std::string_view uniform, const glm::mat4& value,
                             std::source_location loc = std::source_location::current()) {
            return set(uniform, value, loc);
        }

        Shader* operator->();
        const Shader* operator->() const;
        bool valid() const;

        // Get shader name for debugging
        const std::string& getName() const { return name_; }

        // Get program ID for debugging
        GLuint getProgramID() const {
            return shader_ ? shader_->programID() : 0;
        }
    };

    // RAII scope guard with enhanced debugging
    class ShaderScope {
        ManagedShader* shader_;
        bool bound_ = false;

    public:
        explicit ShaderScope(ManagedShader& shader);
        ~ShaderScope();

        // Deleted copy operations
        ShaderScope(const ShaderScope&) = delete;
        ShaderScope& operator=(const ShaderScope&) = delete;

        ShaderScope(ShaderScope&& other) noexcept
            : shader_(other.shader_),
              bound_(other.bound_) {
            other.bound_ = false;
        }

        ShaderScope& operator=(ShaderScope&& other) = delete; // Prevent move assignment

        ManagedShader* operator->();
        ManagedShader& operator*();

        bool isBound() const { return bound_; }
    };

    // Shader loader with source location tracking for better error reporting
    ShaderResult<ManagedShader> load_shader(
        std::string_view name,
        std::string_view vert_file,
        std::string_view frag_file,
        bool create_buffer = false,
        std::source_location loc = std::source_location::current());

    // Shader loader with geometry shader support
    ShaderResult<ManagedShader> load_shader_with_geometry(
        std::string_view name,
        std::string_view vert_file,
        std::string_view geom_file,
        std::string_view frag_file,
        bool create_buffer = false,
        std::source_location loc = std::source_location::current());

    // Helper function to log shader compilation info
    inline void logShaderInfo(GLuint shader, const std::string& type) {
        GLint compile_status;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);

        GLint info_log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);

        GLint shader_source_length;
        glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &shader_source_length);

        LOG_DEBUG("{} shader {} info: compile_status={}, log_length={}, source_length={}",
                  type, shader, compile_status, info_log_length, shader_source_length);
    }

    // Helper function to log program linking info
    inline void logProgramInfo(GLuint program) {
        GLint link_status;
        glGetProgramiv(program, GL_LINK_STATUS, &link_status);

        GLint info_log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);

        GLint attached_shaders;
        glGetProgramiv(program, GL_ATTACHED_SHADERS, &attached_shaders);

        GLint active_attributes;
        glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &active_attributes);

        GLint active_uniforms;
        glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &active_uniforms);

        LOG_DEBUG("Program {} info: link_status={}, log_length={}, attached_shaders={}, "
                  "active_attributes={}, active_uniforms={}",
                  program, link_status, info_log_length, attached_shaders,
                  active_attributes, active_uniforms);
    }

} // namespace lfs::rendering