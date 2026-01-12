/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "shader_manager.hpp"
#include "core/logger.hpp"
#include "shader_paths.hpp"
#include <filesystem>

namespace lfs::rendering {

    std::string ShaderError::what() const {
        return std::format("[{}:{}] {}",
                           std::filesystem::path(location.file_name()).filename().string(),
                           location.line(), message);
    }

    ManagedShader::ManagedShader(std::shared_ptr<Shader> shader, std::string_view name)
        : shader_(shader),
          name_(name) {
    }

    Result<void> ManagedShader::bind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized", name_);
            return std::unexpected(std::format("Shader '{}' not initialized", name_));
        }

        try {
            shader_->bind(false);

            GLint current_program = 0;
            glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
            if (current_program != static_cast<GLint>(shader_->programID())) {
                LOG_ERROR("Failed to bind shader '{}': expected {}, got {}",
                          name_, shader_->programID(), current_program);
                return std::unexpected(std::format("Failed to bind shader '{}': program mismatch", name_));
            }
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Exception binding shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to bind shader '{}': {}", name_, e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception binding shader '{}'", name_);
            return std::unexpected(std::format("Failed to bind shader '{}': unknown exception", name_));
        }
    }

    Result<void> ManagedShader::unbind() {
        if (!shader_) {
            LOG_ERROR("Shader '{}' not initialized", name_);
            return std::unexpected(std::format("Shader '{}' not initialized", name_));
        }

        try {
            shader_->unbind(false);
            return {};
        } catch (const std::exception& e) {
            LOG_ERROR("Exception unbinding shader '{}': {}", name_, e.what());
            return std::unexpected(std::format("Failed to unbind shader '{}': {}", name_, e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception unbinding shader '{}'", name_);
            return std::unexpected(std::format("Failed to unbind shader '{}': unknown exception", name_));
        }
    }

    Shader* ManagedShader::operator->() {
        return shader_.get();
    }

    const Shader* ManagedShader::operator->() const {
        return shader_.get();
    }

    bool ManagedShader::valid() const {
        return shader_ != nullptr;
    }

    ShaderScope::ShaderScope(ManagedShader& shader) : shader_(&shader) {
        if (!shader_->valid()) {
            LOG_ERROR("ShaderScope: invalid shader '{}'", shader_->name_);
            bound_ = false;
            return;
        }

        if (auto result = shader_->bind(); result) {
            bound_ = true;
        } else {
            LOG_WARN("ShaderScope bind failed: {}", result.error());
            bound_ = false;
        }
    }

    ShaderScope::~ShaderScope() {
        if (bound_) {
            if (auto result = shader_->unbind(); !result) {
                LOG_ERROR("ShaderScope unbind failed: {}", result.error());
            }
        }
    }

    ManagedShader* ShaderScope::operator->() {
        return shader_;
    }

    ManagedShader& ShaderScope::operator*() {
        if (!shader_) {
            throw std::runtime_error("ShaderScope: null shader pointer");
        }
        return *shader_;
    }

    ShaderResult<ManagedShader> load_shader(
        std::string_view name,
        std::string_view vert_file,
        std::string_view frag_file,
        bool create_buffer,
        std::source_location loc) {

        try {
            const auto vert_path = getShaderPath(std::string(vert_file));
            const auto frag_path = getShaderPath(std::string(frag_file));

            if (!std::filesystem::exists(vert_path)) {
                LOG_ERROR("Vertex shader not found: {}", vert_path.string());
                return std::unexpected(ShaderError{
                    std::format("Vertex shader not found: {}", vert_path.string()), loc});
            }

            if (!std::filesystem::exists(frag_path)) {
                LOG_ERROR("Fragment shader not found: {}", frag_path.string());
                return std::unexpected(ShaderError{
                    std::format("Fragment shader not found: {}", frag_path.string()), loc});
            }

            if (std::filesystem::file_size(vert_path) == 0) {
                LOG_ERROR("Vertex shader empty: {}", vert_path.string());
                return std::unexpected(ShaderError{
                    std::format("Vertex shader empty: {}", vert_path.string()), loc});
            }

            if (std::filesystem::file_size(frag_path) == 0) {
                LOG_ERROR("Fragment shader empty: {}", frag_path.string());
                return std::unexpected(ShaderError{
                    std::format("Fragment shader empty: {}", frag_path.string()), loc});
            }

            while (glGetError() != GL_NO_ERROR) {}

            auto shader = std::make_shared<Shader>(
                vert_path.string().c_str(),
                frag_path.string().c_str(),
                create_buffer);

            const GLuint program_id = shader->programID();
            if (program_id == 0) {
                LOG_ERROR("Shader '{}' has invalid program ID", name);
                return std::unexpected(ShaderError{
                    std::format("Shader '{}' has invalid program ID", name), loc});
            }

            LOG_INFO("Loaded shader '{}' (program {})", name, program_id);
            return ManagedShader(shader, name);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load shader '{}': {}", name, e.what());
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': {}", name, e.what()), loc});
        } catch (...) {
            LOG_ERROR("Failed to load shader '{}': unknown exception", name);
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': unknown exception", name), loc});
        }
    }

    ShaderResult<ManagedShader> load_shader_with_geometry(
        std::string_view name,
        std::string_view vert_file,
        std::string_view geom_file,
        std::string_view frag_file,
        bool create_buffer,
        std::source_location loc) {

        try {
            const auto vert_path = getShaderPath(std::string(vert_file));
            const auto geom_path = getShaderPath(std::string(geom_file));
            const auto frag_path = getShaderPath(std::string(frag_file));

            if (!std::filesystem::exists(vert_path)) {
                LOG_ERROR("Vertex shader not found: {}", vert_path.string());
                return std::unexpected(ShaderError{
                    std::format("Vertex shader not found: {}", vert_path.string()), loc});
            }

            if (!std::filesystem::exists(geom_path)) {
                LOG_ERROR("Geometry shader not found: {}", geom_path.string());
                return std::unexpected(ShaderError{
                    std::format("Geometry shader not found: {}", geom_path.string()), loc});
            }

            if (!std::filesystem::exists(frag_path)) {
                LOG_ERROR("Fragment shader not found: {}", frag_path.string());
                return std::unexpected(ShaderError{
                    std::format("Fragment shader not found: {}", frag_path.string()), loc});
            }

            while (glGetError() != GL_NO_ERROR) {}

            auto shader = std::make_shared<Shader>(
                vert_path.string().c_str(),
                frag_path.string().c_str(),
                geom_path.string().c_str(),
                create_buffer);

            const GLuint program_id = shader->programID();
            if (program_id == 0) {
                LOG_ERROR("Shader '{}' has invalid program ID", name);
                return std::unexpected(ShaderError{
                    std::format("Shader '{}' has invalid program ID", name), loc});
            }

            LOG_INFO("Loaded shader '{}' with geometry (program {})", name, program_id);
            return ManagedShader(shader, name);

        } catch (const std::exception& e) {
            LOG_ERROR("Failed to load shader '{}': {}", name, e.what());
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': {}", name, e.what()), loc});
        } catch (...) {
            LOG_ERROR("Failed to load shader '{}': unknown exception", name);
            return std::unexpected(ShaderError{
                std::format("Failed to load shader '{}': unknown exception", name), loc});
        }
    }

} // namespace lfs::rendering