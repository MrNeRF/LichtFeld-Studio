/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// clang-format off
// GLAD must be included before GLFW
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include "core/logger.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::rendering {

    // Helper function to get OpenGL error string
    inline std::string getGLErrorString(GLenum error) {
        switch (error) {
        case GL_NO_ERROR: return "GL_NO_ERROR";
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
        case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        default: return std::format("Unknown GL error: 0x{:x}", error);
        }
    }

// Macro for checking OpenGL errors with detailed context
#define CHECK_GL_ERROR(operation)                                                 \
    do {                                                                          \
        GLenum err = glGetError();                                                \
        if (err != GL_NO_ERROR) {                                                 \
            LOG_ERROR("OpenGL error after {}: {} (0x{:x}) at {}:{}",              \
                      operation, getGLErrorString(err), err, __FILE__, __LINE__); \
        }                                                                         \
    } while (0)

    template <typename E>
    inline GLenum is_type_integral() {
        return GL_FALSE;
    }

    template <>
    inline GLenum is_type_integral<GLbyte>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLshort>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLint>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLubyte>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLushort>() {
        return GL_TRUE;
    }

    template <>
    inline GLenum is_type_integral<GLuint>() {
        return GL_TRUE;
    }

    template <typename E>
    inline GLenum get_type_enum() {
        LOG_ERROR("Error getting type enum: unsupported type at {}:{}", __FILE__, __LINE__);
        throw std::runtime_error(std::format("Error getting type enum: unsupported type at {}:{}",
                                             __FILE__, __LINE__));
    }

    template <>
    inline GLenum get_type_enum<GLbyte>() {
        return GL_BYTE;
    }

    template <>
    inline GLenum get_type_enum<GLshort>() {
        return GL_SHORT;
    }

    template <>
    inline GLenum get_type_enum<GLint>() {
        return GL_INT;
    }

    template <>
    inline GLenum get_type_enum<GLubyte>() {
        return GL_UNSIGNED_BYTE;
    }

    template <>
    inline GLenum get_type_enum<GLushort>() {
        return GL_UNSIGNED_SHORT;
    }

    template <>
    inline GLenum get_type_enum<GLuint>() {
        return GL_UNSIGNED_INT;
    }

    template <>
    inline GLenum get_type_enum<GLfloat>() {
        return GL_FLOAT;
    }

    class Shader {
    public:
        Shader(const char* vshader_path, const char* fshader_path, bool create_buffer = true)
            : Shader(vshader_path, fshader_path, nullptr, create_buffer) {}

        Shader(const char* vshader_path, const char* fshader_path, const char* gshader_path, bool create_buffer)
            : vshader_path_(vshader_path),
              fshader_path_(fshader_path),
              gshader_path_(gshader_path ? gshader_path : "") {
            LOG_TIMER_TRACE("Shader::Shader");
            LOG_DEBUG("Creating shader with vertex: {}", vshader_path);
            LOG_DEBUG("Creating shader with fragment: {}", fshader_path);
            if (gshader_path && gshader_path[0] != '\0') {
                LOG_DEBUG("Creating shader with geometry: {}", gshader_path);
            }

            // Clear any existing GL errors before we start
            while (glGetError() != GL_NO_ERROR) {}

            GLint status;

            std::string vshader_source = readShaderSourceFromFile(vshader_path);
            std::string fshader_source = readShaderSourceFromFile(fshader_path);

            if (vshader_source.empty()) {
                LOG_ERROR("Vertex shader source is empty for file: {}", vshader_path);
                throw std::runtime_error(std::format("ERROR: Vertex shader source is empty for file: {}", vshader_path));
            }
            if (fshader_source.empty()) {
                LOG_ERROR("Fragment shader source is empty for file: {}", fshader_path);
                throw std::runtime_error(std::format("ERROR: Fragment shader source is empty for file: {}", fshader_path));
            }

            constexpr GLsizei MAX_INFO_LOG_LENGTH = 2000;
            GLsizei info_log_length;
            GLchar info_log[MAX_INFO_LOG_LENGTH];
            GLint compilation_status;
            auto check_comp_status = [&](GLuint shader, const char* shader_type, const char* shader_file) {
                glGetShaderiv(shader, GL_COMPILE_STATUS, &compilation_status);
                if (compilation_status == GL_TRUE) {
                    LOG_TRACE("{} shader compiled successfully: {}", shader_type, shader_file);
                    return;
                }
                glGetShaderInfoLog(shader, MAX_INFO_LOG_LENGTH, &info_log_length, info_log);

                // Extract line numbers from error messages if possible
                std::string error_details(info_log);
                LOG_ERROR("{} shader compilation error in file: {}", shader_type, shader_file);
                LOG_ERROR("Compilation error details:\n{}", error_details);

                // Try to show the problematic lines from source
                std::istringstream source_stream(shader_type == std::string("Vertex") ? vshader_source : fshader_source);
                std::string line;
                int line_num = 1;
                LOG_ERROR("Shader source preview:");
                while (std::getline(source_stream, line) && line_num <= 10) {
                    LOG_ERROR("  {:3}: {}", line_num++, line);
                }

                throw std::runtime_error(std::format("{} shader compilation error in {}: {}",
                                                     shader_type, shader_file, info_log));
            };

            // Create and compile vertex shader
            vshader = glCreateShader(GL_VERTEX_SHADER);
            if (vshader == 0) {
                GLenum err = glGetError();
                LOG_ERROR("Failed to create vertex shader object: {}", getGLErrorString(err));
                throw std::runtime_error(std::format("Failed to create vertex shader object: {}", getGLErrorString(err)));
            }

            const char* vshader_code = vshader_source.c_str();
            glShaderSource(vshader, 1, &vshader_code, nullptr);
            CHECK_GL_ERROR("glShaderSource (vertex)");

            glCompileShader(vshader);
            CHECK_GL_ERROR("glCompileShader (vertex)");
            check_comp_status(vshader, "Vertex", vshader_path);

            // Create and compile fragment shader
            fshader = glCreateShader(GL_FRAGMENT_SHADER);
            if (fshader == 0) {
                GLenum err = glGetError();
                LOG_ERROR("Failed to create fragment shader object: {}", getGLErrorString(err));
                glDeleteShader(vshader); // Clean up vertex shader
                throw std::runtime_error(std::format("Failed to create fragment shader object: {}", getGLErrorString(err)));
            }

            const char* fshader_code = fshader_source.c_str();
            glShaderSource(fshader, 1, &fshader_code, nullptr);
            CHECK_GL_ERROR("glShaderSource (fragment)");

            glCompileShader(fshader);
            CHECK_GL_ERROR("glCompileShader (fragment)");
            check_comp_status(fshader, "Fragment", fshader_path);

            // Create and compile geometry shader (optional)
            if (gshader_path && gshader_path[0] != '\0') {
                std::string gshader_source = readShaderSourceFromFile(gshader_path);
                if (gshader_source.empty()) {
                    LOG_ERROR("Geometry shader source is empty for file: {}", gshader_path);
                    glDeleteShader(vshader);
                    glDeleteShader(fshader);
                    throw std::runtime_error(std::format("ERROR: Geometry shader source is empty for file: {}", gshader_path));
                }

                gshader = glCreateShader(GL_GEOMETRY_SHADER);
                if (gshader == 0) {
                    GLenum err = glGetError();
                    LOG_ERROR("Failed to create geometry shader object: {}", getGLErrorString(err));
                    glDeleteShader(vshader);
                    glDeleteShader(fshader);
                    throw std::runtime_error(std::format("Failed to create geometry shader object: {}", getGLErrorString(err)));
                }

                const char* gshader_code = gshader_source.c_str();
                glShaderSource(gshader, 1, &gshader_code, nullptr);
                CHECK_GL_ERROR("glShaderSource (geometry)");

                glCompileShader(gshader);
                CHECK_GL_ERROR("glCompileShader (geometry)");
                check_comp_status(gshader, "Geometry", gshader_path);
            }

            // Create and link program
            program = glCreateProgram();
            if (program == 0) {
                GLenum err = glGetError();
                LOG_ERROR("Failed to create shader program object: {}", getGLErrorString(err));
                glDeleteShader(vshader);
                glDeleteShader(fshader);
                throw std::runtime_error(std::format("Failed to create shader program object: {}", getGLErrorString(err)));
            }

            glAttachShader(program, vshader);
            CHECK_GL_ERROR("glAttachShader (vertex)");

            glAttachShader(program, fshader);
            CHECK_GL_ERROR("glAttachShader (fragment)");

            if (gshader != 0) {
                glAttachShader(program, gshader);
                CHECK_GL_ERROR("glAttachShader (geometry)");
            }

            glLinkProgram(program);
            CHECK_GL_ERROR("glLinkProgram");

            glGetProgramiv(program, GL_LINK_STATUS, &status);
            if (status != GL_TRUE) {
                glGetProgramInfoLog(program, MAX_INFO_LOG_LENGTH, nullptr, info_log);
                LOG_ERROR("Shader link error for program using:");
                LOG_ERROR("  Vertex shader: {}", vshader_path);
                LOG_ERROR("  Fragment shader: {}", fshader_path);
                LOG_ERROR("Link error details:\n{}", info_log);

                // Clean up
                glDeleteShader(vshader);
                glDeleteShader(fshader);
                glDeleteProgram(program);

                throw std::runtime_error(std::format("Shader link error:\n{}", info_log));
            }

            // Validate program
            glValidateProgram(program);
            glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
            if (status != GL_TRUE) {
                glGetProgramInfoLog(program, MAX_INFO_LOG_LENGTH, nullptr, info_log);
                LOG_WARN("Shader validation warning for program: {}", info_log);
            }

            LOG_DEBUG("Shader program {} linked and validated successfully", program);

            if (create_buffer) {
                glGenBuffers(1, &index_buffer);
                CHECK_GL_ERROR("glGenBuffers (index)");

                glGenVertexArrays(1, &vertex_array);
                CHECK_GL_ERROR("glGenVertexArrays");

                LOG_TRACE("Created index buffer {} and vertex array {}", index_buffer, vertex_array);
            }

            // Log shader program info
            logProgramInfo();
        }

        ~Shader() {
            LOG_TRACE("Destroying shader program {} (vertex: {}, fragment: {})",
                      program, vshader_path_, fshader_path_);

            for (auto [attrib, buffer] : attribute_buffers) {
                glDeleteBuffers(1, &buffer);
                LOG_TRACE("Deleted attribute buffer {} for attribute {}", buffer, attrib);
            }

            if (vertex_array != 0) {
                glDeleteVertexArrays(1, &vertex_array);
                LOG_TRACE("Deleted vertex array {}", vertex_array);
            }

            if (index_buffer != 0) {
                glDeleteBuffers(1, &index_buffer);
                LOG_TRACE("Deleted index buffer {}", index_buffer);
            }

            glDetachShader(program, fshader);
            glDetachShader(program, vshader);
            if (gshader != 0) {
                glDetachShader(program, gshader);
                glDeleteShader(gshader);
            }
            glDeleteProgram(program);
            glDeleteShader(fshader);
            glDeleteShader(vshader);
        }

        void bind(bool use_buffer = true) {
            // Clear any pre-existing OpenGL errors before binding
            while (glGetError() != GL_NO_ERROR)
                ;

            if (use_buffer) {
                glBindVertexArray(vertex_array);
                CHECK_GL_ERROR("glBindVertexArray");

                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
                CHECK_GL_ERROR("glBindBuffer (element)");
            }
            glUseProgram(program);
            CHECK_GL_ERROR("glUseProgram");

            LOG_TRACE("Bound shader program {}", program);
        }

        void unbind(bool use_buffer = true) {
            if (use_buffer) {
                glBindVertexArray(0);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            }
            glUseProgram(0);
            LOG_TRACE("Unbound shader program {}", program);
        }

        GLuint programID() const {
            return program;
        }

        void set_uniform(const std::string& name, const size_t& value) {
            GLint uni = uniform(name);
            glUniform1i(uni, static_cast<GLint>(value));
            CHECK_GL_ERROR(std::format("glUniform1i({}, {})", name, value));
            LOG_TRACE("Set uniform '{}' to {}", name, value);
        }

        void set_uniform(const std::string& name, const int& value) {
            GLint uni = uniform(name);
            glUniform1i(uni, value);
            CHECK_GL_ERROR(std::format("glUniform1i({}, {})", name, value));
            LOG_TRACE("Set uniform '{}' to {}", name, value);
        }

        void set_uniform(const std::string& name, const float& value) {
            GLint uni = uniform(name);
            glUniform1f(uni, value);
            CHECK_GL_ERROR(std::format("glUniform1f({}, {})", name, value));
            LOG_TRACE("Set uniform '{}' to {}", name, value);
        }

        void set_uniform(const std::string& name, const glm::vec2& vector) {
            GLint uni = uniform(name);
            glUniform2fv(uni, 1, &vector[0]);
            CHECK_GL_ERROR(std::format("glUniform2fv({})", name));
            LOG_TRACE("Set uniform '{}' to vec2({}, {})", name, vector.x, vector.y);
        }

        void set_uniform(const std::string& name, const glm::vec3& vector) {
            GLint uni = uniform(name);
            glUniform3fv(uni, 1, &vector[0]);
            CHECK_GL_ERROR(std::format("glUniform3fv({})", name));
            LOG_TRACE("Set uniform '{}' to vec3({}, {}, {})", name, vector.x, vector.y, vector.z);
        }

        void set_uniform(const std::string& name, const glm::vec4& vector) {
            GLint uni = uniform(name);
            glUniform4fv(uni, 1, &vector[0]);
            CHECK_GL_ERROR(std::format("glUniform4fv({})", name));
            LOG_TRACE("Set uniform '{}' to vec4({}, {}, {}, {})", name, vector.x, vector.y, vector.z, vector.w);
        }

        void set_uniform(const std::string& name, const glm::mat4& matrix) {
            GLint uni = uniform(name);
            glUniformMatrix4fv(uni, 1, GL_FALSE, &matrix[0][0]);
            CHECK_GL_ERROR(std::format("glUniformMatrix4fv({})", name));
            LOG_TRACE("Set uniform '{}' to mat4", name);
        }

        // texture
        void set_uniform(const std::string& name) {
            GLint uni = uniform(name);
            glUniform1i(uni, 0);
            CHECK_GL_ERROR(std::format("glUniform1i({}, 0) [texture]", name));
            LOG_TRACE("Set uniform '{}' to texture unit 0", name);
        }

        template <typename E, int N>
        void set_attribute(const std::string& name,
                           const std::vector<glm::vec<N, E, glm::defaultp>>& data) {
            LOG_TIMER_TRACE("Shader::set_attribute");

            if (data.empty()) {
                LOG_WARN("Attempting to set attribute '{}' with empty data", name);
                return;
            }

            GLint attrib = attribute(name);
            if (attribute_buffers.count(attrib) == 0) {
                GLuint buffer;
                glGenBuffers(1, &buffer);
                CHECK_GL_ERROR(std::format("glGenBuffers for attribute '{}'", name));

                attribute_buffers[attrib] = buffer;
                LOG_TRACE("Created attribute buffer {} for '{}'", buffer, name);
            }

            GLuint buffer = attribute_buffers.at(attrib);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            CHECK_GL_ERROR(std::format("glBindBuffer for attribute '{}'", name));

            size_t data_size = sizeof(E) * N * data.size();
            glBufferData(GL_ARRAY_BUFFER, data_size, data.data(), GL_DYNAMIC_DRAW);
            CHECK_GL_ERROR(std::format("glBufferData ({} bytes) for attribute '{}'", data_size, name));

            glEnableVertexAttribArray(attrib);
            CHECK_GL_ERROR(std::format("glEnableVertexAttribArray for attribute '{}'", name));

            GLenum type_enum = get_type_enum<E>();
            glVertexAttribPointer(attrib, N, type_enum, is_type_integral<E>(), 0, nullptr);
            CHECK_GL_ERROR(std::format("glVertexAttribPointer for attribute '{}'", name));

            glBindBuffer(GL_ARRAY_BUFFER, 0);
            LOG_TRACE("Set attribute '{}' (location {}) with {} elements ({} bytes)",
                      name, attrib, data.size(), data_size);
        }

        void set_indices(const std::vector<unsigned int>& indices) {
            LOG_TIMER_TRACE("Shader::set_indices");

            if (indices.empty()) {
                LOG_WARN("Attempting to set empty indices");
                return;
            }

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
            CHECK_GL_ERROR("glBindBuffer (element array)");

            size_t data_size = sizeof(unsigned int) * indices.size();
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, data_size, &indices[0], GL_DYNAMIC_DRAW);
            CHECK_GL_ERROR(std::format("glBufferData ({} bytes) for indices", data_size));

            LOG_TRACE("Set {} indices ({} bytes)", indices.size(), data_size);
        }

        void draw(GLenum mode, GLuint start, GLuint count) {
            glDrawArrays(mode, start, count);
            CHECK_GL_ERROR(std::format("glDrawArrays(mode={}, start={}, count={})", mode, start, count));
            LOG_TRACE("Drew {} vertices starting at {} with mode {}", count, start, mode);
        }

        void draw_indexed(GLenum mode, GLuint start, GLuint count) {
            glDrawElements(mode, count, GL_UNSIGNED_INT, (const void*)(start * sizeof(GLuint)));
            CHECK_GL_ERROR(std::format("glDrawElements(mode={}, count={}, start={})", mode, count, start));
            LOG_TRACE("Drew {} indexed elements starting at {} with mode {}", count, start, mode);
        }

    private:
        std::string readShaderSourceFromFile(const std::string& filePath) {
            LOG_TIMER_TRACE("Shader::readShaderSourceFromFile");
            LOG_TRACE("Reading shader source from: {}", filePath);

            // Check if file exists
            if (!std::filesystem::exists(filePath)) {
                LOG_ERROR("Shader file does not exist: {}", filePath);
                throw std::runtime_error(std::format("Shader file does not exist: {}", filePath));
            }

            // Check if it's a regular file
            if (!std::filesystem::is_regular_file(filePath)) {
                LOG_ERROR("Shader path is not a regular file: {}", filePath);
                throw std::runtime_error(std::format("Shader path is not a regular file: {}", filePath));
            }

            std::ifstream file(filePath);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open shader file: {} (error: {})", filePath, strerror(errno));
                throw std::runtime_error(std::format("Failed to open shader file: {} (error: {})",
                                                     filePath, strerror(errno)));
            }

            std::stringstream buffer;
            buffer << file.rdbuf();

            if (file.bad()) {
                LOG_ERROR("Error reading shader file: {}", filePath);
                throw std::runtime_error(std::format("Error reading shader file: {}", filePath));
            }

            std::string source = buffer.str();
            LOG_TRACE("Read {} bytes from shader file {}", source.size(), filePath);

            // Validate shader source has some content
            if (source.find_first_not_of(" \t\n\r") == std::string::npos) {
                LOG_WARN("Shader file contains only whitespace: {}", filePath);
            }

            return source;
        }

        GLint uniform(const std::string& name) {
            if (uniforms.count(name) == 0) {
                GLint location = glGetUniformLocation(program, name.c_str());
                if (location == -1) {
                    LOG_ERROR("Cannot find uniform '{}' in shader program {} (vertex: {}, fragment: {})",
                              name, program, vshader_path_, fshader_path_);

                    // Log all available uniforms
                    logAvailableUniforms();

                    throw std::runtime_error(std::format("Error: cannot find uniform '{}' in shader program {}",
                                                         name, program));
                }
                uniforms[name] = location;
                LOG_TRACE("Found uniform '{}' at location {} in program {}", name, location, program);
            }
            return uniforms.at(name);
        }

        GLint attribute(const std::string& name) {
            if (attributes.count(name) == 0) {
                GLint location = glGetAttribLocation(program, name.c_str());
                if (location == -1) {
                    LOG_ERROR("Cannot find attribute '{}' in shader program {} (vertex: {}, fragment: {})",
                              name, program, vshader_path_, fshader_path_);

                    // Log all available attributes
                    logAvailableAttributes();

                    throw std::runtime_error(std::format("Error getting attribute location for '{}' in program {}",
                                                         name, program));
                }
                attributes[name] = location;
                LOG_TRACE("Found attribute '{}' at location {} in program {}", name, location, program);
            }
            return attributes.at(name);
        }

        void logProgramInfo() {
            GLint num_uniforms = 0;
            glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &num_uniforms);

            GLint num_attributes = 0;
            glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &num_attributes);

            LOG_DEBUG("Shader program {} has {} active uniforms and {} active attributes",
                      program, num_uniforms, num_attributes);
        }

        void logAvailableUniforms() {
            GLint count;
            glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);
            LOG_DEBUG("Available uniforms in program {}:", program);

            for (GLint i = 0; i < count; i++) {
                GLsizei length;
                GLint size;
                GLenum type;
                GLchar name[256];
                glGetActiveUniform(program, i, sizeof(name), &length, &size, &type, name);
                GLint location = glGetUniformLocation(program, name);
                LOG_DEBUG("  - '{}' (location: {}, type: 0x{:x}, size: {})", name, location, type, size);
            }
        }

        void logAvailableAttributes() {
            GLint count;
            glGetProgramiv(program, GL_ACTIVE_ATTRIBUTES, &count);
            LOG_DEBUG("Available attributes in program {}:", program);

            for (GLint i = 0; i < count; i++) {
                GLsizei length;
                GLint size;
                GLenum type;
                GLchar name[256];
                glGetActiveAttrib(program, i, sizeof(name), &length, &size, &type, name);
                GLint location = glGetAttribLocation(program, name);
                LOG_DEBUG("  - '{}' (location: {}, type: 0x{:x}, size: {})", name, location, type, size);
            }
        }

        GLuint program;
        GLuint vshader;
        GLuint fshader;
        GLuint gshader = 0;
        std::string vshader_path_;
        std::string fshader_path_;
        std::string gshader_path_;
        std::map<std::string, GLint> uniforms;
        std::map<std::string, GLint> attributes;
        std::map<GLint, GLuint> attribute_buffers;
        GLuint index_buffer = 0;
        GLuint vertex_array = 0;
    };
} // namespace lfs::rendering