/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/splash_screen.hpp"
#include "core/executable_path.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

namespace lfs::app {

    namespace {

        constexpr int SPLASH_WIDTH = 500;
        constexpr int SPLASH_HEIGHT = 160;
        constexpr int SPINNER_SEGMENTS = 12;
        constexpr float SPINNER_RADIUS = 10.0f;
        constexpr float SPINNER_THICKNESS = 2.0f;
        constexpr float PI = 3.14159265358979f;

        constexpr float SPINNER_R = 0.4f;
        constexpr float SPINNER_G = 0.7f;
        constexpr float SPINNER_B = 1.0f;

        constexpr float BG_R = 0.11f;
        constexpr float BG_G = 0.11f;
        constexpr float BG_B = 0.14f;

        constexpr float LOGO_Y = 0.70f;
        constexpr float TEXT_Y = 0.35f;
        constexpr float SPINNER_Y = 0.15f;

        const char* const SPINNER_VS = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
uniform vec2 uOffset;
uniform vec2 uScale;
void main() {
    gl_Position = vec4((aPos * uScale + uOffset) * 2.0 - 1.0, 0.0, 1.0);
}
)";

        const char* const SPINNER_FS = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    FragColor = uColor;
}
)";

        const char* const TEXTURED_VS = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
uniform vec2 uOffset;
uniform vec2 uScale;
void main() {
    gl_Position = vec4((aPos * uScale + uOffset) * 2.0 - 1.0, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

        const char* const TEXTURED_FS = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() {
    FragColor = texture(uTexture, TexCoord);
}
)";

        GLuint compileShader(const GLenum type, const char* const source) {
            const GLuint shader = glCreateShader(type);
            glShaderSource(shader, 1, &source, nullptr);
            glCompileShader(shader);
            return shader;
        }

        GLuint createProgram(const char* const vs, const char* const fs) {
            const GLuint vs_id = compileShader(GL_VERTEX_SHADER, vs);
            const GLuint fs_id = compileShader(GL_FRAGMENT_SHADER, fs);
            const GLuint program = glCreateProgram();
            glAttachShader(program, vs_id);
            glAttachShader(program, fs_id);
            glLinkProgram(program);
            glDeleteShader(vs_id);
            glDeleteShader(fs_id);
            return program;
        }

        struct ImageData {
            GLuint texture = 0;
            GLuint vao = 0;
            GLuint vbo = 0;
            int width = 0;
            int height = 0;
        };

        ImageData loadImage(const std::filesystem::path& path) {
            ImageData img;
            int channels;
            stbi_set_flip_vertically_on_load(true);
            unsigned char* const data = stbi_load(path.string().c_str(), &img.width, &img.height, &channels, 4);
            if (!data)
                return img;

            glGenTextures(1, &img.texture);
            glBindTexture(GL_TEXTURE_2D, img.texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            stbi_image_free(data);

            constexpr float VERTICES[] = {
                -0.5f,
                -0.5f,
                0.0f,
                0.0f,
                0.5f,
                -0.5f,
                1.0f,
                0.0f,
                0.5f,
                0.5f,
                1.0f,
                1.0f,
                -0.5f,
                -0.5f,
                0.0f,
                0.0f,
                0.5f,
                0.5f,
                1.0f,
                1.0f,
                -0.5f,
                0.5f,
                0.0f,
                1.0f,
            };

            glGenVertexArrays(1, &img.vao);
            glGenBuffers(1, &img.vbo);
            glBindVertexArray(img.vao);
            glBindBuffer(GL_ARRAY_BUFFER, img.vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES), VERTICES, GL_STATIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
            glEnableVertexAttribArray(1);
            glBindVertexArray(0);

            return img;
        }

        void drawImage(const ImageData& img, const GLuint program, const float center_x, const float center_y) {
            if (!img.texture)
                return;
            const float scale_x = static_cast<float>(img.width) / SPLASH_WIDTH;
            const float scale_y = static_cast<float>(img.height) / SPLASH_HEIGHT;
            glUseProgram(program);
            glUniform2f(glGetUniformLocation(program, "uOffset"), center_x, center_y);
            glUniform2f(glGetUniformLocation(program, "uScale"), scale_x, scale_y);
            glBindTexture(GL_TEXTURE_2D, img.texture);
            glBindVertexArray(img.vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);
        }

        void freeImage(ImageData& img) {
            if (img.texture)
                glDeleteTextures(1, &img.texture);
            if (img.vao)
                glDeleteVertexArrays(1, &img.vao);
            if (img.vbo)
                glDeleteBuffers(1, &img.vbo);
            img = {};
        }

        struct SpinnerData {
            GLuint program = 0;
            GLuint vao = 0;
            GLuint vbo = 0;
            GLint color_loc = -1;
            GLint offset_loc = -1;
            GLint scale_loc = -1;
        };

        SpinnerData createSpinner() {
            SpinnerData s;
            s.program = createProgram(SPINNER_VS, SPINNER_FS);
            s.color_loc = glGetUniformLocation(s.program, "uColor");
            s.offset_loc = glGetUniformLocation(s.program, "uOffset");
            s.scale_loc = glGetUniformLocation(s.program, "uScale");

            glGenVertexArrays(1, &s.vao);
            glGenBuffers(1, &s.vbo);
            glBindVertexArray(s.vao);
            glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
            glBufferData(GL_ARRAY_BUFFER, SPINNER_SEGMENTS * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
            glEnableVertexAttribArray(0);
            glBindVertexArray(0);

            return s;
        }

        void drawSpinner(const SpinnerData& s, const float time, const float center_x, const float center_y) {
            glUseProgram(s.program);
            glBindVertexArray(s.vao);
            glUniform2f(s.offset_loc, center_x, center_y);
            glUniform2f(s.scale_loc, SPINNER_RADIUS / SPLASH_WIDTH, SPINNER_RADIUS / SPLASH_HEIGHT);
            glLineWidth(SPINNER_THICKNESS);

            for (int i = 0; i < SPINNER_SEGMENTS; ++i) {
                const float angle = (2.0f * PI * static_cast<float>(i) / SPINNER_SEGMENTS) - time * 4.0f;
                const float alpha = static_cast<float>(i) / SPINNER_SEGMENTS;
                glUniform4f(s.color_loc, SPINNER_R, SPINNER_G, SPINNER_B, alpha);
                const float cos_a = std::cos(angle);
                const float sin_a = std::sin(angle);
                const float vertices[] = {cos_a * 0.5f, sin_a * 0.5f, cos_a, sin_a};
                glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
                glDrawArrays(GL_LINES, 0, 2);
            }
            glBindVertexArray(0);
        }

        void freeSpinner(SpinnerData& s) {
            if (s.program)
                glDeleteProgram(s.program);
            if (s.vao)
                glDeleteVertexArrays(1, &s.vao);
            if (s.vbo)
                glDeleteBuffers(1, &s.vbo);
            s = {};
        }

    } // namespace

    int SplashScreen::runWithDelay(std::function<int()> task, const int delay_ms) {
        std::atomic<bool> done{false};
        std::atomic<int> result{0};
        std::thread worker([&]() {
            result = task();
            done = true;
        });

        // Wait for delay; if task completes quickly, skip splash
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(delay_ms);
        while (!done && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        if (done) {
            worker.join();
            return result;
        }

        // Task still running - show splash
        if (!glfwInit()) {
            worker.join();
            return result;
        }

        GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* const mode = glfwGetVideoMode(monitor);
        int monitor_x, monitor_y;
        glfwGetMonitorPos(monitor, &monitor_x, &monitor_y);

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);

        const int xpos = monitor_x + (mode->width - SPLASH_WIDTH) / 2;
        const int ypos = monitor_y + (mode->height - SPLASH_HEIGHT) / 2;

        GLFWwindow* const window = glfwCreateWindow(SPLASH_WIDTH, SPLASH_HEIGHT, "LichtFeld Studio", nullptr, nullptr);
        if (!window) {
            glfwTerminate();
            worker.join();
            return result;
        }

        glfwSetWindowPos(window, xpos, ypos);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            glfwDestroyWindow(window);
            glfwTerminate();
            worker.join();
            return result;
        }

        const GLuint textured_program = createProgram(TEXTURED_VS, TEXTURED_FS);
        SpinnerData spinner = createSpinner();

        const auto assets_dir = core::getAssetsDir();
        ImageData logo = loadImage(assets_dir / "lichtfeld-splash-logo.png");
        ImageData loading_text = loadImage(assets_dir / "lichtfeld-splash-loading.png");

        glClearColor(BG_R, BG_G, BG_B, 1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        const double start_time = glfwGetTime();

        while (!glfwWindowShouldClose(window) && !done) {
            const float elapsed = static_cast<float>(glfwGetTime() - start_time);
            glClear(GL_COLOR_BUFFER_BIT);
            drawImage(logo, textured_program, 0.5f, LOGO_Y);
            drawImage(loading_text, textured_program, 0.5f, TEXT_Y);
            drawSpinner(spinner, elapsed, 0.5f, SPINNER_Y);
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        worker.join();

        freeImage(logo);
        freeImage(loading_text);
        freeSpinner(spinner);
        glDeleteProgram(textured_program);
        glfwDestroyWindow(window);
        glfwDefaultWindowHints();
        glfwTerminate();

        return result;
    }

} // namespace lfs::app
