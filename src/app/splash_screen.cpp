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
constexpr double MIN_DISPLAY_TIME = 1.5;

// Spinner colors
constexpr float SPINNER_R = 0.4f;
constexpr float SPINNER_G = 0.7f;
constexpr float SPINNER_B = 1.0f;

// Background color (matches app theme)
constexpr float BG_R = 0.11f;
constexpr float BG_G = 0.11f;
constexpr float BG_B = 0.14f;

// Layout positions (normalized, Y: 0=bottom, 1=top)
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
    const GLuint vsId = compileShader(GL_VERTEX_SHADER, vs);
    const GLuint fsId = compileShader(GL_FRAGMENT_SHADER, fs);
    const GLuint program = glCreateProgram();
    glAttachShader(program, vsId);
    glAttachShader(program, fsId);
    glLinkProgram(program);
    glDeleteShader(vsId);
    glDeleteShader(fsId);
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
    if (!data) return img;

    glGenTextures(1, &img.texture);
    glBindTexture(GL_TEXTURE_2D, img.texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_image_free(data);

    constexpr float VERTICES[] = {
        -0.5f, -0.5f, 0.0f, 0.0f,
         0.5f, -0.5f, 1.0f, 0.0f,
         0.5f,  0.5f, 1.0f, 1.0f,
        -0.5f, -0.5f, 0.0f, 0.0f,
         0.5f,  0.5f, 1.0f, 1.0f,
        -0.5f,  0.5f, 0.0f, 1.0f,
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

void drawImage(const ImageData& img, const GLuint program, const float centerX, const float centerY) {
    if (!img.texture) return;

    const float scaleX = static_cast<float>(img.width) / SPLASH_WIDTH;
    const float scaleY = static_cast<float>(img.height) / SPLASH_HEIGHT;

    glUseProgram(program);
    glUniform2f(glGetUniformLocation(program, "uOffset"), centerX, centerY);
    glUniform2f(glGetUniformLocation(program, "uScale"), scaleX, scaleY);
    glBindTexture(GL_TEXTURE_2D, img.texture);
    glBindVertexArray(img.vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

void freeImage(ImageData& img) {
    if (img.texture) glDeleteTextures(1, &img.texture);
    if (img.vao) glDeleteVertexArrays(1, &img.vao);
    if (img.vbo) glDeleteBuffers(1, &img.vbo);
    img = {};
}

struct SpinnerData {
    GLuint program = 0;
    GLuint vao = 0;
    GLuint vbo = 0;
    GLint colorLoc = -1;
    GLint offsetLoc = -1;
    GLint scaleLoc = -1;
};

SpinnerData createSpinner() {
    SpinnerData s;
    s.program = createProgram(SPINNER_VS, SPINNER_FS);
    s.colorLoc = glGetUniformLocation(s.program, "uColor");
    s.offsetLoc = glGetUniformLocation(s.program, "uOffset");
    s.scaleLoc = glGetUniformLocation(s.program, "uScale");

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

void drawSpinner(const SpinnerData& s, const float time, const float centerX, const float centerY) {
    glUseProgram(s.program);
    glBindVertexArray(s.vao);
    glUniform2f(s.offsetLoc, centerX, centerY);
    glUniform2f(s.scaleLoc, SPINNER_RADIUS / SPLASH_WIDTH, SPINNER_RADIUS / SPLASH_HEIGHT);
    glLineWidth(SPINNER_THICKNESS);

    for (int i = 0; i < SPINNER_SEGMENTS; ++i) {
        const float angle = (2.0f * PI * static_cast<float>(i) / SPINNER_SEGMENTS) - time * 4.0f;
        const float alpha = static_cast<float>(i) / SPINNER_SEGMENTS;

        glUniform4f(s.colorLoc, SPINNER_R, SPINNER_G, SPINNER_B, alpha);

        const float cosA = std::cos(angle);
        const float sinA = std::sin(angle);
        const float vertices[] = {cosA * 0.5f, sinA * 0.5f, cosA, sinA};

        glBindBuffer(GL_ARRAY_BUFFER, s.vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_LINES, 0, 2);
    }
    glBindVertexArray(0);
}

void freeSpinner(SpinnerData& s) {
    if (s.program) glDeleteProgram(s.program);
    if (s.vao) glDeleteVertexArrays(1, &s.vao);
    if (s.vbo) glDeleteBuffers(1, &s.vbo);
    s = {};
}

int runSplashImpl(std::function<int()> task, MonitorInfo* outMonitor, const bool keepGlfwAlive) {
    if (!glfwInit()) {
        return task();
    }

    GLFWmonitor* const monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* const mode = glfwGetVideoMode(monitor);
    int monitorX, monitorY;
    glfwGetMonitorPos(monitor, &monitorX, &monitorY);

    if (outMonitor) {
        outMonitor->x = monitorX;
        outMonitor->y = monitorY;
        outMonitor->width = mode->width;
        outMonitor->height = mode->height;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);

    const int xpos = monitorX + (mode->width - SPLASH_WIDTH) / 2;
    const int ypos = monitorY + (mode->height - SPLASH_HEIGHT) / 2;

    GLFWwindow* const window = glfwCreateWindow(SPLASH_WIDTH, SPLASH_HEIGHT, "LichtFeld Studio", nullptr, nullptr);
    if (!window) {
        if (!keepGlfwAlive) glfwTerminate();
        return task();
    }

    glfwSetWindowPos(window, xpos, ypos);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        if (!keepGlfwAlive) glfwTerminate();
        return task();
    }

    const GLuint texturedProgram = createProgram(TEXTURED_VS, TEXTURED_FS);
    SpinnerData spinner = createSpinner();

    const auto assetsDir = core::getAssetsDir();
    ImageData logo = loadImage(assetsDir / "lichtfeld-splash-logo.png");
    ImageData loadingText = loadImage(assetsDir / "lichtfeld-splash-loading.png");

    std::atomic<bool> done{false};
    std::atomic<int> result{0};

    std::thread worker([&]() {
        result = task();
        done = true;
    });

    const double startTime = glfwGetTime();
    glClearColor(BG_R, BG_G, BG_B, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT);

        const double elapsed = glfwGetTime() - startTime;

        drawImage(logo, texturedProgram, 0.5f, LOGO_Y);
        drawImage(loadingText, texturedProgram, 0.5f, TEXT_Y);
        drawSpinner(spinner, static_cast<float>(elapsed), 0.5f, SPINNER_Y);

        glfwSwapBuffers(window);
        glfwPollEvents();

        if (done && elapsed >= MIN_DISPLAY_TIME) {
            break;
        }
    }

    worker.join();

    freeImage(logo);
    freeImage(loadingText);
    freeSpinner(spinner);
    glDeleteProgram(texturedProgram);
    glfwDestroyWindow(window);
    glfwDefaultWindowHints();

    if (!keepGlfwAlive) {
        glfwTerminate();
    }

    return result;
}

} // namespace

int SplashScreen::run(std::function<int()> task) {
    return runSplashImpl(std::move(task), nullptr, false);
}

int SplashScreen::runAndGetMonitor(std::function<int()> task, MonitorInfo& monitor) {
    return runSplashImpl(std::move(task), &monitor, true);
}

} // namespace lfs::app
