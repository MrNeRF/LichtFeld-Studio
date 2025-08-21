#include "grid_renderer.hpp"
#include "core/logger.hpp"
#include "shader_paths.hpp"
#include <format>
#include <random>
#include <vector>

namespace gs::rendering {

    Result<void> RenderInfiniteGrid::init() {
        if (initialized_)
            return {};

        LOG_TIMER("RenderInfiniteGrid::init");
        LOG_INFO("Initializing infinite grid renderer");

        // Create shader for infinite grid rendering
        auto result = load_shader("infinite_grid", "infinite_grid.vert", "infinite_grid.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load infinite grid shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        // Create OpenGL objects using RAII
        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // Full-screen quad vertices (-1 to 1)
        float vertices[] = {
            -1.0f, -1.0f,
            1.0f, -1.0f,
            -1.0f, 1.0f,
            1.0f, 1.0f};

        std::span<const float> vertices_span(vertices, sizeof(vertices) / sizeof(float));

        builder.attachVBO(vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 2 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0});

        vao_ = builder.build();

        // Create blue noise texture
        if (auto noise_result = createBlueNoiseTexture(); !noise_result) {
            return noise_result;
        }

        initialized_ = true;
        LOG_INFO("Infinite grid renderer initialized successfully");
        return {};
    }

    Result<void> RenderInfiniteGrid::createBlueNoiseTexture() {
        LOG_TIMER_TRACE("RenderInfiniteGrid::createBlueNoiseTexture");

        const int size = 32;
        std::vector<float> noise_data(size * size);

        // Generate white noise pattern using uniform random distribution
        std::mt19937 rng(42); // Fixed seed for consistency
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < size * size; ++i) {
            noise_data[i] = dist(rng);
        }

        // Create texture using RAII
        GLuint tex_id;
        glGenTextures(1, &tex_id);
        blue_noise_texture_ = Texture(tex_id);

        glBindTexture(GL_TEXTURE_2D, blue_noise_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, size, size, 0, GL_RED, GL_FLOAT, noise_data.data());

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        glBindTexture(GL_TEXTURE_2D, 0);

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            LOG_ERROR("Failed to create blue noise texture: OpenGL error {}", err);
            return std::unexpected(std::format("Failed to create blue noise texture: OpenGL error {}", err));
        }

        LOG_DEBUG("Blue noise texture created: {}x{}", size, size);
        return {};
    }

    void RenderInfiniteGrid::calculateFrustumCorners(const glm::mat4& inv_viewproj,
                                                     glm::vec3& near_origin, glm::vec3& near_x, glm::vec3& near_y,
                                                     glm::vec3& far_origin, glm::vec3& far_x, glm::vec3& far_y) {
        // Transform NDC corners to world space
        auto unproject = [&inv_viewproj](float x, float y, float z) -> glm::vec3 {
            glm::vec4 p = inv_viewproj * glm::vec4(x, y, z, 1.0f);
            return glm::vec3(p) / p.w;
        };

        // Near plane corners in NDC
        glm::vec3 near_bl = unproject(-1.0f, -1.0f, -1.0f);
        glm::vec3 near_br = unproject(1.0f, -1.0f, -1.0f);
        glm::vec3 near_tl = unproject(-1.0f, 1.0f, -1.0f);

        // Far plane corners in NDC
        glm::vec3 far_bl = unproject(-1.0f, -1.0f, 1.0f);
        glm::vec3 far_br = unproject(1.0f, -1.0f, 1.0f);
        glm::vec3 far_tl = unproject(-1.0f, 1.0f, 1.0f);

        // Calculate origins and axes
        near_origin = near_bl;
        near_x = near_br - near_bl;
        near_y = near_tl - near_bl;

        far_origin = far_bl;
        far_x = far_br - far_bl;
        far_y = far_tl - far_bl;
    }

    Result<void> RenderInfiniteGrid::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid()) {
            LOG_ERROR("Grid renderer not initialized");
            return std::unexpected("Grid renderer not initialized");
        }

        LOG_TIMER_TRACE("RenderInfiniteGrid::render");

        // Calculate matrices
        glm::mat4 viewProj = projection * view;
        glm::mat4 invViewProj = glm::inverse(viewProj);

        // Calculate frustum corners
        glm::vec3 near_origin, near_x, near_y, far_origin, far_x, far_y;
        calculateFrustumCorners(invViewProj, near_origin, near_x, near_y, far_origin, far_x, far_y);

        // Camera position in world space (from view matrix)
        glm::mat4 viewInv = glm::inverse(view);
        glm::vec3 view_position = glm::vec3(viewInv[3]);

        // Save current OpenGL state
        GLboolean depth_mask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
        GLint blend_src, blend_dst;
        glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src);
        glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);
        GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);

        // Clear depth buffer so grid is always visible
        glClear(GL_DEPTH_BUFFER_BIT);

        // Set rendering state for grid
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE); // Grid writes to depth buffer

        LOG_TRACE("Rendering grid with plane type: {}, opacity: {}", static_cast<int>(plane_), opacity_);

        // Bind shader and set uniforms
        ShaderScope s(shader_);

        if (auto result = s->set("near_origin", near_origin); !result)
            return result;
        if (auto result = s->set("near_x", near_x); !result)
            return result;
        if (auto result = s->set("near_y", near_y); !result)
            return result;
        if (auto result = s->set("far_origin", far_origin); !result)
            return result;
        if (auto result = s->set("far_x", far_x); !result)
            return result;
        if (auto result = s->set("far_y", far_y); !result)
            return result;

        if (auto result = s->set("view_position", view_position); !result)
            return result;
        if (auto result = s->set("matrix_viewProjection", viewProj); !result)
            return result;
        if (auto result = s->set("plane", static_cast<int>(plane_)); !result)
            return result;
        if (auto result = s->set("opacity", opacity_); !result)
            return result;

        // Bind blue noise texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, blue_noise_texture_);
        if (auto result = s->set("blueNoiseTex32", 0); !result)
            return result;

        // Render the grid
        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        // Restore OpenGL state
        glDepthMask(depth_mask);
        if (!blend_enabled) {
            glDisable(GL_BLEND);
        } else {
            glBlendFunc(blend_src, blend_dst);
        }
        if (!depth_test_enabled) {
            glDisable(GL_DEPTH_TEST);
        }

        return {};
    }

} // namespace gs::rendering