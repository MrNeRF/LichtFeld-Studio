#include "text_renderer.hpp"
#include "gl_state_guard.hpp"
#include <format>
#include <ft2build.h>
#include <iostream>
#include FT_FREETYPE_H

namespace gs::rendering {

    TextRenderer::TextRenderer(unsigned int width, unsigned int height)
        : screenWidth(width),
          screenHeight(height) {
        if (auto result = initRenderData(); !result) {
            throw std::runtime_error(result.error());
        }
    }

    TextRenderer::~TextRenderer() {
        for (auto& pair : characters) {
            glDeleteTextures(1, &pair.second.textureID);
        }
    }

    Result<void> TextRenderer::LoadFont(const std::string& fontPath, unsigned int fontSize) {
        // Clear existing characters
        for (auto& pair : characters) {
            glDeleteTextures(1, &pair.second.textureID);
        }
        characters.clear();

        FT_Library ft;
        if (FT_Init_FreeType(&ft)) {
            return std::unexpected("Could not init FreeType Library");
        }

        // RAII wrapper for FreeType library
        struct FTLibraryGuard {
            FT_Library lib;
            ~FTLibraryGuard() {
                if (lib)
                    FT_Done_FreeType(lib);
            }
        } ft_guard{ft};

        FT_Face face;
        if (FT_New_Face(ft, fontPath.c_str(), 0, &face)) {
            return std::unexpected(std::format("Failed to load font from: {}", fontPath));
        }

        // RAII wrapper for FreeType face
        struct FTFaceGuard {
            FT_Face face;
            ~FTFaceGuard() {
                if (face)
                    FT_Done_Face(face);
            }
        } face_guard{face};

        FT_Set_Pixel_Sizes(face, 0, fontSize);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Load only needed characters
        for (unsigned char c : {'X', 'Y', 'Z'}) {
            if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
                // Continue with other characters even if one fails
                std::cerr << "Warning: Failed to load glyph for character: " << c << std::endl;
                continue;
            }

            GLuint texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                face->glyph->bitmap.width,
                face->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                face->glyph->bitmap.buffer);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            Character character = {
                texture,
                glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                static_cast<GLuint>(face->glyph->advance.x)};
            characters.insert(std::pair<char, Character>(c, character));
        }

        glBindTexture(GL_TEXTURE_2D, 0);

        if (characters.empty()) {
            return std::unexpected("Failed to load any characters from font");
        }

        return {};
    }

    Result<void> TextRenderer::initRenderData() {
        // Load shader using shader manager
        auto result = load_shader("text_renderer", "text_renderer.vert", "text_renderer.frag", false);
        if (!result) {
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        // Create VAO and VBO
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        VAOBinder vao_bind(vao_);
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);

        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

        VertexAttribute attr{
            .index = 0,
            .size = 4,
            .type = GL_FLOAT,
            .normalized = GL_FALSE,
            .stride = 4 * sizeof(float),
            .offset = nullptr};
        attr.apply();

        return {};
    }

    void TextRenderer::updateScreenSize(unsigned int width, unsigned int height) {
        screenWidth = width;
        screenHeight = height;
    }

    Result<void> TextRenderer::RenderText(const std::string& text, float x, float y, float scale,
                                          const glm::vec3& color) {
        if (!shader_.valid()) {
            return std::unexpected("Text renderer shader not initialized");
        }

        if (characters.empty()) {
            return std::unexpected("No font loaded");
        }

        // Use RAII for OpenGL state management
        GLStateGuard state_guard;

        // Set up our rendering state
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glDisable(GL_SCISSOR_TEST);
        glDepthMask(GL_FALSE);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Use shader with RAII scope
        ShaderScope s(shader_);

        // Set up orthographic projection
        glm::mat4 projection = glm::ortho(0.0f, static_cast<float>(screenWidth),
                                          0.0f, static_cast<float>(screenHeight));
        if (auto result = s->set("projection", projection); !result)
            return result;
        if (auto result = s->set("textColor", color); !result)
            return result;

        // Ensure we're using texture unit 0
        glActiveTexture(GL_TEXTURE0);
        if (auto result = s->set("text", 0); !result)
            return result;

        // Bind our VAO
        VAOBinder vao_bind(vao_);

        // Iterate through all characters
        for (char c : text) {
            auto it = characters.find(c);
            if (it == characters.end())
                continue;

            Character ch = it->second;

            float xpos = x + ch.bearing.x * scale;
            float ypos = y - (ch.size.y - ch.bearing.y) * scale;

            float w = ch.size.x * scale;
            float h = ch.size.y * scale;

            // Update VBO for each character
            float vertices[6][4] = {
                {xpos, ypos + h, 0.0f, 0.0f},
                {xpos, ypos, 0.0f, 1.0f},
                {xpos + w, ypos, 1.0f, 1.0f},

                {xpos, ypos + h, 0.0f, 0.0f},
                {xpos + w, ypos, 1.0f, 1.0f},
                {xpos + w, ypos + h, 1.0f, 0.0f}};

            // Bind glyph texture
            glBindTexture(GL_TEXTURE_2D, ch.textureID);

            // Update content of VBO memory
            BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

            // Render quad
            glDrawArrays(GL_TRIANGLES, 0, 6);

            // Advance cursors for next glyph (advance is in 1/64 pixels)
            x += (ch.advance >> 6) * scale;
        }

        // State automatically restored by GLStateGuard destructor
        return {};
    }

} // namespace gs::rendering
