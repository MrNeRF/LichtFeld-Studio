/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rmlui_render_interface.hpp"
#include "core/logger.hpp"

#include <stb_image.h>

namespace lfs::vis::gui {

    RmlRenderInterface::RmlRenderInterface() {
        if (!*this)
            LOG_ERROR("RmlUI GL3 render interface failed to initialize");
    }

    RmlRenderInterface::~RmlRenderInterface() = default;

    Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& dimensions,
                                                       const Rml::String& source) {
        int w = 0, h = 0, channels = 0;
        unsigned char* data = stbi_load(source.c_str(), &w, &h, &channels, 4);
        if (!data) {
            LOG_WARN("RmlUI LoadTexture failed: {}", source);
            return 0;
        }

        dimensions.x = w;
        dimensions.y = h;

        const int pixel_count = w * h;
        for (int i = 0; i < pixel_count; ++i) {
            unsigned char* p = data + i * 4;
            const unsigned int a = p[3];
            p[0] = static_cast<unsigned char>((p[0] * a + 127) / 255);
            p[1] = static_cast<unsigned char>((p[1] * a + 127) / 255);
            p[2] = static_cast<unsigned char>((p[2] * a + 127) / 255);
        }

        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return static_cast<Rml::TextureHandle>(tex);
    }

} // namespace lfs::vis::gui
