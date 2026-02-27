/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace lfs::vis::gui {

    class LFS_VIS_API GLLineRenderer {
    public:
        GLLineRenderer() = default;

        void begin(int screen_w, int screen_h);
        void addLine(glm::vec2 p0, glm::vec2 p1, glm::vec4 color, float thickness = 1.0f);
        void addTriangleFilled(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec4 color);
        void addCircleFilled(glm::vec2 center, float radius, glm::vec4 color, int segments = 16);
        void end();

        static glm::vec4 fromU32(uint32_t abgr);

        void destroyGLResources();

    private:
        struct Vertex {
            float x, y;
            float r, g, b, a;
        };

        static void ensureProgram();

        static GLuint program_;
        static GLuint vao_;
        static GLuint vbo_;

        std::vector<Vertex> vertices_;
        int screen_w_ = 0;
        int screen_h_ = 0;
    };

} // namespace lfs::vis::gui
