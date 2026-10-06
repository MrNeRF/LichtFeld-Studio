/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

namespace lfs::vis {

    // Input event types shared across the system
    struct MouseButtonEvent {
        int button;
        int action;
        int mods;
        glm::dvec2 position;
    };

    struct MouseMoveEvent {
        glm::dvec2 position;
        glm::dvec2 delta;
    };

    struct MouseScrollEvent {
        double xoffset;
        double yoffset;
    };

    struct KeyEvent {
        int key;
        int scancode;
        int action;
        int mods;
    };

    struct FileDropEvent {
        std::vector<std::string> paths;
    };

    // Callback types
    using FileDropCallback = std::function<void(const FileDropEvent&)>;

} // namespace lfs::vis
