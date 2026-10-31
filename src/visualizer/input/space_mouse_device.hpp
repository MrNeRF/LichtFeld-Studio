/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "space_mouse.hpp"
#include <core/export.hpp>
#include <functional>
#include <memory>

struct SDL_Window;

namespace lfs::vis::input {
    class LFS_VIS_API SpaceMouseDevice {
    public:
        SpaceMouseDevice(SDL_Window* window, std::function<void()> wake);
        ~SpaceMouseDevice();
        SpaceMouseDevice(const SpaceMouseDevice&) = delete;
        SpaceMouseDevice& operator=(const SpaceMouseDevice&) = delete;

        SpaceMouseFrame takeFrame();
        [[nodiscard]] std::optional<SpaceMouseConnection> connection() const;
        void setInputEnabled(bool enabled);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

#ifdef _WIN32
    std::unique_ptr<SpaceMouseSource> createSpaceMouseRawInput(SDL_Window* window);
#endif
} // namespace lfs::vis::input
