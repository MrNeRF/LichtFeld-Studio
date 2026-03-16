/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "framebuffer.hpp"
#include <exception>
#include <glm/vec2.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

namespace lfs::rendering {

    class RenderTargetPool {
    public:
        Result<std::shared_ptr<FrameBuffer>> acquire(std::string_view key, const glm::ivec2& size) {
            if (size.x <= 0 || size.y <= 0) {
                return std::unexpected("Render target size must be positive");
            }

            auto& target = targets_[std::string(key)];
            if (!target) {
                try {
                    target = std::make_shared<FrameBuffer>();
                } catch (const std::exception& e) {
                    return std::unexpected(std::string("Failed to create render target: ") + e.what());
                }
            }

            if (target->getWidth() != size.x || target->getHeight() != size.y) {
                target->resize(size.x, size.y);
            }

            return target;
        }

        void clear() { targets_.clear(); }

    private:
        std::unordered_map<std::string, std::shared_ptr<FrameBuffer>> targets_;
    };

} // namespace lfs::rendering
