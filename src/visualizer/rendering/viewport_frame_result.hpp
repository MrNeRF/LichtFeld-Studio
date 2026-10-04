/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {

    struct ViewportFrameResult {
        std::shared_ptr<const lfs::core::Tensor> image{};
        std::uint64_t image_generation = 0;
        std::uint64_t split_left_image_generation = 0;
        glm::ivec2 size{0, 0};
        glm::ivec2 alloc_size{0, 0};
        bool flip_y = false;
        bool matches_viewport_extent = false;
        bool rendered = false;
        std::shared_ptr<const lfs::core::Tensor> split_right_image{};
        std::uint64_t split_right_image_generation = 0;
        glm::ivec2 split_right_size{0, 0};
        bool split_right_flip_y = false;
    };

} // namespace lfs::vis
