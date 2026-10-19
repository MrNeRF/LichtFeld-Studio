/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <glm/vec3.hpp>

namespace lfs::core {
    class Tensor;
    namespace detail {
        glm::vec3 selected_centroid_cuda(const Tensor& means, const Tensor& selected);
    }
} // namespace lfs::core
