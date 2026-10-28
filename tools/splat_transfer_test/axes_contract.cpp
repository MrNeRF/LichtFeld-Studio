/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/utils/splat_drag_axes.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

static void check(bool value) {
    if (!value)
        throw std::runtime_error("Splat drag axis contract failed");
}
int main() {
    try {
        using lfs::vis::SplatDragAxes;
        const std::array<int, 4> axes{-2, 2, -3, 3};
        const std::array<glm::vec3, 4> expected{{{1, 2, 3}, {1, -2, -3}, {1, -3, 2}, {1, 3, -2}}};
        check(SplatDragAxes{}.up == -2);
        check(SplatDragAxes::blender().up == 3);
        for (size_t i = 0; i < axes.size(); ++i) {
            const auto matrix = SplatDragAxes{axes[i]}.matrix();
            check(glm::length(glm::vec3(matrix * glm::vec4(1, 2, 3, 1)) - expected[i]) < 1e-6f);
            const auto rotation = glm::mat3(matrix);
            check(std::abs(glm::determinant(rotation) - 1.0f) < 1e-6f);
            check(glm::length(rotation[0] - glm::vec3(1, 0, 0)) < 1e-6f);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    check(std::abs(glm::dot(rotation[a], rotation[b]) - float(a == b)) < 1e-6f);
        }
        for (int invalid : {0, 1, -1, 4, -4, std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}) {
            check(!SplatDragAxes{invalid}.valid());
            bool rejected = false;
            try {
                (void)SplatDragAxes{invalid}.matrix();
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            check(rejected);
        }
        std::cout << "PASS four up axes, original default, Blender conversion, proper rotations and invalid values\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
