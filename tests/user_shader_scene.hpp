/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "vulkan_external_tensor_test.hpp"
#include "window/vulkan_context.hpp"
#include <cmath>
#include <glm/glm.hpp>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace lfs::test {
    constexpr std::size_t kNodes = 3, kColumns = 16, kRows = 12;
    constexpr float kC0 = 0.28209479177387814f;

    // Host arrays of the generated scene, in the stored domains (log scale,
    // logit opacity, w-first quaternions); oracles edit these.
    struct Arrays {
        std::vector<float> means, sh0, shN, scales, rotations, opacities;
        std::vector<int> nodes;
        [[nodiscard]] std::size_t size() const { return nodes.size(); }
    };

    // A grid per node; denser grids get proportionally smaller elements.
    inline Arrays generate(const std::size_t columns = kColumns, const std::size_t rows = kRows,
                           const std::optional<float> gray_node_x = std::nullopt) {
        std::mt19937 rng(17);
        std::uniform_real_distribution<float> unit(-1.0f, 1.0f), alpha(0.35f, 0.9f), log_scale(-3.4f, -2.6f);
        const float shrink = std::log(static_cast<float>(kColumns) / static_cast<float>(columns));
        Arrays a;
        for (std::size_t node = 0; node < (gray_node_x ? 1 : kNodes); ++node) {
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    a.nodes.push_back(static_cast<int>(node));
                    if (gray_node_x) {
                        a.means.insert(a.means.end(), {*gray_node_x + 0.12f * (static_cast<float>(column) - 2.0f),
                                                       0.12f * (static_cast<float>(row) - 2.0f), 0.0f});
                        a.sh0.insert(a.sh0.end(), 3, (0.4f - 0.5f) / kC0);
                        a.scales.insert(a.scales.end(), 3, std::log(0.07f));
                        a.rotations.insert(a.rotations.end(), {1.0f, 0.0f, 0.0f, 0.0f});
                        a.opacities.push_back(std::log(0.95f / 0.05f));
                        continue;
                    }
                    a.means.insert(a.means.end(), {-1.2f + 2.4f * column / (columns - 1) + 0.03f * unit(rng),
                                                   -0.8f + 1.6f * row / (rows - 1) + 0.03f * unit(rng),
                                                   -4.0f - 0.4f * node + 0.1f * unit(rng)});
                    for (int c = 0; c < 3; ++c)
                        a.sh0.push_back(0.45f * unit(rng) / kC0);
                    for (int k = 0; k < 9; ++k)
                        a.shN.push_back(0.15f * unit(rng));
                    for (int axis = 0; axis < 3; ++axis)
                        a.scales.push_back(log_scale(rng) + shrink);
                    glm::vec4 q(unit(rng), unit(rng), unit(rng), unit(rng));
                    q = glm::normalize(q + glm::vec4(1.5f, 0.0f, 0.0f, 0.0f));
                    a.rotations.insert(a.rotations.end(), {q.x, q.y, q.z, q.w});
                    const float o = alpha(rng);
                    a.opacities.push_back(std::log(o / (1.0f - o)));
                }
            }
        }
        return a;
    }

    inline lfs::core::Tensor externalTensor(lfs::vis::VulkanContext& context,
                                            const lfs::core::Tensor& values, const std::size_t capacity,
                                            const char* label) {
        auto tensor = lfs::test::makeVulkanExternalTensor(context, values.shape(), values.dtype(), capacity, label);
        if (!tensor)
            throw std::runtime_error(tensor.error());
        tensor->copy_from(values);
        return std::move(*tensor);
    }
} // namespace lfs::test
