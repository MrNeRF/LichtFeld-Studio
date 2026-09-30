/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/command_api.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace {

    // Model commands (set/scale/clamp_attribute) blend through a row mask
    // expanded to the attribute shape. sh0 and canonical shN have rank 3.
    TEST(TrainingCommandRowMask, IndexMaskBroadcastsOverEveryTrailingAttributeAxis) {
        constexpr std::size_t kRows = 5;
        const auto mask = lfs::core::Tensor::from_vector(
            std::vector<bool>{false, true, true, true, false}, {kRows}, lfs::core::Device::CPU);

        // means/scaling [N, 3], opacity [N, 1], sh0 [N, 1, 3], canonical shN [N, 15, 3].
        const std::array<std::vector<std::size_t>, 4> attribute_dims{{
            {kRows, 3},
            {kRows, 1},
            {kRows, 1, 3},
            {kRows, 15, 3},
        }};
        for (const auto& dims : attribute_dims) {
            const lfs::core::TensorShape shape{dims};
            const auto expanded = lfs::training::expand_row_mask(mask, shape);
            ASSERT_EQ(expanded.shape(), shape) << shape.str();

            const auto values = expanded.contiguous().to_vector_bool();
            ASSERT_EQ(values.size(), shape.elements()) << shape.str();
            const std::size_t per_row = values.size() / kRows;
            for (std::size_t i = 0; i < values.size(); ++i) {
                const std::size_t row = i / per_row;
                EXPECT_EQ(values[i], row >= 1 && row <= 3) << shape.str() << " element " << i;
            }
        }
    }

} // namespace
