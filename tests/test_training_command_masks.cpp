/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/control/command_api.hpp"

#include <array>
#include <gtest/gtest.h>
#include <vector>

namespace {

    TEST(TrainingCommandRowMask, IndexMaskBroadcastsAcrossShAttributes) {
        constexpr std::size_t rows = 5;
        const auto mask = lfs::core::Tensor::from_vector(
            std::vector<bool>{false, true, true, true, false}, {rows}, lfs::core::Device::CPU);
        const std::array<std::vector<std::size_t>, 4> dimensions{{
            {rows, 3},
            {rows, 1},
            {rows, 1, 3},
            {rows, 15, 3},
        }};

        for (const auto& dims : dimensions) {
            const lfs::core::TensorShape shape{dims};
            const auto expanded = lfs::training::expand_row_mask(mask, shape).contiguous();
            ASSERT_EQ(expanded.shape(), shape);
            const auto values = expanded.to_vector_bool();
            const std::size_t row_width = shape.elements() / rows;
            for (std::size_t i = 0; i < values.size(); ++i) {
                EXPECT_EQ(values[i], i / row_width >= 1 && i / row_width <= 3) << shape.str();
            }
        }
    }

} // namespace
