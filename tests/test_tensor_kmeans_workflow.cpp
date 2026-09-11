/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

using namespace lfs::core;

TEST(TensorKMeansWorkflowTest, KMeansPlusPlusSelectionPipeline) {
    const auto data = Tensor::from_vector(
        std::vector<float>{0.0f, 0.0f,
                           0.0f, 1.0f,
                           1.0f, 0.0f,
                           2.0f, 2.0f},
        {4, 2}, Device::CUDA);
    auto centroids = Tensor::zeros({2, 2}, Device::CUDA);
    centroids[0] = data[0];

    const auto active_centroids = centroids.slice(0, 0, 1);
    const auto distances = data.cdist(active_centroids).min(1);
    const auto probabilities = distances.square() / distances.square().sum();
    const auto cumulative = probabilities.cumsum(0);
    const auto candidates = cumulative.ge(0.15f).nonzero();

    const auto cumulative_values = cumulative.cpu().to_vector();
    ASSERT_EQ(cumulative_values.size(), 4u);
    constexpr std::array expected_cumulative = {0.0f, 0.1f, 0.2f, 1.0f};
    for (size_t i = 0; i < expected_cumulative.size(); ++i) {
        EXPECT_NEAR(cumulative_values[i], expected_cumulative[i], 1e-6f);
    }
    ASSERT_EQ(candidates.shape(), TensorShape({2, 1}));
    const int64_t next_index = candidates[0].item_int64();
    EXPECT_EQ(next_index, 2);

    centroids[1] = data[next_index];
    EXPECT_EQ(centroids.cpu().to_vector(),
              (std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f}));
}
