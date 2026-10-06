/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "visualizer/core/training_manager.hpp"

#include <gtest/gtest.h>

class TrainingManagerCapacityTest : public ::testing::Test {
protected:
    static std::size_t estimate(const lfs::core::param::TrainingParameters& params,
                                const std::size_t min_capacity) {
        return lfs::vis::TrainerManager::initialSplatLiveEstimate(params, min_capacity);
    }
};

TEST_F(TrainingManagerCapacityTest, RandomInitializationUsesConfiguredPointCountWhenPointCloudIsSmaller) {
    lfs::core::param::TrainingParameters params;
    params.optimization.random = true;
    params.optimization.init_num_pts = 100'000;

    EXPECT_EQ(estimate(params, 9'510), 100'000u);
}

TEST_F(TrainingManagerCapacityTest, RandomInitializationKeepsLargerExistingModelEstimate) {
    lfs::core::param::TrainingParameters params;
    params.optimization.random = true;
    params.optimization.init_num_pts = 100'000;

    EXPECT_EQ(estimate(params, 250'000), 250'000u);
}

TEST_F(TrainingManagerCapacityTest, PointCloudInitializationUsesAvailablePointCount) {
    lfs::core::param::TrainingParameters params;
    params.optimization.random = false;

    EXPECT_EQ(estimate(params, 9'510), 9'510u);
    EXPECT_EQ(estimate(params, 0), 1u);
}
