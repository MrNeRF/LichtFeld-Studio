/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/registry.hpp"

#include <gtest/gtest.h>

TEST(TrainingOpsCapability, SessionFamilyIsCudaOnly) {
    using namespace lfs::training;
    FamilySet required;
    required.set(static_cast<size_t>(Family::Session));
    const auto& cuda = training_ops(lfs::core::GpuBackend::CUDA);
    ASSERT_NE(cuda.session, nullptr);
    EXPECT_NE(cuda.session->profile, nullptr);
    EXPECT_TRUE(missing_training_families(cuda, required).empty());
    EXPECT_EQ(missing_training_families(TrainingOps{}, required),
              std::vector<std::string_view>{"Session"});
    for (const auto backend : {lfs::core::GpuBackend::Vulkan, lfs::core::GpuBackend::Metal}) {
        const auto& table = training_ops(backend);
        EXPECT_EQ(table.session, nullptr);
        EXPECT_EQ(missing_training_families(table, required),
                  std::vector<std::string_view>{"Session"});
    }
}
