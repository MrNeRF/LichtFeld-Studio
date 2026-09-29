/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "lfs/training/ops/session.hpp"

#include <gtest/gtest.h>

TEST(TrainingOpsCapability, SessionFamilyIsAvailableOnCudaAndVulkan) {
    using namespace lfs::training;
    FamilySet required;
    required.set(static_cast<size_t>(Family::Session));
    const auto& cuda = training_ops(lfs::core::GpuBackend::CUDA);
    ASSERT_NE(cuda.session, nullptr);
    EXPECT_NE(cuda.session->profile, nullptr);
    EXPECT_TRUE(missing_training_families(cuda, required).empty());
    EXPECT_EQ(missing_training_families(TrainingOps{}, required),
              std::vector<std::string_view>{"Session"});
    const auto& vulkan = training_ops(lfs::core::GpuBackend::Vulkan);
    ASSERT_NE(vulkan.session, nullptr);
    EXPECT_NE(vulkan.session->profile, nullptr);
    EXPECT_TRUE(missing_training_families(vulkan, required).empty());
    const auto& metal = training_ops(lfs::core::GpuBackend::Metal);
    EXPECT_EQ(metal.session, nullptr);
    EXPECT_EQ(missing_training_families(metal, required),
              std::vector<std::string_view>{"Session"});
}

TEST(TrainingVulkanSession, ArenaBorrowIsNullableAndDoesNotGrow) {
    using lfs::core::GpuBackend;
    if (!lfs::core::gpu_backend_available(GpuBackend::Vulkan))
        GTEST_SKIP() << "Vulkan tensor backend unavailable";
    const lfs::core::GpuBackendScope scope(GpuBackend::Vulkan);
    const auto* session = lfs::training::training_ops(GpuBackend::Vulkan).session;
    ASSERT_NE(session, nullptr);
    session->reset_arena();
    const auto target = lfs::core::TensorExecutionTarget::current();
    const auto first = session->borrow_idle_arena(4096, 0, target);
    EXPECT_EQ(first.data, nullptr);
    EXPECT_EQ(first.bytes, 0u);
    EXPECT_EQ(session->borrow_idle_arena(0, 0, target).data, nullptr);
    EXPECT_EQ(session->zero_idle_arena(first, 0, target), nullptr);
    EXPECT_FALSE(session->arena_memory_info().has_value());
    session->reset_arena();
}

TEST(TrainingVulkanSession, AllocationBytesMatchPoolBuckets) {
    const auto* session = lfs::training::training_ops(lfs::core::GpuBackend::Vulkan).session;
    ASSERT_NE(session, nullptr);
    constexpr size_t k256K = 256 * 1024;
    constexpr size_t k1M = 1024 * 1024;
    constexpr size_t k16M = 16 * k1M;
    constexpr size_t k256M = 256 * k1M;
    constexpr size_t k1G = 1024 * k1M;
    constexpr size_t k8G = 8 * k1G;
    EXPECT_EQ(session->allocation_bytes(0), k256K);
    EXPECT_EQ(session->allocation_bytes(k256K + 1), 2 * k256K);
    EXPECT_EQ(session->allocation_bytes(k1M + 1), 2 * k1M);
    EXPECT_EQ(session->allocation_bytes(k16M + 1), 2 * k16M);
    EXPECT_EQ(session->allocation_bytes(k256M + 1), k256M + 64 * k1M);
    EXPECT_EQ(session->allocation_bytes(k1G + 1), k1G + 256 * k1M);
    EXPECT_EQ(session->allocation_bytes(k8G + 1), 9 * k1G);
}
