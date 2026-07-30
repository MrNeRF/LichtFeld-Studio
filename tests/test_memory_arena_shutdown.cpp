/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdexcept>

namespace {

    lfs::core::RasterizerMemoryArena::Config test_config() {
        lfs::core::RasterizerMemoryArena::Config config;
        config.enable_vmm = false;
        return config;
    }

    class GlobalArenaShutdownTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            auto& manager = lfs::core::GlobalArenaManager::instance();
            if (auto* arena = manager.try_get_arena()) {
                arena->full_reset();
            }
            manager.reconfigure_for_testing(test_config());
            manager.reset();
        }

        void TearDown() override {
            auto& manager = lfs::core::GlobalArenaManager::instance();
            if (auto* arena = manager.try_get_arena()) {
                arena->full_reset();
            }
            manager.reconfigure_for_testing(test_config());
            manager.reset();
        }
    };

} // namespace

TEST_F(GlobalArenaShutdownTest, ShutdownIsIdempotentWithoutConstructingArena) {
    auto& manager = lfs::core::GlobalArenaManager::instance();
    ASSERT_EQ(manager.try_get_arena(), nullptr);

    manager.shutdown();
    manager.shutdown();

    EXPECT_EQ(manager.try_get_arena(), nullptr);
}

TEST(MemoryArenaShutdownTest, FullResetDestroysLastFrameEvent) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    lfs::core::RasterizerMemoryArena arena(test_config());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const uint64_t frame = arena.begin_frame(stream);
    arena.end_frame(frame, stream);
    ASSERT_TRUE(arena.has_last_frame_event_for_testing());

    arena.full_reset();
    EXPECT_FALSE(arena.has_last_frame_event_for_testing());

    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(GlobalArenaShutdownTest, ShutdownLatchesUntilTestingReconfigure) {
    auto& manager = lfs::core::GlobalArenaManager::instance();
    auto& created_arena = manager.get_arena();
    ASSERT_EQ(manager.try_get_arena(), &created_arena);

    manager.shutdown();

    EXPECT_EQ(manager.try_get_arena(), nullptr);
    EXPECT_THROW(static_cast<void>(manager.get_arena()), std::runtime_error);

    manager.reconfigure_for_testing(test_config());
    auto& arena = manager.get_arena();
    const uint64_t frame = arena.begin_frame();
    arena.end_frame(frame);
    arena.full_reset();
}
