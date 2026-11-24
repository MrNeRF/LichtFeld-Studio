// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/sharpness_score.hpp"
#include <atomic>
#include <cstdint>
#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>
#include <vector>

TEST(VideoSharpnessScore, LargeImageUsesTbbWorkers) {
    if (tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) < 2 ||
        tbb::info::default_concurrency() < 2)
        GTEST_SKIP() << "Requires a TBB worker";

    struct WorkerObserver final : tbb::task_scheduler_observer {
        explicit WorkerObserver(tbb::task_arena& arena) : task_scheduler_observer(arena) {}
        void on_scheduler_entry(bool worker) override {
            if (worker)
                entered.store(true, std::memory_order_relaxed);
        }
        std::atomic<bool> entered{false};
    };

    constexpr int width = 4096;
    constexpr int height = 2160;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 3, 128);
    tbb::task_arena arena(2);
    WorkerObserver observer(arena);
    observer.observe();
    for (int attempt = 0; attempt < 3 && !observer.entered.load(std::memory_order_relaxed); ++attempt) {
        const auto score = arena.execute([&] {
            return lfs::io::computeSharpnessScore(pixels.data(), width, height, lfs::io::SharpnessAlgorithm::COMBINED);
        });
        EXPECT_EQ(score, 0.0);
    }
    observer.observe(false);
    EXPECT_TRUE(observer.entered.load(std::memory_order_relaxed));
}
