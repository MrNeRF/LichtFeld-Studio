/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/app_store.hpp"
#include "visualizer/training/training_progress_publisher.hpp"

#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace {
    using lfs::vis::TrainingProgressPublisher;
    using Clock = TrainingProgressPublisher::Clock;
    using std::chrono::milliseconds;

    int publishedIteration() { return lfs::vis::app_store().iteration.get(); }

    // Catches a publisher that forwards every step (the UI then redraws per
    // step), or one that drops the newest values so a paused run shows an old
    // step count.
    TEST(TrainingProgressPublisherTest, PublishesOncePerIntervalAndKeepsTheNewestValue) {
        TrainingProgressPublisher publisher;
        const auto t0 = Clock::now();
        publisher.offer({.iteration = 1000, .loss = 0.5f, .num_gaussians = 10}, t0);
        EXPECT_EQ(publishedIteration(), 1000);
        EXPECT_FALSE(publisher.secondsUntilDue(t0).has_value());

        publisher.offer({.iteration = 1010, .loss = 0.4f, .num_gaussians = 11}, t0 + milliseconds(30));
        publisher.offer({.iteration = 1020, .loss = 0.3f, .num_gaussians = 12}, t0 + milliseconds(60));
        EXPECT_EQ(publishedIteration(), 1000);
        const auto wait = publisher.secondsUntilDue(t0 + milliseconds(60));
        ASSERT_TRUE(wait.has_value());
        EXPECT_NEAR(*wait,
                    std::chrono::duration<double>(TrainingProgressPublisher::kInterval - milliseconds(60)).count(),
                    1e-3);

        publisher.flushDue(t0 + milliseconds(99));
        EXPECT_EQ(publishedIteration(), 1000);
        publisher.flushDue(t0 + TrainingProgressPublisher::kInterval);
        EXPECT_EQ(publishedIteration(), 1020);
        EXPECT_FLOAT_EQ(lfs::vis::app_store().loss.get(), 0.3f);
        EXPECT_EQ(lfs::vis::app_store().num_gaussians.get(), 12);
        EXPECT_FALSE(publisher.secondsUntilDue(t0 + TrainingProgressPublisher::kInterval).has_value());
    }

    // Races training threads offering progress against the UI thread flushing
    // and draining the store. Catches a publisher that lets more than one
    // update per interval through, or loses the last value.
    TEST(TrainingProgressPublisherTest, ConcurrentOffersStayWithinTheRate) {
        TrainingProgressPublisher publisher;
        auto& store = lfs::vis::app_store();
        (void)store.store().drain_dirty_into_frame();
        std::atomic<int> next_iteration{2000};
        std::atomic<bool> stop{false};
        std::vector<std::thread> trainers;
        for (int i = 0; i < 4; ++i) {
            trainers.emplace_back([&] {
                while (!stop.load(std::memory_order_acquire)) {
                    const int iteration = next_iteration.fetch_add(1, std::memory_order_acq_rel);
                    publisher.offer({.iteration = iteration, .loss = 0.1f, .num_gaussians = 1}, Clock::now());
                }
            });
        }
        const auto start = Clock::now();
        int changes = 0;
        int last_seen = publishedIteration();
        while (Clock::now() - start < milliseconds(550)) {
            publisher.flushDue(Clock::now());
            (void)store.store().drain_dirty_into_frame();
            const int seen = publishedIteration();
            if (seen != last_seen) {
                ++changes;
                last_seen = seen;
            }
            std::this_thread::sleep_for(milliseconds(1));
        }
        stop.store(true, std::memory_order_release);
        for (auto& trainer : trainers) {
            trainer.join();
        }
        EXPECT_GE(changes, 4) << "progress stopped reaching the store";
        EXPECT_LE(changes, 7) << "progress reached the store more than once per interval";

        const int final_iteration = next_iteration.load() + 1000;
        const auto final_offer = Clock::now();
        publisher.offer({.iteration = final_iteration, .loss = 0.1f, .num_gaussians = 1}, final_offer);
        publisher.flushDue(final_offer + TrainingProgressPublisher::kInterval);
        EXPECT_FALSE(publisher.secondsUntilDue(final_offer + TrainingProgressPublisher::kInterval).has_value());
        EXPECT_EQ(publishedIteration(), final_iteration) << "the newest progress was lost";
    }
} // namespace
