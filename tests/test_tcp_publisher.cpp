/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "tcp_publisher.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

TEST(TcpPublisherTest, ConcurrentEmittersStopAndDrainCleanly) {
    auto& bridge = lfs::event::EventBridge::instance();
    bridge.clear_all();

    lfs::tcp::PublisherServer publisher(
        0, nullptr, lfs::core::LogLevel::Off, false);
    publisher.start();
    EXPECT_FALSE(publisher.getEndpoint().empty());

    constexpr int thread_count = 8;
    constexpr int events_per_thread = 1000;
    std::vector<std::thread> emitters;
    emitters.reserve(thread_count);
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        emitters.emplace_back([thread_index] {
            for (int event_index = 0; event_index < events_per_thread; ++event_index) {
                lfs::core::events::state::TrainingProgress{
                    .iteration = thread_index * events_per_thread + event_index,
                    .loss = 1.0f,
                    .num_gaussians = 100,
                    .is_refining = false,
                }
                    .emit();
            }
        });
    }
    for (auto& emitter : emitters)
        emitter.join();

    publisher.stop();
    EXPECT_EQ(
        bridge.handler_count(typeid(lfs::core::events::state::TrainingProgress)),
        0u);
    bridge.clear_all();
}
