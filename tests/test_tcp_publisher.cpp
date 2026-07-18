/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error_envelope.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "tcp_publisher.hpp"

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
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

TEST(TcpPublisherTest, WireErrorSerializesFiveCoreFieldsWithoutDetails) {
    const lfs::core::WireError wire{
        .code = "ResourceExhausted",
        .domain = "CUDA",
        .message = "Out of GPU memory",
        .operation_id = 42,
        .retryable = false};

    const nlohmann::json payload = wire;
    EXPECT_EQ(payload["code"], "ResourceExhausted");
    EXPECT_EQ(payload["domain"], "CUDA");
    EXPECT_EQ(payload["message"], "Out of GPU memory");
    EXPECT_EQ(payload["operation_id"].get<std::uint64_t>(), 42u);
    EXPECT_FALSE(payload["retryable"].get<bool>());
    EXPECT_FALSE(payload.contains("details"));
}

TEST(TcpPublisherTest, OptionalWireErrorSerializesPresenceAndNull) {
    const std::optional<lfs::core::WireError> present{lfs::core::WireError{
        .code = "NotFound",
        .domain = "TCP",
        .message = "gone",
        .operation_id = 1,
        .retryable = false}};
    const nlohmann::json present_json = present;
    ASSERT_TRUE(present_json.is_object());
    EXPECT_EQ(present_json["code"], "NotFound");

    const std::optional<lfs::core::WireError> absent;
    const nlohmann::json absent_json = absent;
    EXPECT_TRUE(absent_json.is_null());
}
