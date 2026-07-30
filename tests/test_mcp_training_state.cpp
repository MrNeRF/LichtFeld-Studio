/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "mcp/mcp_tools.hpp"
#include "training/control/command_api.hpp"

#include <gtest/gtest.h>

#include <array>

namespace {

    class McpTrainingStateTest : public testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();

            auto& command_center = lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(command_center.snapshot().trainer);
            command_center.clear_loss_history();
            lfs::event::CommandCenterBridge::instance().set(&command_center);
        }

        void TearDown() override {
            auto& registry = lfs::mcp::ToolRegistry::instance();
            for (const auto* name : registered_tool_names_) {
                registry.unregister_tool(name);
            }

            auto& command_center = lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(command_center.snapshot().trainer);
            command_center.clear_loss_history();
            lfs::event::CommandCenterBridge::instance().set(nullptr);
            lfs::event::EventBridge::instance().clear_all();
        }

        static void populate_snapshot() {
            auto& command_center = lfs::training::CommandCenter::instance();
            const lfs::training::HookContext context{
                .iteration = kIteration,
                .loss = kLoss,
                .num_gaussians = kNumGaussians,
                .is_refining = true,
                .trainer = reinterpret_cast<lfs::training::Trainer*>(0x1)};

            command_center.update_snapshot(
                context,
                kMaxIterations,
                true,
                true,
                false,
                lfs::training::TrainingPhase::OptimizerStep);
            command_center.set_phase(lfs::training::TrainingPhase::OptimizerStep);
        }

        static constexpr int kIteration = 137;
        static constexpr int kMaxIterations = 30'000;
        static constexpr float kLoss = 0.625f;
        static constexpr std::size_t kNumGaussians = 42'000;

        static constexpr std::array registered_tool_names_{
            "training.get_state",
            "training.list_operations",
            "training.get_loss_history",
            "model.set_attribute",
            "model.scale_attribute",
            "model.clamp_attribute",
            "optimizer.set_lr",
            "optimizer.scale_lr",
            "session.pause",
            "session.resume",
            "session.request_stop",
        };
    };

    TEST_F(McpTrainingStateTest, BridgeReadsTheProcessWideCommandCenterSnapshot) {
        populate_snapshot();

        auto* const bridged_command_center = lfs::event::command_center();
        ASSERT_NE(bridged_command_center, nullptr);
        EXPECT_EQ(bridged_command_center, &lfs::training::CommandCenter::instance());

        const auto bridged_snapshot = bridged_command_center->snapshot();
        EXPECT_EQ(bridged_snapshot.iteration, kIteration);
        EXPECT_EQ(bridged_snapshot.max_iterations, kMaxIterations);
        EXPECT_FLOAT_EQ(bridged_snapshot.loss, kLoss);
        EXPECT_EQ(bridged_snapshot.num_gaussians, kNumGaussians);
        EXPECT_TRUE(bridged_snapshot.is_refining);
        EXPECT_TRUE(bridged_snapshot.is_paused);
        EXPECT_TRUE(bridged_snapshot.is_running);
        EXPECT_FALSE(bridged_snapshot.stop_requested);
        EXPECT_EQ(bridged_snapshot.phase, lfs::training::TrainingPhase::OptimizerStep);
        EXPECT_NE(bridged_snapshot.trainer, nullptr);
    }

    TEST_F(McpTrainingStateTest, TrainingGetStateReturnsTheProcessWideSnapshot) {
        populate_snapshot();
        lfs::mcp::register_core_tools();

        const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
            "training.get_state", nlohmann::json::object());

        EXPECT_EQ(result.at("iteration"), kIteration);
        EXPECT_EQ(result.at("max_iterations"), kMaxIterations);
        EXPECT_FLOAT_EQ(result.at("loss").get<float>(), kLoss);
        EXPECT_EQ(result.at("num_gaussians"), kNumGaussians);
        EXPECT_TRUE(result.at("is_refining").get<bool>());
        EXPECT_TRUE(result.at("is_paused").get<bool>());
        EXPECT_TRUE(result.at("is_running").get<bool>());
    }

} // namespace
