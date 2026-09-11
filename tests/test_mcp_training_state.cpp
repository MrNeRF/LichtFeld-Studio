/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/command_center_bridge.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "mcp/mcp_tools.hpp"
#include "training/control/command_api.hpp"
#include "training/trainer.hpp"

#include <gtest/gtest.h>

#include <array>
#include <filesystem>
#include <memory>

namespace {

    [[nodiscard]] std::shared_ptr<lfs::core::Camera> make_command_camera() {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(),
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
    }

    class McpTrainingStateTest : public testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();

            const auto cameras = scene_.addGroup("Cameras");
            scene_.addCamera("camera.png", cameras, make_command_camera());
            trainer_ = std::make_unique<lfs::training::Trainer>(scene_);

            auto& command_center = lfs::training::CommandCenter::instance();
            command_center.clear_snapshot(command_center.snapshot().trainer);
            command_center.clear_loss_history();
            command_center.bind_state_events();
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
            trainer_.reset();
        }

        void populate_snapshot() const {
            auto& command_center = lfs::training::CommandCenter::instance();
            const lfs::training::HookContext context{
                .iteration = kIteration,
                .loss = kLoss,
                .num_gaussians = kNumGaussians,
                .is_refining = true,
                .trainer = trainer_.get()};

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

        lfs::core::Scene scene_;
        std::unique_ptr<lfs::training::Trainer> trainer_;
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

    TEST_F(McpTrainingStateTest, TrainingGetStateReflectsTrainingPausedEvent) {
        lfs::mcp::register_core_tools();

        lfs::core::events::state::TrainingPaused{.iteration = 8200}.emit();

        const auto result = lfs::mcp::ToolRegistry::instance().call_tool(
            "training.get_state", nlohmann::json::object());

        EXPECT_EQ(result.at("iteration"), 8200);
        EXPECT_TRUE(result.at("is_paused").get<bool>());
        EXPECT_FALSE(result.at("is_running").get<bool>());
    }

} // namespace
