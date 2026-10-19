/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/events.hpp"
#include "core/scene.hpp"
#include "gui/rml_toast_overlay.hpp"
#include "operation/undo_entry.hpp"
#include "scene/scene_manager.hpp"
#include "training/camera_training_notice.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace {

    using namespace std::chrono_literals;
    constexpr auto kBase = std::chrono::steady_clock::time_point{};
    constexpr std::uint64_t kCameraTrainingFingerprint = 0x4c465343414d0001ULL;

    class ScopedSceneChangedSubscription {
    public:
        explicit ScopedSceneChangedSubscription(const lfs::event::HandlerId id) : id_(id) {}
        ~ScopedSceneChangedSubscription() {
            if (id_ != 0)
                lfs::event::EventBridge::instance().unsubscribe(
                    typeid(lfs::core::events::state::SceneChanged), id_);
        }

        ScopedSceneChangedSubscription(const ScopedSceneChangedSubscription&) = delete;
        ScopedSceneChangedSubscription& operator=(const ScopedSceneChangedSubscription&) = delete;

    private:
        lfs::event::HandlerId id_;
    };

    std::shared_ptr<lfs::core::Camera> make_camera() {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE,
            "frame.png", std::filesystem::path{}, std::filesystem::path{}, 64, 64, 0);
    }

    lfs::vis::gui::ToastRequest camera_notice() {
        return lfs::vis::gui::ToastRequest{
            .title = "Training",
            .message = "Camera changes apply to the next training run.",
            .level = lfs::vis::gui::ErrorNoticeLevel::Info,
            .fingerprint = kCameraTrainingFingerprint,
        };
    }

    struct NoticeDelivery {
        lfs::vis::gui::ToastStack stack;
        std::size_t posted = 0;

        void post() {
            ++posted;
            stack.push(camera_notice(), kBase + std::chrono::milliseconds(posted));
        }
    };

} // namespace

TEST(CameraTrainingNoticeTest, OneEffectiveTogglePostsOneToastAndNoOpPostsNothing) {
    lfs::core::Scene scene;
    const auto group = scene.addCameraGroup("Training", lfs::core::NULL_NODE, 1);
    const auto camera = scene.addCamera("frame.png", group, make_camera());
    ASSERT_NE(camera, lfs::core::NULL_NODE);
    NoticeDelivery delivery;
    ScopedSceneChangedSubscription handler(lfs::core::events::state::SceneChanged::when([&](const auto& event) {
        lfs::vis::detail::dispatch_camera_training_notice(event.mutation_flags, true,
                                                          [&] { delivery.post(); });
    }));

    scene.setCameraTrainingEnabled(camera, false);
    scene.setCameraTrainingEnabled(camera, false);

    EXPECT_EQ(delivery.posted, 1u);
    ASSERT_EQ(delivery.stack.entries.size(), 1u);
    EXPECT_EQ(delivery.stack.entries.front().count, 1u);
}

TEST(CameraTrainingNoticeTest, TenRapidEffectiveChangesPostAndCoalesce) {
    lfs::core::Scene scene;
    const auto group = scene.addCameraGroup("Training", lfs::core::NULL_NODE, 1);
    const auto camera = scene.addCamera("frame.png", group, make_camera());
    ASSERT_NE(camera, lfs::core::NULL_NODE);
    scene.setCameraTrainingEnabled(camera, false);

    NoticeDelivery delivery;
    ScopedSceneChangedSubscription handler(lfs::core::events::state::SceneChanged::when([&](const auto& event) {
        lfs::vis::detail::dispatch_camera_training_notice(event.mutation_flags, true,
                                                          [&] { delivery.post(); });
    }));

    for (int index = 0; index < 10; ++index)
        scene.setCameraTrainingEnabled(camera, index % 2 == 0);

    EXPECT_EQ(delivery.posted, 10u);
    ASSERT_EQ(delivery.stack.entries.size(), 1u);
    EXPECT_EQ(delivery.stack.entries.front().count, 10u);
}

TEST(CameraTrainingNoticeTest, UndoAndRedoEachPostNotice) {
    lfs::vis::SceneManager scene_manager;
    auto& scene = scene_manager.getScene();
    const auto group = scene.addCameraGroup("Training", lfs::core::NULL_NODE, 1);
    const auto camera = scene.addCamera("frame.png", group, make_camera());
    ASSERT_NE(camera, lfs::core::NULL_NODE);

    auto before = lfs::vis::op::SceneGraphMetadataEntry::captureNodes(scene_manager, {"frame.png"});
    ASSERT_EQ(before.size(), 1u);
    scene.setCameraTrainingEnabled(camera, false);
    auto after = lfs::vis::op::SceneGraphMetadataEntry::captureNodes(scene_manager, {"frame.png"});
    ASSERT_EQ(after.size(), 1u);

    NoticeDelivery delivery;
    ScopedSceneChangedSubscription handler(lfs::core::events::state::SceneChanged::when([&](const auto& event) {
        lfs::vis::detail::dispatch_camera_training_notice(event.mutation_flags, true,
                                                          [&] { delivery.post(); });
    }));
    lfs::vis::op::SceneGraphMetadataEntry entry(
        scene_manager, "camera.training",
        {{.before = before.front(), .after = after.front()}});

    entry.undo();
    entry.redo();

    EXPECT_EQ(delivery.posted, 2u);
    ASSERT_EQ(delivery.stack.entries.size(), 1u);
    EXPECT_EQ(delivery.stack.entries.front().count, 2u);
}

TEST(CameraTrainingNoticeTest, ChangeImmediatelyAfterRestartIsDelivered) {
    NoticeDelivery delivery;
    constexpr auto camera_flag = static_cast<std::uint32_t>(
        lfs::core::Scene::MutationType::CAMERA_TRAINING_CHANGED);

    EXPECT_TRUE(lfs::vis::detail::dispatch_camera_training_notice(camera_flag, true,
                                                                  [&] { delivery.post(); }));
    EXPECT_FALSE(lfs::vis::detail::dispatch_camera_training_notice(camera_flag, false,
                                                                   [&] { delivery.post(); }));
    EXPECT_TRUE(lfs::vis::detail::dispatch_camera_training_notice(camera_flag, true,
                                                                  [&] { delivery.post(); }));

    EXPECT_EQ(delivery.posted, 2u);
    ASSERT_EQ(delivery.stack.entries.size(), 1u);
    EXPECT_EQ(delivery.stack.entries.front().count, 2u);
}
