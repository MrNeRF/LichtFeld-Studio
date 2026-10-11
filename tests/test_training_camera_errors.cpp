/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/error_envelope.hpp"
#include "core/error_latch.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "training/trainer.hpp"

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string_view>

namespace lfs::training {
    struct TrainerCameraErrorTestAccess {
        static auto step(Trainer& trainer, core::Camera& camera,
                         bool headless, bool gut = false, bool undistort = false) {
            trainer.params_.optimization.headless = headless;
            trainer.params_.optimization.gut = gut;
            trainer.params_.optimization.undistort = undistort;
            return trainer.train_step(1, &camera, {});
        }
    };
} // namespace lfs::training

namespace {
    using namespace lfs::core;

    class TrainingCameraErrors : public testing::Test {
    protected:
        Scene scene;

        void check(bool headless, bool radial, bool tangential, CameraModelType model,
                   std::string_view expected, std::string_view expected_key = {},
                   bool gut = false, bool undistort = false) {
            auto camera = std::make_shared<Camera>(
                Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU),
                100.0f, 100.0f, 32.0f, 32.0f,
                radial ? Tensor::full({4}, 0.01f, Device::CPU) : Tensor{},
                tangential ? Tensor::full({2}, 0.01f, Device::CPU) : Tensor{},
                model, "camera.png", std::filesystem::path{}, std::filesystem::path{}, 64, 64, 0);
            scene.addCamera("camera.png", scene.addGroup("Cameras"), camera);
            lfs::training::Trainer trainer(scene);
            const auto result = lfs::training::TrainerCameraErrorTestAccess::step(
                trainer, *camera, headless, gut, undistort);
            ASSERT_FALSE(result.has_value());
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(result.error().domain(), lfs::ErrorDomain::Training);
            EXPECT_EQ(result.error().user_message(), expected);
            // The stored error, log formatter and MCP envelope must remain readable.
            lfs::core::ErrorLatch stored;
            stored.set(result.error());
            ASSERT_TRUE(stored.get().has_value());
            EXPECT_EQ(stored.get()->user_message(), expected);
            EXPECT_EQ(lfs::format_for_developer(result.error()).find("messages.err_"), std::string::npos);
            const auto wire = lfs::core::to_wire_envelope(result.error());
            EXPECT_EQ(wire["message"], expected);
            EXPECT_EQ(wire.dump().find("messages.err_"), std::string::npos);
            std::string_view key;
            for (const auto& field : result.error().frames().front().fields.entries()) {
                if (field.key == "message_key") {
                    const auto* value = std::get_if<std::string>(&field.value);
                    ASSERT_NE(value, nullptr);
                    key = *value;
                }
            }
            EXPECT_EQ(key, expected_key);
            EXPECT_EQ(trainer.get_current_iteration(), 0);
        }
    };

    TEST_F(TrainingCameraErrors, GuiRadialDistortionPointsToPanel) {
        check(false, true, false, CameraModelType::PINHOLE,
              "Distorted images detected. Enable GUT or Undistort in the Training panel to train on cameras with distortion.", "messages.err_distorted_images");
    }

    TEST_F(TrainingCameraErrors, GuiTangentialDistortionPointsToPanel) {
        check(false, false, true, CameraModelType::PINHOLE,
              "Distorted images detected. Enable GUT or Undistort in the Training panel to train on cameras with distortion.", "messages.err_distorted_images");
    }

    TEST_F(TrainingCameraErrors, GuiNonPinholePointsToPanel) {
        check(false, false, false, CameraModelType::FISHEYE,
              "Enable GUT or Undistort in the Training panel to train on cameras with a non-pinhole model.", "messages.err_non_pinhole");
    }

    TEST_F(TrainingCameraErrors, HeadlessDistortionKeepsFlags) {
        check(true, true, false, CameraModelType::PINHOLE,
              "Distorted images detected. Use --gut or --undistort to train on cameras with distortion.");
    }

    TEST_F(TrainingCameraErrors, HeadlessNonPinholeKeepsFlags) {
        check(true, false, false, CameraModelType::FISHEYE,
              "Use --gut or --undistort to train on cameras with non-pinhole model.");
    }

    TEST_F(TrainingCameraErrors, UnpreparedUndistortionStillRejectsDistortion) {
        check(true, true, false, CameraModelType::PINHOLE,
              "Distorted images detected. Use --gut or --undistort to train on cameras with distortion.", {}, false, true);
    }

    TEST_F(TrainingCameraErrors, GutOrthoRejectionIsUnchanged) {
        check(false, false, false, CameraModelType::ORTHO,
              "Training on cameras with ortho model is not supported yet.", {}, true);
    }
} // namespace
