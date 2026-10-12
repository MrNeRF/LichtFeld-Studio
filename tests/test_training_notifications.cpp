/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"
#include "core/events.hpp"
#include "python/lfs/notification_bridge.hpp"
#include "python/lfs/py_ui.hpp"

#include <filesystem>
#include <gtest/gtest.h>

namespace {
    using namespace lfs::core::events;

    class TrainingNotificationTest : public testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            ASSERT_TRUE(locale_.initialize((std::filesystem::path(PROJECT_ROOT_PATH) /
                                            "src/visualizer/gui/resources/locales")
                                               .string()));
            ASSERT_TRUE(locale_.setLanguage("en"));
            registry_.clear_for_test();
            registry_.set_enqueue_callback([this](lfs::core::ModalRequest request) {
                requests_.push_back(std::move(request));
            });
            lfs::python::setup_notification_handlers();
            state::TrainingStarted{.total_iterations = 30000}.emit();
        }

        void TearDown() override {
            locale_.reset();
            registry_.set_enqueue_callback({});
            registry_.clear_for_test();
            lfs::event::EventBridge::instance().clear_all();
        }

        void complete(bool stopped = false, bool success = true, bool suppressed = false) {
            state::TrainingCompleted{
                .iteration = 2117,
                .final_loss = 0.125f,
                .elapsed_seconds = 65.0f,
                .success = success,
                .user_stopped = stopped,
                .suppress_notification = suppressed}
                .emit();
            registry_.draw_modals();
        }

        lfs::event::LocalizationManager& locale_ = lfs::event::LocalizationManager::getInstance();
        lfs::python::PyModalRegistry& registry_ = lfs::python::PyModalRegistry::instance();
        std::vector<lfs::core::ModalRequest> requests_;
    };

    TEST_F(TrainingNotificationTest, StoppedExplainsSaveAndReopenBeforeResume) {
        complete(true);
        ASSERT_EQ(requests_.size(), 1);
        const auto& request = requests_.front();
        EXPECT_EQ(request.title, "Training Stopped");
        EXPECT_NE(request.body_rml.find("iteration 2117"), std::string::npos);
        EXPECT_NE(request.body_rml.find("Save the project to keep this state."), std::string::npos);
        EXPECT_NE(request.body_rml.find("Reopen the saved project to resume training."), std::string::npos);
        EXPECT_EQ(request.body_rml.find("Resume is available from the training panel"), std::string::npos);
        ASSERT_EQ(request.buttons.size(), 1);
        EXPECT_EQ(request.buttons.front().label, "OK");
    }

    TEST_F(TrainingNotificationTest, StoppedBodyUsesSelectedLanguage) {
        ASSERT_TRUE(locale_.setLanguage("de"));
        complete(true);
        ASSERT_EQ(requests_.size(), 1);
        EXPECT_NE(requests_.back().body_rml.find("Training wurde bei Iteration 2117"), std::string::npos);
        EXPECT_NE(requests_.back().body_rml.find("Öffnen Sie das gespeicherte Projekt erneut"), std::string::npos);
        EXPECT_EQ(requests_.back().body_rml.find("Save the project"), std::string::npos);
    }

    TEST_F(TrainingNotificationTest, StoppedBodyIsLocalizedInAllLanguages) {
        for (const auto* language : {"en", "de", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"}) {
            SCOPED_TRACE(language);
            ASSERT_TRUE(locale_.setLanguage(language));
            ASSERT_TRUE(locale_.hasKey("messages.training_stopped_body"));
            const auto expected = LOCF("messages.training_stopped_body", 2117);
            const auto paragraph = expected.find("\n\n");
            ASSERT_NE(paragraph, std::string::npos);
            complete(true);
            const auto& body = requests_.back().body_rml;
            EXPECT_NE(body.find(expected.substr(0, paragraph)), std::string::npos);
            EXPECT_NE(body.find(expected.substr(paragraph + 2)), std::string::npos);
            EXPECT_NE(body.find("2117"), std::string::npos);
            EXPECT_EQ(body.find("{}"), std::string::npos);
        }
    }

    TEST_F(TrainingNotificationTest, SuccessfulCompletionKeepsSummaryAndEditAction) {
        complete();
        ASSERT_EQ(requests_.size(), 1);
        const auto& request = requests_.front();
        EXPECT_EQ(request.title, "Training Complete");
        EXPECT_NE(request.body_rml.find("Training completed successfully."), std::string::npos);
        EXPECT_NE(request.body_rml.find("2117 iterations | loss 0.125000 | 1m 5s"), std::string::npos);
        EXPECT_EQ(request.body_rml.find("resume"), std::string::npos);
        ASSERT_EQ(request.buttons.size(), 2);
        EXPECT_EQ(request.buttons.back().label, "OK");
        int edit_actions = 0;
        cmd::SwitchToEditMode::when([&](const auto&) { ++edit_actions; });
        request.on_result({.button_label = "OK"});
        EXPECT_EQ(edit_actions, 0);
        request.on_result({.button_label = request.buttons.front().label});
        EXPECT_EQ(edit_actions, 1);
    }

    TEST_F(TrainingNotificationTest, FailedAndSuppressedCompletionsDoNotShowSuccessModal) {
        complete(false, false);
        complete(true, true, true);
        complete(false, true, true);
        EXPECT_TRUE(requests_.empty());
    }

    TEST_F(TrainingNotificationTest, FinalAndEarlierEvaluationMetricsRemainVisible) {
        state::EvaluationCompleted{.iteration = 2117, .psnr = 30.5f, .ssim = 0.95f, .lpips = 0.125f}.emit();
        complete();
        ASSERT_EQ(requests_.size(), 1);
        EXPECT_NE(requests_.back().body_rml.find("Final metrics: PSNR 30.50 | SSIM 0.9500 | LPIPS 0.1250"), std::string::npos);

        state::EvaluationCompleted{.iteration = 2000, .psnr = 29.0f, .ssim = 0.9f}.emit();
        complete(true);
        ASSERT_EQ(requests_.size(), 2);
        EXPECT_NE(requests_.back().body_rml.find("Last eval @ 2000: PSNR 29.00 | SSIM 0.9000"), std::string::npos);
        EXPECT_EQ(requests_.back().body_rml.find("LPIPS"), std::string::npos);
    }

    TEST_F(TrainingNotificationTest, StartingAnotherRunClearsPreviousEvaluationMetrics) {
        state::EvaluationCompleted{.iteration = 2000, .psnr = 30.0f, .ssim = 0.9f}.emit();
        state::TrainingStarted{.total_iterations = 30000}.emit();
        complete(true);
        ASSERT_EQ(requests_.size(), 1);
        EXPECT_EQ(requests_.front().body_rml.find("PSNR"), std::string::npos);
    }
} // namespace
