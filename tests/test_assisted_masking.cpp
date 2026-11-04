/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/events.hpp"
#include "gui/assisted_masking.hpp"
#include "gui/windows/video_extractor_dialog.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

namespace lfs::gui {
    class VideoExtractorDialogTestAccess {
    public:
        static void completedFrames(VideoExtractorDialog& dialog) {
            dialog.review_frames_ = {{.path = "late.png", .timestamp = 2, .source_frame = 8},
                                     {.path = "second.png", .timestamp = 1, .source_frame = 5},
                                     {.path = "first.png", .timestamp = 1, .source_frame = 3}};
            dialog.review_weights_ = "local-model.lfw";
        }
        static void review(VideoExtractorDialog& dialog, bool extracting) {
            dialog.extracting_.store(extracting);
            dialog.handleClick("btn-review-frames");
            dialog.extracting_.store(false);
        }
        static void clear(VideoExtractorDialog& dialog) { dialog.clearBatchVideos(); }
    };
} // namespace lfs::gui

namespace {
    using namespace lfs::vis;
    using namespace lfs::vis::gui;

    AssistedMaskRequest missing_source() {
        return {.image = "missing-sam-preview-input", .revision = "r1", .points = {{0.5f, 0.5f, 1}}};
    }

    TEST(AssistedMasking, InvalidInputAndAdmissionDoNotReserveAJob) {
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return false; });
        auto input = missing_source();
        input.points[0][0] = 1.5f;
        EXPECT_THROW(service.submit(input), std::invalid_argument);
        input.points[0][0] = 0.5f;
        EXPECT_THROW(service.submit(input), std::runtime_error);
        EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
    }

    TEST(AssistedMasking, CancelQueuedOrFailedRequestRetiresAndAllowsANewSession) {
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return true; });
        // The deliberately missing source fails before any CUDA initialization.
        for (int session = 0; session < 3; ++session) {
            ASSERT_NE(service.submit(missing_source()), 0);
            ASSERT_TRUE(jobs.anyRunning(JobType::AssistedMask));
            service.close();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            do {
                (void)service.poll();
                std::this_thread::yield();
            } while (jobs.anyRunning(JobType::AssistedMask) && std::chrono::steady_clock::now() < deadline);
            EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
            EXPECT_EQ(service.poll().phase, "idle");
        }
        service.shutdown();
        service.shutdown();
    }

    TEST(AssistedMasking, ExtractorReviewUsesCompletedFramesInSourceOrderAndPreservesWeights) {
        using Event = lfs::core::events::cmd::OpenImagePreview;
        struct Subscription {
            lfs::event::HandlerId id;
            ~Subscription() { lfs::event::EventBridge::instance().unsubscribe(typeid(Event), id); }
        };
        std::vector<Event> opened;
        const Subscription subscription{Event::when([&](const Event& event) { opened.push_back(event); })};
        lfs::gui::VideoExtractorDialog dialog;
        using Access = lfs::gui::VideoExtractorDialogTestAccess;
        Access::review(dialog, false);
        EXPECT_TRUE(opened.empty());
        Access::completedFrames(dialog);
        Access::review(dialog, true);
        EXPECT_TRUE(opened.empty());
        Access::review(dialog, false);
        ASSERT_EQ(opened.size(), 1);
        EXPECT_EQ(opened[0].paths, (std::vector<std::filesystem::path>{"first.png", "second.png", "late.png"}));
        EXPECT_EQ(opened[0].weights, "local-model.lfw");
        Access::clear(dialog);
        Access::review(dialog, false);
        EXPECT_EQ(opened.size(), 1);
    }
} // namespace
