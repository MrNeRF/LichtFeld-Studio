/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "gui/assisted_masking.hpp"
#include "gui/windows/video_extractor_dialog.hpp"
#include <chrono>
#include <cstdlib>
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

    TEST(AssistedMaskingGpu, RepeatedPromptsPreserveMasksAfterArenaWarmup) {
        const auto* image = std::getenv("LFS_SAM_TEST_IMAGE");
        const auto* weights = std::getenv("LFS_SAM_TEST_WEIGHTS");
        if (!image || !weights)
            GTEST_SKIP() << "Set LFS_SAM_TEST_IMAGE and LFS_SAM_TEST_WEIGHTS for the real GPU regression";
        AssistedMaskRequest input{.image = lfs::core::utf8_to_path(image),
                                  .weights = lfs::core::utf8_to_path(weights),
                                  .revision = "repeat"};
        const auto stamp = lfs::core::native_fs::last_write_time(input.image);
#if defined(_MSC_VER)
        const auto system_stamp = std::chrono::clock_cast<std::chrono::system_clock>(stamp);
#else
        const auto system_stamp = std::filesystem::file_time_type::clock::to_sys(stamp);
#endif
        input.mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(system_stamp.time_since_epoch()).count();
        input.size_bytes = lfs::core::native_fs::file_size(input.image);
        input.points = {{0.15875f, 0.42083f, 1}, {0.52813f, 0.46262f, 1}, {0.75218f, 0.48079f, 1}};
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return true; });
        std::vector<std::vector<uint8_t>> reference;
        for (int repetition = 0; repetition < 6; ++repetition) {
            const auto id = service.submit(input);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            AssistedMaskSnapshot state;
            do {
                state = service.poll();
                if (state.request == id && (state.phase == "ready" || state.phase == "error"))
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now() < deadline);
            ASSERT_EQ(state.phase, "ready") << state.error;
            ASSERT_EQ(state.request, id);
            ASSERT_EQ(state.overlays.size(), 3);
            for (int c = 0; c < 3; ++c) {
                auto [raw, width, height, channels] = lfs::core::load_image_with_alpha(state.overlays[c]);
                const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> pixels(raw, lfs::core::free_image);
                ASSERT_NE(raw, nullptr);
                ASSERT_EQ(channels, 4);
                std::vector<uint8_t> alpha(size_t(width) * height);
                for (size_t p = 0; p < alpha.size(); ++p)
                    alpha[p] = raw[p * 4 + 3];
                if (!repetition) {
                    reference.push_back(std::move(alpha));
                    continue;
                }
                ASSERT_EQ(alpha.size(), reference[c].size());
                size_t changed = 0;
                for (size_t p = 0; p < alpha.size(); ++p)
                    changed += alpha[p] != reference[c][p];
                EXPECT_LT(double(changed) / alpha.size(), 0.0001) << "repeat=" << repetition << " candidate=" << c;
            }
        }
    }
} // namespace
