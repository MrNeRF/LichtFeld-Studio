/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/native_fs.hpp"
#include "core/nn/models/sam2.hpp"
#include "core/path_utils.hpp"
#include "core/utc_time.hpp"
#include "gui/assisted_masking.hpp"
#include "gui/windows/video_extractor_dialog.hpp"
#include "long_path_test_support.hpp"
#include "preprocessing/preprocess.hpp"
#include "preprocessing/video_mask_extraction.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <thread>

namespace lfs::gui {
    class VideoExtractorDialogTestAccess {
    public:
        static void completedFrames(VideoExtractorDialog& dialog) {
            dialog.review_frames_.resize(3);
            for (int i = 0; i < 3; ++i) {
                auto& frame = dialog.review_frames_[i];
                frame.path = std::array{"late.png", "second.png", "first.png"}[i];
                frame.timestamp = i == 0 ? 2 : 1;
                frame.source_frame = std::array{8, 5, 3}[i];
                frame.width = 32;
                frame.height = 16;
            }
            dialog.review_weights_ = "local-model.lfw";
            dialog.review_run_ = lfs::core::generate_uuid_v4();
            dialog.review_video_ = "clip.mp4";
            dialog.review_output_ = "frames";
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
        AssistedMaskRequest input{};
        input.image = "missing-sam-preview-input";
        input.revision = "r1";
        input.points = {{0.5f, 0.5f, 1}};
        return input;
    }

    TEST(AssistedMasking, CancelledModelAcquisitionDoesNotReadOrDownloadWeights) {
        const auto result = lfs::preprocessing::ensure_sam2_weights(true, [] { return true; });
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
    }

    TEST(AssistedMasking, CancellationDuringCachedModelVerificationKeepsVerifiedWeights) {
        const auto weights = lfs::preprocessing::ensure_sam2_weights(false);
        if (!weights)
            GTEST_SKIP() << "Verified SAM2.1 model is not cached";
        const auto stamp = lfs::core::native_fs::last_write_time(lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::Sam2)->path);
        int polls = 0;
        // Pass the preparation gates, then interrupt the streamed hash.
        const auto cancelled = lfs::preprocessing::ensure_sam2_weights(false, [&] { return ++polls > 4; });
        ASSERT_FALSE(cancelled);
        EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
        EXPECT_GT(polls, 4);
        EXPECT_EQ(lfs::core::native_fs::last_write_time(lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::Sam2)->path), stamp);
        EXPECT_TRUE(lfs::preprocessing::ensure_sam2_weights(false));
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

    TEST(AssistedMasking, ShutdownBeforeFirstRequestRejectsNewWork) {
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return true; });
        service.shutdown();
        service.shutdown();
        EXPECT_THROW(service.submit(missing_source()), std::runtime_error);
        EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
    }

    TEST(AssistedMasking, ExtractionIdentityBoundsAndStaleRunsDoNotCrossSequences) {
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return true; });
        const auto first = lfs::core::generate_uuid_v4(), second = lfs::core::generate_uuid_v4();
        std::vector<lfs::core::ExtractedFrame> frames{{"one.png", 32, 16, 0, 3}, {"two.png", 32, 16, .25, 9}};
        service.beginSequence(first, frames, "out", {});
        auto equivalent = frames;
        equivalent[0].path = "./one.png";
        service.beginSequence(first, equivalent, "out", {});
        EXPECT_EQ(service.sequenceFrames(first, 1, 999).size(), 1);
        EXPECT_TRUE(service.sequenceFrames(first, 99).empty());
        EXPECT_FALSE(service.approveFrames(first, std::array{AssistedMaskApproval{0, 1, true}}));
        service.beginSequence(second, frames, "other", {});
        service.endSequence(first);
        EXPECT_EQ(service.sequenceFrames(second).size(), 2);
        auto stale = missing_source();
        stale.run = first;
        EXPECT_THROW(service.submit(stale), std::invalid_argument);
        EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
        frames[1].timestamp = -1;
        EXPECT_THROW(service.beginSequence(first, frames, "out", {}), std::invalid_argument);
        EXPECT_EQ(service.sequenceFrames(second).size(), 2);
    }

    TEST(AssistedMasking, DestructorRetiresAnActiveSessionWithoutExplicitClose) {
        JobRegistry jobs;
        {
            AssistedMasking service(jobs, [] { return true; });
            ASSERT_NE(service.submit(missing_source()), 0);
        }
        EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
    }

    TEST(AssistedMasking, ImmediateResubmitCannotReuseARetiringJob) {
        JobRegistry jobs;
        AssistedMasking service(jobs, [] { return true; });
        ASSERT_NE(service.submit(missing_source()), 0);
        for (int session = 0; session < 256; ++session) {
            const auto old = jobs.active(JobType::AssistedMask)->handle;
            service.close();
            uint64_t next = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!next && std::chrono::steady_clock::now() < deadline) {
                next = service.submit(missing_source());
                std::this_thread::yield();
            }
            ASSERT_NE(next, 0);
            const auto active = jobs.active(JobType::AssistedMask);
            ASSERT_TRUE(active);
            EXPECT_NE(active->handle, old);
            EXPECT_FALSE(active->cancel_requested);
        }
        service.shutdown();
        EXPECT_FALSE(jobs.anyRunning(JobType::AssistedMask));
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
        EXPECT_EQ(opened[0].video, "clip.mp4");
        EXPECT_EQ(opened[0].output, "frames");
        EXPECT_FALSE(opened[0].run.is_nil());
        ASSERT_EQ(opened[0].frames.size(), 3);
        EXPECT_EQ(opened[0].frames[0].source_frame, 3);
        EXPECT_EQ(opened[0].frames[0].width, 32);
        Access::clear(dialog);
        Access::review(dialog, false);
        EXPECT_EQ(opened.size(), 1);
    }

    TEST(AssistedMaskingGpu, ReviewedSequenceProtectsSeedsAndSavesOnlyApprovedOriginalMasks) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            GTEST_SKIP() << "CUDA device unavailable";
        const auto weights = lfs::preprocessing::ensure_sam2_weights(false);
        if (!weights)
            GTEST_SKIP() << "SAM2.1 weights are not cached";
        lfs::test::long_paths::LongPathDirectory directory;
        const auto fixture = std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/lpips_crop_a.png";
        const auto [width, height, channels] = lfs::core::probe_image_info(fixture);
        std::vector<lfs::core::ExtractedFrame> frames;
        for (int i = 0; i < 3; ++i) {
            const auto path = directory.path / (std::to_string(i) + ".png");
            ASSERT_TRUE(lfs::core::native_fs::copy_file(fixture, path));
            frames.push_back({path, width, height, i * .25, i * 3});
        }
        const auto run = lfs::core::generate_uuid_v4();
        JobRegistry jobs;
        uint64_t document_revision = 1;
        AssistedMasking service(jobs, [] { return true; }, [&] { return std::optional{std::array<uint64_t, 2>{1, document_revision}}; });
        service.beginSequence(run, frames, directory.path, lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::Sam2)->path);
        AssistedMaskRequest input{};
        input.run = run;
        input.image = frames[1].path;
        input.revision = "seed";
        const auto stamp = lfs::core::native_fs::last_write_time(input.image);
#if defined(_MSC_VER)
        const auto time = std::chrono::clock_cast<std::chrono::system_clock>(stamp);
#else
        const auto time = std::filesystem::file_time_type::clock::to_sys(stamp);
#endif
        input.mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
        input.size_bytes = lfs::core::native_fs::file_size(input.image);
        input.points = {{.5f, .5f, 1}};
        const auto wait = [&](uint64_t id, const std::string& phase) {
            AssistedMaskSnapshot state;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(90);
            do {
                state = service.poll();
                if (state.request == id && (state.phase == phase || state.phase == "error"))
                    return state;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now() < deadline);
            return state;
        };
        const auto initial_request = service.submit(input);
        EXPECT_THROW(service.invalidateFrame(run, 1), std::runtime_error);
        EXPECT_EQ(service.poll().request, initial_request);
        auto state = wait(initial_request, "ready");
        ASSERT_EQ(state.phase, "ready") << state.error;
        const auto revision = service.sequenceFrames(run)[1].revision;
        ASSERT_TRUE(service.approveFrames(run, std::array{AssistedMaskApproval{1, revision, true}}));
        service.beginSequence(run, frames, directory.path, lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::Sam2)->path);
        EXPECT_TRUE(service.sequenceFrames(run)[1].approved);
        input.refine_request = state.request;
        input.refine_candidate = 0;
        input.tracking = AssistedMaskTracking{.first = 0, .last = 2, .seed = 1};
        state = wait(service.submit(input), "tracked");
        ASSERT_EQ(state.phase, "tracked") << state.error;
        auto summaries = service.sequenceFrames(run);
        ASSERT_EQ(summaries.size(), 3);
        EXPECT_TRUE(summaries[0].revision);
        EXPECT_TRUE(summaries[2].revision);
        EXPECT_TRUE(summaries[1].approved);
        EXPECT_EQ(summaries[1].revision, revision);
        const auto preview_key = "sam:" + run.to_string() + ":0:" + std::to_string(summaries[0].revision);
        EXPECT_FALSE(service.sequencePreview(preview_key).expired());
        EXPECT_TRUE(service.sequencePreview(preview_key + "invalid").expired());
        EXPECT_TRUE(service.sequencePreview("sam:" + run.to_string() + ":0:0").expired());
        EXPECT_TRUE(service.sequencePreview("sam:" + run.to_string() + ":-1:1").expired());
        // A stale or duplicate member must leave the entire review group unchanged.
        std::array group{AssistedMaskApproval{0, summaries[0].revision, true},
                         AssistedMaskApproval{2, summaries[2].revision + 1, true}};
        const auto sequence_revision = service.poll().sequence_revision;
        EXPECT_FALSE(service.approveFrames(run, group));
        EXPECT_EQ(service.poll().approved, 1);
        EXPECT_EQ(service.poll().sequence_revision, sequence_revision);
        EXPECT_FALSE(service.sequenceFrames(run)[0].approved);
        group[1] = {0, summaries[0].revision, false};
        EXPECT_FALSE(service.approveFrames(run, group));
        group[1] = {2, summaries[2].revision, true};
        ASSERT_TRUE(service.approveFrames(run, group));
        EXPECT_EQ(service.poll().approved, 3);
        EXPECT_EQ(service.poll().sequence_revision, sequence_revision + 1);
        EXPECT_TRUE(service.approveFrames(run, group));
        EXPECT_EQ(service.poll().sequence_revision, sequence_revision + 1);
        group[0].approved = group[1].approved = false;
        ASSERT_TRUE(service.approveFrames(run, group));
        EXPECT_EQ(service.poll().approved, 1);
        EXPECT_FALSE(lfs::core::native_fs::exists(directory.path / "masks"));
        input.tracking.reset();
        input.refine_candidate = -1;
        input.review_frame = 1;
        state = wait(service.submit(input), "ready");
        ASSERT_EQ(state.phase, "ready") << state.error;
        // Forward tracking must not inspect or require sources before the seed.
        ASSERT_TRUE(lfs::core::native_fs::remove(frames[0].path));
        input.review_frame = -1;
        input.refine_request = state.request;
        input.refine_candidate = 0;
        input.tracking = AssistedMaskTracking{.first = 0, .last = 2, .seed = 1, .direction = lfs::preprocessing::VideoMaskDirection::Forward};
        state = wait(service.submit(input), "tracked");
        ASSERT_EQ(state.phase, "tracked") << state.error;
        input.tracking.reset();
        input.refine_candidate = -1;
        // Retaining an extracted-frame proposal is independent of the document.
        const auto camera_input = input;
        input.image = frames[2].path;
        input.mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(lfs::core::file_time_to_system_clock(lfs::core::native_fs::last_write_time(input.image)).time_since_epoch()).count();
        input.size_bytes = lfs::core::native_fs::file_size(input.image);
        input.review_frame = 2;
        state = wait(service.submit(input), "ready");
        ASSERT_EQ(state.phase, "ready") << state.error;
        input.review_frame = -1;
        input.export_candidate = 0;
        input.retain_frame = true;
        input.refine_request = state.request;
        const auto retained = service.submit(input);
        ++document_revision;
        state = wait(retained, "mask_ready");
        ASSERT_EQ(state.phase, "mask_ready") << state.error;
        EXPECT_TRUE(service.sequenceFrames(run)[2].keyframe);
        input = camera_input;
        input.review_frame = 1;
        state = wait(service.submit(input), "ready");
        ASSERT_EQ(state.phase, "ready") << state.error;
        input.review_frame = -1;
        input.export_candidate = 0;
        input.refine_request = state.request;
        state = wait(service.submit(input), "mask_ready");
        ASSERT_EQ(state.phase, "mask_ready") << state.error;
        const auto expected = state.mask;
        ASSERT_TRUE(expected);
        // A camera Apply expires after a document edit; sequence review stays editable.
        ++document_revision;
        EXPECT_EQ(service.poll().phase, "expired");
        ASSERT_TRUE(service.approveFrames(run, std::array{AssistedMaskApproval{2, service.sequenceFrames(run)[2].revision, false}}));
        input.export_candidate = -1;
        input.publish_approved = true;
        const auto blocked_directory = directory.path / "masks";
        {
            std::ofstream blocked(lfs::core::native_fs::native_path(blocked_directory));
            blocked << "not a directory";
            ASSERT_TRUE(blocked);
        }
        state = wait(service.submit(input), "error");
        EXPECT_EQ(state.phase, "error");
        EXPECT_EQ(service.poll().approved, 1);
        ASSERT_TRUE(lfs::core::native_fs::remove(blocked_directory));
        state = wait(service.submit(input), "saved");
        ASSERT_EQ(state.phase, "saved") << state.error;
        EXPECT_EQ(state.completed, 1);
        const auto destination = lfs::preprocessing::video_mask_path(directory.path, input.image.filename());
        std::ifstream file(lfs::core::native_fs::native_path(destination), std::ios::binary);
        const std::vector<char> written((std::istreambuf_iterator<char>(file)), {});
        file.close();
        ASSERT_EQ(written.size(), expected->size());
        EXPECT_EQ(std::memcmp(written.data(), expected->data(), written.size()), 0);
        EXPECT_FALSE(lfs::core::native_fs::exists(lfs::preprocessing::video_mask_path(directory.path, frames[0].path.filename())));
        EXPECT_FALSE(lfs::core::native_fs::exists(lfs::preprocessing::video_mask_path(directory.path, frames[2].path.filename())));
        input.exclude = true;
        state = wait(service.submit(input), "saved");
        ASSERT_EQ(state.phase, "saved") << state.error;
        auto [original, ow, oh, oc] = lfs::core::load_image_from_memory(reinterpret_cast<const uint8_t*>(expected->data()), expected->size());
        auto [excluded, ew, eh, ec] = lfs::core::load_image(destination);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> original_pixels(original, lfs::core::free_image);
        const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> excluded_pixels(excluded, lfs::core::free_image);
        ASSERT_NE(original, nullptr);
        ASSERT_NE(excluded, nullptr);
        ASSERT_EQ(ew, ow);
        ASSERT_EQ(eh, oh);
        for (size_t i = 0; i < size_t(ow) * oh; ++i)
            ASSERT_EQ(original[i * oc] + excluded[i * ec], 255) << i;
        // Retiring a completed publication must preserve the exported asset
        // and its review bookkeeping after the request snapshot is cleared.
        service.close();
        state = wait(0, "idle");
        ASSERT_EQ(state.phase, "idle");
        EXPECT_EQ(state.approved, 1);
        EXPECT_EQ(service.sequenceFrames(run)[1].mask_path, destination);
        EXPECT_EQ(service.sequenceFrames(run)[1].revision, revision);
        EXPECT_TRUE(lfs::core::native_fs::exists(destination));
        auto thumbnail = service.sequencePreview(preview_key);
        ASSERT_FALSE(thumbnail.expired());
        input.publish_approved = false;
        input.review_frame = 1;
        state = wait(service.submit(input), "ready");
        ASSERT_EQ(state.phase, "ready") << state.error;
        // Revoking another frame must preserve the current result and its
        // source identity, even after publishing masks for the sequence.
        const auto current = service.poll();
        ASSERT_EQ(current.image, frames[1].path);
        service.invalidateFrame(run, 0);
        EXPECT_EQ(service.poll().request, current.request);
        EXPECT_EQ(service.poll().phase, current.phase);
        EXPECT_EQ(service.poll().overlays.size(), 1);
        EXPECT_TRUE(thumbnail.expired());
        EXPECT_TRUE(service.sequencePreview(preview_key).expired());
        service.invalidateFrame(run, 1);
        EXPECT_EQ(service.poll().approved, 0);
        EXPECT_FALSE(service.approveFrames(run, std::array{AssistedMaskApproval{1, revision, true}}));
        EXPECT_TRUE(lfs::core::native_fs::exists(destination)); // Draft edits do not delete saved masks.
        EXPECT_EQ(service.sequenceFrames(run)[1].mask_path, destination);
    }

    TEST(AssistedMaskingGpu, RepeatedPromptsPreserveMasksAfterArenaWarmup) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
            GTEST_SKIP() << "CUDA device unavailable";
        auto weights = lfs::preprocessing::ensure_sam2_weights(false);
        if (!weights)
            GTEST_SKIP() << "SAM2.1 weights are not cached; this regression never downloads models";
        {
            // Preview can omit the original-size outputs without changing the
            // decoder's logits or scores. Existing callers keep their full masks.
            auto model = lfs::core::nn::models::Sam2::load(*weights, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            ASSERT_TRUE(model);
            *weights = {}; // Loaded tensors remain owned after the archive mapping closes.
            ASSERT_TRUE(model->set_image(lfs::core::Tensor::zeros({1, 3, 64, 96}, lfs::core::Device::CUDA)));
            const std::vector<lfs::core::nn::models::Sam2PointPrompt> points{{48, 32, 1}};
            const auto full = model->predict(points, std::nullopt);
            ASSERT_TRUE(full);
            ASSERT_EQ(full->masks.ndim(), 3);
            EXPECT_EQ(full->masks.size(0), 3);
            EXPECT_EQ(full->masks.size(1), 64);
            EXPECT_EQ(full->masks.size(2), 96);
            const auto low = model->predict(points, std::nullopt, true, nullptr,
                                            lfs::core::nn::models::Sam2MaskOutput::LowResolutionOnly);
            ASSERT_TRUE(low);
            EXPECT_FALSE(low->masks.is_valid());
            const auto expected = full->low_res_logits.cpu().contiguous().to_vector();
            const auto actual = low->low_res_logits.cpu().contiguous().to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                ASSERT_NEAR(actual[i], expected[i], 1e-3f);
        }
        AssistedMaskRequest input{};
        const auto fixture = std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/lpips_crop_a.png";
        if (!lfs::core::native_fs::is_regular_file(fixture))
            GTEST_SKIP() << "Missing versioned image fixture: " << fixture;
        lfs::test::long_paths::LongPathDirectory temporary;
        input.image = temporary.path / "source.png";
        {
            // Derive a large source from the versioned photo using the canonical
            // resampler, exercising preview downsampling and original-size Apply.
            auto [raw, width, height, channels] = lfs::core::load_image_float(fixture);
            const std::unique_ptr<float, decltype(&lfs::core::free_image)> source(raw, lfs::core::free_image);
            ASSERT_NE(raw, nullptr);
            std::vector<float> enlarged(1792 * 1664 * channels);
            lfs::core::resample_bilinear_f32(raw, width, height, channels, enlarged.data(), 1792, 1664);
            std::vector<uint8_t> pixels(enlarged.size());
            std::ranges::transform(enlarged, pixels.begin(), [](float value) {
                return uint8_t(std::clamp(value * 255.0f, 0.0f, 255.0f));
            });
            ASSERT_TRUE(lfs::core::save_png(input.image, pixels.data(), 1792, 1664, channels, 8, 1));
        }
        input.weights = lfs::preprocessing::inspect_cached_model(lfs::preprocessing::CachedModel::Sam2)->path;
        input.revision = "repeat";
        const auto stamp = lfs::core::native_fs::last_write_time(input.image);
#if defined(_MSC_VER)
        const auto system_stamp = std::chrono::clock_cast<std::chrono::system_clock>(stamp);
#else
        const auto system_stamp = std::filesystem::file_time_type::clock::to_sys(stamp);
#endif
        input.mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(system_stamp.time_since_epoch()).count();
        input.size_bytes = lfs::core::native_fs::file_size(input.image);
        input.points = {{0.5f, 0.5f, 1}};
        JobRegistry jobs;
        std::array<uint64_t, 2> document_revision{1, 1};
        AssistedMasking service(jobs, [] { return true; }, [&] { return document_revision; });
        const auto wait = [&](const uint64_t id, const std::string& phase) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            AssistedMaskSnapshot state;
            do {
                state = service.poll();
                if (state.request == id && (state.phase == phase || state.phase == "error"))
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now() < deadline);
            return state;
        };
        std::vector<std::vector<uint8_t>> reference;
        uint64_t last_prediction = 0;
        for (int repetition = 0; repetition < 6; ++repetition) {
            const auto id = service.submit(input);
            const auto state = wait(id, "ready");
            ASSERT_EQ(state.phase, "ready") << state.error;
            ASSERT_EQ(state.request, id);
            last_prediction = id;
            ASSERT_EQ(state.overlays.size(), 3);
            for (int c = 0; c < 3; ++c) {
                ASSERT_EQ(state.overlays[c].device(), lfs::core::Device::CUDA);
                ASSERT_EQ(state.overlays[c].size(2), 4);
                EXPECT_EQ(state.overlays[c].size(1), 1536);
                EXPECT_LT(state.overlays[c].size(0), 1664);
                const auto mask = state.overlays[c].slice(2, 3, 4).to(lfs::core::DataType::UInt8).to(lfs::core::Device::CPU).contiguous();
                const auto* raw = mask.ptr<uint8_t>();
                std::vector<uint8_t> alpha(raw, raw + mask.numel());
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
        // A recoverable source-proof failure must preserve the current model and
        // candidate, so Apply still works without loading or predicting again.
        ++input.mtime_ns;
        EXPECT_EQ(wait(service.submit(input), "error").phase, "error");
        --input.mtime_ns;
        const auto [source_width, source_height, source_channels] = lfs::core::probe_image_info(input.image);
        std::vector<uint8_t> kept;
        for (const bool exclude : {false, true}) {
            input.refine_request = last_prediction;
            input.export_candidate = 0;
            input.exclude = exclude;
            const auto id = service.submit(input);
            const auto state = wait(id, "mask_ready");
            ASSERT_EQ(state.phase, "mask_ready") << state.error;
            ASSERT_TRUE(state.mask);
            EXPECT_EQ(state.width, source_width);
            EXPECT_EQ(state.height, source_height);
            auto [raw, width, height, channels] = lfs::core::load_image_from_memory(
                reinterpret_cast<const uint8_t*>(state.mask->data()), state.mask->size());
            EXPECT_EQ(width, source_width);
            EXPECT_EQ(height, source_height);
            EXPECT_EQ(channels, 3); // The public in-memory image loader expands grayscale to RGB.
            const std::unique_ptr<unsigned char, decltype(&lfs::core::free_image)> pixels(raw, lfs::core::free_image);
            ASSERT_NE(raw, nullptr);
            const size_t count = size_t(width) * height * channels;
            if (!exclude) {
                kept.assign(raw, raw + count);
                EXPECT_GT(std::ranges::count(kept, uint8_t{0}), 0);
                EXPECT_GT(std::ranges::count(kept, uint8_t{255}), 0);
            } else {
                ASSERT_EQ(count, kept.size());
                for (size_t i = 0; i < count; ++i)
                    ASSERT_EQ(raw[i], 255 - kept[i]) << i;
            }
            ++document_revision[exclude ? 0 : 1]; // Replaced document or persistent scene edit.
            EXPECT_EQ(service.poll().phase, "expired");
        }
    }
} // namespace
