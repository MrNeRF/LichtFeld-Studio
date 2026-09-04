/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/sequencer_ui_state.hpp"
#include "io/video/video_export_options.hpp"
#include "sequencer/animation_clip.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/timeline.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>

namespace {

    TEST(VideoReconstructionExportRequestTest, SharedRequestSnapshotsSelectionWithoutChangingExportOptions) {
        lfs::vis::gui::panels::SequencerUIState state;
        const auto native = state.videoExportRequest(1920, 1080, 30, 18);
        EXPECT_EQ(native.reconstruction_backend_id, "native");
        EXPECT_EQ(native.reconstruction_preset_id, "native");
        EXPECT_EQ(native.reconstruction_fallback, "abort");
        EXPECT_TRUE(native.path.empty());
        EXPECT_TRUE(native.include_provenance);

        state.reconstruction = {.backend_id = "missing", .preset_id = "quality", .fallback = lfs::io::video::VideoReconstructionFallback::Native};
        const auto request = state.videoExportRequest(1280, 720, 48, 22, "export.mp4", false);
        state.reconstruction = {}; // An already queued export owns its selection snapshot.
        EXPECT_EQ(request.reconstruction_backend_id, "missing");
        EXPECT_EQ(request.reconstruction_preset_id, "quality");
        EXPECT_EQ(request.reconstruction_fallback, "native");
        EXPECT_EQ(request.width, 1280);
        EXPECT_EQ(request.height, 720);
        EXPECT_EQ(request.framerate, 48);
        EXPECT_EQ(request.crf, 22);
        EXPECT_EQ(request.path, "export.mp4");
        EXPECT_FALSE(request.include_provenance);
        EXPECT_EQ(state.videoExportRequest(1920, 1080, 30, 18).reconstruction_backend_id, "native");
    }

    using lfs::sequencer::AnimationClip;
    using lfs::sequencer::EasingType;
    using lfs::sequencer::Keyframe;
    using lfs::sequencer::Timeline;
    using lfs::vis::LoopMode;
    using lfs::vis::SequencerController;

    Keyframe makeKeyframe(const float time, const glm::vec3 position = glm::vec3(0.0f),
                          const float focal_length_mm = 35.0f) {
        Keyframe keyframe;
        keyframe.time = time;
        keyframe.position = position;
        keyframe.focal_length_mm = focal_length_mm;
        return keyframe;
    }

    void expectVec3Eq(const glm::vec3& actual, const glm::vec3& expected) {
        EXPECT_FLOAT_EQ(actual.x, expected.x);
        EXPECT_FLOAT_EQ(actual.y, expected.y);
        EXPECT_FLOAT_EQ(actual.z, expected.z);
    }

    struct TempJsonPath {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     std::format("sequencer-regression-{}.json",
                                                 std::chrono::steady_clock::now().time_since_epoch().count());

        ~TempJsonPath() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    };

    TEST(SequencerTimelineRegressionTest, SaveSkipsSyntheticLoopPoint) {
        Timeline timeline;

        timeline.addKeyframe(makeKeyframe(0.0f, {1.0f, 0.0f, 0.0f}));
        timeline.addKeyframe(makeKeyframe(2.0f, {2.0f, 0.0f, 0.0f}));

        auto loop_point = makeKeyframe(3.0f, {1.0f, 0.0f, 0.0f});
        loop_point.is_loop_point = true;
        timeline.addKeyframe(loop_point);

        TempJsonPath file;
        ASSERT_TRUE(timeline.saveToJson(file.path.string()));

        std::ifstream input(file.path);
        ASSERT_TRUE(input.is_open());
        const auto json = nlohmann::json::parse(input);

        ASSERT_TRUE(json.contains("keyframes"));
        ASSERT_EQ(json["keyframes"].size(), 2u);
        EXPECT_FLOAT_EQ(json["keyframes"][0]["time"].get<float>(), 0.0f);
        EXPECT_FLOAT_EQ(json["keyframes"][1]["time"].get<float>(), 2.0f);

        const std::string temp_prefix = file.path.filename().string() + ".";
        for (const auto& entry : std::filesystem::directory_iterator(file.path.parent_path())) {
            const std::string name = entry.path().filename().string();
            EXPECT_FALSE(name.starts_with(temp_prefix) && name.ends_with(".tmp"));
        }
    }

    TEST(SequencerTimelineRegressionTest, SavedTimelineLoadsBackFromTheSameFile) {
        Timeline source;
        source.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}, 35.0f));
        source.addKeyframe(makeKeyframe(2.0f, {4.0f, 5.0f, 6.0f}, 50.0f));

        TempJsonPath file;
        ASSERT_TRUE(source.saveToJson(file.path.string()));

        Timeline loaded;
        ASSERT_TRUE(loaded.loadFromJson(file.path.string()));
        ASSERT_EQ(loaded.realKeyframeCount(), 2u);
        ASSERT_NE(loaded.getKeyframe(0), nullptr);
        ASSERT_NE(loaded.getKeyframe(1), nullptr);
        expectVec3Eq(loaded.getKeyframe(0)->position, {1.0f, 2.0f, 3.0f});
        expectVec3Eq(loaded.getKeyframe(1)->position, {4.0f, 5.0f, 6.0f});
        EXPECT_FLOAT_EQ(loaded.getKeyframe(0)->focal_length_mm, 35.0f);
        EXPECT_FLOAT_EQ(loaded.getKeyframe(1)->focal_length_mm, 50.0f);
    }

    TEST(SequencerTimelineRegressionTest, LoadAcceptsCrlfAndBomTimelineFile) {
        Timeline source;
        source.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}));
        source.addKeyframe(makeKeyframe(2.0f, {4.0f, 5.0f, 6.0f}));

        TempJsonPath file;
        ASSERT_TRUE(source.saveToJson(file.path.string()));

        std::ifstream input(file.path, std::ios::binary);
        ASSERT_TRUE(input.is_open());
        const std::string saved((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        input.close();

        std::string crlf_bom = "\xEF\xBB\xBF";
        crlf_bom.reserve(saved.size() * 2 + 3);
        for (const char ch : saved) {
            if (ch == '\n')
                crlf_bom += "\r\n";
            else
                crlf_bom += ch;
        }

        TempJsonPath crlf_file;
        std::ofstream output(crlf_file.path, std::ios::binary);
        ASSERT_TRUE(output.is_open());
        output << crlf_bom;
        output.close();

        Timeline loaded;
        ASSERT_TRUE(loaded.loadFromJson(crlf_file.path.string()));
        ASSERT_EQ(loaded.realKeyframeCount(), 2u);
        ASSERT_NE(loaded.getKeyframe(0), nullptr);
        ASSERT_NE(loaded.getKeyframe(1), nullptr);
        expectVec3Eq(loaded.getKeyframe(0)->position, {1.0f, 2.0f, 3.0f});
        expectVec3Eq(loaded.getKeyframe(1)->position, {4.0f, 5.0f, 6.0f});
    }

    TEST(SequencerTimelineRegressionTest, SavedTimelineContainsNoCarriageReturns) {
        Timeline timeline;
        timeline.addKeyframe(makeKeyframe(0.0f));

        TempJsonPath file;
        ASSERT_TRUE(timeline.saveToJson(file.path.string()));

        std::ifstream input(file.path, std::ios::binary);
        ASSERT_TRUE(input.is_open());
        const std::string saved((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

        EXPECT_EQ(saved.find('\r'), std::string::npos);
        EXPECT_FALSE(saved.empty());
    }

    TEST(SequencerTimelineRegressionTest, LoadReplacesStateAndClearsAbsentClip) {
        Timeline timeline;
        timeline.addKeyframe(makeKeyframe(9.0f, {9.0f, 0.0f, 0.0f}));
        auto clip = std::make_unique<AnimationClip>("stale");
        clip->addTrack(lfs::sequencer::ValueType::Float, "camera.exposure");
        timeline.setAnimationClip(std::move(clip));
        ASSERT_TRUE(timeline.hasAnimationClip());

        TempJsonPath file;
        nlohmann::json json;
        json["version"] = 3;
        json["keyframes"] = nlohmann::json::array({
            {
                {"time", 1.0f},
                {"position", {1.0f, 2.0f, 3.0f}},
                {"rotation", {1.0f, 0.0f, 0.0f, 0.0f}},
                {"focal_length_mm", 40.0f},
                {"easing", static_cast<int>(EasingType::EASE_OUT)},
            },
            {
                {"time", 4.0f},
                {"position", {4.0f, 5.0f, 6.0f}},
                {"rotation", {1.0f, 0.0f, 0.0f, 0.0f}},
                {"focal_length_mm", 55.0f},
                {"easing", static_cast<int>(EasingType::EASE_IN_OUT)},
            },
        });

        std::ofstream output(file.path);
        ASSERT_TRUE(output.is_open());
        output << json.dump(2);
        output.close();

        ASSERT_TRUE(timeline.loadFromJson(file.path.string()));
        ASSERT_EQ(timeline.realKeyframeCount(), 2u);
        ASSERT_FALSE(timeline.hasAnimationClip());
        ASSERT_NE(timeline.getKeyframe(0), nullptr);
        ASSERT_NE(timeline.getKeyframe(1), nullptr);
        EXPECT_FLOAT_EQ(timeline.getKeyframe(0)->time, 1.0f);
        EXPECT_EQ(timeline.getKeyframe(0)->easing, EasingType::EASE_OUT);
        EXPECT_FLOAT_EQ(timeline.getKeyframe(1)->time, 4.0f);
        EXPECT_EQ(timeline.getKeyframe(1)->easing, EasingType::EASE_IN_OUT);
    }

    TEST(SequencerTimelineRegressionTest, AnimationClipLoadPreservesSerializedTrackIds) {
        nlohmann::json json;
        json["name"] = "clip";
        json["tracks"] = nlohmann::json::array({
            {
                {"id", 7u},
                {"type", "float"},
                {"target", "camera.exposure"},
                {"keyframes", nlohmann::json::array({
                                  {
                                      {"time", 0.0f},
                                      {"value", 1.0f},
                                      {"easing", "linear"},
                                  },
                              })},
            },
            {
                {"id", 42u},
                {"type", "vec3"},
                {"target", "light.color"},
                {"keyframes", nlohmann::json::array({
                                  {
                                      {"time", 1.0f},
                                      {"value", {0.1f, 0.2f, 0.3f}},
                                      {"easing", "ease_out"},
                                  },
                              })},
            },
        });

        auto clip = AnimationClip::fromJson(json);

        ASSERT_EQ(clip.trackCount(), 2u);
        ASSERT_NE(clip.getTrack(7u), nullptr);
        ASSERT_NE(clip.getTrack(42u), nullptr);
        ASSERT_NE(clip.getTrackByPath("camera.exposure"), nullptr);
        ASSERT_NE(clip.getTrackByPath("light.color"), nullptr);
        EXPECT_EQ(clip.getTrackByPath("camera.exposure")->id(), 7u);
        EXPECT_EQ(clip.getTrackByPath("light.color")->id(), 42u);
    }

    TEST(SequencerTimelineRegressionTest, LoadRejectsInvalidStateTransactionally) {
        Timeline timeline;
        timeline.addKeyframe(makeKeyframe(9.0f, {9.0f, 0.0f, 0.0f}));

        nlohmann::json json = {
            {"version", 4},
            {"clip_duration", 10.0f},
            {"keyframes", nlohmann::json::array({
                              {
                                  {"time", 1.0f},
                                  {"position", {1.0f, 2.0f, 3.0f}},
                                  {"rotation", {1.0f, 0.0f, 0.0f, 0.0f}},
                                  {"focal_length_mm", 40.0f},
                                  {"easing", 99},
                              },
                          })},
        };

        TempJsonPath file;
        {
            std::ofstream output(file.path);
            ASSERT_TRUE(output.is_open());
            output << json.dump();
        }
        EXPECT_FALSE(timeline.loadFromJson(file.path.string()));
        ASSERT_EQ(timeline.realKeyframeCount(), 1u);
        EXPECT_FLOAT_EQ(timeline.getKeyframe(0)->time, 9.0f);

        json["keyframes"][0]["easing"] = static_cast<int>(EasingType::LINEAR);
        json["keyframes"][0]["rotation"] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            std::ofstream output(file.path, std::ios::trunc);
            ASSERT_TRUE(output.is_open());
            output << json.dump();
        }
        EXPECT_FALSE(timeline.loadFromJson(file.path.string()));
        ASSERT_EQ(timeline.realKeyframeCount(), 1u);
        EXPECT_FLOAT_EQ(timeline.getKeyframe(0)->time, 9.0f);
    }

    TEST(SequencerTimelineRegressionTest, LoadNormalizesCameraAndAnimationQuaternions) {
        nlohmann::json json = {
            {"version", 4},
            {"keyframes", nlohmann::json::array({
                              {
                                  {"time", 1.0f},
                                  {"position", {1.0f, 2.0f, 3.0f}},
                                  {"rotation", {2.0f, 0.0f, 0.0f, 0.0f}},
                                  {"focal_length_mm", 40.0f},
                                  {"easing", static_cast<int>(EasingType::LINEAR)},
                              },
                          })},
            {"animation_clip",
             {
                 {"tracks", nlohmann::json::array({
                                {
                                    {"id", 1u},
                                    {"type", "quat"},
                                    {"target", "node.rotation"},
                                    {"keyframes", nlohmann::json::array({
                                                      {{"time", 0.0f},
                                                       {"value", {0.0f, 2.0f, 0.0f, 0.0f}},
                                                       {"easing", "linear"}},
                                                  })},
                                },
                            })},
             }},
        };

        TempJsonPath file;
        {
            std::ofstream output(file.path);
            ASSERT_TRUE(output.is_open());
            output << json.dump();
        }

        Timeline timeline;
        ASSERT_TRUE(timeline.loadFromJson(file.path.string()));
        ASSERT_NE(timeline.getKeyframe(0), nullptr);
        EXPECT_NEAR(glm::length(timeline.getKeyframe(0)->rotation), 1.0f, 1e-6f);
        ASSERT_NE(timeline.animationClip(), nullptr);
        const auto* track = timeline.animationClip()->getTrack(1u);
        ASSERT_NE(track, nullptr);
        const auto* rotation = std::get_if<glm::quat>(&track->keyframe(0).value);
        ASSERT_NE(rotation, nullptr);
        EXPECT_NEAR(glm::length(*rotation), 1.0f, 1e-6f);
    }

    TEST(SequencerTimelineRegressionTest, RejectsInvalidAndUnboundedPathRequests) {
        Timeline timeline;
        timeline.addKeyframe(makeKeyframe(0.0f));
        timeline.addKeyframe(makeKeyframe(4.0f));

        EXPECT_THROW(
            (void)timeline.generatePathAtTimeStep(std::numeric_limits<float>::quiet_NaN()),
            std::invalid_argument);
        EXPECT_THROW(
            (void)timeline.generatePathAtTimeStep(std::numeric_limits<float>::denorm_min()),
            std::length_error);
        EXPECT_THROW((void)timeline.generatePath(0), std::invalid_argument);
    }

    TEST(SequencerTimelineRegressionTest, AnimationClipRejectsUnknownTypesAndDuplicateTargets) {
        nlohmann::json invalid_type = {
            {"tracks", nlohmann::json::array({
                           {{"id", 1u}, {"type", "opaque"}, {"target", "node.value"}},
                       })},
        };
        EXPECT_THROW((void)AnimationClip::fromJson(invalid_type), std::runtime_error);

        nlohmann::json duplicate_target = {
            {"tracks", nlohmann::json::array({
                           {{"id", 1u}, {"type", "float"}, {"target", "node.value"}},
                           {{"id", 2u}, {"type", "float"}, {"target", "node.value"}},
                       })},
        };
        EXPECT_THROW((void)AnimationClip::fromJson(duplicate_target), std::runtime_error);
    }

    TEST(SequencerControllerRegressionTest, SelectionTracksKeyframeIdentityAcrossResort) {
        SequencerController controller;
        const auto first_id = controller.addKeyframe(makeKeyframe(1.0f, {1.0f, 0.0f, 0.0f}));
        const auto second_id = controller.addKeyframe(makeKeyframe(3.0f, {2.0f, 0.0f, 0.0f}));

        ASSERT_TRUE(controller.selectKeyframeById(second_id));
        ASSERT_EQ(controller.selectedKeyframeId(), second_id);

        const auto selection_revision_before = controller.selectionRevision();
        ASSERT_TRUE(controller.setKeyframeTimeById(second_id, 0.5f));

        ASSERT_EQ(controller.selectedKeyframeId(), second_id);
        ASSERT_TRUE(controller.selectedKeyframe().has_value());
        EXPECT_EQ(*controller.selectedKeyframe(), 0u);
        EXPECT_EQ(controller.timeline().getKeyframe(0)->id, second_id);
        EXPECT_EQ(controller.timeline().getKeyframe(1)->id, first_id);
        EXPECT_EQ(controller.selectionRevision(), selection_revision_before);
    }

    TEST(SequencerControllerRegressionTest, LoopModeBuildsAndProtectsDerivedEndpoint) {
        SequencerController controller;
        const auto first_id = controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}, 30.0f));
        controller.addKeyframe(makeKeyframe(2.0f, {4.0f, 5.0f, 6.0f}, 50.0f));
        controller.setClipDuration(8.0f);

        controller.toggleLoop();

        ASSERT_EQ(controller.loopMode(), LoopMode::LOOP);
        ASSERT_EQ(controller.timeline().size(), 3u);
        ASSERT_TRUE(controller.isLoopKeyframe(2));

        const auto* loop_point = controller.timeline().getKeyframe(2);
        ASSERT_NE(loop_point, nullptr);
        EXPECT_TRUE(loop_point->is_loop_point);
        EXPECT_FLOAT_EQ(loop_point->time, 8.0f);
        expectVec3Eq(loop_point->position, {1.0f, 2.0f, 3.0f});
        EXPECT_FLOAT_EQ(loop_point->focal_length_mm, 30.0f);

        EXPECT_FALSE(controller.selectKeyframe(2));
        EXPECT_FALSE(controller.setKeyframeTime(2, 10.0f));
        EXPECT_FALSE(controller.removeKeyframeById(loop_point->id));

        ASSERT_TRUE(controller.setKeyframeTimeById(first_id, 1.0f));
        loop_point = controller.timeline().getKeyframe(controller.timeline().size() - 1);
        ASSERT_NE(loop_point, nullptr);
        EXPECT_TRUE(loop_point->is_loop_point);
        // Loop keyframe is anchored to clipDuration, not realEndTime, so reordering keyframes
        // doesn't move it.
        EXPECT_FLOAT_EQ(loop_point->time, 8.0f);

        ASSERT_TRUE(controller.updateKeyframeById(first_id, {7.0f, 8.0f, 9.0f}, glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 45.0f));
        loop_point = controller.timeline().getKeyframe(controller.timeline().size() - 1);
        ASSERT_NE(loop_point, nullptr);
        expectVec3Eq(loop_point->position, {7.0f, 8.0f, 9.0f});
        EXPECT_FLOAT_EQ(loop_point->focal_length_mm, 45.0f);

        controller.setClipDuration(12.0f);
        loop_point = controller.timeline().getKeyframe(controller.timeline().size() - 1);
        ASSERT_NE(loop_point, nullptr);
        EXPECT_FLOAT_EQ(loop_point->time, 12.0f);
    }

    TEST(SequencerControllerRegressionTest, PreviewKeyframeTimeDefersSortAndLoopRebuildUntilCommit) {
        SequencerController controller;
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 0.0f, 0.0f}));
        const auto middle_id = controller.addKeyframe(makeKeyframe(2.0f, {2.0f, 0.0f, 0.0f}));
        controller.addKeyframe(makeKeyframe(4.0f, {3.0f, 0.0f, 0.0f}));
        controller.setClipDuration(10.0f);
        controller.toggleLoop();

        const auto timeline_revision_before = controller.timelineRevision();
        ASSERT_TRUE(controller.previewKeyframeTimeById(middle_id, 5.0f));
        EXPECT_GT(controller.timelineRevision(), timeline_revision_before);
        EXPECT_EQ(controller.timeline().size(), 3u);
        ASSERT_NE(controller.timeline().getKeyframe(1), nullptr);
        EXPECT_EQ(controller.timeline().getKeyframe(1)->id, middle_id);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframe(1)->time, 5.0f);

        ASSERT_TRUE(controller.commitKeyframeTimeById(middle_id));
        ASSERT_EQ(controller.timeline().size(), 4u);
        ASSERT_NE(controller.timeline().getKeyframe(2), nullptr);
        EXPECT_EQ(controller.timeline().getKeyframe(2)->id, middle_id);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframe(2)->time, 5.0f);

        const auto* loop_point = controller.timeline().getKeyframe(3);
        ASSERT_NE(loop_point, nullptr);
        EXPECT_TRUE(loop_point->is_loop_point);
        EXPECT_FLOAT_EQ(loop_point->time, 10.0f);
    }

    TEST(SequencerControllerRegressionTest, SeekToLastKeyframeSkipsSyntheticLoopPoint) {
        SequencerController controller;
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 0.0f, 0.0f}));
        controller.addKeyframe(makeKeyframe(2.0f, {2.0f, 0.0f, 0.0f}));
        controller.toggleLoop();

        ASSERT_EQ(controller.timeline().size(), 3u);
        ASSERT_TRUE(controller.isLoopKeyframe(2));

        controller.seekToLastKeyframe();

        EXPECT_FLOAT_EQ(controller.playhead(), 2.0f);
    }

    TEST(SequencerMappingRegressionTest, RulerMajorTicksAreExactMultiplesOfRoundIntervals) {
        Timeline timeline;
        ASSERT_FLOAT_EQ(timeline.clipDuration(), 30.0f);

        constexpr float kRoundIntervals[] = {0.25f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
        constexpr float kZooms[] = {1.0f, 2.0f, 3.0f, 4.0f};

        for (const float zoom : kZooms) {
            const float visible = lfs::vis::sequencer_ui::displayEndTime(timeline, zoom);
            const float major = lfs::vis::sequencer_ui::rulerMajorInterval(visible);
            const float minor = major / 4.0f;
            ASSERT_GT(minor, 0.0f);

            bool is_round = false;
            for (const float interval : kRoundIntervals) {
                if (std::abs(major - interval) < 1e-6f)
                    is_round = true;
            }
            EXPECT_TRUE(is_round) << "zoom=" << zoom << " major=" << major;

            int major_count = 0;
            const int first_index = 0;
            for (int i = 0;; ++i) {
                const int index = first_index + i;
                const float t = static_cast<float>(index) * minor;
                if (t > visible)
                    break;
                if ((index % 4) != 0)
                    continue;
                ++major_count;
                const float quotient = t / major;
                EXPECT_NEAR(quotient, std::round(quotient), 1e-5f)
                    << "zoom=" << zoom << " t=" << t << " major=" << major;
            }
            EXPECT_GT(major_count, 0) << "zoom=" << zoom;
        }

        // Zoom 3 on a 30s clip used to divide the 1s ladder by 3 again (ticks at 1/3 s).
        EXPECT_FLOAT_EQ(lfs::vis::sequencer_ui::rulerMajorInterval(
                            lfs::vis::sequencer_ui::displayEndTime(timeline, 3.0f)),
                        1.0f);
    }

    TEST(SequencerMappingRegressionTest, TimeScreenMappingRoundTripsWithZoomAndPan) {
        Timeline timeline;
        timeline.addKeyframe(makeKeyframe(0.0f));
        timeline.addKeyframe(makeKeyframe(12.0f));

        constexpr float zoom = 2.0f;
        constexpr float pan = 1.75f;
        constexpr float timeline_x = 100.0f;
        constexpr float timeline_width = 640.0f;
        const float display_end = lfs::vis::sequencer_ui::displayEndTime(timeline, zoom);

        constexpr float original_time = 5.25f;
        const float x = lfs::vis::sequencer_ui::timeToScreenX(
            original_time, timeline_x, timeline_width, display_end, pan);
        const float roundtrip_time = lfs::vis::sequencer_ui::screenXToTime(
            x, timeline_x, timeline_width, display_end, pan);

        EXPECT_NEAR(roundtrip_time, original_time, 1e-5f);
    }

    TEST(SequencerMappingRegressionTest, ThumbnailSlotUsesCenterSampleTime) {
        constexpr float timeline_x = 50.0f;
        constexpr float timeline_width = 600.0f;
        constexpr float display_end = 6.0f;
        constexpr float pan = 1.0f;

        const auto slot = lfs::vis::sequencer_ui::thumbnailSlotAt(
            2, 6, timeline_x, timeline_width, display_end, pan);

        EXPECT_NEAR(slot.sample_time,
                    lfs::vis::sequencer_ui::screenXToTime(
                        slot.screen_center_x, timeline_x, timeline_width, display_end, pan),
                    1e-5f);
        EXPECT_NEAR(slot.interval_start_time,
                    lfs::vis::sequencer_ui::screenXToTime(
                        slot.screen_x, timeline_x, timeline_width, display_end, pan),
                    1e-5f);
        EXPECT_NEAR(slot.interval_end_time,
                    lfs::vis::sequencer_ui::screenXToTime(
                        slot.screen_x + slot.screen_width, timeline_x, timeline_width, display_end, pan),
                    1e-5f);
        EXPECT_NEAR(slot.sample_time,
                    (slot.interval_start_time + slot.interval_end_time) * 0.5f,
                    1e-5f);
    }

    TEST(SequencerMappingRegressionTest, ThumbnailDensityIncreasesWithZoom) {
        constexpr float timeline_width = 800.0f;
        constexpr float base_thumb_width = 96.0f;

        const int zoomed_out = lfs::vis::sequencer_ui::thumbnailCount(
            timeline_width, base_thumb_width, 0.5f);
        const int zoomed_in = lfs::vis::sequencer_ui::thumbnailCount(
            timeline_width, base_thumb_width, 4.0f);

        EXPECT_GT(zoomed_out, 0);
        EXPECT_GT(zoomed_in, zoomed_out);
    }

    TEST(SequencerMappingRegressionTest, ThumbnailSamplingAnchorsAnimationEndpoints) {
        constexpr float content_start = 0.0f;
        constexpr float content_end = 2.0f;

        EXPECT_FLOAT_EQ(
            lfs::vis::sequencer_ui::resolvedThumbnailSampleTime(0.28f, 0.0f, 0.56f, content_start, content_end),
            content_start);
        EXPECT_FLOAT_EQ(
            lfs::vis::sequencer_ui::resolvedThumbnailSampleTime(1.72f, 1.44f, 2.0f, content_start, content_end),
            content_end);
        EXPECT_FLOAT_EQ(
            lfs::vis::sequencer_ui::resolvedThumbnailSampleTime(1.12f, 0.84f, 1.40f, content_start, content_end),
            1.12f);
    }

    TEST(SequencerTimelineRegressionTest, TimeSampledPathUsesTimelineEvaluationAtUniformTimes) {
        Timeline timeline;

        auto first = makeKeyframe(0.0f, {0.0f, 0.0f, 0.0f});
        first.easing = EasingType::EASE_IN;
        timeline.addKeyframe(first);
        timeline.addKeyframe(makeKeyframe(1.0f, {1.0f, 2.0f, 0.0f}));
        timeline.addKeyframe(makeKeyframe(4.0f, {5.0f, 3.0f, 0.0f}));

        const auto points = timeline.generatePathAtTimeStep(1.0f);

        ASSERT_EQ(points.size(), 5u);
        expectVec3Eq(points[0], timeline.evaluate(0.0f).position);
        expectVec3Eq(points[1], timeline.evaluate(1.0f).position);
        expectVec3Eq(points[2], timeline.evaluate(2.0f).position);
        expectVec3Eq(points[3], timeline.evaluate(3.0f).position);
        expectVec3Eq(points[4], timeline.evaluate(4.0f).position);
    }

    [[nodiscard]] float clampCenteredSpan(const float center, const float extent, const float span) {
        if (extent <= 0.0f)
            return 0.0f;
        const float half_span = std::max(span * 0.5f, 0.0f);
        if (extent <= span)
            return extent * 0.5f;
        return std::clamp(center, half_span, extent - half_span);
    }

    TEST(SequencerTimelineRegressionTest, PlayheadDrawAndHitUseTheSameClampSpan) {
        using lfs::vis::panel_config::PLAYHEAD_HANDLE_WIDTH;
        using lfs::vis::panel_config::PLAYHEAD_HIT_RADIUS;

        EXPECT_FLOAT_EQ(PLAYHEAD_HANDLE_WIDTH, 14.0f);
        EXPECT_GE(PLAYHEAD_HIT_RADIUS, 8.0f);

        constexpr float timeline_width = 200.0f;
        constexpr float dp = 1.0f;
        const float draw_span = PLAYHEAD_HANDLE_WIDTH * dp;
        const float hit_span = PLAYHEAD_HANDLE_WIDTH * dp;
        EXPECT_FLOAT_EQ(draw_span, hit_span);

        EXPECT_FLOAT_EQ(clampCenteredSpan(0.0f, timeline_width, draw_span), 7.0f);
        EXPECT_FLOAT_EQ(clampCenteredSpan(0.0f, timeline_width, hit_span), 7.0f);
        EXPECT_FLOAT_EQ(clampCenteredSpan(timeline_width, timeline_width, draw_span), 193.0f);
        EXPECT_FLOAT_EQ(clampCenteredSpan(timeline_width, timeline_width, hit_span), 193.0f);

        // Old panel.cpp used an 8dp span while input.cpp used 14dp (~3dp edge mismatch).
        EXPECT_FLOAT_EQ(clampCenteredSpan(0.0f, timeline_width, 8.0f), 4.0f);
        EXPECT_NE(clampCenteredSpan(0.0f, timeline_width, draw_span),
                  clampCenteredSpan(0.0f, timeline_width, 8.0f));
    }

    TEST(SequencerTimelineRegressionTest, ParseVideoResolutionAcceptsRejectsAndClamps) {
        using lfs::io::video::parseVideoResolution;

        {
            const auto parsed = parseVideoResolution("1920x1080");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 1920);
            EXPECT_EQ(parsed->height, 1080);
        }
        {
            const auto parsed = parseVideoResolution("1920 x 1080");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 1920);
            EXPECT_EQ(parsed->height, 1080);
        }
        {
            const auto parsed = parseVideoResolution(" 1280X720 ");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 1280);
            EXPECT_EQ(parsed->height, 720);
        }
        {
            const auto parsed = parseVideoResolution("1080x1920");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 1080);
            EXPECT_EQ(parsed->height, 1920);
        }
        {
            const auto parsed = parseVideoResolution("17x17");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 16);
            EXPECT_EQ(parsed->height, 16);
        }
        {
            const auto parsed = parseVideoResolution("1x99999");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 16);
            EXPECT_EQ(parsed->height, 8192);
        }
        {
            const auto parsed = parseVideoResolution("8193x16");
            ASSERT_TRUE(parsed.has_value());
            EXPECT_EQ(parsed->width, 8192);
            EXPECT_EQ(parsed->height, 16);
        }

        EXPECT_FALSE(parseVideoResolution("").has_value());
        EXPECT_FALSE(parseVideoResolution("1920").has_value());
        EXPECT_FALSE(parseVideoResolution("1920x").has_value());
        EXPECT_FALSE(parseVideoResolution("x1080").has_value());
        EXPECT_FALSE(parseVideoResolution("1920x1080x30").has_value());
        EXPECT_FALSE(parseVideoResolution("abc").has_value());
        EXPECT_FALSE(parseVideoResolution("-1920x1080").has_value());
        EXPECT_FALSE(parseVideoResolution("0x0").has_value());
        EXPECT_FALSE(parseVideoResolution("1920 x").has_value());
    }

} // namespace
