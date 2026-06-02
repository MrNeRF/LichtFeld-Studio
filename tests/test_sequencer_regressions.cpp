/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "sequencer/animation_clip.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/timeline.hpp"
#include "sequencer/timeline_view_math.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

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

} // namespace
