/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/services.hpp"
#include "gui/film_strip_renderer.hpp"
#include "gui/gui_manager.hpp"
#include "io/video/video_export_options.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "scene/scene_manager.hpp"

#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/animation_clip.hpp"
#include "sequencer/keyframe.hpp"
#include "sequencer/rml_sequencer_panel.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/timeline.hpp"
#include "sequencer/timeline_view_math.hpp"
#include "visualizer_impl.hpp"

#include <RmlUi/Core.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/RenderInterface.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <regex>

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

    class SequencerHistoryRegressionTest : public ::testing::Test {
    protected:
        void SetUp() override {
            if (const char* previous = std::getenv("LFS_HOME"))
                previous_home_ = previous;
            setHome(temporary_.path.string().c_str());
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            ASSERT_TRUE(lfs::event::LocalizationManager::getInstance().initialize(
                (std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui/resources/locales").string()));
        }

        void TearDown() override {
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            lfs::core::event::bus().clear_all();
            lfs::event::EventBridge::instance().clear_all();
            lfs::event::LocalizationManager::getInstance().reset();
            setHome(previous_home_ ? previous_home_->c_str() : nullptr);
        }

        static void setHome(const char* value) {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", value ? value : "");
#else
            if (value)
                (void)setenv("LFS_HOME", value, 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

        lfs::vis::ViewerOptions options() const {
            lfs::vis::ViewerOptions result;
            result.show_startup_overlay = false;
            result.project_lifecycle_settings_path = temporary_.path / "lifecycle.json";
            return result;
        }

        lfs::test::licht::TemporaryDirectory temporary_{"sequencer-history"};
        std::optional<std::string> previous_home_;
    };

    TEST_F(SequencerHistoryRegressionTest, ClearKeyframesPreservesPlySequence) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencer();
        controller.setPlySequence("frames", "sequence",
                                  {"frame_0.ply", "frame_1.ply", "frame_2.ply", "frame_3.ply"},
                                  {"frame_0", "frame_1", "frame_2", "frame_3"}, 5.0f);
        const auto* sequence = controller.plySequence();
        ASSERT_NE(sequence, nullptr);
        const auto* frames = sequence->frames.data();
        for (int repeat = 0; repeat < 2; ++repeat) {
            const auto id = controller.addKeyframe(makeKeyframe(0.0f));
            controller.addKeyframe(makeKeyframe(1.0f));
            controller.selectKeyframeById(id);
            controller.play();
            lfs::python::clear_keyframes();
            EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
            EXPECT_FALSE(controller.hasSelection());
            EXPECT_FALSE(controller.isPlaying());
            ASSERT_TRUE(controller.hasPlySequence());
            EXPECT_EQ(controller.plySequence(), sequence);
            EXPECT_EQ(controller.plySequence()->frames.data(), frames);
            EXPECT_EQ(sequence->directory, "frames");
            EXPECT_EQ(sequence->node_name, "sequence");
            EXPECT_EQ(sequence->frames.size(), 4u);
            EXPECT_FLOAT_EQ(controller.plySequenceFps(), 5.0f);
            EXPECT_FLOAT_EQ(controller.timeline().clipDuration(), 0.8f);
            controller.seek(0.6f);
            EXPECT_EQ(controller.currentPlySequenceFrameIndex(), 3u);
            controller.play();
            EXPECT_TRUE(controller.isPlaying());
        }
    }

    TEST_F(SequencerHistoryRegressionTest, ClearKeyframesLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencer();
        std::vector<double> samples;
        for (int repeat = 0; repeat < 101; ++repeat) {
            controller.setPlySequence("frames", "sequence", {"frame_0.ply", "frame_1.ply"},
                                      {"frame_0", "frame_1"}, 5.0f);
            controller.addKeyframe(makeKeyframe(0.0f));
            controller.addKeyframe(makeKeyframe(1.0f));
            const auto start = std::chrono::steady_clock::now();
            lfs::python::clear_keyframes();
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
            ASSERT_EQ(controller.timeline().realKeyframeCount(), 0u);
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_clear_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, ClearKeyframesWithoutSequenceKeepsResetBehavior) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencer();
        const auto id = controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(2.0f));
        controller.timeline().ensureAnimationClip();
        controller.selectKeyframeById(id);
        controller.play();
        lfs::python::clear_keyframes();
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
        EXPECT_FALSE(controller.timeline().hasAnimationClip());
        EXPECT_FALSE(controller.hasPlySequence());
        EXPECT_FALSE(controller.hasSelection());
        EXPECT_FALSE(controller.isPlaying());
        EXPECT_FLOAT_EQ(controller.timeline().clipDuration(), lfs::sequencer::DEFAULT_CLIP_DURATION_SECONDS);
    }

    TEST_F(SequencerHistoryRegressionTest, SceneClearStillDropsKeysAndPlySequence) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencer();
        controller.setPlySequence("frames", "sequence", {"frame_0.ply", "frame_1.ply"},
                                  {"frame_0", "frame_1"}, 5.0f);
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.play();
        lfs::core::events::state::SceneCleared{}.emit();
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
        EXPECT_FALSE(controller.hasPlySequence());
        EXPECT_FALSE(controller.isPlaying());
        EXPECT_FLOAT_EQ(controller.timeline().clipDuration(), lfs::sequencer::DEFAULT_CLIP_DURATION_SECONDS);
    }

    TEST_F(SequencerHistoryRegressionTest, AddCommandParticipatesInSharedUndoHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        ASSERT_EQ(controller.timeline().realKeyframeCount(), 0u);
        viewer.getViewport().camera.R = glm::mat3_cast(glm::angleAxis(
            0.73f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f))));
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
        ASSERT_EQ(controller.timeline().realKeyframeCount(), 1u);
        const auto added = *controller.timeline().getKeyframe(0);
        ASSERT_TRUE(history.canUndo());
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
        ASSERT_TRUE(history.redo().success);
        ASSERT_EQ(controller.timeline().realKeyframeCount(), 1u);
        const auto restored = *controller.timeline().getKeyframe(0);
        EXPECT_EQ(restored.id, added.id);
        EXPECT_EQ(restored.time, added.time);
        EXPECT_EQ(restored.position, added.position);
        EXPECT_EQ(restored.rotation, added.rotation);
        EXPECT_EQ(restored.focal_length_mm, added.focal_length_mm);
    }

    TEST_F(SequencerHistoryRegressionTest, DeleteCommandRestoresExactKeyAndPreservesOtherTimelineContent) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto first = controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}));
        auto key = makeKeyframe(4.5f, {3.0f, 5.0f, 7.0f}, 73.0f);
        key.rotation = glm::angleAxis(0.73f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
        key.easing = EasingType::EASE_OUT;
        const auto deleted = controller.addKeyframe(key);
        const auto captured = *controller.timeline().getKeyframeById(deleted);
        const auto other = controller.addKeyframe(makeKeyframe(8.0f));
        auto* clip = &controller.timeline().ensureAnimationClip();
        controller.setLoopMode(LoopMode::LOOP);
        const float duration = controller.clipDuration();
        history.clear();
        lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = 1}.emit();
        ASSERT_EQ(controller.timeline().getKeyframeById(deleted), nullptr);
        ASSERT_EQ(history.undoCount(), 1u);
        EXPECT_EQ(history.undoName(), "Delete Keyframe");
        for (int repetition = 0; repetition < 2; ++repetition) {
            ASSERT_TRUE(history.undo().success);
            const auto* restored = controller.timeline().getKeyframeById(deleted);
            ASSERT_NE(restored, nullptr);
            EXPECT_EQ(restored->id, captured.id);
            EXPECT_EQ(restored->time, captured.time);
            EXPECT_EQ(restored->position, captured.position);
            EXPECT_EQ(restored->rotation, captured.rotation);
            EXPECT_EQ(restored->focal_length_mm, captured.focal_length_mm);
            EXPECT_EQ(restored->easing, captured.easing);
            EXPECT_EQ(controller.timeline().animationClip(), clip);
            EXPECT_EQ(controller.loopMode(), LoopMode::LOOP);
            EXPECT_EQ(controller.clipDuration(), duration);
            EXPECT_NE(controller.timeline().getKeyframeById(first), nullptr);
            EXPECT_NE(controller.timeline().getKeyframeById(other), nullptr);
            EXPECT_EQ(history.undoCount(), 0u);
            EXPECT_EQ(history.redoCount(), 1u);
            ASSERT_TRUE(history.redo().success);
            EXPECT_EQ(controller.timeline().getKeyframeById(deleted), nullptr);
            EXPECT_EQ(history.undoCount(), 1u);
            EXPECT_EQ(history.redoCount(), 0u);
        }
    }

    TEST_F(SequencerHistoryRegressionTest, DeleteUndoPrecedesEarlierSceneVisibilityChange) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        const auto deleted = controller.addKeyframe(makeKeyframe(1.0f));
        auto* manager = viewer.getSceneManager();
        const auto group = manager->getScene().addGroup("group");
        manager->setNodeVisibility(group, false);
        ASSERT_EQ(history.undoName(), "Set Visibility");
        lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = 1}.emit();
        ASSERT_TRUE(history.undo().success);
        EXPECT_FALSE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
        EXPECT_NE(controller.timeline().getKeyframeById(deleted), nullptr);
        ASSERT_TRUE(history.undo().success);
        EXPECT_TRUE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
        ASSERT_TRUE(history.redo().success);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().getKeyframeById(deleted), nullptr);
    }

    TEST_F(SequencerHistoryRegressionTest, ProtectedAndMissingDeleteCommandsDoNotCreateHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        const auto first = controller.addKeyframe(makeKeyframe(0.0f));
        auto& history = lfs::vis::op::undoHistory();
        lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = 0}.emit();
        lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = 99}.emit();
        EXPECT_NE(controller.timeline().getKeyframeById(first), nullptr);
        EXPECT_EQ(history.undoCount(), 0u);
    }

    TEST_F(SequencerHistoryRegressionTest, GroupedControllerDeletionIsOneUndoStep) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        const auto second = controller.addKeyframe(makeKeyframe(1.0f));
        const auto third = controller.addKeyframe(makeKeyframe(2.0f));
        {
            lfs::vis::op::TransactionGuard transaction("Delete Keyframes");
            ASSERT_TRUE(controller.removeKeyframeById(second));
            ASSERT_TRUE(controller.removeKeyframeById(third));
            transaction.commit();
        }
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_NE(controller.timeline().getKeyframeById(second), nullptr);
        EXPECT_NE(controller.timeline().getKeyframeById(third), nullptr);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 1u);
        EXPECT_EQ(history.undoCount(), 1u);
    }

    TEST_F(SequencerHistoryRegressionTest, DeleteHistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            auto& controller = viewer.getGuiManager()->sequencerUI().controller();
            controller.addKeyframe(makeKeyframe(0.0f));
            controller.addKeyframe(makeKeyframe(1.0f));
            lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = 1}.emit();
            ASSERT_EQ(history.undoCount(), 1u);
        }
        EXPECT_FALSE(history.undo().success);
        EXPECT_EQ(history.undoCount(), 0u);
    }

    TEST_F(SequencerHistoryRegressionTest, DeleteCommandLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        std::vector<double> samples;
        for (int batch = 0; batch < 9; ++batch) {
            controller.clear();
            history.clear();
            for (int i = 0; i < 51; ++i)
                controller.addKeyframe(makeKeyframe(static_cast<float>(i)));
            const auto start = std::chrono::steady_clock::now();
            for (int i = 50; i > 0; --i)
                lfs::core::events::cmd::SequencerDeleteKeyframe{.keyframe_index = static_cast<size_t>(i)}.emit();
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              50.0);
            ASSERT_EQ(controller.timeline().realKeyframeCount(), 1u);
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_delete_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeEditUsesSharedHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 0.0f}.emit();
        const auto id = controller.addKeyframe(makeKeyframe(1.0f, {1.0f, 2.0f, 3.0f}));
        const auto before = *controller.timeline().getKeyframeById(id);
        ASSERT_TRUE(controller.setKeyframeTime(1, 6.5f));
        ASSERT_EQ(history.undoCount(), 2u);
        ASSERT_TRUE(history.undo().success);
        ASSERT_NE(controller.timeline().getKeyframeById(id), nullptr);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->time, before.time);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->position, before.position);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 6.5f);
        ASSERT_TRUE(history.undo().success);
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 1u);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 1.0f);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeDragCommitsOneHistoryEntry) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        const auto id = controller.addKeyframe(makeKeyframe(1.0f));
        controller.setLoopMode(LoopMode::LOOP);
        for (const float time : {2.0f, 3.0f, 4.0f}) {
            ASSERT_TRUE(controller.previewKeyframeTimeById(id, time));
            EXPECT_FALSE(history.canUndo());
        }
        ASSERT_TRUE(controller.commitKeyframeTimeById(id));
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 1.0f);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 30.0f);
        EXPECT_FLOAT_EQ(controller.timeline().keyframes().back().time, 30.0f);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 4.0f);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 30.0f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeNoOpsDoNotRecordHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto id = controller.addKeyframe(makeKeyframe(1.0f));
        (void)controller.setKeyframeTimeById(id, 1.0f);
        EXPECT_FALSE(controller.setKeyframeTimeById(id, std::numeric_limits<float>::infinity()));
        EXPECT_FALSE(controller.previewKeyframeTimeById(id, std::numeric_limits<float>::quiet_NaN()));
        ASSERT_TRUE(controller.previewKeyframeTimeById(id, 2.0f));
        ASSERT_TRUE(controller.previewKeyframeTimeById(id, 1.0f));
        ASSERT_TRUE(controller.commitKeyframeTimeById(id));
        ASSERT_TRUE(controller.commitKeyframeTimeById(id));
        EXPECT_FALSE(history.canUndo());
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 1.0f);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeHistoryPreservesOtherEdits) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto first_id = controller.addKeyframe(makeKeyframe(0.0f));
        const auto id = controller.addKeyframe(makeKeyframe(1.0f));
        controller.setLoopMode(LoopMode::LOOP);
        auto* const clip = &controller.timeline().ensureAnimationClip();
        ASSERT_TRUE(controller.setKeyframeTimeById(id, 3.0f));
        ASSERT_TRUE(controller.updateKeyframeById(id, {4.0f, 5.0f, 6.0f}, lfs::sequencer::IDENTITY_ROTATION, 60.0f));
        controller.selectKeyframeById(first_id);
        controller.setClipDuration(80.0f);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 1.0f);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->position, glm::vec3(4.0f, 5.0f, 6.0f));
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->focal_length_mm, 60.0f);
        EXPECT_EQ(controller.selectedKeyframeId(), first_id);
        EXPECT_EQ(controller.timeline().animationClip(), clip);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 80.0f);
        EXPECT_FLOAT_EQ(controller.timeline().keyframes().back().time, 80.0f);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 3.0f);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->position, glm::vec3(4.0f, 5.0f, 6.0f));
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeHistoryPreservesNegativeTimeAndReordering) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}));
        const auto id = controller.addKeyframe(makeKeyframe(1.0f, {4.0f, 5.0f, 6.0f}));
        const auto before = controller.saveToJson();
        ASSERT_TRUE(controller.setKeyframeTimeById(id, -1.0f));
        EXPECT_EQ(controller.timeline().getKeyframe(0)->id, id);
        const auto edited = controller.saveToJson();
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.saveToJson(), before);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.saveToJson(), edited);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeHistoryRejectsReloadedTimeline) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto id = controller.addKeyframe(makeKeyframe(1.0f));
        ASSERT_TRUE(controller.setKeyframeTimeById(id, 3.0f));
        const auto replacement = controller.saveToJson();
        ASSERT_TRUE(controller.loadFromJson(replacement));
        ASSERT_TRUE(history.canUndo());
        EXPECT_FALSE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.timeline().getKeyframeById(id)->time, 3.0f);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimeHistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            auto& controller = viewer.getGuiManager()->sequencerUI().controller();
            const auto id = controller.addKeyframe(makeKeyframe(1.0f));
            ASSERT_TRUE(controller.setKeyframeTimeById(id, 3.0f));
            ASSERT_TRUE(history.canUndo());
        }
        EXPECT_FALSE(history.undo().success);
    }

    TEST_F(SequencerHistoryRegressionTest, KeyframeTimePreviewLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        controller.addKeyframe(makeKeyframe(0.0f));
        const auto id = controller.addKeyframe(makeKeyframe(1.0f));
        std::vector<double> samples;
        for (int batch = 0; batch < 11; ++batch) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10000; ++i)
                (void)controller.previewKeyframeTimeById(id, 1.0f + (i % 2));
            samples.push_back(std::chrono::duration<double, std::nano>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              10000.0);
            ASSERT_TRUE(controller.commitKeyframeTimeById(id));
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_preview_ns", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, EasingCommandParticipatesInSharedUndoHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}, 73.0f));
        const auto second = controller.addKeyframe(makeKeyframe(4.5f, {3.0f, 5.0f, 7.0f}, 45.0f));
        controller.selectKeyframeById(second);
        auto* clip = &controller.timeline().ensureAnimationClip();
        controller.setLoopMode(LoopMode::LOOP);
        const auto original = controller.saveToJson();
        for (size_t index = 0; index < 2; ++index) {
            for (int mode = 1; mode <= 3; ++mode) {
                history.clear();
                lfs::core::events::cmd::SequencerSetKeyframeEasing{.keyframe_index = index, .easing_type = mode}.emit();
                const auto edited = controller.saveToJson();
                ASSERT_NE(edited, original);
                ASSERT_EQ(history.undoCount(), 1u);
                EXPECT_EQ(history.undoName(), "Set Keyframe Easing");
                for (int repetition = 0; repetition < 2; ++repetition) {
                    ASSERT_TRUE(history.undo().success);
                    EXPECT_EQ(controller.saveToJson(), original);
                    EXPECT_EQ(history.undoCount(), 0u);
                    ASSERT_TRUE(history.redo().success);
                    EXPECT_EQ(controller.saveToJson(), edited);
                    EXPECT_EQ(history.undoCount(), 1u);
                }
                ASSERT_TRUE(history.undo().success);
                EXPECT_EQ(controller.timeline().animationClip(), clip);
                EXPECT_EQ(controller.selectedKeyframeId(), second);
                EXPECT_EQ(controller.loopMode(), LoopMode::LOOP);
            }
        }
    }

    TEST_F(SequencerHistoryRegressionTest, EasingUndoPrecedesEarlierVisibilityChange) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto key = controller.addKeyframe(makeKeyframe(0.0f));
        auto* manager = viewer.getSceneManager();
        const auto group = manager->getScene().addGroup("group");
        manager->setNodeVisibility(group, false);
        lfs::core::events::cmd::SequencerSetKeyframeEasing{.keyframe_index = 0, .easing_type = 1}.emit();
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().getKeyframeById(key)->easing, EasingType::LINEAR);
        EXPECT_FALSE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
        ASSERT_TRUE(history.undo().success);
        EXPECT_TRUE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
        ASSERT_TRUE(history.redo().success);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().getKeyframeById(key)->easing, EasingType::EASE_IN);
    }

    TEST_F(SequencerHistoryRegressionTest, UnchangedAndInvalidEasingDoesNotCreateHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto key = controller.addKeyframe(makeKeyframe(0.0f));
        const auto original = controller.saveToJson();
        EXPECT_TRUE(controller.setKeyframeEasingById(key, EasingType::LINEAR));
        EXPECT_FALSE(controller.setKeyframeEasingById(key, static_cast<EasingType>(99)));
        EXPECT_FALSE(controller.setKeyframeEasingById(99999, EasingType::EASE_IN));
        EXPECT_EQ(controller.saveToJson(), original);
        EXPECT_EQ(history.undoCount(), 0u);
    }

    TEST_F(SequencerHistoryRegressionTest, EasingHistoryCannotOverwriteLaterKeyEdit) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto key = controller.addKeyframe(makeKeyframe(0.0f));
        ASSERT_TRUE(controller.setKeyframeEasingById(key, EasingType::EASE_IN));
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(controller.setKeyframeFocalLengthById(key, 73.0f));
        const auto edited = controller.saveToJson();
        EXPECT_FALSE(history.undo().success);
        EXPECT_EQ(controller.saveToJson(), edited);
    }

    TEST_F(SequencerHistoryRegressionTest, EasingHistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            auto& controller = viewer.getGuiManager()->sequencerUI().controller();
            const auto key = controller.addKeyframe(makeKeyframe(0.0f));
            ASSERT_TRUE(controller.setKeyframeEasingById(key, EasingType::EASE_IN));
            ASSERT_EQ(history.undoCount(), 1u);
        }
        EXPECT_FALSE(history.undo().success);
        EXPECT_EQ(history.undoCount(), 0u);
    }

    TEST_F(SequencerHistoryRegressionTest, EasingCommandLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        std::vector<double> samples;
        for (int batch = 0; batch < 9; ++batch) {
            history.clear();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i)
                lfs::core::events::cmd::SequencerSetKeyframeEasing{.keyframe_index = 0, .easing_type = 1 + i % 2}.emit();
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              50.0);
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_easing_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationEditUsesSharedHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}));
        controller.addKeyframe(makeKeyframe(1.0f, {4.0f, 5.0f, 6.0f}));
        controller.setClipDuration(1.0f);
        const auto keys = controller.saveToJson()["keyframes"];
        controller.editClipDuration(2.0f);
        ASSERT_FLOAT_EQ(controller.clipDuration(), 2.0f);
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 2.0f);
        controller.editClipDuration(3.0f);
        ASSERT_EQ(history.undoCount(), 2u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 2.0f);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        ASSERT_TRUE(history.redo().success);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
        EXPECT_EQ(controller.saveToJson()["keyframes"], keys);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationEditFollowsEarlierHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 0.0f}.emit();
        controller.addKeyframe(makeKeyframe(1.0f));
        controller.setClipDuration(1.0f);
        controller.editClipDuration(3.0f);
        ASSERT_EQ(history.undoCount(), 2u);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
        EXPECT_EQ(history.undoName(), "Add Keyframe");
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationEditKeepsClampLoopAndNoOpBehavior) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        controller.setClipDuration(3.0f);
        controller.setLoopMode(LoopMode::LOOP);
        controller.seek(2.5f);
        controller.editClipDuration(3.0f);
        EXPECT_FALSE(history.canUndo());
        EXPECT_THROW(controller.editClipDuration(std::numeric_limits<float>::infinity()), std::invalid_argument);
        EXPECT_FALSE(history.canUndo());
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
        controller.editClipDuration(0.25f);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        EXPECT_FLOAT_EQ(controller.playhead(), 1.0f);
        EXPECT_FLOAT_EQ(controller.timeline().keyframes().back().time, 1.0f);
        controller.editClipDuration(0.25f);
        ASSERT_EQ(history.undoCount(), 1u);
        controller.seek(0.5f);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
        EXPECT_FLOAT_EQ(controller.playhead(), 0.5f);
        EXPECT_FLOAT_EQ(controller.timeline().keyframes().back().time, 3.0f);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        EXPECT_FLOAT_EQ(controller.playhead(), 0.5f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationHistoryPreservesOtherTimelineContent) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto id = controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        controller.setPlySequence("frames", "Sequence", {"frame0.ply", "frame1.ply"}, {"Frame 0", "Frame 1"}, 2.0f);
        controller.setClipDuration(1.0f);
        const auto* sequence = controller.plySequence();
        const auto* frames = sequence->frames.data();
        auto* clip = &controller.timeline().ensureAnimationClip();
        controller.editClipDuration(3.0f);
        controller.selectKeyframeById(id);
        controller.updateKeyframeById(id, {4.0f, 5.0f, 6.0f}, lfs::sequencer::IDENTITY_ROTATION, 60.0f);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 1.0f);
        EXPECT_EQ(controller.plySequence(), sequence);
        EXPECT_EQ(controller.plySequence()->frames.data(), frames);
        EXPECT_EQ(controller.timeline().animationClip(), clip);
        EXPECT_EQ(controller.selectedKeyframeId(), id);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->position, glm::vec3(4.0f, 5.0f, 6.0f));
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
        EXPECT_EQ(controller.plySequence()->frames.data(), frames);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationHistoryRejectsReloadAndLaterConflictingKeys) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        controller.setClipDuration(1.0f);
        controller.editClipDuration(3.0f);
        controller.addKeyframe(makeKeyframe(2.0f));
        ASSERT_TRUE(history.canUndo());
        EXPECT_FALSE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 3.0f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 3u);
        history.clear();
        controller.editClipDuration(4.0f);
        const auto replacement = controller.saveToJson();
        ASSERT_TRUE(controller.loadFromJson(replacement));
        ASSERT_TRUE(history.canUndo());
        EXPECT_FALSE(history.undo().success);
        EXPECT_EQ(controller.saveToJson(), replacement);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationHistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            viewer.getGuiManager()->sequencerUI().controller().editClipDuration(40.0f);
            ASSERT_TRUE(history.canUndo());
        }
        EXPECT_FALSE(history.undo().success);
    }

    TEST_F(SequencerHistoryRegressionTest, ClipDurationEditLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        std::vector<double> samples;
        for (int i = 0; i < 101; ++i) {
            history.clear();
            controller.setClipDuration(1.0f);
            const auto start = std::chrono::steady_clock::now();
            controller.editClipDuration(2.0f);
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count());
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_duration_edit_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, CurrentViewUpdateRestoresExactPoseAndFocalLength) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}, 41.0f));
        controller.addKeyframe(makeKeyframe(4.5f, {3.0f, 5.0f, 7.0f}, 73.0f));
        controller.setKeyframeEasing(0, EasingType::EASE_IN);
        controller.setKeyframeEasing(1, EasingType::EASE_OUT);
        auto* clip = &controller.timeline().ensureAnimationClip();
        controller.setLoopMode(LoopMode::LOOP);
        const auto original = controller.saveToJson();
        for (size_t index = 0; index < 2; ++index) {
            history.clear();
            ASSERT_TRUE(controller.selectKeyframe(index));
            const auto id = *controller.selectedKeyframeId();
            auto& camera = viewer.getViewport().camera;
            camera.t = {9.0f, 8.0f, 7.0f};
            camera.R = glm::mat3_cast(glm::angleAxis(0.73f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f))));
            viewer.getRenderingManager()->setFocalLength(63.0f);
            lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
            const auto edited = controller.saveToJson();
            ASSERT_NE(edited, original);
            const auto* updated = controller.timeline().getKeyframeById(id);
            ASSERT_NE(updated, nullptr);
            EXPECT_EQ(updated->position, camera.t);
            EXPECT_EQ(updated->rotation, glm::normalize(glm::quat_cast(camera.R)));
            EXPECT_EQ(updated->focal_length_mm, 63.0f);
            ASSERT_EQ(history.undoCount(), 1u);
            EXPECT_EQ(history.undoName(), "Update Keyframe");
            lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
            EXPECT_EQ(history.undoCount(), 1u);
            for (int repetition = 0; repetition < 2; ++repetition) {
                ASSERT_TRUE(history.undo().success);
                EXPECT_EQ(controller.saveToJson(), original);
                EXPECT_EQ(history.undoCount(), 0u);
                ASSERT_TRUE(history.redo().success);
                EXPECT_EQ(controller.saveToJson(), edited);
                EXPECT_EQ(history.undoCount(), 1u);
            }
            ASSERT_TRUE(history.undo().success);
            EXPECT_EQ(controller.timeline().animationClip(), clip);
            EXPECT_EQ(controller.selectedKeyframeId(), id);
            EXPECT_EQ(controller.loopMode(), LoopMode::LOOP);
            EXPECT_EQ(camera.t, glm::vec3(9.0f, 8.0f, 7.0f));
        }
    }

    TEST_F(SequencerHistoryRegressionTest, CurrentViewUpdateUndoPrecedesVisibilityChange) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        ASSERT_TRUE(controller.selectKeyframe(1));
        const auto original = controller.saveToJson();
        auto* manager = viewer.getSceneManager();
        const auto group = manager->getScene().addGroup("group");
        manager->setNodeVisibility(group, false);
        viewer.getViewport().camera.t = {7.0f, 8.0f, 9.0f};
        lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.saveToJson(), original);
        EXPECT_FALSE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
        ASSERT_TRUE(history.undo().success);
        EXPECT_TRUE(static_cast<bool>(manager->getScene().getNodeById(group)->visible));
    }

    TEST_F(SequencerHistoryRegressionTest, UnchangedInvalidAndUnselectedViewUpdatesDoNotCreateHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        auto& camera = viewer.getViewport().camera;
        auto key = makeKeyframe(0.0f, camera.t, viewer.getRenderingManager()->getFocalLengthMm());
        key.rotation = glm::quat_cast(camera.R);
        controller.addKeyframe(key);
        ASSERT_TRUE(controller.selectKeyframe(0));
        const auto original = controller.saveToJson();
        lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
        EXPECT_EQ(history.undoCount(), 0u);
        camera.t.x = std::numeric_limits<float>::quiet_NaN();
        lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
        controller.deselectKeyframe();
        camera.t = {9.0f, 8.0f, 7.0f};
        lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
        EXPECT_EQ(controller.saveToJson(), original);
        EXPECT_EQ(history.undoCount(), 0u);
    }

    TEST_F(SequencerHistoryRegressionTest, CurrentViewHistoryCannotOverwriteLaterKeyEdit) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto key = controller.addKeyframe(makeKeyframe(0.0f));
        ASSERT_TRUE(controller.selectKeyframe(0));
        viewer.getViewport().camera.t = {7.0f, 8.0f, 9.0f};
        lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
        ASSERT_EQ(history.undoCount(), 1u);
        ASSERT_TRUE(controller.setKeyframeFocalLengthById(key, 73.0f));
        const auto edited = controller.saveToJson();
        EXPECT_FALSE(history.undo().success);
        EXPECT_EQ(controller.saveToJson(), edited);
    }

    TEST_F(SequencerHistoryRegressionTest, CurrentViewHistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            auto& controller = viewer.getGuiManager()->sequencerUI().controller();
            controller.addKeyframe(makeKeyframe(0.0f));
            ASSERT_TRUE(controller.selectKeyframe(0));
            viewer.getViewport().camera.t = {7.0f, 8.0f, 9.0f};
            lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
            ASSERT_EQ(history.undoCount(), 1u);
        }
        EXPECT_FALSE(history.undo().success);
    }

    TEST_F(SequencerHistoryRegressionTest, CurrentViewUpdateLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(1.0f));
        ASSERT_TRUE(controller.selectKeyframe(1));
        std::vector<double> samples;
        for (int batch = 0; batch < 9; ++batch) {
            history.clear();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i) {
                viewer.getViewport().camera.t.x = static_cast<float>(i);
                lfs::core::events::cmd::SequencerUpdateKeyframe{}.emit();
            }
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              50.0);
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_update_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, AddCommandLatency) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        std::vector<double> samples;
        for (int batch = 0; batch < 9; ++batch) {
            controller.clear();
            history.clear();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i)
                lfs::core::events::cmd::SequencerAddKeyframe{.time = static_cast<float>(i)}.emit();
            samples.push_back(std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              50.0);
            ASSERT_EQ(controller.timeline().realKeyframeCount(), 50u);
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_add_us", std::to_string(samples[samples.size() / 2]));
    }

    TEST_F(SequencerHistoryRegressionTest, ReplacementRestoresExactKeyAndDoesNotRecordNoOp) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        viewer.getViewport().camera.t = {1.0f, 2.0f, 3.0f};
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
        const auto first = *controller.timeline().getKeyframe(0);
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
        EXPECT_EQ(history.undoCount(), 1u);
        viewer.getViewport().camera.t = {4.0f, 5.0f, 6.0f};
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
        const auto replaced = *controller.timeline().getKeyframe(0);
        ASSERT_EQ(replaced.id, first.id);
        ASSERT_EQ(history.undoCount(), 2u);
        ASSERT_TRUE(history.undo().success);
        const auto restored = *controller.timeline().getKeyframeById(first.id);
        EXPECT_EQ(restored.position, first.position);
        EXPECT_EQ(restored.rotation, first.rotation);
        EXPECT_EQ(restored.focal_length_mm, first.focal_length_mm);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().getKeyframeById(first.id)->position, replaced.position);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 1u);
    }

    TEST_F(SequencerHistoryRegressionTest, AdditionPreservesOtherKeysTracksAndLoopOnUndo) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        const auto first_id = controller.addKeyframe(makeKeyframe(0.0f, {1.0f, 2.0f, 3.0f}));
        controller.setLoopMode(LoopMode::LOOP);
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 40.0f}.emit();
        ASSERT_FLOAT_EQ(controller.clipDuration(), 40.0f);
        const auto other_id = controller.addKeyframe(makeKeyframe(5.0f, {3.0f, 2.0f, 1.0f}));
        auto* const clip = &controller.timeline().ensureAnimationClip();
        controller.selectKeyframeById(other_id);
        ASSERT_TRUE(history.undo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 30.0f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 2u);
        EXPECT_NE(controller.timeline().getKeyframeById(first_id), nullptr);
        EXPECT_NE(controller.timeline().getKeyframeById(other_id), nullptr);
        EXPECT_EQ(controller.selectedKeyframeId(), other_id);
        EXPECT_EQ(controller.timeline().animationClip(), clip);
        EXPECT_EQ(controller.loopMode(), LoopMode::LOOP);
        ASSERT_TRUE(history.redo().success);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 40.0f);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 3u);
        EXPECT_EQ(controller.timeline().animationClip(), clip);
        EXPECT_EQ(controller.selectedKeyframeId(), other_id);
    }

    TEST_F(SequencerHistoryRegressionTest, LaterKeyEditCannotBeOverwrittenByStaleHistory) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
        const auto id = controller.timeline().getKeyframe(0)->id;
        controller.updateKeyframeById(id, {7.0f, 8.0f, 9.0f}, lfs::sequencer::IDENTITY_ROTATION, 60.0f);
        ASSERT_TRUE(history.canUndo());
        EXPECT_FALSE(history.undo().success);
        ASSERT_NE(controller.timeline().getKeyframeById(id), nullptr);
        EXPECT_EQ(controller.timeline().getKeyframeById(id)->position, glm::vec3(7.0f, 8.0f, 9.0f));
    }

    TEST_F(SequencerHistoryRegressionTest, LaterClipDurationEditIsPreserved) {
        lfs::vis::VisualizerImpl viewer(options());
        auto& controller = viewer.getGuiManager()->sequencerUI().controller();
        auto& history = lfs::vis::op::undoHistory();
        lfs::core::events::cmd::SequencerAddKeyframe{.time = 40.0f}.emit();
        controller.setClipDuration(80.0f);
        ASSERT_TRUE(history.undo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 0u);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 80.0f);
        ASSERT_TRUE(history.redo().success);
        EXPECT_EQ(controller.timeline().realKeyframeCount(), 1u);
        EXPECT_FLOAT_EQ(controller.clipDuration(), 80.0f);
    }

    TEST_F(SequencerHistoryRegressionTest, HistoryCannotAccessDestroyedSequencer) {
        auto& history = lfs::vis::op::undoHistory();
        {
            lfs::vis::VisualizerImpl viewer(options());
            lfs::core::events::cmd::SequencerAddKeyframe{.time = 2.5f}.emit();
            ASSERT_TRUE(history.canUndo());
        }
        EXPECT_FALSE(history.undo().success);
    }

    TEST(SequencerTimelineRegressionTest, ExportCameraPreservesScreenCornersAcrossPoses) {
        // Independent screen-space oracle: right stays right, up maps to smaller
        // image rows, and visible points have positive depth in a dataset Camera.
        const std::array<glm::vec3, 4> eyes{{{0, 0, 5}, {2, 4, 1}, {-2, -4, 1}, {1, 2, 5}}};
        const std::array<glm::vec3, 4> ups{{{0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {1, 0, 0}}};
        for (size_t pose = 0; pose < eyes.size(); ++pose) {
            SCOPED_TRACE(pose);
            const auto rotation = lfs::rendering::tryMakeVisualizerLookAtRotation(
                eyes[pose], glm::vec3(0), ups[pose]);
            ASSERT_TRUE(rotation.has_value());
            Timeline timeline;
            auto keyframe = makeKeyframe(0, eyes[pose]);
            keyframe.rotation = glm::quat_cast(*rotation);
            timeline.addKeyframe(keyframe);
            const auto camera = timeline.evaluate(0);
            const auto view = lfs::rendering::dataWorldToCameraFromVisualizerPose(
                glm::mat3_cast(camera.rotation), camera.position);
            for (const float x : {-1.0f, 1.0f}) {
                for (const float y : {-1.0f, 1.0f}) {
                    const auto visualizer_point = eyes[pose] + *rotation * glm::vec3(x, y, -3);
                    // Raw PLY world uses the opposite Y/Z axes to the visualizer.
                    const glm::vec3 data_point(visualizer_point.x, -visualizer_point.y, -visualizer_point.z);
                    const auto projected = glm::vec3(view * glm::vec4(data_point, 1));
                    EXPECT_NEAR(projected.x, x, 1e-5f);
                    EXPECT_NEAR(projected.y, -y, 1e-5f);
                    EXPECT_NEAR(projected.z, 3, 1e-5f);
                }
            }
        }
    }

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

    TEST(SequencerControllerRegressionTest, ClearResetsPlayheadAfterRemovingContent) {
        for (const auto mode : {LoopMode::ONCE, LoopMode::LOOP, LoopMode::PING_PONG}) {
            for (const float first_time : {0.0f, 5.0f, 59.0f}) {
                for (const bool playing : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << "first=" << first_time << " playing=" << playing
                                                      << " loop=" << static_cast<int>(mode));
                    SequencerController controller;
                    const auto id = controller.addKeyframe(makeKeyframe(first_time));
                    controller.addKeyframe(makeKeyframe(first_time + 2.0f));
                    controller.setLoopMode(mode);
                    controller.setPlaybackSpeed(2.0f);
                    ASSERT_TRUE(controller.selectKeyframeById(id));
                    if (playing)
                        controller.play();
                    controller.seek(first_time + 1.0f);
                    const auto revision = controller.timelineRevision();

                    controller.clear();

                    EXPECT_FLOAT_EQ(controller.playhead(), 0.0f);
                    EXPECT_FLOAT_EQ(controller.clipDuration(), 30.0f);
                    EXPECT_TRUE(controller.isStopped());
                    EXPECT_FALSE(controller.hasPlayableContent());
                    EXPECT_FALSE(controller.hasSelection());
                    EXPECT_EQ(controller.timelineRevision(), revision + 1);
                    EXPECT_EQ(controller.loopMode(), mode);
                    EXPECT_FLOAT_EQ(controller.playbackSpeed(), 2.0f);
                    controller.clear();
                    EXPECT_FLOAT_EQ(controller.playhead(), 0.0f);
                }
            }
        }
    }

    TEST(SequencerControllerRegressionTest, ClearAlsoResetsEmptyAndPlyTimelines) {
        SequencerController controller;
        controller.seek(25.0f);
        controller.clear();
        EXPECT_FLOAT_EQ(controller.playhead(), 0.0f);
        controller.setPlySequence({}, "sequence", {"frame.ply"}, {"frame"}, 24.0f);
        controller.addKeyframe(makeKeyframe(59.0f));
        controller.play();
        controller.seek(59.0f);
        controller.clear();
        EXPECT_FALSE(controller.hasPlySequence());
        EXPECT_FALSE(controller.hasPlayableContent());
        EXPECT_TRUE(controller.isStopped());
        EXPECT_FLOAT_EQ(controller.playhead(), 0.0f);
    }

    TEST(SequencerControllerRegressionTest, ClearLatency) {
        std::vector<double> samples;
        std::vector<SequencerController> controllers(1000);
        for (int batch = 0; batch < 11; ++batch) {
            for (auto& controller : controllers) {
                controller.addKeyframe(makeKeyframe(59.0f));
                controller.seek(59.0f);
            }
            const auto start = std::chrono::steady_clock::now();
            for (auto& controller : controllers)
                controller.clear();
            samples.push_back(std::chrono::duration<double, std::nano>(
                                  std::chrono::steady_clock::now() - start)
                                  .count() /
                              controllers.size());
        }
        std::sort(samples.begin(), samples.end());
        RecordProperty("median_clear_ns", std::to_string(samples[samples.size() / 2]));
    }

    TEST(SequencerControllerRegressionTest, StopStillReturnsToFirstKeyframeWithoutClearing) {
        SequencerController controller;
        const auto id = controller.addKeyframe(makeKeyframe(59.0f));
        controller.addKeyframe(makeKeyframe(61.0f));
        ASSERT_TRUE(controller.selectKeyframeById(id));
        const auto saved = controller.saveToJson();
        const auto revision = controller.timelineRevision();
        controller.play();
        controller.seek(60.0f);

        controller.stop();

        EXPECT_FLOAT_EQ(controller.playhead(), 59.0f);
        EXPECT_TRUE(controller.isStopped());
        EXPECT_EQ(controller.selectedKeyframeId(), id);
        EXPECT_EQ(controller.timelineRevision(), revision);
        EXPECT_EQ(controller.saveToJson(), saved);
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

namespace lfs::vis {

    class SequencerMarkupRegressionTest : public ::testing::Test {
    protected:
        class CountingElement : public Rml::Element {
        public:
            explicit CountingElement(const Rml::String& tag) : Rml::Element(tag) {}
            void SetInnerRML(const Rml::String& markup) override {
                ++replacements;
                Rml::Element::SetInnerRML(markup);
            }
            size_t replacements = 0;
        };

        class StubRenderer final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = {16, 16};
                return 1;
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };

        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            Rml::Factory::RegisterElementInstancer("counted-sequencer", &instancer_);
        }
        static void TearDownTestSuite() { Rml::Shutdown(); }

        void SetUp() override {
            context_ = Rml::CreateContext("sequencer_markup", {1000, 300}, &renderer_);
            ASSERT_NE(context_, nullptr);
            panel_ = std::make_unique<RmlSequencerPanel>(controller_, ui_, &manager_);
            createDocument();
            for (int i = 0; i < 500; ++i) {
                sequencer::Keyframe keyframe;
                keyframe.time = static_cast<float>(i) * 0.1f;
                controller_.addKeyframeAtTime(keyframe, keyframe.time);
            }
            panel_->setFilmStripAttached(true);
            rebuild();
        }
        void TearDown() override {
            panel_.reset();
            ASSERT_TRUE(Rml::RemoveContext("sequencer_markup"));
        }

        void createDocument() {
            document_ = context_->LoadDocumentFromMemory("<rml><head/><body/></rml>");
            ASSERT_NE(document_, nullptr);
            for (auto& element : elements_) {
                auto owned = document_->CreateElement("counted-sequencer");
                element = static_cast<CountingElement*>(owned.get());
                document_->AppendChild(std::move(owned));
            }
            bindElements();
        }
        void bindElements() {
            panel_->document_ = document_;
            panel_->elements_cached_ = true;
            panel_->cached_panel_width_ = 1000.0f;
            panel_->el_film_strip_dividers_ = elements_[0];
            panel_->el_film_strip_sprockets_top_ = elements_[1];
            panel_->el_film_strip_sprockets_bottom_ = elements_[2];
            panel_->el_film_strip_gaps_ = elements_[3];
            panel_->el_film_strip_markers_ = elements_[4];
            panel_->el_easing_segments_ = elements_[5];
            panel_->el_easing_curves_ = elements_[6];
            panel_->el_easing_indicators_ = elements_[7];
        }
        void rebuild() {
            PanelInputState input;
            input.mouse_x = input.mouse_y = -1.0f;
            panel_->rebuildFilmStrip(0.0f, width_, 200.0f, input, nullptr, nullptr, film_strip_);
            panel_->rebuildEasingStripe(0.0f, width_);
        }
        void rebuildFresh() {
            panel_->clearElementCache();
            bindElements();
            rebuild();
        }
        auto markup() const {
            std::array<std::string, 8> result;
            for (size_t i = 0; i < elements_.size(); ++i)
                result[i] = elements_[i]->GetInnerRML();
            return result;
        }
        void expectFreshEquivalent() {
            rebuild();
            const auto cached = markup();
            rebuildFresh();
            EXPECT_EQ(markup(), cached);
        }
        size_t replacements() const {
            size_t result = 0;
            for (const auto* element : elements_)
                result += element->replacements;
            return result;
        }
        void resetCounters() {
            for (auto* element : elements_)
                element->replacements = 0;
        }
        void replaceDocument() {
            panel_->clearElementCache();
            context_->UnloadDocument(document_);
            context_->Update();
            createDocument();
        }
        void destroyGraphics() { panel_->destroyGraphicsResources(); }
        void hover(std::optional<size_t> index) { panel_->hovered_keyframe_ = index; }

        inline static StubRenderer renderer_;
        inline static Rml::ElementInstancerGeneric<CountingElement> instancer_;
        gui::RmlUIManager manager_;
        SequencerController controller_;
        gui::panels::SequencerUIState ui_;
        gui::FilmStripRenderer film_strip_;
        std::unique_ptr<RmlSequencerPanel> panel_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
        std::array<CountingElement*, 8> elements_{};
        float width_ = 960.0f;
    };

    TEST_F(SequencerMarkupRegressionTest, PlaybackRetainsUnchangedSubtrees) {
        const auto expected = markup();
        resetCounters();
        for (int i = 1; i <= 20; ++i) {
            controller_.seek(static_cast<float>(i) * 0.1f);
            rebuild();
        }
        EXPECT_EQ(replacements(), 0u);
        EXPECT_EQ(markup(), expected);
    }

    TEST_F(SequencerMarkupRegressionTest, ChangesAndResourceResetsMatchFreshMarkup) {
        ASSERT_EQ(elements_[4]->GetNumChildren(), 500);
        controller_.selectKeyframe(12);
        expectFreshEquivalent();
        EXPECT_TRUE(elements_[4]->GetChild(12)->IsClassSet("selected"));
        hover(18);
        expectFreshEquivalent();
        EXPECT_TRUE(elements_[4]->GetChild(18)->IsClassSet("hovered"));
        hover(std::nullopt);
        expectFreshEquivalent();
        controller_.removeSelectedKeyframe();
        expectFreshEquivalent();
        EXPECT_EQ(elements_[4]->GetNumChildren(), 499);
        width_ = 640.0f;
        panel_->setTimelineView(2.0f, 4.0f);
        expectFreshEquivalent();
        controller_.setKeyframeEasing(18, sequencer::EasingType::EASE_IN_OUT);
        expectFreshEquivalent();
        controller_.timeline().setClipDuration(75.0f);
        expectFreshEquivalent();
        panel_->setFilmStripAttached(false);
        expectFreshEquivalent();
        EXPECT_EQ(elements_[4]->GetNumChildren(), 0);
        panel_->setFilmStripAttached(true);
        expectFreshEquivalent();
        const auto expected = markup();
        destroyGraphics();
        rebuild();
        EXPECT_EQ(markup(), expected);
        replaceDocument();
        rebuild();
        EXPECT_EQ(markup(), expected);
        controller_.clear();
        expectFreshEquivalent();
        EXPECT_EQ(elements_[4]->GetNumChildren(), 0);
        EXPECT_EQ(elements_[5]->GetNumChildren(), 0);
    }

    TEST_F(SequencerMarkupRegressionTest, RepeatedUpdatesAreCheaperThanRebuildingMarkup) {
        // Same production formatter and Rml parser, with only cache reuse changed.
        // Interleave batches and use a generous relative margin, not a wall-time limit.
        const auto measure = [&](const bool fresh) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) {
                if (fresh)
                    rebuildFresh();
                else
                    rebuild();
            }
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        };
        std::array<double, 5> cached{}, fresh{};
        for (size_t i = 0; i < cached.size(); ++i) {
            if (i % 2 == 0) {
                cached[i] = measure(false);
                fresh[i] = measure(true);
            } else {
                fresh[i] = measure(true);
                cached[i] = measure(false);
            }
        }
        std::sort(cached.begin(), cached.end());
        std::sort(fresh.begin(), fresh.end());
        RecordProperty("cached_batch_us", std::to_string(cached[2]));
        RecordProperty("fresh_batch_us", std::to_string(fresh[2]));
        EXPECT_LT(cached[2], fresh[2] * 0.5);
    }

} // namespace lfs::vis

namespace {
    class SequencerToolbarLayoutTest : public ::testing::Test {
    protected:
        class Renderer final : public Rml::RenderInterface {
        public:
            Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex>, Rml::Span<const int>) override { return 1; }
            void RenderGeometry(Rml::CompiledGeometryHandle, Rml::Vector2f, Rml::TextureHandle) override {}
            void ReleaseGeometry(Rml::CompiledGeometryHandle) override {}
            Rml::TextureHandle LoadTexture(Rml::Vector2i& dimensions, const Rml::String&) override {
                dimensions = {16, 16};
                return 1;
            }
            Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte>, Rml::Vector2i) override { return 1; }
            void ReleaseTexture(Rml::TextureHandle) override {}
            void EnableScissorRegion(bool) override {}
            void SetScissorRegion(Rml::Rectanglei) override {}
        };
        static void SetUpTestSuite() {
            ASSERT_TRUE(Rml::Initialise());
            ASSERT_TRUE(Rml::LoadFontFace((std::filesystem::path(PROJECT_ROOT_PATH) /
                                           "src/visualizer/gui/assets/fonts/Inter-Regular.ttf")
                                              .string()));
        }
        static void TearDownTestSuite() {
            Rml::Shutdown();
            lfs::event::LocalizationManager::getInstance().reset();
        }
        void SetUp() override {
            context_ = Rml::CreateContext("sequencer_toolbar", {1240, 320}, &renderer_);
            ASSERT_NE(context_, nullptr);
        }
        void TearDown() override { ASSERT_TRUE(Rml::RemoveContext("sequencer_toolbar")); }
        static std::string read(const std::filesystem::path& path) {
            std::ifstream file(path);
            return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        }
        void load(int width, float scale = 1.0f, const std::string& language = "en") {
            if (document_)
                context_->UnloadDocument(document_);
            context_->SetDimensions({width, 640});
            context_->SetDensityIndependentPixelRatio(scale);
            const auto root = std::filesystem::path(PROJECT_ROOT_PATH) / "src/visualizer/gui";
            const auto resources = root / "rmlui/resources";
            auto rml = read(resources / "sequencer.rml");
            auto& locales = lfs::event::LocalizationManager::getInstance();
            ASSERT_TRUE(locales.initialize((root / "resources/locales").string()));
            ASSERT_TRUE(locales.setLanguage(language));
            const std::regex token("@tr:([A-Za-z0-9_.-]+)");
            std::string localized;
            size_t offset = 0;
            for (std::sregex_iterator it(rml.begin(), rml.end(), token), end; it != end; ++it) {
                localized.append(rml, offset, static_cast<size_t>(it->position()) - offset);
                localized += Rml::StringUtilities::EncodeRml(locales.get((*it)[1].str()));
                offset = static_cast<size_t>(it->position() + it->length());
            }
            localized.append(rml, offset, std::string::npos);
            const auto begin = localized.find("<link");
            const auto end = localized.find("/>", begin) + 2;
            localized.replace(begin, end - begin, "<style>" + read(resources / "components.rcss") + read(resources / "sequencer.rcss") + "</style>");
            document_ = context_->LoadDocumentFromMemory(localized);
            ASSERT_NE(document_, nullptr);
            document_->GetElementById("floating-header")->SetClass("hidden", false);
            document_->GetElementById("panel")->SetClass("is-floating", true);
            document_->Show();
            context_->Update();
        }
        Rml::Element* row() { return document_->GetElementById("transport-row"); }
        float right(Rml::Element* element) {
            return element->GetAbsoluteOffset(Rml::BoxArea::Border).x + element->GetBox().GetSize(Rml::BoxArea::Border).x;
        }
        void expectReachable(const char* id) {
            SCOPED_TRACE(id);
            auto* element = document_->GetElementById(id);
            ASSERT_NE(element, nullptr);
            ASSERT_TRUE(element->IsVisible(true));
            const auto origin = row()->GetAbsoluteOffset();
            const auto x = element->GetAbsoluteOffset(Rml::BoxArea::Border).x;
            row()->SetScrollLeft(row()->GetScrollLeft() + x - origin.x);
            context_->Update();
            EXPECT_GE(element->GetAbsoluteOffset(Rml::BoxArea::Border).x, origin.x - 1.0f);
            EXPECT_LE(right(element), origin.x + row()->GetClientWidth() + 1.0f);
            EXPECT_GE(element->GetAbsoluteOffset(Rml::BoxArea::Border).y, origin.y - 1.0f);
            EXPECT_LE(element->GetAbsoluteOffset(Rml::BoxArea::Border).y + element->GetBox().GetSize(Rml::BoxArea::Border).y,
                      origin.y + row()->GetClientHeight() + 1.0f);
        }
        inline static Renderer renderer_;
        Rml::Context* context_ = nullptr;
        Rml::ElementDocument* document_ = nullptr;
    };

    TEST_F(SequencerToolbarLayoutTest, OverflowControlsRemainReachable) {
        for (const auto& language : {"en", "de"}) {
            for (const auto [width, scale] : {std::pair{1240, 1.0f}, {600, 1.0f}, {1240, 2.0f}}) {
                SCOPED_TRACE(std::format("{} width={} scale={}", language, width, scale));
                load(width, scale, language);
                ASSERT_GT(row()->GetScrollWidth(), row()->GetClientWidth());
                // A scroll range without a visible affordance still leaves these controls unreachable.
                ASSERT_GT(row()->GetOffsetHeight() - row()->GetClientHeight(), 2.0f * scale);
                for (const auto* id : {"quality-scrub", "btn-export", "btn-clear", "btn-dock-toggle", "btn-play"})
                    expectReachable(id);
            }
        }
    }

    TEST_F(SequencerToolbarLayoutTest, WideToolbarNeedsNoScrollingAndKeepsTimelinePosition) {
        load(2400);
        EXPECT_FLOAT_EQ(row()->GetScrollWidth(), row()->GetClientWidth());
        EXPECT_FLOAT_EQ(row()->GetScrollLeft(), 0.0f);
        EXPECT_FLOAT_EQ(row()->GetBox().GetSize(Rml::BoxArea::Content).y, 36.0f);
        const auto timeline_y = document_->GetElementById("timeline")->GetAbsoluteOffset().y;
        for (const auto* id : {"btn-play", "quality-scrub", "btn-export", "btn-clear", "btn-dock-toggle"})
            expectReachable(id);
        load(1240);
        EXPECT_FLOAT_EQ(document_->GetElementById("timeline")->GetAbsoluteOffset().y, timeline_y);
    }
    TEST_F(SequencerToolbarLayoutTest, RecordsLayoutUpdateCost) {
        load(1240);
        std::array<double, 7> batches;
        for (auto& elapsed : batches) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 100; ++i) {
                context_->SetDimensions({1240 + i % 2, 640});
                context_->Update();
            }
            elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 100.0;
        }
        std::sort(batches.begin(), batches.end());
        RecordProperty("median_layout_update_us", std::to_string(batches[3]));
    }

    TEST_F(SequencerToolbarLayoutTest, RecordsSteadyUpdateCostAgainstUnclippedReference) {
        // At this width the old media rules hide no controls. Overriding overflow
        // reproduces the previous layout for a same-process steady-update control.
        const auto measure = [&](const bool reference) {
            load(1240);
            if (reference) {
                row()->SetProperty("overflow-x", "visible");
                row()->SetProperty("overflow-y", "visible");
                context_->Update();
            }
            const auto origin = row()->GetAbsoluteOffset();
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 2000; ++i) {
                context_->ProcessMouseMove(static_cast<int>(origin.x) + 30 + i % 2,
                                           static_cast<int>(origin.y) + 12, 0);
                context_->Update();
            }
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 2000.0;
        };
        std::array<double, 7> current{}, reference{};
        for (size_t i = 0; i < current.size(); ++i) {
            if (i % 2 == 0) {
                current[i] = measure(false);
                reference[i] = measure(true);
            } else {
                reference[i] = measure(true);
                current[i] = measure(false);
            }
        }
        std::sort(current.begin(), current.end());
        std::sort(reference.begin(), reference.end());
        RecordProperty("steady_update_us", std::to_string(current[3]));
        RecordProperty("unclipped_reference_us", std::to_string(reference[3]));
    }

} // namespace
