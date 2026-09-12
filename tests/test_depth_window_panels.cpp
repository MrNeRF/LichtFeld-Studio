/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/tensor.hpp"
#include "operation/undo_history.hpp"
#include "operator/operator_registry.hpp"
#include "tools/selection_tool.hpp"
#include "visualizer/rendering/depth_window_state.hpp"
#include "visualizer/rendering/dirty_flags.hpp"
#include "visualizer/rendering/render_pass.hpp"
#include "visualizer/rendering/rendering_manager.hpp"
#include "visualizer/rendering/rendering_types.hpp"
#include "visualizer/rendering/viewport_request_builder.hpp"
#include "visualizer_impl.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

namespace lfs::vis {

    namespace {

        DepthWindowState makeWindow(const float depth_near_val,
                                    const float depth_far_val,
                                    const float scale_x = 0.35f,
                                    const float scale_y = 0.35f,
                                    const float offset_x = 0.0f,
                                    const float offset_y = 0.0f) {
            return {
                .near_plane = depth_near_val,
                .far_plane = depth_far_val,
                .scale_x = scale_x,
                .scale_y = scale_y,
                .offset_x = offset_x,
                .offset_y = offset_y,
            };
        }

        void expectProjectionWindow(const RenderSettings& settings, const DepthWindowState& expected) {
            EXPECT_FLOAT_EQ(-settings.depth_filter_max.z, expected.near_plane);
            EXPECT_FLOAT_EQ(-settings.depth_filter_min.z, expected.far_plane);
            EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, expected.scale_x);
            EXPECT_FLOAT_EQ(settings.depth_filter_scale_y, expected.scale_y);
            EXPECT_FLOAT_EQ(settings.depth_filter_offset_x, expected.offset_x);
            EXPECT_FLOAT_EQ(settings.depth_filter_offset_y, expected.offset_y);
        }

        void setMode(RenderingManager& manager, const SplitViewMode mode) {
            auto settings = manager.getSettings();
            settings.split_view_mode = mode;
            manager.updateSettings(settings);
        }

        void writeGlobalWindow(RenderingManager& manager, const DepthWindowState& window) {
            auto settings = manager.getSettings();
            settings.depth_filter_min.z = -window.far_plane;
            settings.depth_filter_max.z = -window.near_plane;
            settings.depth_filter_scale_x = window.scale_x;
            settings.depth_filter_scale_y = window.scale_y;
            settings.depth_filter_offset_x = window.offset_x;
            settings.depth_filter_offset_y = window.offset_y;
            manager.updateSettings(settings);
        }

        std::array<DepthWindowState, 2> parkPair(RenderingManager& manager, const bool disabled = true) {
            setMode(manager, SplitViewMode::IndependentDual);
            manager.setDepthWindowSync(false);
            const std::array pair{makeWindow(1.0f, 10.0f, 0.3f, 0.4f, 0.1f, 0.2f),
                                  makeWindow(2.0f, 20.0f, 0.6f, 0.7f, -0.2f, -0.3f)};
            manager.setDepthWindowForPanel(SplitViewPanelId::Left, pair[0]);
            manager.setDepthWindowForPanel(SplitViewPanelId::Right, pair[1]);
            manager.setFocusedSplitPanel(SplitViewPanelId::Right);
            setMode(manager, SplitViewMode::GTComparison);
            if (disabled)
                setMode(manager, SplitViewMode::Disabled);
            return pair;
        }

    } // namespace

    class DepthWindowPanelsTest : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();

            options_.show_startup_overlay = false;
            options_.width = 200;
            options_.height = 200;
            viewer_ = std::make_unique<lfs::vis::VisualizerImpl>(options_);
            rendering_manager_ = viewer_->getRenderingManager();
            ASSERT_NE(rendering_manager_, nullptr);

            const std::vector<float> means_data{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
            const std::vector<float> rotation_data{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
            auto means = lfs::core::Tensor::from_vector(means_data, {size_t{2}, size_t{3}}, lfs::core::Device::CUDA).to(lfs::core::DataType::Float32);
            auto sh0 = lfs::core::Tensor::zeros({size_t{2}, size_t{1}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto shN = lfs::core::Tensor::zeros({size_t{2}, size_t{3}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto scaling = lfs::core::Tensor::zeros({size_t{2}, size_t{3}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            auto rotation = lfs::core::Tensor::from_vector(rotation_data, {size_t{2}, size_t{4}}, lfs::core::Device::CUDA).to(lfs::core::DataType::Float32);
            auto opacity = lfs::core::Tensor::zeros({size_t{2}, size_t{1}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            viewer_->getSceneManager()->getScene().addSplat(
                "depth_window_test",
                std::make_unique<lfs::core::SplatData>(1, std::move(means), std::move(sh0), std::move(shN), std::move(scaling), std::move(rotation), std::move(opacity), 1.0f));
            viewer_->getSceneManager()->initSelectionService();
            if (auto* const service = viewer_->getSceneManager()->getSelectionService()) {
                service->setTestingViewport({
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(options_.width),
                    .height = static_cast<float>(options_.height),
                    .render_width = options_.width,
                    .render_height = options_.height,
                });
            }
            auto settings = rendering_manager_->getSettings();
            settings.depth_filter_enabled = true;
            settings.depth_filter_scale_x = 0.5f;
            settings.depth_filter_scale_y = 0.5f;
            rendering_manager_->updateSettings(settings);
        }

        void TearDown() override {
            lfs::vis::op::operators().cancelModalOperator();
            lfs::event::EventBridge::instance().clear_all();
            lfs::core::event::bus().clear_all();
            lfs::vis::services().clear();
            viewer_.reset();
            lfs::vis::op::undoHistory().clear();
        }

        void enterIndependentDual() {
            lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        }

        lfs::vis::ViewerOptions options_{};
        std::unique_ptr<lfs::vis::VisualizerImpl> viewer_;
        lfs::vis::RenderingManager* rendering_manager_ = nullptr;
    };

    TEST_F(DepthWindowPanelsTest, RetainedGtPairSurvivesDisabledAndRepeatedGt) {
        auto& manager = *rendering_manager_;
        enterIndependentDual();
        manager.setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f));
        manager.setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f));
        manager.setFocusedSplitPanel(SplitViewPanelId::Right);
        const auto pair = manager.depthWindowSnapshot().panels;
        const auto lineage = manager.getDepthWindowCollapseRecord().generation;
        for (int cycle = 0; cycle < 2; ++cycle) {
            lfs::core::events::cmd::ToggleGTComparison{}.emit();
            ASSERT_EQ(manager.getSettings().split_view_mode, SplitViewMode::GTComparison);
            writeGlobalWindow(manager, makeWindow(7.0f, 70.0f));
            lfs::core::events::cmd::ToggleGTComparison{}.emit();
            ASSERT_EQ(manager.getSettings().split_view_mode, SplitViewMode::Disabled);
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, lineage);
        }
        enterIndependentDual();
        EXPECT_EQ(manager.depthWindowSnapshot().panels, pair);
        EXPECT_EQ(manager.getFocusedSplitPanel(), SplitViewPanelId::Left);
        expectProjectionWindow(manager.getSettings(), pair[0]);
        EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, lineage);
    }

    TEST_F(DepthWindowPanelsTest, EveryDisabledGeometryFieldDiscardsRetainedPair) {
        const std::array fields{&DepthWindowState::near_plane, &DepthWindowState::far_plane,
                                &DepthWindowState::scale_x, &DepthWindowState::scale_y,
                                &DepthWindowState::offset_x, &DepthWindowState::offset_y};
        for (const bool use_settings : {false, true}) {
            for (const auto field : fields) {
                RenderingManager manager;
                parkPair(manager);
                const auto generation = manager.getDepthWindowCollapseRecord().generation;
                auto changed = manager.depthWindowSnapshot().projection;
                changed.*field = std::nextafter(changed.*field, std::numeric_limits<float>::infinity());
                if (use_settings)
                    writeGlobalWindow(manager, changed);
                else
                    ASSERT_TRUE(manager.setDepthWindowForPanel(SplitViewPanelId::Right, changed));
                const auto record = manager.getDepthWindowCollapseRecord();
                EXPECT_EQ(record.generation, generation + 1);
                EXPECT_EQ(record.kind, RenderingManager::DepthWindowLineageKind::RetainedPairDiscard);
                setMode(manager, SplitViewMode::IndependentDual);
                EXPECT_EQ(manager.depthWindowSnapshot().panels, (std::array{changed, changed}));
            }
        }
    }

    TEST_F(DepthWindowPanelsTest, NormalizedNoopsAndEnableVizChangesPreserveRetainedPair) {
        for (const bool use_settings : {false, true}) {
            RenderingManager manager;
            auto pair = parkPair(manager, false);
            const auto extreme = makeWindow(0.0f, 1000.0f, 1.0f, 0.05f, 1.0f, -1.0f);
            writeGlobalWindow(manager, extreme);
            setMode(manager, SplitViewMode::Disabled);
            const auto generation = manager.getDepthWindowCollapseRecord().generation;
            auto normalized_equal = extreme;
            normalized_equal.scale_x = 2.0f;
            normalized_equal.scale_y = -2.0f;
            normalized_equal.offset_x = 2.0f;
            normalized_equal.offset_y = -2.0f;
            if (use_settings)
                writeGlobalWindow(manager, normalized_equal);
            else {
                normalized_equal.near_plane = -1.0f;
                normalized_equal.far_plane = 2000.0f;
                ASSERT_TRUE(manager.setDepthWindowForPanel(SplitViewPanelId::Right, normalized_equal));
            }
            auto settings = manager.getSettings();
            settings.depth_filter_enabled = !settings.depth_filter_enabled;
            settings.depth_filter_viz_mode = (settings.depth_filter_viz_mode + 1) % 3;
            manager.updateSettings(settings);
            manager.setDepthWindowSync(manager.getDepthWindowSync());
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, generation);
            setMode(manager, SplitViewMode::IndependentDual);
            EXPECT_EQ(manager.depthWindowSnapshot().panels, pair);
        }
    }

    TEST_F(DepthWindowPanelsTest, DisabledSyncChangeDiscardsBeforeDormantRefusal) {
        for (const bool initial_sync : {false, true}) {
            RenderingManager manager;
            setMode(manager, SplitViewMode::IndependentDual);
            manager.setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f));
            manager.setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f));
            manager.setDepthWindowSync(initial_sync);
            setMode(manager, SplitViewMode::GTComparison);
            const auto generation = manager.getDepthWindowCollapseRecord().generation;
            manager.setDepthWindowSync(!initial_sync);
            ASSERT_EQ(manager.getDepthWindowSync(), initial_sync);
            setMode(manager, SplitViewMode::Disabled);
            const auto global = manager.depthWindowSnapshot().projection;
            manager.setDepthWindowSync(!initial_sync);
            EXPECT_EQ(manager.getDepthWindowSync(), !initial_sync);
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, generation + 1);
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().kind,
                      RenderingManager::DepthWindowLineageKind::RetainedPairDiscard);
            setMode(manager, SplitViewMode::IndependentDual);
            EXPECT_EQ(manager.depthWindowSnapshot().panels, (std::array{global, global}));
        }
    }

    TEST_F(DepthWindowPanelsTest, DisabledDragInvalidatesOnlyChangedAcceptedCommit) {
        for (const int outcome : {0, 1, 2, 3, 4}) {
            SCOPED_TRACE(outcome); // subthreshold, cancel, unchanged commit, changed commit, refused commit
            RenderingManager manager;
            const auto pair = parkPair(manager);
            const auto generation = manager.getDepthWindowCollapseRecord().generation;
            const auto baseline = manager.depthWindowSnapshot().projection;
            ASSERT_EQ(manager.depthWindowSnapshot().panels, (std::array{baseline, baseline}));
            const auto epoch = manager.depthWindowModeEpoch();
            uint64_t token = 0;
            manager.beginDepthWindowDrag(SplitViewPanelId::Left, token);
            auto preview = baseline;
            preview.scale_x += 0.1f;
            if (outcome != 0)
                ASSERT_TRUE(manager.applyDepthWindowForPanelIfEpoch(SplitViewPanelId::Left, preview, epoch, token));
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, generation);
            manager.setDepthWindowSync(true); // An owned drag refuses sync without discarding.
            EXPECT_FALSE(manager.getDepthWindowSync());
            if (outcome == 1)
                ASSERT_TRUE(manager.restorePinnedDepthWindowSlots(SplitViewPanelId::Left, baseline, baseline, epoch, token));
            if (outcome >= 2) {
                op::DepthWindowModeSnapshot after;
                EXPECT_EQ(manager.commitDepthWindowForPanelIfEpoch(
                              SplitViewPanelId::Left, outcome == 2 ? baseline : preview,
                              outcome == 4 ? epoch + 1 : epoch, token, after),
                          outcome != 4);
            }
            manager.endDepthWindowDrag(SplitViewPanelId::Left, token);
            EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, generation + (outcome == 3 ? 1 : 0));
            setMode(manager, SplitViewMode::IndependentDual);
            EXPECT_EQ(manager.depthWindowSnapshot().panels, outcome == 3 ? (std::array{preview, preview}) : pair);
        }
    }

    TEST_F(DepthWindowPanelsTest, OtherComparisonAndProjectOrSceneResetDiscardRetainedPair) {
        for (const bool disabled : {false, true}) {
            for (const int invalidator : {0, 1, 2, 3}) {
                RenderingManager manager;
                parkPair(manager, disabled);
                const auto generation = manager.getDepthWindowCollapseRecord().generation;
                const auto global = manager.depthWindowSnapshot().projection;
                if (invalidator == 0)
                    setMode(manager, SplitViewMode::PLYComparison);
                else if (invalidator == 1)
                    manager.restoreDepthWindowStateFromProject();
                else if (invalidator == 2)
                    lfs::core::events::state::SceneCleared{}.emit();
                else
                    lfs::core::events::state::SceneLoaded{
                        .scene = &viewer_->getSceneManager()->getScene(),
                        .path = {},
                        .type = lfs::core::events::state::SceneLoaded::Type::PLY,
                        .num_gaussians = 2}
                        .emit();
                EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, generation + 1);
                EXPECT_EQ(manager.getDepthWindowCollapseRecord().kind,
                          invalidator == 1 ? RenderingManager::DepthWindowLineageKind::ProjectRestore
                                           : RenderingManager::DepthWindowLineageKind::RetainedPairDiscard);
                setMode(manager, SplitViewMode::IndependentDual);
                EXPECT_EQ(manager.depthWindowSnapshot().panels, (std::array{global, global}));
            }
        }
    }

    TEST_F(DepthWindowPanelsTest, ParkedGtPanelWriteRefusesWithoutMutation) {
        RenderingManager manager;
        const auto pair = parkPair(manager, false);
        const auto before = manager.depthWindowSnapshot();
        const auto generation = manager.depthWindowProjectionGeneration();
        const auto lineage = manager.getDepthWindowCollapseRecord().generation;
        manager.dirty_mask_.store(0);
        EXPECT_FALSE(manager.setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(7.0f, 70.0f)));
        EXPECT_EQ(manager.dirty_mask_.load(), 0u);
        EXPECT_EQ(manager.depthWindowSnapshot().panels, before.panels);
        EXPECT_EQ(manager.depthWindowSnapshot().projection, before.projection);
        EXPECT_EQ(manager.depthWindowProjectionGeneration(), generation);
        EXPECT_EQ(manager.getDepthWindowCollapseRecord().generation, lineage);
        setMode(manager, SplitViewMode::IndependentDual);
        EXPECT_EQ(manager.depthWindowSnapshot().panels, pair);
    }

    TEST_F(DepthWindowPanelsTest, SetGetRoundtripAndClamping) {
        RenderingManager manager;
        DepthWindowState extreme = makeWindow(500.0f, -1.0f, 2.0f, -2.0f, 5.0f, -5.0f);
        manager.setDepthWindowForPanel(SplitViewPanelId::Left, extreme);
        const auto stored = manager.getDepthWindowForPanel(SplitViewPanelId::Left);
        EXPECT_FLOAT_EQ(stored.near_plane, 500.0f);
        EXPECT_FLOAT_EQ(stored.far_plane, 500.01f);
        EXPECT_FLOAT_EQ(stored.scale_x, 1.0f);
        EXPECT_FLOAT_EQ(stored.scale_y, 0.05f);
        EXPECT_FLOAT_EQ(stored.offset_x, 1.0f);
        EXPECT_FLOAT_EQ(stored.offset_y, -1.0f);
    }

    TEST_F(DepthWindowPanelsTest, FocusSwitchSwapsProjection) {
        // Sync defaults OFF: per-panel independence is the default experience.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);

        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.4f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.5f));

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);
        const auto left_settings = rendering_manager_->getSettings();
        EXPECT_FLOAT_EQ(-left_settings.depth_filter_max.z, 1.0f);
        EXPECT_FLOAT_EQ(-left_settings.depth_filter_min.z, 10.0f);
        EXPECT_FLOAT_EQ(left_settings.depth_filter_scale_x, 0.4f);

        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        const auto right_settings = rendering_manager_->getSettings();
        EXPECT_FLOAT_EQ(-right_settings.depth_filter_max.z, 2.0f);
        EXPECT_FLOAT_EQ(-right_settings.depth_filter_min.z, 20.0f);
        EXPECT_FLOAT_EQ(right_settings.depth_filter_scale_x, 0.5f);
    }

    TEST_F(DepthWindowPanelsTest, UpdateSettingsBackRoutesFocusedSlotWhenUnsynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.45f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.55f;
        settings.depth_filter_min.z = -15.0f;
        settings.depth_filter_max.z = -3.0f;
        rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.55f);
        EXPECT_FLOAT_EQ(left.near_plane, 3.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 15.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 2.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 20.0f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.45f);
    }

    TEST_F(DepthWindowPanelsTest, FanOutWhenSynced) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(4.0f, 40.0f, 0.6f));

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.near_plane, 4.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 4.0f);
        EXPECT_FLOAT_EQ(left.scale_x, 0.6f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.6f);
    }

    TEST_F(DepthWindowPanelsTest, ModeEnterSeedsBothSlots) {
        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.42f;
        settings.depth_filter_min.z = -25.0f;
        settings.depth_filter_max.z = -5.0f;
        rendering_manager_->updateSettings(settings);

        enterIndependentDual();

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.42f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.42f);
        EXPECT_FLOAT_EQ(left.near_plane, 5.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 5.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 25.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 25.0f);
    }

    TEST_F(DepthWindowPanelsTest, ModeLeaveCollapsesToPreTransitionFocus) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.3f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.7f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);

        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();

        const auto settings = rendering_manager_->getSettings();
        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::Disabled);
        EXPECT_FLOAT_EQ(-settings.depth_filter_max.z, 2.0f);
        EXPECT_FLOAT_EQ(-settings.depth_filter_min.z, 20.0f);
        EXPECT_FLOAT_EQ(settings.depth_filter_scale_x, 0.7f);

        // The sync choice survives leaving and re-entering split view.
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
        lfs::core::events::cmd::ToggleIndependentSplitView{.viewport = &viewer_->getViewport()}.emit();
        EXPECT_EQ(rendering_manager_->getSettings().split_view_mode, SplitViewMode::IndependentDual);
        EXPECT_FALSE(rendering_manager_->getDepthWindowSync());
    }

    TEST_F(DepthWindowPanelsTest, StaleProxyWithoutSelectionMaskDoesNotMoveOtherPanel) {
        ToolContext context(rendering_manager_, nullptr, &viewer_->getViewport(), nullptr);
        tools::SelectionTool tool;
        ASSERT_TRUE(tool.initialize(context));
        tool.setEnabled(true);
        tool.setDepthFilterEnabled(true);
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        const auto left = makeWindow(1.0f, 10.0f, 0.35f, 0.36f, 0.10f, 0.11f);
        const auto right = makeWindow(2.0f, 20.0f, 0.45f, 0.46f, -0.20f, -0.21f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right);

        auto stale = rendering_manager_->getSettings();
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        tool.update(context);
        ASSERT_FLOAT_EQ(tool.getDepthNear(), right.near_plane);
        ASSERT_FLOAT_EQ(tool.getDepthFar(), right.far_plane);
        const auto generation = rendering_manager_->depthWindowProjectionGeneration();
        stale.background_color = glm::vec3(0.1f, 0.2f, 0.3f);
        rendering_manager_->updateSettings(stale, DirtyFlag::BACKGROUND);

        const auto applied = rendering_manager_->getSettings();
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right);
        expectProjectionWindow(applied, right);
        EXPECT_EQ(applied.background_color, stale.background_color);
        EXPECT_EQ(rendering_manager_->depthWindowProjectionGeneration(), generation);

        for (int tick = 0; tick < 2; ++tick) {
            SCOPED_TRACE(tick);
            tool.update(context);
            EXPECT_FLOAT_EQ(tool.getDepthNear(), right.near_plane);
            EXPECT_FLOAT_EQ(tool.getDepthFar(), right.far_plane);
            expectProjectionWindow(rendering_manager_->getSettings(), right);
            EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left);
            EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), right);
            EXPECT_EQ(rendering_manager_->depthWindowProjectionGeneration(), generation);
        }
        tool.shutdown();
    }

    TEST_F(DepthWindowPanelsTest, StaleNonSelectionWritePreservesActiveDepthWindowPreview) {
        enterIndependentDual();
        const auto left = makeWindow(1.0f, 10.0f, 0.35f, 0.36f, 0.10f, 0.11f);
        const auto right = makeWindow(2.0f, 20.0f, 0.45f, 0.46f, -0.20f, -0.21f);
        const auto preview = makeWindow(3.0f, 30.0f, 0.55f, 0.56f, 0.30f, 0.31f);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, left);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, right);
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Right);
        auto stale = rendering_manager_->getSettings();
        std::uint64_t drag_token = 0;
        ASSERT_FALSE(rendering_manager_->beginDepthWindowDrag(SplitViewPanelId::Right, drag_token));
        ASSERT_NE(drag_token, 0u);
        rendering_manager_->beginDepthWindowPreview(SplitViewPanelId::Right);
        const auto epoch = rendering_manager_->depthWindowModeEpoch();
        ASSERT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Right, preview, epoch, drag_token));
        const auto generation = rendering_manager_->depthWindowProjectionGeneration();
        stale.background_color = glm::vec3(0.1f, 0.2f, 0.3f);

        rendering_manager_->updateSettings(stale, DirtyFlag::BACKGROUND);

        expectProjectionWindow(rendering_manager_->getSettings(), preview);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left), left);
        EXPECT_EQ(rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right), preview);
        EXPECT_EQ(rendering_manager_->getSettings().background_color, stale.background_color);
        EXPECT_EQ(rendering_manager_->depthWindowProjectionGeneration(), generation);
        EXPECT_EQ(rendering_manager_->depthWindowModeEpoch(), epoch);
        EXPECT_TRUE(rendering_manager_->depthWindowDragPreview());
        EXPECT_TRUE(rendering_manager_->applyDepthWindowForPanelIfEpoch(
            SplitViewPanelId::Right, preview, epoch, drag_token));
        rendering_manager_->endDepthWindowPreview(SplitViewPanelId::Right);
        rendering_manager_->endDepthWindowDrag(SplitViewPanelId::Right, drag_token);
    }

    TEST_F(DepthWindowPanelsTest, NonSelectionWritesStillApplyInGlobalAndSyncedModes) {
        for (const auto mode : {SplitViewMode::Disabled, SplitViewMode::PLYComparison,
                                SplitViewMode::GTComparison, SplitViewMode::IndependentDual}) {
            SCOPED_TRACE(static_cast<int>(mode));
            RenderingManager manager;
            auto settings = manager.getSettings();
            settings.split_view_mode = mode;
            manager.updateSettings(settings);
            manager.setDepthWindowSync(mode == SplitViewMode::IndependentDual);
            manager.setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f));
            const auto generation = manager.depthWindowProjectionGeneration();
            settings = manager.getSettings();
            settings.depth_filter_min.z = -30.0f;
            settings.depth_filter_max.z = -3.0f;
            settings.depth_filter_scale_x = 0.55f;
            settings.depth_filter_scale_y = 0.56f;
            settings.depth_filter_offset_x = 0.30f;
            settings.depth_filter_offset_y = 0.31f;
            const auto written = makeWindow(3.0f, 30.0f, 0.55f, 0.56f, 0.30f, 0.31f);

            manager.updateSettings(settings, DirtyFlag::BACKGROUND);

            expectProjectionWindow(manager.getSettings(), written);
            EXPECT_EQ(manager.getDepthWindowForPanel(SplitViewPanelId::Left), written);
            EXPECT_EQ(manager.getDepthWindowForPanel(SplitViewPanelId::Right), written);
            EXPECT_EQ(manager.depthWindowProjectionGeneration(), generation + 1);
        }
    }

    TEST_F(DepthWindowPanelsTest, NonSelectionModeEntryStillSeedsIncomingDepthWindow) {
        RenderingManager manager;
        manager.setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f));
        const auto generation = manager.depthWindowProjectionGeneration();
        auto settings = manager.getSettings();
        settings.split_view_mode = SplitViewMode::IndependentDual;
        settings.depth_filter_min.z = -30.0f;
        settings.depth_filter_max.z = -3.0f;
        settings.depth_filter_scale_x = 0.55f;
        settings.depth_filter_scale_y = 0.56f;
        settings.depth_filter_offset_x = 0.30f;
        settings.depth_filter_offset_y = 0.31f;
        const auto written = makeWindow(3.0f, 30.0f, 0.55f, 0.56f, 0.30f, 0.31f);

        manager.updateSettings(settings, DirtyFlag::BACKGROUND);

        EXPECT_EQ(manager.getSettings().split_view_mode, SplitViewMode::IndependentDual);
        expectProjectionWindow(manager.getSettings(), written);
        EXPECT_EQ(manager.getDepthWindowForPanel(SplitViewPanelId::Left), written);
        EXPECT_EQ(manager.getDepthWindowForPanel(SplitViewPanelId::Right), written);
        EXPECT_EQ(manager.depthWindowProjectionGeneration(), generation + 1);
    }

    TEST_F(DepthWindowPanelsTest, SyncOnCopiesFocusedToOther) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(false);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.31f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.62f));
        rendering_manager_->setFocusedSplitPanel(SplitViewPanelId::Left);

        rendering_manager_->setDepthWindowSync(true);

        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(right.near_plane, 1.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 10.0f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.31f);
    }

    TEST_F(DepthWindowPanelsTest, UpdateSettingsSyncedFansOutToBothSlots) {
        enterIndependentDual();
        rendering_manager_->setDepthWindowSync(true);
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Left, makeWindow(1.0f, 10.0f, 0.35f));
        rendering_manager_->setDepthWindowForPanel(SplitViewPanelId::Right, makeWindow(2.0f, 20.0f, 0.45f));

        auto settings = rendering_manager_->getSettings();
        settings.depth_filter_scale_x = 0.66f;
        settings.depth_filter_min.z = -30.0f;
        settings.depth_filter_max.z = -6.0f;
        rendering_manager_->updateSettings(settings, DirtyFlag::SELECTION);

        const auto left = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Left);
        const auto right = rendering_manager_->getDepthWindowForPanel(SplitViewPanelId::Right);
        EXPECT_FLOAT_EQ(left.scale_x, 0.66f);
        EXPECT_FLOAT_EQ(right.scale_x, 0.66f);
        EXPECT_FLOAT_EQ(left.near_plane, 6.0f);
        EXPECT_FLOAT_EQ(right.near_plane, 6.0f);
        EXPECT_FLOAT_EQ(left.far_plane, 30.0f);
        EXPECT_FLOAT_EQ(right.far_plane, 30.0f);
    }

    TEST(ViewportRequestBuilderTest, DisabledModeUsesGlobalSettingsNotDefaultSlots) {
        Viewport viewport;
        RenderSettings settings;
        settings.depth_filter_enabled = true;
        settings.split_view_mode = SplitViewMode::Disabled;
        settings.depth_filter_min = glm::vec3(-50.0f, -50.0f, -80.0f);
        settings.depth_filter_max = glm::vec3(50.0f, 50.0f, -7.0f);
        settings.depth_filter_scale_x = 0.42f;
        settings.depth_filter_scale_y = 0.51f;
        settings.depth_filter_offset_x = 0.11f;
        settings.depth_filter_offset_y = -0.22f;

        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .panel_depth_windows = {
                makeWindow(1.0f, 10.0f, 0.25f, 0.30f, 0.1f, 0.0f),
                makeWindow(2.0f, 20.0f, 0.55f, 0.60f, -0.2f, 0.15f),
            },
        };

        const auto request = buildViewportRenderRequest(ctx, {800, 600}, &ctx.viewport, std::nullopt);
        ASSERT_TRUE(request.filters.screen_window.has_value());
        ASSERT_TRUE(request.filters.view_volume.has_value());
        EXPECT_FLOAT_EQ(request.filters.screen_window->scale_x, 0.42f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->scale_y, 0.51f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->offset_x, 0.11f);
        EXPECT_FLOAT_EQ(request.filters.screen_window->offset_y, -0.22f);
        EXPECT_FLOAT_EQ(request.filters.view_volume->min.z, -80.0f);
        EXPECT_FLOAT_EQ(request.filters.view_volume->max.z, -7.0f);
    }

    TEST(ViewportRequestBuilderTest, IndependentDualPanelsResolveDifferentDepthWindows) {
        Viewport viewport;
        RenderSettings settings;
        settings.depth_filter_enabled = true;
        settings.split_view_mode = SplitViewMode::IndependentDual;
        settings.depth_filter_min = glm::vec3(-50.0f, -50.0f, -100.0f);
        settings.depth_filter_max = glm::vec3(50.0f, 50.0f, 0.0f);

        FrameContext ctx{
            .viewport = viewport,
            .settings = settings,
            .render_size = {800, 600},
            .panel_depth_windows = {
                makeWindow(1.0f, 10.0f, 0.25f, 0.30f, 0.1f, 0.0f),
                makeWindow(2.0f, 20.0f, 0.55f, 0.60f, -0.2f, 0.15f),
            },
        };

        const auto left_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Left);
        const auto right_request = buildViewportRenderRequest(
            ctx, {400, 600}, &ctx.viewport, SplitViewPanelId::Right);

        ASSERT_TRUE(left_request.filters.screen_window.has_value());
        ASSERT_TRUE(right_request.filters.screen_window.has_value());
        ASSERT_TRUE(left_request.filters.view_volume.has_value());
        ASSERT_TRUE(right_request.filters.view_volume.has_value());
        EXPECT_FLOAT_EQ(left_request.filters.screen_window->scale_x, 0.25f);
        EXPECT_FLOAT_EQ(right_request.filters.screen_window->scale_x, 0.55f);
        EXPECT_FLOAT_EQ(left_request.filters.view_volume->min.z, -10.0f);
        EXPECT_FLOAT_EQ(right_request.filters.view_volume->min.z, -20.0f);
        EXPECT_FLOAT_EQ(left_request.filters.view_volume->max.z, -1.0f);
        EXPECT_FLOAT_EQ(right_request.filters.view_volume->max.z, -2.0f);
    }

} // namespace lfs::vis
