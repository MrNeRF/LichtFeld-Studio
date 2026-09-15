/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace/viewport_workspace.hpp"

#include "rendering/coordinate_conventions.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <unordered_set>
#include <vector>

namespace lfs::vis {
    namespace {

        constexpr ViewRect kOuter{0, 0, 800, 600};

        [[nodiscard]] bool rectsOverlap(const ViewRect& a, const ViewRect& b) {
            if (a.empty() || b.empty())
                return false;
            return a.x < b.right() && b.x < a.right() && a.y < b.bottom() && b.y < a.bottom();
        }

        void expectNoOverlap(const std::vector<ViewRect>& rects) {
            for (std::size_t i = 0; i < rects.size(); ++i) {
                for (std::size_t j = i + 1; j < rects.size(); ++j)
                    EXPECT_FALSE(rectsOverlap(rects[i], rects[j])) << i << " vs " << j;
            }
        }

        void expectCoverage(const ViewRect& outer, const std::vector<LayoutPane>& panes,
                            const std::vector<SplitterRect>& splitters) {
            std::vector<ViewRect> rects;
            rects.reserve(panes.size() + splitters.size());
            long long covered = 0;
            for (const auto& pane : panes) {
                rects.push_back(pane.rect);
                covered += pane.rect.area();
            }
            for (const auto& splitter : splitters) {
                rects.push_back(splitter.rect);
                covered += splitter.rect.area();
            }
            expectNoOverlap(rects);
            EXPECT_EQ(covered, static_cast<long long>(outer.area()));

            if (outer.empty() || outer.area() > 200000)
                return;
            for (int y = outer.y; y < outer.bottom(); ++y) {
                for (int x = outer.x; x < outer.right(); ++x) {
                    int hits = 0;
                    for (const auto& pane : panes) {
                        if (pane.rect.contains(x, y))
                            ++hits;
                    }
                    for (const auto& splitter : splitters) {
                        if (splitter.rect.contains(x, y))
                            ++hits;
                    }
                    EXPECT_EQ(hits, 1) << "pixel " << x << "," << y;
                }
            }
        }

        [[nodiscard]] ViewPersistentState defaultViewState(const ViewId id) {
            ViewPersistentState state;
            state.id = id;
            state.window_size = {1280, 720};
            state.framebuffer_size = {1280, 720};
            return state;
        }

        class ViewportWorkspaceTest : public ::testing::Test {
        protected:
            ViewportWorkspace workspace_{glm::ivec2{1280, 720}};
        };

        TEST_F(ViewportWorkspaceTest, StartsWithSingleViewAndInitializedFramebuffer) {
            EXPECT_EQ(workspace_.layout().leafCount(), 1u);
            const ViewId primary = workspace_.primaryView();
            ASSERT_NE(primary, kInvalidViewId);
            const ViewRecord* record = workspace_.views().find(primary);
            ASSERT_NE(record, nullptr);
            EXPECT_EQ(record->camera.windowSize, glm::ivec2(1280, 720));
            EXPECT_EQ(record->camera.frameBufferSize, glm::ivec2(1280, 720));
            EXPECT_EQ(record->id, primary);
            EXPECT_TRUE(workspace_.layout().validate());
        }

        TEST_F(ViewportWorkspaceTest, CloseLastPaneFails) {
            const ViewId primary = workspace_.primaryView();
            const auto status = workspace_.close(primary);
            EXPECT_FALSE(status);
            EXPECT_EQ(status.error().code(), lfs::ErrorCode::FailedPrecondition);
            EXPECT_EQ(workspace_.layout().leafCount(), 1u);
            EXPECT_TRUE(workspace_.views().contains(primary));
        }

        TEST_F(ViewportWorkspaceTest, SplitSideBySideAndStackedPresetsCoverOuter) {
            const auto dual_h = workspace_.setPreset(LayoutPreset::DualHorizontal);
            ASSERT_TRUE(dual_h);
            auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 2u);
            EXPECT_EQ(snap.panes[0].rect.y, snap.panes[1].rect.y);
            EXPECT_LT(snap.panes[0].rect.x, snap.panes[1].rect.x);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);

            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::DualVertical));
            snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 2u);
            EXPECT_EQ(snap.panes[0].rect.x, snap.panes[1].rect.x);
            EXPECT_LT(snap.panes[0].rect.y, snap.panes[1].rect.y);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);
        }

        TEST_F(ViewportWorkspaceTest, QuadPresetSeedsFrontTopRightOrthoAndKeepsPerspective) {
            const ViewId primary = workspace_.primaryView();
            auto* source = workspace_.findView(primary);
            ASSERT_NE(source, nullptr);
            source->camera.camera.t = glm::vec3(4.0f, 5.0f, 6.0f);
            source->camera.camera.pivot = glm::vec3(1.0f, 2.0f, 3.0f);
            source->projection.focal_length_mm = 50.0f;

            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            const auto ids = workspace_.layout().leafIds();
            ASSERT_EQ(ids.size(), 4u);
            EXPECT_EQ(ids[0], primary);
            EXPECT_EQ(workspace_.layout().focused(), primary);

            const ViewRecord* persp = workspace_.views().find(ids[0]);
            const ViewRecord* front = workspace_.views().find(ids[1]);
            const ViewRecord* top = workspace_.views().find(ids[2]);
            const ViewRecord* right = workspace_.views().find(ids[3]);
            ASSERT_NE(persp, nullptr);
            ASSERT_NE(front, nullptr);
            ASSERT_NE(top, nullptr);
            ASSERT_NE(right, nullptr);

            EXPECT_FALSE(persp->projection.orthographic);
            EXPECT_EQ(persp->projection.focal_length_mm, 50.0f);
            EXPECT_EQ(persp->camera.camera.t, glm::vec3(4.0f, 5.0f, 6.0f));

            EXPECT_TRUE(front->projection.orthographic);
            EXPECT_TRUE(top->projection.orthographic);
            EXPECT_TRUE(right->projection.orthographic);
            EXPECT_EQ(front->grid_plane, 2);
            EXPECT_EQ(top->grid_plane, 1);
            EXPECT_EQ(right->grid_plane, 0);

            const glm::vec3 front_fwd = lfs::rendering::cameraForward(front->camera.camera.R);
            const glm::vec3 top_fwd = lfs::rendering::cameraForward(top->camera.camera.R);
            const glm::vec3 right_fwd = lfs::rendering::cameraForward(right->camera.camera.R);
            EXPECT_NEAR(front_fwd.z, -1.0f, 1e-5f);
            EXPECT_NEAR(top_fwd.y, -1.0f, 1e-5f);
            EXPECT_NEAR(right_fwd.x, -1.0f, 1e-5f);

            auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 4u);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);
        }

        TEST_F(ViewportWorkspaceTest, RepeatedFrameExtentsAndSettingsDoNotChangeDurableGenerations) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            workspace_.syncExtents(kOuter, 64, 4, {2.0f, 2.0f});
            const auto before = workspace_.exportState();
            for (int frame = 0; frame < 5; ++frame) {
                workspace_.syncExtents(kOuter, 64, 4, {2.0f, 2.0f});
                for (const auto& state : before.views) {
                    ASSERT_TRUE(workspace_.setViewProjection(state.id, state.projection));
                    ASSERT_TRUE(workspace_.setViewDepthWindow(state.id, state.depth));
                    ASSERT_TRUE(workspace_.setViewGridPlane(state.id, state.grid_plane));
                }
            }
            const auto after = workspace_.exportState();
            EXPECT_EQ(after.registry_generation, before.registry_generation);
            for (const auto& state : before.views)
                EXPECT_EQ(workspace_.findView(state.id)->state_generation, state.state_generation);
            workspace_.syncExtents({0, 0, 1000, 700}, 64, 4, {2.0f, 2.0f});
            EXPECT_GT(workspace_.views().generation(), before.registry_generation);
        }

        TEST_F(ViewportWorkspaceTest, QuadPresetIsIdempotentAndDoesNotResetPoses) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            const auto ids = workspace_.layout().leafIds();
            ASSERT_EQ(ids.size(), 4u);
            const auto layout_generation = workspace_.layout().generation();
            const auto registry_generation = workspace_.views().generation();
            auto* front = workspace_.findView(ids[1]);
            ASSERT_NE(front, nullptr);
            const glm::vec3 front_t = front->camera.camera.t;
            const auto front_r = front->camera.camera.R;
            front->projection.focal_length_mm = 80.0f;

            ASSERT_TRUE(workspace_.focus(ids[2]));
            const auto focused_layout_generation = workspace_.layout().generation();
            const auto focused_registry_generation = workspace_.views().generation();
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            const auto ids_again = workspace_.layout().leafIds();
            EXPECT_EQ(ids_again, ids);
            EXPECT_EQ(workspace_.layout().focused(), ids[2]);
            auto* front_again = workspace_.findView(ids[1]);
            ASSERT_NE(front_again, nullptr);
            EXPECT_EQ(front_again->camera.camera.t, front_t);
            EXPECT_EQ(front_again->camera.camera.R, front_r);
            EXPECT_EQ(front_again->projection.focal_length_mm, 80.0f);
            EXPECT_EQ(workspace_.layout().generation(), focused_layout_generation);
            EXPECT_EQ(workspace_.views().generation(), focused_registry_generation);
            EXPECT_GT(focused_layout_generation, layout_generation);
            EXPECT_EQ(focused_registry_generation, registry_generation);
        }

        TEST_F(ViewportWorkspaceTest, NestedSplitCollapseKeepsCoverageAndUniqueLeaves) {
            const ViewId a = workspace_.primaryView();
            const auto b = workspace_.split(a, SplitAxis::Horizontal, 0.4f);
            ASSERT_TRUE(b);
            const auto c = workspace_.split(*b, SplitAxis::Vertical, 0.3f);
            ASSERT_TRUE(c);
            const auto d = workspace_.split(*c, SplitAxis::Horizontal, 0.6f);
            ASSERT_TRUE(d);
            EXPECT_EQ(workspace_.layout().leafCount(), 4u);
            EXPECT_TRUE(workspace_.layout().validate());

            auto snap = workspace_.snapshot(kOuter);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);

            ASSERT_TRUE(workspace_.close(*c));
            EXPECT_FALSE(workspace_.views().contains(*c));
            EXPECT_TRUE(workspace_.views().contains(*d));
            EXPECT_EQ(workspace_.layout().leafCount(), 3u);
            EXPECT_TRUE(workspace_.layout().validate());
            snap = workspace_.snapshot(kOuter);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);

            ASSERT_TRUE(workspace_.close(*d));
            ASSERT_TRUE(workspace_.close(*b));
            EXPECT_EQ(workspace_.layout().leafCount(), 1u);
            EXPECT_EQ(workspace_.primaryView(), a);
            snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 1u);
            EXPECT_EQ(snap.panes[0].rect, snap.outer);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);
        }

        TEST_F(ViewportWorkspaceTest, NestedSplittersExposeContainingNodeBounds) {
            const ViewId first = workspace_.primaryView();
            const auto second = workspace_.split(first, SplitAxis::Horizontal);
            ASSERT_TRUE(second);
            const auto third = workspace_.split(first, SplitAxis::Horizontal);
            ASSERT_TRUE(third);

            const auto snap = workspace_.snapshot(kOuter, 1, kDefaultDividerPixels);
            ASSERT_EQ(snap.splitters.size(), 2u);
            for (const auto& splitter : snap.splitters) {
                EXPECT_FALSE(splitter.parent.empty());
                EXPECT_GE(splitter.rect.x, splitter.parent.x);
                EXPECT_GE(splitter.rect.y, splitter.parent.y);
                EXPECT_LE(splitter.rect.right(), splitter.parent.right());
                EXPECT_LE(splitter.rect.bottom(), splitter.parent.bottom());
            }
            EXPECT_EQ(snap.splitters[0].parent, kOuter);
            EXPECT_LT(snap.splitters[1].parent.width, kOuter.width);
            EXPECT_EQ(snap.splitters[1].parent.height, kOuter.height);
        }

        TEST_F(ViewportWorkspaceTest, FocusMaximizeRestoreSurviveResize) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            const auto ids = workspace_.layout().leafIds();
            ASSERT_EQ(ids.size(), 4u);
            const ViewId target = ids[2];
            ASSERT_TRUE(workspace_.focus(target));
            ASSERT_TRUE(workspace_.maximize(target));

            auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 1u);
            EXPECT_EQ(snap.panes[0].id, target);
            EXPECT_EQ(snap.panes[0].rect, snap.outer);
            EXPECT_TRUE(snap.panes[0].focused);
            EXPECT_TRUE(snap.panes[0].maximized);
            EXPECT_TRUE(snap.splitters.empty());
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);

            const ViewRect resized{10, 20, 400, 240};
            snap = workspace_.snapshot(resized);
            ASSERT_EQ(snap.panes.size(), 1u);
            EXPECT_EQ(snap.panes[0].id, target);
            EXPECT_EQ(snap.focused, target);
            EXPECT_EQ(snap.maximized, target);
            EXPECT_EQ(snap.panes[0].rect, snap.outer);

            ASSERT_TRUE(workspace_.restoreMaximized());
            snap = workspace_.snapshot(resized);
            ASSERT_EQ(snap.panes.size(), 4u);
            EXPECT_EQ(snap.focused, target);
            EXPECT_FALSE(snap.maximized.has_value());
            bool saw_focus = false;
            for (const auto& pane : snap.panes) {
                if (pane.id == target) {
                    saw_focus = pane.focused;
                    EXPECT_FALSE(pane.maximized);
                }
            }
            EXPECT_TRUE(saw_focus);
            expectCoverage(snap.outer, workspace_.layout().solve(resized).panes, snap.splitters);
        }

        TEST_F(ViewportWorkspaceTest, StableIdsAreMonotonicAndNeverReused) {
            const ViewId a = workspace_.primaryView();
            const auto b = workspace_.split(a, SplitAxis::Horizontal);
            ASSERT_TRUE(b);
            EXPECT_GT(*b, a);
            const ViewId retired = *b;
            const auto next_after_create = workspace_.views().nextId();
            ASSERT_TRUE(workspace_.close(retired));
            EXPECT_FALSE(workspace_.views().contains(retired));
            EXPECT_GE(workspace_.views().nextId(), next_after_create);

            const auto c = workspace_.split(a, SplitAxis::Vertical);
            ASSERT_TRUE(c);
            EXPECT_NE(*c, retired);
            EXPECT_GT(*c, retired);
            EXPECT_NE(*c, a);
            EXPECT_NE(*c, kInvalidViewId);
        }

        TEST_F(ViewportWorkspaceTest, SplitFailureRollsBackRegistry) {
            const auto before_ids = workspace_.views().ids();
            const auto before_next = workspace_.views().nextId();
            const auto nan_split =
                workspace_.split(workspace_.primaryView(), SplitAxis::Horizontal,
                                 std::numeric_limits<float>::quiet_NaN());
            EXPECT_FALSE(nan_split);
            EXPECT_EQ(workspace_.views().ids(), before_ids);
            EXPECT_EQ(workspace_.layout().leafCount(), 1u);

            const auto missing = workspace_.split(ViewId{999}, SplitAxis::Horizontal);
            EXPECT_FALSE(missing);
            EXPECT_EQ(workspace_.views().ids(), before_ids);
            EXPECT_EQ(workspace_.views().nextId(), before_next);
        }

        TEST_F(ViewportWorkspaceTest, InvalidSplitAxisDoesNotAllocateOrMutateLayout) {
            const auto before_ids = workspace_.views().ids();
            const auto before_next = workspace_.views().nextId();
            const auto result = workspace_.split(
                workspace_.primaryView(), static_cast<SplitAxis>(255));
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(workspace_.views().ids(), before_ids);
            EXPECT_EQ(workspace_.views().nextId(), before_next);
            EXPECT_EQ(workspace_.layout().leafCount(), 1u);
            EXPECT_TRUE(workspace_.layout().validate());
        }

        TEST_F(ViewportWorkspaceTest, DefaultWorkspaceAllowsRepeatedSplitsBeyondFourViews) {
            ViewportWorkspace workspace({1600, 1000});
            ViewId source = workspace.primaryView();
            for (int i = 0; i < 11; ++i) {
                const auto created = workspace.split(
                    source, i % 2 == 0 ? SplitAxis::Horizontal : SplitAxis::Vertical);
                ASSERT_TRUE(created);
                source = *created;
            }
            EXPECT_EQ(workspace.views().size(), 12u);
            EXPECT_EQ(workspace.layout().leafCount(), 12u);
            const auto saved = workspace.exportState();
            ViewportWorkspace restored;
            ASSERT_TRUE(restored.importState(saved));
            EXPECT_EQ(restored.layout().leafIds(), workspace.layout().leafIds());
        }

        TEST_F(ViewportWorkspaceTest, DividerResizeChangesRectangles) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::DualHorizontal));
            auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.splitters.size(), 1u);
            const LayoutNodeId split = snap.splitters.front().split;
            const int left_before = snap.panes.front().rect.width;
            ASSERT_TRUE(workspace_.resize(split, 0.25f));
            snap = workspace_.snapshot(kOuter);
            EXPECT_LT(snap.panes.front().rect.width, left_before);
            expectCoverage(snap.outer, workspace_.layout().solve(kOuter).panes, snap.splitters);
            EXPECT_NEAR(*workspace_.layout().splitRatio(split), 0.25f, 1e-6f);
        }

        TEST_F(ViewportWorkspaceTest, PoseInertiaAndProjectionAreIndependent) {
            const ViewId source_id = workspace_.primaryView();
            ViewRecord* source = workspace_.findView(source_id);
            ASSERT_NE(source, nullptr);
            source->camera.camera.t = glm::vec3(9.0f, 8.0f, 7.0f);
            source->camera.camera.pivot = glm::vec3(1.0f, 0.0f, -1.0f);
            source->camera.camera.prePos = glm::vec2(12.0f, 34.0f);
            source->camera.camera.isOrbiting = true;
            source->camera.camera.advanceWasd(0.16f, true, false, false, false, false, false);
            ASSERT_TRUE(source->camera.camera.hasWasdMomentum());
            source->projection.focal_length_mm = 24.0f;
            source->projection.equirectangular = true;
            source->depth.far_plane = 42.0f;
            source->grid_plane = 2;

            const glm::vec3 source_t = source->camera.camera.t;
            const glm::vec3 source_pivot = source->camera.camera.pivot;
            const auto dest_id = workspace_.split(source_id, SplitAxis::Horizontal);
            ASSERT_TRUE(dest_id);
            ViewRecord* dest = workspace_.findView(*dest_id);
            ASSERT_NE(dest, nullptr);

            EXPECT_EQ(dest->camera.camera.t, source_t);
            EXPECT_EQ(dest->camera.camera.pivot, source_pivot);
            EXPECT_EQ(dest->camera.camera.prePos, glm::vec2(0.0f, 0.0f));
            EXPECT_FALSE(dest->camera.camera.isOrbiting);
            EXPECT_FALSE(dest->camera.camera.hasWasdMomentum());
            EXPECT_FALSE(dest->camera.camera.hasOrbitMomentum());
            EXPECT_FALSE(dest->camera.camera.hasPanMomentum());
            EXPECT_TRUE(source->camera.camera.hasWasdMomentum());
            EXPECT_TRUE(source->camera.camera.isOrbiting);
            EXPECT_EQ(source->camera.camera.prePos, glm::vec2(12.0f, 34.0f));
            EXPECT_EQ(source->camera.camera.t, source_t);

            EXPECT_EQ(dest->projection.focal_length_mm, 24.0f);
            EXPECT_TRUE(dest->projection.equirectangular);
            EXPECT_EQ(dest->depth.far_plane, 42.0f);
            EXPECT_EQ(dest->grid_plane, 2);

            ViewProjectionState changed = dest->projection;
            changed.orthographic = true;
            changed.focal_length_mm = 100.0f;
            changed.equirectangular = false;
            ASSERT_TRUE(workspace_.setViewProjection(*dest_id, changed));
            EXPECT_FALSE(workspace_.findView(source_id)->projection.orthographic);
            EXPECT_EQ(workspace_.findView(source_id)->projection.focal_length_mm, 24.0f);
            EXPECT_TRUE(workspace_.findView(source_id)->projection.equirectangular);
            EXPECT_TRUE(workspace_.findView(*dest_id)->projection.orthographic);

            const auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 2u);
            for (const auto& pane : snap.panes)
                EXPECT_EQ(pane.translation, source_t);
        }

        TEST_F(ViewportWorkspaceTest, EmptyAndTinyLayoutsAreSafe) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::Quad));
            for (const ViewRect outer : {ViewRect{0, 0, 0, 0}, ViewRect{0, 0, 1, 0},
                                         ViewRect{0, 0, 0, 1}, ViewRect{5, 5, 1, 1},
                                         ViewRect{0, 0, 2, 2}, ViewRect{-4, -4, 3, 3}}) {
                const auto snap = workspace_.snapshot(outer);
                EXPECT_EQ(snap.outer.width, std::max(outer.width, 0));
                EXPECT_EQ(snap.outer.height, std::max(outer.height, 0));
                EXPECT_EQ(snap.panes.size(), 4u);
                expectCoverage(snap.outer, workspace_.layout().solve(outer).panes, snap.splitters);
            }
        }

        TEST_F(ViewportWorkspaceTest, SnapshotHasNoPointersAndStableGenerations) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::DualHorizontal));
            const auto first = workspace_.snapshot(kOuter);
            const auto second = workspace_.snapshot(kOuter);
            EXPECT_EQ(first.layout_generation, second.layout_generation);
            EXPECT_EQ(first.registry_generation, second.registry_generation);
            ASSERT_EQ(first.panes.size(), 2u);
            EXPECT_NE(first.panes[0].id, kInvalidViewId);
            EXPECT_NE(first.panes[0].state_generation, 0u);
            ASSERT_TRUE(workspace_.focus(first.panes[1].id));
            const auto third = workspace_.snapshot(kOuter);
            EXPECT_GT(third.layout_generation, first.layout_generation);
            EXPECT_EQ(third.focused, first.panes[1].id);
        }

        TEST_F(ViewportWorkspaceTest, ActiveViewportSurvivesFocusOnEditorArea) {
            const ViewId viewport = workspace_.primaryView();
            const auto editor = workspace_.split(viewport, SplitAxis::Horizontal);
            ASSERT_TRUE(editor);
            ASSERT_TRUE(workspace_.setAreaEditor(*editor, "lfs.scene"));

            ASSERT_TRUE(workspace_.focus(*editor));
            EXPECT_EQ(workspace_.layout().focused(), std::optional<ViewId>(*editor));
            EXPECT_EQ(workspace_.activeViewport(), std::optional<ViewId>(viewport));
        }

        TEST_F(ViewportWorkspaceTest, SnapshotUsesCurrentSolveDimensionsAndExplicitFramebufferScale) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::DualHorizontal));
            workspace_.syncExtents({0, 0, 800, 600});

            const auto snap = workspace_.snapshot({10, 20, 400, 240}, kDefaultMinPanePixels,
                                                  kDefaultDividerPixels, {2.0f, 1.5f});
            ASSERT_EQ(snap.panes.size(), 2u);
            for (const PaneSnapshot& pane : snap.panes) {
                EXPECT_EQ(pane.window_size, glm::ivec2(pane.rect.width, pane.rect.height));
                EXPECT_EQ(pane.framebuffer_size,
                          glm::ivec2(pane.rect.width * 2, pane.rect.height * 3 / 2));
            }
        }

        TEST_F(ViewportWorkspaceTest, DatasetCameraBindingExpiresWhenPoseOrProjectionChanges) {
            const ViewId source = workspace_.primaryView();
            auto* record = workspace_.findView(source);
            ASSERT_NE(record, nullptr);
            record->camera_binding = ViewRecord::CameraBinding{
                .uid = 17,
                .camera_list_generation = 4,
                .rotation = record->camera.camera.R,
                .translation = record->camera.camera.t,
                .pivot = record->camera.camera.pivot,
                .projection = record->projection};

            auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 1u);
            ASSERT_TRUE(snap.panes.front().camera_uid);
            EXPECT_EQ(*snap.panes.front().camera_uid, 17);
            EXPECT_EQ(snap.panes.front().camera_binding_generation, 4u);

            record->camera.camera.t.x += 1.0f;
            snap = workspace_.snapshot(kOuter);
            EXPECT_FALSE(snap.panes.front().camera_uid);

            record->camera.camera.t.x -= 1.0f;
            record->projection.focal_length_mm += 1.0f;
            snap = workspace_.snapshot(kOuter);
            EXPECT_FALSE(snap.panes.front().camera_uid);
        }

        TEST_F(ViewportWorkspaceTest, DatasetCameraBindingClonesWithSplitView) {
            const ViewId source = workspace_.primaryView();
            auto* record = workspace_.findView(source);
            ASSERT_NE(record, nullptr);
            record->camera_binding = ViewRecord::CameraBinding{
                .uid = 23,
                .camera_list_generation = 8,
                .rotation = record->camera.camera.R,
                .translation = record->camera.camera.t,
                .pivot = record->camera.camera.pivot,
                .projection = record->projection};

            const auto split = workspace_.split(source, SplitAxis::Horizontal);
            ASSERT_TRUE(split);
            const auto snap = workspace_.snapshot(kOuter);
            ASSERT_EQ(snap.panes.size(), 2u);
            for (const auto& pane : snap.panes) {
                ASSERT_TRUE(pane.camera_uid);
                EXPECT_EQ(*pane.camera_uid, 23);
                EXPECT_EQ(pane.camera_binding_generation, 8u);
            }
        }

        TEST_F(ViewportWorkspaceTest, TransactionalInvalidImportsLeaveWorkspaceUntouched) {
            ASSERT_TRUE(workspace_.setPreset(LayoutPreset::DualHorizontal));
            const ViewId live = workspace_.primaryView();
            workspace_.findView(live)->camera.camera.t = glm::vec3(3.0f, 1.0f, 2.0f);
            const auto original = workspace_.exportState();
            const auto original_ids = workspace_.views().ids();
            const auto original_next = workspace_.views().nextId();

            auto bad_zero = original;
            bad_zero.views.front().id = kInvalidViewId;
            EXPECT_FALSE(workspace_.importState(bad_zero));

            auto bad_dup = original;
            if (bad_dup.views.size() >= 2)
                bad_dup.views[1].id = bad_dup.views[0].id;
            EXPECT_FALSE(workspace_.importState(bad_dup));

            auto bad_ratio = original;
            for (auto& node : bad_ratio.layout.nodes) {
                if (node.kind == LayoutNodeKind::Split)
                    node.ratio = std::numeric_limits<float>::infinity();
            }
            EXPECT_FALSE(workspace_.importState(bad_ratio));

            auto bad_nan_cam = original;
            bad_nan_cam.views.front().translation.x = std::numeric_limits<float>::quiet_NaN();
            EXPECT_FALSE(workspace_.importState(bad_nan_cam));

            auto bad_rotation = original;
            bad_rotation.views.front().rotation[0][0] = 2.0f;
            EXPECT_FALSE(workspace_.importState(bad_rotation));

            auto bad_focus = original;
            bad_focus.layout.focused = ViewId{123456};
            EXPECT_FALSE(workspace_.importState(bad_focus));

            auto bad_missing = original;
            bad_missing.views.pop_back();
            EXPECT_FALSE(workspace_.importState(bad_missing));

            auto bad_cycle = original;
            for (auto& node : bad_cycle.layout.nodes) {
                if (node.kind == LayoutNodeKind::Split)
                    node.first = node.id;
            }
            EXPECT_FALSE(workspace_.importState(bad_cycle));

            EXPECT_EQ(workspace_.views().ids(), original_ids);
            EXPECT_EQ(workspace_.views().nextId(), original_next);
            EXPECT_EQ(workspace_.findView(live)->camera.camera.t, glm::vec3(3.0f, 1.0f, 2.0f));
            EXPECT_EQ(workspace_.layout().leafCount(), 2u);
        }

        TEST_F(ViewportWorkspaceTest, SparseIdsImportAndNeverAliasRetired) {
            const ViewId first = workspace_.primaryView();
            const auto second = workspace_.split(first, SplitAxis::Horizontal);
            ASSERT_TRUE(second);
            ASSERT_TRUE(workspace_.close(*second));
            const ViewId retired = *second;
            const ViewId high_water = workspace_.views().nextId();

            WorkspacePersistentState state;
            state.format_version = 1;
            state.layout.root = 10;
            state.layout.next_node_id = 13;
            state.layout.focused = 7;
            LayoutNodeState split;
            split.id = 10;
            split.kind = LayoutNodeKind::Split;
            split.axis = SplitAxis::Horizontal;
            split.ratio = 0.5f;
            split.first = 11;
            split.second = 12;
            LayoutNodeState leaf_a;
            leaf_a.id = 11;
            leaf_a.kind = LayoutNodeKind::Leaf;
            leaf_a.view_id = 2;
            LayoutNodeState leaf_b;
            leaf_b.id = 12;
            leaf_b.kind = LayoutNodeKind::Leaf;
            leaf_b.view_id = 7;
            state.layout.nodes = {split, leaf_a, leaf_b};
            state.views = {defaultViewState(2), defaultViewState(7)};
            state.views[1].translation = glm::vec3(0.5f, 1.5f, 2.5f);
            state.next_view_id = 40;

            ASSERT_TRUE(workspace_.importState(state));
            EXPECT_TRUE(workspace_.views().contains(2));
            EXPECT_TRUE(workspace_.views().contains(7));
            EXPECT_FALSE(workspace_.views().contains(first));
            EXPECT_EQ(workspace_.layout().leafCount(), 2u);
            EXPECT_GE(workspace_.views().nextId(), 40u);
            EXPECT_GE(workspace_.views().nextId(), high_water);

            const auto created = workspace_.split(7, SplitAxis::Vertical);
            ASSERT_TRUE(created);
            EXPECT_NE(*created, retired);
            EXPECT_NE(*created, first);
            EXPECT_NE(*created, ViewId{2});
            EXPECT_NE(*created, ViewId{7});
            EXPECT_GE(*created, 40u);
            EXPECT_EQ(workspace_.findView(7)->camera.camera.t, glm::vec3(0.5f, 1.5f, 2.5f));
        }

        TEST_F(ViewportWorkspaceTest, InvalidSparseImportIsRejected) {
            WorkspacePersistentState state;
            state.format_version = 1;
            state.layout.root = 1;
            LayoutNodeState leaf;
            leaf.id = 1;
            leaf.kind = LayoutNodeKind::Leaf;
            leaf.view_id = 0;
            state.layout.nodes = {leaf};
            state.views = {defaultViewState(0)};
            const auto ids = workspace_.views().ids();
            EXPECT_FALSE(workspace_.importState(state));
            EXPECT_EQ(workspace_.views().ids(), ids);
        }

        TEST(WorkspaceLayoutStandalone, TreeUniqueness) {
            WorkspaceLayout layout = WorkspaceLayout::single(1);
            EXPECT_TRUE(layout.split(1, 2, SplitAxis::Horizontal, 0.5f));
            EXPECT_FALSE(layout.split(1, 2, SplitAxis::Vertical, 0.5f));
            EXPECT_FALSE(layout.split(1, 0, SplitAxis::Vertical, 0.5f));
            EXPECT_TRUE(layout.validate());

            const auto snap = layout.solve(kOuter);
            ASSERT_EQ(snap.panes.size(), 2u);
            EXPECT_TRUE(layout.close(1));
            EXPECT_EQ(layout.leafCount(), 1u);
            EXPECT_FALSE(layout.close(2));
            EXPECT_EQ(layout.leafCount(), 1u);
            EXPECT_TRUE(layout.contains(2));
        }

        TEST(ViewRegistryStandalone, CreateSetsFramebufferAndRejectsMissingClone) {
            ViewRegistry registry;
            const auto id = registry.create({100, 50});
            ASSERT_TRUE(id);
            const ViewRecord* record = registry.find(*id);
            ASSERT_NE(record, nullptr);
            EXPECT_EQ(record->camera.windowSize, glm::ivec2(100, 50));
            EXPECT_EQ(record->camera.frameBufferSize, glm::ivec2(100, 50));
            const auto missing = registry.create({100, 50}, ViewId{99});
            EXPECT_FALSE(missing);
            EXPECT_EQ(registry.size(), 1u);
        }

        TEST(ViewRegistryStandalone, RetiredIdHighWaterNeverWraps) {
            ViewRegistry registry;
            ViewPersistentState max_id = defaultViewState(std::numeric_limits<ViewId>::max());
            ASSERT_TRUE(registry.adopt(max_id));
            EXPECT_EQ(registry.nextId(), std::numeric_limits<ViewId>::max());
            const auto exhausted = registry.create({100, 50});
            EXPECT_FALSE(exhausted);
            EXPECT_EQ(exhausted.error().code(), lfs::ErrorCode::ResourceExhausted);
        }

    } // namespace
} // namespace lfs::vis
