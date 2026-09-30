/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/screen_host.hpp"
#include "gui/screen_host_logic.hpp"
#include "rendering/rendering_manager.hpp"
#include "screen/area_gestures.hpp"
#include "screen/screen.hpp"
#include "screen/screen_service.hpp"
#include "screen/view3d_space.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>

namespace lfs::vis::screen {

    namespace {

        const Rect kBounds{0.0f, 0.0f, 1200.0f, 800.0f};
        const LayoutMetrics kMetrics{2.0f, 48.0f, 32.0f};

        class ScreenTest : public ::testing::Test {
        protected:
            void SetUp() override {
                registerBuiltinEditorTypes(registry);
                registry.add(EditorType{.id = "lfs.histogram",
                                        .label = "Histogram",
                                        .placement = {.anchor = EditorPlacement::Anchor::ActiveView,
                                                      .side = Side::Bottom,
                                                      .fraction = 0.3f}});
            }

            EditorTypeRegistry registry;
        };

        AreaId areaShowing(const Screen& screen, std::string_view editor) { return screen.findEditor(editor); }

        const Rect& rectOf(const LayoutGeometry& g, const AreaId id) {
            static const Rect empty{};
            const auto* found = g.find(id);
            return found ? found->rect : empty;
        }

    } // namespace

    TEST_F(ScreenTest, DefaultScreenHasViewportSceneAndProperties) {
        const Screen screen = Screen::makeDefault(registry);
        EXPECT_EQ(screen.areas().size(), 3u);
        const AreaId view = areaShowing(screen, editors::kView3D);
        const AreaId scene = areaShowing(screen, editors::kScene);
        const AreaId properties = areaShowing(screen, editors::kProperties);
        ASSERT_TRUE(view.valid() && scene.valid() && properties.valid());
        EXPECT_EQ(screen.activeView(), view);
        ASSERT_NE(screen.view(view), nullptr);

        const auto g = screen.solve(kBounds, kMetrics);
        EXPECT_GT(rectOf(g, view).w, rectOf(g, scene).w * 2.0f);
        EXPECT_LT(rectOf(g, scene).y, rectOf(g, properties).y);
        EXPECT_FLOAT_EQ(rectOf(g, scene).x, rectOf(g, properties).x);
    }

    TEST_F(ScreenTest, SplittingAViewCopiesItsCameraIndependently) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        screen.view(view)->camera.camera.t = glm::vec3(1.0f, 2.0f, 3.0f);
        screen.view(view)->settings.show_grid = false;

        const AreaId added = screen.split(view, SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(added.valid());
        auto* copy = screen.view(added);
        ASSERT_NE(copy, nullptr);
        EXPECT_EQ(copy->camera.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_FALSE(copy->settings.show_grid);

        copy->camera.camera.t = glm::vec3(9.0f);
        copy->settings.orthographic = true;
        EXPECT_EQ(screen.view(view)->camera.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_FALSE(screen.view(view)->settings.orthographic);
        EXPECT_EQ(screen.views().size(), 2u);
    }

    TEST_F(ScreenTest, SplittingASingleInstanceEditorOpensAViewport) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId scene = areaShowing(screen, editors::kScene);
        const AreaId added = screen.split(scene, SplitAxis::Rows, 0.5f);
        ASSERT_TRUE(added.valid());
        EXPECT_EQ(screen.area(added)->editor, editors::kView3D);
        EXPECT_EQ(screen.findEditor(editors::kScene), scene);
    }

    TEST_F(ScreenTest, LastViewportCannotBeClosedJoinedAwayOrReplaced) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        const AreaId scene = areaShowing(screen, editors::kScene);
        EXPECT_FALSE(screen.canClose(view));
        EXPECT_FALSE(screen.close(view));
        EXPECT_FALSE(screen.setEditor(view, editors::kConsole));
        EXPECT_EQ(screen.area(view)->editor, editors::kView3D);

        const AreaId second = screen.split(view, SplitAxis::Rows, 0.5f);
        ASSERT_TRUE(second.valid());
        EXPECT_TRUE(screen.canJoin(second, view));
        ASSERT_TRUE(screen.join(second, view));
        EXPECT_EQ(screen.activeView(), second) << "the active view moves to a surviving viewport";
        EXPECT_FALSE(screen.canClose(second));
        EXPECT_TRUE(screen.close(scene));
    }

    TEST_F(ScreenTest, SingleInstanceEditorTradesPlacesInsteadOfDuplicating) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        const AreaId second = screen.split(view, SplitAxis::Columns, 0.5f);
        const AreaId scene = areaShowing(screen, editors::kScene);
        const auto before = screen.solve(kBounds, kMetrics);

        ASSERT_TRUE(screen.setEditor(second, editors::kScene));
        const auto after = screen.solve(kBounds, kMetrics);
        EXPECT_EQ(rectOf(after, scene), rectOf(before, second)) << "the scene editor now shows where the view was";
        EXPECT_EQ(rectOf(after, second), rectOf(before, scene));
        EXPECT_EQ(screen.area(second)->editor, editors::kView3D);
        EXPECT_EQ(screen.area(scene)->editor, editors::kScene);
    }

    TEST_F(ScreenTest, SwitchingEditorKeepsEachEditorsState) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        const AreaId second = screen.split(view, SplitAxis::Columns, 0.5f);
        screen.view(second)->camera.camera.t = glm::vec3(4.0f, 5.0f, 6.0f);
        ASSERT_TRUE(screen.setEditor(second, editors::kConsole));
        EXPECT_EQ(screen.view(second), nullptr);
        EXPECT_EQ(screen.views().size(), 1u);
        ASSERT_TRUE(screen.setEditor(second, editors::kView3D));
        ASSERT_NE(screen.view(second), nullptr);
        EXPECT_EQ(screen.view(second)->camera.camera.t, glm::vec3(4.0f, 5.0f, 6.0f));
    }

    TEST_F(ScreenTest, UnknownEditorsAreRejected) {
        Screen screen = Screen::makeDefault(registry);
        EXPECT_FALSE(screen.setEditor(areaShowing(screen, editors::kScene), "no.such.editor"));
        EXPECT_FALSE(screen.openEditor("no.such.editor").valid());
    }

    TEST_F(ScreenTest, OpenEditorUsesPlacementAndReusesSingleInstances) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        const auto before = screen.solve(kBounds, kMetrics);

        const AreaId histogram = screen.openEditor("lfs.histogram");
        ASSERT_TRUE(histogram.valid());
        const auto after = screen.solve(kBounds, kMetrics);
        EXPECT_FLOAT_EQ(rectOf(after, histogram).x, rectOf(before, view).x);
        EXPECT_GT(rectOf(after, histogram).y, rectOf(after, view).y) << "opens below the active view";
        EXPECT_EQ(screen.openEditor("lfs.histogram"), histogram);

        ASSERT_TRUE(screen.closeEditor("lfs.histogram"));
        EXPECT_FALSE(screen.findEditor("lfs.histogram").valid());
        EXPECT_EQ(rectOf(screen.solve(kBounds, kMetrics), view), rectOf(before, view));
    }

    TEST_F(ScreenTest, ClosedSceneReopensAboveProperties) {
        Screen screen = Screen::makeDefault(registry);
        const auto before = screen.solve(kBounds, kMetrics);
        const AreaId properties = areaShowing(screen, editors::kProperties);
        ASSERT_TRUE(screen.closeEditor(editors::kScene));
        const AreaId scene = screen.openEditor(editors::kScene);
        ASSERT_TRUE(scene.valid());
        const auto after = screen.solve(kBounds, kMetrics);
        EXPECT_FLOAT_EQ(rectOf(after, scene).x, rectOf(before, properties).x);
        EXPECT_LT(rectOf(after, scene).y, rectOf(after, properties).y);
    }

    TEST_F(ScreenTest, OpenWithoutAnchorUsesTheScreenEdge) {
        Screen screen = Screen::makeDefault(registry);
        ASSERT_TRUE(screen.closeEditor(editors::kScene));
        ASSERT_TRUE(screen.closeEditor(editors::kProperties));
        const AreaId properties = screen.openEditor(editors::kProperties);
        ASSERT_TRUE(properties.valid());
        const auto g = screen.solve(kBounds, kMetrics);
        EXPECT_FLOAT_EQ(rectOf(g, properties).right(), kBounds.right());
        EXPECT_FLOAT_EQ(rectOf(g, properties).h, kBounds.h);
    }

    TEST_F(ScreenTest, MaximizeTogglesAndClearsOnStructuralChange) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        ASSERT_TRUE(screen.toggleMaximized(view));
        const auto g = screen.solve(kBounds, kMetrics);
        ASSERT_EQ(g.areas.size(), 1u);
        EXPECT_EQ(g.areas[0].rect, kBounds);
        ASSERT_TRUE(screen.toggleMaximized(view));
        EXPECT_FALSE(screen.maximized().valid());

        ASSERT_TRUE(screen.toggleMaximized(view));
        screen.split(view, SplitAxis::Rows, 0.5f);
        EXPECT_FALSE(screen.maximized().valid());
    }

    TEST_F(ScreenTest, GenerationChangesOnlyWithStructure) {
        Screen screen = Screen::makeDefault(registry);
        const auto g0 = screen.generation();
        screen.view(screen.activeView())->camera.camera.t = glm::vec3(1.0f);
        EXPECT_EQ(screen.generation(), g0);
        screen.split(screen.activeView(), SplitAxis::Rows, 0.5f);
        EXPECT_GT(screen.generation(), g0);
    }

    TEST_F(ScreenTest, SaveLoadRoundTrip) {
        Screen screen = Screen::makeDefault(registry);
        const AreaId view = screen.activeView();
        const AreaId second = screen.split(view, SplitAxis::Rows, 0.35f);
        screen.view(second)->camera.setViewMatrix(glm::mat3(1.0f), glm::vec3(1.0f, 2.0f, 3.0f));
        screen.view(second)->settings.orthographic = true;
        screen.view(second)->settings.ortho_scale = 42.0f;
        screen.view(second)->settings.show_camera_frustums = true;
        screen.openEditor("lfs.histogram");
        ASSERT_TRUE(screen.setActiveView(second));

        const auto json = screen.save();
        const auto restored = Screen::load(json, registry);
        ASSERT_TRUE(restored.has_value());
        EXPECT_EQ(restored->layout(), screen.layout());
        EXPECT_EQ(restored->activeView(), second);
        const auto* copy = restored->view(second);
        ASSERT_NE(copy, nullptr);
        EXPECT_EQ(copy->camera.camera.t, glm::vec3(1.0f, 2.0f, 3.0f));
        EXPECT_TRUE(copy->settings.orthographic);
        EXPECT_FLOAT_EQ(copy->settings.ortho_scale, 42.0f);
        EXPECT_TRUE(copy->settings.show_camera_frustums);
        EXPECT_EQ(restored->area(areaShowing(screen, "lfs.histogram"))->editor, "lfs.histogram");

        Screen grown = *restored;
        const AreaId fresh = grown.split(second, SplitAxis::Columns, 0.5f);
        for (const AreaId id : screen.areas())
            EXPECT_NE(fresh, id) << "ids are never reused after a load";
    }

    TEST_F(ScreenTest, LoadRejectsInconsistentScreens) {
        const Screen screen = Screen::makeDefault(registry);
        auto json = screen.save();

        auto no_view = json;
        for (auto& area : no_view["areas"]) {
            if (area["editor"] == "view3d")
                area["editor"] = "console";
        }
        EXPECT_FALSE(Screen::load(no_view, registry)) << "a screen needs a 3D viewport";

        auto missing_area = json;
        missing_area["areas"].erase(missing_area["areas"].begin());
        EXPECT_FALSE(Screen::load(missing_area, registry));

        auto bad_space = json;
        for (auto& area : bad_space["areas"]) {
            if (area["editor"] == "view3d")
                area["spaces"]["view3d"]["camera"]["translation"] = "north";
        }
        EXPECT_FALSE(Screen::load(bad_space, registry));

        auto unknown_editor = json;
        unknown_editor["areas"][1]["editor"] = "plugin.gone";
        const auto restored = Screen::load(unknown_editor, registry);
        ASSERT_TRUE(restored) << "an editor from a missing plugin keeps its area";
        EXPECT_TRUE(restored->findEditor("plugin.gone").valid());
    }

    TEST_F(ScreenTest, LoadRejectsDuplicateSingleInstanceEditors) {
        auto json = Screen::makeDefault(registry).save();
        for (auto& area : json["areas"]) {
            if (area["editor"] == editors::kProperties)
                area["editor"] = std::string(editors::kScene);
        }
        EXPECT_FALSE(Screen::load(json, registry));
    }

    TEST_F(ScreenTest, LoadRejectsIdsThatExhaustAreaOrSplitAllocators) {
        Screen only_view = Screen::makeDefault(registry);
        ASSERT_TRUE(only_view.closeEditor(editors::kScene));
        ASSERT_TRUE(only_view.closeEditor(editors::kProperties));
        auto json = only_view.save();
        json["areas"][0]["id"] = std::numeric_limits<std::uint32_t>::max();
        json["layout"]["area"] = std::numeric_limits<std::uint32_t>::max();
        EXPECT_FALSE(Screen::load(json, registry));

        json = Screen::makeDefault(registry).save();
        json["layout"]["split"] = std::numeric_limits<std::uint32_t>::max();
        EXPECT_FALSE(Screen::load(json, registry));
    }

    TEST_F(ScreenTest, QuadViewCollapsesInsideDefaultNaryParent) {
        Screen screen = Screen::makeDefault(registry);
        const auto scene = screen.findEditor(editors::kScene);
        const auto properties = screen.findEditor(editors::kProperties);
        const auto view = screen.activeView();
        const auto before = screen.solve(kBounds, kMetrics);

        ASSERT_TRUE(screen.toggleQuadView(view, 800.0f));
        ASSERT_EQ(screen.views().size(), 4u);
        ASSERT_TRUE(screen.toggleQuadView(screen.views()[2], 800.0f));

        EXPECT_EQ(screen.views().size(), 1u);
        EXPECT_EQ(screen.findEditor(editors::kScene), scene);
        EXPECT_EQ(screen.findEditor(editors::kProperties), properties);
        const auto after = screen.solve(kBounds, kMetrics);
        EXPECT_EQ(rectOf(after, scene), rectOf(before, scene));
        EXPECT_EQ(rectOf(after, properties), rectOf(before, properties));

        Screen single = Screen::makeDefault(registry);
        ASSERT_TRUE(single.closeEditor(editors::kScene));
        ASSERT_TRUE(single.closeEditor(editors::kProperties));
        const auto only = single.activeView();
        ASSERT_TRUE(only.valid());
        ASSERT_TRUE(single.toggleQuadView(only, 800.0f));
        ASSERT_EQ(single.views().size(), 4u);
        ASSERT_TRUE(single.toggleQuadView(single.views()[0], 800.0f));
        EXPECT_EQ(single.views().size(), 1u);
    }

    TEST(ScreenHostLogic, HiddenOverlayConsumesDirtyState) {
        bool dirty = true;
        gui::screen_host_detail::clearDirtyWhenOverlayHidden(false, dirty);
        EXPECT_FALSE(dirty);
        dirty = true;
        gui::screen_host_detail::clearDirtyWhenOverlayHidden(true, dirty);
        EXPECT_TRUE(dirty);
    }

    TEST(ScreenHostLogic, HiddenUiTargetsTheFullDisplayedViewport) {
        const AreaId active{7};
        const AreaId stale_area{2};
        const Rect full_window{0.0f, 0.0f, 1920.0f, 1080.0f};
        EXPECT_EQ(gui::screen_host_detail::displayedViewAt(true, active, {}, full_window, 1200.0f, 700.0f),
                  active);
        EXPECT_FALSE(gui::screen_host_detail::displayedViewAt(true, active, {}, full_window, 1920.0f, 700.0f)
                         .valid());
        EXPECT_EQ(gui::screen_host_detail::displayedViewAt(false, active, stale_area, full_window, 1200.0f, 700.0f),
                  stale_area);
    }

    TEST(ScreenHostLogic, DisappearedPanelEditorIsIdentifiedFromHistory) {
        EXPECT_TRUE(gui::screen_host_detail::shouldCloseMissingPanelEditor(false, true));
        EXPECT_FALSE(gui::screen_host_detail::shouldCloseMissingPanelEditor(false, false));
        EXPECT_FALSE(gui::screen_host_detail::shouldCloseMissingPanelEditor(true, true));
    }

    TEST(ScreenHostLogic, CornerGestureZoneWinsInsideHeader) {
        ScreenLayout layout(AreaId{1});
        const auto geometry = layout.solve(kBounds, kMetrics);
        EXPECT_TRUE(gui::screen_host_detail::cornerGestureZone(geometry, false, 12.0f, 2.0f, 2.0f));
        EXPECT_FALSE(gui::screen_host_detail::cornerGestureZone(geometry, true, 12.0f, 2.0f, 2.0f));
        EXPECT_FALSE(gui::screen_host_detail::cornerGestureZone(geometry, false, 12.0f, 100.0f, 100.0f));
    }

    TEST_F(ScreenTest, ViewSettingsJsonRoundTripsEveryField) {
        ViewSettings s;
        s.focal_length_mm = 50.0f;
        s.equirectangular = true;
        s.orthographic = true;
        s.ortho_scale = 12.0f;
        s.show_coord_axes = true;
        s.axes_size = 3.0f;
        s.axes_visibility = {true, false, true};
        s.show_grid = false;
        s.grid_plane = 2;
        s.grid_opacity = 0.25f;
        s.point_cloud_mode = true;
        s.voxel_size = 0.05f;
        s.show_rings = true;
        s.ring_width = 0.02f;
        s.show_center_markers = true;
        s.show_camera_frustums = true;
        s.camera_frustum_scale = 0.5f;
        s.show_pivot = true;
        s.split_view_mode = SplitViewMode::GTComparison;
        s.gt_comparison_mode = GTComparisonMode::Depth;
        s.split_position = 0.3f;
        s.split_view_offset = 4;
        s.depth_view = true;
        s.depth_view_min = 1.0f;
        s.depth_view_max = 20.0f;
        s.depth_visualization_mode = lfs::rendering::DepthVisualizationMode::Grayscale;
        s.depth_filter_enabled = true;
        s.depth_filter_min = glm::vec3(-1.0f, -2.0f, 0.5f);
        s.depth_filter_max = glm::vec3(1.0f, 2.0f, 30.0f);
        s.depth_filter_transform = lfs::geometry::EuclideanTransform(glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                                                                     glm::vec3(0.5f, 0.25f, 2.0f));
        s.depth_filter_scale_x = 0.5f;
        s.depth_filter_scale_y = 0.6f;
        s.depth_filter_offset_x = 0.1f;
        s.depth_filter_offset_y = -0.1f;
        s.depth_filter_viz_mode = 2;

        const auto restored = viewSettingsFromJson(viewSettingsToJson(s), ViewSettings{});
        ASSERT_TRUE(restored);
        EXPECT_EQ(viewSettingsToJson(*restored), viewSettingsToJson(s));
        EXPECT_NE(viewSettingsToJson(ViewSettings{}), viewSettingsToJson(s));
        EXPECT_FALSE(viewSettingsFromJson(nlohmann::json{{"grid_plane", 7}}, ViewSettings{}));
        EXPECT_FALSE(viewSettingsFromJson(nlohmann::json{{"orthographic", 1}}, ViewSettings{}));
    }

    TEST(ViewLabel, NamesAxisAlignedAndUserViews) {
        View3DSpace view;
        view.camera.camera.setAxisAlignedView(1, false);
        view.settings.orthographic = true;
        EXPECT_EQ(viewLabel(view), "Top Orthographic");
        view.camera.camera.setAxisAlignedView(2, false);
        view.settings.orthographic = false;
        EXPECT_EQ(viewLabel(view), "Front Perspective");
        view.camera.camera.setAxisAlignedView(0, true);
        EXPECT_EQ(viewLabel(view), "Left Perspective");
        view.camera.setViewMatrix(lfs::rendering::makeVisualizerLookAtRotation(glm::vec3(3.0f, 2.0f, 1.0f),
                                                                               glm::vec3(0.0f)),
                                  glm::vec3(3.0f, 2.0f, 1.0f));
        EXPECT_EQ(viewLabel(view), "User Perspective");
    }

    // ---- Gestures ---------------------------------------------------------

    class GestureTest : public ScreenTest {
    protected:
        void SetUp() override {
            ScreenTest::SetUp();
            screen = std::make_unique<Screen>(Screen::makeDefault(registry));
            view = screen->activeView();
        }

        LayoutGeometry geometry() const { return screen->solve(kBounds, kMetrics); }

        void apply(const GestureCommand& c) {
            switch (c.kind) {
            case GestureCommand::Kind::MoveDivider: screen->moveDivider(c.divider, c.position); break;
            case GestureCommand::Kind::Split: screen->split(c.area, c.axis, c.fraction, c.new_first); break;
            case GestureCommand::Kind::Join: screen->join(c.area, c.other); break;
            case GestureCommand::Kind::Swap: screen->swap(c.area, c.other); break;
            case GestureCommand::Kind::None: break;
            }
        }

        std::unique_ptr<Screen> screen;
        AreaId view;
        AreaGestures gestures;
    };

    TEST_F(GestureTest, DividerDragResizesLive) {
        auto g = geometry();
        const Rect before = rectOf(g, view);
        const auto& divider = g.dividers.front();
        const float x = divider.rect.x + 1.0f;
        const float y = 400.0f;
        EXPECT_EQ(gestures.hoverCursor(g, x, y), GestureCursor::ResizeColumns);
        ASSERT_TRUE(gestures.press(g, x, y, false));
        for (float px = x; px >= x - 200.0f; px -= 50.0f) {
            apply(gestures.move(geometry(), *screen, px, y));
        }
        (void)gestures.release(geometry(), *screen, x - 200.0f, y);
        EXPECT_FALSE(gestures.active());
        EXPECT_NEAR(rectOf(geometry(), view).w, before.w - 200.0f, 1.0f);
    }

    TEST_F(GestureTest, CornerDragInwardSplitsAtThePointer) {
        const auto g = geometry();
        const Rect r = rectOf(g, view);
        // Bottom-left corner of the viewport, dragged right: a vertical
        // divider follows the pointer and the new area is on the left.
        const float x0 = r.x + 2.0f;
        const float y0 = r.bottom() - 2.0f;
        EXPECT_EQ(gestures.hoverCursor(g, x0, y0), GestureCursor::Crosshair);
        ASSERT_TRUE(gestures.press(g, x0, y0, false));
        (void)gestures.move(g, *screen, x0 + 30.0f, y0 - 2.0f);
        EXPECT_EQ(gestures.preview().kind, GesturePreview::Kind::Split);
        (void)gestures.move(g, *screen, r.x + 300.0f, y0 - 2.0f);
        EXPECT_NEAR(gestures.preview().second.w, 300.0f, 1.0f);
        apply(gestures.release(g, *screen, r.x + 300.0f, y0 - 2.0f));

        ASSERT_EQ(screen->views().size(), 2u);
        const auto after = geometry();
        const AreaId added = screen->views().front() == view ? screen->views().back() : screen->views().front();
        EXPECT_NEAR(rectOf(after, added).w, 300.0f, 2.0f);
        EXPECT_LT(rectOf(after, added).x, rectOf(after, view).x);
    }

    TEST_F(GestureTest, VerticalCornerDragSplitsIntoRows) {
        const auto g = geometry();
        const Rect r = rectOf(g, view);
        const float x0 = r.right() - 2.0f;
        const float y0 = r.y + 2.0f;
        ASSERT_TRUE(gestures.press(g, x0, y0, false));
        (void)gestures.move(g, *screen, x0 - 1.0f, y0 + 40.0f);
        apply(gestures.release(g, *screen, x0 - 1.0f, r.y + 250.0f));
        ASSERT_EQ(screen->views().size(), 2u);
        const auto after = geometry();
        EXPECT_FLOAT_EQ(rectOf(after, screen->views()[0]).x, rectOf(after, screen->views()[1]).x);
    }

    TEST_F(GestureTest, CornerDragOutwardJoinsTheNeighbour) {
        const AreaId scene = screen->findEditor(editors::kScene);
        const AreaId properties = screen->findEditor(editors::kProperties);
        const auto g = geometry();
        const Rect r = rectOf(g, properties);
        // From the properties' top-right corner upwards into the scene area.
        const float x0 = r.right() - 2.0f;
        const float y0 = r.y + 2.0f;
        ASSERT_TRUE(gestures.press(g, x0, y0, false));
        (void)gestures.move(g, *screen, x0 - 1.0f, y0 - 30.0f);
        EXPECT_EQ(gestures.preview().kind, GesturePreview::Kind::Join);
        EXPECT_EQ(gestures.preview().target, scene);
        EXPECT_TRUE(gestures.preview().allowed);
        EXPECT_EQ(gestures.preview().direction, Side::Top);
        apply(gestures.release(g, *screen, x0 - 1.0f, y0 - 60.0f));
        EXPECT_FALSE(screen->findEditor(editors::kScene).valid());
        EXPECT_EQ(rectOf(geometry(), properties).h, kBounds.h);
    }

    TEST_F(GestureTest, JoinIntoANonNeighbourIsRefused) {
        const AreaId properties = screen->findEditor(editors::kProperties);
        const auto g = geometry();
        const Rect r = rectOf(g, properties);
        // The viewport's right edge spans both scene and properties, so the
        // properties area cannot absorb it.
        ASSERT_TRUE(gestures.press(g, r.x + 2.0f, r.bottom() - 2.0f, false));
        (void)gestures.move(g, *screen, r.x - 40.0f, r.bottom() - 3.0f);
        EXPECT_EQ(gestures.preview().kind, GesturePreview::Kind::Join);
        EXPECT_FALSE(gestures.preview().allowed);
        const auto command = gestures.release(g, *screen, r.x - 80.0f, r.bottom() - 3.0f);
        EXPECT_EQ(command.kind, GestureCommand::Kind::None);
        EXPECT_EQ(screen->areas().size(), 3u);
    }

    TEST_F(GestureTest, SwapModifierExchangesAreas) {
        const AreaId scene = screen->findEditor(editors::kScene);
        const auto g = geometry();
        const Rect view_rect = rectOf(g, view);
        const Rect scene_rect = rectOf(g, scene);
        ASSERT_TRUE(gestures.press(g, view_rect.x + 2.0f, view_rect.y + 2.0f, true));
        (void)gestures.move(g, *screen, scene_rect.x + 50.0f, scene_rect.y + 50.0f);
        EXPECT_EQ(gestures.preview().kind, GesturePreview::Kind::Swap);
        EXPECT_TRUE(gestures.preview().allowed);
        apply(gestures.release(g, *screen, scene_rect.x + 50.0f, scene_rect.y + 50.0f));
        EXPECT_EQ(rectOf(geometry(), scene), view_rect);
    }

    TEST_F(GestureTest, CancelLeavesTheScreenUntouched) {
        const auto g = geometry();
        const Rect r = rectOf(g, view);
        const auto before = screen->save();
        ASSERT_TRUE(gestures.press(g, r.x + 2.0f, r.y + 2.0f, false));
        (void)gestures.move(g, *screen, r.x + 200.0f, r.y + 3.0f);
        gestures.cancel();
        EXPECT_FALSE(gestures.active());
        EXPECT_EQ(gestures.release(g, *screen, r.x + 200.0f, r.y + 3.0f).kind, GestureCommand::Kind::None);
        EXPECT_EQ(screen->save(), before);
    }

    TEST_F(GestureTest, SmallDragsAndMaximizedScreensDoNothing) {
        auto g = geometry();
        const Rect r = rectOf(g, view);
        ASSERT_TRUE(gestures.press(g, r.x + 2.0f, r.y + 2.0f, false));
        EXPECT_EQ(gestures.release(g, *screen, r.x + 4.0f, r.y + 3.0f).kind, GestureCommand::Kind::None);

        screen->toggleMaximized(view);
        g = geometry();
        EXPECT_FALSE(gestures.press(g, 2.0f, 2.0f, false));
        EXPECT_EQ(gestures.hoverCursor(g, 2.0f, 2.0f), GestureCursor::Default);
    }

    TEST_F(GestureTest, SplitIsRefusedWhenTheAreaIsTooSmall) {
        const AreaId scene = screen->findEditor(editors::kScene);
        const auto small = Rect{0.0f, 0.0f, 400.0f, 180.0f};
        const auto g = screen->solve(small, kMetrics);
        const Rect r = rectOf(g, scene);
        ASSERT_GT(r.w, 0.0f);
        if (!gestures.press(g, r.x + 2.0f, r.y + 2.0f, false))
            GTEST_SKIP() << "corner zone suppressed on a tiny area";
        (void)gestures.move(g, *screen, r.x + 20.0f, r.y + 3.0f);
        EXPECT_FALSE(gestures.preview().allowed);
        EXPECT_EQ(gestures.release(g, *screen, r.x + 20.0f, r.y + 3.0f).kind, GestureCommand::Kind::None);
    }

} // namespace lfs::vis::screen

namespace lfs::vis::screen {
    TEST(ViewSettingsOwnershipTest, ComposesSceneWithEachViewsOwnSettings) {
        ScreenService source;
        const auto first = source.screen().activeView();
        const auto second = source.screen().split(first, SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(second.valid());
        RenderingManager renderer(source);
        source.editViewSettings(first.value, [](ViewSettings& s) { s.focal_length_mm = 21.0f; });
        source.editViewSettings(second.value, [](ViewSettings& s) { s.focal_length_mm = 80.0f; s.point_cloud_mode = true; });
        auto settings = renderer.getSettings();
        settings.background_color = {0.1f, 0.2f, 0.3f};
        renderer.updateSettings(settings);
        const auto other = renderer.settingsForView(second.value);
        EXPECT_FLOAT_EQ(other.focal_length_mm, 80.0f);
        EXPECT_TRUE(other.point_cloud_mode);
        EXPECT_EQ(other.background_color, settings.background_color);
        source.screen().setActiveView(second);
        EXPECT_FLOAT_EQ(renderer.getSettings().focal_length_mm, 80.0f);
        settings = renderer.getSettings();
        settings.focal_length_mm = 55.0f;
        renderer.updateSettings(settings);
        EXPECT_FLOAT_EQ(source.viewSettings(second.value)->focal_length_mm, 55.0f);
        EXPECT_FLOAT_EQ(source.viewSettings(first.value)->focal_length_mm, 21.0f);
    }

    TEST(ScreenAreaQueryTest, StructuralMutationReturnsRectsSolvedAgainstCurrentWorkArea) {
        ScreenService source;
        gui::ScreenHost host(source);
        const Rect work{17.0f, 31.0f, 1280.0f, 720.0f};
        host.layout(work, 1.0f);

        const auto added = source.screen().split(source.screen().activeView(), SplitAxis::Columns, 0.4f);
        ASSERT_TRUE(added.valid());
        const auto rect = host.currentAreaRect(added);

        EXPECT_GE(rect.x, work.x);
        EXPECT_GE(rect.y, work.y);
        EXPECT_GT(rect.w, 0.0f);
        EXPECT_GT(rect.h, 0.0f);
        EXPECT_LE(rect.x + rect.w, work.x + work.w);
        EXPECT_LE(rect.y + rect.h, work.y + work.h);
    }

    TEST(ViewSettingsOwnershipTest, ComparisonBelongsToOnlyOneViewAndIsNotCloned) {
        ScreenService source;
        const auto first = source.screen().activeView();
        source.editViewSettings(first.value, [](ViewSettings& s) { s.split_view_mode = SplitViewMode::GTComparison; });
        const auto second = source.screen().split(first, SplitAxis::Columns, 0.5f);
        ASSERT_TRUE(second.valid());
        EXPECT_EQ(source.viewSettings(second.value)->split_view_mode, SplitViewMode::Disabled);
        source.editViewSettings(second.value, [](ViewSettings& s) { s.split_view_mode = SplitViewMode::PLYComparison; });
        EXPECT_EQ(source.viewSettings(first.value)->split_view_mode, SplitViewMode::Disabled);
        EXPECT_EQ(source.viewSettings(second.value)->split_view_mode, SplitViewMode::PLYComparison);
    }
} // namespace lfs::vis::screen
