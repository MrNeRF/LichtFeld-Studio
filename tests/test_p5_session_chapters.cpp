/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/scene.hpp"
#include "gui/editor/python_editor.hpp"
#include "io/project_document.hpp"
#include "p5_matrix_rows.hpp"
#include "p8_matrix_session_fixture.hpp"
#include "project/session_state.hpp"
#include "sequencer/timeline.hpp"
#include "training/control/command_api.hpp"
#include "training/training_manager.hpp"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

namespace {

    namespace fs = std::filesystem;
    using Json = lfs::io::JsonChapterDom::Json;
    using namespace lfs::io::project;
    using namespace lfs::vis::project;

    template <typename T>
    T require_result(lfs::Result<T> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(
                    result.error()));
        }
        return std::move(*result);
    }

    void require_status(lfs::Result<void> result) {
        if (!result) {
            throw std::runtime_error(
                lfs::format_for_developer(
                    result.error()));
        }
    }

    Json root_json(
        const lfs::io::JsonChapterDom& dom) {
        return Json::parse(dom.dump());
    }

    void overwrite_u32_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const std::uint32_t value) {
        ASSERT_LE(offset + sizeof(value), bytes.size());
        for (std::size_t byte = 0;
             byte < sizeof(value); ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(
                    (value >> (byte * 8u)) & 0xffu);
        }
    }

    void overwrite_u64_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const std::uint64_t value) {
        ASSERT_LE(offset + sizeof(value), bytes.size());
        for (std::size_t byte = 0;
             byte < sizeof(value); ++byte) {
            bytes[offset + byte] =
                static_cast<std::byte>(
                    (value >> (byte * 8u)) & 0xffu);
        }
    }

    void overwrite_f32_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const float value) {
        overwrite_u32_le(
            bytes, offset,
            std::bit_cast<std::uint32_t>(value));
    }

    void overwrite_f64_le(
        std::vector<std::byte>& bytes,
        const std::size_t offset,
        const double value) {
        overwrite_u64_le(
            bytes, offset,
            std::bit_cast<std::uint64_t>(value));
    }

    class TemporaryProjectDirectory {
    public:
        TemporaryProjectDirectory()
            : path(
                  fs::temp_directory_path() /
                  ("licht-p5-" +
                   lfs::core::generate_uuid_v4()
                       .to_string())) {
            fs::create_directories(path);
        }

        ~TemporaryProjectDirectory() {
            std::error_code error;
            fs::remove_all(path, error);
        }

        fs::path path;
    };

    PanelCameraProjectState rolled_camera(
        const float tag) {
        PanelCameraProjectState result;
        // Column-major +90-degree roll. This cannot be reconstructed
        // losslessly through the viewer's yaw/pitch controls.
        result.rotation = {
            0.0f, 1.0f, 0.0f,
            -1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f};
        result.translation =
            {tag, tag + 1.0f, tag + 2.0f};
        result.pivot =
            {tag + 3.0f, tag + 4.0f,
             tag + 5.0f};
        result.home_rotation = {
            1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            0.0f, -1.0f, 0.0f};
        result.home_translation =
            {tag + 6.0f, tag + 7.0f,
             tag + 8.0f};
        result.home_pivot =
            {tag + 9.0f, tag + 10.0f,
             tag + 11.0f};
        result.home_saved = true;
        result.zoom_speed = 12.0f + tag;
        result.max_zoom_speed = 80.0f + tag;
        result.rotate_speed = 0.002f + tag * 0.0001f;
        result.centre_speed = 0.003f + tag * 0.0001f;
        result.roll_speed = 0.02f + tag * 0.001f;
        result.translate_speed =
            0.004f + tag * 0.0001f;
        result.wasd_speed = 9.0f + tag;
        result.max_wasd_speed = 90.0f + tag;
        result.ortho_scale = 120.0f + tag;
        return result;
    }

    ReferenceFingerprint fake_reference_fingerprint(
        const FingerprintKind kind,
        const std::uint8_t tag) {
        ReferenceFingerprint result;
        result.kind = kind;
        result.size = 1000 + tag;
        result.mtime_unix_ns = 2000 + tag;
        result.head_xxh3.bytes.fill(tag);
        result.tail_xxh3.bytes.fill(
            static_cast<std::uint8_t>(tag + 1));
        return result;
    }

    ProjectSessionChapters make_matrix_session() {
        ProjectSessionChapters session;

        Json gui = root_json(
            session.gui_layout.dom());
        auto& spaces =
            gui["layouts"][0]["areas"][0]["spaces"];
        auto& fixed =
            spaces[0]["opaque_payload"];
        fixed["right_panel_width"] = 417.0f;
        fixed["scene_panel_ratio"] = 0.61f;
        fixed["python_console_width"] = 511.0f;
        fixed["bottom_dock_height"] = 288.0f;
        fixed["left_dock_width"] = 271.0f;
        fixed["sequencer_visible"] = true;
        fixed["python_console_visible"] = true;
        fixed["window"] = {
            {"x", 101},
            {"y", 202},
            {"width", 1440},
            {"height", 900},
            {"fullscreen", false},
            {"maximized", true},
            {"restore_x", 11},
            {"restore_y", 22},
            {"restore_width", 1280},
            {"restore_height", 720},
        };
        auto& registry =
            spaces[1]["opaque_payload"];
        registry["panels"] = Json::array({
            {
                {"id", "plugin.matrix"},
                {"parent_id", "main"},
                {"space", "floating"},
                {"order", 7},
                {"enabled", true},
                {"float_x", 31.0f},
                {"float_y", 47.0f},
                {"float_user_height", 333.0f},
                {"float_last_bounds_valid", true},
                {"float_last_x", 31.0f},
                {"float_last_y", 47.0f},
                {"float_last_w", 640.0f},
                {"float_last_h", 333.0f},
                {"float_auto_center", false},
                {"float_stack_order", 12},
                {"vendor_extension", "retained"},
            },
        });
        registry["active_tabs"] = {
            {"main_panel", "training"},
            {"scene_panel", "history"},
        };
        spaces[2]["opaque_payload"] = {
            {"active_tab", 1},
            {"font_scale", 1.7f},
        };
        session.gui_layout =
            require_result(
                GuiLayoutChapter::parse(
                    gui.dump()));

        const Json editor{
            {"version", 2},
            {"open_files",
             Json::array({
                 {
                     {"locator",
                      "project://scripts/a.py"},
                     {"modified", false},
                     {"cursor_byte", 4},
                     {"selection_anchor_byte", 1},
                     {"scroll_x", 12.5f},
                     {"scroll_y", 44.0f},
                     {"folds",
                      Json::array({
                          {
                              {"start_byte", 0},
                              {"end_byte", 8},
                              {"collapsed", true},
                          },
                      })},
                 },
                 {
                     {"locator",
                      "project://scripts/b.py"},
                     {"modified", true},
                     {"embedded_buffer",
                      "token = 'secret'\\n"},
                     {"share_warning", true},
                     {"cursor_byte", 18},
                     {"selection_anchor_byte",
                      nullptr},
                     {"scroll_x", 2.0f},
                     {"scroll_y", 91.0f},
                     {"folds", Json::array()},
                 },
             })},
            {"active_file",
             "project://scripts/b.py"},
            {"vim_mode", true},
            {"contains_embedded_secrets", true},
        };
        session.editor =
            require_result(
                EditorSessionChapter::parse(
                    editor.dump()));

        lfs::vis::RenderSettings settings;
        settings.focal_length_mm = 73.0f;
        settings.scaling_modifier = 0.73f;
        settings.antialiasing = true;
        settings.mip_filter = true;
        settings.sh_degree = 2;
        settings.render_scale = 0.75f;
        settings.camera_metrics_mode =
            lfs::vis::RenderSettings::
                CameraMetricsMode::PSNRSSIM;
        settings.show_crop_box = true;
        settings.use_crop_box = true;
        settings.show_ellipsoid = true;
        settings.use_ellipsoid = true;
        settings.desaturate_unselected = true;
        settings.desaturate_cropping = true;
        settings.hide_outside_depth_box = true;
        settings.crop_filter_for_selection = true;
        settings.apply_appearance_correction = true;
        settings.ppisp_mode =
            lfs::vis::RenderSettings::PPISPMode::MANUAL;
        settings.ppisp_overrides.exposure_offset =
            1.25f;
        settings.ppisp_overrides.vignette_strength =
            1.4f;
        settings.ppisp_overrides.wb_temperature =
            0.2f;
        settings.ppisp_overrides.gamma_multiplier =
            1.3f;
        settings.background_color =
            {0.1f, 0.2f, 0.3f};
        settings.environment_mode =
            lfs::vis::
                EnvironmentBackgroundMode::
                    Equirectangular;
        settings.environment_map_path.clear();
        settings.environment_exposure = 1.75f;
        settings.environment_rotation_degrees =
            42.0f;
        settings.show_coord_axes = true;
        settings.axes_size = 3.5f;
        settings.axes_visibility =
            {true, false, true};
        settings.show_grid = true;
        settings.grid_plane = 2;
        settings.grid_opacity = 0.65f;
        settings.point_cloud_mode = true;
        settings.voxel_size = 0.025f;
        settings.show_rings = true;
        settings.ring_width = 0.04f;
        settings.show_center_markers = true;
        settings.show_camera_frustums = true;
        settings.camera_frustum_scale = 0.9f;
        settings.train_camera_color =
            {0.3f, 0.4f, 0.5f};
        settings.eval_camera_color =
            {0.6f, 0.7f, 0.8f};
        settings.show_pivot = true;
        settings.split_view_mode =
            lfs::vis::SplitViewMode::
                IndependentDual;
        settings.gt_comparison_mode =
            lfs::vis::GTComparisonMode::Depth;
        settings.split_position = 0.37f;
        settings.split_view_offset = 5;
        settings.raster_backend =
            lfs::rendering::
                GaussianRasterBackend::ThreeDgut;
        settings.gut = false;
        settings.equirectangular = true;
        settings.orthographic = true;
        settings.ortho_scale = 77.0f;
        settings.depth_view = true;
        settings.depth_view_min = 0.5f;
        settings.depth_view_max = 55.0f;
        settings.depth_visualization_mode =
            lfs::rendering::
                DepthVisualizationMode::Grayscale;
        settings.selection_color_committed =
            {0.11f, 0.22f, 0.33f};
        settings.selection_color_preview =
            {0.44f, 0.55f, 0.66f};
        settings.selection_color_center_marker =
            {0.77f, 0.88f, 0.99f};
        settings.depth_clip_enabled = true;
        settings.depth_clip_far = 34.0f;
        settings.mesh_wireframe = true;
        settings.mesh_wireframe_color =
            {0.2f, 0.4f, 0.6f};
        settings.mesh_wireframe_width = 2.5f;
        settings.mesh_light_dir =
            {0.6f, 0.7f, 0.8f};
        settings.mesh_light_intensity = 0.85f;
        settings.mesh_ambient = 0.25f;
        settings.mesh_backface_culling = false;
        settings.mesh_shadow_enabled = true;
        settings.mesh_shadow_resolution = 4096;
        settings.depth_filter_enabled = true;
        settings.depth_filter_min =
            {-8.0f, -7.0f, -6.0f};
        settings.depth_filter_max =
            {6.0f, 7.0f, 8.0f};
        settings.depth_filter_transform =
            lfs::geometry::EuclideanTransform(
                glm::angleAxis(
                    glm::half_pi<float>(),
                    glm::vec3{0.0f, 0.0f, 1.0f}),
                glm::vec3{1.0f, 2.0f, 3.0f});
        settings.lod_enabled = true;
        settings.lod_auto_enable_rad = true;
        settings.lod_max_splats = 1'234'567;
        settings.lod_render_scale = 0.8f;
        settings.lod_behind_camera_penalty =
            0.31f;
        settings.lod_cone_foveation = 0.51f;
        settings.lod_cone_inner_degrees = 61.0f;
        settings.lod_cone_outer_degrees = 111.0f;
        settings.lod_page_pool_splats = 765'432;
        settings.lod_pool_vram_fraction = 0.22f;
        settings.lod_fade_frames = 19;
        settings.lod_debug_colors = true;

        auto render_settings =
            renderSettingsToProjectJson(settings);
        render_settings
            ["environment_reference_uuid"] =
                lfs::core::generate_uuid_v4()
                    .to_string();
        render_settings["environment_builtin"] =
            nullptr;

        auto primary = panelCameraProjectStateToJson(
            "primary", rolled_camera(1.0f));
        auto secondary =
            panelCameraProjectStateToJson(
                "secondary",
                rolled_camera(20.0f));
        auto bookmark = panelCameraProjectStateToJson(
            "bookmark", rolled_camera(40.0f));
        bookmark.erase("panel");
        bookmark["id"] = "bookmark.matrix";
        bookmark["name"] = "Rolled view";

        Json view = root_json(session.view.dom());
        view["render_settings"] =
            std::move(render_settings);
        view["panel_cameras"] =
            Json::array(
                {std::move(primary),
                 std::move(secondary)});
        view["navigation"] = {
            {"mode", "drone"},
            {"view_snap", true},
        };
        view["split"] = {
            {"focused_panel", "right"},
            {"gt_camera_id", 41},
            {"panel_grid_planes",
             Json::array({0, 2})},
        };
        view["camera_bookmarks"] =
            Json::array({std::move(bookmark)});
        view["tools"] = {
            {"active_tool_id", "crop"},
            {"active_submode_id", "brush"},
            {"selection_submode", "add"},
            {"gizmo_operation", "rotate"},
            {"transform_space", "local"},
            {"pivot_mode", "bounds"},
            {"multi_transform_mode", "group"},
            {"crop_shape", "sphere"},
            {"crop_operation", "subtract"},
            {"selection",
             {
                 {"brush_radius", 37.0f},
                 {"crop_filter", true},
                 {"depth_filter", true},
                 {"restrict_to_selected_nodes",
                  true},
             }},
        };
        view["sequencer_view"] = {
            {"show_camera_path", false},
        };
        session.view =
            require_result(
                ViewSessionChapter::parse(
                    view.dump()));

        lfs::sequencer::Timeline timeline;
        timeline.setClipDuration(48.0f);
        timeline.addKeyframe({
            .time = 3.5f,
            .position = {1.0f, 2.0f, 3.0f},
            .rotation =
                glm::angleAxis(
                    glm::quarter_pi<float>(),
                    glm::vec3{0.0f, 1.0f, 0.0f}),
            .focal_length_mm = 61.0f,
            .easing =
                lfs::sequencer::EasingType::
                    EASE_IN_OUT,
        });
        auto& animation =
            timeline.ensureAnimationClip();
        animation.setName("Matrix animation");
        const auto track_id = animation.addTrack(
            lfs::sequencer::ValueType::Float,
            "node.opacity");
        animation.getTrack(track_id)
            ->addKeyframe(
                1.25f, 0.75f,
                lfs::sequencer::EasingType::
                    EASE_OUT);

        const auto clip_uuid =
            lfs::core::generate_uuid_v4();
        const auto frame_uuid =
            lfs::core::generate_uuid_v4();
        const Json sequencer{
            {"version", 1},
            {"timeline",
             Json::parse(
                 timeline.saveToJson().dump())},
            {"ply_sequences",
             Json::array({
                 {
                     {"node_name",
                      "Matrix sequence"},
                     {"node_uuid",
                      clip_uuid.to_string()},
                     {"directory_reference_uuid",
                      lfs::core::generate_uuid_v4()
                          .to_string()},
                     {"directory_hint",
                      "matrix-frames"},
                     {"frames",
                      Json::array({
                          {
                              {"locator",
                               "frame_0007.ply"},
                              {"node_name",
                               "Frame 7"},
                              {"node_uuid",
                               frame_uuid.to_string()},
                          },
                      })},
                     {"fps", 17.5f},
                 },
             })},
            {"playhead", 3.5f},
            {"loop_mode", "ping_pong"},
            {"playback_speed", 1.75f},
            {"preferences",
             {
                 {"snap_to_grid", true},
                 {"snap_interval", 0.25f},
                 {"follow_playback", true},
                 {"show_pip_preview", false},
                 {"pip_preview_scale", 1.4f},
                 {"show_film_strip", false},
             }},
        };
        session.sequencer =
            require_result(
                SequencerSessionChapter::parse(
                    sequencer.dump()));

        session.metrics.loss_history = {
            {.iteration = 10, .value = 0.42f},
            {.iteration = 20, .value = 0.21f},
        };
        session.metrics.psnr_history = {
            {.iteration = 10, .value = 21.5f},
            {.iteration = 20, .value = 24.75f},
        };
        session.metrics
            .accumulated_training_seconds = 37.5;
        session.metrics.last_evaluation = {
            .iteration = 20,
            .psnr = 24.75f,
            .ssim = 0.91f,
        };
        require_status(
            session.metrics.validate());

        require_result(
            prepareGuiSessionRestore(session));
        return session;
    }

    TEST(P5SessionChapterTest,
         GuilUsesFrozenAreaTreeAndStripsExcludedStateOnLoad) {
        GuiLayoutChapter chapter;
        const Json root = root_json(chapter.dom());
        ASSERT_EQ(root["layouts"].size(), 1u);
        const auto& layout = root["layouts"][0];
        ASSERT_TRUE(layout["active"].get<bool>());
        ASSERT_EQ(layout["areas"].size(), 1u);
        const auto& area = layout["areas"][0];
        EXPECT_TRUE(
            area.contains(
                "rect_or_split_position"));
        EXPECT_TRUE(area.contains("active_space"));
        ASSERT_EQ(area["spaces"].size(), 3u);
        for (const auto& space : area["spaces"]) {
            EXPECT_TRUE(space.contains("type"));
            EXPECT_TRUE(space.contains("version"));
            EXPECT_TRUE(
                space.contains("opaque_payload"));
        }

        Json imgui = root;
        imgui["layouts"][0]["areas"][0]
             ["spaces"][0]["opaque_payload"]
             ["imgui_ini"] = "dock-id";
        auto rejected_imgui =
            GuiLayoutChapter::parse(imgui.dump());
        ASSERT_FALSE(rejected_imgui);
        EXPECT_EQ(
            rejected_imgui.error().code(),
            lfs::ErrorCode::DataLoss);

        for (const auto key : {
                 "theme",
                 "language",
                 "scale",
                 "ui_scale",
                 "hud",
                 "vram_hud",
                 "vram_hud_visible",
             }) {
            Json user_global = root;
            user_global["layouts"][0]["areas"][0]
                       ["spaces"][0]
                       ["opaque_payload"][key] =
                           "forbidden";
            auto loaded =
                GuiLayoutChapter::parse(
                    user_global.dump());
            ASSERT_TRUE(loaded)
                << key;
            const auto sanitized =
                root_json(loaded->dom());
            EXPECT_FALSE(
                sanitized["layouts"][0]["areas"][0]
                         ["spaces"][0]
                         ["opaque_payload"]
                             .contains(key))
                << key;
        }
    }

    TEST(P5SessionChapterTest,
         GuilSaveRefusesExcludedUserGlobalState) {
        TemporaryProjectDirectory temporary;
        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        require_status(
            document->edit_gui_layout()
                .dom()
                .set("theme", "must-not-save"));

        auto saved = document->save(
            temporary.path /
                "excluded-user-global.licht",
            ProjectDocumentSaveOptions{
                .disk_reserve_bytes = 0,
            });
        ASSERT_FALSE(saved);
        EXPECT_EQ(
            saved.error().code(),
            lfs::ErrorCode::DataLoss);
        EXPECT_FALSE(fs::exists(
            temporary.path /
            "excluded-user-global.licht"));
    }

    TEST(P5SessionChapterTest,
         GuilRetainsUnknownSpaceTypeRoundTrip) {
        GuiLayoutChapter chapter;
        Json root = root_json(chapter.dom());
        const Json unknown{
            {"type", "vendor.future_graph"},
            {"version", 7},
            {"opaque_payload",
             {
                 {"opaque_token", "retain-me"},
                 {"nodes",
                  Json::array({
                      {
                          {"id", 41},
                          {"weight", 0.75f},
                      },
                  })},
             }},
        };
        root["layouts"][0]["areas"][0]["spaces"]
            .push_back(unknown);

        auto loaded =
            GuiLayoutChapter::parse(root.dump());
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(
                   loaded.error());
        const Json known_update{
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"spaces",
                               Json::array({
                                   {
                                       {"type",
                                        "fixed_arrangement"},
                                       {"opaque_payload",
                                        {
                                            {"right_panel_width",
                                             492.0f},
                                        }},
                                   },
                               })},
                          },
                      })},
                 },
             })},
        };
        require_status(
            loaded->merge_known_state(
                known_update));

        auto reopened =
            GuiLayoutChapter::from_bytes(
                loaded->to_bytes());
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(
                   reopened.error());
        const auto retained =
            root_json(reopened->dom());
        const auto& spaces =
            retained["layouts"][0]["areas"][0]
                    ["spaces"];
        const auto found =
            std::ranges::find_if(
                spaces,
                [](const Json& space) {
                    return space.value(
                               "type",
                               std::string{}) ==
                           "vendor.future_graph";
                });
        ASSERT_NE(found, spaces.end());
        EXPECT_EQ(*found, unknown);

        ProjectSessionChapters session =
            make_matrix_session();
        session.gui_layout =
            std::move(*reopened);
        EXPECT_TRUE(
            prepareGuiSessionRestore(
                std::move(session)));
    }

    TEST(P5SessionChapterTest,
         GuilAreaTreeRejectsCyclesMissingParentsTwoRootsAndIncompleteOpaqueSpaces) {
        const auto area_with_identity =
            [](Json area, std::string id,
               std::optional<std::string> parent) {
                area["id"] = std::move(id);
                area["parent_id"] = parent
                                        ? Json(*parent)
                                        : Json(nullptr);
                return area;
            };
        const auto expect_data_loss = [](const Json& candidate) {
            auto parsed = GuiLayoutChapter::parse(candidate.dump());
            ASSERT_FALSE(parsed);
            EXPECT_EQ(parsed.error().code(),
                      lfs::ErrorCode::DataLoss);
        };

        GuiLayoutChapter chapter;
        const Json baseline = root_json(chapter.dom());
        const Json implicit_area = baseline["layouts"][0]["areas"][0];

        Json cycle = baseline;
        cycle["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root", std::nullopt),
            area_with_identity(implicit_area, "loop-a", "loop-b"),
            area_with_identity(implicit_area, "loop-b", "loop-a"),
        });
        expect_data_loss(cycle);

        Json missing_parent = baseline;
        missing_parent["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root", std::nullopt),
            area_with_identity(implicit_area, "child", "absent"),
        });
        expect_data_loss(missing_parent);

        Json two_roots = baseline;
        two_roots["layouts"][0]["areas"] = Json::array({
            area_with_identity(implicit_area, "root-a", std::nullopt),
            area_with_identity(implicit_area, "root-b", std::nullopt),
        });
        expect_data_loss(two_roots);

        Json missing_opaque = baseline;
        missing_opaque["layouts"][0]["areas"][0]["spaces"].push_back(
            Json{{"type", "vendor.missing_payload"}, {"version", 1}});
        expect_data_loss(missing_opaque);
    }

    TEST(P5SessionChapterTest,
         MissingViewCameraIsDocumentedDegradedStateButSequencerCameraRefusesHydration) {
        const auto missing = lfs::core::generate_uuid_v4();

        auto view_document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(view_document);
        require_status(view_document->edit_view().dom().set(
            "active_camera_uuid", missing.to_string()));
        lfs::core::Scene view_scene;
        auto view_plan = view_document->stage_hydration(view_scene);
        ASSERT_TRUE(view_plan)
            << lfs::format_for_developer(view_plan.error());
        EXPECT_NE(std::ranges::find(
                      view_document->degraded_states(),
                      ProjectDocumentDegradedState::MissingActiveCamera),
                  view_document->degraded_states().end());

        auto sequence_document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(sequence_document);
        require_status(sequence_document->edit_sequencer().dom().set_json(
            "timeline.keyframes",
            Json::array({Json{{"camera_uuid", missing.to_string()}}})));
        lfs::core::Scene sequence_scene;
        auto sequence_plan =
            sequence_document->stage_hydration(sequence_scene);
        ASSERT_FALSE(sequence_plan);
        EXPECT_EQ(sequence_plan.error().code(),
                  lfs::ErrorCode::DataLoss);
    }

    TEST(P5SessionChapterTest,
         EditorWorkspaceCaptureUsesEveryRealOpenBuffer) {
        using namespace lfs::vis::editor;

        PythonEditor editor;
        PythonEditorWorkspaceSessionState
            restored;
        restored.open_files = {
            {
                .locator =
                    "/tmp/p5-editor-a.py",
                .text =
                    "alpha = 1\nbeta = 2\n",
                .modified = false,
                .editor =
                    {
                        .cursor_byte = 8,
                        .selection_anchor_byte =
                            2,
                        .scroll_x = 11.0f,
                        .scroll_y = 23.0f,
                        .folds =
                            {
                                {
                                    .start_byte =
                                        10,
                                    .end_byte =
                                        18,
                                    .start_line =
                                        1,
                                    .end_line = 2,
                                    .kind =
                                        "block",
                                    .collapsed =
                                        true,
                                },
                            },
                    },
            },
            {
                .locator =
                    "untitled://p5-editor-b",
                .text =
                    "secret = 'embedded'\n",
                .modified = true,
                .editor =
                    {
                        .cursor_byte = 18,
                        .selection_anchor_byte =
                            9,
                        .scroll_x = 3.5f,
                        .scroll_y = 71.0f,
                        .folds = {},
                    },
            },
        };
        restored.active_file =
            "untitled://p5-editor-b";
        restored.vim_mode = true;

        editor.restoreWorkspaceSessionState(
            restored);
        const auto captured =
            editor.captureWorkspaceSessionState(
                "/tmp/stale-console-path.py",
                false);

        ASSERT_EQ(
            captured.open_files.size(), 2u);
        EXPECT_EQ(
            captured.active_file,
            "untitled://p5-editor-b");
        EXPECT_TRUE(captured.vim_mode);
        for (std::size_t index = 0;
             index < restored.open_files.size();
             ++index) {
            const auto& expected =
                restored.open_files[index];
            const auto& actual =
                captured.open_files[index];
            EXPECT_EQ(
                actual.locator,
                expected.locator)
                << index;
            EXPECT_EQ(actual.text, expected.text)
                << index;
            EXPECT_EQ(
                actual.modified,
                expected.modified)
                << index;
            EXPECT_EQ(
                actual.editor.cursor_byte,
                expected.editor.cursor_byte)
                << index;
            EXPECT_EQ(
                actual.editor
                    .selection_anchor_byte,
                expected.editor
                    .selection_anchor_byte)
                << index;
            EXPECT_FLOAT_EQ(
                actual.editor.scroll_x,
                expected.editor.scroll_x)
                << index;
            EXPECT_FLOAT_EQ(
                actual.editor.scroll_y,
                expected.editor.scroll_y)
                << index;
            EXPECT_EQ(
                actual.editor.folds,
                expected.editor.folds)
                << index;
        }
    }

    TEST(P5SessionChapterTest,
         RetainedDomPreservesUnknownFieldsOnKnownPanel) {
        auto session = make_matrix_session();
        const Json known{
            {"version", 1},
            {"layouts",
             Json::array({
                 {
                     {"areas",
                      Json::array({
                          {
                              {"rect_or_split_position",
                               {
                                   {"kind", "rect"},
                               }},
                              {"active_space",
                               "viewport"},
                              {"spaces",
                               Json::array({
                                   {
                                       {"type",
                                        "panel_registry"},
                                       {"version", 1},
                                       {"opaque_payload",
                                        {
                                            {"panels",
                                             Json::array({
                                                 {
                                                     {"id",
                                                      "plugin.matrix"},
                                                     {"enabled",
                                                      false},
                                                 },
                                             })},
                                        }},
                                   },
                               })},
                          },
                      })},
                     {"active", true},
                 },
             })},
        };
        require_status(
            session.gui_layout
                .merge_known_state(known));
        const auto merged = root_json(
            session.gui_layout.dom());
        const auto& panels =
            merged["layouts"][0]["areas"][0]
                  ["spaces"][1]["opaque_payload"]
                  ["panels"];
        ASSERT_EQ(panels.size(), 1u);
        EXPECT_FALSE(
            panels[0]["enabled"].get<bool>());
        EXPECT_EQ(
            panels[0]["vendor_extension"],
            "retained");
    }

    TEST(P5SessionChapterTest,
         MissingViewOrSequencerReferenceFailsDocumentValidation) {
        const auto session = make_matrix_session();
        TemporaryProjectDirectory temporary;

        auto missing_environment =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4(),
                100);
        ASSERT_TRUE(missing_environment);
        missing_environment->edit_view() =
            session.view;
        auto environment_save =
            missing_environment->save(
                temporary.path /
                    "missing-environment.licht",
                ProjectDocumentSaveOptions{
                    .disk_reserve_bytes = 0,
                });
        ASSERT_FALSE(environment_save);
        EXPECT_EQ(
            environment_save.error().code(),
            lfs::ErrorCode::DataLoss);

        auto missing_sequence =
            ProjectDocument::create(
                lfs::core::generate_uuid_v4(),
                100);
        ASSERT_TRUE(missing_sequence);
        missing_sequence->edit_sequencer() =
            session.sequencer;
        auto sequence_save =
            missing_sequence->save(
                temporary.path /
                    "missing-sequence.licht",
                ProjectDocumentSaveOptions{
                    .disk_reserve_bytes = 0,
                });
        ASSERT_FALSE(sequence_save);
        EXPECT_EQ(
            sequence_save.error().code(),
            lfs::ErrorCode::DataLoss);
    }

    TEST(P5SessionChapterTest,
         RenderBackendAndPanelRotationRoundTripCanonically) {
        lfs::vis::RenderSettings settings;
        settings.raster_backend =
            lfs::rendering::
                GaussianRasterBackend::ThreeDgut;
        settings.gut = false;
        const auto json =
            renderSettingsToProjectJson(settings);
        EXPECT_FALSE(json.contains("gut"));
        EXPECT_EQ(
            json["raster_backend"], "3dgut");
        auto restored =
            renderSettingsFromProjectJson(json);
        ASSERT_TRUE(restored)
            << lfs::format_for_developer(
                   restored.error());
        EXPECT_EQ(
            restored->raster_backend,
            lfs::rendering::
                GaussianRasterBackend::ThreeDgut);
        EXPECT_TRUE(restored->gut);

        Viewport viewport;
        const auto expected = rolled_camera(7.0f);
        applyPanelCameraProjectState(
            viewport, expected);
        EXPECT_EQ(
            capturePanelCameraProjectState(
                viewport),
            expected);
    }

    TEST(P5SessionChapterTest,
         RestoreCoordinatorRequiresBothEventGates) {
        auto session = make_matrix_session();
        GuiSessionRestoreCoordinator coordinator;
        require_status(
            coordinator.stage(session));
        EXPECT_TRUE(coordinator.hasPending());
        EXPECT_FALSE(coordinator.ready());

        coordinator.onPanelsReady(73);
        EXPECT_FALSE(coordinator.ready());
        EXPECT_FALSE(coordinator.takeReady());

        coordinator.onFirstGuiFrame();
        ASSERT_TRUE(coordinator.ready());
        auto prepared = coordinator.takeReady();
        ASSERT_TRUE(prepared);
        EXPECT_EQ(
            coordinator
                .panelsRegistrationRevision(),
            73u);
        EXPECT_FALSE(coordinator.hasPending());
        EXPECT_EQ(
            prepared->chapters.editor.dom().dump(),
            session.editor.dom().dump());
    }

    TEST(P5SessionChapterTest,
         AutoloadOffNotStartedTerminalAppliesStagedRestore) {
        GuiSessionRestoreCoordinator coordinator;
        require_status(
            coordinator.stage(
                make_matrix_session()));
        coordinator.onFirstGuiFrame();

        EXPECT_FALSE(
            pluginPreloadTerminalForGuiPanels(
                false, "not_started"));
        EXPECT_FALSE(
            pluginPreloadTerminalForGuiPanels(
                true, "discovering"));
        EXPECT_FALSE(
            pluginPreloadTerminalForGuiPanels(
                true, "loading"));
        EXPECT_TRUE(
            pluginPreloadTerminalForGuiPanels(
                true, "completed"));
        EXPECT_TRUE(
            pluginPreloadTerminalForGuiPanels(
                true, "cancelled"));
        ASSERT_TRUE(
            pluginPreloadTerminalForGuiPanels(
                true, "not_started"));

        coordinator.onPanelsReady(91);
        auto restored = coordinator.takeReady();
        ASSERT_TRUE(restored);
        EXPECT_EQ(
            restored->chapters.editor.dom().dump(),
            make_matrix_session()
                .editor.dom()
                .dump());
    }

    TEST(P5SessionChapterTest,
         MetricsHostileBinaryIsRejectedBeforeAllocation) {
        constexpr std::size_t LOSS_COUNT_OFFSET = 16;
        constexpr std::size_t ACCUMULATED_OFFSET =
            32;
        constexpr std::size_t LAST_PSNR_OFFSET =
            44;
        constexpr std::size_t LOSS_VALUE_OFFSET =
            56;
        constexpr std::size_t PSNR_VALUE_OFFSET =
            64;

        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 3, .value = 0.5f},
        };
        metrics.psnr_history = {
            {.iteration = 3, .value = 20.0f},
        };
        metrics.accumulated_training_seconds =
            1.5;
        metrics.last_evaluation = {
            .iteration = 3,
            .psnr = 20.0f,
            .ssim = 0.9f,
        };
        const auto valid =
            require_result(metrics.to_bytes());
        ASSERT_EQ(valid.size(), 68u);

        const auto expect_data_loss =
            [](const std::vector<std::byte>& bytes) {
                auto parsed =
                    MetricsChapter::from_bytes(
                        bytes);
                ASSERT_FALSE(parsed);
                EXPECT_EQ(
                    parsed.error().code(),
                    lfs::ErrorCode::DataLoss);
            };

        auto truncated_header = valid;
        truncated_header.resize(31);
        expect_data_loss(truncated_header);

        auto truncated_payload = valid;
        truncated_payload.pop_back();
        expect_data_loss(truncated_payload);

        auto oversized = valid;
        overwrite_u64_le(
            oversized, LOSS_COUNT_OFFSET,
            MetricsChapter::MAX_HISTORY_SAMPLES +
                1ull);
        auto oversized_result =
            MetricsChapter::from_bytes(
                oversized);
        ASSERT_FALSE(oversized_result);
        EXPECT_EQ(
            oversized_result.error().code(),
            lfs::ErrorCode::ResourceExhausted);

        auto nan_accumulated = valid;
        overwrite_f64_le(
            nan_accumulated,
            ACCUMULATED_OFFSET,
            std::numeric_limits<double>::
                quiet_NaN());
        expect_data_loss(nan_accumulated);

        auto infinite_last = valid;
        overwrite_f32_le(
            infinite_last, LAST_PSNR_OFFSET,
            std::numeric_limits<float>::
                infinity());
        expect_data_loss(infinite_last);

        auto nan_loss = valid;
        overwrite_f32_le(
            nan_loss, LOSS_VALUE_OFFSET,
            std::numeric_limits<float>::
                quiet_NaN());
        expect_data_loss(nan_loss);

        auto infinite_psnr = valid;
        overwrite_f32_le(
            infinite_psnr, PSNR_VALUE_OFFSET,
            -std::numeric_limits<float>::
                infinity());
        expect_data_loss(infinite_psnr);

        auto misaligned = valid;
        misaligned.push_back(
            std::byte{0x7f});
        expect_data_loss(misaligned);
    }

    class P5MetricsRestoreTest
        : public ::testing::Test {
    protected:
        void SetUp() override {
            lfs::event::EventBridge::instance()
                .clear_all();
            lfs::training::CommandCenter::instance()
                .clear_loss_history();
        }

        void TearDown() override {
            lfs::training::CommandCenter::instance()
                .clear_loss_history();
            lfs::event::EventBridge::instance()
                .clear_all();
        }
    };

    TEST_F(P5MetricsRestoreTest,
           ResumeRebuildsPopulatedGraphBuffers) {
        MetricsChapter metrics;
        metrics.loss_history = {
            {.iteration = 3, .value = 0.8f},
            {.iteration = 9, .value = 0.4f},
        };
        metrics.psnr_history = {
            {.iteration = 3, .value = 18.0f},
            {.iteration = 9, .value = 23.0f},
        };
        metrics.accumulated_training_seconds =
            125.5;
        metrics.last_evaluation = {
            .iteration = 9,
            .psnr = 23.0f,
            .ssim = 0.93f,
        };

        lfs::vis::TrainerManager manager;
        manager.restoreProjectMetrics(metrics);
        const auto loss = manager.getLossBuffer();
        const auto psnr = manager.getPSNRBuffer();
        ASSERT_EQ(loss.size(), 2u);
        ASSERT_EQ(psnr.size(), 2u);
        EXPECT_FLOAT_EQ(loss.back(), 0.4f);
        EXPECT_FLOAT_EQ(psnr.back(), 23.0f);
        EXPECT_FLOAT_EQ(manager.getLastPSNR(), 23.0f);
        EXPECT_NEAR(
            manager.getElapsedSeconds(),
            125.5f, 0.001f);
        const auto last =
            manager.getLastEvaluationMetrics();
        ASSERT_TRUE(last);
        EXPECT_EQ(last->iteration, 9);
        EXPECT_FLOAT_EQ(last->ssim, 0.93f);

        const auto recaptured =
            manager.captureProjectMetrics();
        EXPECT_EQ(recaptured, metrics);

        lfs::core::Scene checkpoint_scene;
        const auto cameras =
            checkpoint_scene.addGroup("Cameras");
        const auto training_cameras =
            checkpoint_scene.addCameraGroup(
                "Training", cameras, 1);
        checkpoint_scene.addCamera(
            "train.png",
            training_cameras,
            std::make_shared<lfs::core::Camera>());
        manager.setTrainerFromCheckpoint(
            std::make_unique<
                lfs::training::Trainer>(
                checkpoint_scene),
            9);
        EXPECT_FLOAT_EQ(
            manager.getElapsedSeconds(), 0.0f);
        manager.restoreProjectMetrics(metrics);
        EXPECT_NEAR(
            manager.getElapsedSeconds(),
            125.5f, 0.001f);
    }

    TEST_F(P5MetricsRestoreTest,
           IdleManagerShutdownCannotMissReaperStopWake) {
        for (int attempt = 0; attempt < 64; ++attempt) {
            lfs::vis::TrainerManager manager;
        }
    }

    TEST(P5MatrixProof,
         EveryAssignedRowSurvivesSaveLoadAndStagesBeforeMutation) {
        const auto session = make_matrix_session();
        TemporaryProjectDirectory temporary;
        const auto project_path =
            temporary.path / "p5-matrix.licht";

        auto document = ProjectDocument::create(
            lfs::core::generate_uuid_v4(), 100);
        ASSERT_TRUE(document)
            << lfs::format_for_developer(
                   document.error());
        const Json session_view =
            root_json(session.view.dom());
        const auto environment_reference =
            lfs::core::Uuid::from_string(
                session_view["render_settings"]
                            ["environment_reference_uuid"]
                                .get<std::string>());
        const Json session_sequencer =
            root_json(session.sequencer.dom());
        const auto sequence_reference =
            lfs::core::Uuid::from_string(
                session_sequencer["ply_sequences"][0]
                                 ["directory_reference_uuid"]
                                     .get<std::string>());
        ASSERT_TRUE(environment_reference);
        ASSERT_TRUE(sequence_reference);
        require_status(
            document->edit_references().upsert(
                ReferenceRecord{
                    .uuid = *environment_reference,
                    .key = "view.environment",
                    .kind = "environment_map",
                    .locator =
                        {
                            .preferred =
                                "assets/matrix.hdr",
                            .base =
                                LocatorBase::Project,
                        },
                    .fingerprint =
                        fake_reference_fingerprint(
                            FingerprintKind::File,
                            31),
                    .unresolved = true,
                }));
        require_status(
            document->edit_references().upsert(
                ReferenceRecord{
                    .uuid = *sequence_reference,
                    .key =
                        "sequencer.matrix_frames",
                    .kind =
                        "ply_sequence_directory",
                    .locator =
                        {
                            .preferred =
                                "matrix-frames",
                            .base =
                                LocatorBase::Project,
                        },
                    .fingerprint =
                        fake_reference_fingerprint(
                            FingerprintKind::Directory,
                            32),
                    .unresolved = true,
                }));
        document->edit_gui_layout() =
            session.gui_layout;
        document->edit_editor() = session.editor;
        document->edit_view() = session.view;
        document->edit_sequencer() =
            session.sequencer;
        document->edit_metrics() =
            session.metrics;
        auto saved = document->save(
            project_path,
            ProjectDocumentSaveOptions{
                .disk_reserve_bytes = 0,
            });
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());

        auto reopened =
            ProjectDocument::open(project_path);
        ASSERT_TRUE(reopened)
            << lfs::format_for_developer(
                   reopened.error());
        auto reverse =
            reopened->reverse_reference_index();
        ASSERT_TRUE(reverse)
            << lfs::format_for_developer(
                   reverse.error());
        EXPECT_TRUE(std::ranges::any_of(
            reverse->at(*environment_reference),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "VIEW" &&
                       binding.field ==
                           "render_settings.environment_reference_uuid";
            }));
        EXPECT_TRUE(std::ranges::any_of(
            reverse->at(*sequence_reference),
            [](const ReferenceOwnerBinding& binding) {
                return binding.chapter == "SEQR" &&
                       binding.owner_uuid.has_value() &&
                       binding.field ==
                           "ply_sequences.directory_reference_uuid";
            }));

        lfs::core::Scene live_scene;
        ASSERT_NE(
            live_scene.addGroup("Live sentinel"),
            lfs::core::NULL_NODE);
        auto staged =
            reopened->stage_hydration(live_scene);
        ASSERT_TRUE(staged)
            << lfs::format_for_developer(
                   staged.error());
        EXPECT_EQ(live_scene.getNodeCount(), 1u);
        ASSERT_TRUE(
            staged->report().gui_session_pending);
        EXPECT_EQ(
            staged->report()
                .pending_session.gui_layout
                .dom()
                .dump(),
            reopened->gui_layout().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.view.dom()
                .dump(),
            reopened->view().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.editor.dom()
                .dump(),
            reopened->editor().dom().dump());
        EXPECT_EQ(
            staged->report()
                .pending_session.sequencer
                .dom()
                .dump(),
            reopened->sequencer().dom().dump());
        EXPECT_EQ(
            staged->report().pending_session.metrics,
            reopened->metrics());

        GuiSessionRestoreCoordinator coordinator;
        require_status(coordinator.stage(
            staged->report().pending_session));
        coordinator.onFirstGuiFrame();
        EXPECT_FALSE(coordinator.takeReady());
        coordinator.onPanelsReady(88);
        ASSERT_TRUE(coordinator.takeReady());
        EXPECT_EQ(live_scene.getNodeCount(), 1u);

        const auto hydration =
            ProjectDocument::commit_hydration(
                live_scene,
                std::move(*staged));
        EXPECT_TRUE(hydration.gui_session_pending);
        EXPECT_EQ(live_scene.getNodeCount(), 0u);

        const Json gui =
            root_json(reopened->gui_layout().dom());
        const auto& spaces =
            gui["layouts"][0]["areas"][0]["spaces"];
        const auto& fixed =
            spaces[0]["opaque_payload"];
        const auto& registry =
            spaces[1]["opaque_payload"];
        const auto& panel =
            registry["panels"][0];
        const auto& console =
            spaces[2]["opaque_payload"];

        std::set<std::string> proven;
        const auto prove =
            [&proven](
                const std::string_view row) {
                EXPECT_TRUE(
                    proven
                        .emplace(row)
                        .second)
                    << row;
            };

        EXPECT_FLOAT_EQ(
            fixed["right_panel_width"]
                .get<float>(),
            417.0f);
        EXPECT_FLOAT_EQ(
            fixed["scene_panel_ratio"]
                .get<float>(),
            0.61f);
        EXPECT_FLOAT_EQ(
            fixed["python_console_width"]
                .get<float>(),
            511.0f);
        EXPECT_FLOAT_EQ(
            fixed["bottom_dock_height"]
                .get<float>(),
            288.0f);
        EXPECT_FLOAT_EQ(
            fixed["left_dock_width"]
                .get<float>(),
            271.0f);
        prove("GUIL-166");
        EXPECT_TRUE(
            fixed["sequencer_visible"]
                .get<bool>());
        EXPECT_TRUE(
            fixed["python_console_visible"]
                .get<bool>());
        EXPECT_TRUE(panel["enabled"].get<bool>());
        prove("GUIL-167");
        EXPECT_EQ(panel["parent_id"], "main");
        EXPECT_EQ(panel["space"], "floating");
        EXPECT_EQ(panel["order"], 7);
        EXPECT_EQ(
            registry["active_tabs"]
                    ["main_panel"],
            "training");
        EXPECT_EQ(
            registry["active_tabs"]
                    ["scene_panel"],
            "history");
        prove("GUIL-168");
        EXPECT_FLOAT_EQ(
            panel["float_x"].get<float>(),
            31.0f);
        EXPECT_FLOAT_EQ(
            panel["float_last_w"].get<float>(),
            640.0f);
        EXPECT_EQ(
            panel["float_stack_order"], 12);
        EXPECT_EQ(
            panel["vendor_extension"],
            "retained");
        prove("GUIL-169");
        EXPECT_EQ(fixed["window"]["x"], 101);
        EXPECT_EQ(
            fixed["window"]["width"], 1440);
        EXPECT_TRUE(
            fixed["window"]["maximized"]
                .get<bool>());
        EXPECT_EQ(
            fixed["window"]["restore_width"],
            1280);
        prove("GUIL-170");
        EXPECT_EQ(console["active_tab"], 1);
        EXPECT_FLOAT_EQ(
            console["font_scale"].get<float>(),
            1.7f);
        prove("GUIL-171");

        const Json editor =
            root_json(reopened->editor().dom());
        ASSERT_EQ(editor["open_files"].size(), 2u);
        EXPECT_EQ(
            editor["open_files"][0]["locator"],
            "project://scripts/a.py");
        EXPECT_EQ(
            editor["open_files"][1]["locator"],
            "project://scripts/b.py");
        prove("EDTR-179");
        EXPECT_EQ(
            editor["active_file"],
            "project://scripts/b.py");
        prove("EDTR-180");
        EXPECT_TRUE(
            editor["open_files"][1]["modified"]
                .get<bool>());
        EXPECT_EQ(
            editor["open_files"][1]
                  ["embedded_buffer"],
            "token = 'secret'\\n");
        EXPECT_TRUE(
            editor["open_files"][1]
                  ["share_warning"]
                      .get<bool>());
        EXPECT_TRUE(
            editor["contains_embedded_secrets"]
                .get<bool>());
        prove("EDTR-181");
        EXPECT_EQ(
            editor["open_files"][0]
                  ["cursor_byte"],
            4);
        EXPECT_EQ(
            editor["open_files"][0]
                  ["selection_anchor_byte"],
            1);
        prove("EDTR-182");
        EXPECT_FLOAT_EQ(
            editor["open_files"][0]
                  ["scroll_y"]
                      .get<float>(),
            44.0f);
        EXPECT_TRUE(
            editor["open_files"][0]["folds"][0]
                  ["collapsed"]
                      .get<bool>());
        prove("EDTR-183");
        EXPECT_TRUE(
            editor["vim_mode"].get<bool>());
        prove("EDTR-184");

        const Json view =
            root_json(reopened->view().dom());
        const auto& render =
            view["render_settings"];
        EXPECT_FLOAT_EQ(
            render["focal_length_mm"]
                .get<float>(),
            73.0f);
        EXPECT_TRUE(
            render["antialiasing"].get<bool>());
        EXPECT_TRUE(
            render["mip_filter"].get<bool>());
        EXPECT_EQ(render["sh_degree"], 2);
        EXPECT_EQ(
            render["camera_metrics_mode"], 2);
        prove("VIEW-192");
        EXPECT_TRUE(
            render["show_crop_box"].get<bool>());
        EXPECT_TRUE(
            render["use_ellipsoid"].get<bool>());
        EXPECT_TRUE(
            render["desaturate_cropping"]
                .get<bool>());
        EXPECT_TRUE(
            render["crop_filter_for_selection"]
                .get<bool>());
        prove("VIEW-193");
        EXPECT_TRUE(
            render["apply_appearance_correction"]
                .get<bool>());
        EXPECT_EQ(render["ppisp_mode"], 0);
        EXPECT_FLOAT_EQ(
            render["ppisp_overrides"]
                  ["exposure_offset"]
                      .get<float>(),
            1.25f);
        EXPECT_FLOAT_EQ(
            render["ppisp_overrides"]
                  ["gamma_multiplier"]
                      .get<float>(),
            1.3f);
        prove("VIEW-194");
        EXPECT_EQ(
            render["background_color"],
            Json::array({0.1f, 0.2f, 0.3f}));
        EXPECT_TRUE(
            render["environment_reference_uuid"]
                .is_string());
        EXPECT_TRUE(
            render["environment_builtin"]
                .is_null());
        EXPECT_FLOAT_EQ(
            render["environment_exposure"]
                .get<float>(),
            1.75f);
        prove("VIEW-195");
        EXPECT_TRUE(
            render["show_coord_axes"].get<bool>());
        EXPECT_EQ(
            render["axes_visibility"],
            Json::array({true, false, true}));
        EXPECT_EQ(render["grid_plane"], 2);
        prove("VIEW-196");
        EXPECT_TRUE(
            render["point_cloud_mode"].get<bool>());
        EXPECT_TRUE(
            render["show_rings"].get<bool>());
        EXPECT_TRUE(
            render["show_camera_frustums"]
                .get<bool>());
        EXPECT_TRUE(
            render["show_pivot"].get<bool>());
        prove("VIEW-197");
        EXPECT_EQ(render["split_view_mode"], 3);
        EXPECT_EQ(
            render["gt_comparison_mode"], 2);
        EXPECT_EQ(
            render["raster_backend"], "3dgut");
        EXPECT_FALSE(render.contains("gut"));
        EXPECT_TRUE(
            render["orthographic"].get<bool>());
        EXPECT_TRUE(
            render["depth_view"].get<bool>());
        prove("VIEW-198");
        EXPECT_EQ(
            render["selection_color_committed"],
            Json::array(
                {0.11f, 0.22f, 0.33f}));
        EXPECT_TRUE(
            render["depth_clip_enabled"]
                .get<bool>());
        EXPECT_FLOAT_EQ(
            render["depth_clip_far"].get<float>(),
            34.0f);
        prove("VIEW-199");
        EXPECT_TRUE(
            render["mesh_wireframe"].get<bool>());
        EXPECT_FLOAT_EQ(
            render["mesh_wireframe_width"]
                .get<float>(),
            2.5f);
        EXPECT_TRUE(
            render["mesh_shadow_enabled"]
                .get<bool>());
        EXPECT_EQ(
            render["mesh_shadow_resolution"],
            4096);
        prove("VIEW-200");
        EXPECT_TRUE(
            render["depth_filter_enabled"]
                .get<bool>());
        EXPECT_EQ(
            render["depth_filter_transform"]
                  ["translation"],
            Json::array({1.0f, 2.0f, 3.0f}));
        EXPECT_EQ(
            render["depth_filter_transform"]
                  ["rotation"]
                      .size(),
            9u);
        prove("VIEW-201");
        EXPECT_TRUE(
            render["lod_enabled"].get<bool>());
        EXPECT_EQ(
            render["lod_max_splats"],
            1'234'567);
        EXPECT_EQ(
            render["lod_page_pool_splats"],
            765'432);
        EXPECT_TRUE(
            render["lod_debug_colors"].get<bool>());
        prove("VIEW-202");
        EXPECT_EQ(
            view["split"]["panel_grid_planes"],
            Json::array({0, 2}));
        prove("VIEW-203");
        ASSERT_EQ(
            view["panel_cameras"].size(), 2u);
        EXPECT_EQ(
            view["panel_cameras"][0]["R"],
            Json::array({0.0f, 1.0f, 0.0f,
                         -1.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 1.0f}));
        EXPECT_TRUE(
            view["panel_cameras"][1]
                ["ortho_scale"]
                    .is_number());
        prove("VIEW-204");
        EXPECT_EQ(
            view["navigation"]["mode"],
            "drone");
        EXPECT_TRUE(
            view["navigation"]["view_snap"]
                .get<bool>());
        prove("VIEW-205");
        EXPECT_EQ(
            view["split"]["focused_panel"],
            "right");
        EXPECT_EQ(
            view["split"]["gt_camera_id"], 41);
        prove("VIEW-206");
        ASSERT_EQ(
            view["camera_bookmarks"].size(), 1u);
        EXPECT_EQ(
            view["camera_bookmarks"][0]["id"],
            "bookmark.matrix");
        EXPECT_EQ(
            view["camera_bookmarks"][0]["name"],
            "Rolled view");
        prove("VIEW-207");
        EXPECT_EQ(
            view["tools"]["active_tool_id"],
            "crop");
        EXPECT_EQ(
            view["tools"]["active_submode_id"],
            "brush");
        EXPECT_EQ(
            view["tools"]["gizmo_operation"],
            "rotate");
        EXPECT_EQ(
            view["tools"]["transform_space"],
            "local");
        prove("VIEW-208");
        EXPECT_FLOAT_EQ(
            view["tools"]["selection"]
                ["brush_radius"]
                    .get<float>(),
            37.0f);
        EXPECT_TRUE(
            view["tools"]["selection"]
                ["restrict_to_selected_nodes"]
                    .get<bool>());
        prove("VIEW-209");
        EXPECT_FALSE(
            view["sequencer_view"]
                ["show_camera_path"]
                    .get<bool>());
        prove("VIEW-210");

        const Json sequencer = root_json(
            reopened->sequencer().dom());
        EXPECT_FLOAT_EQ(
            sequencer["timeline"]
                     ["clip_duration"]
                         .get<float>(),
            48.0f);
        ASSERT_EQ(
            sequencer["timeline"]["keyframes"]
                .size(),
            1u);
        EXPECT_EQ(
            sequencer["timeline"]["keyframes"][0]
                     ["position"],
            Json::array({1.0f, 2.0f, 3.0f}));
        prove("SEQR-218");
        EXPECT_EQ(
            sequencer["timeline"]
                     ["animation_clip"]["name"],
            "Matrix animation");
        ASSERT_EQ(
            sequencer["timeline"]
                     ["animation_clip"]["tracks"]
                         .size(),
            1u);
        EXPECT_EQ(
            sequencer["timeline"]
                     ["animation_clip"]["tracks"][0]
                     ["target"],
            "node.opacity");
        prove("SEQR-219");
        ASSERT_EQ(
            sequencer["ply_sequences"].size(),
            1u);
        EXPECT_EQ(
            sequencer["ply_sequences"][0]
                     ["frames"][0]["locator"],
            "frame_0007.ply");
        EXPECT_FLOAT_EQ(
            sequencer["ply_sequences"][0]["fps"]
                .get<float>(),
            17.5f);
        EXPECT_FALSE(
            sequencer["ply_sequences"][0]
                .contains("sequence_fps"));
        prove("SEQR-220");
        EXPECT_FLOAT_EQ(
            sequencer["playhead"].get<float>(),
            3.5f);
        EXPECT_EQ(
            sequencer["loop_mode"],
            "ping_pong");
        EXPECT_FLOAT_EQ(
            sequencer["playback_speed"]
                .get<float>(),
            1.75f);
        prove("SEQR-221");
        EXPECT_TRUE(
            sequencer["preferences"]
                     ["snap_to_grid"]
                         .get<bool>());
        EXPECT_TRUE(
            sequencer["preferences"]
                     ["follow_playback"]
                         .get<bool>());
        EXPECT_FALSE(
            sequencer["preferences"]
                     ["show_film_strip"]
                         .get<bool>());
        prove("SEQR-222");

        const auto& metrics = reopened->metrics();
        ASSERT_EQ(metrics.loss_history.size(), 2u);
        EXPECT_EQ(
            metrics.loss_history[1].iteration,
            20);
        EXPECT_FLOAT_EQ(
            metrics.loss_history[1].value,
            0.21f);
        prove("METR-230");
        ASSERT_EQ(metrics.psnr_history.size(), 2u);
        EXPECT_FLOAT_EQ(
            metrics.psnr_history[1].value,
            24.75f);
        prove("METR-231");
        EXPECT_DOUBLE_EQ(
            metrics.accumulated_training_seconds,
            37.5);
        prove("METR-232");
        ASSERT_TRUE(metrics.last_evaluation);
        EXPECT_EQ(
            metrics.last_evaluation->iteration,
            20);
        EXPECT_FLOAT_EQ(
            metrics.last_evaluation->psnr,
            24.75f);
        EXPECT_FLOAT_EQ(
            metrics.last_evaluation->ssim,
            0.91f);
        prove("METR-233");

        const std::set<std::string> registered(
            lfs::test::P5_MATRIX_ROWS.begin(),
            lfs::test::P5_MATRIX_ROWS.end());
        EXPECT_EQ(proven, registered)
            << "Every registered P5 row must reach an explicit "
               "save->load->Phase-A assertion";
    }

} // namespace

namespace lfs::test {

    io::project::ProjectSessionChapters
    make_p8_matrix_session() {
        return make_matrix_session();
    }

} // namespace lfs::test
