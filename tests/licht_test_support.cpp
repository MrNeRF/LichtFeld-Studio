/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "licht_test_support.hpp"

#include "io/project_document.hpp"

#include "project/session_state.hpp"
#include "sequencer/timeline.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <utility>

namespace lfs::test::licht {

    using core::DataType;
    using core::Device;
    using core::Tensor;
    using Json = io::JsonChapterDom::Json;
    using namespace io::project;
    using namespace vis::project;

    PanelCameraProjectState rolled_panel_camera(
        const float tag) {
        PanelCameraProjectState result;
        // Column-major +90-degree roll. This cannot be reconstructed
        // losslessly through the viewer's yaw/pitch controls.
        result.rotation = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        result.translation = {tag, tag + 1.0f, tag + 2.0f};
        result.pivot = {tag + 3.0f, tag + 4.0f, tag + 5.0f};
        result.home_rotation = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f};
        result.home_translation = {tag + 6.0f, tag + 7.0f, tag + 8.0f};
        result.home_pivot = {tag + 9.0f, tag + 10.0f, tag + 11.0f};
        result.home_saved = true;
        result.zoom_speed = 12.0f + tag;
        result.max_zoom_speed = 80.0f + tag;
        result.rotate_speed = 0.002f + tag * 0.0001f;
        result.centre_speed = 0.003f + tag * 0.0001f;
        result.roll_speed = 0.02f + tag * 0.001f;
        result.translate_speed = 0.004f + tag * 0.0001f;
        result.wasd_speed = 9.0f + tag;
        result.max_wasd_speed = 90.0f + tag;
        result.ortho_scale = 120.0f + tag;
        return result;
    }

    ProjectSessionChapters make_populated_session_chapters() {
        ProjectSessionChapters session;

        Json gui = json_root(
            session.gui_layout.dom());
        auto& spaces =
            gui["layouts"][0]["areas"][0]["spaces"];
        spaces[0]["opaque_payload"] = Json::parse(R"({
        "right_panel_width":417.0,"scene_panel_ratio":0.61,"python_console_width":511.0,
        "bottom_dock_height":288.0,"left_dock_width":271.0,
        "sequencer_visible":true,"python_console_visible":true
    })");
        spaces[1]["opaque_payload"] = Json::parse(R"({
        "panels":[{"id":"plugin.matrix","parent_id":"main","space":"floating",
          "order":7,"enabled":true,"float_x":31.0,"float_y":47.0,"float_user_height":333.0,
          "float_last_bounds_valid":true,"float_last_x":31.0,"float_last_y":47.0,
          "float_last_w":640.0,"float_last_h":333.0,"float_auto_center":false,
          "float_stack_order":12,"vendor_extension":"retained"}],
        "active_tabs":{"main_panel":"training","scene_panel":"history"}
    })");
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

        Json view = json_root(session.view.dom());
        auto& render = view["render_settings"];
        render.update(Json::parse(R"({
        "focal_length_mm":73.0,"scaling_modifier":0.73,"antialiasing":true,"mip_filter":true,
        "sh_degree":2,"render_scale":0.75,"camera_metrics_mode":2,
        "show_crop_box":true,"use_crop_box":true,"show_ellipsoid":true,"use_ellipsoid":true,
        "desaturate_unselected":true,"desaturate_cropping":true,"hide_outside_depth_box":true,
        "depth_filter_viz_mode":2,
        "crop_filter_for_selection":true,"apply_appearance_correction":true,"ppisp_mode":0,
        "background_color":[0.1,0.2,0.3],"environment_builtin":null,
        "environment_exposure":1.75,"environment_rotation_degrees":42.0,
        "show_coord_axes":true,"axes_size":3.5,"axes_visibility":[true,false,true],
        "show_grid":true,"grid_plane":2,"grid_opacity":0.65,
        "point_cloud_mode":true,"voxel_size":0.025,"show_rings":true,"ring_width":0.04,
        "show_center_markers":true,"show_camera_frustums":true,"camera_frustum_scale":0.9,
        "train_camera_color":[0.3,0.4,0.5],"eval_camera_color":[0.6,0.7,0.8],"show_pivot":true,
        "split_view_mode":1,"gt_comparison_mode":2,"split_position":0.37,"split_view_offset":5,
        "raster_backend":"3dgut","equirectangular":true,"orthographic":true,"ortho_scale":77.0,
        "depth_view":true,"depth_view_min":0.5,"depth_view_max":55.0,"depth_visualization_mode":0,
        "selection_color_preview":[0.44,0.55,0.66],
        "selection_color_center_marker":[0.77,0.88,0.99],
        "depth_clip_enabled":true,"depth_clip_far":34.0,"mesh_wireframe":true,
        "mesh_wireframe_color":[0.2,0.4,0.6],"mesh_wireframe_width":2.5,
        "mesh_light_dir":[0.6,0.7,0.8],"mesh_light_intensity":0.85,"mesh_ambient":0.25,
        "mesh_backface_culling":false,"mesh_shadow_enabled":true,"mesh_shadow_resolution":4096,
        "depth_filter_enabled":true,"depth_filter_min":[-8.0,-7.0,-6.0],
        "depth_filter_max":[6.0,7.0,8.0],
        "depth_filter_transform":{"rotation":[0.0,1.0,0.0,-1.0,0.0,0.0,0.0,0.0,1.0],
                                  "translation":[1.0,2.0,3.0]},
        "lod_enabled":true,"lod_auto_enable_rad":true,"lod_max_splats":1234567,
        "lod_render_scale":0.8,"lod_behind_camera_penalty":0.31,"lod_cone_foveation":0.51,
        "lod_cone_inner_degrees":61.0,"lod_cone_outer_degrees":111.0,
        "lod_page_pool_splats":765432,"lod_pool_vram_fraction":0.22,
        "lod_fade_frames":19,"lod_debug_colors":true
    })"));
        render["ppisp_overrides"]["exposure_offset"] = 1.25f;
        render["ppisp_overrides"]["vignette_strength"] = 1.4f;
        render["ppisp_overrides"]["wb_temperature"] = 0.2f;
        render["ppisp_overrides"]["gamma_multiplier"] = 1.3f;
        render["background_color"] = {0.1f, 0.2f, 0.3f};
        render["environment_reference_uuid"] = core::generate_uuid_v4().to_string();
        render.erase("gut");

        auto primary = panelCameraProjectStateToJson(
            "primary", rolled_panel_camera(1.0f));
        auto bookmark = panelCameraProjectStateToJson(
            "bookmark", rolled_panel_camera(40.0f));
        bookmark.erase("panel");
        bookmark["id"] = "bookmark.matrix";
        bookmark["name"] = "Rolled view";

        view["panel_cameras"] =
            Json::array(
                {std::move(primary)});
        view["navigation"] = {
            {"mode", "drone"},
            {"view_snap", true},
        };
        view["split"] = {
            {"gt_camera_id", 41},
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

        (void)require_result(prepareGuiSessionRestore(session));
        return session;
    }

    std::vector<std::byte> one_pixel_png() {
        static constexpr char PNG[] =
            "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52"
            "\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
            "\x89\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x60\x00\x00\x00"
            "\x02\x00\x01\xe5\x27\xd4\xa2\x00\x00\x00\x00\x49\x45\x4e\x44\xae"
            "\x42\x60\x82";
        const auto bytes = std::as_bytes(
            std::span(PNG, sizeof(PNG) - 1));
        return {bytes.begin(), bytes.end()};
    }

    std::unique_ptr<core::SplatData> make_splat(const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index);
            rotations[index * 4] = 1.0f;
        }
        return std::make_unique<core::SplatData>(
            0, Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::zeros({count, std::size_t{1}, std::size_t{3}}, Device::CPU,
                          DataType::Float32),
            Tensor{}, Tensor::zeros({count, std::size_t{3}}, Device::CPU, DataType::Float32),
            Tensor::from_vector(rotations, {count, std::size_t{4}}, Device::CPU),
            Tensor::zeros({count, std::size_t{1}}, Device::CPU, DataType::Float32), 1.0f);
    }

    std::shared_ptr<core::PointCloud> make_point_cloud(const std::size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> colors(count * 3, 0.5f);
        for (std::size_t index = 0; index < count; ++index) {
            means[index * 3] = static_cast<float>(index + 10);
        }
        return std::make_shared<core::PointCloud>(
            Tensor::from_vector(means, {count, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(colors, {count, std::size_t{3}}, Device::CPU));
    }

    std::shared_ptr<core::MeshData> make_triangle_mesh() {
        return std::make_shared<core::MeshData>(
            Tensor::from_vector(std::vector<float>{-1, -1, 0, 1, -1, 0, 0, 1, 0},
                                {std::size_t{3}, std::size_t{3}}, Device::CPU),
            Tensor::from_vector(std::vector<std::int32_t>{0, 1, 2},
                                {std::size_t{1}, std::size_t{3}}, Device::CPU));
    }

    ProjectDocumentSaveOptions deterministic_document_save_options(
        const std::uint32_t uuid_namespace, const std::uint64_t identity_tag,
        const std::uint64_t wallclock_unix_ns) {
        return {
            .commit =
                {
                    .kind = CommitKind::Explicit,
                    .commit_uuid = fixed_uuid_in_namespace(uuid_namespace, identity_tag),
                    .snapshot_uuid = fixed_uuid_in_namespace(uuid_namespace, identity_tag + 1),
                    .wallclock_unix_ns = wallclock_unix_ns,
                },
            .file_uuid = fixed_uuid_in_namespace(uuid_namespace, identity_tag + 2),
            .index_compression = IndexCompression::StoredForDeterministicTests,
            .disk_reserve_bytes = 0,
        };
    }

    std::unique_ptr<ProjectDocument> make_empty_document(
        const core::Uuid project_uuid, const std::uint64_t created_at_unix_ns) {
        return require_result_ptr(ProjectDocument::create(project_uuid, created_at_unix_ns));
    }

} // namespace lfs::test::licht
