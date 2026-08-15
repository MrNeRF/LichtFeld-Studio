/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "licht_test_support.hpp"

#include "io/project_document.hpp"

#include "project/session_state.hpp"
#include "sequencer/timeline.hpp"

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <ranges>

namespace lfs::test::licht {

    using core::DataType;
    using core::Device;
    using core::Tensor;
    using core::Uuid;
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
        "sequencer_visible":true,"python_console_visible":true,
        "window":{"x":101,"y":202,"width":1440,"height":900,"fullscreen":false,
                  "maximized":true,"restore_x":11,"restore_y":22,
                  "restore_width":1280,"restore_height":720}
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
        "crop_filter_for_selection":true,"apply_appearance_correction":true,"ppisp_mode":0,
        "background_color":[0.1,0.2,0.3],"environment_builtin":null,
        "environment_exposure":1.75,"environment_rotation_degrees":42.0,
        "show_coord_axes":true,"axes_size":3.5,"axes_visibility":[true,false,true],
        "show_grid":true,"grid_plane":2,"grid_opacity":0.65,
        "point_cloud_mode":true,"voxel_size":0.025,"show_rings":true,"ring_width":0.04,
        "show_center_markers":true,"show_camera_frustums":true,"camera_frustum_scale":0.9,
        "train_camera_color":[0.3,0.4,0.5],"eval_camera_color":[0.6,0.7,0.8],"show_pivot":true,
        "split_view_mode":3,"gt_comparison_mode":2,"split_position":0.37,"split_view_offset":5,
        "raster_backend":"3dgut","equirectangular":true,"orthographic":true,"ortho_scale":77.0,
        "depth_view":true,"depth_view_min":0.5,"depth_view_max":55.0,"depth_visualization_mode":0,
        "selection_color_committed":[0.11,0.22,0.33],
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
        render["selection_color_committed"] = {0.11f, 0.22f, 0.33f};
        render["environment_reference_uuid"] = core::generate_uuid_v4().to_string();
        render.erase("gut");

        auto primary = panelCameraProjectStateToJson(
            "primary", rolled_panel_camera(1.0f));
        auto secondary =
            panelCameraProjectStateToJson(
                "secondary",
                rolled_panel_camera(20.0f));
        auto bookmark = panelCameraProjectStateToJson(
            "bookmark", rolled_panel_camera(40.0f));
        bookmark.erase("panel");
        bookmark["id"] = "bookmark.matrix";
        bookmark["name"] = "Rolled view";

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

        (void)require_result(prepareGuiSessionRestore(session));
        return session;
    }

    PopulatedProjectFixture::PopulatedProjectFixture() = default;
    PopulatedProjectFixture::~PopulatedProjectFixture() = default;
    PopulatedProjectFixture::PopulatedProjectFixture(PopulatedProjectFixture&&) noexcept = default;
    PopulatedProjectFixture& PopulatedProjectFixture::operator=(PopulatedProjectFixture&&) noexcept = default;

    namespace {

        core::Uuid matrix_uuid(const std::uint64_t tag) {
            return fixed_uuid_in_namespace(0x73000000, tag);
        }

        lfs::core::param::OptimizationParameters
        distinct_pending_parameters(
            const std::string_view strategy,
            const std::uint32_t tag) {
            using lfs::core::param::BackgroundMode;
            using lfs::core::param::MaskMode;
            using lfs::core::param::NormalLossSpace;

            const float delta = static_cast<float>(tag) * 0.001f;
            const int tag_i = static_cast<int>(tag);
            return {
                .iterations = 31'000 + tag,
                .sh_degree_interval = 900 + tag,
                .means_lr = 0.010f + delta,
                .means_lr_end = 0.001f + delta,
                .shs_lr = 0.020f + delta,
                .opacity_lr = 0.030f + delta,
                .scaling_lr = 0.040f + delta,
                .scaling_lr_end = 0.050f + delta,
                .rotation_lr = 0.060f + delta,
                .cropbox_lr_scale = 0.21f + delta,
                .cropbox_loss_weight = 0.22f + delta,
                .lambda_dssim = 0.23f + delta,
                .min_opacity = 0.024f + delta,
                .refine_every = 70 + tag,
                .start_refine = 700 + tag,
                .stop_refine = 15'000 + tag,
                .sh_degree = 2,
                .opacity_reg = 0.080f + delta,
                .scale_reg = 0.090f + delta,
                .init_opacity = 0.41f + delta,
                .init_scaling = 0.12f + delta,
                .max_cap = 900'000 + tag_i,
                .eval_steps = {1'111 + tag, 2'222 + tag},
                .save_steps = {3'333 + tag, 4'444 + tag},
                .bg_modulation = true,
                .enable_eval = true,
                .enable_save_eval_images = false,
                .headless = true,
                .auto_train = true,
                .no_splash = true,
                .debug_python = true,
                .debug_python_port = 10'000 + tag_i,
                .strategy = std::string(strategy),
                .mask_mode = MaskMode::AlphaConsistent,
                .invert_masks = true,
                .mask_threshold = 0.61f + delta,
                .mask_opacity_penalty_weight = 0.71f + delta,
                .mask_opacity_penalty_power = 1.51f + delta,
                .use_alpha_as_mask = false,
                .use_depth_loss = true,
                .depth_loss_weight = 1.71f + delta,
                .depth_loss_mode = "ssi-depth",
                .use_normal_loss = true,
                .normal_loss_weight = 0.31f + delta,
                .normal_consistency_weight = 0.32f + delta,
                .normal_flatten_weight = 0.33f + delta,
                .normal_loss_space = NormalLossSpace::World,
                .mip_filter = true,
                .bg_mode = BackgroundMode::Image,
                .bg_color = {0.11f + delta, 0.22f + delta, 0.33f + delta},
                .bg_image_path = std::format("excluded-background-{}.png", tag),
                .use_bilateral_grid = true,
                .bilateral_grid_X = 17 + tag_i,
                .bilateral_grid_Y = 18 + tag_i,
                .bilateral_grid_W = 9 + tag_i,
                .bilateral_grid_lr = 0.014f + delta,
                .tv_loss_weight = 7.1f + delta,
                .use_ppisp = true,
                .ppisp_lr = 0.015f + delta,
                .ppisp_reg_weight = 0.016f + delta,
                .ppisp_warmup_steps = 600 + tag_i,
                .ppisp_freeze_from_sidecar = true,
                .ppisp_sidecar_path = std::format("excluded-ppisp-{}.bin", tag),
                .ppisp_use_controller = true,
                .ppisp_freeze_gaussians_on_distill = false,
                .ppisp_controller_activation_step = 2'000 + tag_i,
                .ppisp_controller_lr = 0.017f + delta,
                .prune_opacity = 0.18f + delta,
                .reset_every = 2'500 + tag,
                .gut = strategy != lfs::core::param::kStrategyIGSPlus,
                .undistort = true,
                .steps_scaler = 1.2f + delta,
                .growth_grad_threshold = 0.023f + delta,
                .grow_fraction = 0.24f + delta,
                .grow_until_iter = 12'000 + tag,
                .opacity_decay = 0.025f + delta,
                .scale_decay = 0.026f + delta,
                .means_noise_weight = 40.0f + delta,
                .bounds_percentile = 0.72f + delta,
                .use_error_map = false,
                .use_edge_map = false,
                .random = true,
                .init_num_pts = 80'000 + tag_i,
                .init_extent = 4.0f + delta,
                .enable_sparsity = true,
                .sparsify_steps = 12'000 + tag_i,
                .init_rho = 0.027f + delta,
                .prune_ratio = 0.55f + delta,
                .config_file = std::format("excluded-config-{}.json", tag),
            };
        }

        ReferenceFingerprint matrix_fingerprint(
            const std::uint8_t tag,
            const FingerprintKind kind = FingerprintKind::File) {
            ReferenceFingerprint result;
            result.kind = kind;
            result.size = 10'000 + tag;
            result.mtime_unix_ns = 20'000 + tag;
            result.head_xxh3.bytes.fill(tag);
            result.tail_xxh3.bytes.fill(
                static_cast<std::uint8_t>(tag + 1));
            Hash128 full;
            full.bytes.fill(static_cast<std::uint8_t>(tag + 2));
            result.full_xxh3 = full;
            return result;
        }

        Tensor uint8_tensor(
            const std::initializer_list<std::uint8_t> values,
            const lfs::core::TensorShape& shape,
            const Device device = Device::CPU) {
            Tensor result =
                Tensor::empty(shape, Device::CPU, DataType::UInt8);
            std::ranges::copy(values, result.ptr<std::uint8_t>());
            return result.to(device);
        }

        PointCloudPayload make_matrix_point_cloud() {
            auto point_cloud = std::make_shared<lfs::core::PointCloud>();
            point_cloud->means = Tensor::from_vector(
                std::vector<float>{0.0f, 1.0f, 2.0f, -3.0f, 4.5f, 6.0f},
                {std::size_t{2}, std::size_t{3}}, Device::CPU);
            point_cloud->colors = uint8_tensor(
                {255, 128, 64, 0, 192, 255},
                {std::size_t{2}, std::size_t{3}});
            point_cloud->normals = Tensor::from_vector(
                std::vector<float>{0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f},
                {std::size_t{2}, std::size_t{3}}, Device::CPU);
            point_cloud->sh0 = Tensor::from_vector(
                std::vector<float>{1, 2, 3, 4, 5, 6},
                {std::size_t{2}, std::size_t{3}, std::size_t{1}},
                Device::CPU);
            point_cloud->shN = Tensor::from_vector(
                std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                {std::size_t{2}, std::size_t{3}, std::size_t{2}},
                Device::CPU);
            point_cloud->opacity = Tensor::from_vector(
                std::vector<float>{0.1f, 0.9f},
                {std::size_t{2}, std::size_t{1}}, Device::CPU);
            point_cloud->scaling = Tensor::from_vector(
                std::vector<float>{1, 2, 3, 4, 5, 6},
                {std::size_t{2}, std::size_t{3}}, Device::CPU);
            point_cloud->rotation = Tensor::from_vector(
                std::vector<float>{1, 0, 0, 0, 0.5f, 0.5f, 0.5f, 0.5f},
                {std::size_t{2}, std::size_t{4}}, Device::CPU);
            point_cloud->attribute_names = {
                "x", "y", "z", "red", "green", "blue", "opacity"};

            PointCloudPayload result(std::move(point_cloud));
            require_status(result.add_opaque_property(
                GeometryPropertyPlane{
                    .name = "vendor_score",
                    .components = 1,
                    .dtype = GeometryDtype::UInt16,
                    .encoding = 77,
                    .bytes = byte_values({0x10, 0x20, 0x30, 0x40}),
                }));
            return result;
        }

        MeshPayload make_matrix_mesh() {
            auto mesh = std::make_shared<lfs::core::MeshData>();
            mesh->vertices = Tensor::from_vector(
                std::vector<float>{-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f,
                                   1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f},
                {std::size_t{4}, std::size_t{3}}, Device::CPU);
            mesh->normals = Tensor::from_vector(
                std::vector<float>{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1},
                {std::size_t{4}, std::size_t{3}}, Device::CPU);
            mesh->tangents = Tensor::from_vector(
                std::vector<float>{1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1},
                {std::size_t{4}, std::size_t{4}}, Device::CPU);
            mesh->texcoords = Tensor::from_vector(
                std::vector<float>{0, 0, 1, 0, 1, 1, 0, 1},
                {std::size_t{4}, std::size_t{2}}, Device::CPU);
            mesh->colors = Tensor::from_vector(
                std::vector<float>{1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1},
                {std::size_t{4}, std::size_t{4}}, Device::CPU);
            mesh->indices = Tensor::from_vector(
                std::vector<std::int32_t>{0, 1, 2, 0, 2, 3},
                {std::size_t{2}, std::size_t{3}}, Device::CPU);
            mesh->texture_images.push_back(lfs::core::TextureImage{
                .pixels = {1, 2, 3, 4, 5, 6},
                .width = 2,
                .height = 1,
                .channels = 3,
            });
            lfs::core::Material material;
            material.name = "matrix-plane";
            material.base_color = {0.25f, 0.5f, 0.75f, 1.0f};
            material.emissive = {0.1f, 0.2f, 0.3f};
            material.metallic = 0.4f;
            material.roughness = 0.6f;
            material.ao = 0.8f;
            material.albedo_tex = 1;
            material.normal_tex = 1;
            material.metallic_roughness_tex = 1;
            material.emissive_tex = 1;
            material.ao_tex = 1;
            material.albedo_tex_path = "textures/albedo.png";
            material.normal_tex_path = "textures/normal.png";
            material.metallic_roughness_tex_path = "textures/mr.png";
            material.double_sided = true;
            mesh->materials.push_back(material);
            mesh->submeshes.push_back(lfs::core::Submesh{
                .start_index = 0,
                .index_count = 6,
                .material_index = 0,
            });

            MeshPayload result(std::move(mesh));
            require_status(result.add_opaque_property(
                GeometryPropertyPlane{
                    .name = "vendor_ids",
                    .components = 1,
                    .dtype = GeometryDtype::UInt32,
                    .encoding = 91,
                    .bytes = byte_values({0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
                                          0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00}),
                }));
            return result;
        }

    } // namespace

    std::vector<std::byte> byte_values(
        const std::initializer_list<std::uint8_t> values) {
        std::vector<std::byte> result;
        result.reserve(values.size());
        for (const std::uint8_t value : values) {
            result.push_back(static_cast<std::byte>(value));
        }
        return result;
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

    std::unique_ptr<core::SplatData> make_matrix_splat(const bool cuda) {
        const Device device = cuda ? Device::CUDA : Device::CPU;
        constexpr std::size_t count = 4;
        std::vector<float> means{
            1.0f,
            2.0f,
            3.0f,
            4.0f,
            5.0f,
            6.0f,
            7.0f,
            8.0f,
            9.0f,
            10.0f,
            11.0f,
            12.0f,
        };
        std::vector<float> sh0(count * 3);
        std::vector<float> shn(count * 3 * 3);
        std::vector<float> scaling(count * 3);
        std::vector<float> rotation(count * 4, 0.0f);
        std::vector<float> opacity(count);
        for (std::size_t index = 0; index < count; ++index) {
            sh0[index * 3] = static_cast<float>(index) + 0.1f;
            sh0[index * 3 + 1] = static_cast<float>(index) + 0.2f;
            sh0[index * 3 + 2] = static_cast<float>(index) + 0.3f;
            scaling[index * 3] = -1.0f - static_cast<float>(index);
            scaling[index * 3 + 1] = -2.0f;
            scaling[index * 3 + 2] = -3.0f;
            rotation[index * 4] = 1.0f;
            opacity[index] = -0.5f + static_cast<float>(index) * 0.1f;
        }
        for (std::size_t index = 0; index < shn.size(); ++index) {
            shn[index] = static_cast<float>(index) / 100.0f;
        }

        auto result = std::make_unique<lfs::core::SplatData>(
            1,
            Tensor::from_vector(
                means, {count, std::size_t{3}}, device),
            Tensor::from_vector(
                sh0, {count, std::size_t{1}, std::size_t{3}},
                device),
            Tensor::from_vector(
                shn, {count, std::size_t{3}, std::size_t{3}},
                device),
            Tensor::from_vector(
                scaling, {count, std::size_t{3}}, device),
            Tensor::from_vector(
                rotation, {count, std::size_t{4}}, device),
            Tensor::from_vector(
                opacity, {count, std::size_t{1}}, device),
            2.5f);
        result->set_active_sh_degree(0);
        result->deleted() =
            uint8_tensor({0, 1, 0, 1}, {count}, device);
        result->_densification_info = Tensor::from_vector(
            std::vector<float>{
                0.1f,
                0.2f,
                0.3f,
                0.4f,
                1.1f,
                1.2f,
                1.3f,
                1.4f,
            },
            {std::size_t{2}, count}, device);
        result->set_frozen_ranges({
            {.start = 1, .count = 2},
        });
        return result;
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

    PopulatedProjectFixture make_populated_project_fixture() {
        const Uuid project_uuid = matrix_uuid(1);
        const Uuid dataset_ref = matrix_uuid(2);
        const Uuid colmap_ref = matrix_uuid(3);
        const Uuid rad_ref = matrix_uuid(4);
        const Uuid rad_meta_ref = matrix_uuid(5);
        const Uuid background_ref = matrix_uuid(6);
        const Uuid environment_ref = matrix_uuid(7);
        const Uuid sequence_ref = matrix_uuid(8);
        const Uuid ppisp_ref = matrix_uuid(9);
        const Uuid root_node = matrix_uuid(20);
        const Uuid dataset_node = matrix_uuid(21);
        const Uuid camera_group_node = matrix_uuid(22);
        const Uuid image_group_node = matrix_uuid(23);
        const Uuid image_node = matrix_uuid(24);
        const Uuid sequence_node = matrix_uuid(25);
        const Uuid training_node = matrix_uuid(26);
        const Uuid imported_node = matrix_uuid(27);
        const Uuid point_node = matrix_uuid(28);
        const Uuid mesh_node = matrix_uuid(29);
        const Uuid live_rad_node = matrix_uuid(30);
        const Uuid crop_node = matrix_uuid(31);
        const Uuid ellipsoid_node = matrix_uuid(32);
        const Uuid camera_node = matrix_uuid(33);
        const Uuid checkpoint_uuid = matrix_uuid(34);

        auto document = require_result_ptr(
            ProjectDocument::create(project_uuid, 100));

        ProjectManifest manifest{
            .application_name = "LichtFeld Matrix Proof",
            .application_version = {2, 7, 3},
            .schema_version = {2, 0, 0},
            .minimum_reader_version = {1, 4, 0},
            .minimum_safe_writer_version = {1, 8, 0},
            .required_capabilities = {"p3-core", "retained-json"},
            .optional_capabilities = {"future-capability"},
        };
        auto& project = document->edit_project();
        require_status(project.set_manifest(manifest));
        const std::array lineage{matrix_uuid(40), matrix_uuid(41)};
        require_status(project.set_project_lineage(lineage));
        require_status(project.set_dataset_reference(dataset_ref));
        const ProjectGeoreference georeference{
            .crs = "EPSG:2056",
            .world_origin = {2'600'000.25, 1'200'000.5, 450.125},
            .world_unit_scale = 0.01,
            .world_origin_provenance =
                WorldOriginProvenance::CentralizeByPointCloud,
        };
        require_status(project.set_georeference(georeference));

        std::vector<ReferenceRecord> expected_references;
        const auto add_reference =
            [&](const Uuid& uuid, std::string key, std::string kind,
                std::string preferred, const LocatorBase base,
                const FingerprintKind fingerprint_kind,
                const std::uint8_t tag, const bool unresolved = false) {
                ReferenceRecord record{
                    .uuid = uuid,
                    .key = std::move(key),
                    .kind = std::move(kind),
                    .locator =
                        {
                            .preferred = std::move(preferred),
                            .base = base,
                            .absolute_fallback =
                                std::format("/fallback/{}", tag),
                        },
                    .fingerprint =
                        matrix_fingerprint(tag, fingerprint_kind),
                    .unresolved = unresolved,
                };
                expected_references.push_back(record);
                require_status(document->edit_references().upsert(record));
            };
        add_reference(dataset_ref, "dataset.root", "dataset",
                      "../dataset", LocatorBase::Project,
                      FingerprintKind::Directory, 1);
        add_reference(colmap_ref, "dataset.colmap", "colmap",
                      "sparse/0", LocatorBase::Dataset,
                      FingerprintKind::Directory, 2);
        add_reference(rad_ref, "splat.live_rad", "rad",
                      "assets/live.rad", LocatorBase::Project,
                      FingerprintKind::File, 3);
        add_reference(rad_meta_ref, "splat.live_rad.meta",
                      "rad_meta_cache", "assets/live.rad.meta",
                      LocatorBase::Project, FingerprintKind::File, 4,
                      true);
        add_reference(background_ref, "training.background",
                      "background_image", "images/background.png",
                      LocatorBase::Dataset, FingerprintKind::File, 5);
        add_reference(environment_ref, "view.environment",
                      "environment_map", "assets/studio.hdr",
                      LocatorBase::Project, FingerprintKind::File, 6);
        add_reference(sequence_ref, "sequence.directory",
                      "ply_sequence", "sequences/review",
                      LocatorBase::Project, FingerprintKind::Directory, 7);
        add_reference(ppisp_ref, "training.ppisp",
                      "ppisp_sidecar", "appearance/model.ppisp",
                      LocatorBase::Project, FingerprintKind::File, 8);

        const std::array<float, 16> edited_transform{
            1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1};
        std::vector<SceneNodeRecord> expected_nodes;
        const auto add_node = [&](SceneNodeRecord node) {
            expected_nodes.push_back(node);
            require_status(document->edit_scene_graph().upsert_node(node));
        };
        add_node(SceneNodeRecord{
            .uuid = root_node,
            .type = "group",
            .name = "Root group",
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = dataset_node,
            .type = "dataset",
            .name = "Dataset",
            .parent_uuid = root_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = camera_group_node,
            .type = "camera_group",
            .name = "Cameras",
            .parent_uuid = dataset_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = image_group_node,
            .type = "image_group",
            .name = "Train images",
            .parent_uuid = camera_group_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = image_node,
            .type = "image",
            .name = "Image placeholder",
            .parent_uuid = image_group_node,
            .child_order = 0,
        });
        add_node(SceneNodeRecord{
            .uuid = sequence_node,
            .type = "ply_sequence",
            .name = "Review sequence",
            .parent_uuid = root_node,
            .child_order = 1,
        });
        add_node(SceneNodeRecord{
            .uuid = training_node,
            .type = "splat",
            .name = "Training model",
            .parent_uuid = root_node,
            .child_order = 2,
            .payload =
                PayloadBinding{
                    .fourcc = "CKPT",
                    .instance_uuid = checkpoint_uuid,
                    .source_kind = "checkpoint",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = imported_node,
            .type = "splat",
            .name = "Edited imported splat",
            .parent_uuid = root_node,
            .child_order = 3,
            .local_transform = edited_transform,
            .visible = false,
            .locked = true,
            .payload_diverged = true,
            .payload =
                PayloadBinding{
                    .fourcc = "SPLT",
                    .instance_uuid = imported_node,
                    .source_kind = "spz",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = point_node,
            .type = "pointcloud",
            .name = "Survey points",
            .parent_uuid = root_node,
            .child_order = 4,
            .georef_pose =
                GeorefPose{
                    .rotation = {0.9238795325, 0.0, 0.3826834324, 0.0},
                    .translation = {1000.25, 2000.5, -10.75},
                },
            .payload =
                PayloadBinding{
                    .fourcc = "PCLD",
                    .instance_uuid = point_node,
                    .source_kind = "ply",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = mesh_node,
            .type = "mesh",
            .name = "Textured mesh",
            .parent_uuid = root_node,
            .child_order = 5,
            .payload =
                PayloadBinding{
                    .fourcc = "MESH",
                    .instance_uuid = mesh_node,
                    .source_kind = "obj",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = live_rad_node,
            .type = "splat",
            .name = "Live RAD",
            .parent_uuid = root_node,
            .child_order = 6,
            .payload =
                PayloadBinding{
                    .fourcc = "REFS",
                    .instance_uuid = rad_ref,
                    .reference_uuid = rad_ref,
                    .source_kind = "rad",
                },
        });
        add_node(SceneNodeRecord{
            .uuid = crop_node,
            .type = "cropbox",
            .name = "Crop region",
            .parent_uuid = root_node,
            .child_order = 7,
            .cropbox =
                CropBoxRecord{
                    .min = {-2.0f, -3.0f, -4.0f},
                    .max = {5.0f, 6.0f, 7.0f},
                    .inverse = true,
                    .enabled = true,
                    .color = {0.1f, 0.2f, 0.3f},
                    .line_width = 3.5f,
                },
        });
        add_node(SceneNodeRecord{
            .uuid = ellipsoid_node,
            .type = "ellipsoid",
            .name = "Ellipsoid region",
            .parent_uuid = root_node,
            .child_order = 8,
            .ellipsoid =
                EllipsoidRecord{
                    .radii = {2.0f, 3.0f, 4.0f},
                    .inverse = true,
                    .enabled = true,
                    .color = {0.4f, 0.5f, 0.6f},
                    .line_width = 4.5f,
                },
        });
        const CameraRecord camera{
            .uid = 71,
            .camera_id = 72,
            .rotation = {1, 0, 0, 0, 0, -1, 0, 1, 0},
            .translation = {1.25f, 2.5f, 3.75f},
            .focal_x = 1200.5f,
            .focal_y = 1199.5f,
            .center_x = 640.25f,
            .center_y = 360.75f,
            .radial_distortion = {0.1f, -0.01f, 0.001f},
            .tangential_distortion = {0.002f, -0.003f},
            .camera_model_type = 4,
            .camera_width = 1280,
            .camera_height = 720,
            .image_width = 1920,
            .image_height = 1080,
            .image_name = "frame_0071.png",
            .image_path = "images/frame_0071.png",
            .mask_path = "masks/frame_0071.png",
            .depth_path = "depth/frame_0071.png",
            .normal_path = "normals/frame_0071.png",
            .has_alpha = true,
            .split = "eval",
        };
        add_node(SceneNodeRecord{
            .uuid = camera_node,
            .type = "camera",
            .name = "Evaluation camera",
            .parent_uuid = image_group_node,
            .child_order = 1,
            .training_enabled = false,
            .camera = camera,
        });
        require_status(
            document->edit_scene_graph().set_training_model_uuid(
                training_node));

        auto splat = require_result(SplatChapterPayload::capture(
            *make_matrix_splat(false), SplatSourceKind::ImportedSpz, false));
        require_status(document->set_splat(imported_node, std::move(splat)));
        require_status(document->set_point_cloud(
            point_node, make_matrix_point_cloud()));
        require_status(document->set_mesh(mesh_node, make_matrix_mesh()));

        for (const auto& [node_uuid, fourcc, locator, tag] :
             std::array{
                 std::tuple{imported_node, std::string{"SPLT"},
                            std::string{"imports/source.spz"},
                            std::uint8_t{11}},
                 std::tuple{point_node, std::string{"PCLD"},
                            std::string{"imports/points.ply"},
                            std::uint8_t{12}},
                 std::tuple{mesh_node, std::string{"MESH"},
                            std::string{"imports/mesh.obj"},
                            std::uint8_t{13}},
             }) {
            require_status(project.upsert_embed_decision(EmbedDecision{
                .uuid = node_uuid,
                .node_uuid = node_uuid,
                .payload_fourcc = fourcc,
                .decision = "embedded",
                .reason = "matrix proof embedded payload",
            }));
            require_status(project.upsert_embedded_payload_provenance(
                EmbeddedPayloadProvenance{
                    .uuid = node_uuid,
                    .node_uuid = node_uuid,
                    .fourcc = fourcc,
                    .import_locator =
                        {
                            .preferred = locator,
                            .base = LocatorBase::Project,
                        },
                    .import_fingerprint =
                        matrix_fingerprint(tag),
                    .content_xxh3_128 = {},
                }));
        }
        require_status(project.upsert_embed_decision(EmbedDecision{
            .uuid = live_rad_node,
            .node_uuid = live_rad_node,
            .payload_fourcc = "REFS",
            .decision = "external",
            .reference_uuid = rad_ref,
            .reason = "live RAD remains external",
        }));
        require_status(project.upsert_provenance(ProvenanceRecord{
            .uuid = matrix_uuid(50),
            .kind = "legacy_view_path",
            .value = "imports/source.spz",
        }));
        require_status(project.upsert_provenance(ProvenanceRecord{
            .uuid = matrix_uuid(51),
            .kind = "mesh_texture_sources",
            .value =
                "textures/albedo.png;textures/normal.png;textures/mr.png",
        }));

        auto& selection = document->edit_selection();
        require_status(selection.set_groups(
            {
                lfs::core::SelectionGroup{
                    .id = 3,
                    .name = "Review",
                    .color = {0.2f, 0.4f, 0.8f},
                    .locked = false,
                },
                lfs::core::SelectionGroup{
                    .id = 7,
                    .name = "Protected",
                    .color = {0.9f, 0.1f, 0.3f},
                    .locked = true,
                },
            },
            7, 8));
        require_status(selection.upsert_slice(SelectionMaskSlice{
            .node_uuid = imported_node,
            .domain = lfs::core::SelectionDomain::Splat,
            .encoding = SelectionMaskEncoding::RawU8,
            .mask = {3, 0, 7, 3},
        }));
        require_status(selection.upsert_slice(SelectionMaskSlice{
            .node_uuid = point_node,
            .domain = lfs::core::SelectionDomain::PointCloud,
            .encoding = SelectionMaskEncoding::RawU8,
            .mask = {0, 7},
        }));
        require_status(selection.set_selected_node_uuids(
            {point_node, imported_node, camera_node}));

        ParameterManagerSnapshot parameters;
        parameters.active_strategy =
            std::string(lfs::core::param::kStrategyIGSPlus);
        parameters.mcmc_session = distinct_pending_parameters(lfs::core::param::kStrategyMCMC, 1);
        parameters.mrnf_session = distinct_pending_parameters(lfs::core::param::kStrategyMRNF, 2);
        parameters.igs_session = distinct_pending_parameters(lfs::core::param::kStrategyIGSPlus, 3);
        parameters.mcmc_current = distinct_pending_parameters(lfs::core::param::kStrategyMCMC, 4);
        parameters.mrnf_current = distinct_pending_parameters(lfs::core::param::kStrategyMRNF, 5);
        parameters.igs_current = distinct_pending_parameters(lfs::core::param::kStrategyIGSPlus, 6);
        for (auto* references : {
                 &parameters.mcmc_session_references,
                 &parameters.mrnf_session_references,
                 &parameters.igs_session_references,
                 &parameters.mcmc_current_references,
                 &parameters.mrnf_current_references,
                 &parameters.igs_current_references,
             }) {
            references->background_image_reference = background_ref;
            references->ppisp_reference = ppisp_ref;
        }
        parameters.dataset.images = "images_matrix";
        parameters.dataset.resize_factor = 4;
        parameters.dataset.test_every = 11;
        parameters.dataset.timelapse_images = {"frame_a.png", "frame_b.png"};
        parameters.dataset.timelapse_every = 77;
        parameters.dataset.max_width = 2048;
        parameters.dataset.min_track_length = 5;
        parameters.dataset.invert_masks = true;
        parameters.dataset.mask_threshold = 0.625f;
        parameters.dataset.centralize_dataset = "cameras";
        parameters.dataset.loading_params.use_cpu_memory = false;
        parameters.dataset.loading_params.min_cpu_free_memory_ratio = 0.25f;
        parameters.dataset.loading_params.min_cpu_free_GB = 3.5f;
        parameters.dataset.loading_params.print_cache_status = false;
        parameters.dataset.loading_params.print_status_freq_num = 123;
        parameters.dataset.loading_params.use_16bit_color = true;
        require_status(document->edit_parameters().set_snapshot(parameters));

        auto session = lfs::test::licht::make_populated_session_chapters();
        Json view = Json::parse(session.view.dom().dump());
        view["render_settings"]["environment_reference_uuid"] =
            environment_ref.to_string();
        session.view = require_result(ViewSessionChapter::parse(view.dump()));
        Json sequencer = Json::parse(session.sequencer.dom().dump());
        sequencer["ply_sequences"][0]["directory_reference_uuid"] =
            sequence_ref.to_string();
        session.sequencer = require_result(
            SequencerSessionChapter::parse(sequencer.dump()));
        document->edit_gui_layout() = session.gui_layout;
        document->edit_editor() = session.editor;
        document->edit_view() = session.view;
        document->edit_sequencer() = session.sequencer;
        document->edit_metrics() = session.metrics;

        PopulatedProjectFixture fixture;
        fixture.document = std::move(document);
        fixture.project_uuid = project_uuid;
        fixture.dataset_reference = dataset_ref;
        fixture.background_reference = background_ref;
        fixture.ppisp_reference = ppisp_ref;
        fixture.root_node = root_node;
        fixture.training_node = training_node;
        fixture.imported_node = imported_node;
        fixture.point_node = point_node;
        fixture.mesh_node = mesh_node;
        fixture.crop_node = crop_node;
        fixture.ellipsoid_node = ellipsoid_node;
        fixture.camera_node = camera_node;
        fixture.checkpoint_uuid = checkpoint_uuid;
        fixture.manifest = std::move(manifest);
        fixture.georeference = georeference;
        fixture.edited_transform = edited_transform;
        fixture.camera = camera;
        fixture.references = std::move(expected_references);
        fixture.nodes = std::move(expected_nodes);
        fixture.parameters = std::move(parameters);
        return fixture;
    }

} // namespace lfs::test::licht
