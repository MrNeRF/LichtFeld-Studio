/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Localization string keys for type-safe access via LOC(Strings::X::Y)

namespace lichtfeld::Strings {

    namespace Menu {
        namespace File {
            inline constexpr const char* MENU = "menu.file";
            inline constexpr const char* NEW_PROJECT = "menu.file.new_project";
            inline constexpr const char* IMPORT_DATASET = "menu.file.import_dataset";
            inline constexpr const char* IMPORT_PLY = "menu.file.import_ply";
            inline constexpr const char* IMPORT_CHECKPOINT = "menu.file.import_checkpoint";
            inline constexpr const char* IMPORT_CONFIG = "menu.file.import_config";
            inline constexpr const char* EXPORT = "menu.file.export";
            inline constexpr const char* EXPORT_CONFIG = "menu.file.export_config";
            inline constexpr const char* EXIT = "menu.file.exit";
        } // namespace File

        namespace Edit {
            inline constexpr const char* MENU = "menu.edit";
            inline constexpr const char* INPUT_SETTINGS = "menu.edit.input_settings";
            inline constexpr const char* PREFERENCES = "menu.edit.preferences";
        } // namespace Edit

        namespace View {
            inline constexpr const char* MENU = "menu.view";
            inline constexpr const char* THEME = "menu.view.theme";
            inline constexpr const char* THEME_DARK = "menu.view.theme.dark";
            inline constexpr const char* THEME_LIGHT = "menu.view.theme.light";
            inline constexpr const char* DEBUG_INFO = "menu.view.debug_info";
        } // namespace View

        namespace Help {
            inline constexpr const char* MENU = "menu.help";
            inline constexpr const char* GETTING_STARTED = "menu.help.getting_started";
            inline constexpr const char* ABOUT = "menu.help.about";
        } // namespace Help
    } // namespace Menu

    namespace Window {
        inline constexpr const char* GETTING_STARTED = "window.getting_started";
        inline constexpr const char* ABOUT = "window.about";
        inline constexpr const char* INPUT_SETTINGS = "window.input_settings";
        inline constexpr const char* DEBUG_INFO = "window.debug_info";
        inline constexpr const char* EXPORT = "window.export";
        inline constexpr const char* SCENE = "window.scene";
        inline constexpr const char* RENDERING = "window.rendering";
        inline constexpr const char* TRAINING = "window.training";
        inline constexpr const char* PREFERENCES = "window.preferences";
    } // namespace Window

    namespace GettingStarted {
        inline constexpr const char* TITLE = "getting_started.title";
        inline constexpr const char* DESCRIPTION = "getting_started.description";
        inline constexpr const char* WIKI_SECTION = "getting_started.wiki_section";
        inline constexpr const char* VIDEO_LATEST = "getting_started.video_latest";
        inline constexpr const char* VIDEO_REALITY_SCAN = "getting_started.video_reality_scan";
        inline constexpr const char* VIDEO_COLMAP = "getting_started.video_colmap";
        inline constexpr const char* VIDEO_LICHTFELD = "getting_started.video_lichtfeld";
        inline constexpr const char* LOADING = "getting_started.loading";
    } // namespace GettingStarted

    namespace About {
        inline constexpr const char* TITLE = "about.title";
        inline constexpr const char* DESCRIPTION = "about.description";
        inline constexpr const char* BUILD_INFO = "about.build_info";
        inline constexpr const char* LINKS = "about.links";
        inline constexpr const char* REPOSITORY = "about.repository";
        inline constexpr const char* WEBSITE = "about.website";
        inline constexpr const char* AUTHORS = "about.authors";
        inline constexpr const char* LICENSE = "about.license";

        namespace BuildInfo {
            inline constexpr const char* VERSION = "about.build_info.version";
            inline constexpr const char* COMMIT = "about.build_info.commit";
            inline constexpr const char* BUILD_TYPE = "about.build_info.build_type";
            inline constexpr const char* PLATFORM = "about.build_info.platform";
            inline constexpr const char* CUDA_GL_INTEROP = "about.build_info.cuda_gl_interop";
        } // namespace BuildInfo

        namespace BuildType {
            inline constexpr const char* DEBUG = "about.build_type.debug";
            inline constexpr const char* RELEASE = "about.build_type.release";
        } // namespace BuildType

        namespace Platform {
            inline constexpr const char* WINDOWS = "about.platform.windows";
            inline constexpr const char* LINUX = "about.platform.linux";
            inline constexpr const char* UNKNOWN = "about.platform.unknown";
        } // namespace Platform

        namespace Interop {
            inline constexpr const char* ENABLED = "about.interop.enabled";
            inline constexpr const char* DISABLED = "about.interop.disabled";
        } // namespace Interop
    } // namespace About

    namespace Training {
        namespace Section {
            inline constexpr const char* DATASET = "training.section.dataset";
            inline constexpr const char* OPTIMIZATION = "training.section.optimization";
            inline constexpr const char* REFINEMENT = "training.section.refinement";
            inline constexpr const char* BILATERAL_GRID = "training.section.bilateral_grid";
            inline constexpr const char* MASKING = "training.section.masking";
            inline constexpr const char* EVALUATION = "training.section.evaluation";
            inline constexpr const char* LOSSES = "training.section.losses";
            inline constexpr const char* INITIALIZATION = "training.section.initialization";
            inline constexpr const char* THRESHOLDS = "training.section.thresholds";
            inline constexpr const char* SAVE_STEPS = "training.section.save_steps";
            inline constexpr const char* BASIC_PARAMS = "training.section.basic_params";
            inline constexpr const char* ADVANCED_PARAMS = "training.section.advanced_params";
            inline constexpr const char* SPARSITY = "training.section.sparsity";
            inline constexpr const char* PRUNING_GROWING = "training.section.pruning_growing";
        } // namespace Section

        namespace Dataset {
            inline constexpr const char* PATH = "training.dataset.path";
            inline constexpr const char* IMAGES = "training.dataset.images";
            inline constexpr const char* RESIZE_FACTOR = "training.dataset.resize_factor";
            inline constexpr const char* MAX_WIDTH = "training.dataset.max_width";
            inline constexpr const char* CPU_CACHE = "training.dataset.cpu_cache";
            inline constexpr const char* FS_CACHE = "training.dataset.fs_cache";
            inline constexpr const char* TEST_EVERY = "training.dataset.test_every";
            inline constexpr const char* OUTPUT = "training.dataset.output";
        } // namespace Dataset

        namespace Opt {
            inline constexpr const char* STRATEGY = "training.opt.strategy";
            inline constexpr const char* LEARNING_RATES = "training.opt.learning_rates";
            inline constexpr const char* ITERATIONS = "training.opt.iterations";
            inline constexpr const char* SH_DEGREE = "training.opt.sh_degree";
            inline constexpr const char* USE_BILATERAL = "training.opt.use_bilateral";
            inline constexpr const char* MASK_MODE = "training.opt.mask_mode";
            inline constexpr const char* ENABLE_EVAL = "training.opt.enable_eval";
            inline constexpr const char* SPARSITY = "training.opt.sparsity";
            inline constexpr const char* GUT = "training.opt.gut";
            inline constexpr const char* MIP_FILTER = "training.opt.mip_filter";
            inline constexpr const char* BG_MODULATION = "training.opt.bg_modulation";
            inline constexpr const char* LR_POSITION = "training.opt.lr.position";
            inline constexpr const char* LR_SH_COEFF = "training.opt.lr.sh_coeff";
            inline constexpr const char* LR_OPACITY = "training.opt.lr.opacity";
            inline constexpr const char* LR_SCALING = "training.opt.lr.scaling";
            inline constexpr const char* LR_ROTATION = "training.opt.lr.rotation";
        } // namespace Opt

        namespace Refinement {
            inline constexpr const char* REFINE_EVERY = "training.refinement.refine_every";
            inline constexpr const char* START_REFINE = "training.refinement.start_refine";
            inline constexpr const char* STOP_REFINE = "training.refinement.stop_refine";
            inline constexpr const char* GRADIENT_THR = "training.refinement.gradient_thr";
            inline constexpr const char* RESET_EVERY = "training.refinement.reset_every";
            inline constexpr const char* SH_UPGRADE_EVERY = "training.refinement.sh_upgrade_every";
        } // namespace Refinement

        namespace Mask {
            inline constexpr const char* INVERT = "training.mask.invert";
            inline constexpr const char* THRESHOLD = "training.mask.threshold";
            inline constexpr const char* PENALTY_WEIGHT = "training.mask.penalty_weight";
            inline constexpr const char* PENALTY_POWER = "training.mask.penalty_power";
            inline constexpr const char* NO_MASKS = "training.mask.no_masks";
        } // namespace Mask

        namespace Bilateral {
            inline constexpr const char* GRID_X = "training.bilateral.grid_x";
            inline constexpr const char* GRID_Y = "training.bilateral.grid_y";
            inline constexpr const char* GRID_W = "training.bilateral.grid_w";
            inline constexpr const char* LEARNING_RATE = "training.bilateral.learning_rate";
        } // namespace Bilateral

        namespace Masking {
            inline constexpr const char* INVERT_MASKS = "training.masking.invert_masks";
            inline constexpr const char* THRESHOLD = "training.masking.threshold";
            inline constexpr const char* PENALTY_WEIGHT = "training.masking.penalty_weight";
            inline constexpr const char* PENALTY_POWER = "training.masking.penalty_power";
        } // namespace Masking

        namespace Eval {
            inline constexpr const char* SAVE_IMAGES = "training.eval.save_images";
            inline constexpr const char* EVALUATION_STEPS = "training.eval.evaluation_steps";
        } // namespace Eval

        namespace Losses {
            inline constexpr const char* LAMBDA_DSSIM = "training.losses.lambda_dssim";
            inline constexpr const char* OPACITY_REG = "training.losses.opacity_reg";
            inline constexpr const char* SCALE_REG = "training.losses.scale_reg";
            inline constexpr const char* TV_LOSS_WEIGHT = "training.losses.tv_loss_weight";
        } // namespace Losses

        namespace Init {
            inline constexpr const char* INIT_OPACITY = "training.init.init_opacity";
            inline constexpr const char* INIT_SCALING = "training.init.init_scaling";
            inline constexpr const char* RANDOM_INIT = "training.init.random_init";
            inline constexpr const char* NUM_POINTS = "training.init.num_points";
            inline constexpr const char* EXTENT = "training.init.extent";
        } // namespace Init

        namespace Thresholds {
            inline constexpr const char* MIN_OPACITY = "training.thresholds.min_opacity";
            inline constexpr const char* PRUNE_OPACITY = "training.thresholds.prune_opacity";
            inline constexpr const char* GROW_SCALE_3D = "training.thresholds.grow_scale_3d";
            inline constexpr const char* GROW_SCALE_2D = "training.thresholds.grow_scale_2d";
            inline constexpr const char* PRUNE_SCALE_3D = "training.thresholds.prune_scale_3d";
            inline constexpr const char* PRUNE_SCALE_2D = "training.thresholds.prune_scale_2d";
            inline constexpr const char* PAUSE_AFTER_RESET = "training.thresholds.pause_after_reset";
            inline constexpr const char* REVISED_OPACITY = "training.thresholds.revised_opacity";
        } // namespace Thresholds

        namespace Tooltip {
            inline constexpr const char* INVERT_MASKS = "training.tooltip.invert_masks";
            inline constexpr const char* PENALTY_WEIGHT = "training.tooltip.penalty_weight";
            inline constexpr const char* PENALTY_POWER = "training.tooltip.penalty_power";
            inline constexpr const char* MASK_THRESHOLD = "training.tooltip.mask_threshold";
            inline constexpr const char* MIP_FILTER = "training.tooltip.mip_filter";
            inline constexpr const char* KEEP_MODEL = "training.tooltip.keep_model";
        } // namespace Tooltip

        namespace Status {
            inline constexpr const char* ENABLED = "training.status.enabled";
            inline constexpr const char* DISABLED = "training.status.disabled";
            inline constexpr const char* YES = "training.status.yes";
            inline constexpr const char* NO = "training.status.no";
        } // namespace Status

        namespace Button {
            inline constexpr const char* START = "training.button.start";
            inline constexpr const char* RESUME = "training.button.resume";
            inline constexpr const char* PAUSE = "training.button.pause";
            inline constexpr const char* STOP = "training.button.stop";
            inline constexpr const char* RESET = "training.button.reset";
            inline constexpr const char* CLEAR = "training.button.clear";
            inline constexpr const char* SAVE_CHECKPOINT = "training.button.save_checkpoint";
            inline constexpr const char* SWITCH_EDIT_MODE = "training.button.switch_edit_mode";
            inline constexpr const char* ADD = "training.button.add";
            inline constexpr const char* REMOVE = "training.button.remove";
        } // namespace Button
    } // namespace Training

    namespace Scene {
        inline constexpr const char* ADD_PLY = "scene.add_ply";
        inline constexpr const char* ADD_GROUP = "scene.add_group";
        inline constexpr const char* ADD_GROUP_ELLIPSIS = "scene.add_group_ellipsis";
        inline constexpr const char* EXPORT = "scene.export";
        inline constexpr const char* DELETE_ITEM = "scene.delete";
        inline constexpr const char* RENAME = "scene.rename";
        inline constexpr const char* DUPLICATE_ITEM = "scene.duplicate";
        inline constexpr const char* GO_TO_CAMERA_VIEW = "scene.go_to_camera_view";
        inline constexpr const char* GO_TO_CAM_VIEW = "scene.go_to_cam_view";
        inline constexpr const char* FIT_TO_SCENE = "scene.fit_to_scene";
        inline constexpr const char* FIT_TO_SCENE_TRIMMED = "scene.fit_to_scene_trimmed";
        inline constexpr const char* MERGE_TO_SINGLE_PLY = "scene.merge_to_single_ply";
        inline constexpr const char* MOVE_TO = "scene.move_to";
        inline constexpr const char* MOVE_TO_ROOT = "scene.move_to_root";
        inline constexpr const char* IMAGES = "scene.images";
        inline constexpr const char* NO_IMAGES = "scene.no_images";
        inline constexpr const char* USE_FILE_BROWSER = "scene.use_file_browser";
        inline constexpr const char* MOVE_NODE = "scene.move_node";
        inline constexpr const char* MODELS = "scene.models";
        inline constexpr const char* FILTER = "scene.filter";
        inline constexpr const char* NO_DATA_LOADED = "scene.no_data_loaded";
        inline constexpr const char* USE_FILE_MENU = "scene.use_file_menu";
        inline constexpr const char* NO_MODELS_LOADED = "scene.no_models_loaded";
        inline constexpr const char* RIGHT_CLICK_TO_ADD = "scene.right_click_to_add";
        inline constexpr const char* NO_ACTIONS = "scene.no_actions";
        inline constexpr const char* NO_GROUPS_AVAILABLE = "scene.no_groups_available";
        inline constexpr const char* DELETE_NODE = "scene.delete_node";
        inline constexpr const char* CANNOT_DELETE_TRAINING = "scene.cannot_delete_training";
    } // namespace Scene

    namespace Export {
        inline constexpr const char* TITLE = "export.title";
        inline constexpr const char* FORMAT_PLY_STANDARD = "export.format.ply_standard";
        inline constexpr const char* FORMAT_SOG_SUPERSPLAT = "export.format.sog_supersplat";
        inline constexpr const char* FORMAT_SPZ_NIANTIC = "export.format.spz_niantic";
        inline constexpr const char* FORMAT_HTML_VIEWER = "export.format.html_viewer";
        inline constexpr const char* SELECT_MODELS = "export.select_models";
        inline constexpr const char* ALL = "export.all";
        inline constexpr const char* NONE = "export.none";
        inline constexpr const char* CANCEL = "export.cancel";
        inline constexpr const char* EXPORT = "export.export";
        inline constexpr const char* EXPORTING = "export.exporting";
        inline constexpr const char* WRITING_PLY = "export.writing_ply";
        inline constexpr const char* WRITING_SPZ = "export.writing_spz";
        inline constexpr const char* COMPLETE = "export.complete";
        inline constexpr const char* FAILED = "export.failed";
        inline constexpr const char* SELECT_AT_LEAST_ONE = "export.select_at_least_one";
    } // namespace Export

    namespace Common {
        inline constexpr const char* OK = "common.ok";
        inline constexpr const char* CANCEL = "common.cancel";
        inline constexpr const char* CLOSE = "common.close";
        inline constexpr const char* SAVE = "common.save";
        inline constexpr const char* LOAD = "common.load";
        inline constexpr const char* ADD = "common.add";
        inline constexpr const char* REMOVE = "common.remove";
        inline constexpr const char* DELETE_ITEM = "common.delete";
        inline constexpr const char* EDIT = "common.edit";
        inline constexpr const char* BROWSE = "common.browse";
        inline constexpr const char* APPLY = "common.apply";
        inline constexpr const char* RESET = "common.reset";
    } // namespace Common

    namespace Status {
        inline constexpr const char* READY = "status.ready";
        inline constexpr const char* TRAINING = "status.training";
        inline constexpr const char* PAUSED = "status.paused";
        inline constexpr const char* STOPPED = "status.stopped";
        inline constexpr const char* STOPPING = "status.stopping";
        inline constexpr const char* COMPLETE = "status.complete";
        inline constexpr const char* ERROR_STATE = "status.error";
        inline constexpr const char* LOADING = "status.loading";
        inline constexpr const char* EMPTY = "status.empty";
        inline constexpr const char* MODE = "status.mode";
        inline constexpr const char* GAUSSIANS = "status.gaussians";
        inline constexpr const char* ITERATION = "status.iteration";
        inline constexpr const char* FPS = "status.fps";
        inline constexpr const char* STEP = "status.step";
        inline constexpr const char* LOSS = "status.loss";
        inline constexpr const char* ETA = "status.eta";
        inline constexpr const char* UNKNOWN = "status.unknown";
        inline constexpr const char* DATASET_NO_TRAINER = "status.dataset_no_trainer";
        inline constexpr const char* DATASET_READY = "status.dataset_ready";
        inline constexpr const char* TRAINING_PAUSED = "status.training_paused";
        inline constexpr const char* TRAINING_FINISHED = "status.training_finished";
        inline constexpr const char* PLY_MODELS_COUNT = "status.ply_models_count";
    } // namespace Status

    namespace Mode {
        inline constexpr const char* EMPTY = "mode.empty";
        inline constexpr const char* DATASET = "mode.dataset";
        inline constexpr const char* EDIT_MODE = "mode.edit_mode";
        inline constexpr const char* PLY_MODELS = "mode.ply_models";
    } // namespace Mode

    namespace Messages {
        inline constexpr const char* NO_DATA_LOADED = "messages.no_data_loaded";
        inline constexpr const char* USE_FILE_MENU = "messages.use_file_menu";
        inline constexpr const char* NO_MODELS_LOADED = "messages.no_models_loaded";
        inline constexpr const char* RIGHT_CLICK_TO_ADD = "messages.right_click_to_add";
        inline constexpr const char* TRAINING_COMPLETE = "messages.training_complete";
        inline constexpr const char* TRAINING_STOPPED = "messages.training_stopped";
        inline constexpr const char* TRAINING_ERROR = "messages.training_error";
        inline constexpr const char* PARAM_MANAGER_UNAVAILABLE = "messages.param_manager_unavailable";
        inline constexpr const char* FAILED_TO_LOAD_PARAMS = "messages.failed_to_load_params";
    } // namespace Messages

    namespace Controls {
        inline constexpr const char* WASD = "controls.wasd";
        inline constexpr const char* ZOOM = "controls.zoom";
    } // namespace Controls

    namespace Preferences {
        inline constexpr const char* TITLE = "preferences.title";
        inline constexpr const char* LANGUAGE = "preferences.language";
        inline constexpr const char* SELECT_LANGUAGE = "preferences.select_language";
    } // namespace Preferences

    namespace MainPanel {
        inline constexpr const char* WINDOWS = "main_panel.windows";
        inline constexpr const char* SCENE_PANEL = "main_panel.scene_panel";
        inline constexpr const char* SHOW_CONSOLE = "main_panel.show_console";
        inline constexpr const char* HIDE_CONSOLE = "main_panel.hide_console";
        inline constexpr const char* BACKGROUND = "main_panel.background";
        inline constexpr const char* COLOR = "main_panel.color";
        inline constexpr const char* SHOW_COORD_AXES = "main_panel.show_coord_axes";
        inline constexpr const char* VISIBLE_AXES = "main_panel.visible_axes";
        inline constexpr const char* AXES_SIZE = "main_panel.axes_size";
        inline constexpr const char* SHOW_PIVOT = "main_panel.show_pivot";
        inline constexpr const char* SHOW_GRID = "main_panel.show_grid";
        inline constexpr const char* GRID_OPACITY = "main_panel.grid_opacity";
        inline constexpr const char* PLANE = "main_panel.plane";
        inline constexpr const char* PLANE_YZ = "main_panel.plane_yz";
        inline constexpr const char* PLANE_XZ = "main_panel.plane_xz";
        inline constexpr const char* PLANE_XY = "main_panel.plane_xy";
        inline constexpr const char* CAMERA_FRUSTUMS = "main_panel.camera_frustums";
        inline constexpr const char* POINT_CLOUD_MODE = "main_panel.point_cloud_mode";
        inline constexpr const char* DESATURATE_UNSELECTED = "main_panel.desaturate_unselected";
        inline constexpr const char* FOV = "main_panel.fov";
        inline constexpr const char* SH_DEGREE = "main_panel.sh_degree";
        inline constexpr const char* EQUIRECTANGULAR = "main_panel.equirectangular";
        inline constexpr const char* GUT_MODE = "main_panel.gut_mode";
        inline constexpr const char* MIP_FILTER = "main_panel.mip_filter";
        inline constexpr const char* SELECTION_COLORS = "main_panel.selection_colors";
        inline constexpr const char* COMMITTED = "main_panel.committed";
        inline constexpr const char* PREVIEW = "main_panel.preview";
        inline constexpr const char* CENTER_MARKER = "main_panel.center_marker";
        inline constexpr const char* SELECTION_GROUPS = "main_panel.selection_groups";
        inline constexpr const char* ADD_GROUP = "main_panel.add_group";
        inline constexpr const char* NO_SELECTION_GROUPS = "main_panel.no_selection_groups";
        inline constexpr const char* CLEAR = "main_panel.clear";
    } // namespace MainPanel

    namespace Toolbar {
        inline constexpr const char* SELECTION = "toolbar.selection";
        inline constexpr const char* TRANSLATE = "toolbar.translate";
        inline constexpr const char* ROTATE = "toolbar.rotate";
        inline constexpr const char* SCALE = "toolbar.scale";
        inline constexpr const char* MIRROR = "toolbar.mirror";
        inline constexpr const char* PAINTING = "toolbar.painting";
        inline constexpr const char* ALIGN_3POINT = "toolbar.align_3point";
        inline constexpr const char* CROP_BOX = "toolbar.crop_box";
        inline constexpr const char* BRUSH_SELECTION = "toolbar.brush_selection";
        inline constexpr const char* RECT_SELECTION = "toolbar.rect_selection";
        inline constexpr const char* POLYGON_SELECTION = "toolbar.polygon_selection";
        inline constexpr const char* LASSO_SELECTION = "toolbar.lasso_selection";
        inline constexpr const char* RING_SELECTION = "toolbar.ring_selection";
        inline constexpr const char* LOCAL_SPACE = "toolbar.local_space";
        inline constexpr const char* WORLD_SPACE = "toolbar.world_space";
        inline constexpr const char* RESIZE_BOUNDS = "toolbar.resize_bounds";
        inline constexpr const char* MIRROR_X = "toolbar.mirror_x";
        inline constexpr const char* MIRROR_Y = "toolbar.mirror_y";
        inline constexpr const char* MIRROR_Z = "toolbar.mirror_z";
        inline constexpr const char* RESET_DEFAULT = "toolbar.reset_default";
        inline constexpr const char* HOME = "toolbar.home";
        inline constexpr const char* FULLSCREEN = "toolbar.fullscreen";
        inline constexpr const char* TOGGLE_UI = "toolbar.toggle_ui";
        inline constexpr const char* SPLAT_RENDERING = "toolbar.splat_rendering";
        inline constexpr const char* POINT_CLOUD = "toolbar.point_cloud";
        inline constexpr const char* GAUSSIAN_RINGS = "toolbar.gaussian_rings";
        inline constexpr const char* CENTER_MARKERS = "toolbar.center_markers";
        inline constexpr const char* PERSPECTIVE = "toolbar.perspective";
        inline constexpr const char* ORTHOGRAPHIC = "toolbar.orthographic";
    } // namespace Toolbar

    namespace Transform {
        inline constexpr const char* NODE = "transform.node";
        inline constexpr const char* SPACE = "transform.space";
        inline constexpr const char* WORLD = "transform.world";
        inline constexpr const char* LOCAL = "transform.local";
        inline constexpr const char* POSITION = "transform.position";
        inline constexpr const char* ROTATION = "transform.rotation";
        inline constexpr const char* ROTATION_DEGREES = "transform.rotation_degrees";
        inline constexpr const char* SCALE = "transform.scale";
        inline constexpr const char* UNIFORM_SCALE = "transform.uniform_scale";
        inline constexpr const char* USE_GIZMO = "transform.use_gizmo";
        inline constexpr const char* RESET_ALL = "transform.reset_all";
        inline constexpr const char* RESET_TRANSFORM = "transform.reset_transform";
        inline constexpr const char* NODES_SELECTED = "transform.nodes_selected";
    } // namespace Transform

    namespace CropBox {
        inline constexpr const char* TITLE = "cropbox.title";
        inline constexpr const char* NOT_VISIBLE = "cropbox.not_visible";
        inline constexpr const char* NO_SELECTION = "cropbox.no_selection";
        inline constexpr const char* INVALID = "cropbox.invalid";
        inline constexpr const char* POSITION = "cropbox.position";
        inline constexpr const char* ROTATION = "cropbox.rotation";
        inline constexpr const char* SIZE = "cropbox.size";
        inline constexpr const char* APPEARANCE = "cropbox.appearance";
        inline constexpr const char* LINE_WIDTH = "cropbox.line_width";
        inline constexpr const char* INSTRUCTIONS = "cropbox.instructions";
    } // namespace CropBox

    namespace FileBrowser {
        inline constexpr const char* TITLE = "file_browser.title";
        inline constexpr const char* QUICK_ACCESS = "file_browser.quick_access";
        inline constexpr const char* CURRENT_DIR = "file_browser.current_dir";
        inline constexpr const char* HOME = "file_browser.home";
        inline constexpr const char* CURRENT_PATH = "file_browser.current_path";
        inline constexpr const char* PARENT_DIR = "file_browser.parent_dir";
        inline constexpr const char* DIRECTORY = "file_browser.directory";
        inline constexpr const char* SELECTED = "file_browser.selected";
        inline constexpr const char* LOAD_DATASET = "file_browser.load_dataset";
        inline constexpr const char* LOAD_SOG = "file_browser.load_sog";
        inline constexpr const char* LOAD_PLY = "file_browser.load_ply";
        inline constexpr const char* ENTER_DIR = "file_browser.enter_dir";
    } // namespace FileBrowser

    namespace TrainingPanel {
        inline constexpr const char* START_TRAINING = "training_panel.start_training";
        inline constexpr const char* RESUME_TRAINING = "training_panel.resume_training";
        inline constexpr const char* PAUSE = "training_panel.pause";
        inline constexpr const char* RESUME = "training_panel.resume";
        inline constexpr const char* STOP = "training_panel.stop";
        inline constexpr const char* RESET = "training_panel.reset";
        inline constexpr const char* CLEAR = "training_panel.clear";
        inline constexpr const char* SWITCH_EDIT_MODE = "training_panel.switch_edit_mode";
        inline constexpr const char* SAVE_CHECKPOINT = "training_panel.save_checkpoint";
        inline constexpr const char* CHECKPOINT_SAVED = "training_panel.checkpoint_saved";
        inline constexpr const char* IDLE = "training_panel.idle";
        inline constexpr const char* RUNNING = "training_panel.running";
        inline constexpr const char* FINISHED = "training_panel.finished";
        inline constexpr const char* SAVE_STEPS = "training_panel.save_steps";
        inline constexpr const char* NEW_STEP = "training_panel.new_step";
        inline constexpr const char* NO_SAVE_STEPS = "training_panel.no_save_steps";
        inline constexpr const char* SPARSITY = "training_panel.sparsity";
        inline constexpr const char* PRUNING_GROWING = "training_panel.pruning_growing";
    } // namespace TrainingPanel

    namespace Tooltip {
        inline constexpr const char* GUT_MODE = "tooltip.gut_mode";
        inline constexpr const char* MIP_FILTER = "tooltip.mip_filter";
        inline constexpr const char* POINT_CLOUD_FORCED = "tooltip.point_cloud_forced";
        inline constexpr const char* DESATURATE_UNSELECTED = "tooltip.desaturate_unselected";
        inline constexpr const char* LOCKED = "tooltip.locked";
        inline constexpr const char* UNLOCKED = "tooltip.unlocked";
        inline constexpr const char* POINT_SIZE = "tooltip.point_size";
        inline constexpr const char* SCALE_CAMERA = "tooltip.scale_camera";
    } // namespace Tooltip

    namespace ExitPopup {
        inline constexpr const char* TITLE = "exit_popup.title";
        inline constexpr const char* MESSAGE = "exit_popup.message";
        inline constexpr const char* UNSAVED_WARNING = "exit_popup.unsaved_warning";
        inline constexpr const char* EXIT = "exit_popup.exit";
    } // namespace ExitPopup

    namespace SaveDirPopup {
        inline constexpr const char* TITLE = "save_dir_popup.title";
        inline constexpr const char* CONFIGURE_OUTPUT = "save_dir_popup.configure_output";
        inline constexpr const char* DATASET_LABEL = "save_dir_popup.dataset_label";
        inline constexpr const char* OUTPUT_DIR = "save_dir_popup.output_dir";
        inline constexpr const char* HELP_TEXT = "save_dir_popup.help_text";
    } // namespace SaveDirPopup

    namespace ExportDialog {
        inline constexpr const char* FORMAT = "export_dialog.format";
        inline constexpr const char* MODELS = "export_dialog.models";
        inline constexpr const char* NO_MODELS = "export_dialog.no_models";
        inline constexpr const char* SH_DEGREE = "export_dialog.sh_degree";
        inline constexpr const char* EXPORT_MERGED = "export_dialog.export_merged";
    } // namespace ExportDialog

    namespace SelectionGroup {
        inline constexpr const char* LOCK = "selection_group.lock";
        inline constexpr const char* UNLOCK = "selection_group.unlock";
    } // namespace SelectionGroup

    namespace FileBrowserExt {
        inline constexpr const char* DATASET = "file_browser_ext.dataset";
        inline constexpr const char* SOG = "file_browser_ext.sog";
        inline constexpr const char* NO_FILE_SELECTED = "file_browser_ext.no_file_selected";
        inline constexpr const char* NOT_A_DATASET = "file_browser_ext.not_a_dataset";
        inline constexpr const char* SOG_DIRECTORY = "file_browser_ext.sog_directory";
        inline constexpr const char* SOG_META = "file_browser_ext.sog_meta";
        inline constexpr const char* ERROR_MSG = "file_browser_ext.error";
    } // namespace FileBrowserExt

    namespace Progress {
        inline constexpr const char* LOSS = "progress.loss";
        inline constexpr const char* GAUSSIANS_COUNT = "progress.gaussians_count";
        inline constexpr const char* EXPORTING = "progress.exporting";
        inline constexpr const char* NUM_SPLATS = "progress.num_splats";
        inline constexpr const char* STATUS_LABEL = "progress.status_label";
    } // namespace Progress

    namespace InputSettings {
        inline constexpr const char* ACTIVE_PROFILE = "input_settings.active_profile";
        inline constexpr const char* SAVE_CURRENT_PROFILE = "input_settings.save_current_profile";
        inline constexpr const char* RESET_TO_DEFAULT = "input_settings.reset_to_default";
        inline constexpr const char* EXPORT = "input_settings.export";
        inline constexpr const char* IMPORT = "input_settings.import";
        inline constexpr const char* ACTION = "input_settings.action";
        inline constexpr const char* BINDING = "input_settings.binding";
        inline constexpr const char* REBIND = "input_settings.rebind";
        inline constexpr const char* CANCEL = "input_settings.cancel";
        inline constexpr const char* PRESS_KEY_OR_CLICK = "input_settings.press_key_or_click";
        inline constexpr const char* CLICK_AGAIN_DOUBLE = "input_settings.click_again_double";
        inline constexpr const char* TOOL_MODE = "input_settings.tool_mode";
        inline constexpr const char* SELECT_TOOL_MODE = "input_settings.select_tool_mode";
        inline constexpr const char* CURRENT_BINDINGS = "input_settings.current_bindings";
        inline constexpr const char* GLOBAL_BINDINGS_HINT = "input_settings.global_bindings_hint";
        inline constexpr const char* TOOL_BINDINGS_HINT = "input_settings.tool_bindings_hint";
        inline constexpr const char* SECTION_NAVIGATION = "input_settings.section.navigation";
        inline constexpr const char* SECTION_SELECTION = "input_settings.section.selection";
        inline constexpr const char* SECTION_BRUSH = "input_settings.section.brush";
        inline constexpr const char* SECTION_CROP_BOX = "input_settings.section.crop_box";
        inline constexpr const char* SECTION_EDITING = "input_settings.section.editing";
        inline constexpr const char* SECTION_VIEW = "input_settings.section.view";
        inline constexpr const char* MODE_GLOBAL = "input_settings.mode.global";
        inline constexpr const char* MODE_SELECTION = "input_settings.mode.selection";
        inline constexpr const char* MODE_BRUSH = "input_settings.mode.brush";
        inline constexpr const char* MODE_TRANSLATE = "input_settings.mode.translate";
        inline constexpr const char* MODE_ROTATE = "input_settings.mode.rotate";
        inline constexpr const char* MODE_SCALE = "input_settings.mode.scale";
        inline constexpr const char* MODE_ALIGN = "input_settings.mode.align";
        inline constexpr const char* MODE_CROP_BOX = "input_settings.mode.crop_box";
        inline constexpr const char* MODE_UNKNOWN = "input_settings.mode.unknown";
    } // namespace InputSettings

    namespace DebugInfo {
        inline constexpr const char* FREE_MEMORY = "debug_info.free_memory";
        inline constexpr const char* ENABLE_TRACING = "debug_info.enable_tracing";
        inline constexpr const char* RECORDED_OPERATIONS = "debug_info.recorded_operations";
        inline constexpr const char* CLEAR_HISTORY = "debug_info.clear_history";
        inline constexpr const char* PRINT_TO_LOG = "debug_info.print_to_log";
    } // namespace DebugInfo

    namespace TrainingParams {
        inline constexpr const char* STRATEGY = "training_params.strategy";
        inline constexpr const char* ITERATIONS = "training_params.iterations";
        inline constexpr const char* MAX_GAUSSIANS = "training_params.max_gaussians";
        inline constexpr const char* SH_DEGREE = "training_params.sh_degree";
        inline constexpr const char* TILE_MODE = "training_params.tile_mode";
        inline constexpr const char* NUM_WORKERS = "training_params.num_workers";
        inline constexpr const char* STEPS_SCALER = "training_params.steps_scaler";
        inline constexpr const char* BILATERAL_GRID = "training_params.bilateral_grid";
        inline constexpr const char* MASK_MODE = "training_params.mask_mode";
        inline constexpr const char* INVERT_MASKS = "training_params.invert_masks";
        inline constexpr const char* OPACITY_PENALTY_WEIGHT = "training_params.opacity_penalty_weight";
        inline constexpr const char* OPACITY_PENALTY_POWER = "training_params.opacity_penalty_power";
        inline constexpr const char* MASK_THRESHOLD = "training_params.mask_threshold";
        inline constexpr const char* SPARSITY = "training_params.sparsity";
        inline constexpr const char* GUT = "training_params.gut";
        inline constexpr const char* MIP_FILTER = "training_params.mip_filter";
        inline constexpr const char* BG_MODULATION = "training_params.bg_modulation";
        inline constexpr const char* EVALUATION = "training_params.evaluation";
        inline constexpr const char* INIT_OPACITY = "training_params.init_opacity";
        inline constexpr const char* INIT_SCALING = "training_params.init_scaling";
        inline constexpr const char* RANDOM_INIT = "training_params.random_init";
        inline constexpr const char* NUM_POINTS = "training_params.num_points";
        inline constexpr const char* EXTENT = "training_params.extent";
        inline constexpr const char* PRUNE_SCALE_3D = "training_params.prune_scale_3d";
        inline constexpr const char* PRUNE_OPACITY = "training_params.prune_opacity";
        inline constexpr const char* PRUNE_SCALE_2D = "training_params.prune_scale_2d";
        inline constexpr const char* PAUSE_AFTER_RESET = "training_params.pause_after_reset";
        inline constexpr const char* REVISED_OPACITY = "training_params.revised_opacity";
        inline constexpr const char* SPARSIFY_STEPS = "training_params.sparsify_steps";
        inline constexpr const char* INIT_RHO = "training_params.init_rho";
        inline constexpr const char* PRUNE_RATIO = "training_params.prune_ratio";
        inline constexpr const char* NEW_EVAL_STEP = "training_params.new_eval_step";
        inline constexpr const char* DISABLED = "training_params.disabled";
    } // namespace TrainingParams

    namespace ImagePreview {
        inline constexpr const char* VIEW = "image_preview.view";
        inline constexpr const char* NAVIGATE = "image_preview.navigate";
        inline constexpr const char* FIT_TO_WINDOW = "image_preview.fit_to_window";
        inline constexpr const char* SHOW_INFO_PANEL = "image_preview.show_info_panel";
        inline constexpr const char* SHOW_MASK_OVERLAY = "image_preview.show_mask_overlay";
        inline constexpr const char* RESET_VIEW = "image_preview.reset_view";
        inline constexpr const char* ACTUAL_SIZE = "image_preview.actual_size";
        inline constexpr const char* PREVIOUS = "image_preview.previous";
        inline constexpr const char* NEXT = "image_preview.next";
        inline constexpr const char* FIRST = "image_preview.first";
        inline constexpr const char* LAST = "image_preview.last";
        inline constexpr const char* NAME = "image_preview.name";
        inline constexpr const char* FORMAT = "image_preview.format";
        inline constexpr const char* SIZE_MB = "image_preview.size_mb";
        inline constexpr const char* SIZE_KB = "image_preview.size_kb";
        inline constexpr const char* SIZE_BYTES = "image_preview.size_bytes";
        inline constexpr const char* MODIFIED = "image_preview.modified";
        inline constexpr const char* PATH = "image_preview.path";
        inline constexpr const char* MEGAPIXELS = "image_preview.megapixels";
        inline constexpr const char* CHANNELS = "image_preview.channels";
        inline constexpr const char* COLOR_SPACE = "image_preview.color_space";
        inline constexpr const char* CAMERA = "image_preview.camera";
        inline constexpr const char* LENS = "image_preview.lens";
        inline constexpr const char* FOCAL_LENGTH = "image_preview.focal_length";
        inline constexpr const char* FOCAL_35MM = "image_preview.focal_35mm";
        inline constexpr const char* EXPOSURE = "image_preview.exposure";
        inline constexpr const char* APERTURE = "image_preview.aperture";
        inline constexpr const char* DATE = "image_preview.date";
        inline constexpr const char* SOFTWARE = "image_preview.software";
        inline constexpr const char* FIT_STATUS = "image_preview.fit_status";
        inline constexpr const char* OVERLAY_STATUS = "image_preview.overlay_status";
        inline constexpr const char* FILE_LABEL = "image_preview.file_label";
        inline constexpr const char* VISIBLE = "image_preview.visible";
        inline constexpr const char* HIDDEN = "image_preview.hidden";
        inline constexpr const char* MASK_SECTION = "image_preview.mask_section";
    } // namespace ImagePreview

    namespace Startup {
        inline constexpr const char* SUPPORTED_BY = "startup.supported_by";
        inline constexpr const char* CLICK_TO_CONTINUE = "startup.click_to_continue";
        inline constexpr const char* DROP_FILES_TITLE = "startup.drop_files_title";
        inline constexpr const char* DROP_FILES_SUBTITLE = "startup.drop_files_subtitle";
        inline constexpr const char* DROP_FILES_HINT = "startup.drop_files_hint";
        inline constexpr const char* DROP_TO_IMPORT = "startup.drop_to_import";
        inline constexpr const char* DROP_TO_IMPORT_SUBTITLE = "startup.drop_to_import_subtitle";
    } // namespace Startup

    namespace Axis {
        inline constexpr const char* X = "axis.x";
        inline constexpr const char* Y = "axis.y";
        inline constexpr const char* Z = "axis.z";
        inline constexpr const char* U = "axis.u";
    } // namespace Axis

} // namespace lichtfeld::Strings
