/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/environment.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <args.hxx>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <print>
#include <set>
#include <string_view>
#include <unordered_map>
#ifdef _WIN32
#include <Windows.h>
#endif

namespace {

    enum class ParseResult {
        Success,
        Help
    };

    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "mrnf", "mnrf", "lfs", "igs+"};

    std::optional<lfs::core::param::BackgroundMode> parse_bg_mode(const std::string& mode) {
        using lfs::core::param::BackgroundMode;
        if (mode == "solidcolor")
            return BackgroundMode::SolidColor;
        if (mode == "modulation")
            return BackgroundMode::Modulation;
        if (mode == "image")
            return BackgroundMode::Image;
        if (mode == "random")
            return BackgroundMode::Random;
        return std::nullopt;
    }

    std::string_view trim_view(std::string_view value) {
        constexpr std::string_view whitespace = " \t\n\r\v\f";
        const auto first = value.find_first_not_of(whitespace);
        if (first == std::string_view::npos)
            return {};
        const auto last = value.find_last_not_of(whitespace);
        return value.substr(first, (last - first + 1));
    }

    std::optional<std::array<float, 3>> parse_bg_hex_color(std::string_view color) {
        if (color.size() != 7 || color[0] != '#')
            return std::nullopt;

        std::array<float, 3> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            int channel = 0;
            auto [ptr, ec] = std::from_chars(color.data() + 1 + i * 2, color.data() + 1 + i * 2 + 2, channel, 16);
            if (ec != std::errc() || ptr != color.data() + 1 + i * 2 + 2)
                return std::nullopt;
            values[i] = static_cast<float>(channel) / 255.0f;
        }

        return values;
    }

    std::optional<std::array<float, 3>> parse_bg_rgb_color(std::string_view color) {
        if (color.size() < 7 || color.front() != '(' || color.back() != ')')
            return std::nullopt;

        std::array<float, 3> values{};
        std::string_view inner = color.substr(1, color.size() - 2);

        for (size_t i = 0; i < values.size(); ++i) {
            auto comma_pos = inner.find(',');
            if (i < 2 && comma_pos == std::string_view::npos)
                return std::nullopt;
            if (i == 2 && comma_pos != std::string_view::npos)
                return std::nullopt;

            std::string_view token = inner.substr(0, comma_pos);
            token = trim_view(token);

            int channel = 0;
            auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), channel);
            if (ec != std::errc() || ptr != token.data() + token.size() || channel < 0 || channel > 255)
                return std::nullopt;

            values[i] = static_cast<float>(channel) / 255.0f;
            if (i < 2) {
                inner.remove_prefix(comma_pos + 1);
            }
        }

        return values;
    }

    std::optional<std::array<float, 3>> parse_bg_color(const std::string& color) {
        const auto trimmed = trim_view(color);
        if (auto value = parse_bg_hex_color(trimmed))
            return value;
        if (auto value = parse_bg_rgb_color(trimmed))
            return value;
        return std::nullopt;
    }

    std::expected<std::vector<bool>, std::string> parse_add_splat_freeze_modifiers(
        const std::vector<std::string>& args) {
        std::vector<bool> freeze;
        for (size_t i = 1; i < args.size(); ++i) {
            const auto& arg = args[i];
            if (arg == "--add-splat") {
                if (i + 1 >= args.size()) {
                    return std::unexpected("--add-splat requires a path");
                }
                const bool frozen = (i + 2 < args.size() && args[i + 2] == "--freeze");
                freeze.push_back(frozen);
                i += frozen ? 2 : 1;
                continue;
            }
            if (arg.starts_with("--add-splat=")) {
                const bool frozen = (i + 1 < args.size() && args[i + 1] == "--freeze");
                freeze.push_back(frozen);
                if (frozen) {
                    ++i;
                }
                continue;
            }
            if (arg == "--freeze") {
                return std::unexpected("--freeze must immediately follow --add-splat <path>");
            }
        }
        return freeze;
    }

    // Parse log level from string
    lfs::core::LogLevel parse_log_level(const std::string& level_str) {
        if (level_str == "trace")
            return lfs::core::LogLevel::Trace;
        if (level_str == "debug")
            return lfs::core::LogLevel::Debug;
        if (level_str == "info")
            return lfs::core::LogLevel::Info;
        if (level_str == "perf" || level_str == "performance")
            return lfs::core::LogLevel::Performance;
        if (level_str == "warn" || level_str == "warning")
            return lfs::core::LogLevel::Warn;
        if (level_str == "error")
            return lfs::core::LogLevel::Error;
        if (level_str == "critical")
            return lfs::core::LogLevel::Critical;
        if (level_str == "off")
            return lfs::core::LogLevel::Off;
        return lfs::core::LogLevel::Info; // Default
    }

    std::expected<std::tuple<ParseResult, std::function<void()>>, std::string> parse_arguments(
        const std::vector<std::string>& args,
        lfs::core::param::TrainingParameters& params) {

        try {
            ::args::ArgumentParser parser(
                "LichtFeld Studio: High-performance CUDA implementation of 3D Gaussian Splatting algorithm.\n",
                "\nSUBCOMMANDS:\n"
                "convert -- Convert between .ply, .sog, .spz, .usd/.usda/.usdc, .html\n"
                "mesh2splat -- Convert a mesh file to Gaussian splats\n"
                "preprocess -- Generate depth and/or normal maps for an image dataset\n"
                "plugin -- Manage plugins (create, check, list)\n"
                "\n"
                "Run '<subcommand> --help' for details.\n"
                "\n"
                "EXAMPLES:\n"
                "lichtfeld-studio -d ./data -o ./output\n"
                "lichtfeld-studio --resume checkpoint.resume\n"
                "lichtfeld-studio --render-camera-path path.json --render-load model.ply --render-output out.mp4\n"
                "lichtfeld-studio -v model.ply\n"
                "lichtfeld-studio convert in.ply out.spz\n"
                "lichtfeld-studio mesh2splat model.obj -o model_splat.ply\n"
                "lichtfeld-studio preprocess ./data/scene --mode both\n"
                "lichtfeld-studio plugin create my_plugin\n"
                "\n"
                "ENVIRONMENT:\n"
                "LFS_LOG_LEVEL -- Set log level (trace/debug/info/perf/warn/error)\n");
            parser.helpParams.width = 240;

            // =============================================================================
            // MODE SELECTION
            // =============================================================================
            ::args::Group mode_group(parser, "MODE SELECTION:");
            ::args::HelpFlag help(mode_group, "help", "Display help menu", {'h', "help"});
            ::args::Flag version(mode_group, "version", "Display version information", {'V', "version"});
            ::args::ValueFlag<std::string> view_ply(mode_group, "path", "View file(s). Supports splat (.ply, .sog, .spz, .rad, .usd, .usda, .usdc, .usdz) and mesh (.obj, .fbx, .gltf, .glb, .stl) formats. If directory, loads all.", {'v', "view"});
            ::args::ValueFlag<std::string> resume_checkpoint(mode_group, "checkpoint", "Resume training from checkpoint file", {"resume"});
            ::args::ValueFlag<std::string> render_camera_path(mode_group, "path", "Render a JSON camera-keyframe path to video, headless (no GUI/window). Requires --render-load and --render-output; see RENDER PATH options.", {"render-camera-path"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // =============================================================================
            // TRAINING PATHS
            // =============================================================================
            ::args::Group paths_sep(parser, " ");
            ::args::Group paths_group(parser, "TRAINING PATHS:");
            ::args::ValueFlag<std::string> data_path(paths_group, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(paths_group, "output_path", "Path to output", {'o', "output-path"});
            ::args::ValueFlag<std::string> output_name(paths_group, "output_name", "Output filename (replaces default splat_ITER.ply stem)", {"output-name"});
            ::args::ValueFlag<std::string> config_file(paths_group, "config_file", "LichtFeldStudio config file (json)", {"config"});
            ::args::ValueFlag<std::string> init_path(paths_group, "path", "Initialize from splat file (.ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume)", {"init"});
            ::args::ValueFlagList<std::string> add_splats(paths_group, "path", "Append trained splat file(s) to the training model before optimizer initialization", {"add-splat"});
            ::args::CounterFlag freeze(paths_group, "freeze", "Freeze the immediately preceding --add-splat rows from optimizer gradients and densification", {"freeze"});
            ::args::Flag exclude_export(paths_group, "exclude_export", "Exclude frozen --add-splat rows from PLY exports", {"exclude-export"});

            ::args::ValueFlag<std::string> import_cameras(paths_group, "path", "Import COLMAP cameras from sparse folder (no images required)", {"import-cameras"});

            // =============================================================================
            // RENDER PATH (used with --render-camera-path)
            // =============================================================================
            ::args::Group render_path_sep(parser, " ");
            ::args::Group render_path_group(parser, "RENDER PATH (used with --render-camera-path):");
            ::args::ValueFlag<std::string> render_load(render_path_group, "path", "Trained scene to render (.ply/.sog/.spz or .resume checkpoint)", {"render-load"});
            ::args::ValueFlag<std::string> render_output(render_path_group, "path", "Output video file (.mp4)", {"render-output"});
            ::args::ValueFlag<int> render_width(render_path_group, "width", "Output width (default 1920)", {"render-width"});
            ::args::ValueFlag<int> render_height(render_path_group, "height", "Output height (default 1080)", {"render-height"});
            ::args::ValueFlag<int> render_fps(render_path_group, "fps", "Output framerate (default 30)", {"render-fps"});
            ::args::ValueFlag<int> render_crf(render_path_group, "crf", "Video quality, lower=better (default 18)", {"render-crf"});

            // =============================================================================
            // TRAINING PARAMETERS
            // =============================================================================
            ::args::Group training_sep(parser, " ");
            ::args::Group training_group(parser, "TRAINING PARAMETERS:");
            ::args::ValueFlag<uint32_t> iterations(training_group, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<std::string> strategy(training_group, "strategy", "Optimization strategy: mcmc, mrnf, igs+ (legacy aliases: mnrf, lfs)", {"strategy"});
            ::args::ValueFlag<int> sh_degree(training_group, "sh_degree", "Max SH degree [0-3]", {"sh-degree"});
            ::args::ValueFlag<int> sh_degree_interval(training_group, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> max_cap(training_group, "max_cap", "Maximum number of Gaussians", {"max-cap"});
            ::args::ValueFlag<float> min_opacity(training_group, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<float> steps_scaler(training_group, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::Flag no_error_map(training_group, "no_error_map", "Disable per-pixel SSIM error-map weighting of the MRNF refine signal (default: enabled)", {"no-error-map"});
            ::args::Flag no_edge_map(training_group, "no_edge_map", "Disable Sobel edge-map weighting of the MRNF refine signal (default: enabled)", {"no-edge-map"});
            ::args::ValueFlag<std::string> bg_mode(training_group, "mode", "Background mode: solidcolor, modulation, image, random (default: solidcolor)", {"bg-mode"});
            ::args::ValueFlag<std::string> bg_color(training_group, "color", "solidcolor background color as #RRGGBB or (R,G,B) with 0-255 channels (default: #000000)", {"bg-color"});
            ::args::ValueFlag<std::string> bg_image_path(training_group, "path", "Background image path (required when --bg-mode image)", {"bg-image-path"});

            // =============================================================================
            // INITIALIZATION
            // =============================================================================
            ::args::Group init_sep(parser, " ");
            ::args::Group init_group(parser, "INITIALIZATION:");
            ::args::Flag random(init_group, "random", "Use random initialization instead of SfM", {"random"});
            ::args::ValueFlag<int> init_num_pts(init_group, "init_num_pts", "Number of random initialization points", {"init-num-pts"});
            ::args::ValueFlag<float> init_extent(init_group, "init_extent", "Extent of random initialization", {"init-extent"});

            // =============================================================================
            // DATASET OPTIONS
            // =============================================================================
            ::args::Group dataset_sep(parser, " ");
            ::args::Group dataset_group(parser, "DATASET OPTIONS:");
            ::args::ValueFlag<std::string> images_folder(dataset_group, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(dataset_group, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::MapFlag<std::string, int> resize_factor(dataset_group, "resize_factor",
                                                            "Resize resolution by factor: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});
            ::args::ValueFlag<int> max_width(dataset_group, "max_width", "Max width of images in px; 0 disables the cap (default: 3840)", {"max-width"});
            ::args::ValueFlag<int> min_track_length(dataset_group, "min_track_length", "Minimum point track length for COLMAP sparse point import; 0 disables filtering", {"min-track-length"});
            ::args::Flag no_cpu_cache(dataset_group, "no_cpu_cache", "Disable CPU memory caching (default: enabled)", {"no-cpu-cache"});
            ::args::Flag no_fs_cache(dataset_group, "no_fs_cache", "Disable filesystem caching (default: enabled)", {"no-fs-cache"});
            ::args::Flag use_16bit(dataset_group, "use_16bit", "Train with 16-bit color images (HDR); caches losslessly as JPEG 2000 (default: 8-bit)", {"use-16bit"});
            ::args::Flag undistort(dataset_group, "undistort", "Undistort images on-the-fly before training", {"undistort"});
            ::args::MapFlag<std::string, std::string> centralize(dataset_group, "mode",
                                                                 "Centralize dataset origin: off, by_pointcloud, by_cameras (default: off)",
                                                                 {"centralize"},
                                                                 std::unordered_map<std::string, std::string>{
                                                                     {"off", "off"},
                                                                     {"by_pointcloud", "by_pointcloud"},
                                                                     {"by_cameras", "by_cameras"}});

            // =============================================================================
            // MASK / DEPTH / NORMAL OPTIONS
            // =============================================================================
            ::args::Group mask_sep(parser, " ");
            ::args::Group mask_group(parser, "MASK / DEPTH / NORMAL OPTIONS:");
            ::args::MapFlag<std::string, lfs::core::param::MaskMode> mask_mode(mask_group, "mask_mode",
                                                                               "Mask mode: none, segment, ignore, segment_and_ignore, alpha_consistent (default: none)",
                                                                               {"mask-mode"},
                                                                               std::unordered_map<std::string, lfs::core::param::MaskMode>{
                                                                                   {"none", lfs::core::param::MaskMode::None},
                                                                                   {"segment", lfs::core::param::MaskMode::Segment},
                                                                                   {"ignore", lfs::core::param::MaskMode::Ignore},
                                                                                   {"segment_and_ignore", lfs::core::param::MaskMode::SegmentAndIgnore},
                                                                                   {"alpha_consistent", lfs::core::param::MaskMode::AlphaConsistent}});
            ::args::Flag invert_masks(mask_group, "invert_masks", "Invert mask values (swap object/background)", {"invert-masks"});
            ::args::Flag no_alpha_as_mask(mask_group, "no_alpha_as_mask", "Disable automatic alpha-as-mask for RGBA images", {"no-alpha-as-mask"});
            ::args::Flag use_depth_loss(mask_group, "use_depth_loss", "Load depth maps and enable depth-map supervision", {"use-depth-loss"});
            ::args::ValueFlag<float> depth_loss_weight(mask_group, "depth_loss_weight", "Depth loss weight (default: 2.0)", {"depth-loss-weight"});
            ::args::ValueFlag<std::string> depth_loss_mode(mask_group, "depth_loss_mode", "Depth prior convention: ssi (auto-detect), ssi-disparity, ssi-depth (default: ssi)", {"depth-loss-mode"});
            ::args::Flag use_normal_loss(mask_group, "use_normal_loss", "Load normal maps and enable normal supervision", {"use-normal-loss"});
            ::args::ValueFlag<float> normal_loss_weight(mask_group, "normal_loss_weight", "Prior normal loss weight (default: 0.05)", {"normal-loss-weight"});
            ::args::ValueFlag<float> normal_consistency_weight(mask_group, "normal_consistency_weight", "Depth-normal consistency weight (default: 0.05)", {"normal-consistency-weight"});
            ::args::ValueFlag<float> normal_flatten_weight(mask_group, "normal_flatten_weight", "Min-axis scale flattening weight while normal supervision is active (default: 1.0)", {"normal-flatten-weight"});
            ::args::ValueFlag<std::string> normal_loss_space(mask_group, "normal_loss_space", "Normal prior space: auto, camera-opencv, camera-opengl, world (default: auto)", {"normal-loss-space"});

            // =============================================================================
            // SPARSITY OPTIMIZATION
            // =============================================================================
            ::args::Group sparsity_sep(parser, " ");
            ::args::Group sparsity_group(parser, "SPARSITY OPTIMIZATION:");
            ::args::Flag enable_sparsity(sparsity_group, "enable_sparsity", "Enable sparsity optimization", {"enable-sparsity"});
            ::args::ValueFlag<int> sparsify_steps(sparsity_group, "sparsify_steps", "Number of sparsification steps to run after regular training (default: 15000)", {"sparsify-steps"});
            ::args::ValueFlag<float> init_rho(sparsity_group, "init_rho", "Initial ADMM penalty parameter (default: 0.0005)", {"init-rho"});
            ::args::ValueFlag<float> prune_ratio(sparsity_group, "prune_ratio", "Final pruning ratio for sparsity (default: 0.6)", {"prune-ratio"});

            // =============================================================================
            // RENDERING OPTIONS
            // =============================================================================
            ::args::Group rendering_sep(parser, " ");
            ::args::Group rendering_group(parser, "RENDERING OPTIONS:");
            ::args::Flag enable_mip(rendering_group, "enable_mip", "Enable mip filter (anti-aliasing)", {"enable-mip"});
            ::args::Flag use_bilateral_grid(rendering_group, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag use_ppisp(rendering_group, "ppisp", "Enable PPISP for per-camera appearance modeling", {"ppisp"});
            ::args::Flag ppisp_controller(rendering_group, "ppisp_controller", "Enable PPISP controller for novel views", {"ppisp-controller"});
            ::args::Flag ppisp_freeze_from_sidecar(rendering_group, "ppisp_freeze", "Freeze PPISP learning and load PPISP weights from a sidecar file", {"ppisp-freeze"});
            ::args::ValueFlag<std::string> ppisp_sidecar_path(rendering_group, "path", "Path to PPISP sidecar (.ppisp) used for frozen PPISP training", {"ppisp-sidecar"});
            ::args::Flag gut(rendering_group, "gut", "Enable GUT mode", {"gut"});

            // =============================================================================
            // OUTPUT OPTIONS
            // =============================================================================
            ::args::Group output_sep(parser, " ");
            ::args::Group output_group(parser, "OUTPUT OPTIONS:");
            ::args::Flag enable_eval(output_group, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag no_save_eval_images(output_group, "no_save_eval_images", "Disable saving of evaluation comparison images (GT vs rendered) during eval (default: enabled)", {"no-save-eval-images"});
            ::args::ValueFlagList<std::string> timelapse_images(output_group, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});
            ::args::ValueFlag<int> timelapse_every(output_group, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});

            // =============================================================================
            // UI OPTIONS
            // =============================================================================
            ::args::Group ui_sep(parser, " ");
            ::args::Group ui_group(parser, "UI OPTIONS:");
            ::args::Flag headless(ui_group, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag auto_train(ui_group, "train", "Start training immediately on startup", {"train"});
#ifndef LFS_BUILD_PORTABLE
            ::args::Flag no_splash(ui_group, "no_splash", "Skip splash screen on startup", {"no-splash"});
#endif
            ::args::Flag debug_python(ui_group, "debug_python", "Start debugpy listener on port 5678 for plugin debugging", {"debug-python"});
            ::args::ValueFlag<int> debug_python_port(ui_group, "port", "Port for debugpy listener (default: 5678)", {"debug-python-port"});

            // =============================================================================
            // LOGGING
            // =============================================================================
            ::args::Group logging_sep(parser, " ");
            ::args::Group logging_group(parser, "LOGGING:");
            ::args::ValueFlag<std::string> log_level(logging_group, "level", "Log level: trace, debug, info, perf, warn, error, critical, off (default: info)", {"log-level"});
            ::args::Flag verbose(logging_group, "verbose", "Verbose output (equivalent to --log-level debug)", {"verbose"});
            ::args::Flag quiet(logging_group, "quiet", "Suppress non-error output (equivalent to --log-level error)", {'q', "quiet"});
            ::args::ValueFlag<std::string> log_file(logging_group, "file", "Optional log file path", {"log-file"});
            ::args::ValueFlag<std::string> log_filter(logging_group, "pattern", "Filter log messages (glob: *foo*, regex: \\\\d+)", {"log-filter"});
            ::args::Flag tcp_connection(parser, "tcp_connection", "Use TCP connection for signals and events", {"tcp-connection"});
            ::args::ValueFlag<int> tcp_server_connection_port(parser, "tcp_server_connection_port", "TCP connection port when tcp connection is in use for server requests, -1 for auto", {"tcp-server-port"});
            ::args::ValueFlag<int> tcp_broadcast_connection_port(parser, "tcp_broadcast_connection_port", "TCP connection port when tcp connection is in use for broadcasting, -1 for auto", {"tcp-broadcast-port"});

            // =============================================================================
            // EXTENSIONS
            // =============================================================================
            ::args::Group extensions_sep(parser, " ");
            ::args::Group extensions_group(parser, "EXTENSIONS:");
            ::args::ValueFlagList<std::string> python_scripts(extensions_group, "path", "Python script(s) for custom training callbacks", {"python-script"});

            // Parse arguments
            try {
                parser.Prog(args.front());
                parser.ParseArgs(std::vector<std::string>(args.begin() + 1, args.end()));
            } catch (const ::args::Help&) {
                std::print("{}", parser.Help());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::Completion& e) {
                std::print("{}", e.what());
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            } catch (const ::args::ParseError& e) {
                return std::unexpected(std::format("Parse error: {}\n{}", e.what(), parser.Help()));
            }

            // Initialize logger (CLI args override environment variable)
            {
                auto level = lfs::core::LogLevel::Info;
                std::string log_file_path;
                std::string filter_pattern;

                // Check environment variable first
                if (const auto env_level = lfs::core::environment::value("LFS_LOG_LEVEL")) {
                    level = parse_log_level(std::string(*env_level));
                }
                // Verbose/quiet flags override environment variable
                if (verbose) {
                    level = lfs::core::LogLevel::Debug;
                }
                if (quiet) {
                    level = lfs::core::LogLevel::Error;
                }
                // CLI --log-level takes final precedence
                if (log_level) {
                    level = parse_log_level(::args::get(log_level));
                }
                if (log_file) {
                    log_file_path = ::args::get(log_file);
                }
                if (log_filter) {
                    filter_pattern = ::args::get(log_filter);
                }

                lfs::core::Logger::get().init(level, log_file_path, filter_pattern);

                LOG_DEBUG("Logger initialized with level: {}", static_cast<int>(level));
                if (!filter_pattern.empty()) {
                    LOG_DEBUG("Log filter: {}", filter_pattern);
                }
                if (!log_file_path.empty()) {
                    LOG_DEBUG("Logging to file: {}", log_file_path);
                }
            }

            // Check if explicitly displaying help
            if (help) {
                return std::make_tuple(ParseResult::Help, std::function<void()>{});
            }

            // NO ARGUMENTS = VIEWER MODE (empty)
            if (args.size() == 1) {
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Viewer mode: file or directory
            if (view_ply) {
                const auto& view_path_str = ::args::get(view_ply);
                if (!view_path_str.empty()) {
                    const std::filesystem::path view_path = lfs::core::utf8_to_path(view_path_str);

                    if (!std::filesystem::exists(view_path)) {
                        return std::unexpected(std::format("Path does not exist: {}", lfs::core::path_to_utf8(view_path)));
                    }

                    constexpr std::array<std::string_view, 13> SUPPORTED_EXTENSIONS = {
                        ".ply", ".sog", ".spz", ".rad", ".resume",
                        ".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".blend"};
                    const auto is_supported = [&](const std::filesystem::path& p) {
                        auto ext = p.extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        return std::ranges::find(SUPPORTED_EXTENSIONS, ext) != SUPPORTED_EXTENSIONS.end();
                    };

                    if (std::filesystem::is_directory(view_path)) {
                        for (const auto& entry : std::filesystem::directory_iterator(view_path)) {
                            if (entry.is_regular_file() && is_supported(entry.path())) {
                                params.view_paths.push_back(entry.path());
                            }
                        }
                        std::ranges::sort(params.view_paths);

                        if (params.view_paths.empty()) {
                            return std::unexpected(std::format(
                                "No supported files found in: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        LOG_DEBUG("Found {} view files in directory", params.view_paths.size());
                    } else {
                        if (!is_supported(view_path)) {
                            return std::unexpected(std::format(
                                "Unsupported file format: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        params.view_paths.push_back(view_path);
                    }
                }

                if (gut) {
                    params.optimization.gut = true;
                }
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Headless camera-path -> video render mode: no training, no window.
            if (render_camera_path) {
                const auto& camera_path_str = ::args::get(render_camera_path);
                const auto camera_path = lfs::core::utf8_to_path(camera_path_str);
                if (!std::filesystem::exists(camera_path)) {
                    return std::unexpected(std::format("Camera path file does not exist: {}", camera_path_str));
                }
                if (!render_load || ::args::get(render_load).empty()) {
                    return std::unexpected(std::format(
                        "ERROR: --render-camera-path requires --render-load\n\n{}", parser.Help()));
                }
                if (!render_output || ::args::get(render_output).empty()) {
                    return std::unexpected(std::format(
                        "ERROR: --render-camera-path requires --render-output\n\n{}", parser.Help()));
                }
                const auto load_path = lfs::core::utf8_to_path(::args::get(render_load));
                if (!std::filesystem::exists(load_path)) {
                    return std::unexpected(std::format("Scene to render does not exist: {}", ::args::get(render_load)));
                }

                lfs::core::param::RenderPathConfig cfg;
                cfg.camera_path = camera_path;
                cfg.load_path = load_path;
                cfg.output_path = lfs::core::utf8_to_path(::args::get(render_output));
                if (render_width) {
                    cfg.width = ::args::get(render_width);
                }
                if (render_height) {
                    cfg.height = ::args::get(render_height);
                }
                if (render_fps) {
                    cfg.fps = ::args::get(render_fps);
                }
                if (render_crf) {
                    cfg.crf = ::args::get(render_crf);
                }
                params.render_path = cfg;

                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Import COLMAP cameras only (no images required)
            if (import_cameras) {
                const auto& import_path_str = ::args::get(import_cameras);
                if (!import_path_str.empty()) {
                    const std::filesystem::path import_path = lfs::core::utf8_to_path(import_path_str);
                    if (!std::filesystem::exists(import_path)) {
                        return std::unexpected(std::format("Path does not exist: {}", lfs::core::path_to_utf8(import_path)));
                    }
                    if (!std::filesystem::is_directory(import_path)) {
                        return std::unexpected(std::format("Expected directory for --import-cameras: {}", lfs::core::path_to_utf8(import_path)));
                    }
                    params.import_cameras_path = import_path;
                }
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Check for resume mode
            if (resume_checkpoint) {
                const auto ckpt_path_str = ::args::get(resume_checkpoint);
                if (!ckpt_path_str.empty()) {
                    const auto ckpt_path = lfs::core::utf8_to_path(ckpt_path_str);
                    if (!std::filesystem::exists(ckpt_path)) {
                        return std::unexpected(std::format("Checkpoint file does not exist: {}", ckpt_path_str));
                    }
                    params.resume_checkpoint = ckpt_path;
                }
            }

            if (init_path) {
                const auto path_str = ::args::get(init_path);
                params.init_path = path_str;

                if (!std::filesystem::exists(lfs::core::utf8_to_path(path_str))) {
                    return std::unexpected(std::format("Initialization file does not exist: {}", path_str));
                }
            }

            auto add_splat_freeze = parse_add_splat_freeze_modifiers(args);
            if (!add_splat_freeze) {
                return std::unexpected(add_splat_freeze.error());
            }
            if (add_splats) {
                const auto add_splat_values = ::args::get(add_splats);
                if (add_splat_freeze->size() != add_splat_values.size()) {
                    return std::unexpected("--add-splat parser metadata is inconsistent");
                }
                for (size_t i = 0; i < add_splat_values.size(); ++i) {
                    const auto& path_str = add_splat_values[i];
                    const auto splat_path = lfs::core::utf8_to_path(path_str);
                    if (!std::filesystem::exists(splat_path)) {
                        return std::unexpected(std::format("Added splat does not exist: {}", path_str));
                    }
                    params.add_splat_paths.push_back(splat_path);
                    params.add_splat_freeze.push_back((*add_splat_freeze)[i]);
                }
            }

            // Training mode
            const bool has_data_path = data_path && !::args::get(data_path).empty();
            const bool has_output_path = output_path && !::args::get(output_path).empty();
            const bool has_resume = params.resume_checkpoint.has_value();

            // If headless mode, require data path or resume checkpoint
            if (headless && !has_data_path && !has_resume) {
                return std::unexpected(std::format(
                    "ERROR: Headless mode requires --data-path or --resume\n\n{}",
                    parser.Help()));
            }

            if (tcp_connection && !headless) {
                return std::unexpected(std::format(
                    "ERROR: TCP connection mode requires --headless\n\n{}",
                    parser.Help()));
            }

            // Training/resume mode requires both data-path and output-path
            // Exception: resume mode can work without explicit paths (extracted from checkpoint)
            if (has_data_path && has_output_path) {
                params.dataset.data_path = lfs::core::utf8_to_path(::args::get(data_path));
                params.dataset.output_path = lfs::core::utf8_to_path(::args::get(output_path));

                // Create output directory
                std::error_code ec;
                std::filesystem::create_directories(params.dataset.output_path, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        lfs::core::path_to_utf8(params.dataset.output_path), ec.message()));
                }
            } else if (has_data_path != has_output_path && !has_resume) {
                // Only require both if not in resume mode
                return std::unexpected(std::format(
                    "ERROR: Training mode requires both --data-path and --output-path\n\n{}",
                    parser.Help()));
            } else if (has_resume) {
                // Resume mode: paths are optional (will be read from checkpoint)
                if (has_data_path) {
                    params.dataset.data_path = lfs::core::utf8_to_path(::args::get(data_path));
                }
                if (has_output_path) {
                    params.dataset.output_path = lfs::core::utf8_to_path(::args::get(output_path));

                    // Create output directory if provided
                    std::error_code ec;
                    std::filesystem::create_directories(params.dataset.output_path, ec);
                    if (ec) {
                        return std::unexpected(std::format(
                            "Failed to create output directory '{}': {}",
                            lfs::core::path_to_utf8(params.dataset.output_path), ec.message()));
                    }
                }
            }

            if (strategy) {
                const auto strat = ::args::get(strategy);
                if (VALID_STRATEGIES.find(strat) == VALID_STRATEGIES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, mrnf, igs+ (legacy aliases: mnrf, lfs)",
                        strat));
                }

                // Unlike other parameters that will be set later as overrides,
                // strategy must be set immediately to ensure correct JSON loading
                // in `read_optim_params_from_json()`
                params.optimization.strategy = std::string(lfs::core::param::canonical_strategy_name(strat));
            }

            if (config_file) {
                params.optimization.config_file = ::args::get(config_file);
                if (!strategy) {
                    params.optimization.strategy = ""; // Clear strategy to avoid using default strategy for evaluation of conflict
                }
            }

            if (max_width) {
                int width = ::args::get(max_width);
                if (width < 0) {
                    return std::unexpected("ERROR: --max-width must be 0 or greater");
                }
            }
            if (min_track_length) {
                int min_track = ::args::get(min_track_length);
                if (min_track < 0) {
                    return std::unexpected("ERROR: --min-track-length must be 0 or greater");
                }
            }

            // Validate sh_degree (0-3)
            if (sh_degree) {
                int degree = ::args::get(sh_degree);
                if (degree < 0 || degree > 3) {
                    return std::unexpected("ERROR: --sh-degree must be 0, 1, 2, or 3");
                }
            }

            // Validate min_opacity (0.0-1.0)
            if (min_opacity) {
                float opacity = ::args::get(min_opacity);
                if (opacity < 0.0f || opacity > 1.0f) {
                    return std::unexpected("ERROR: --min-opacity must be between 0.0 and 1.0");
                }
            }

            // Validate init_num_pts (> 0)
            if (init_num_pts) {
                int pts = ::args::get(init_num_pts);
                if (pts <= 0) {
                    return std::unexpected("ERROR: --init-num-pts must be greater than 0");
                }
            }

            // Validate prune_ratio (0.0-1.0)
            if (prune_ratio) {
                float ratio = ::args::get(prune_ratio);
                if (ratio < 0.0f || ratio > 1.0f) {
                    return std::unexpected("ERROR: --prune-ratio must be between 0.0 and 1.0");
                }
            }

            std::optional<lfs::core::param::BackgroundMode> parsed_bg_mode;
            if (bg_mode) {
                parsed_bg_mode = parse_bg_mode(::args::get(bg_mode));
                if (!parsed_bg_mode) {
                    return std::unexpected("ERROR: --bg-mode must be one of solidcolor, modulation, image, or random");
                }
            }

            std::optional<std::array<float, 3>> parsed_bg_color;
            if (bg_color) {
                parsed_bg_color = parse_bg_color(::args::get(bg_color));
                if (!parsed_bg_color) {
                    return std::unexpected("ERROR: --bg-color must be #RRGGBB or (R,G,B) with 0-255 channels");
                }
            }

            if (parsed_bg_mode == lfs::core::param::BackgroundMode::Image &&
                (!bg_image_path || ::args::get(bg_image_path).empty())) {
                return std::unexpected("ERROR: --bg-image-path is required when --bg-mode image");
            }

            const auto cli_option_present = [&args](const std::initializer_list<std::string_view> names) {
                for (size_t i = 1; i < args.size(); ++i) {
                    const std::string_view arg = args[i];
                    for (const std::string_view name : names) {
                        if (arg == name) {
                            return true;
                        }
                        if (name.starts_with("--") &&
                            arg.size() > name.size() &&
                            arg.starts_with(name) &&
                            arg[name.size()] == '=') {
                            return true;
                        }
                    }
                }
                return false;
            };

            // Create lambda to apply command line overrides after JSON loading
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = cli_option_present({"-i", "--iter"}) ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // default 1
                                        max_width_val = max_width ? std::optional<int>(::args::get(max_width)) : std::optional<int>(3840),
                                        min_track_length_val = cli_option_present({"--min-track-length"}) ? std::optional<int>(::args::get(min_track_length)) : std::optional<int>(),
                                        no_cpu_cache_flag = static_cast<bool>(no_cpu_cache),
                                        no_fs_cache_flag = static_cast<bool>(no_fs_cache),
                                        use_16bit_flag = static_cast<bool>(use_16bit),
                                        tcp_server_connection_port_val = tcp_server_connection_port ? std::optional<int>(::args::get(tcp_server_connection_port)) : std::optional<int>(),
                                        tcp_broadcast_connection_port_val = tcp_broadcast_connection_port ? std::optional<int>(::args::get(tcp_broadcast_connection_port)) : std::optional<int>(),
                                        tcp_connection_flag = bool(tcp_connection),
                                        max_cap_val = cli_option_present({"--max-cap"}) ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        config_file_val = cli_option_present({"--config"}) ? std::optional<std::string>(::args::get(config_file)) : std::optional<std::string>(),
                                        images_folder_val = cli_option_present({"--images"}) ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = cli_option_present({"--test-every"}) ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = cli_option_present({"--steps-scaler"}) ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = cli_option_present({"--sh-degree-interval"}) ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = cli_option_present({"--sh-degree"}) ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = cli_option_present({"--min-opacity"}) ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        init_num_pts_val = cli_option_present({"--init-num-pts"}) ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = cli_option_present({"--init-extent"}) ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        strategy_val = cli_option_present({"--strategy"}) ? std::optional<std::string>(::args::get(strategy)) : std::optional<std::string>(),
                                        timelapse_images_val = cli_option_present({"--timelapse-images"}) ? std::optional<std::vector<std::string>>(::args::get(timelapse_images)) : std::optional<std::vector<std::string>>(),
                                        timelapse_every_val = cli_option_present({"--timelapse-every"}) ? std::optional<int>(::args::get(timelapse_every)) : std::optional<int>(),
                                        // Sparsity parameters
                                        sparsify_steps_val = cli_option_present({"--sparsify-steps"}) ? std::optional<int>(::args::get(sparsify_steps)) : std::optional<int>(),
                                        init_rho_val = cli_option_present({"--init-rho"}) ? std::optional<float>(::args::get(init_rho)) : std::optional<float>(),
                                        prune_ratio_val = cli_option_present({"--prune-ratio"}) ? std::optional<float>(::args::get(prune_ratio)) : std::optional<float>(),
                                        // Mask parameters
                                        mask_mode_val = cli_option_present({"--mask-mode"}) ? std::optional<lfs::core::param::MaskMode>(::args::get(mask_mode)) : std::optional<lfs::core::param::MaskMode>(),
                                        depth_loss_weight_val = cli_option_present({"--depth-loss-weight"}) ? std::optional<float>(::args::get(depth_loss_weight)) : std::optional<float>(),
                                        depth_loss_mode_val = cli_option_present({"--depth-loss-mode"}) ? std::optional<std::string>(::args::get(depth_loss_mode)) : std::optional<std::string>(),
                                        normal_loss_weight_val = cli_option_present({"--normal-loss-weight"}) ? std::optional<float>(::args::get(normal_loss_weight)) : std::optional<float>(),
                                        normal_consistency_weight_val = cli_option_present({"--normal-consistency-weight"}) ? std::optional<float>(::args::get(normal_consistency_weight)) : std::optional<float>(),
                                        normal_flatten_weight_val = cli_option_present({"--normal-flatten-weight"}) ? std::optional<float>(::args::get(normal_flatten_weight)) : std::optional<float>(),
                                        normal_loss_space_val = cli_option_present({"--normal-loss-space"}) ? std::optional<std::string>(::args::get(normal_loss_space)) : std::optional<std::string>(),
                                        // Python scripts
                                        python_scripts_val = cli_option_present({"--python-script"}) ? std::optional<std::vector<std::string>>(::args::get(python_scripts)) : std::optional<std::vector<std::string>>(),
                                        centralize_val = cli_option_present({"--centralize"}) ? std::optional<std::string>(::args::get(centralize)) : std::optional<std::string>(),
                                        // Capture flag states
                                        enable_mip_flag = bool(enable_mip),
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        use_ppisp_flag = bool(use_ppisp),
                                        ppisp_controller_flag = bool(ppisp_controller),
                                        ppisp_freeze_from_sidecar_flag = bool(ppisp_freeze_from_sidecar),
                                        ppisp_sidecar_path_val = cli_option_present({"--ppisp-sidecar"}) ? std::optional<std::string>(::args::get(ppisp_sidecar_path)) : std::optional<std::string>(),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        auto_train_flag = bool(auto_train),
#ifdef LFS_BUILD_PORTABLE
                                        no_splash_flag = false,
#else
                                        no_splash_flag = bool(no_splash),
#endif
                                        debug_python_flag = bool(debug_python),
                                        debug_python_port_val = cli_option_present({"--debug-python-port"}) ? std::optional<int>(::args::get(debug_python_port)) : std::optional<int>(),
                                        no_save_eval_images_flag = bool(no_save_eval_images),
                                        bg_mode_val = parsed_bg_mode,
                                        bg_color_val = parsed_bg_color,
                                        bg_image_path_val = cli_option_present({"--bg-image-path"}) ? std::optional<std::string>(::args::get(bg_image_path)) : std::optional<std::string>(),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut),
                                        undistort_flag = bool(undistort),
                                        enable_sparsity_flag = bool(enable_sparsity),
                                        invert_masks_flag = bool(invert_masks),
                                        no_alpha_as_mask_flag = bool(no_alpha_as_mask),
                                        use_depth_loss_flag = bool(use_depth_loss),
                                        use_normal_loss_flag = bool(use_normal_loss),
                                        no_error_map_flag = bool(no_error_map),
                                        no_edge_map_flag = bool(no_edge_map),
                                        exclude_export_flag = bool(exclude_export),
                                        output_name_val = cli_option_present({"--output-name"}) ? std::optional<std::string>(::args::get(output_name)) : std::optional<std::string>()]() {
                auto& opt = params.optimization;
                auto& svs = params.server;
                auto& ds = params.dataset;

                // Simple lambdas to apply if flag/value exists
                auto setVal = [](const auto& flag, auto& target) {
                    if (flag)
                        target = *flag;
                };

                auto setFlag = [](bool flag, auto& target) {
                    if (flag)
                        target = true;
                };

                // Apply all overrides
                setVal(iterations_val, opt.iterations);
                setVal(resize_factor_val, ds.resize_factor);
                setVal(max_width_val, ds.max_width);
                setVal(min_track_length_val, ds.min_track_length);
                if (no_cpu_cache_flag)
                    ds.loading_params.use_cpu_memory = false;
                if (no_fs_cache_flag)
                    ds.loading_params.use_fs_cache = false;
                setFlag(use_16bit_flag, ds.loading_params.use_16bit_color);
                setVal(max_cap_val, opt.max_cap);
                setVal(tcp_server_connection_port_val, svs.tcp_server_connection_port);
                setVal(tcp_broadcast_connection_port_val, svs.tcp_broadcast_connection_port);
                setFlag(tcp_connection_flag, svs.tcp_connection);
                setVal(images_folder_val, ds.images);
                setVal(test_every_val, ds.test_every);
                setVal(steps_scaler_val, opt.steps_scaler);
                setVal(sh_degree_interval_val, opt.sh_degree_interval);
                setVal(sh_degree_val, opt.sh_degree);
                setVal(min_opacity_val, opt.min_opacity);
                setVal(init_num_pts_val, opt.init_num_pts);
                setVal(init_extent_val, opt.init_extent);
                setVal(strategy_val, opt.strategy);
                setVal(timelapse_images_val, ds.timelapse_images);
                setVal(timelapse_every_val, ds.timelapse_every);
                setVal(output_name_val, ds.output_name);

                // Sparsity parameters
                setVal(sparsify_steps_val, opt.sparsify_steps);
                setVal(init_rho_val, opt.init_rho);
                setVal(prune_ratio_val, opt.prune_ratio);

                setFlag(enable_mip_flag, opt.mip_filter);
                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);
                setFlag(use_ppisp_flag, opt.use_ppisp);
                setFlag(ppisp_controller_flag, opt.ppisp_use_controller);
                setFlag(ppisp_freeze_from_sidecar_flag, opt.ppisp_freeze_from_sidecar);
                if (ppisp_sidecar_path_val) {
                    opt.ppisp_sidecar_path = lfs::core::utf8_to_path(*ppisp_sidecar_path_val);
                }
                if (opt.ppisp_use_controller)
                    opt.use_ppisp = true;
                if (opt.ppisp_freeze_from_sidecar)
                    opt.use_ppisp = true;
                setFlag(enable_eval_flag, opt.enable_eval);
                setFlag(headless_flag, opt.headless);
                setFlag(auto_train_flag, opt.auto_train);
                setFlag(no_splash_flag, opt.no_splash);
                setFlag(debug_python_flag, opt.debug_python);
                setVal(debug_python_port_val, opt.debug_python_port);
                if (no_save_eval_images_flag)
                    opt.enable_save_eval_images = false;
                if (bg_mode_val) {
                    opt.bg_mode = *bg_mode_val;
                    opt.bg_modulation = *bg_mode_val == lfs::core::param::BackgroundMode::Modulation;
                }
                setVal(bg_color_val, opt.bg_color);
                if (bg_color_val) {
                    params.cli_bg_color_set = true;
                }
                if (bg_image_path_val) {
                    opt.bg_image_path = lfs::core::utf8_to_path(*bg_image_path_val);
                }
                setFlag(random_flag, opt.random);
                setFlag(gut_flag, opt.gut);
                setFlag(undistort_flag, opt.undistort);
                setFlag(enable_sparsity_flag, opt.enable_sparsity);
                if (no_error_map_flag)
                    opt.use_error_map = false;
                if (no_edge_map_flag)
                    opt.use_edge_map = false;
                setFlag(exclude_export_flag, params.exclude_frozen_add_splats_from_export);

                // Mask parameters
                setVal(mask_mode_val, opt.mask_mode);
                setFlag(invert_masks_flag, opt.invert_masks);
                if (no_alpha_as_mask_flag)
                    opt.use_alpha_as_mask = false;
                setFlag(use_depth_loss_flag, opt.use_depth_loss);
                setVal(depth_loss_weight_val, opt.depth_loss_weight);
                if (depth_loss_mode_val) {
                    opt.depth_loss_mode = *depth_loss_mode_val;
                }
                setFlag(use_normal_loss_flag, opt.use_normal_loss);
                setVal(normal_loss_weight_val, opt.normal_loss_weight);
                setVal(normal_consistency_weight_val, opt.normal_consistency_weight);
                setVal(normal_flatten_weight_val, opt.normal_flatten_weight);
                setVal(normal_loss_space_val, opt.normal_loss_space);
                // Also propagate to dataset config for loading
                ds.invert_masks = opt.invert_masks;
                ds.mask_threshold = opt.mask_threshold;
                setVal(centralize_val, ds.centralize_dataset);

                // Python scripts
                if (python_scripts_val) {
                    for (const auto& script : *python_scripts_val) {
                        params.python_scripts.emplace_back(script);
                    }
                }
            };

            return std::make_tuple(ParseResult::Success, apply_cmd_overrides);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Unexpected error during argument parsing: {}", e.what()));
        }
    }

    void apply_step_scaling(lfs::core::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        opt.apply_step_scaling();
    }

    void apply_ppisp_defaults(lfs::core::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        if (!opt.ppisp_use_controller)
            return;

        if (opt.ppisp_controller_activation_step < 0) {
            opt.ppisp_controller_activation_step =
                opt.resolved_ppisp_controller_activation_step(opt.resolved_total_iterations());
        }
    }

    std::vector<std::string> convert_args(int argc, const char* const argv[]) {
        return std::vector<std::string>(argv, argv + argc);
    }
} // anonymous namespace

// Public interface
std::expected<std::unique_ptr<lfs::core::param::TrainingParameters>, std::string>
lfs::core::args::parse_args_and_params(int argc, const char* const argv[]) {

    auto params = std::make_unique<lfs::core::param::TrainingParameters>();
    auto args = convert_args(argc, argv);

    if (args.size() >= 2 && !args[1].starts_with('-') &&
        args[1] != "convert" && args[1] != "mesh2splat" &&
        args[1] != "mesh-to-splat" && args[1] != "preprocess" &&
        args[1] != "plugin") {
        const std::filesystem::path p = lfs::core::utf8_to_path(args[1]);
        std::error_code ec;
        if (std::filesystem::exists(p, ec))
            args.insert(args.begin() + 1, "-v");
    }

    auto parse_result = parse_arguments(args, *params);
    const std::string& strategy = params->optimization.strategy;
    const std::string& config_file = params->optimization.config_file;

    if (!parse_result) {
        return std::unexpected(parse_result.error());
    }

    const auto [result, apply_overrides] = *parse_result;
    if (result == ParseResult::Help) {
        std::exit(0);
    }

    // Load from --config or use hardcoded defaults
    if (!config_file.empty()) {
        const auto opt_result = lfs::core::param::read_optim_params_from_json(lfs::core::utf8_to_path(config_file));
        if (!opt_result) {
            return std::unexpected(std::format("Config load failed: {}", opt_result.error()));
        }
        params->optimization = *opt_result;

        if (!strategy.empty() &&
            !lfs::core::param::strategy_names_match(strategy, params->optimization.strategy)) {
            return std::unexpected("--strategy conflicts with config file");
        }
    } else {
        if (lfs::core::param::is_mrnf_strategy(strategy))
            params->optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
        else if (strategy == "igs+")
            params->optimization = lfs::core::param::OptimizationParameters::igs_plus_defaults();
        else if (strategy == "mcmc")
            params->optimization = lfs::core::param::OptimizationParameters::mcmc_defaults();
        else
            params->optimization = lfs::core::param::OptimizationParameters::mrnf_defaults();
    }

    params->dataset.loading_params = lfs::core::param::LoadingParams{};

    if (apply_overrides) {
        apply_overrides();
    }
    apply_step_scaling(*params);
    apply_ppisp_defaults(*params);

    if (auto error = params->validate(); !error.empty())
        return std::unexpected("ERROR: " + error);

    return params;
}

namespace {
    constexpr const char* CONVERT_HELP_HEADER = "LichtFeld Studio - Convert splat files between formats\n";
    constexpr const char* CONVERT_HELP_FOOTER =
        "\n"
        "EXAMPLES:\n"
        "  LichtFeld-Studio convert input.ply output.spz --sh-degree 0\n"
        "  LichtFeld-Studio convert input.ply -f html\n"
        "  LichtFeld-Studio convert ./splats/ -f sog --sh-degree 2\n"
        "\n"
        "SUPPORTED FORMATS:\n"
        "  Input:  .ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume (checkpoint)\n"
        "  Output: .ply, .sog, .spz, .usd, .usda, .usdc, .html, .rad\n"
        "\n";

    constexpr const char* MESH2SPLAT_HELP_HEADER = "LichtFeld Studio - Convert mesh files to Gaussian splats\n";
    constexpr const char* MESH2SPLAT_HELP_FOOTER =
        "\n"
        "EXAMPLES:\n"
        "  LichtFeld-Studio mesh2splat model.obj -o model_splat.ply\n"
        "  LichtFeld-Studio mesh2splat model.glb output.spz --resolution 1024 --sigma 0.65\n"
        "  LichtFeld-Studio mesh2splat model.glb -o ./splats/model -f ply,spz,html --overwrite\n"
        "  LichtFeld-Studio mesh2splat ./meshes/ -o ./splats/ -f ply,spz --overwrite\n"
        "\n"
        "SUPPORTED FORMATS:\n"
        "  Input:  .obj, .fbx, .gltf, .glb, .stl, .dae, .3ds, .ply\n"
        "  Output: .ply, .sog, .spz, .usd, .usda, .usdc, .html, .rad\n"
        "  Multiple output formats: pass a comma-separated list to --format\n"
        "\n";

    constexpr const char* PREPROCESS_HELP_HEADER = "LichtFeld Studio - Generate dataset depth and normal maps with MoGe-2 ONNX\n";
    constexpr const char* PREPROCESS_HELP_FOOTER =
        "\n"
        "EXAMPLES:\n"
        "  LichtFeld-Studio preprocess ./data/scene\n"
        "  LichtFeld-Studio preprocess ./data/scene --mode both --overwrite\n"
        "  LichtFeld-Studio preprocess ./data/scene --png-compression 0 --threads 8\n"
        "  LichtFeld-Studio preprocess --download-only\n"
        "\n"
        "OUTPUTS:\n"
        "  depth/<relative-image>.png   8-bit grayscale depth map\n"
        "  normals/<relative-image>.png RGB normal map encoded from [-1,1] to [0,255]\n"
        "  Existing outputs are skipped by default for resumable runs.\n"
        "  Use --overwrite to recreate existing outputs.\n"
        "  PNG compression defaults to level 1; use 0 for fastest/largest files.\n"
        "  The auto-downloaded default model is SHA-256 verified before use.\n"
        "\n";

    std::optional<lfs::core::param::OutputFormat> parseFormat(const std::string& str) {
        using lfs::core::param::OutputFormat;
        if (str == "ply" || str == ".ply")
            return OutputFormat::PLY;
        if (str == "sog" || str == ".sog")
            return OutputFormat::SOG;
        if (str == "spz" || str == ".spz")
            return OutputFormat::SPZ;
        if (str == "html" || str == ".html")
            return OutputFormat::HTML;
        if (str == "usd" || str == ".usd")
            return OutputFormat::USD;
        if (str == "usda" || str == ".usda")
            return OutputFormat::USDA;
        if (str == "usdc" || str == ".usdc")
            return OutputFormat::USDC;
        if (str == "rad" || str == ".rad")
            return OutputFormat::RAD;
        return std::nullopt;
    }

    std::expected<std::vector<lfs::core::param::OutputFormat>, std::string> parseFormatList(const std::string& formats_str) {
        std::vector<lfs::core::param::OutputFormat> formats;
        size_t start = 0;
        while (start < formats_str.size()) {
            size_t end = formats_str.find(',', start);
            if (end == std::string::npos)
                end = formats_str.size();
            std::string token = formats_str.substr(start, end - start);
            const size_t first = token.find_first_not_of(" \t");
            const size_t last = token.find_last_not_of(" \t");
            if (first != std::string::npos && last != std::string::npos) {
                token = token.substr(first, last - first + 1);
            }
            if (!token.empty()) {
                auto fmt = parseFormat(token);
                if (!fmt) {
                    return std::unexpected(std::format("Invalid format '{}'. Use: ply, sog, spz, html, usd, usda, usdc, rad", token));
                }
                if (std::ranges::find(formats, *fmt) == formats.end()) {
                    formats.push_back(*fmt);
                }
            }
            start = end + 1;
        }
        if (formats.empty()) {
            return std::unexpected("No output formats specified");
        }
        return formats;
    }

    std::expected<lfs::core::args::ParsedArgs, std::string> parseConvertArgs(const int argc, const char* const argv[]) {
        namespace core_args = lfs::core::args;
        namespace param = lfs::core::param;

        ::args::ArgumentParser parser(CONVERT_HELP_HEADER, CONVERT_HELP_FOOTER);
        ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
        ::args::Positional<std::string> input(parser, "input", "Input file or directory");
        ::args::Positional<std::string> output(parser, "output", "Output file (optional)");
        ::args::ValueFlag<int> sh_degree(parser, "degree", "SH degree [0-3], -1 to keep original (default: -1)", {"sh-degree"});
        ::args::ValueFlag<std::string> format(parser, "format", "Output format: ply, sog, spz, html, usd, usda, usdc, rad", {'f', "format"});
        ::args::ValueFlag<int> sog_iter(parser, "iterations", "K-means iterations for SOG (default: 10)", {"sog-iterations"});
        ::args::ValueFlag<std::string> tiles(parser, "AxB", "Replicate a PLY source across an AxB ground-plane grid (RAD output only)", {"tiles"});
        ::args::ValueFlag<std::string> lod_builder(parser, "builder", "PLY->RAD LOD tree builder: bhatt (default) or octree (hybrid: octree fine levels + similarity-ordered bhatt top, much faster)", {"lod-builder"});
        ::args::Flag rad_stream(parser, "stream", "RAD output: streamable Spark-compatible chunks (default)", {"stream"});
        ::args::Flag rad_none_stream(parser, "none-stream", "RAD output: native LichtFeld chunks", {"none-stream"});
        ::args::Flag overwrite(parser, "overwrite", "Overwrite existing files without prompting", {'y', "overwrite"});

        std::vector<std::string> args_vec(argv + 1, argv + argc);
        args_vec[0] = std::string(argv[0]) + " convert";
        parser.Prog(args_vec[0]);

        try {
            parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
        } catch (const ::args::Help&) {
            std::print("{}", parser.Help());
            return core_args::HelpMode{};
        } catch (const ::args::ParseError& e) {
            return std::unexpected(std::format("{}\n\n{}", e.what(), parser.Help()));
        }

        if (!input) {
            return std::unexpected(std::format("Missing input path\n\n{}", parser.Help()));
        }

        param::ConvertParameters params;
        params.input_path = lfs::core::utf8_to_path(::args::get(input));
        params.sh_degree = sh_degree ? ::args::get(sh_degree) : -1;

        if (!std::filesystem::exists(params.input_path)) {
            return std::unexpected(std::format("Input not found: {}", lfs::core::path_to_utf8(params.input_path)));
        }

        if (params.sh_degree < -1 || params.sh_degree > 3) {
            return std::unexpected("SH degree must be -1 (keep) or 0-3");
        }

        if (output)
            params.output_path = lfs::core::utf8_to_path(::args::get(output));
        if (sog_iter)
            params.sog_iterations = ::args::get(sog_iter);
        params.overwrite = overwrite;

        if (format) {
            if (const auto fmt = parseFormat(::args::get(format))) {
                params.format = *fmt;
            } else {
                return std::unexpected(std::format("Invalid format '{}'. Use: ply, sog, spz, html, usd, usda, usdc, rad", ::args::get(format)));
            }
        } else if (!params.output_path.empty()) {
            if (const auto fmt = parseFormat(params.output_path.extension().string())) {
                params.format = *fmt;
            } else {
                return std::unexpected(std::format("Unknown extension '{}'. Use --format", params.output_path.extension().string()));
            }
        }

        if (tiles) {
            const std::string& spec = ::args::get(tiles);
            const std::size_t sep = spec.find_first_of("xX");
            std::uint32_t a = 0;
            std::uint32_t b = 0;
            bool ok = sep != std::string::npos && sep > 0 && sep + 1 < spec.size();
            if (ok) {
                const auto [pa, ea] = std::from_chars(spec.data(), spec.data() + sep, a);
                const auto [pb, eb] = std::from_chars(spec.data() + sep + 1, spec.data() + spec.size(), b);
                ok = ea == std::errc{} && eb == std::errc{} &&
                     pa == spec.data() + sep && pb == spec.data() + spec.size() &&
                     a > 0 && b > 0;
            }
            if (!ok) {
                return std::unexpected(std::format("Invalid --tiles '{}'. Use AxB, e.g. 3x2", spec));
            }
            if (params.format != param::OutputFormat::RAD) {
                return std::unexpected("--tiles requires RAD output (--format rad)");
            }
            params.tiles_x = a;
            params.tiles_y = b;
        }

        if (lod_builder) {
            const std::string& name = ::args::get(lod_builder);
            if (name == "bhatt") {
                params.lod_builder = param::LodBuilder::BHATT;
            } else if (name == "octree") {
                params.lod_builder = param::LodBuilder::OCTREE;
            } else {
                return std::unexpected(std::format("Invalid --lod-builder '{}'. Use: bhatt, octree", name));
            }
            if (params.format != param::OutputFormat::RAD) {
                return std::unexpected("--lod-builder requires RAD output (--format rad)");
            }
        }

        if (rad_stream && rad_none_stream) {
            return std::unexpected("--stream and --none-stream are mutually exclusive");
        }
        if ((rad_stream || rad_none_stream) && params.format != param::OutputFormat::RAD) {
            return std::unexpected("--stream/--none-stream require RAD output (--format rad)");
        }
        params.rad_export_mode = rad_none_stream ? param::RadExportMode::NonStream
                                                 : param::RadExportMode::Stream;

        return core_args::ConvertMode{params};
    }

    std::expected<lfs::core::args::ParsedArgs, std::string> parseMesh2SplatArgs(const int argc, const char* const argv[]) {
        namespace core_args = lfs::core::args;
        namespace param = lfs::core::param;

        ::args::ArgumentParser parser(MESH2SPLAT_HELP_HEADER, MESH2SPLAT_HELP_FOOTER);
        ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
        ::args::Positional<std::string> input(parser, "input", "Input mesh file or directory");
        ::args::Positional<std::string> output(parser, "output", "Output file or directory (optional)");
        ::args::ValueFlag<std::string> output_flag(parser, "path", "Output file or directory", {'o', "output"});
        ::args::ValueFlag<std::string> format(parser, "formats", "Output format(s): ply, sog, spz, html, usd, usda, usdc, rad. Use commas for multiple outputs", {'f', "format"});
        ::args::ValueFlag<int> resolution(parser, "pixels", "Mesh2Splat raster resolution target (default: 1024)", {"resolution"});
        ::args::ValueFlag<float> sigma(parser, "scale", "Gaussian scale sigma (default: 0.65)", {"sigma"});
        ::args::ValueFlag<int> sog_iter(parser, "iterations", "K-means iterations for SOG/HTML output (default: 10)", {"sog-iterations"});
        ::args::Flag overwrite(parser, "overwrite", "Overwrite existing files without prompting", {'y', "overwrite"});

        std::vector<std::string> args_vec(argv + 1, argv + argc);
        args_vec[0] = std::string(argv[0]) + " mesh2splat";
        parser.Prog(args_vec[0]);

        try {
            parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
        } catch (const ::args::Help&) {
            std::print("{}", parser.Help());
            return core_args::HelpMode{};
        } catch (const ::args::ParseError& e) {
            return std::unexpected(std::format("{}\n\n{}", e.what(), parser.Help()));
        }

        if (!input) {
            return std::unexpected(std::format("Missing input mesh path\n\n{}", parser.Help()));
        }
        if (output && output_flag) {
            return std::unexpected("Use either positional output or --output, not both");
        }

        param::Mesh2SplatParameters params;
        params.input_path = lfs::core::utf8_to_path(::args::get(input));

        if (!std::filesystem::exists(params.input_path)) {
            return std::unexpected(std::format("Input not found: {}", lfs::core::path_to_utf8(params.input_path)));
        }

        if (output_flag) {
            params.output_path = lfs::core::utf8_to_path(::args::get(output_flag));
        } else if (output) {
            params.output_path = lfs::core::utf8_to_path(::args::get(output));
        }
        if (resolution)
            params.options.resolution_target = ::args::get(resolution);
        if (sigma)
            params.options.sigma = ::args::get(sigma);
        if (sog_iter)
            params.sog_iterations = ::args::get(sog_iter);
        params.overwrite = overwrite;

        if (params.options.resolution_target < lfs::core::Mesh2SplatOptions::kMinResolution) {
            return std::unexpected(std::format("Mesh2Splat resolution must be at least {}", lfs::core::Mesh2SplatOptions::kMinResolution));
        }
        if (params.options.sigma <= 0.0f) {
            return std::unexpected("Mesh2Splat sigma must be positive");
        }

        if (format) {
            auto formats = parseFormatList(::args::get(format));
            if (!formats)
                return std::unexpected(formats.error());
            params.formats = std::move(*formats);
            params.format = params.formats.front();
        } else if (!params.output_path.empty()) {
            if (const auto fmt = parseFormat(params.output_path.extension().string())) {
                params.format = *fmt;
                params.formats = {*fmt};
            } else if (!params.output_path.extension().empty() && !std::filesystem::is_directory(params.output_path)) {
                return std::unexpected(std::format("Unknown extension '{}'. Use --format", params.output_path.extension().string()));
            }
        }

        return core_args::Mesh2SplatMode{params};
    }

    std::expected<lfs::core::args::ParsedArgs, std::string> parsePreprocessArgs(const int argc, const char* const argv[]) {
        namespace core_args = lfs::core::args;
        namespace param = lfs::core::param;

        ::args::ArgumentParser parser(PREPROCESS_HELP_HEADER, PREPROCESS_HELP_FOOTER);
        ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
        ::args::Positional<std::string> input(parser, "dataset", "Dataset root containing the images folder");
        ::args::MapFlag<std::string, param::PreprocessOutputMode> mode(parser, "mode",
                                                                       "Output mode: depth, normals, both (default: both)",
                                                                       {"mode"},
                                                                       std::unordered_map<std::string, param::PreprocessOutputMode>{
                                                                           {"depth", param::PreprocessOutputMode::Depth},
                                                                           {"normal", param::PreprocessOutputMode::Normals},
                                                                           {"normals", param::PreprocessOutputMode::Normals},
                                                                           {"both", param::PreprocessOutputMode::Both}});
        ::args::ValueFlag<std::string> images(parser, "folder", "Images subfolder (default: images)", {"images"});
        ::args::ValueFlag<std::string> model(parser, "path", "Local ONNX model path; skips default cache download", {"model"});
        ::args::ValueFlag<int> max_side(parser, "pixels", "Inference longest side, rounded to /14 (default: 518; 0 disables resize)", {"max-side"});
        ::args::ValueFlag<std::int64_t> num_tokens(parser, "tokens", "MoGe dynamic-token input when present (default: 1800)", {"num-tokens"});
        ::args::ValueFlag<int> threads(parser, "count", "ONNX Runtime CPU threads (default: all available cores)", {"threads"});
        ::args::ValueFlag<int> png_compression(parser, "level", "PNG compression level 0-9 (default: 1; 0 is fastest/largest)", {"png-compression"});
        ::args::ValueFlag<int> bit_depth(parser, "bits", "Output PNG bit depth, 8 or 16 (default: 16; 8-bit depth priors quantize visibly)", {"bit-depth"});
        ::args::Flag cpu(parser, "cpu", "Force CPU inference even if CUDA is available", {"cpu"});
        ::args::Flag overwrite(parser, "overwrite", "Overwrite existing depth/normal files", {'y', "overwrite"});
        ::args::Flag no_download(parser, "no-download", "Fail if the default model is not already cached", {"no-download"});
        ::args::Flag download_only(parser, "download-only", "Download/verify the default model and exit", {"download-only"});

        std::vector<std::string> args_vec(argv + 1, argv + argc);
        args_vec[0] = std::string(argv[0]) + " preprocess";
        parser.Prog(args_vec[0]);

        try {
            parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
        } catch (const ::args::Help&) {
            std::print("{}", parser.Help());
            return core_args::HelpMode{};
        } catch (const ::args::ParseError& e) {
            return std::unexpected(std::format("{}\n\n{}", e.what(), parser.Help()));
        }

        param::PreprocessParameters params;
        if (input) {
            params.dataset_path = lfs::core::utf8_to_path(::args::get(input));
        }
        if (images) {
            params.images_folder = ::args::get(images);
        }
        if (model) {
            params.model_path = lfs::core::utf8_to_path(::args::get(model));
        }
        if (mode) {
            params.mode = ::args::get(mode);
        }
        if (max_side) {
            params.max_side = ::args::get(max_side);
        }
        if (num_tokens) {
            params.num_tokens = ::args::get(num_tokens);
        }
        if (threads) {
            params.threads = ::args::get(threads);
        }
        if (png_compression) {
            params.png_compression = ::args::get(png_compression);
        }
        if (bit_depth) {
            params.bit_depth = ::args::get(bit_depth);
        }
        params.force_cpu = cpu;
        params.overwrite = overwrite;
        params.no_download = no_download;
        params.download_only = download_only;

        if (!params.download_only) {
            if (params.dataset_path.empty()) {
                return std::unexpected(std::format("Missing dataset path\n\n{}", parser.Help()));
            }
            if (!std::filesystem::exists(params.dataset_path)) {
                return std::unexpected(std::format("Dataset not found: {}", lfs::core::path_to_utf8(params.dataset_path)));
            }
            if (!std::filesystem::is_directory(params.dataset_path)) {
                return std::unexpected(std::format("Expected dataset directory: {}", lfs::core::path_to_utf8(params.dataset_path)));
            }
        }
        if (params.max_side < 0) {
            return std::unexpected("--max-side must be 0 or greater");
        }
        if (params.num_tokens <= 0) {
            return std::unexpected("--num-tokens must be greater than 0");
        }
        if (params.threads < 0) {
            return std::unexpected("--threads must be 0 or greater");
        }
        if (params.png_compression < 0 || params.png_compression > 9) {
            return std::unexpected("--png-compression must be between 0 and 9");
        }
        if (params.bit_depth != 8 && params.bit_depth != 16) {
            return std::unexpected("--bit-depth must be 8 or 16");
        }
        if (!params.model_path.empty() && !std::filesystem::exists(params.model_path)) {
            return std::unexpected(std::format("Model not found: {}", lfs::core::path_to_utf8(params.model_path)));
        }

        return core_args::PreprocessMode{params};
    }
} // namespace

std::expected<lfs::core::args::ParsedArgs, std::string>
lfs::core::args::parse_args(const int argc, const char* const argv[]) {
    if (argc >= 2) {
        const std::string_view arg1 = argv[1];

        if (arg1 == "-V" || arg1 == "--version") {
            return VersionMode{};
        }

        if (arg1 == "--warmup") {
            return WarmupMode{};
        }

        if (arg1 == "convert") {
            return parseConvertArgs(argc, argv);
        } else if (arg1 == "mesh2splat" || arg1 == "mesh-to-splat") {
            return parseMesh2SplatArgs(argc, argv);
        } else if (arg1 == "preprocess") {
            return parsePreprocessArgs(argc, argv);
        } else if (arg1 == "plugin") {
            if (argc < 3) {
                return std::unexpected("Usage: LichtFeld-Studio plugin <create|check|list> [name]");
            }

            const std::string_view subcmd = argv[2];
            PluginMode mode;

            if (subcmd == "create") {
                if (argc < 4) {
                    return std::unexpected("Usage: LichtFeld-Studio plugin create <name>");
                }
                mode.command = PluginMode::Command::CREATE;
                mode.name = argv[3];
            } else if (subcmd == "check") {
                if (argc < 4) {
                    return std::unexpected("Usage: LichtFeld-Studio plugin check <name>");
                }
                mode.command = PluginMode::Command::CHECK;
                mode.name = argv[3];
            } else if (subcmd == "list") {
                mode.command = PluginMode::Command::LIST;
            } else if (subcmd == "-h" || subcmd == "--help") {
                std::print(R"(Usage: LichtFeld-Studio plugin <command> [name]

Commands:
  create <name>   Create plugin with venv and VS Code config
  check <name>    Validate plugin structure
  list            List installed plugins
)");
                return HelpMode{};
            } else {
                return std::unexpected(std::format("Unknown plugin command: {}", subcmd));
            }

            return mode;
        } else {
            auto result = parse_args_and_params(argc, argv);
            if (!result)
                return std::unexpected(result.error());
            return TrainingMode{std::move(*result)};
        }
    } else {
        auto result = parse_args_and_params(argc, argv);
        if (!result)
            return std::unexpected(result.error());
        return TrainingMode{std::move(*result)};
    }
}
