/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include <algorithm>
#include <args.hxx>
#include <array>
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

    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "adc"};

    void scale_steps_vector(std::vector<size_t>& steps, float scaler) {
        std::set<size_t> unique_steps;
        for (const auto& step : steps) {
            size_t scaled = static_cast<size_t>(std::lround(static_cast<float>(step) * scaler));
            if (scaled > 0) {
                unique_steps.insert(scaled);
            }
        }
        steps.assign(unique_steps.begin(), unique_steps.end());
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
                "convert -- Convert between .ply, .sog, .spz, .html\n"
                "plugin -- Manage plugins (create, check, list)\n"
                "\n"
                "Run '<subcommand> --help' for details.\n"
                "\n"
                "EXAMPLES:\n"
                "lichtfeld-studio -d ./data -o ./output\n"
                "lichtfeld-studio --resume checkpoint.resume\n"
                "lichtfeld-studio -v model.ply\n"
                "lichtfeld-studio convert in.ply out.spz\n"
                "lichtfeld-studio plugin create my_plugin\n"
                "\n"
                "ENVIRONMENT:\n"
                "LOG_LEVEL -- Set log level (trace/debug/info/perf/warn/error)\n");
            parser.helpParams.width = 240;

            // =============================================================================
            // MODE SELECTION
            // =============================================================================
            ::args::Group mode_group(parser, "MODE SELECTION:");
            ::args::HelpFlag help(mode_group, "help", "Display help menu", {'h', "help"});
            ::args::Flag version(mode_group, "version", "Display version information", {'V', "version"});
            ::args::ValueFlag<std::string> view_ply(mode_group, "path", "View splat file(s). Supports .ply, .sog, .resume. If directory, loads all.", {'v', "view"});
            ::args::ValueFlag<std::string> resume_checkpoint(mode_group, "checkpoint", "Resume training from checkpoint file", {"resume"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // =============================================================================
            // TRAINING PATHS
            // =============================================================================
            ::args::Group paths_sep(parser, " ");
            ::args::Group paths_group(parser, "TRAINING PATHS:");
            ::args::ValueFlag<std::string> data_path(paths_group, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(paths_group, "output_path", "Path to output", {'o', "output-path"});
            ::args::ValueFlag<std::string> config_file(paths_group, "config_file", "LichtFeldStudio config file (json)", {"config"});
            ::args::ValueFlag<std::string> init_path(paths_group, "path", "Initialize from splat file (.ply, .sog, .spz, .resume)", {"init"});

            ::args::ValueFlag<std::string> import_cameras(paths_group, "path", "Import COLMAP cameras from sparse folder (no images required)", {"import-cameras"});

            // =============================================================================
            // TRAINING PARAMETERS
            // =============================================================================
            ::args::Group training_sep(parser, " ");
            ::args::Group training_group(parser, "TRAINING PARAMETERS:");
            ::args::ValueFlag<uint32_t> iterations(training_group, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<std::string> strategy(training_group, "strategy", "Optimization strategy: mcmc, adc", {"strategy"});
            ::args::ValueFlag<int> sh_degree(training_group, "sh_degree", "Max SH degree [0-3]", {"sh-degree"});
            ::args::ValueFlag<int> sh_degree_interval(training_group, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> max_cap(training_group, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
            ::args::ValueFlag<float> min_opacity(training_group, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<float> steps_scaler(training_group, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> tile_mode(training_group, "tile_mode", "Tile mode for memory-efficient training: 1=1 tile, 2=2 tiles, 4=4 tiles (default: 1)", {"tile-mode"});

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
            ::args::ValueFlag<int> max_width(dataset_group, "max_width", "Max width of images in px (default: 3840)", {"max-width"});
            ::args::Flag no_cpu_cache(dataset_group, "no_cpu_cache", "Disable CPU memory caching (default: enabled)", {"no-cpu-cache"});
            ::args::Flag no_fs_cache(dataset_group, "no_fs_cache", "Disable filesystem caching (default: enabled)", {"no-fs-cache"});
            ::args::Flag undistort(dataset_group, "undistort", "Undistort images on-the-fly before training", {"undistort"});

            // =============================================================================
            // MASK OPTIONS
            // =============================================================================
            ::args::Group mask_sep(parser, " ");
            ::args::Group mask_group(parser, "MASK OPTIONS:");
            ::args::MapFlag<std::string, lfs::core::param::MaskMode> mask_mode(mask_group, "mask_mode",
                                                                               "Mask mode: none, segment, ignore, alpha_consistent (default: none)",
                                                                               {"mask-mode"},
                                                                               std::unordered_map<std::string, lfs::core::param::MaskMode>{
                                                                                   {"none", lfs::core::param::MaskMode::None},
                                                                                   {"segment", lfs::core::param::MaskMode::Segment},
                                                                                   {"ignore", lfs::core::param::MaskMode::Ignore},
                                                                                   {"alpha_consistent", lfs::core::param::MaskMode::AlphaConsistent}});
            ::args::Flag invert_masks(mask_group, "invert_masks", "Invert mask values (swap object/background)", {"invert-masks"});
            ::args::Flag no_alpha_as_mask(mask_group, "no_alpha_as_mask", "Disable automatic alpha-as-mask for RGBA images", {"no-alpha-as-mask"});

            // =============================================================================
            // SPARSITY OPTIMIZATION
            // =============================================================================
            ::args::Group sparsity_sep(parser, " ");
            ::args::Group sparsity_group(parser, "SPARSITY OPTIMIZATION:");
            ::args::Flag enable_sparsity(sparsity_group, "enable_sparsity", "Enable sparsity optimization", {"enable-sparsity"});
            ::args::ValueFlag<int> sparsify_steps(sparsity_group, "sparsify_steps", "Number of steps for sparsification (default: 15000)", {"sparsify-steps"});
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
            ::args::Flag bg_modulation(rendering_group, "bg_modulation", "Enable sinusoidal background modulation", {"bg-modulation"});
            ::args::Flag gut(rendering_group, "gut", "Enable GUT mode", {"gut"});

            // =============================================================================
            // OUTPUT OPTIONS
            // =============================================================================
            ::args::Group output_sep(parser, " ");
            ::args::Group output_group(parser, "OUTPUT OPTIONS:");
            ::args::Flag enable_eval(output_group, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag enable_save_eval_images(output_group, "save_eval_images", "Save evaluation comparison images (GT vs rendered)", {"save-eval-images"});
            ::args::Flag save_depth(output_group, "save_depth", "[TODO] Save depth maps during training (not yet implemented)", {"save-depth"});
            ::args::ValueFlagList<std::string> timelapse_images(output_group, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});
            ::args::ValueFlag<int> timelapse_every(output_group, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});

            // =============================================================================
            // UI OPTIONS
            // =============================================================================
            ::args::Group ui_sep(parser, " ");
            ::args::Group ui_group(parser, "UI OPTIONS:");
            ::args::Flag headless(ui_group, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag auto_train(ui_group, "train", "Start training immediately on startup", {"train"});
            ::args::Flag no_splash(ui_group, "no_splash", "Skip splash screen on startup", {"no-splash"});
            ::args::Flag no_interop(ui_group, "no_interop", "Disable CUDA-GL interop (use CPU fallback for display)", {"no-interop"});
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
                if (const char* env_level = std::getenv("LOG_LEVEL")) {
                    level = parse_log_level(env_level);
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

                    constexpr std::array<std::string_view, 4> SUPPORTED_EXTENSIONS = {".ply", ".sog", ".spz", ".resume"};
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
                                "No supported files (.ply, .sog, .spz, .resume) found in: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        LOG_DEBUG("Found {} view files in directory", params.view_paths.size());
                    } else {
                        if (!is_supported(view_path)) {
                            return std::unexpected(std::format(
                                "Unsupported format. Expected: .ply, .sog, .spz, .resume. Got: {}", lfs::core::path_to_utf8(view_path)));
                        }
                        params.view_paths.push_back(view_path);
                    }
                }

                if (gut) {
                    params.optimization.gut = true;
                }
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
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, adc",
                        strat));
                }

                // Unlike other parameters that will be set later as overrides,
                // strategy must be set immediately to ensure correct JSON loading
                // in `read_optim_params_from_json()`
                params.optimization.strategy = strat;
            }

            if (config_file) {
                params.optimization.config_file = ::args::get(config_file);
                if (!strategy) {
                    params.optimization.strategy = ""; // Clear strategy to avoid using default strategy for evaluation of conflict
                }
            }

            if (max_width) {
                int width = ::args::get(max_width);
                if (width <= 0) {
                    return std::unexpected("ERROR: --max-width must be greather than 0");
                }
                if (width > 4096) {
                    return std::unexpected("ERROR: --max-width cannot be higher than 4096");
                }
            }

            if (tile_mode) {
                int mode = ::args::get(tile_mode);
                if (mode != 1 && mode != 2 && mode != 4) {
                    return std::unexpected("ERROR: --tile-mode must be 1 (1 tile), 2 (2 tiles), or 4 (4 tiles)");
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

            // Create lambda to apply command line overrides after JSON loading
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // default 1
                                        max_width_val = max_width ? std::optional<int>(::args::get(max_width)) : std::optional<int>(3840),          // default 3840
                                        no_cpu_cache_flag = static_cast<bool>(no_cpu_cache),
                                        no_fs_cache_flag = static_cast<bool>(no_fs_cache),
                                        max_cap_val = max_cap ? std::optional<int>(::args::get(max_cap)) : std::optional<int>(),
                                        config_file_val = config_file ? std::optional<std::string>(::args::get(config_file)) : std::optional<std::string>(),
                                        images_folder_val = images_folder ? std::optional<std::string>(::args::get(images_folder)) : std::optional<std::string>(),
                                        test_every_val = test_every ? std::optional<int>(::args::get(test_every)) : std::optional<int>(),
                                        steps_scaler_val = steps_scaler ? std::optional<float>(::args::get(steps_scaler)) : std::optional<float>(),
                                        sh_degree_interval_val = sh_degree_interval ? std::optional<int>(::args::get(sh_degree_interval)) : std::optional<int>(),
                                        sh_degree_val = sh_degree ? std::optional<int>(::args::get(sh_degree)) : std::optional<int>(),
                                        min_opacity_val = min_opacity ? std::optional<float>(::args::get(min_opacity)) : std::optional<float>(),
                                        init_num_pts_val = init_num_pts ? std::optional<int>(::args::get(init_num_pts)) : std::optional<int>(),
                                        init_extent_val = init_extent ? std::optional<float>(::args::get(init_extent)) : std::optional<float>(),
                                        strategy_val = strategy ? std::optional<std::string>(::args::get(strategy)) : std::optional<std::string>(),
                                        timelapse_images_val = timelapse_images ? std::optional<std::vector<std::string>>(::args::get(timelapse_images)) : std::optional<std::vector<std::string>>(),
                                        timelapse_every_val = timelapse_every ? std::optional<int>(::args::get(timelapse_every)) : std::optional<int>(),
                                        tile_mode_val = tile_mode ? std::optional<int>(::args::get(tile_mode)) : std::optional<int>(),
                                        // Sparsity parameters
                                        sparsify_steps_val = sparsify_steps ? std::optional<int>(::args::get(sparsify_steps)) : std::optional<int>(),
                                        init_rho_val = init_rho ? std::optional<float>(::args::get(init_rho)) : std::optional<float>(),
                                        prune_ratio_val = prune_ratio ? std::optional<float>(::args::get(prune_ratio)) : std::optional<float>(),
                                        // Mask parameters
                                        mask_mode_val = mask_mode ? std::optional<lfs::core::param::MaskMode>(::args::get(mask_mode)) : std::optional<lfs::core::param::MaskMode>(),
                                        // Python scripts
                                        python_scripts_val = python_scripts ? std::optional<std::vector<std::string>>(::args::get(python_scripts)) : std::optional<std::vector<std::string>>(),
                                        // Capture flag states
                                        enable_mip_flag = bool(enable_mip),
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        use_ppisp_flag = bool(use_ppisp),
                                        ppisp_controller_flag = bool(ppisp_controller),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        auto_train_flag = bool(auto_train),
                                        no_splash_flag = bool(no_splash),
                                        no_interop_flag = bool(no_interop),
                                        debug_python_flag = bool(debug_python),
                                        debug_python_port_val = debug_python_port ? std::optional<int>(::args::get(debug_python_port)) : std::optional<int>(),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        bg_modulation_flag = bool(bg_modulation),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut),
                                        undistort_flag = bool(undistort),
                                        enable_sparsity_flag = bool(enable_sparsity),
                                        invert_masks_flag = bool(invert_masks),
                                        no_alpha_as_mask_flag = bool(no_alpha_as_mask)]() {
                auto& opt = params.optimization;
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
                if (no_cpu_cache_flag)
                    ds.loading_params.use_cpu_memory = false;
                if (no_fs_cache_flag)
                    ds.loading_params.use_fs_cache = false;
                setVal(max_cap_val, opt.max_cap);
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
                setVal(tile_mode_val, opt.tile_mode);

                // Sparsity parameters
                setVal(sparsify_steps_val, opt.sparsify_steps);
                setVal(init_rho_val, opt.init_rho);
                setVal(prune_ratio_val, opt.prune_ratio);

                setFlag(enable_mip_flag, opt.mip_filter);
                setFlag(use_bilateral_grid_flag, opt.use_bilateral_grid);
                setFlag(use_ppisp_flag, opt.use_ppisp);
                setFlag(ppisp_controller_flag, opt.ppisp_use_controller);
                if (opt.ppisp_use_controller)
                    opt.use_ppisp = true;
                setFlag(enable_eval_flag, opt.enable_eval);
                setFlag(headless_flag, opt.headless);
                setFlag(auto_train_flag, opt.auto_train);
                setFlag(no_splash_flag, opt.no_splash);
                setFlag(no_interop_flag, opt.no_interop);
                setFlag(debug_python_flag, opt.debug_python);
                setVal(debug_python_port_val, opt.debug_python_port);
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images);
                setFlag(bg_modulation_flag, opt.bg_modulation);
                setFlag(random_flag, opt.random);
                setFlag(gut_flag, opt.gut);
                setFlag(undistort_flag, opt.undistort);
                setFlag(enable_sparsity_flag, opt.enable_sparsity);

                // Mask parameters
                setVal(mask_mode_val, opt.mask_mode);
                setFlag(invert_masks_flag, opt.invert_masks);
                if (no_alpha_as_mask_flag)
                    opt.use_alpha_as_mask = false;
                // Also propagate to dataset config for loading
                ds.invert_masks = opt.invert_masks;
                ds.mask_threshold = opt.mask_threshold;

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
        const float scaler = opt.steps_scaler;

        if (scaler > 0) {
            LOG_INFO("Scaling training steps by factor: {}", scaler);

            const auto scale = [scaler](const size_t v) {
                return static_cast<size_t>(std::lround(static_cast<float>(v) * scaler));
            };
            opt.iterations = scale(opt.iterations);
            opt.start_refine = scale(opt.start_refine);
            opt.reset_every = scale(opt.reset_every);
            opt.stop_refine = scale(opt.stop_refine);
            opt.refine_every = scale(opt.refine_every);
            opt.sh_degree_interval = scale(opt.sh_degree_interval);

            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
        }
    }

    void apply_ppisp_defaults(lfs::core::param::TrainingParameters& params) {
        auto& opt = params.optimization;
        if (!opt.ppisp_use_controller)
            return;

        if (opt.ppisp_controller_activation_step < 0) {
            constexpr int CONTROLLER_TRAINING_ITERS = 5000;
            opt.ppisp_controller_activation_step =
                std::max(0, static_cast<int>(opt.iterations) - CONTROLLER_TRAINING_ITERS);
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
    auto parse_result = parse_arguments(convert_args(argc, argv), *params);
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

        if (!strategy.empty() && strategy != params->optimization.strategy) {
            return std::unexpected("--strategy conflicts with config file");
        }
    } else {
        params->optimization = (strategy == "adc")
                                   ? lfs::core::param::OptimizationParameters::adc_defaults()
                                   : lfs::core::param::OptimizationParameters::mcmc_defaults();
    }

    params->dataset.loading_params = lfs::core::param::LoadingParams{};

    if (apply_overrides) {
        apply_overrides();
    }
    apply_step_scaling(*params);
    apply_ppisp_defaults(*params);

    if (auto error = params->optimization.validate(); !error.empty())
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
        "  Input:  .ply, .sog, .spz, .resume (checkpoint)\n"
        "  Output: .ply, .sog, .spz, .html\n"
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
        return std::nullopt;
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

        if (arg1 == "--mcp") {
            McpMode mode;
            if (argc >= 3) {
                const std::filesystem::path scene_path = lfs::core::utf8_to_path(argv[2]);
                if (std::filesystem::exists(scene_path)) {
                    mode.scene_path = scene_path;
                }
            }
            return mode;
        }

        if (arg1 == "convert") {
            // Handle convert subcommand below
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

    // Convert subcommand
    ::args::ArgumentParser parser(CONVERT_HELP_HEADER, CONVERT_HELP_FOOTER);
    ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    ::args::Positional<std::string> input(parser, "input", "Input file or directory");
    ::args::Positional<std::string> output(parser, "output", "Output file (optional)");
    ::args::ValueFlag<int> sh_degree(parser, "degree", "SH degree [0-3], -1 to keep original (default: -1)", {"sh-degree"});
    ::args::ValueFlag<std::string> format(parser, "format", "Output format: ply, sog, spz, html", {'f', "format"});
    ::args::ValueFlag<int> sog_iter(parser, "iterations", "K-means iterations for SOG (default: 10)", {"sog-iterations"});
    ::args::Flag overwrite(parser, "overwrite", "Overwrite existing files without prompting", {'y', "overwrite"});

    std::vector<std::string> args_vec(argv + 1, argv + argc);
    args_vec[0] = std::string(argv[0]) + " convert";
    parser.Prog(args_vec[0]);

    try {
        parser.ParseArgs(std::vector<std::string>(args_vec.begin() + 1, args_vec.end()));
    } catch (const ::args::Help&) {
        std::print("{}", parser.Help());
        return HelpMode{};
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
            return std::unexpected(std::format("Invalid format '{}'. Use: ply, sog, html", ::args::get(format)));
        }
    } else if (!params.output_path.empty()) {
        if (const auto fmt = parseFormat(params.output_path.extension().string())) {
            params.format = *fmt;
        } else {
            return std::unexpected(std::format("Unknown extension '{}'. Use --format", params.output_path.extension().string()));
        }
    }

    return ConvertMode{params};
}
