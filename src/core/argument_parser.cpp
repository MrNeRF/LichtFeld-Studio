/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/argument_parser.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include <algorithm>
#include <args.hxx>
#include <array>
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

    const std::set<std::string> VALID_STRATEGIES = {"mcmc", "default"};

    void scale_steps_vector(std::vector<size_t>& steps, float scaler) {
        std::set<size_t> unique_steps;
        for (const auto& step : steps) {
            size_t scaled = static_cast<size_t>(static_cast<float>(step) * scaler);
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
                "LichtFeld Studio: High-performance CUDA implementation of 3D Gaussian Splatting algorithm. \n",
                "Usage:\n"
                "  Training: LichtFeld-Studio --data-path <path> --output-path <path> [options]\n"
                "  Resume:   LichtFeld-Studio --resume <checkpoint.resume> [options]\n"
                "  Viewing:  LichtFeld-Studio --view <file_or_directory> [options]\n");

            // Define all arguments
            ::args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
            ::args::CompletionFlag completion(parser, {"complete"});

            // PLY viewing mode (supports single file or directory with multiple files)
            ::args::ValueFlag<std::string> view_ply(parser, "path", "View splat file(s). Supports .ply, .sog, .resume. If directory, loads all.", {'v', "view"});

            // Resume from checkpoint
            ::args::ValueFlag<std::string> resume_checkpoint(parser, "checkpoint", "Resume training from checkpoint file", {"resume"});

            // Training mode arguments
            ::args::ValueFlag<std::string> data_path(parser, "data_path", "Path to training data", {'d', "data-path"});
            ::args::ValueFlag<std::string> output_path(parser, "output_path", "Path to output", {'o', "output-path"});

            // config file argument
            ::args::ValueFlag<std::string> config_file(parser, "config_file", "LichtFeldStudio config file (json)", {"config"});

            // Optional value arguments
            ::args::ValueFlag<uint32_t> iterations(parser, "iterations", "Number of iterations", {'i', "iter"});
            ::args::ValueFlag<int> num_workers(parser, "num_threads", "Number of workers", {"num-workers"});
            ::args::ValueFlag<int> max_cap(parser, "max_cap", "Max Gaussians for MCMC", {"max-cap"});
            ::args::ValueFlag<std::string> images_folder(parser, "images", "Images folder name", {"images"});
            ::args::ValueFlag<int> test_every(parser, "test_every", "Use every Nth image as test", {"test-every"});
            ::args::ValueFlag<float> steps_scaler(parser, "steps_scaler", "Scale training steps by factor", {"steps-scaler"});
            ::args::ValueFlag<int> sh_degree_interval(parser, "sh_degree_interval", "SH degree interval", {"sh-degree-interval"});
            ::args::ValueFlag<int> sh_degree(parser, "sh_degree", "Max SH degree [0-3]", {"sh-degree"});
            ::args::ValueFlag<float> min_opacity(parser, "min_opacity", "Minimum opacity threshold", {"min-opacity"});
            ::args::ValueFlag<std::string> strategy(parser, "strategy", "Optimization strategy: mcmc, default", {"strategy"});
            ::args::ValueFlag<int> init_num_pts(parser, "init_num_pts", "Number of random initialization points", {"init-num-pts"});
            ::args::ValueFlag<float> init_extent(parser, "init_extent", "Extent of random initialization", {"init-extent"});
            ::args::ValueFlagList<std::string> timelapse_images(parser, "timelapse_images", "Image filenames to render timelapse images for", {"timelapse-images"});
            ::args::ValueFlag<int> timelapse_every(parser, "timelapse_every", "Render timelapse image every N iterations (default: 50)", {"timelapse-every"});
            ::args::ValueFlag<std::string> init_path(parser, "path", "Initialize from splat file (.ply, .sog, .spz, .resume)", {"init"});
            ::args::ValueFlag<int> tile_mode(parser, "tile_mode", "Tile mode for memory-efficient training: 1=1 tile, 2=2 tiles, 4=4 tiles (default: 1)", {"tile-mode"});

            // Sparsity optimization arguments
            ::args::ValueFlag<int> sparsify_steps(parser, "sparsify_steps", "Number of steps for sparsification (default: 15000)", {"sparsify-steps"});
            ::args::ValueFlag<float> init_rho(parser, "init_rho", "Initial ADMM penalty parameter (default: 0.0005)", {"init-rho"});
            ::args::ValueFlag<float> prune_ratio(parser, "prune_ratio", "Final pruning ratio for sparsity (default: 0.6)", {"prune-ratio"});

            // Logging options
            ::args::ValueFlag<std::string> log_level(parser, "level", "Log level: trace, debug, info, perf, warn, error, critical, off (default: info)", {"log-level"});
            ::args::ValueFlag<std::string> log_file(parser, "file", "Optional log file path", {"log-file"});
            ::args::ValueFlag<std::string> log_filter(parser, "pattern", "Filter log messages (glob: *foo*, regex: \\\\d+)", {"log-filter"});

            // Optional flag arguments
            ::args::Flag enable_mip(parser, "enable_mip", "Enable mip filter (anti-aliasing)", {"enable-mip"});
            ::args::Flag use_bilateral_grid(parser, "bilateral_grid", "Enable bilateral grid filtering", {"bilateral-grid"});
            ::args::Flag enable_eval(parser, "eval", "Enable evaluation during training", {"eval"});
            ::args::Flag headless(parser, "headless", "Disable visualization during training", {"headless"});
            ::args::Flag no_splash(parser, "no_splash", "Skip splash screen on startup", {"no-splash"});
            ::args::Flag no_interop(parser, "no_interop", "Disable CUDA-GL interop (use CPU fallback for display)", {"no-interop"});
            ::args::Flag enable_save_eval_images(parser, "save_eval_images", "Save eval images and depth maps", {"save-eval-images"});
            ::args::Flag save_depth(parser, "save_depth", "Save depth maps during training", {"save-depth"});
            ::args::Flag bg_modulation(parser, "bg_modulation", "Enable sinusoidal background modulation mixed with base background", {"bg-modulation"});
            ::args::Flag random(parser, "random", "Use random initialization instead of SfM", {"random"});
            ::args::Flag gut(parser, "gut", "Enable GUT mode", {"gut"});
            ::args::Flag enable_sparsity(parser, "enable_sparsity", "Enable sparsity optimization", {"enable-sparsity"});

            // Mask-related arguments
            ::args::MapFlag<std::string, lfs::core::param::MaskMode> mask_mode(parser, "mask_mode",
                                                                               "Mask mode: none, segment, ignore, alpha_consistent (default: none)",
                                                                               {"mask-mode"},
                                                                               std::unordered_map<std::string, lfs::core::param::MaskMode>{
                                                                                   {"none", lfs::core::param::MaskMode::None},
                                                                                   {"segment", lfs::core::param::MaskMode::Segment},
                                                                                   {"ignore", lfs::core::param::MaskMode::Ignore},
                                                                                   {"alpha_consistent", lfs::core::param::MaskMode::AlphaConsistent}});
            ::args::Flag invert_masks(parser, "invert_masks", "Invert mask values (swap object/background)", {"invert-masks"});

            ::args::MapFlag<std::string, int> resize_factor(parser, "resize_factor",
                                                            "resize resolution by this factor. Options: auto, 1, 2, 4, 8 (default: auto)",
                                                            {'r', "resize_factor"},
                                                            // load_image only supports those resizes
                                                            std::unordered_map<std::string, int>{
                                                                {"auto", 1},
                                                                {"1", 1},
                                                                {"2", 2},
                                                                {"4", 4},
                                                                {"8", 8}});

            ::args::ValueFlag<int> max_width(parser, "max_width", "Max width of images in px (default: 3840)", {"max-width"});
            ::args::ValueFlag<bool> use_cpu_cache(parser, "use_cpu_cache", "if true - try using cpu memory to cache images (default: true)", {"use_cpu_cache"});
            ::args::ValueFlag<bool> use_fs_cache(parser, "use_fs_cache", "if true - try using temporary file system to cache images (default: true)", {"use_fs_cache"});

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
                // CLI argument overrides environment variable
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
                    const std::filesystem::path view_path{view_path_str};

                    if (!std::filesystem::exists(view_path)) {
                        return std::unexpected(std::format("Path does not exist: {}", view_path.string()));
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
                                "No supported files (.ply, .sog, .spz, .resume) found in: {}", view_path.string()));
                        }
                        LOG_DEBUG("Found {} view files in directory", params.view_paths.size());
                    } else {
                        if (!is_supported(view_path)) {
                            return std::unexpected(std::format(
                                "Unsupported format. Expected: .ply, .sog, .spz, .resume. Got: {}", view_path.string()));
                        }
                        params.view_paths.push_back(view_path);
                    }
                }

                if (gut) {
                    params.optimization.gut = true;
                }
                return std::make_tuple(ParseResult::Success, std::function<void()>{});
            }

            // Check for resume mode
            if (resume_checkpoint) {
                const auto ckpt_path = ::args::get(resume_checkpoint);
                if (!ckpt_path.empty()) {
                    if (!std::filesystem::exists(ckpt_path)) {
                        return std::unexpected(std::format("Checkpoint file does not exist: {}", ckpt_path));
                    }
                    params.resume_checkpoint = ckpt_path;
                }
            }

            if (init_path) {
                const auto path = ::args::get(init_path);
                params.init_path = path;

                if (!std::filesystem::exists(path)) {
                    return std::unexpected(std::format("Initialization file does not exist: {}", path));
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
                params.dataset.data_path = ::args::get(data_path);
                params.dataset.output_path = ::args::get(output_path);

                // Create output directory
                std::error_code ec;
                std::filesystem::create_directories(params.dataset.output_path, ec);
                if (ec) {
                    return std::unexpected(std::format(
                        "Failed to create output directory '{}': {}",
                        params.dataset.output_path.string(), ec.message()));
                }
            } else if (has_data_path != has_output_path && !has_resume) {
                // Only require both if not in resume mode
                return std::unexpected(std::format(
                    "ERROR: Training mode requires both --data-path and --output-path\n\n{}",
                    parser.Help()));
            } else if (has_resume) {
                // Resume mode: paths are optional (will be read from checkpoint)
                if (has_data_path) {
                    params.dataset.data_path = ::args::get(data_path);
                }
                if (has_output_path) {
                    params.dataset.output_path = ::args::get(output_path);

                    // Create output directory if provided
                    std::error_code ec;
                    std::filesystem::create_directories(params.dataset.output_path, ec);
                    if (ec) {
                        return std::unexpected(std::format(
                            "Failed to create output directory '{}': {}",
                            params.dataset.output_path.string(), ec.message()));
                    }
                }
            }

            if (strategy) {
                const auto strat = ::args::get(strategy);
                if (VALID_STRATEGIES.find(strat) == VALID_STRATEGIES.end()) {
                    return std::unexpected(std::format(
                        "ERROR: Invalid optimization strategy '{}'. Valid strategies are: mcmc, default",
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

            // Create lambda to apply command line overrides after JSON loading
            auto apply_cmd_overrides = [&params,
                                        // Capture values, not references
                                        iterations_val = iterations ? std::optional<uint32_t>(::args::get(iterations)) : std::optional<uint32_t>(),
                                        resize_factor_val = resize_factor ? std::optional<int>(::args::get(resize_factor)) : std::optional<int>(1), // default 1
                                        max_width_val = max_width ? std::optional<int>(::args::get(max_width)) : std::optional<int>(3840),          // default 3840
                                        use_cpu_cache_val = use_cpu_cache ? std::optional<bool>(::args::get(use_cpu_cache)) : std::optional<bool>(),
                                        use_fs_cache_val = use_fs_cache ? std::optional<bool>(::args::get(use_fs_cache)) : std::optional<bool>(),
                                        num_workers_val = num_workers ? std::optional<int>(::args::get(num_workers)) : std::optional<int>(),
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
                                        // Capture flag states
                                        enable_mip_flag = bool(enable_mip),
                                        use_bilateral_grid_flag = bool(use_bilateral_grid),
                                        enable_eval_flag = bool(enable_eval),
                                        headless_flag = bool(headless),
                                        no_splash_flag = bool(no_splash),
                                        no_interop_flag = bool(no_interop),
                                        enable_save_eval_images_flag = bool(enable_save_eval_images),
                                        bg_modulation_flag = bool(bg_modulation),
                                        random_flag = bool(random),
                                        gut_flag = bool(gut),
                                        enable_sparsity_flag = bool(enable_sparsity),
                                        invert_masks_flag = bool(invert_masks)]() {
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
                setVal(use_cpu_cache_val, ds.loading_params.use_cpu_memory);
                setVal(use_fs_cache_val, ds.loading_params.use_fs_cache);
                setVal(num_workers_val, opt.num_workers);
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
                setFlag(enable_eval_flag, opt.enable_eval);
                setFlag(headless_flag, opt.headless);
                setFlag(no_splash_flag, opt.no_splash);
                setFlag(no_interop_flag, opt.no_interop);
                setFlag(enable_save_eval_images_flag, opt.enable_save_eval_images);
                setFlag(bg_modulation_flag, opt.bg_modulation);
                setFlag(random_flag, opt.random);
                setFlag(gut_flag, opt.gut);
                setFlag(enable_sparsity_flag, opt.enable_sparsity);

                // Mask parameters
                setVal(mask_mode_val, opt.mask_mode);
                setFlag(invert_masks_flag, opt.invert_masks);
                // Also propagate to dataset config for loading
                ds.invert_masks = opt.invert_masks;
                ds.mask_threshold = opt.mask_threshold;
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

            opt.iterations *= scaler;
            opt.start_refine *= scaler;
            opt.reset_every *= scaler;
            opt.stop_refine *= scaler;
            opt.refine_every *= scaler;
            opt.sh_degree_interval *= scaler;

            scale_steps_vector(opt.eval_steps, scaler);
            scale_steps_vector(opt.save_steps, scaler);
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

    // Parse command line arguments
    auto parse_result = parse_arguments(convert_args(argc, argv), *params);

    std::string strategy = params->optimization.strategy; // empty when config files is used and not passed as command line argument
    std::string config_file = params->optimization.config_file;
    std::filesystem::path config_file_to_read = config_file != ""
                                                    ? std::filesystem::path(reinterpret_cast<const char8_t*>(config_file.c_str()))
                                                    : lfs::core::param::get_parameter_file_path(params->optimization.strategy + "_optimization_params.json");

    if (!parse_result) {
        return std::unexpected(parse_result.error());
    }

    auto [result, apply_overrides] = *parse_result;

    // Handle help case
    if (result == ParseResult::Help) {
        std::exit(0);
    }

    auto opt_params_result = lfs::core::param::read_optim_params_from_json(config_file_to_read);
    if (!opt_params_result) {
        return std::unexpected(std::format("Failed to load optimization parameters: {}",
                                           opt_params_result.error()));
    }
    params->optimization = *opt_params_result;

    std::filesystem::path config_file_loading = lfs::core::param::get_parameter_file_path("loading_params.json");
    auto loading_result = lfs::core::param::read_loading_params_from_json(config_file_loading);
    if (!loading_result) {
        return std::unexpected(std::format("Failed to load loading parameters: {}",
                                           loading_result.error()));
    }
    params->dataset.loading_params = *loading_result;

    // if a config file was used and strategy was also passed as command line argument, ensure they match
    if (config_file != "" && strategy != "" && strategy != params->optimization.strategy) {
        LOG_ERROR("Conflict between strategy in config file and --strategy on command line");
        return std::unexpected(std::format("Conflict between strategy in config file and --strategy on command line"));
    }

    // Apply command line overrides
    if (apply_overrides) {
        apply_overrides();
    }

    // Apply step scaling
    apply_step_scaling(*params);

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

        if (arg1 == "--warmup") {
            return WarmupMode{};
        }

        if (arg1 == "convert") {
            // Handle convert subcommand below
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
    params.input_path = ::args::get(input);
    params.sh_degree = sh_degree ? ::args::get(sh_degree) : -1;

    if (!std::filesystem::exists(params.input_path)) {
        return std::unexpected(std::format("Input not found: {}", params.input_path.string()));
    }

    if (params.sh_degree < -1 || params.sh_degree > 3) {
        return std::unexpected("SH degree must be -1 (keep) or 0-3");
    }

    if (output)
        params.output_path = ::args::get(output);
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
