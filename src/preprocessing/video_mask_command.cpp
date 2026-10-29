/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "preprocessing/video_mask_command.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "preprocessing/preprocess.hpp"
#include "preprocessing/video_mask_extraction.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace lfs::preprocessing {
    namespace {
        bool frame_less(const std::filesystem::path& a, const std::filesystem::path& b) {
            const auto left = core::path_to_utf8(a.filename()), right = core::path_to_utf8(b.filename());
            size_t i = 0, j = 0;
            while (i < left.size() && j < right.size()) {
                if (std::isdigit(static_cast<unsigned char>(left[i])) && std::isdigit(static_cast<unsigned char>(right[j]))) {
                    const size_t begin_i = i, begin_j = j;
                    while (i < left.size() && std::isdigit(static_cast<unsigned char>(left[i])))
                        ++i;
                    while (j < right.size() && std::isdigit(static_cast<unsigned char>(right[j])))
                        ++j;
                    size_t first_i = begin_i, first_j = begin_j;
                    while (first_i + 1 < i && left[first_i] == '0')
                        ++first_i;
                    while (first_j + 1 < j && right[first_j] == '0')
                        ++first_j;
                    if (i - first_i != j - first_j)
                        return i - first_i < j - first_j;
                    const int compared = left.compare(first_i, i - first_i, right, first_j, j - first_j);
                    if (compared)
                        return compared < 0;
                } else {
                    if (left[i] != right[j])
                        return left[i] < right[j];
                    ++i;
                    ++j;
                }
            }
            return i == left.size() && j != right.size() ? true : (j == right.size() && i != left.size() ? false : left < right);
        }
    } // namespace

    int run_video_masking(const core::param::VideoMaskParameters& params) {
        try {
            if (!core::Logger::get().is_ready()) {
                const auto paths = core::UserPaths::resolve();
                if (!paths)
                    throw std::runtime_error(std::string(paths.error().user_message()));
                core::Logger::get().init(core::LogLevel::Info, {}, {}, false, core::path_to_utf8(paths->rootDir()));
            }
            if (params.download_only) {
                const auto weights = ensure_sam2_video_weights(!params.no_download);
                if (!weights)
                    throw std::runtime_error(std::string(weights.error().user_message()));
                std::cout << core::path_to_utf8(*weights) << '\n';
                return 0;
            }
            const auto started = std::chrono::steady_clock::now();
            VideoMaskExtractionOptions options;
            options.enabled = true;
            options.circle = {params.circle[0], params.circle[1], params.circle[2], params.circle[3]};
            options.masking.weights = params.model_path;
            options.masking.allow_download = !params.no_download;
            options.masking.recenter_fisheye = params.recenter;
            options.masking.fisheye_fov_degrees = params.fisheye_fov;
            options.masking.view_fov_degrees = params.view_fov;
            options.masking.dilation_radius = params.dilation;
            options.masking.require_operator = params.require_operator;
            if (!params.points.empty())
                options.masking.prompt = {.frame = params.seed_frame, .points = params.points, .box = params.box};
            options.masking.progress = [](const VideoMaskProgress& p) {
                if (p.completed == p.total || p.completed % 25 == 0)
                    LOG_INFO("SAM2 stage {}: {}/{}", int(p.stage), p.completed, p.total);
            };
            size_t count = 0;
            if (core::native_fs::is_directory(params.input_path)) {
                if (!params.output_path.empty() && core::native_fs::weakly_canonical(params.output_path) !=
                                                       core::native_fs::weakly_canonical(params.input_path))
                    throw std::runtime_error("For extracted frames, masks are written in the input dataset; omit --output");
                const auto images = params.input_path / "images";
                std::vector<std::filesystem::path> paths;
                for (const auto& entry : core::native_fs::directory_iterator(images)) {
                    if (!core::native_fs::entry_is_regular_file(images, entry))
                        continue;
                    auto extension = core::path_to_utf8(entry.path().extension());
                    std::ranges::transform(extension, extension.begin(), [](unsigned char c) { return std::tolower(c); });
                    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png")
                        paths.push_back(core::native_fs::entry_path(images, entry.path()));
                }
                std::sort(paths.begin(), paths.end(), frame_less);
                std::vector<VideoMaskFrame> frames;
                for (const auto& path : paths) {
                    const auto mask = video_mask_path(params.input_path, path.filename());
                    if (!params.overwrite && core::native_fs::exists(mask))
                        throw std::runtime_error("Masks already exist; use --overwrite to replace them");
                    const auto [width, height, channels] = core::get_image_info(path);
                    if (width <= 0 || height <= 0)
                        throw std::runtime_error("Cannot read an input frame");
                    frames.push_back({.image_path = path, .mask_path = mask, .width = width, .height = height, .time_seconds = double(frames.size()) / params.fps});
                }
                if (frames.empty())
                    throw std::runtime_error("No image frames found in images/");
                auto result = mask_video_frames(frames, options.circle, options.masking);
                if (!result)
                    throw std::runtime_error(std::string(result.error().user_message()));
                count = frames.size();
            } else {
                if (!core::native_fs::is_regular_file(params.input_path))
                    throw std::runtime_error("Input video does not exist");
                if (params.output_path.empty())
                    throw std::runtime_error("Video input requires --output dataset_root");
                if (!params.overwrite) {
                    for (const auto* folder : {"images", "masks"}) {
                        const auto directory = params.output_path / folder;
                        if (!core::native_fs::is_directory(directory))
                            continue;
                        for (const auto& entry : core::native_fs::directory_iterator(directory)) {
                            auto name = core::path_to_utf8(entry.path().filename());
                            if (std::string_view(folder) == "masks" && name.ends_with(".png"))
                                name.resize(name.size() - 4);
                            if (io::isGeneratedFrameFilename("frame_%06d", name, true))
                                throw std::runtime_error("Generated output already exists; use --overwrite to replace it");
                        }
                    }
                }
                io::VideoFrameExtractor::Params extraction;
                extraction.video_path = params.input_path;
                extraction.output_dir = params.output_path;
                extraction.fps = params.fps;
                extraction.start_time = params.start_time;
                extraction.end_time = params.end_time;
                extraction.filename_pattern = "frame_%06d";
                extraction.format = io::ImageFormat::JPG;
                extraction.frame_written = [&](const auto&) { ++count; };
                std::string error;
                if (extract_video_frames(extraction, options, error) != io::ExtractionOutcome::Completed)
                    throw std::runtime_error(error);
            }
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            LOG_INFO("Video masking: {} frames in {:.3f} seconds ({:.3f} seconds/frame)", count, elapsed, elapsed / count);
            return 0;
        } catch (const std::exception& e) {
            // LFS-CENSUS-OK(empty-catch): the command reports the failure on stderr and returns its exit status.
            std::cerr << "mask-video: " << e.what() << '\n';
            return 1;
        }
    }
} // namespace lfs::preprocessing
