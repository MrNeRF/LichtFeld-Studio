/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "normal_auto_generate.hpp"

#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "io/filesystem_utils.hpp"
#include "preprocessing/preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <expected>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

namespace lfs::training {
    namespace {

        constexpr std::string_view kOriginalImagesFolder = "images";

        void count_prior_maps(std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
                              PriorAutoGenerateOutcome& outcome) {
            outcome.depth = {};
            outcome.normal = {};
            for (const auto& cam : cameras) {
                if (!cam || !cam->has_image())
                    continue;
                ++(cam->has_depth() ? outcome.depth.existing : outcome.depth.missing);
                ++(cam->has_normal() ? outcome.normal.existing : outcome.normal.missing);
            }
        }

        [[nodiscard]] std::string training_images_folder(
            const lfs::core::param::TrainingParameters& params) {
            return params.dataset.images.empty()
                       ? std::string(kOriginalImagesFolder)
                       : params.dataset.images;
        }

        // Full-res images/ is the COLMAP original_width/height source. Maps written
        // at that size pass sidecar_dimensions_match_contract for every --images
        // choice. Fall back to the training folder only when images/ is absent.
        [[nodiscard]] bool original_images_folder_available(
            const std::filesystem::path& dataset_root,
            const std::string& training_folder) {
            if (training_folder == kOriginalImagesFolder)
                return false;
            const auto original_dir = dataset_root / kOriginalImagesFolder;
            if (!lfs::io::safe_is_directory(original_dir))
                return false;
            const auto training_dir = dataset_root / lfs::core::utf8_to_path(training_folder);
            if (lfs::io::paths_equivalent_or_lexically_equal(original_dir, training_dir))
                return false;
            return true;
        }

        [[nodiscard]] std::string resolve_generation_images_folder(
            const std::filesystem::path& dataset_root,
            const std::string& training_folder) {
            if (original_images_folder_available(dataset_root, training_folder))
                return std::string(kOriginalImagesFolder);
            return training_folder;
        }

        [[nodiscard]] std::filesystem::path resolve_output_dir(
            const std::filesystem::path& dataset_root,
            const std::span<const char* const> search_folders,
            const std::string_view fallback) {
            for (const char* folder : search_folders) {
                if (lfs::io::safe_is_directory(dataset_root / folder))
                    return dataset_root / folder;
            }
            return dataset_root / fallback;
        }

        [[nodiscard]] std::filesystem::path relative_prior_filename(
            const std::filesystem::path& image_path,
            const std::string& image_name,
            const std::filesystem::path& dataset_root,
            const std::string& images_folder) {
            const auto images_dir = dataset_root / lfs::core::utf8_to_path(images_folder);
            std::filesystem::path rel;
            if (!image_path.empty() && !images_dir.empty()) {
                rel = image_path.lexically_relative(images_dir);
                const auto generic = rel.generic_string();
                if (rel.empty() || generic == "." || generic == ".." || generic.starts_with("../"))
                    rel.clear();
            }
            if (rel.empty()) {
                rel = lfs::core::utf8_to_path(image_name);
                if (rel.empty())
                    rel = image_path.filename();
            }
            rel.replace_extension(".png");
            return rel;
        }

        [[nodiscard]] std::filesystem::path resolve_generation_image_path(
            const lfs::core::Camera& cam,
            const std::filesystem::path& dataset_root,
            const std::string& training_folder,
            const std::string& source_folder) {
            if (source_folder == training_folder)
                return cam.image_path();

            const auto training_dir = dataset_root / lfs::core::utf8_to_path(training_folder);
            const auto source_dir = dataset_root / lfs::core::utf8_to_path(source_folder);
            std::filesystem::path rel;
            if (!cam.image_path().empty() && !training_dir.empty()) {
                rel = cam.image_path().lexically_relative(training_dir);
                const auto generic = rel.generic_string();
                if (rel.empty() || generic == "." || generic == ".." || generic.starts_with("../"))
                    rel.clear();
            }
            if (rel.empty()) {
                rel = lfs::core::utf8_to_path(cam.image_name());
                if (rel.empty())
                    rel = cam.image_path().filename();
            }
            return source_dir / rel;
        }

        void associate_depth_maps(
            std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
            const std::filesystem::path& dataset_root) {
            lfs::io::DepthDirCache cache(dataset_root);
            if (!cache.has_depth_dirs())
                return;
            for (const auto& cam : cameras) {
                if (!cam || cam->has_depth())
                    continue;
                if (auto lookup = cache.lookup(cam->image_name()); lookup.found()) {
                    cam->set_depth_path(std::move(lookup.path));
                }
            }
        }

        void associate_normal_maps(
            std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
            const std::filesystem::path& dataset_root) {
            lfs::io::NormalDirCache cache(dataset_root);
            if (!cache.has_normal_dirs())
                return;
            for (const auto& cam : cameras) {
                if (!cam || cam->has_normal())
                    continue;
                if (auto lookup = cache.lookup(cam->image_name()); lookup.found()) {
                    cam->set_normal_path(std::move(lookup.path));
                }
            }
        }

        // One preprocess run per output combination so a camera that already has
        // one map never gets it overwritten while the other is generated.
        std::expected<void, lfs::Error> run_moge_estimator(
            const lfs::core::param::TrainingParameters& params,
            const std::string& images_folder,
            const std::string& depth_folder,
            const std::string& normals_folder,
            std::span<const PriorMapJob> jobs,
            const PriorMapProgress& progress) {
            using lfs::core::param::PreprocessOutputMode;
            std::size_t done_before = 0;
            for (const auto mode : {PreprocessOutputMode::Both, PreprocessOutputMode::Depth,
                                    PreprocessOutputMode::Normals}) {
                lfs::core::param::PreprocessParameters pp;
                pp.dataset_path = params.dataset.data_path;
                pp.images_folder = images_folder;
                pp.mode = mode;
                // Jobs are only cameras without a valid map; overwrite stale mismatched files.
                pp.overwrite = true;
                pp.no_download = params.safe_mode;
                pp.depth_folder = depth_folder;
                pp.normals_folder = normals_folder;
                for (const auto& job : jobs) {
                    const bool depth = !job.depth_output_path.empty();
                    const bool normal = !job.normal_output_path.empty();
                    const auto job_mode = depth && normal ? PreprocessOutputMode::Both
                                          : depth         ? PreprocessOutputMode::Depth
                                                          : PreprocessOutputMode::Normals;
                    if (job_mode == mode)
                        pp.image_paths.push_back(job.image_path);
                }
                if (pp.image_paths.empty())
                    continue;

                const std::size_t batch = pp.image_paths.size();
                const auto result = lfs::preprocessing::run_preprocess_ex(
                    pp, [&](const std::size_t done, std::size_t, const std::string_view filename) {
                        if (progress)
                            progress(done_before + done, jobs.size(), filename);
                    });
                if (!result.ok) {
                    return std::unexpected(lfs::make_error(lfs::ErrorInit{
                        .code = lfs::ErrorCode::Internal,
                        .domain = lfs::ErrorDomain::Training,
                        .user_message = "prior map generation failed",
                        .detail = result.error,
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    }));
                }
                done_before += batch;
            }
            return {};
        }

    } // namespace

    bool training_depth_priors_enabled(const lfs::core::param::OptimizationParameters& opt) {
        return opt.use_depth_loss && opt.depth_loss_weight > 0.0f;
    }

    bool training_normal_priors_enabled(const lfs::core::param::OptimizationParameters& opt) {
        return opt.use_normal_loss && opt.normal_loss_weight > 0.0f;
    }

    bool normal_auto_generate_needed(
        const bool use_normal_loss,
        const bool normal_auto_generate,
        const float normal_loss_weight,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras) {
        if (!use_normal_loss || !normal_auto_generate || normal_loss_weight <= 0.0f)
            return false;
        for (const auto& cam : cameras) {
            if (cam && cam->has_image() && !cam->has_normal())
                return true;
        }
        return false;
    }

    PriorAutoGenerateOutcome ensure_training_prior_maps(
        const lfs::core::param::TrainingParameters& params,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const PriorMapEstimator& estimator) {
        PriorAutoGenerateOutcome outcome;
        const auto& opt = params.optimization;
        const bool depth_enabled = training_depth_priors_enabled(opt);
        const bool normal_enabled = training_normal_priors_enabled(opt);
        if (!depth_enabled && !normal_enabled)
            return outcome;

        count_prior_maps(cameras, outcome);
        const bool generate_depth = depth_enabled && opt.depth_auto_generate && outcome.depth.missing > 0;
        const bool generate_normal = normal_enabled && opt.normal_auto_generate && outcome.normal.missing > 0;

        const auto log_skip = [](const std::string_view kind, const bool enabled, const bool auto_generate,
                                 const PriorMapCounts& counts) {
            if (!enabled)
                return;
            if (!auto_generate) {
                if (counts.existing > 0) {
                    LOG_INFO("{} maps available for {} training cameras (auto-generate disabled)",
                             kind, counts.existing);
                }
            } else if (counts.missing == 0) {
                LOG_INFO("{} auto-generate: found {} existing maps, skipping generation",
                         kind, counts.existing);
            }
        };
        log_skip("Depth", depth_enabled, opt.depth_auto_generate, outcome.depth);
        log_skip("Normal", normal_enabled, opt.normal_auto_generate, outcome.normal);

        if (!generate_depth && !generate_normal)
            return outcome;

        const std::string_view kind = generate_depth && generate_normal ? "Depth and normal"
                                      : generate_depth                  ? "Depth"
                                                                        : "Normal";
        const std::string_view maps = generate_depth && generate_normal ? "depth and normal maps"
                                      : generate_depth                  ? "depth maps"
                                                                        : "normal maps";
        if (params.dataset.data_path.empty()) {
            outcome.failed = true;
            outcome.warning = std::format(
                "{} auto-generate skipped because the dataset path is empty; "
                "continuing without the prior loss",
                kind);
            LOG_WARN("{}", outcome.warning);
            return outcome;
        }

        const auto dataset_root = params.dataset.data_path;
        const auto training_folder = training_images_folder(params);
        const auto source_folder = resolve_generation_images_folder(dataset_root, training_folder);
        const auto depth_dir = resolve_output_dir(dataset_root, lfs::io::DEPTH_SEARCH_FOLDERS, "depth");
        const auto normal_dir = resolve_output_dir(dataset_root, lfs::io::NORMAL_SEARCH_FOLDERS, "normals");

        std::vector<PriorMapJob> jobs;
        jobs.reserve(std::max(outcome.depth.missing, outcome.normal.missing));
        for (const auto& cam : cameras) {
            if (!cam || !cam->has_image() || cam->image_path().empty())
                continue;
            const bool needs_depth = generate_depth && !cam->has_depth();
            const bool needs_normal = generate_normal && !cam->has_normal();
            if (!needs_depth && !needs_normal)
                continue;
            const auto image_path = resolve_generation_image_path(
                *cam, dataset_root, training_folder, source_folder);
            const auto rel = relative_prior_filename(image_path, cam->image_name(), dataset_root, source_folder);
            jobs.push_back(PriorMapJob{
                .image_path = image_path,
                .depth_output_path = needs_depth ? depth_dir / rel : std::filesystem::path{},
                .normal_output_path = needs_normal ? normal_dir / rel : std::filesystem::path{},
            });
        }

        if (jobs.empty()) {
            LOG_INFO("{} auto-generate: no camera has an image to generate from", kind);
            return outcome;
        }

        outcome.attempted = true;
        const auto start = std::chrono::steady_clock::now();
        if (source_folder != training_folder) {
            LOG_INFO("{} auto-generate: generating maps for {} images from '{}' "
                     "at original resolution for {} training cameras (training folder '{}')",
                     kind, jobs.size(), source_folder, cameras.size(), training_folder);
        } else {
            LOG_INFO("{} auto-generate: generating maps for {} images from '{}' for {} training cameras",
                     kind, jobs.size(), source_folder, cameras.size());
        }

        lfs::core::events::state::DatasetLoadStarted{.path = dataset_root}.emit();
        lfs::core::events::state::DatasetLoadProgress{
            .path = dataset_root,
            .progress = 0.0f,
            .step = std::format("Generating {}", maps)}
            .emit();

        std::size_t last_logged_bucket = 0;
        const PriorMapProgress progress =
            [dataset_root, kind, maps, &last_logged_bucket](
                const std::size_t done, const std::size_t total, const std::string_view filename) {
                const std::size_t denom = total == 0 ? 1 : total;
                const float pct = 100.0f * static_cast<float>(done) / static_cast<float>(denom);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = dataset_root,
                    .progress = pct,
                    .step = std::format("Generating {} {}/{}: {}", maps, done, total, filename)}
                    .emit();
                const std::size_t bucket = static_cast<std::size_t>(pct) / 10;
                if (done == 1 || done == total || bucket > last_logged_bucket) {
                    last_logged_bucket = bucket;
                    LOG_INFO("{} auto-generate: {}/{} images ({:.0f}%) {}",
                             kind, done, total, pct, filename);
                }
            };

        const auto generated = estimator
                                   ? estimator(jobs, progress)
                                   : run_moge_estimator(params, source_folder,
                                                        depth_dir.filename().string(),
                                                        normal_dir.filename().string(), jobs, progress);
        if (generate_depth)
            associate_depth_maps(cameras, dataset_root);
        if (generate_normal)
            associate_normal_maps(cameras, dataset_root);
        count_prior_maps(cameras, outcome);

        const double elapsed_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();

        if (!generated) {
            outcome.failed = true;
            const auto mode = generate_depth && generate_normal ? "both" : generate_depth ? "depth"
                                                                                          : "normals";
            const auto skip_flags = generate_depth && generate_normal
                                        ? "--no-depth-auto-generate and --no-normal-auto-generate"
                                    : generate_depth ? "--no-depth-auto-generate"
                                                     : "--no-normal-auto-generate";
            outcome.warning = std::format(
                "{} auto-generate failed ({}). Continuing without the prior loss. "
                "Generate maps with `preprocess <dataset> --mode {}` or pass {} to skip this step.",
                kind,
                generated.error().detail().empty() ? generated.error().user_message() : generated.error().detail(),
                mode, skip_flags);
            LOG_WARN("{}", outcome.warning);
        } else {
            outcome.generated = true;
            LOG_INFO("{} auto-generate: finished {} images in {:.1f}s ({} cameras have depth maps, {} have normal maps)",
                     kind, jobs.size(), elapsed_s, outcome.depth.existing, outcome.normal.existing);
        }

        lfs::core::events::state::DatasetLoadCompleted{
            .path = dataset_root,
            .success = true,
            .error = std::nullopt,
            .num_images = jobs.size(),
            .num_points = 0}
            .emit();

        return outcome;
    }

} // namespace lfs::training
