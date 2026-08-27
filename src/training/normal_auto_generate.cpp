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

#include <chrono>
#include <expected>
#include <format>
#include <optional>
#include <string_view>
#include <utility>

namespace lfs::training {
    namespace {

        constexpr std::string_view kOriginalImagesFolder = "images";

        [[nodiscard]] bool camera_needs_normal_map(const lfs::core::Camera& cam) {
            return cam.has_image() && !cam.has_normal();
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

        [[nodiscard]] std::filesystem::path resolve_normals_output_dir(
            const std::filesystem::path& dataset_root) {
            const auto first = dataset_root / lfs::io::NORMAL_SEARCH_FOLDERS.front();
            if (lfs::io::safe_is_directory(first))
                return first;
            return dataset_root / "normals";
        }

        [[nodiscard]] std::filesystem::path relative_normal_filename(
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

        std::expected<void, lfs::Error> run_moge_estimator(
            const lfs::core::param::TrainingParameters& params,
            const std::string& images_folder,
            const std::string& normals_folder,
            std::span<const NormalAutoGenerateJob> jobs,
            const NormalGenerateProgress& progress) {
            lfs::core::param::PreprocessParameters pp;
            pp.dataset_path = params.dataset.data_path;
            pp.images_folder = images_folder;
            pp.mode = lfs::core::param::PreprocessOutputMode::Normals;
            // Jobs are only cameras without a valid map; overwrite stale mismatched files.
            pp.overwrite = true;
            pp.no_download = params.safe_mode;
            pp.normals_folder = normals_folder;
            pp.image_paths.reserve(jobs.size());
            for (const auto& job : jobs)
                pp.image_paths.push_back(job.image_path);

            const auto result = lfs::preprocessing::run_preprocess_ex(pp, progress);
            if (!result.ok) {
                return std::unexpected(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::Internal,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "normal map generation failed",
                    .detail = result.error,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            }
            return {};
        }

    } // namespace

    bool normal_auto_generate_needed(
        const bool use_normal_loss,
        const bool normal_auto_generate,
        const float normal_loss_weight,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras) {
        if (!use_normal_loss || !normal_auto_generate || normal_loss_weight <= 0.0f)
            return false;
        for (const auto& cam : cameras) {
            if (cam && camera_needs_normal_map(*cam))
                return true;
        }
        return false;
    }

    NormalAutoGenerateOutcome ensure_training_normal_maps(
        const lfs::core::param::TrainingParameters& params,
        const std::span<const std::shared_ptr<lfs::core::Camera>> cameras,
        const NormalEstimator& estimator) {
        NormalAutoGenerateOutcome outcome;
        const auto& opt = params.optimization;
        if (!opt.use_normal_loss || opt.normal_loss_weight <= 0.0f)
            return outcome;

        for (const auto& cam : cameras) {
            if (!cam || !cam->has_image())
                continue;
            if (cam->has_normal())
                ++outcome.existing_count;
            else
                ++outcome.missing_count;
        }

        if (!opt.normal_auto_generate) {
            if (outcome.existing_count > 0) {
                LOG_INFO("Normal maps available for {} training cameras (auto-generate disabled)",
                         outcome.existing_count);
            }
            return outcome;
        }

        if (outcome.missing_count == 0) {
            LOG_INFO("Normal auto-generate: found {} existing maps, skipping generation",
                     outcome.existing_count);
            return outcome;
        }

        if (params.dataset.data_path.empty()) {
            outcome.failed = true;
            outcome.warning =
                "Normal auto-generate skipped because the dataset path is empty; "
                "continuing without the prior normal loss";
            LOG_WARN("{}", outcome.warning);
            return outcome;
        }

        const auto dataset_root = params.dataset.data_path;
        const auto training_folder = training_images_folder(params);
        const auto source_folder = resolve_generation_images_folder(dataset_root, training_folder);
        const auto output_dir = resolve_normals_output_dir(dataset_root);

        std::vector<NormalAutoGenerateJob> jobs;
        jobs.reserve(outcome.missing_count);
        for (const auto& cam : cameras) {
            if (!cam || !camera_needs_normal_map(*cam) || cam->image_path().empty())
                continue;
            const auto image_path = resolve_generation_image_path(
                *cam, dataset_root, training_folder, source_folder);
            jobs.push_back(NormalAutoGenerateJob{
                .image_path = image_path,
                .output_path = output_dir / relative_normal_filename(
                                                image_path, cam->image_name(), dataset_root, source_folder),
            });
        }

        if (jobs.empty()) {
            LOG_INFO("Normal auto-generate: found {} existing maps, skipping generation",
                     outcome.existing_count);
            associate_normal_maps(cameras, dataset_root);
            return outcome;
        }

        outcome.attempted = true;
        const auto start = std::chrono::steady_clock::now();
        if (source_folder != training_folder) {
            LOG_INFO("Normal auto-generate: generating {} missing maps ({} existing) from '{}' "
                     "at original resolution for {} training cameras (training folder '{}')",
                     jobs.size(), outcome.existing_count, source_folder, cameras.size(),
                     training_folder);
        } else {
            LOG_INFO("Normal auto-generate: generating {} missing maps ({} existing) from '{}' "
                     "for {} training cameras",
                     jobs.size(), outcome.existing_count, source_folder, cameras.size());
        }

        lfs::core::events::state::DatasetLoadStarted{.path = dataset_root}.emit();
        lfs::core::events::state::DatasetLoadProgress{
            .path = dataset_root,
            .progress = 0.0f,
            .step = "Generating normal maps"}
            .emit();

        std::size_t last_logged_bucket = 0;
        const NormalGenerateProgress progress =
            [dataset_root, &last_logged_bucket](
                const std::size_t done, const std::size_t total, const std::string_view filename) {
                const std::size_t denom = total == 0 ? 1 : total;
                const float pct = 100.0f * static_cast<float>(done) / static_cast<float>(denom);
                lfs::core::events::state::DatasetLoadProgress{
                    .path = dataset_root,
                    .progress = pct,
                    .step = std::format("Generating normals {}/{}: {}", done, total, filename)}
                    .emit();
                const std::size_t bucket = static_cast<std::size_t>(pct) / 10;
                if (done == 1 || done == total || bucket > last_logged_bucket) {
                    last_logged_bucket = bucket;
                    LOG_INFO("Normal auto-generate: {}/{} images ({:.0f}%) {}",
                             done, total, pct, filename);
                }
            };

        const auto generated = estimator
                                   ? estimator(jobs, progress)
                                   : run_moge_estimator(
                                         params, source_folder, output_dir.filename().string(), jobs, progress);
        associate_normal_maps(cameras, dataset_root);

        outcome.existing_count = 0;
        outcome.missing_count = 0;
        for (const auto& cam : cameras) {
            if (!cam || !cam->has_image())
                continue;
            if (cam->has_normal())
                ++outcome.existing_count;
            else
                ++outcome.missing_count;
        }

        const double elapsed_s = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();

        if (!generated) {
            outcome.failed = true;
            outcome.warning = std::format(
                "Normal auto-generate failed ({}). Continuing without the prior normal loss. "
                "Generate maps with `preprocess <dataset> --mode normals` or pass "
                "--no-normal-auto-generate to skip this step.",
                generated.error().detail().empty() ? generated.error().user_message() : generated.error().detail());
            LOG_WARN("{}", outcome.warning);
        } else {
            outcome.generated = true;
            LOG_INFO("Normal auto-generate: finished {} maps in {:.1f}s ({} cameras now have maps)",
                     jobs.size(), elapsed_s, outcome.existing_count);
        }

        lfs::core::events::state::DatasetLoadCompleted{
            .path = dataset_root,
            .success = true,
            .error = std::nullopt,
            .num_images = outcome.existing_count,
            .num_points = 0}
            .emit();

        return outcome;
    }

} // namespace lfs::training
