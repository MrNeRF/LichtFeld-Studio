/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/licht_command.hpp"
#include "core/native_fs.hpp"

#include "app/terminal_progress_bar.hpp"
#include "core/path_utils.hpp"
#include "io/embedded_dataset.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/project_chapters.hpp"
#include "io/project_document.hpp"
#include "io/project_inspector.hpp"
#include "io/project_operations.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <format>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::app {

    namespace {

        std::string dataset_type_name(const lfs::io::DatasetType type) {
            switch (type) {
            case lfs::io::DatasetType::COLMAP:
                return "COLMAP";
            case lfs::io::DatasetType::Transforms:
                return "Transforms";
            case lfs::io::DatasetType::Unknown:
                return "Unknown";
            }
            return "Unknown";
        }

        std::string dataset_layouts(const std::filesystem::path& root) {
            std::string layouts;
            for (const auto& candidate : lfs::io::get_colmap_search_paths(root)) {
                if (!layouts.empty()) {
                    layouts += ", ";
                }
                const auto relative = candidate.lexically_relative(root);
                layouts += relative.empty() || relative == "."
                               ? "flat root"
                               : lfs::core::path_to_generic_utf8(relative);
            }
            std::string transforms;
            for (const auto marker : lfs::io::TRANSFORMS_DATASET_MARKERS) {
                if (!transforms.empty()) {
                    transforms += ", ";
                }
                transforms += marker;
            }
            return std::format("COLMAP layouts: {}; Transforms markers: {}",
                               layouts, transforms);
        }

        int print_error(const std::string_view message) {
            std::println(stderr, "Error: {}", message);
            return 1;
        }

        int print_io_error(const lfs::Error& error) {
            return print_error(error.user_message());
        }

        bool complete_embedded_dataset(const lfs::io::project::ProjectDocument& document) {
            auto embedded = document.parameters().embedded_dataset();
            return embedded && *embedded && (*embedded)->complete &&
                   !(*embedded)->entries.empty();
        }

        lfs::Result<bool> matches_embedded_dataset_sources(
            const std::filesystem::path& dataset,
            const lfs::io::project::EmbeddedDatasetManifest& manifest) {
            for (const auto& entry : manifest.entries) {
                const auto source = dataset / lfs::core::utf8_to_path(entry.rel_path);
                std::error_code error;
                if (!lfs::core::native_fs::is_regular_file(source, error) || error) {
                    return false;
                }
                const auto size = lfs::core::native_fs::file_size(source, error);
                if (error || size != entry.bytes) {
                    return false;
                }
                auto hash = lfs::io::project::hash_dataset_file(source);
                if (!hash) {
                    return std::move(hash).error();
                }
                if (*hash != entry.xxh3_128) {
                    return false;
                }
            }
            return true;
        }

        std::string format_size(const std::uint64_t bytes) {
            constexpr std::array units{"bytes", "KiB", "MiB", "GiB", "TiB"};
            auto value = static_cast<double>(bytes);
            std::size_t unit = 0;
            while (value >= 1024.0 && unit + 1 < units.size()) {
                value /= 1024.0;
                ++unit;
            }
            return unit == 0 ? std::format("{} bytes", bytes) : std::format("{:.1f} {}", value, units[unit]);
        }

        std::vector<lfs::io::project::ProjectInspectorCheckpoint>
        newest_first(std::vector<lfs::io::project::ProjectInspectorCheckpoint> checkpoints) {
            std::erase_if(checkpoints, [](const auto& checkpoint) { return !checkpoint.retained; });
            std::ranges::stable_sort(checkpoints, std::ranges::greater{},
                                     &lfs::io::project::ProjectInspectorCheckpoint::iteration);
            return checkpoints;
        }

        int list_checkpoints(const std::filesystem::path& project) {
            auto details = lfs::io::project::inspect_project_details(project);
            if (!details) {
                return print_io_error(details.error());
            }
            const auto checkpoints = newest_first(details->retained_checkpoints);
            if (checkpoints.empty()) {
                std::println("The project has no checkpoints.");
            } else {
                std::println("{:>10}  {:>10}  {:<36}", "Iteration", "Gaussians", "Checkpoint");
                for (const auto& checkpoint : checkpoints) {
                    std::println("{:>10}  {:>10}  {}{}", checkpoint.iteration, checkpoint.gaussians,
                                 checkpoint.instance_uuid.to_string(),
                                 checkpoint.binds_scene_graph ? "  resume" : "");
                }
            }
            std::println("Project: {} ({})", lfs::core::path_to_utf8(project),
                         format_size(details->card.physical_file_size));
            return 0;
        }

        int remove_checkpoints(const std::filesystem::path& project,
                               const io::args::LichtMode& mode) {
            auto details = lfs::io::project::inspect_project_details(project);
            if (!details) {
                return print_io_error(details.error());
            }
            const auto checkpoints = newest_first(details->retained_checkpoints);
            std::vector<lfs::io::project::ProjectInspectorCheckpoint> removed;
            if (mode.action == io::args::LichtMode::Action::PRUNE_CHECKPOINTS) {
                std::size_t kept = 0;
                for (const auto& checkpoint : checkpoints) {
                    if (!checkpoint.binds_scene_graph && kept++ >= mode.keep_checkpoints) {
                        removed.push_back(checkpoint);
                    }
                }
            } else {
                for (const auto iteration : mode.checkpoint_iterations) {
                    const auto before = removed.size();
                    for (const auto& checkpoint : checkpoints) {
                        if (checkpoint.iteration == iteration) {
                            removed.push_back(checkpoint);
                        }
                    }
                    if (removed.size() == before) {
                        return print_error(std::format("The project has no checkpoint at iteration {}.", iteration));
                    }
                }
            }
            if (removed.empty()) {
                std::println("No checkpoints to remove.");
                std::println("Project: {} ({})", lfs::core::path_to_utf8(project),
                             format_size(details->card.physical_file_size));
                return 0;
            }

            lfs::io::project::ProjectReduceSelection selection;
            for (const auto& checkpoint : removed) {
                selection.checkpoints.push_back(checkpoint.instance_uuid);
                if (checkpoint.binds_scene_graph) {
                    std::println("Removing the resume checkpoint at iteration {}: the trained model stays in the "
                                 "project, but training can no longer resume from it.",
                                 checkpoint.iteration);
                }
            }
            TerminalProgressBar bar("Removing checkpoints");
            auto reduced = lfs::io::project::reduce_size(
                project, false, false,
                [&bar](const float value, const std::string& stage) { bar.report(value, stage); },
                {}, selection);
            if (!reduced) {
                bar.abort();
                return print_io_error(reduced.error());
            }
            bar.complete();
            std::string iterations;
            for (const auto& checkpoint : removed) {
                iterations += std::format("{}{}", iterations.empty() ? "" : ", ", checkpoint.iteration);
            }
            const auto plural = reduced->checkpoints_removed == 1 ? "" : "s";
            std::println("Removed {} checkpoint{} (iteration{} {}), reclaimed {}.", reduced->checkpoints_removed,
                         plural, plural, iterations, format_size(reduced->bytes_reclaimed));
            std::println("Project: {} ({})", lfs::core::path_to_utf8(project),
                         format_size(reduced->card.physical_file_size));
            std::println("Recovery copy of the previous project ({}): {}",
                         format_size(details->card.physical_file_size),
                         lfs::core::path_to_utf8(reduced->recovery_copy));
            return 0;
        }

    } // namespace

    int run_licht_command(const io::args::LichtMode& mode) {
        const auto project = std::filesystem::absolute(mode.project_path).lexically_normal();
        switch (mode.action) {
        case io::args::LichtMode::Action::LIST_CHECKPOINTS:
            return list_checkpoints(project);
        case io::args::LichtMode::Action::PRUNE_CHECKPOINTS:
        case io::args::LichtMode::Action::REMOVE_CHECKPOINTS:
            return remove_checkpoints(project, mode);
        case io::args::LichtMode::Action::EMBED:
            break;
        }
        auto document = lfs::io::project::ProjectDocument::open(project);
        if (!document) {
            return print_io_error(document.error());
        }

        auto existing_reference = document->project().dataset_reference();
        if (!existing_reference) {
            return print_io_error(existing_reference.error());
        }
        const bool had_dataset_reference = existing_reference->has_value();
        std::optional<lfs::io::project::ReferenceRecord> previous_reference;
        std::optional<std::filesystem::path> previous_dataset;
        if (*existing_reference) {
            auto record = document->references().find(**existing_reference);
            if (!record) {
                return print_io_error(record.error());
            }
            previous_reference = *record;
            auto resolved = lfs::io::project::resolve_path_reference(
                document->references(), project.parent_path(), **existing_reference);
            if (resolved) {
                previous_dataset = std::filesystem::absolute(*resolved).lexically_normal();
            }
        }

        std::filesystem::path dataset;
        if (mode.dataset_path) {
            dataset = *mode.dataset_path;
        } else {
            if (!*existing_reference) {
                return print_error(
                    "The project has no dataset reference. Specify --embed <dataset>.");
            }
            auto resolved = lfs::io::project::resolve_path_reference(
                document->references(), project.parent_path(), **existing_reference);
            if (!resolved || !lfs::core::native_fs::is_directory(*resolved)) {
                return print_error(
                    "The referenced dataset folder is unavailable. Specify --embed <dataset>.");
            }
            dataset = *resolved;
        }

        if (!dataset.is_absolute()) {
            dataset = lfs::core::native_fs::current_path() / dataset;
        }
        dataset = std::filesystem::absolute(dataset).lexically_normal();
        std::error_code equivalence_error;
        const bool same_dataset = previous_dataset &&
                                  lfs::core::native_fs::equivalent(
                                      *previous_dataset, dataset, equivalence_error) &&
                                  !equivalence_error;
        if (!lfs::core::native_fs::is_directory(dataset)) {
            return print_error(std::format("Dataset folder not found: {}",
                                           lfs::core::path_to_utf8(dataset)));
        }

        const auto type = lfs::io::Loader::getDatasetType(dataset);
        if (type != lfs::io::DatasetType::COLMAP &&
            type != lfs::io::DatasetType::Transforms) {
            return print_error(std::format(
                "Unsupported dataset type '{}'. Accepted kinds are COLMAP and Transforms ({}).",
                dataset_type_name(type), dataset_layouts(dataset)));
        }

        bool reference_matches_dataset = false;
        if (same_dataset && previous_reference) {
            auto fingerprint = lfs::io::project::fingerprint_path(dataset);
            if (!fingerprint) {
                return print_io_error(fingerprint.error());
            }
            reference_matches_dataset = *fingerprint == previous_reference->fingerprint;
        }

        auto reference = lfs::io::project::set_dataset_reference(
            project, dataset, mode.dataset_path.has_value());
        if (!reference) {
            return print_io_error(reference.error());
        }
        std::println("Dataset reference changed: {}",
                     reference->content_replaced ? "yes" : "no");
        if (had_dataset_reference && same_dataset && reference_matches_dataset &&
            !reference->content_replaced &&
            complete_embedded_dataset(*document)) {
            auto manifest = document->parameters().embedded_dataset();
            if (!manifest || !*manifest) {
                return print_error("The embedded dataset manifest could not be read.");
            }
            auto source_matches = matches_embedded_dataset_sources(dataset, **manifest);
            if (!source_matches) {
                return print_io_error(source_matches.error());
            }
            if (*source_matches) {
                std::uint64_t images = 0;
                std::uint64_t normals = 0;
                std::uint64_t sparse = 0;
                std::uint64_t bytes = 0;
                for (const auto& entry : (*manifest)->entries) {
                    images += entry.kind == "image";
                    normals += entry.kind == "normal";
                    sparse += entry.kind == "sparse";
                    bytes += entry.bytes;
                }
                std::println("Dataset is already embedded: {} images, {} normals, {} sparse files, {} bytes.",
                             images, normals, sparse, bytes);
                std::println("Project: {}", lfs::core::path_to_utf8(project));
                return 0;
            }
        }

        TerminalProgressBar bar("Embedding");
        auto embedded = lfs::io::project::embed_dataset_file(
            project, [&bar](const float value, const std::string& stage) {
                return bar.report(value, stage);
            });
        if (!embedded) {
            bar.abort();
            return print_io_error(embedded.error());
        }
        bar.complete();
        std::println(
            "Embedded dataset: {} images, {} masks, {} depths, {} normals, {} sparse files, {} bytes.",
            embedded->images_embedded, embedded->masks_embedded,
            embedded->depths_embedded, embedded->normals_embedded,
            embedded->sparse_embedded, embedded->bytes_embedded);
        std::println("Project: {}", lfs::core::path_to_utf8(project));
        return 0;
    }

} // namespace lfs::app
