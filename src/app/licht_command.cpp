/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/licht_command.hpp"

#include "app/terminal_progress_bar.hpp"
#include "core/path_utils.hpp"
#include "io/embedded_dataset.hpp"
#include "io/filesystem_utils.hpp"
#include "io/loader.hpp"
#include "io/project_chapters.hpp"
#include "io/project_document.hpp"
#include "io/project_operations.hpp"

#include <filesystem>
#include <format>
#include <optional>
#include <print>
#include <string>
#include <string_view>

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
                const auto source = dataset / std::filesystem::path(entry.rel_path);
                std::error_code error;
                if (!std::filesystem::is_regular_file(source, error) || error) {
                    return false;
                }
                const auto size = std::filesystem::file_size(source, error);
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

    } // namespace

    int run_licht_command(const core::args::LichtMode& mode) {
        const auto project = std::filesystem::absolute(mode.project_path).lexically_normal();
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
            if (!resolved || !std::filesystem::is_directory(*resolved)) {
                return print_error(
                    "The referenced dataset folder is unavailable. Specify --embed <dataset>.");
            }
            dataset = *resolved;
        }

        if (!dataset.is_absolute()) {
            dataset = std::filesystem::current_path() / dataset;
        }
        dataset = std::filesystem::absolute(dataset).lexically_normal();
        std::error_code equivalence_error;
        const bool same_dataset = previous_dataset &&
                                  std::filesystem::equivalent(
                                      *previous_dataset, dataset, equivalence_error) &&
                                  !equivalence_error;
        if (!std::filesystem::is_directory(dataset)) {
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
