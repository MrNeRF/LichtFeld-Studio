/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/embedded_dataset.hpp"

#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <fstream>
#include <ios>
#include <istream>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace lfs::io::project {

    namespace {

        [[nodiscard]] lfs::Error embed_error(
            const lfs::ErrorCode code,
            std::string message,
            std::string detail) {
            lfs::SmallFields fields;
            fields.add("field", "project.dataset_embed");
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        [[nodiscard]] std::filesystem::path project_root_for(
            const ProjectDocument& document) {
            if (const auto source = document.source_path();
                source && !source->empty()) {
                return source->parent_path();
            }
            return {};
        }

    } // namespace

    lfs::Result<Hash128> hash_dataset_file(const std::filesystem::path& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return embed_error(
                lfs::ErrorCode::PermissionDenied,
                "A dataset file could not be opened.", lfs::core::path_to_utf8(path));
        }
        Hash128Stream hasher;
        std::vector<char> buffer(1024 * 1024);
        while (input) {
            input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            const auto count = input.gcount();
            if (count <= 0) {
                break;
            }
            if (!hasher.update(std::span<const std::byte>(
                    reinterpret_cast<const std::byte*>(buffer.data()),
                    static_cast<std::size_t>(count)))) {
                return embed_error(
                    lfs::ErrorCode::DataLoss,
                    "A dataset file could not be hashed.", lfs::core::path_to_utf8(path));
            }
        }
        if (!input.eof() || !hasher.valid()) {
            return embed_error(
                lfs::ErrorCode::DataLoss,
                "A dataset file could not be hashed.", lfs::core::path_to_utf8(path));
        }
        return hasher.digest();
    }

    lfs::Result<std::filesystem::path>
    embedded_dataset_cache_dir(const ProjectDocument& document) {
        const auto paths = lfs::core::UserPaths::resolve();
        if (!paths) {
            return std::move(paths).error();
        }
        return paths->rootDir() / "cache" / "embedded_datasets" /
               document.project_uuid().to_string();
    }

    lfs::Result<std::optional<std::filesystem::path>>
    extract_embedded_dataset(const ProjectDocument& document,
                             const std::filesystem::path& cache,
                             const std::function<bool(float)>& progress) {
        auto manifest = document.parameters().embedded_dataset();
        if (!manifest) {
            return std::move(manifest).error();
        }
        if (!*manifest || !(*manifest)->complete || (*manifest)->entries.empty()) {
            return std::optional<std::filesystem::path>{};
        }
        std::error_code error;
        std::filesystem::create_directories(cache, error);
        if (error) {
            return embed_error(lfs::ErrorCode::PermissionDenied,
                               "The embedded dataset cache could not be created.",
                               error.message());
        }
        const auto marker = cache / ".complete";
        std::filesystem::remove(marker, error);
        const auto total_bytes = std::ranges::fold_left(
            (*manifest)->entries, std::uint64_t{0},
            [](const std::uint64_t total, const auto& entry) {
                return total + entry.bytes;
            });
        LOG_INFO(
            "Embedded dataset extraction started: {} files, {} bytes, cache {}",
            (*manifest)->entries.size(), total_bytes,
            lfs::core::path_to_utf8(cache));
        std::uint64_t completed_bytes = 0;
        std::size_t reused_files = 0;
        std::size_t extracted_files = 0;
        for (const auto& entry : (*manifest)->entries) {
            if (progress && !progress(
                                total_bytes == 0
                                    ? 0.0F
                                    : static_cast<float>(completed_bytes) /
                                          static_cast<float>(total_bytes))) {
                return embed_error(
                    lfs::ErrorCode::Cancelled,
                    "Embedded dataset extraction was canceled.",
                    "The caller requested cancellation");
            }
            const auto relative = lfs::core::utf8_to_path(entry.rel_path);
            if (relative.empty() || relative.is_absolute() ||
                relative.lexically_normal() != relative) {
                return embed_error(
                    lfs::ErrorCode::DataLoss,
                    "The embedded dataset manifest contains an unsafe path.",
                    entry.rel_path);
            }
            const auto destination = (cache / relative).lexically_normal();
            const auto source = document.find_dataset_source(entry.chunk_uuid);
            if (!source) {
                return embed_error(
                    lfs::ErrorCode::DataLoss,
                    "The embedded dataset chunk is missing.",
                    entry.chunk_uuid.to_string());
            }
            if (source->size() != entry.bytes) {
                return embed_error(
                    lfs::ErrorCode::DataLoss,
                    "The embedded dataset chunk size does not match its manifest.",
                    std::format("{} has {} bytes, manifest expected {}",
                                entry.chunk_uuid.to_string(), source->size(),
                                entry.bytes));
            }
            bool valid_existing = false;
            if (std::filesystem::is_regular_file(destination)) {
                std::error_code size_error;
                valid_existing = std::filesystem::file_size(destination, size_error) ==
                                     entry.bytes &&
                                 !size_error;
                if (valid_existing) {
                    auto hash = hash_dataset_file(destination);
                    valid_existing = hash && *hash == entry.xxh3_128;
                }
            }
            if (valid_existing) {
                completed_bytes += entry.bytes;
                ++reused_files;
                continue;
            }
            std::filesystem::create_directories(destination.parent_path(), error);
            if (error) {
                return embed_error(
                    lfs::ErrorCode::PermissionDenied,
                    "The embedded dataset cache could not be created.",
                    error.message());
            }
            auto temporary = destination;
            temporary += ".part";
            std::ofstream output;
            lfs::core::open_file_for_write(
                temporary, std::ios::binary | std::ios::trunc, output);
            if (!output) {
                return embed_error(
                    lfs::ErrorCode::PermissionDenied,
                    "An embedded dataset file could not be extracted.",
                    lfs::core::path_to_utf8(destination));
            }
            Hash128Stream hasher;
            auto copied = source->visit_stream(
                [&](std::istream& input, const std::uint64_t size) -> lfs::Result<void> {
                    std::vector<char> buffer(1024 * 1024);
                    std::uint64_t copied_bytes = 0;
                    while (input && copied_bytes < size) {
                        const auto remaining = size - copied_bytes;
                        const auto request = static_cast<std::streamsize>(
                            std::min<std::uint64_t>(remaining, buffer.size()));
                        input.read(buffer.data(), request);
                        const auto count = input.gcount();
                        if (count <= 0)
                            break;
                        const auto bytes = std::span<const std::byte>(
                            reinterpret_cast<const std::byte*>(buffer.data()),
                            static_cast<std::size_t>(count));
                        if (!hasher.update(bytes)) {
                            return lfs::Result<void>::failure(embed_error(
                                lfs::ErrorCode::DataLoss,
                                "An embedded dataset file could not be extracted.",
                                lfs::core::path_to_utf8(destination)));
                        }
                        output.write(buffer.data(), count);
                        copied_bytes += static_cast<std::uint64_t>(count);
                    }
                    if (!output || copied_bytes != size) {
                        return lfs::Result<void>::failure(embed_error(
                            lfs::ErrorCode::DataLoss,
                            "An embedded dataset file could not be extracted.",
                            lfs::core::path_to_utf8(destination)));
                    }
                    return {};
                });
            output.close();
            if (!copied || !hasher.valid() || hasher.digest() != entry.xxh3_128) {
                std::filesystem::remove(temporary, error);
                return copied ? embed_error(
                                    lfs::ErrorCode::DataLoss,
                                    "The embedded dataset file failed hash verification.",
                                    lfs::core::path_to_utf8(destination))
                              : std::move(copied).error();
            }
            std::filesystem::rename(temporary, destination, error);
            if (error) {
                std::filesystem::remove(temporary);
                return embed_error(
                    lfs::ErrorCode::PermissionDenied,
                    "The embedded dataset file could not be published.",
                    error.message());
            }
            completed_bytes += entry.bytes;
            ++extracted_files;
        }
        if (progress && !progress(1.0F)) {
            return embed_error(
                lfs::ErrorCode::Cancelled,
                "Embedded dataset extraction was canceled.",
                "The caller requested cancellation");
        }
        std::ofstream complete_marker(marker,
                                      std::ios::binary | std::ios::trunc);
        if (!complete_marker) {
            return embed_error(
                lfs::ErrorCode::PermissionDenied,
                "The embedded dataset cache could not be finalized.",
                lfs::core::path_to_utf8(marker));
        }
        LOG_INFO(
            "Embedded dataset extraction completed: {} files extracted, {} files reused, cache {}",
            extracted_files, reused_files,
            lfs::core::path_to_utf8(cache));
        return std::optional<std::filesystem::path>{cache};
    }

    lfs::Result<std::optional<std::filesystem::path>>
    extract_embedded_dataset_if_needed(
        const ProjectDocument& document,
        const std::function<bool(float)>& progress) {
        auto manifest = document.parameters().embedded_dataset();
        if (!manifest) {
            return std::move(manifest).error();
        }
        if (!*manifest || !(*manifest)->complete || (*manifest)->entries.empty()) {
            return std::optional<std::filesystem::path>{};
        }
        const auto dataset_ref = document.project().dataset_reference();
        if (!dataset_ref) {
            return std::move(dataset_ref).error();
        }
        if (!*dataset_ref) {
            return std::optional<std::filesystem::path>{};
        }
        const auto external = resolve_path_reference(
            document.references(), project_root_for(document), **dataset_ref, {});
        if (external && std::filesystem::exists(*external)) {
            return std::optional<std::filesystem::path>{};
        }
        // Resolve cache dir path from UserPaths
        const auto cache = embedded_dataset_cache_dir(document);
        if (!cache) {
            return std::move(cache).error();
        }
        return extract_embedded_dataset(document, *cache, progress);
    }

} // namespace lfs::io::project
