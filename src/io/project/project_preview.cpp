/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/project_document.hpp"

#include "core/error.hpp"
#include "core/guarded_task.hpp"
#include "core/image_io.hpp"
#include "core/path_utils.hpp"
#include "io/filesystem_utils.hpp"

#include <stb_image_write.h>

#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace lfs::io::project {
    namespace {

        lfs::Error preview_error(const lfs::ErrorCode code,
                                 std::string message,
                                 std::string detail,
                                 const std::string_view field = {}) {
            lfs::SmallFields fields;
            if (!field.empty()) {
                fields.add("field", field);
            }
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

        void png_write_callback(void* context, void* bytes, const int size) {
            auto& destination = *static_cast<std::vector<std::byte>*>(context);
            const auto* begin = static_cast<const std::byte*>(bytes);
            destination.insert(destination.end(), begin, begin + size);
        }

        std::filesystem::path dataset_images_directory(
            const lfs::core::param::DatasetConfig& dataset) {
            const auto folder = dataset.images.empty() ? std::string{"images"}
                                                       : dataset.images;
            auto folder_path = lfs::core::utf8_to_path(folder);
            if (folder_path.is_absolute()) {
                return folder_path;
            }
            return dataset.data_path / folder_path;
        }

        struct LoadedImage {
            unsigned char* pixels = nullptr;

            LoadedImage() = default;
            LoadedImage(const LoadedImage&) = delete;
            LoadedImage& operator=(const LoadedImage&) = delete;

            ~LoadedImage() {
                if (pixels) {
                    lfs::core::free_image(pixels);
                }
            }
        };

    } // namespace

    std::optional<std::filesystem::path>
    first_dataset_image(const lfs::core::param::DatasetConfig& dataset) {
        if (dataset.data_path.empty()) {
            return std::nullopt;
        }
        const auto images_dir = dataset_images_directory(dataset);
        std::error_code ec;
        if (!std::filesystem::is_directory(images_dir, ec) || ec) {
            return std::nullopt;
        }
        std::optional<std::filesystem::path> first;
        std::string first_key;
        std::filesystem::directory_iterator it(
            images_dir,
            std::filesystem::directory_options::skip_permission_denied,
            ec);
        if (ec) {
            return std::nullopt;
        }
        for (; !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            std::error_code file_ec;
            if (!it->is_regular_file(file_ec) || file_ec) {
                continue;
            }
            const auto path = it->path();
            if (!lfs::io::is_image_file(path)) {
                continue;
            }
            auto key = path.filename().generic_string();
            if (!first || key < first_key) {
                first = std::move(path);
                first_key = std::move(key);
            }
        }
        return first;
    }

    std::optional<std::filesystem::path>
    first_dataset_image(const ProjectChapter& project,
                        const ReferencesChapter& references,
                        const ParametersChapter& parameters,
                        const std::filesystem::path& project_root) {
        auto snapshot = parameters.snapshot();
        if (!snapshot) {
            return std::nullopt;
        }
        auto dataset = snapshot->dataset;
        if (dataset.data_path.empty()) {
            auto dataset_reference = project.dataset_reference();
            if (dataset_reference && *dataset_reference) {
                if (auto resolved = resolve_path_reference(
                        references, project_root, **dataset_reference)) {
                    dataset.data_path = std::move(*resolved);
                }
            }
        }
        return first_dataset_image(dataset);
    }

    lfs::Result<std::vector<std::byte>>
    dataset_preview_png(const std::filesystem::path& first_image, int max_size) {
        if (first_image.empty()) {
            return preview_error(
                lfs::ErrorCode::InvalidArgument,
                "The dataset preview image path is empty.",
                "dataset_preview_png requires a source image",
                "preview.path");
        }
        if (max_size <= 0) {
            max_size = 512;
        }

        LoadedImage image;
        int width = 0;
        int height = 0;
        int channels = 0;
        try {
            std::tie(image.pixels, width, height, channels) =
                lfs::core::load_image(first_image, -1, max_size);
        } catch (...) {
            return lfs::core::detail::normalize_current_exception(
                lfs::core::TaskContext{
                    .name = "io.project.dataset_preview_png",
                    .domain = lfs::ErrorDomain::IO,
                    .operation_id = lfs::OperationId::generate(),
                    .site = LFS_SOURCE_SITE_CURRENT(),
                });
        }
        if (!image.pixels || width <= 0 || height <= 0 || channels < 1 ||
            channels > 4) {
            return preview_error(
                lfs::ErrorCode::DataLoss,
                "The dataset preview image is empty or has an invalid layout.",
                std::format("{}x{}x{}", width, height, channels),
                "preview.image");
        }

        std::vector<std::byte> png;
        const int stride = width * channels;
        if (!stbi_write_png_to_func(
                png_write_callback, &png, width, height, channels,
                image.pixels, stride) ||
            png.empty()) {
            return preview_error(
                lfs::ErrorCode::Unavailable,
                "The dataset preview could not be encoded.",
                "stbi_write_png_to_func failed",
                "preview.png");
        }
        return png;
    }

} // namespace lfs::io::project
