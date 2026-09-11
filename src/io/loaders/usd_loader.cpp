/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "usd_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "formats/nurec_usdz.hpp"
#include "formats/usd.hpp"
#include "io/error.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <string_view>

namespace lfs::io {

    using lfs::core::Device;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    namespace {
        std::string_view project_error_message(const lfs::Error& error) {
            return error.user_message().empty() ? error.detail()
                                                : error.user_message();
        }
    } // namespace

    Result<LoadResult> USDLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        constexpr std::string_view kNurecLoaderName = "NuRec USDZ";

        LOG_TIMER("USD Loading");
        const auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading USD gaussian file...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "USD file does not exist",
                              path);
        }

        if (!std::filesystem::is_regular_file(path)) {
            return make_error(ErrorCode::NOT_A_FILE,
                              "Path is not a regular file",
                              path);
        }

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for USD: {}", lfs::core::path_to_utf8(path));

            std::expected<void, std::string> validation_result = std::unexpected(std::string{"USD validation not started"});
            std::string validation_loader = name();

            if (extension == ".usdz") {
                const auto nurec_hint = is_nurec_usdz(path);
                if (nurec_hint && *nurec_hint) {
                    validation_result = validate_nurec_usdz(path);
                    if (validation_result) {
                        validation_loader = std::string(kNurecLoaderName);
                    }
                } else {
                    validation_result = validate_usd(path);
                    if (!validation_result && !nurec_hint) {
                        auto nurec_validation = validate_nurec_usdz(path);
                        if (nurec_validation) {
                            validation_result = std::move(nurec_validation);
                            validation_loader = std::string(kNurecLoaderName);
                        }
                    }
                }
            } else {
                validation_result = validate_usd(path);
            }

            if (!validation_result) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  std::format("Invalid USD gaussian file: {}", validation_result.error()),
                                  path);
            }

            if (options.progress) {
                options.progress(100.0f, "USD validation complete");
            }

            LoadResult result;
            result.data = std::shared_ptr<SplatData>{};
            result.scene_center = Tensor::zeros({3}, Device::CPU);
            result.loader_used = validation_loader;
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            result.warnings = {};
            return result;
        }

        std::string loader_used = name();
        std::expected<SplatData, std::string> splat_result = std::unexpected(std::string{"USD loader not initialized"});
        std::optional<double> usd_meters_per_unit;

        if (extension == ".usdz") {
            const auto nurec_hint = is_nurec_usdz(path);
            if (nurec_hint && *nurec_hint) {
                if (options.progress) {
                    options.progress(50.0f, "Parsing NuRec USDZ...");
                }

                loader_used = std::string(kNurecLoaderName);
                splat_result = load_nurec_usdz(path);
                if (!splat_result) {
                    return make_error(ErrorCode::CORRUPTED_DATA,
                                      std::format("Failed to load USD gaussian file: {}", splat_result.error()),
                                      path);
                }
            } else {
                if (options.progress) {
                    options.progress(50.0f, "Parsing OpenUSD data...");
                }

                auto openusd_result = load_usd_project_units(path);
                if (openusd_result) {
                    usd_meters_per_unit = openusd_result->meters_per_unit;
                    splat_result = std::move(openusd_result->data);
                } else if (!nurec_hint) {
                    LOG_INFO("USDZ archive inspection failed for {}: {}. Trying NuRec fallback after OpenUSD failure.",
                             lfs::core::path_to_utf8(path), nurec_hint.error());
                    if (options.progress) {
                        options.progress(75.0f, "Parsing NuRec USDZ...");
                    }

                    auto nurec_result = load_nurec_usdz(path);
                    if (nurec_result) {
                        loader_used = std::string(kNurecLoaderName);
                        splat_result = std::move(nurec_result);
                    } else {
                        return make_error(ErrorCode::CORRUPTED_DATA,
                                          std::format("Failed to load USD gaussian file: OpenUSD: {}; NuRec USDZ: {}",
                                                      project_error_message(openusd_result.error()),
                                                      nurec_result.error()),
                                          path);
                    }
                } else {
                    return make_error(ErrorCode::CORRUPTED_DATA,
                                      std::format("Failed to load USD gaussian file: {}",
                                                  project_error_message(openusd_result.error())),
                                      path);
                }
            }
        } else {
            if (options.progress) {
                options.progress(50.0f, "Parsing OpenUSD ParticleField...");
            }

            auto openusd_result = load_usd_project_units(path);
            if (!openusd_result) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Failed to load USD gaussian file: {}",
                                              project_error_message(openusd_result.error())),
                                  path);
            }
            usd_meters_per_unit = openusd_result->meters_per_unit;
            splat_result = std::move(openusd_result->data);
        }

        if (options.progress) {
            options.progress(100.0f, "USD loading complete");
        }

        const auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);

        LoadResult result{
            .data = std::make_shared<SplatData>(std::move(*splat_result)),
            .scene_center = Tensor::zeros({3}, Device::CPU),
            .loader_used = loader_used,
            .load_time = load_time,
            .warnings = {},
            .georeference =
                usd_meters_per_unit
                    ? std::optional<ImportGeoreference>(
                          ImportGeoreference{
                              .crs = std::nullopt,
                              .world_origin = {},
                              .world_unit_scale = *usd_meters_per_unit,
                              .world_origin_provenance =
                                  ImportWorldOriginProvenance::Import,
                          })
                    : std::nullopt};

        LOG_INFO("USD gaussian file loaded successfully in {}ms", load_time.count());
        return result;
    }

    bool USDLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || std::filesystem::is_directory(path)) {
            return false;
        }

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz";
    }

    std::string USDLoader::name() const {
        return "OpenUSD";
    }

    std::vector<std::string> USDLoader::supportedExtensions() const {
        return {".usd", ".USD", ".usda", ".USDA", ".usdc", ".USDC", ".usdz", ".USDZ"};
    }

    int USDLoader::priority() const {
        return 17;
    }

} // namespace lfs::io
