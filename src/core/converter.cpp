/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/converter.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include <cctype>
#include <iostream>
#include <print>

namespace lfs::core {

    namespace {

        constexpr size_t SH_CHANNELS = 3;
        constexpr const char* VALID_EXTENSIONS[] = {".ply", ".sog", ".spz", ".resume"};

        enum class OverwriteChoice { YES,
                                     NO,
                                     ALL };

        OverwriteChoice askOverwrite(const std::filesystem::path& path) {
            std::print("File exists: {}\nOverwrite? [y]es / [n]o / [a]ll: ", path.filename().string());
            std::string input;
            if (!std::getline(std::cin, input) || input.empty()) {
                return OverwriteChoice::NO;
            }
            const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(input[0])));
            if (c == 'y')
                return OverwriteChoice::YES;
            if (c == 'a')
                return OverwriteChoice::ALL;
            return OverwriteChoice::NO;
        }

        void truncateSHDegree(SplatData& splat, const int degree) {
            if (degree < 0 || degree >= splat.get_max_sh_degree())
                return;

            if (degree == 0) {
                splat.shN() = Tensor{};
            } else {
                const size_t keep = static_cast<size_t>((degree + 1) * (degree + 1) - 1);
                auto& shN = splat.shN();
                if (shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > keep) {
                    const auto slice_end = static_cast<int64_t>(shN.ndim() == 3 ? keep : keep * SH_CHANNELS);
                    shN = shN.slice(1, 0, slice_end).contiguous();
                }
            }
            splat.set_max_sh_degree(degree);
            splat.set_active_sh_degree(degree);
        }

        std::vector<std::filesystem::path> getInputFiles(const std::filesystem::path& path) {
            std::vector<std::filesystem::path> files;
            if (std::filesystem::is_directory(path)) {
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    if (!entry.is_regular_file())
                        continue;
                    const auto ext = entry.path().extension().string();
                    for (const auto* valid : VALID_EXTENSIONS) {
                        if (ext == valid) {
                            files.push_back(entry.path());
                            break;
                        }
                    }
                }
            } else {
                files.push_back(path);
            }
            return files;
        }

        const char* getFormatExtension(const param::OutputFormat format) {
            switch (format) {
            case param::OutputFormat::PLY: return ".ply";
            case param::OutputFormat::SOG: return ".sog";
            case param::OutputFormat::SPZ: return ".spz";
            case param::OutputFormat::HTML: return ".html";
            }
            return ".ply";
        }

        std::filesystem::path generateOutputPath(
            const std::filesystem::path& input,
            const std::filesystem::path& output_template,
            const param::OutputFormat format) {

            const auto ext = getFormatExtension(format);
            const auto cwd = std::filesystem::current_path();
            // Use proper path operations for Unicode handling
            const auto converted_name = input.stem().string() + "_converted" + ext;

            if (output_template.empty()) {
                return cwd / converted_name;
            }

            if (std::filesystem::is_directory(output_template)) {
                const auto dir = output_template.is_absolute() ? output_template : cwd / output_template;
                return dir / converted_name;
            }

            auto out = output_template;
            if (out.extension().empty()) {
                // Use path concatenation for proper Unicode handling
                out += ext;
            }
            return out.is_absolute() ? out : cwd / out;
        }

        bool convertFile(
            const std::filesystem::path& input,
            const std::filesystem::path& output,
            const param::ConvertParameters& params) {

            std::println("Converting: {} -> {}", lfs::core::path_to_utf8(input), lfs::core::path_to_utf8(output));

            const auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(input);
            if (!load_result) {
                LOG_ERROR("Load failed: {}", load_result.error().format());
                std::println(stderr, "  Error: {}", load_result.error().message);
                return false;
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&load_result->data);
            if (!splat_ptr || !*splat_ptr) {
                LOG_ERROR("Not a splat file: {}", lfs::core::path_to_utf8(input));
                std::println(stderr, "  Error: not a splat file");
                return false;
            }

            auto splat = std::move(*splat_ptr);
            std::println("  Loaded {} gaussians, SH degree {}", splat->size(), splat->get_max_sh_degree());

            if (params.sh_degree >= 0 && params.sh_degree < splat->get_max_sh_degree()) {
                truncateSHDegree(*splat, params.sh_degree);
                std::println("  Truncated to SH degree {}", params.sh_degree);
            }

            lfs::io::Result<void> result;
            switch (params.format) {
            case param::OutputFormat::PLY:
                result = lfs::io::save_ply(*splat, {.output_path = output, .binary = true});
                break;
            case param::OutputFormat::SOG:
                result = lfs::io::save_sog(*splat, {.output_path = output, .kmeans_iterations = params.sog_iterations});
                break;
            case param::OutputFormat::SPZ:
                result = lfs::io::save_spz(*splat, {.output_path = output});
                break;
            case param::OutputFormat::HTML:
                result = lfs::io::export_html(*splat, {.output_path = output, .kmeans_iterations = params.sog_iterations});
                break;
            }

            if (!result) {
                LOG_ERROR("Save failed: {}", result.error().format());
                std::println(stderr, "  Error: {}", result.error().message);
                return false;
            }

            std::println("  Done");
            return true;
        }

    } // namespace

    int run_converter(const param::ConvertParameters& params) {
        const auto files = getInputFiles(params.input_path);
        if (files.empty()) {
            LOG_ERROR("No convertible files in: {}", lfs::core::path_to_utf8(params.input_path));
            std::println(stderr, "Error: No .ply, .sog, or .resume files found");
            return 1;
        }

        std::println("Found {} file(s) to convert", files.size());

        int succeeded = 0, skipped = 0, failed = 0;
        bool overwrite_all = false;

        for (const auto& input : files) {
            const auto output = generateOutputPath(input, params.output_path, params.format);

            if (std::filesystem::exists(output) && !overwrite_all && !params.overwrite) {
                const auto choice = askOverwrite(output);
                if (choice == OverwriteChoice::NO) {
                    std::println("  Skipped");
                    ++skipped;
                    continue;
                }
                if (choice == OverwriteChoice::ALL) {
                    overwrite_all = true;
                }
            }

            if (convertFile(input, output, params)) {
                ++succeeded;
            } else {
                ++failed;
            }
        }

        std::println("\nDone: {} succeeded, {} skipped, {} failed", succeeded, skipped, failed);
        return failed > 0 ? 1 : 0;
    }

} // namespace lfs::core
