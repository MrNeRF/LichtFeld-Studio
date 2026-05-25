/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/converter.hpp"
#include "core/logger.hpp"
#include "core/mesh_data.hpp"
#include "core/path_utils.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"
#include "rendering/mesh2splat.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <print>

namespace lfs::app {

    using namespace lfs::core;

    namespace {

        constexpr const char* CONVERT_EXTENSIONS[] = {".ply", ".sog", ".spz", ".usd", ".usda", ".usdc", ".usdz", ".resume", ".rad"};
        constexpr const char* MESH_EXTENSIONS[] = {".obj", ".fbx", ".gltf", ".glb", ".stl", ".dae", ".3ds", ".mesh", ".ply"};

        enum class OverwriteChoice { YES,
                                     NO,
                                     ALL };

        struct OutputTarget {
            param::OutputFormat format;
            std::filesystem::path path;
        };

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
            if (degree < 0)
                return;
            splat.set_sh_degree(degree);
        }

        template <size_t N>
        std::vector<std::filesystem::path> getInputFiles(const std::filesystem::path& path, const char* const (&valid_extensions)[N]) {
            std::vector<std::filesystem::path> files;
            if (std::filesystem::is_directory(path)) {
                for (const auto& entry : std::filesystem::directory_iterator(path)) {
                    if (!entry.is_regular_file())
                        continue;
                    auto ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), [](const unsigned char c) {
                        return static_cast<char>(std::tolower(c));
                    });
                    for (const auto* valid : valid_extensions) {
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
            case param::OutputFormat::USD: return ".usd";
            case param::OutputFormat::USDA: return ".usda";
            case param::OutputFormat::USDC: return ".usdc";
            case param::OutputFormat::RAD: return ".rad";
            }
            return ".ply";
        }

        std::filesystem::path generateOutputPath(
            const std::filesystem::path& input,
            const std::filesystem::path& output_template,
            const param::OutputFormat format,
            const char* suffix,
            const bool replace_output_extension = false) {

            const auto ext = getFormatExtension(format);
            const auto cwd = std::filesystem::current_path();
            const auto converted_name = input.stem().string() + suffix + ext;

            if (output_template.empty()) {
                return cwd / converted_name;
            }

            if (std::filesystem::is_directory(output_template)) {
                const auto dir = output_template.is_absolute() ? output_template : cwd / output_template;
                return dir / converted_name;
            }

            auto out = output_template;
            if (replace_output_extension) {
                out.replace_extension(ext);
            } else if (out.extension().empty()) {
                out += ext;
            }
            return out.is_absolute() ? out : cwd / out;
        }

        std::vector<OutputTarget> generateMesh2SplatOutputs(
            const std::filesystem::path& input,
            const param::Mesh2SplatParameters& params) {
            std::vector<OutputTarget> targets;
            targets.reserve(params.formats.size());
            const bool multi_format = params.formats.size() > 1;
            for (const auto format : params.formats) {
                targets.push_back({
                    .format = format,
                    .path = generateOutputPath(input, params.output_path, format, "_splat", multi_format),
                });
            }
            return targets;
        }

        lfs::io::Result<void> saveSplat(
            const SplatData& splat,
            const std::filesystem::path& output,
            const param::OutputFormat format,
            const int sog_iterations,
            const std::vector<float>& rad_lod_levels) {
            switch (format) {
            case param::OutputFormat::PLY:
                return lfs::io::save_ply(splat, {.output_path = output, .binary = true});
            case param::OutputFormat::SOG:
                return lfs::io::save_sog(splat, {.output_path = output, .kmeans_iterations = sog_iterations});
            case param::OutputFormat::SPZ:
                return lfs::io::save_spz(splat, {.output_path = output});
            case param::OutputFormat::HTML:
                return lfs::io::export_html(splat, {.output_path = output, .kmeans_iterations = sog_iterations});
            case param::OutputFormat::USD:
            case param::OutputFormat::USDA:
            case param::OutputFormat::USDC:
                return lfs::io::save_usd(splat, {.output_path = output});
            case param::OutputFormat::RAD:
                return lfs::io::save_rad(splat, {.output_path = output, .lod_ratios = rad_lod_levels});
            }
            return lfs::io::save_ply(splat, {.output_path = output, .binary = true});
        }

        bool convertFile(
            const std::filesystem::path& input,
            const std::filesystem::path& output,
            const param::ConvertParameters& params) {

            std::println("Converting: {} -> {}", path_to_utf8(input), path_to_utf8(output));

            const auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(input);
            if (!load_result) {
                LOG_ERROR("Load failed: {}", load_result.error().format());
                std::println(stderr, "  Error: {}", load_result.error().message);
                return false;
            }

            auto* splat_ptr = std::get_if<std::shared_ptr<SplatData>>(&load_result->data);
            if (!splat_ptr || !*splat_ptr) {
                LOG_ERROR("Not a splat file: {}", path_to_utf8(input));
                std::println(stderr, "  Error: not a splat file");
                return false;
            }

            auto splat = std::move(*splat_ptr);
            std::println("  Loaded {} gaussians, SH degree {}", splat->size(), splat->get_max_sh_degree());

            if (params.sh_degree >= 0 && params.sh_degree != splat->get_max_sh_degree()) {
                truncateSHDegree(*splat, params.sh_degree);
                std::println("  Set SH degree {}", params.sh_degree);
            }

            const auto result = saveSplat(*splat, output, params.format, params.sog_iterations, params.rad_lod_levels);

            if (!result) {
                LOG_ERROR("Save failed: {}", result.error().format());
                std::println(stderr, "  Error: {}", result.error().message);
                return false;
            }

            std::println("  Done");
            return true;
        }

        bool mesh2splatFile(
            const std::filesystem::path& input,
            const std::vector<OutputTarget>& outputs,
            const param::Mesh2SplatParameters& params) {

            std::println("Converting mesh: {}", path_to_utf8(input));

            const auto loader = lfs::io::Loader::create();
            auto load_result = loader->load(input);
            if (!load_result) {
                LOG_ERROR("Mesh load failed: {}", load_result.error().format());
                std::println(stderr, "  Error: {}", load_result.error().message);
                return false;
            }

            auto* mesh_ptr = std::get_if<std::shared_ptr<MeshData>>(&load_result->data);
            if (!mesh_ptr || !*mesh_ptr) {
                LOG_ERROR("Not a mesh file: {}", path_to_utf8(input));
                std::println(stderr, "  Error: not a mesh file");
                return false;
            }

            const auto& mesh = **mesh_ptr;
            std::println("  Loaded {} vertices, {} faces", mesh.vertex_count(), mesh.face_count());

            auto splat = lfs::rendering::mesh_to_splat(
                mesh,
                params.options,
                [](const float progress, const std::string& stage) {
                    std::println("  {:3.0f}% {}", std::clamp(progress, 0.0f, 1.0f) * 100.0f, stage);
                    return true;
                });
            if (!splat) {
                LOG_ERROR("Mesh2Splat conversion failed: {}", splat.error());
                std::println(stderr, "  Error: {}", splat.error());
                return false;
            }

            std::println("  Generated {} gaussians", (*splat)->size());

            bool ok = true;
            for (const auto& output : outputs) {
                std::println("  Saving: {}", path_to_utf8(output.path));
                const auto result = saveSplat(**splat, output.path, output.format, params.sog_iterations, params.rad_lod_levels);
                if (!result) {
                    LOG_ERROR("Save failed: {}", result.error().format());
                    std::println(stderr, "  Error: {}", result.error().message);
                    ok = false;
                }
            }

            if (ok)
                std::println("  Done");
            return ok;
        }

    } // namespace

    int run_converter(const param::ConvertParameters& params) {
        const auto files = getInputFiles(params.input_path, CONVERT_EXTENSIONS);
        if (files.empty()) {
            LOG_ERROR("No convertible files in: {}", path_to_utf8(params.input_path));
            std::println(stderr, "Error: No .ply, .sog, .spz, .usd, .usda, .usdc, .usdz, .resume, or .rad files found");
            return 1;
        }

        std::println("Found {} file(s) to convert", files.size());

        int succeeded = 0, skipped = 0, failed = 0;
        bool overwrite_all = false;

        for (const auto& input : files) {
            const auto output = generateOutputPath(input, params.output_path, params.format, "_converted");

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

    int run_mesh2splat(const param::Mesh2SplatParameters& params) {
        const auto files = getInputFiles(params.input_path, MESH_EXTENSIONS);
        if (files.empty()) {
            LOG_ERROR("No mesh files in: {}", path_to_utf8(params.input_path));
            std::println(stderr, "Error: No .obj, .fbx, .gltf, .glb, .stl, .dae, .3ds, .mesh, or mesh .ply files found");
            return 1;
        }

        std::println("Found {} mesh file(s) to convert", files.size());

        int succeeded = 0, skipped = 0, failed = 0;
        bool overwrite_all = false;

        for (const auto& input : files) {
            const auto outputs = generateMesh2SplatOutputs(input, params);

            bool skip = false;
            for (const auto& output : outputs) {
                if (std::filesystem::exists(output.path) && !overwrite_all && !params.overwrite) {
                    const auto choice = askOverwrite(output.path);
                    if (choice == OverwriteChoice::NO) {
                        skip = true;
                        break;
                    }
                    if (choice == OverwriteChoice::ALL) {
                        overwrite_all = true;
                    }
                }
            }
            if (skip) {
                std::println("  Skipped");
                ++skipped;
                continue;
            }

            if (mesh2splatFile(input, outputs, params)) {
                ++succeeded;
            } else {
                ++failed;
            }
        }

        std::println("\nDone: {} succeeded, {} skipped, {} failed", succeeded, skipped, failed);
        return failed > 0 ? 1 : 0;
    }

} // namespace lfs::app
