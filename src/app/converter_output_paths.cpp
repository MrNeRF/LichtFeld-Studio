/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/converter_output_paths.hpp"

namespace lfs::app {

    namespace {

        [[nodiscard]] const char* format_extension(
            const lfs::core::param::OutputFormat format) {
            using lfs::core::param::OutputFormat;
            switch (format) {
            case OutputFormat::PLY: return ".ply";
            case OutputFormat::SOG: return ".sog";
            case OutputFormat::SSOG: return ".ssog";
            case OutputFormat::SPZ: return ".spz";
            case OutputFormat::GLB: return ".glb";
            case OutputFormat::HTML: return ".html";
            case OutputFormat::USD: return ".usd";
            case OutputFormat::USDA: return ".usda";
            case OutputFormat::USDC: return ".usdc";
            case OutputFormat::RAD: return ".rad";
            }
            return ".ply";
        }

    } // namespace

    std::filesystem::path generate_converter_output_path(
        const std::filesystem::path& input,
        const std::filesystem::path& output_template,
        const lfs::core::param::OutputFormat format,
        const char* const suffix,
        const bool replace_output_extension) {
        if (format == lfs::core::param::OutputFormat::SSOG) {
            auto default_output = input;
            default_output.replace_extension(".ssog");
            return std::filesystem::absolute(
                output_template.empty() ? default_output : output_template);
        }

        const auto extension = format_extension(format);
        const auto working_directory = std::filesystem::current_path();
        auto converted_name = input.stem();
        converted_name += suffix;
        converted_name += extension;

        if (output_template.empty()) {
            return working_directory / converted_name;
        }
        if (std::filesystem::is_directory(output_template)) {
            const auto directory = output_template.is_absolute()
                                       ? output_template
                                       : working_directory / output_template;
            return directory / converted_name;
        }

        auto output = output_template;
        if (replace_output_extension) {
            output.replace_extension(extension);
        } else if (output.extension().empty()) {
            output += extension;
        }
        return output.is_absolute() ? output : working_directory / output;
    }

    std::filesystem::path generate_ssog_batch_output_path(
        const std::filesystem::path& output_directory,
        const std::filesystem::path& input) {
        auto output_name = input.stem();
        output_name += ".ssog";
        return output_directory / output_name;
    }

} // namespace lfs::app
