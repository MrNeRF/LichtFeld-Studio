/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "html.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "html_viewer_resources.hpp"
#include "io/error.hpp"
#include "sogs.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace lfs::io {

    namespace {

        constexpr char BASE64_CHARS[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string base64_encode(const std::vector<uint8_t>& data) {
            std::string result;
            result.reserve(((data.size() + 2) / 3) * 4);

            for (size_t i = 0; i < data.size(); i += 3) {
                const uint32_t b0 = data[i];
                const uint32_t b1 = (i + 1 < data.size()) ? data[i + 1] : 0;
                const uint32_t b2 = (i + 2 < data.size()) ? data[i + 2] : 0;

                result += BASE64_CHARS[(b0 >> 2) & 0x3F];
                result += BASE64_CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F];
                result += (i + 1 < data.size()) ? BASE64_CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=';
                result += (i + 2 < data.size()) ? BASE64_CHARS[b2 & 0x3F] : '=';
            }
            return result;
        }

        std::vector<uint8_t> read_file_binary(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary | std::ios::ate, file))
                return {};

            const auto size = file.tellg();
            file.seekg(0, std::ios::beg);

            std::vector<uint8_t> buffer(size);
            file.read(reinterpret_cast<char*>(buffer.data()), size);
            return buffer;
        }

        std::string replace_placeholder(std::string_view input, std::string_view placeholder, std::string_view replacement) {
            std::string result;
            result.reserve(input.size() + replacement.size());

            size_t pos = 0;
            while (pos < input.size()) {
                const size_t found = input.find(placeholder, pos);
                if (found == std::string_view::npos) {
                    result.append(input.substr(pos));
                    break;
                }
                result.append(input.substr(pos, found - pos));
                result.append(replacement);
                pos = found + placeholder.size();
            }
            return result;
        }

        std::string pad_text(std::string_view text, int spaces) {
            std::string result;
            std::string whitespace(spaces, ' ');
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t newline = text.find('\n', pos);
                if (newline == std::string_view::npos) {
                    result += whitespace;
                    result.append(text.substr(pos));
                    break;
                }
                result += whitespace;
                result.append(text.substr(pos, newline - pos + 1));
                pos = newline + 1;
            }
            return result;
        }

        std::string generate_html(const std::string& base64_sog) {
            const auto tmpl = get_viewer_template();
            const auto css = get_viewer_css();
            const auto js = get_viewer_js();

            std::string html{tmpl};

            const std::string style_link = R"(<link rel="stylesheet" href="./index.css">)";
            const std::string inline_style = "<style>\n" + pad_text(css, 12) + "\n        </style>";
            html = replace_placeholder(html, style_link, inline_style);

            const std::string js_import = "import { main } from './index.js';";
            html = replace_placeholder(html, js_import, js);

            const std::string settings_fetch = "settings: fetch(settingsUrl).then(response => response.json())";
            const std::string inline_settings = R"(settings: {"camera":{"fov":50,"position":[2,2,-2],"target":[0,0,0],"startAnim":"none"},"background":{"color":[0,0,0]},"animTracks":[]})";
            html = replace_placeholder(html, settings_fetch, inline_settings);

            const std::string content_fetch = "fetch(contentUrl)";
            const std::string base64_fetch = "fetch(\"data:application/octet-stream;base64," + base64_sog + "\")";
            html = replace_placeholder(html, content_fetch, base64_fetch);

            html = replace_placeholder(html, ".compressed.ply", ".sog");

            return html;
        }

    } // anonymous namespace

    Result<void> export_html(const SplatData& splat_data, const HtmlExportOptions& options) {
        if (options.progress_callback) {
            options.progress_callback(0.0f, "Exporting SOG...");
        }

        // Estimate HTML file size: SOG data (compressed) + base64 overhead (4/3) + HTML template (~50KB)
        // SOG is roughly 0.4 * (5 textures * width * height * 4 bytes)
        const int64_t num_splats = splat_data.size();
        const int width = static_cast<int>(std::ceil(std::sqrt(num_splats) / 4.0)) * 4;
        const int height = static_cast<int>(std::ceil(static_cast<double>(num_splats) / width / 4.0)) * 4;
        const size_t sog_estimate = static_cast<size_t>(width * height * 4 * 5 * 0.4);
        const size_t base64_estimate = (sog_estimate * 4) / 3 + 4; // Base64 is 4/3 larger
        const size_t html_template_size = 51200;                   // ~50KB for HTML/CSS/JS
        const size_t estimated_size = base64_estimate + html_template_size;

        // Check disk space for output file
        if (auto space_check = check_disk_space(options.output_path, estimated_size, 1.1f); !space_check) {
            return std::unexpected(space_check.error());
        }

        // Verify path is writable
        if (auto writable_check = verify_writable(options.output_path); !writable_check) {
            return std::unexpected(writable_check.error());
        }

        const auto temp_sog = std::filesystem::temp_directory_path() / "lfs_html_export_temp.sog";
        const SogSaveOptions sog_options{
            .output_path = temp_sog,
            .kmeans_iterations = options.kmeans_iterations,
            .use_gpu = true,
            .progress_callback = [&](float p, const std::string& stage) {
                if (options.progress_callback) {
                    options.progress_callback(p * 0.5f, stage);
                }
                return true;
            }};

        if (auto result = save_sog(splat_data, sog_options); !result) {
            // Propagate the SOG error with context
            return make_error(result.error().code,
                              std::format("Failed to create SOG for HTML export: {}", result.error().message),
                              options.output_path);
        }

        if (options.progress_callback) {
            options.progress_callback(0.5f, "Encoding data...");
        }

        const auto sog_data = read_file_binary(temp_sog);
        std::error_code ec;
        std::filesystem::remove(temp_sog, ec); // Best effort cleanup

        if (sog_data.empty()) {
            return make_error(ErrorCode::READ_FAILURE,
                              "Failed to read temporary SOG file", temp_sog);
        }

        const auto base64_data = base64_encode(sog_data);

        if (options.progress_callback) {
            options.progress_callback(0.8f, "Generating HTML...");
        }

        const auto html = generate_html(base64_data);

        std::ofstream out;
        if (!lfs::core::open_file_for_write(options.output_path, out)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open output file for writing", options.output_path);
        }
        out << html;

        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write HTML content (possibly disk full)", options.output_path);
        }
        out.close();

        if (options.progress_callback) {
            options.progress_callback(1.0f, "Done");
        }

        LOG_INFO("Exported HTML viewer: {} ({:.1f} MB)",
                 lfs::core::path_to_utf8(options.output_path),
                 static_cast<float>(html.size()) / (1024 * 1024));

        return {};
    }

} // namespace lfs::io
