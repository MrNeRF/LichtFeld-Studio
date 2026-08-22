/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "html.hpp"
#include "core/base64.hpp"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/splat_data_transform.hpp"
#include "io/atomic_output.hpp"
#include "io/error.hpp"
#include "sogs.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io {

    namespace {

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

        // Viewer export resources (template/css/js/gizmo/measure-tool) are read
        // from disk at export time, so editing them needs no rebuild.
        Result<std::string> read_text_file(const std::filesystem::path& path) {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return make_error(ErrorCode::READ_FAILURE, "Failed to open HTML viewer resource", path);
            }
            std::ostringstream contents;
            contents << file.rdbuf();
            if (file.bad()) {
                return make_error(ErrorCode::READ_FAILURE, "Failed to read HTML viewer resource", path);
            }
            return contents.str();
        }

        // Strips a trailing `export { ... };` statement (only whitespace may
        // follow it) so the source can be concatenated into a plain IIFE
        // instead of a real ES module. Missing or non-trailing `export` is an
        // error: inside the IIFE it is a SyntaxError and the viewer never starts.
        Result<std::string> strip_trailing_export(std::string_view js, std::string_view export_statement,
                                                  const std::filesystem::path& path) {
            std::string result(js);
            const size_t pos = result.rfind(export_statement);
            if (pos == std::string::npos) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  "HTML viewer resource has no trailing export statement", path);
            }
            const size_t after = pos + export_statement.size();
            if (result.find_first_not_of(" \t\r\n", after) != std::string::npos) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  "HTML viewer resource has no trailing export statement", path);
            }
            result.erase(pos);
            return result;
        }

        // Vendored gizmo.js + measure-tool.js, wrapped in an IIFE so their
        // top-level declarations can't collide with index.js's own (they
        // share index.js's classes/constants only via closure), and exposed
        // via a single `window` hook the template calls after `main()` resolves.
        Result<std::string> build_measure_tool_script(const std::string& gizmo_js,
                                                      const std::string& measure_tool_js,
                                                      const std::filesystem::path& gizmo_path,
                                                      const std::filesystem::path& measure_tool_path) {
            auto gizmo_body = strip_trailing_export(gizmo_js, "export { Gizmo, TranslateGizmo };", gizmo_path);
            if (!gizmo_body) {
                return std::unexpected(gizmo_body.error());
            }
            auto measure_body = strip_trailing_export(measure_tool_js, "export { initMeasureTool };",
                                                      measure_tool_path);
            if (!measure_body) {
                return std::unexpected(measure_body.error());
            }

            std::string wrapped;
            wrapped.reserve(gizmo_body->size() + measure_body->size() + 128);
            wrapped += "\n(function () {\n";
            wrapped += *gizmo_body;
            wrapped += "\n";
            wrapped += *measure_body;
            wrapped += "\nwindow.__lfsInitMeasureTool = initMeasureTool;\n})();\n";
            return wrapped;
        }

        Result<std::filesystem::path> resolve_viewer_resources_dir() {
            std::vector<std::filesystem::path> candidates;
            candidates.push_back(lfs::core::getViewerResourcesDir());
#ifdef LFS_DEV_VIEWER_SOURCE_DIR
            candidates.push_back(lfs::core::utf8_to_path(LFS_DEV_VIEWER_SOURCE_DIR));
#endif
            for (const auto& dir : candidates) {
                if (std::filesystem::exists(dir / "viewer_template.html")) {
                    return dir;
                }
            }

            std::string error_msg = "Cannot find HTML viewer resources.\nSearched in:\n";
            for (const auto& dir : candidates) {
                error_msg += "  - " + lfs::core::path_to_utf8(dir) + "\n";
            }
            return make_error(ErrorCode::PATH_NOT_FOUND, error_msg);
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

        Result<std::string> generate_html(const std::string& base64_sog,
                                          const glm::vec3& camera_position,
                                          const glm::vec3& camera_target,
                                          const std::optional<core::ProvenanceStamp>& provenance) {
            auto resource_dir_result = resolve_viewer_resources_dir();
            if (!resource_dir_result)
                return std::unexpected(resource_dir_result.error());
            const auto& resource_dir = *resource_dir_result;

            auto tmpl_result = read_text_file(resource_dir / "viewer_template.html");
            if (!tmpl_result)
                return std::unexpected(tmpl_result.error());
            auto css_result = read_text_file(resource_dir / "index.css");
            if (!css_result)
                return std::unexpected(css_result.error());
            auto js_result = read_text_file(resource_dir / "index.js");
            if (!js_result)
                return std::unexpected(js_result.error());
            auto gizmo_result = read_text_file(resource_dir / "gizmo.js");
            if (!gizmo_result)
                return std::unexpected(gizmo_result.error());
            auto measure_tool_result = read_text_file(resource_dir / "measure-tool.js");
            if (!measure_tool_result)
                return std::unexpected(measure_tool_result.error());

            auto measure_script_result = build_measure_tool_script(*gizmo_result, *measure_tool_result,
                                                                   resource_dir / "gizmo.js",
                                                                   resource_dir / "measure-tool.js");
            if (!measure_script_result)
                return std::unexpected(measure_script_result.error());

            const auto& tmpl = *tmpl_result;
            const auto& css = *css_result;
            const std::string js = *js_result + *measure_script_result;

            std::string html{tmpl};

            const std::string style_link = R"(<link rel="stylesheet" href="./index.css">)";
            const std::string inline_style = "<style>\n" + pad_text(css, 12) + "\n        </style>";
            html = replace_placeholder(html, style_link, inline_style);

            const std::string js_import = "import { main } from './index.js';";
            html = replace_placeholder(html, js_import, js);

            const std::string settings_fetch = "settings: fetch(settingsUrl).then(response => response.json())";
            std::ostringstream settings_stream;
            settings_stream << std::fixed << std::setprecision(6);
            settings_stream << "settings:{\"camera\":{\"fov\":50,\"position\":[" << camera_position.x << "," << camera_position.y << "," << camera_position.z << "],\"target\":[" << camera_target.x << "," << camera_target.y << "," << camera_target.z << "],\"startAnim\":\"none\"},\"background\":{\"color\":[0,0,0]},\"animTracks\":[]}";
            html = replace_placeholder(html, settings_fetch, settings_stream.str());

            const std::string content_fetch = "fetch(contentUrl)";
            const std::string base64_fetch = "fetch(\"data:application/octet-stream;base64," + base64_sog + "\")";
            html = replace_placeholder(html, content_fetch, base64_fetch);

            html = replace_placeholder(html, ".compressed.ply", ".sog");

            if (provenance) {
                const std::string json = core::provenance_to_json(*provenance);
                std::string escaped;
                escaped.reserve(json.size());
                for (const char c : json) {
                    if (c == '<')
                        escaped += "\\u003c";
                    else
                        escaped += c;
                }
                const std::string script =
                    "<script type=\"application/json\" id=\"lichtfeld-provenance\">" + escaped +
                    "</script>\n    </head>";
                html = replace_placeholder(html, "</head>", script);
            }

            return html;
        }

    } // anonymous namespace

    Result<void> export_html(const SplatData& splat_data, const HtmlExportOptions& options_in) {
        HtmlExportOptions options = options_in;
        if (!options.provenance) {
            options.provenance = core::make_minimal_provenance_stamp();
        }

        if (!report_export_progress(options.progress_callback, 0.0f, "Exporting SOG...")) {
            return make_error(ErrorCode::CANCELLED, "HTML export cancelled", options.output_path);
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

        ScopedAtomicOutputFile temp_sog(options.output_path);
        const SogSaveOptions sog_options{
            .output_path = temp_sog.temp_path(),
            .kmeans_iterations = options.kmeans_iterations,
            .use_gpu = true,
            .progress_callback = [&](float p, const std::string& stage) {
                return report_export_progress(options.progress_callback, p * 0.5f, stage);
            },
            .provenance = options.provenance};

        if (auto result = save_sog(splat_data, sog_options); !result) {
            // Propagate the SOG error with context
            return make_error(result.error().code,
                              std::format("Failed to create SOG for HTML export: {}", result.error().message),
                              options.output_path);
        }

        if (!report_export_progress(options.progress_callback, 0.5f, "Encoding data...")) {
            return make_error(ErrorCode::CANCELLED, "HTML export cancelled", options.output_path);
        }

        const auto sog_data = read_file_binary(temp_sog.temp_path());

        if (sog_data.empty()) {
            return make_error(ErrorCode::READ_FAILURE,
                              "Failed to read temporary SOG file", temp_sog.temp_path());
        }

        const auto base64_data = core::base64_encode(sog_data);

        if (!report_export_progress(options.progress_callback, 0.8f, "Generating HTML...")) {
            return make_error(ErrorCode::CANCELLED, "HTML export cancelled", options.output_path);
        }

        // Compute bounding box and derive camera position for a 30° elevation view
        glm::vec3 min_bounds{0.f, 0.f, 0.f};
        glm::vec3 max_bounds{0.f, 0.f, 0.f};
        if (!compute_bounds(splat_data, min_bounds, max_bounds, 0.0f, false)) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              "Failed to compute model bounds for camera setup", options.output_path);
        }

        const glm::vec3 extent = max_bounds - min_bounds;
        const float max_extent = std::max({extent.x, extent.y, extent.z});
        const glm::vec3 center = (min_bounds + max_bounds) * 0.5f;

        // The web viewer's SOG loader (index.js `loadGsplat`) always attaches the
        // gsplat entity with a fixed 180° rotation about Z, while the camera entity
        // itself is left unrotated. That negates X and Y for the model as actually
        // rendered, so the camera's look-at target/position must be computed in that
        // same rotated space (otherwise the orbit pivots around a point away from the
        // model whenever its center isn't near X=0, Y=0).
        const glm::vec3 render_center{-center.x, -center.y, center.z};

        // Distance to fit model in 50° FOV with 1.5× margin
        const float fov_rad = glm::radians(50.0f);
        const float distance = max_extent / std::tan(fov_rad * 0.5f) * 1.5f;

        // 30° elevation angle from horizontal, looking from +X,+Z diagonal
        const float elev_rad = glm::radians(30.0f);
        const float horiz_dist = distance * std::cos(elev_rad);
        const float vert_dist = distance * std::sin(elev_rad);
        const float diag = horiz_dist * 0.70710678118f;

        const glm::vec3 camera_position{render_center.x + diag, render_center.y + vert_dist, render_center.z - diag};
        const glm::vec3 camera_target{render_center.x, render_center.y, render_center.z};

        auto html_result = generate_html(base64_data, camera_position, camera_target, options.provenance);
        if (!html_result) {
            return std::unexpected(html_result.error());
        }
        const auto& html = *html_result;

        if (!report_export_progress(options.progress_callback, 0.9f, "Writing HTML...")) {
            return make_error(ErrorCode::CANCELLED, "HTML export cancelled", options.output_path);
        }

        ScopedAtomicOutputFile atomic_output(options.output_path);
        std::ofstream out;
        if (!lfs::core::open_file_for_write(atomic_output.temp_path(), std::ios::binary | std::ios::out, out)) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to open temporary HTML file for writing",
                              atomic_output.temp_path());
        }
        out.write(html.data(), static_cast<std::streamsize>(html.size()));
        out.close();

        if (!out.good()) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              "Failed to write HTML content (possibly disk full)", atomic_output.temp_path());
        }

        if (!report_export_progress(options.progress_callback, 1.0f, "Done")) {
            return make_error(ErrorCode::CANCELLED, "HTML export cancelled", options.output_path);
        }

        if (auto commit_result = atomic_output.commit(); !commit_result) {
            return std::unexpected(commit_result.error());
        }

        LOG_INFO("Exported HTML viewer: {} ({:.1f} MB)",
                 lfs::core::path_to_utf8(options.output_path),
                 static_cast<float>(html.size()) / (1024 * 1024));

        return {};
    }

} // namespace lfs::io
