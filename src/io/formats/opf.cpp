/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "opf.hpp"

#include <fstream>
#include <format>
#include <nlohmann/json.hpp>
#include <unordered_set>

namespace lfs::io::opf {
    namespace {
        using json = nlohmann::json;

        [[nodiscard]] std::unexpected<Error> invalid(const std::filesystem::path& path,
                                                       const std::string& message) {
            return make_error(ErrorCode::INVALID_DATASET, message, path);
        }

        [[nodiscard]] bool is_inside(const std::filesystem::path& root,
                                     const std::filesystem::path& candidate) {
            const auto root_norm = root.lexically_normal();
            const auto candidate_norm = candidate.lexically_normal();
            auto root_it = root_norm.begin();
            auto candidate_it = candidate_norm.begin();
            for (; root_it != root_norm.end() && candidate_it != candidate_norm.end();
                 ++root_it, ++candidate_it) {
                if (*root_it != *candidate_it)
                    return false;
            }
            return root_it == root_norm.end();
        }

        [[nodiscard]] Result<std::string> required_string(const json& object,
                                                           const char* key,
                                                           const std::filesystem::path& path,
                                                           const std::string& context) {
            const auto it = object.find(key);
            if (it == object.end() || !it->is_string() || it->get<std::string>().empty())
                return invalid(path, std::format("OPF {} requires non-empty string '{}'.", context, key));
            return it->get<std::string>();
        }

        [[nodiscard]] Result<std::filesystem::path> resolve_resource(
            const std::filesystem::path& root,
            const std::string& uri,
            const std::filesystem::path& project_path) {
            const std::filesystem::path relative = std::filesystem::u8path(uri);
            if (relative.empty() || relative.is_absolute() || relative.has_root_name())
                return invalid(project_path, std::format("OPF resource URI '{}' must be relative.", uri));
            const auto resolved = (root / relative).lexically_normal();
            if (!is_inside(root, resolved))
                return invalid(project_path, std::format("OPF resource URI '{}' escapes the project directory.", uri));
            if (!std::filesystem::is_regular_file(resolved))
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  std::format("OPF resource '{}' was not found.", uri), resolved);
            return resolved;
        }

        void collect_extension_warning(const json& object,
                                       const char* context,
                                       Project& project) {
            const auto it = object.find("extensions");
            if (it != object.end() && !it->is_object()) {
                project.warnings.push_back(std::format("OPF {} extensions are not an object; ignored.", context));
            } else if (it != object.end() && !it->empty()) {
                project.warnings.push_back(std::format("OPF {} contains {} extension(s); unknown extensions were retained as metadata warnings.",
                                                       context, it->size()));
            }
        }
    } // namespace

    Result<Project> read_project(const std::filesystem::path& path) {
        std::error_code ec;
        if (!std::filesystem::exists(path, ec))
            return make_error(ErrorCode::PATH_NOT_FOUND, "OPF project does not exist.", path);
        if (!std::filesystem::is_regular_file(path, ec))
            return make_error(ErrorCode::NOT_A_FILE, "OPF project must be a file.", path);

        std::ifstream input(path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF project.", path);

        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF JSON: {}", error.what()), path);
        }
        if (!root.is_object())
            return invalid(path, "OPF project root must be a JSON object.");

        Project project;
        project.path = path;
        auto format = required_string(root, "format", path, "project");
        if (!format)
            return std::unexpected(format.error());
        project.format = *format;
        if (project.format != "application/opf-project+json")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("Unsupported OPF project format '{}'.", project.format), path);

        for (const char* key : {"version", "id", "name"}) {
            auto value = required_string(root, key, path, "project");
            if (!value)
                return std::unexpected(value.error());
            if (std::string_view(key) == "version") project.version = *value;
            if (std::string_view(key) == "id") project.id = *value;
            if (std::string_view(key) == "name") project.name = *value;
        }
        if (!root.contains("items") || !root["items"].is_array() || root["items"].empty())
            return invalid(path, "OPF project requires a non-empty 'items' array.");
        collect_extension_warning(root, "project", project);

        const auto root_dir = path.parent_path().lexically_normal();
        std::unordered_set<std::string> item_ids;
        for (size_t index = 0; index < root["items"].size(); ++index) {
            const auto& value = root["items"][index];
            const auto context = std::format("item {}", index);
            if (!value.is_object())
                return invalid(path, std::format("OPF {} must be an object.", context));

            Item item;
            auto id = required_string(value, "id", path, context);
            auto type = required_string(value, "type", path, context);
            if (!id) return std::unexpected(id.error());
            if (!type) return std::unexpected(type.error());
            item.id = *id;
            item.type = *type;
            if (!item_ids.insert(item.id).second)
                return invalid(path, std::format("OPF item id '{}' is duplicated.", item.id));

            if (!value.contains("sources") || !value["sources"].is_array())
                return invalid(path, std::format("OPF {} requires an array 'sources'.", context));
            for (const auto& source_value : value["sources"]) {
                if (!source_value.is_object())
                    return invalid(path, std::format("OPF {} contains a non-object source.", context));
                auto source_id = required_string(source_value, "id", path, "source");
                auto source_type = required_string(source_value, "type", path, "source");
                if (!source_id) return std::unexpected(source_id.error());
                if (!source_type) return std::unexpected(source_type.error());
                item.sources.push_back({*source_id, *source_type});
            }

            if (!value.contains("resources") || !value["resources"].is_array())
                return invalid(path, std::format("OPF {} requires an array 'resources'.", context));
            for (const auto& resource_value : value["resources"]) {
                if (!resource_value.is_object())
                    return invalid(path, std::format("OPF {} contains a non-object resource.", context));
                auto uri = required_string(resource_value, "uri", path, "resource");
                auto resource_format = required_string(resource_value, "format", path, "resource");
                if (!uri) return std::unexpected(uri.error());
                if (!resource_format) return std::unexpected(resource_format.error());
                auto resolved = resolve_resource(root_dir, *uri, path);
                if (!resolved) return std::unexpected(resolved.error());
                item.resources.push_back({*uri, *resource_format, *resolved});
            }
            collect_extension_warning(value, context.c_str(), project);
            project.items.push_back(std::move(item));
        }
        return project;
    }
} // namespace lfs::io::opf
