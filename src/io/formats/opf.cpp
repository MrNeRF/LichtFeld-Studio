/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "opf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
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

        [[nodiscard]] bool is_extension_item(const std::string& type) {
            return type.starts_with("ext_");
        }

        [[nodiscard]] bool is_extension_resource(const std::string& format) {
            return format.starts_with("application/ext-");
        }

        [[nodiscard]] Result<void> validate_item_formats(const Item& item,
                                                         const std::filesystem::path& path) {
            static const std::unordered_map<std::string, std::vector<std::string>> required = {
                {"camera_list", {"application/opf-camera-list+json"}},
                {"input_cameras", {"application/opf-input-cameras+json"}},
                {"projected_input_cameras", {"application/opf-projected-input-cameras+json"}},
                {"scene_reference_frame", {}},
                {"input_control_points", {"application/opf-input-control-points+json"}},
                {"projected_control_points", {"application/opf-projected-control-points+json"}},
                {"constraints", {"application/opf-constraints+json"}},
                {"calibration", {"application/opf-calibrated-cameras+json"}},
                {"point_cloud", {"model/gltf+json", "application/gltf-buffer+bin"}},
            };
            const auto item_it = required.find(item.type);
            if (item_it == required.end()) {
                if (!is_extension_item(item.type))
                    return invalid(path, std::format("Unsupported OPF item type '{}'.", item.type));
                return {};
            }

            std::unordered_set<std::string> formats;
            for (const auto& resource : item.resources) {
                formats.insert(resource.format);
                if (!is_extension_resource(resource.format) &&
                    resource.format != "application/opf-scene-reference-frame+json" &&
                    resource.format != "application/opf-camera-list+json" &&
                    resource.format != "application/opf-input-cameras+json" &&
                    resource.format != "application/opf-projected-input-cameras+json" &&
                    resource.format != "application/opf-input-control-points+json" &&
                    resource.format != "application/opf-projected-control-points+json" &&
                    resource.format != "application/opf-calibrated-control-points+json" &&
                    resource.format != "application/opf-constraints+json" &&
                    resource.format != "application/opf-calibrated-cameras+json" &&
                    resource.format != "application/opf-gps-bias+json" &&
                    resource.format != "model/gltf+json" &&
                    resource.format != "application/gltf-buffer+bin")
                    return invalid(path, std::format("Unsupported OPF resource format '{}'.", resource.format));
            }
            for (const auto& format : item_it->second) {
                if (!formats.contains(format))
                    return invalid(path, std::format("OPF item '{}' requires resource format '{}'.",
                                                     item.id, format));
            }
            return {};
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
            if (std::string_view(key) == "version")
                project.version = *value;
            if (std::string_view(key) == "id")
                project.id = *value;
            if (std::string_view(key) == "name")
                project.name = *value;
        }
        if (const auto description = root.find("description"); description != root.end()) {
            if (!description->is_string())
                return invalid(path, "OPF project description must be a string when present.");
            project.description = description->get<std::string>();
        } else {
            project.warnings.push_back("OPF project omits optional-in-practice 'description'.");
        }
        if (!root.contains("items") || !root["items"].is_array() || root["items"].empty())
            return invalid(path, "OPF project requires a non-empty 'items' array.");
        collect_extension_warning(root, "project", project);

        const auto root_dir = path.parent_path().lexically_normal();
        std::unordered_set<std::string> item_ids;
        std::unordered_map<std::string, std::string> item_types;
        for (size_t index = 0; index < root["items"].size(); ++index) {
            const auto& value = root["items"][index];
            const auto context = std::format("item {}", index);
            if (!value.is_object())
                return invalid(path, std::format("OPF {} must be an object.", context));

            Item item;
            auto id = required_string(value, "id", path, context);
            auto type = required_string(value, "type", path, context);
            if (!id)
                return std::unexpected(id.error());
            if (!type)
                return std::unexpected(type.error());
            item.id = *id;
            item.type = *type;
            if (!item_ids.insert(item.id).second)
                return invalid(path, std::format("OPF item id '{}' is duplicated.", item.id));
            item_types.emplace(item.id, item.type);

            if (!value.contains("sources") || !value["sources"].is_array())
                return invalid(path, std::format("OPF {} requires an array 'sources'.", context));
            for (const auto& source_value : value["sources"]) {
                if (!source_value.is_object())
                    return invalid(path, std::format("OPF {} contains a non-object source.", context));
                auto source_id = required_string(source_value, "id", path, "source");
                auto source_type = required_string(source_value, "type", path, "source");
                if (!source_id)
                    return std::unexpected(source_id.error());
                if (!source_type)
                    return std::unexpected(source_type.error());
                item.sources.push_back({*source_id, *source_type});
            }

            if (!value.contains("resources") || !value["resources"].is_array())
                return invalid(path, std::format("OPF {} requires an array 'resources'.", context));
            for (const auto& resource_value : value["resources"]) {
                if (!resource_value.is_object())
                    return invalid(path, std::format("OPF {} contains a non-object resource.", context));
                auto uri = required_string(resource_value, "uri", path, "resource");
                auto resource_format = required_string(resource_value, "format", path, "resource");
                if (!uri)
                    return std::unexpected(uri.error());
                if (!resource_format)
                    return std::unexpected(resource_format.error());
                auto resolved = resolve_resource(root_dir, *uri, path);
                if (!resolved)
                    return std::unexpected(resolved.error());
                item.resources.push_back({*uri, *resource_format, *resolved});
            }
            collect_extension_warning(value, context.c_str(), project);
            auto format_validation = validate_item_formats(item, path);
            if (!format_validation)
                return std::unexpected(format_validation.error());
            project.items.push_back(std::move(item));
        }

        for (const auto& item : project.items) {
            for (const auto& source : item.sources) {
                const auto source_it = item_types.find(source.id);
                if (source_it == item_types.end())
                    return invalid(path, std::format("OPF item '{}' references missing source '{}'.", item.id, source.id));
                if (source.type != source_it->second)
                    return invalid(path, std::format("OPF item '{}' source '{}' has type '{}', expected '{}'.",
                                                     item.id, source.id, source.type, source_it->second));
            }
        }

        std::unordered_map<std::string, int> visit_state;
        const auto visit = [&](const auto& self, const std::string& id) -> bool {
            auto& state = visit_state[id];
            if (state == 1)
                return false;
            if (state == 2)
                return true;
            state = 1;
            const auto& item = project.items[std::distance(
                project.items.begin(), std::find_if(project.items.begin(), project.items.end(),
                                                    [&](const Item& candidate) { return candidate.id == id; }))];
            for (const auto& source : item.sources) {
                if (!self(self, source.id))
                    return false;
            }
            state = 2;
            return true;
        };
        for (const auto& item : project.items) {
            if (!visit(visit, item.id))
                return invalid(path, "OPF project contains a circular source dependency.");
        }
        return project;
    }

    Result<std::vector<CameraImage>> read_camera_list(const Resource& resource,
                                                      const std::filesystem::path& project_root) {
        if (resource.format != "application/opf-camera-list+json")
            return invalid(resource.resolved_path, "OPF camera list resource has an unexpected format.");
        std::ifstream input(resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF camera list.", resource.resolved_path);

        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF camera list JSON: {}", error.what()),
                              resource.resolved_path);
        }
        if (!root.is_object())
            return invalid(resource.resolved_path, "OPF camera list root must be a JSON object.");
        auto format = required_string(root, "format", resource.resolved_path, "camera list");
        auto version = required_string(root, "version", resource.resolved_path, "camera list");
        if (!format)
            return std::unexpected(format.error());
        if (!version)
            return std::unexpected(version.error());
        if (*format != "application/opf-camera-list+json")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Unsupported OPF camera list format.", resource.resolved_path);
        if (!root.contains("cameras") || !root["cameras"].is_array())
            return invalid(resource.resolved_path, "OPF camera list requires a 'cameras' array.");

        std::vector<CameraImage> cameras;
        std::unordered_set<std::uint64_t> ids;
        const auto base = project_root.lexically_normal();
        for (const auto& value : root["cameras"]) {
            if (!value.is_object() || !value.contains("id") || !value["id"].is_number_unsigned())
                return invalid(resource.resolved_path, "OPF camera list entries require an unsigned integer 'id'.");
            auto uri = required_string(value, "uri", resource.resolved_path, "camera");
            if (!uri)
                return std::unexpected(uri.error());
            const auto id = value["id"].get<std::uint64_t>();
            if (!ids.insert(id).second)
                return invalid(resource.resolved_path, std::format("OPF camera id '{}' is duplicated.", id));
            auto resolved = resolve_image_uri(base, *uri);
            if (!resolved)
                return std::unexpected(resolved.error());
            cameras.push_back({id, *uri, *resolved});
        }
        return cameras;
    }

    Result<std::filesystem::path> resolve_image_uri(const std::filesystem::path& project_root,
                                                    std::string_view uri) {
        if (uri.empty())
            return invalid(project_root, "OPF image URI must not be empty.");
        const auto requested = std::filesystem::u8path(uri);
        if (requested.is_absolute() || requested.has_root_name())
            return invalid(project_root, std::format("OPF image URI '{}' must be relative.", uri));
        const RecursiveFileCache index(project_root);
        const auto result = index.lookup(requested);
        if (result.ambiguous())
            return invalid(project_root, std::format("OPF image URI '{}' is ambiguous.", uri));
        if (!result.found())
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              std::format("OPF image '{}' was not found.", uri), project_root);
        return result.path;
    }

    Result<std::vector<InputSensor>> read_input_cameras(const Resource& resource) {
        if (resource.format != "application/opf-input-cameras+json")
            return invalid(resource.resolved_path, "OPF input cameras resource has an unexpected format.");
        std::ifstream input(resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF input cameras.", resource.resolved_path);
        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF input cameras JSON: {}", error.what()),
                              resource.resolved_path);
        }
        if (!root.is_object())
            return invalid(resource.resolved_path, "OPF input cameras root must be an object.");
        auto format = required_string(root, "format", resource.resolved_path, "input cameras");
        auto version = required_string(root, "version", resource.resolved_path, "input cameras");
        if (!format)
            return std::unexpected(format.error());
        if (!version)
            return std::unexpected(version.error());
        if (*format != "application/opf-input-cameras+json")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Unsupported OPF input cameras format.", resource.resolved_path);
        if (!root.contains("sensors") || !root["sensors"].is_array())
            return invalid(resource.resolved_path, "OPF input cameras requires a 'sensors' array.");

        std::vector<InputSensor> sensors;
        std::unordered_set<std::uint64_t> ids;
        for (const auto& sensor : root["sensors"]) {
            if (!sensor.is_object() || !sensor.contains("id") || !sensor["id"].is_number_unsigned())
                return invalid(resource.resolved_path, "OPF sensors require unsigned integer ids.");
            if (!ids.insert(sensor["id"].get<std::uint64_t>()).second)
                return invalid(resource.resolved_path, "OPF input sensor ids must be unique.");
            if (!sensor.contains("image_size_px") || !sensor["image_size_px"].is_array() ||
                sensor["image_size_px"].size() != 2)
                return invalid(resource.resolved_path, "OPF sensors require image_size_px [width,height].");
            if (!sensor["image_size_px"][0].is_number_unsigned() ||
                !sensor["image_size_px"][1].is_number_unsigned())
                return invalid(resource.resolved_path, "OPF sensor image dimensions must be unsigned integers.");
            const auto internals = sensor.find("internals");
            if (internals == sensor.end() || !internals->is_object())
                return invalid(resource.resolved_path, "OPF sensors require camera internals.");
            auto model = required_string(*internals, "type", resource.resolved_path, "sensor internals");
            if (!model)
                return std::unexpected(model.error());
            if (*model != "perspective" && *model != "fisheye" && *model != "spherical")
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("Unsupported OPF camera distortion model '{}'.", *model),
                                  resource.resolved_path);
            const auto principal = internals->find("principal_point_px");
            if (principal == internals->end() || !principal->is_array() || principal->size() != 2)
                return invalid(resource.resolved_path, "OPF camera internals require principal_point_px [x,y].");

            InputSensor parsed{sensor["id"].get<std::uint64_t>(),
                               sensor.value("name", std::string{}),
                               sensor["image_size_px"][0].get<std::uint32_t>(),
                               sensor["image_size_px"][1].get<std::uint32_t>(),
                               *model,
                               {principal->at(0).get<double>(), principal->at(1).get<double>()}};
            if (*model == "perspective") {
                if (!internals->contains("focal_length_px") || !internals->contains("radial_distortion") ||
                    !internals->contains("tangential_distortion"))
                    return invalid(resource.resolved_path, "OPF perspective internals are incomplete.");
                parsed.focal_length = (*internals)["focal_length_px"].get<double>();
                parsed.radial_distortion = (*internals)["radial_distortion"].get<std::vector<double>>();
                parsed.tangential_distortion = (*internals)["tangential_distortion"].get<std::vector<double>>();
                if (parsed.radial_distortion.size() != 3 || parsed.tangential_distortion.size() != 2)
                    return invalid(resource.resolved_path, "OPF perspective distortion coefficients have invalid sizes.");
            }
            sensors.push_back(std::move(parsed));
        }
        return sensors;
    }

    Result<std::vector<InputCapture>> read_input_captures(const Resource& resource) {
        if (resource.format != "application/opf-input-cameras+json")
            return invalid(resource.resolved_path, "OPF input cameras resource has an unexpected format.");
        std::ifstream input(resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF input cameras.", resource.resolved_path);
        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF input cameras JSON: {}", error.what()),
                              resource.resolved_path);
        }
        if (!root.is_object() || !root.contains("captures") || !root["captures"].is_array())
            return invalid(resource.resolved_path, "OPF input cameras requires a 'captures' array.");

        std::unordered_set<std::uint64_t> sensor_ids;
        if (root.contains("sensors") && root["sensors"].is_array()) {
            for (const auto& sensor : root["sensors"])
                if (sensor.is_object() && sensor.contains("id") && sensor["id"].is_number_unsigned())
                    sensor_ids.insert(sensor["id"].get<std::uint64_t>());
        }
        std::unordered_set<std::uint64_t> capture_ids;
        std::unordered_set<std::uint64_t> camera_ids;
        std::vector<InputCapture> captures;
        for (const auto& capture : root["captures"]) {
            if (!capture.is_object() || !capture.contains("id") || !capture["id"].is_number_unsigned() ||
                !capture.contains("cameras") || !capture["cameras"].is_array() || capture["cameras"].empty())
                return invalid(resource.resolved_path, "OPF captures require a unique id and non-empty cameras array.");
            const auto capture_id = capture["id"].get<std::uint64_t>();
            if (!capture_ids.insert(capture_id).second)
                return invalid(resource.resolved_path, "OPF capture ids must be unique.");
            InputCapture parsed{capture_id, 0, {}};
            for (const auto& camera : capture["cameras"]) {
                if (!camera.is_object() || !camera.contains("id") || !camera["id"].is_number_unsigned() ||
                    !camera.contains("sensor_id") || !camera["sensor_id"].is_number_unsigned())
                    return invalid(resource.resolved_path, "OPF capture cameras require unsigned id and sensor_id.");
                const auto camera_id = camera["id"].get<std::uint64_t>();
                const auto sensor_id = camera["sensor_id"].get<std::uint64_t>();
                if (!sensor_ids.contains(sensor_id))
                    return invalid(resource.resolved_path, "OPF capture references a missing sensor.");
                if (!camera_ids.insert(camera_id).second)
                    return invalid(resource.resolved_path, "OPF camera ids must be unique across captures.");
                parsed.camera_ids.push_back(camera_id);
            }
            if (!capture.contains("reference_camera_id") || !capture["reference_camera_id"].is_number_unsigned())
                return invalid(resource.resolved_path, "OPF capture requires reference_camera_id.");
            parsed.reference_camera_id = capture["reference_camera_id"].get<std::uint64_t>();
            if (!camera_ids.contains(parsed.reference_camera_id))
                return invalid(resource.resolved_path, "OPF capture reference_camera_id is not in its cameras array.");
            captures.push_back(std::move(parsed));
        }
        return captures;
    }

    Result<PointCloudManifest> read_point_cloud_manifest(const Resource& gltf_resource,
                                                         const std::filesystem::path& /*project_root*/) {
        if (gltf_resource.format != "model/gltf+json")
            return invalid(gltf_resource.resolved_path, "OPF point cloud requires a glTF JSON resource.");
        std::ifstream input(gltf_resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF point cloud glTF.", gltf_resource.resolved_path);
        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF point cloud glTF: {}", error.what()),
                              gltf_resource.resolved_path);
        }
        if (!root.is_object() || !root.contains("asset") || !root["asset"].is_object() ||
            root["asset"].value("version", std::string{}) != "2.0")
            return invalid(gltf_resource.resolved_path, "OPF point cloud requires glTF asset version 2.0.");
        if (!root.contains("buffers") || !root["buffers"].is_array())
            return invalid(gltf_resource.resolved_path, "OPF point cloud glTF requires a buffers array.");

        PointCloudManifest manifest{gltf_resource.resolved_path, {}};
        const auto base = gltf_resource.resolved_path.parent_path().lexically_normal();
        for (const auto& buffer : root["buffers"]) {
            if (!buffer.is_object() || !buffer.contains("uri") || !buffer["uri"].is_string())
                return invalid(gltf_resource.resolved_path, "OPF point cloud buffers require relative URI strings.");
            auto resolved = resolve_resource(base, buffer["uri"].get<std::string>(), gltf_resource.resolved_path);
            if (!resolved)
                return std::unexpected(resolved.error());
            manifest.buffer_paths.push_back(*resolved);
        }
        if (!root.contains("accessors") || !root["accessors"].is_array() ||
            !root.contains("bufferViews") || !root["bufferViews"].is_array() ||
            !root.contains("meshes") || !root["meshes"].is_array() || root["meshes"].empty() ||
            !root["meshes"][0].contains("primitives") || root["meshes"][0]["primitives"].empty())
            return manifest;
        const auto& attributes = root["meshes"][0]["primitives"][0]["attributes"];
        if (!attributes.contains("POSITION") || !attributes.contains("COLOR_0"))
            return invalid(gltf_resource.resolved_path, "OPF sparse point cloud requires POSITION and COLOR_0.");
        const auto read_accessor = [&](const json& index) -> Result<void> {
            if (!index.is_number_unsigned() || index.get<size_t>() >= root["accessors"].size())
                return invalid(gltf_resource.resolved_path, "OPF sparse point cloud has an invalid accessor.");
            return {};
        };
        if (auto valid = read_accessor(attributes["POSITION"]); !valid)
            return std::unexpected(valid.error());
        if (auto valid = read_accessor(attributes["COLOR_0"]); !valid)
            return std::unexpected(valid.error());
        const auto& position_accessor = root["accessors"][attributes["POSITION"].get<size_t>()];
        const auto& color_accessor = root["accessors"][attributes["COLOR_0"].get<size_t>()];
        if (position_accessor.value("componentType", 0) != 5126 ||
            position_accessor.value("type", std::string{}) != "VEC3" ||
            color_accessor.value("componentType", 0) != 5121 ||
            color_accessor.value("type", std::string{}) != "VEC4" ||
            position_accessor.value("count", 0u) != color_accessor.value("count", 0u))
            return invalid(gltf_resource.resolved_path, "OPF sparse point cloud uses unsupported accessor types.");
        manifest.point_count = position_accessor.value("count", 0u);
        const auto position_buffer = root["bufferViews"][position_accessor.value("bufferView", 0u)].value("buffer", 0u);
        const auto color_buffer = root["bufferViews"][color_accessor.value("bufferView", 0u)].value("buffer", 0u);
        if (position_buffer >= manifest.buffer_paths.size() || color_buffer >= manifest.buffer_paths.size())
            return invalid(gltf_resource.resolved_path, "OPF sparse point cloud accessor references an invalid buffer.");
        manifest.positions_path = manifest.buffer_paths[position_buffer];
        manifest.colors_path = manifest.buffer_paths[color_buffer];
        if (root.contains("nodes") && root["nodes"].is_array() && !root["nodes"].empty() &&
            root["nodes"][0].contains("matrix") && root["nodes"][0]["matrix"].is_array() &&
            root["nodes"][0]["matrix"].size() == 16) {
            for (size_t i = 0; i < 16; ++i)
                manifest.node_matrix[i] = root["nodes"][0]["matrix"][i].get<float>();
        }
        return manifest;
    }

    Result<lfs::core::PointCloud> read_sparse_point_cloud(const PointCloudManifest& manifest,
                                                          const SceneReferenceFrame* frame) {
        std::ifstream positions(manifest.positions_path, std::ios::binary);
        std::ifstream colors(manifest.colors_path, std::ios::binary);
        if (!positions || !colors)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF sparse point cloud buffers.", manifest.gltf_path);
        std::vector<float> xyz(static_cast<size_t>(manifest.point_count) * 3);
        std::vector<std::uint8_t> rgba(static_cast<size_t>(manifest.point_count) * 4);
        positions.read(reinterpret_cast<char*>(xyz.data()), static_cast<std::streamsize>(xyz.size() * sizeof(float)));
        colors.read(reinterpret_cast<char*>(rgba.data()), static_cast<std::streamsize>(rgba.size()));
        if (!positions || !colors)
            return make_error(ErrorCode::READ_FAILURE, "OPF sparse point cloud buffers are truncated.", manifest.gltf_path);
        std::vector<float> rgb(static_cast<size_t>(manifest.point_count) * 3);
        for (size_t i = 0; i < manifest.point_count; ++i) {
            const float x = xyz[i * 3], y = xyz[i * 3 + 1], z = xyz[i * 3 + 2];
            float p[3] = {manifest.node_matrix[0] * x + manifest.node_matrix[4] * y + manifest.node_matrix[8] * z + manifest.node_matrix[12],
                          manifest.node_matrix[1] * x + manifest.node_matrix[5] * y + manifest.node_matrix[9] * z + manifest.node_matrix[13],
                          manifest.node_matrix[2] * x + manifest.node_matrix[6] * y + manifest.node_matrix[10] * z + manifest.node_matrix[14]};
            // OPF/glTF scene coordinates are visualizer-style (+Y up, +Z back).
            // LichtFeld stores imported datasets in its COLMAP/data world basis;
            // the visualizer boundary flips these axes back exactly once.
            p[1] = -p[1];
            p[2] = -p[2];
            if (frame) {
                if (frame->swap_xy)
                    std::swap(p[0], p[1]);
                for (size_t axis = 0; axis < 3; ++axis)
                    p[axis] = static_cast<float>(p[axis] * frame->scale[axis] + frame->shift[axis]);
            }
            xyz[i * 3] = p[0];
            xyz[i * 3 + 1] = p[1];
            xyz[i * 3 + 2] = p[2];
            rgb[i * 3] = static_cast<float>(rgba[i * 4]) / 255.0f;
            rgb[i * 3 + 1] = static_cast<float>(rgba[i * 4 + 1]) / 255.0f;
            rgb[i * 3 + 2] = static_cast<float>(rgba[i * 4 + 2]) / 255.0f;
        }
        return lfs::core::PointCloud(
            lfs::core::Tensor::from_vector(std::move(xyz), {static_cast<size_t>(manifest.point_count), size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(std::move(rgb), {static_cast<size_t>(manifest.point_count), size_t{3}}, lfs::core::Device::CPU));
    }

    Result<SceneReferenceFrame> read_scene_reference_frame(const Resource& resource) {
        if (resource.format != "application/opf-scene-reference-frame+json")
            return invalid(resource.resolved_path, "OPF scene reference frame has an unexpected format.");
        std::ifstream input(resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF scene reference frame.", resource.resolved_path);
        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF scene reference frame JSON: {}", error.what()),
                              resource.resolved_path);
        }
        if (!root.is_object() || root.value("format", std::string{}) !=
                                     "application/opf-scene-reference-frame+json")
            return invalid(resource.resolved_path, "Unsupported OPF scene reference frame format.");
        const auto it = root.find("base_to_canonical");
        if (it == root.end() || !it->is_object())
            return invalid(resource.resolved_path, "OPF scene reference frame requires base_to_canonical.");
        SceneReferenceFrame frame;
        for (const char* key : {"scale", "shift"}) {
            const auto value = it->find(key);
            if (value == it->end() || !value->is_array() || value->size() != 3)
                return invalid(resource.resolved_path, std::format("OPF scene reference frame requires {}[3].", key));
            for (size_t i = 0; i < 3; ++i) {
                if (!(*value)[i].is_number())
                    return invalid(resource.resolved_path, std::format("OPF scene reference frame {} must be numeric.", key));
                if (std::string_view(key) == "scale")
                    frame.scale[i] = (*value)[i].get<double>();
                else
                    frame.shift[i] = (*value)[i].get<double>();
            }
        }
        if (const auto swap = it->find("swap_xy"); swap != it->end()) {
            if (!swap->is_boolean())
                return invalid(resource.resolved_path, "OPF scene reference frame swap_xy must be boolean.");
            frame.swap_xy = swap->get<bool>();
        }
        if (const auto crs = root.find("crs"); crs != root.end() && crs->is_object())
            frame.crs_definition = crs->value("definition", std::string{});
        return frame;
    }

    void apply_scene_reference_frame(ImportedCamera& camera, const SceneReferenceFrame& frame) {
        auto position = camera.pose.position;
        if (frame.swap_xy)
            std::swap(position[0], position[1]);
        for (size_t i = 0; i < 3; ++i)
            camera.pose.position[i] = static_cast<float>(position[i] * frame.scale[i] + frame.shift[i]);
    }

    void apply_gltf_node_transform(ImportedCamera& camera, const std::array<float, 16>& matrix) {
        const auto position = camera.pose.position;
        camera.pose.position = {
            matrix[0] * position[0] + matrix[4] * position[1] + matrix[8] * position[2] + matrix[12],
            matrix[1] * position[0] + matrix[5] * position[1] + matrix[9] * position[2] + matrix[13],
            matrix[2] * position[0] + matrix[6] * position[1] + matrix[10] * position[2] + matrix[14]};

        const auto rotation = camera.pose.rotation;
        std::array<float, 9> transformed{};
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                transformed[row * 3 + column] =
                    matrix[row + 0 * 4] * rotation[0 * 3 + column] +
                    matrix[row + 1 * 4] * rotation[1 * 3 + column] +
                    matrix[row + 2 * 4] * rotation[2 * 3 + column];
            }
        }
        camera.pose.rotation = transformed;
    }

    void apply_lichtfeld_coordinate_convention(ImportedCamera& camera) {
        // OPF image axes: +X right, +Y up, +Z from scene to camera.
        // LichtFeld dataset axes: +X right, +Y down, +Z camera forward.
        // The same Y/Z flip is also the OPF/glTF world -> LichtFeld data-world
        // basis conversion. For camera-to-world R this is F * R * F.
        camera.pose.position[1] = -camera.pose.position[1];
        camera.pose.position[2] = -camera.pose.position[2];

        auto& rotation = camera.pose.rotation;
        constexpr std::array<float, 3> signs{1.0f, -1.0f, -1.0f};
        for (size_t row = 0; row < 3; ++row)
            for (size_t column = 0; column < 3; ++column)
                rotation[row * 3 + column] *= signs[row] * signs[column];
    }

    Result<std::vector<CalibratedCamera>> read_calibrated_cameras(const Resource& resource) {
        if (resource.format != "application/opf-calibrated-cameras+json")
            return invalid(resource.resolved_path, "OPF calibrated cameras resource has an unexpected format.");
        std::ifstream input(resource.resolved_path, std::ios::binary);
        if (!input)
            return make_error(ErrorCode::READ_FAILURE, "Cannot open OPF calibrated cameras.", resource.resolved_path);
        json root;
        try {
            root = json::parse(input);
        } catch (const json::parse_error& error) {
            return make_error(ErrorCode::MALFORMED_JSON,
                              std::format("Malformed OPF calibrated cameras JSON: {}", error.what()),
                              resource.resolved_path);
        }
        if (!root.is_object() || root.value("format", std::string{}) != "application/opf-calibrated-cameras+json")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT, "Unsupported OPF calibrated cameras format.", resource.resolved_path);
        if (!root.contains("sensors") || !root["sensors"].is_array() ||
            !root.contains("cameras") || !root["cameras"].is_array())
            return invalid(resource.resolved_path, "OPF calibrated cameras requires sensors and cameras arrays.");
        std::unordered_set<std::uint64_t> sensors;
        for (const auto& sensor : root["sensors"]) {
            if (!sensor.is_object() || !sensor.contains("id") || !sensor["id"].is_number_unsigned())
                return invalid(resource.resolved_path, "OPF calibrated sensors require unsigned ids.");
            if (!sensors.insert(sensor["id"].get<std::uint64_t>()).second)
                return invalid(resource.resolved_path, "OPF calibrated sensor ids must be unique.");
        }
        std::unordered_set<std::uint64_t> camera_ids;
        std::vector<CalibratedCamera> cameras;
        for (const auto& camera : root["cameras"]) {
            if (!camera.is_object() || !camera.contains("id") || !camera["id"].is_number_unsigned() ||
                !camera.contains("sensor_id") || !camera["sensor_id"].is_number_unsigned() ||
                !camera.contains("position") || !camera["position"].is_array() || camera["position"].size() != 3 ||
                !camera.contains("orientation_deg") || !camera["orientation_deg"].is_array() ||
                camera["orientation_deg"].size() != 3)
                return invalid(resource.resolved_path, "OPF calibrated camera requires id, sensor_id, position[3] and orientation_deg[3].");
            const auto id = camera["id"].get<std::uint64_t>();
            const auto sensor_id = camera["sensor_id"].get<std::uint64_t>();
            if (!camera_ids.insert(id).second)
                return invalid(resource.resolved_path, "OPF calibrated camera ids must be unique.");
            if (!sensors.contains(sensor_id))
                return invalid(resource.resolved_path, "OPF calibrated camera references a missing sensor.");
            CalibratedCamera parsed{id, sensor_id, {}, {}};
            for (size_t i = 0; i < 3; ++i) {
                if (!camera["position"][i].is_number() || !camera["orientation_deg"][i].is_number())
                    return invalid(resource.resolved_path, "OPF calibrated camera coordinates must be numeric.");
                parsed.position[i] = camera["position"][i].get<double>();
                parsed.orientation_deg[i] = camera["orientation_deg"][i].get<double>();
            }
            cameras.push_back(parsed);
        }
        return cameras;
    }

    CalibratedPose to_calibrated_pose(const CalibratedCamera& camera) {
        constexpr double pi = 3.14159265358979323846;
        const double omega = camera.orientation_deg[0] * pi / 180.0;
        const double phi = camera.orientation_deg[1] * pi / 180.0;
        const double kappa = camera.orientation_deg[2] * pi / 180.0;
        const double co = std::cos(omega), so = std::sin(omega);
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double ck = std::cos(kappa), sk = std::sin(kappa);

        // OPF defines the image-to-processing rotation as Rx(omega)Ry(phi)Rz(kappa).
        const std::array<double, 9> rotation = {
            cp * ck, -cp * sk, sp,
            co * sk + so * sp * ck, co * ck - so * sp * sk, -so * cp,
            so * sk - co * sp * ck, so * ck + co * sp * sk, co * cp};
        return {{static_cast<float>(rotation[0]), static_cast<float>(rotation[1]),
                 static_cast<float>(rotation[2]), static_cast<float>(rotation[3]),
                 static_cast<float>(rotation[4]), static_cast<float>(rotation[5]),
                 static_cast<float>(rotation[6]), static_cast<float>(rotation[7]),
                 static_cast<float>(rotation[8])},
                {static_cast<float>(camera.position[0]), static_cast<float>(camera.position[1]),
                 static_cast<float>(camera.position[2])}};
    }

    Result<std::vector<ImportedCamera>> assemble_cameras(
        const std::vector<CameraImage>& camera_list,
        const std::vector<InputSensor>& sensors,
        const std::vector<CalibratedCamera>& calibrated_cameras) {
        std::unordered_map<std::uint64_t, const CameraImage*> images;
        for (const auto& image : camera_list)
            images.emplace(image.id, &image);
        std::unordered_map<std::uint64_t, const InputSensor*> sensor_map;
        for (const auto& sensor : sensors)
            sensor_map.emplace(sensor.id, &sensor);

        std::vector<ImportedCamera> result;
        result.reserve(calibrated_cameras.size());
        for (const auto& calibrated : calibrated_cameras) {
            const auto image_it = images.find(calibrated.id);
            if (image_it == images.end())
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("OPF calibrated camera '{}' is missing from camera_list.", calibrated.id));
            const auto sensor_it = sensor_map.find(calibrated.sensor_id);
            if (sensor_it == sensor_map.end())
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("OPF calibrated camera '{}' references missing sensor '{}'.",
                                              calibrated.id, calibrated.sensor_id));
            const auto& sensor = *sensor_it->second;
            ImportedCamera imported{calibrated.id,
                                    image_it->second->uri,
                                    sensor.width,
                                    sensor.height,
                                    sensor.model,
                                    sensor.principal_point,
                                    sensor.focal_length,
                                    sensor.radial_distortion,
                                    sensor.tangential_distortion,
                                    to_calibrated_pose(calibrated),
                                    image_it->second->resolved_path};
            result.push_back(std::move(imported));
        }
        return result;
    }

    CameraTransform to_camera_transform(const CalibratedPose& pose) {
        const auto& r = pose.rotation;
        const std::array<float, 9> transpose = {
            r[0], r[3], r[6],
            r[1], r[4], r[7],
            r[2], r[5], r[8]};
        return {transpose,
                {-static_cast<float>(transpose[0] * pose.position[0] +
                                     transpose[1] * pose.position[1] +
                                     transpose[2] * pose.position[2]),
                 -static_cast<float>(transpose[3] * pose.position[0] +
                                     transpose[4] * pose.position[1] +
                                     transpose[5] * pose.position[2]),
                 -static_cast<float>(transpose[6] * pose.position[0] +
                                     transpose[7] * pose.position[1] +
                                     transpose[8] * pose.position[2])}};
    }

    Result<lfs::io::CameraData> to_camera_data(const ImportedCamera& camera) {
        if (camera.model != "perspective")
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("OPF camera model '{}' has no safe LichtFeld mapping yet.", camera.model));
        if (camera.principal_point.size() != 2 || camera.radial_distortion.size() != 3 ||
            camera.tangential_distortion.size() != 2 || camera.focal_length <= 0.0)
            return make_error(ErrorCode::INVALID_DATASET,
                              "OPF perspective camera has incomplete calibrated intrinsics.");

        const auto transform = to_camera_transform(camera.pose);
        lfs::io::CameraData data;
        data._camera_ID = static_cast<std::uint32_t>(camera.id);
        data._R = lfs::core::Tensor::from_vector(
            std::vector<float>(transform.rotation_world_to_camera.begin(),
                               transform.rotation_world_to_camera.end()),
            {3, 3}, lfs::core::Device::CPU);
        data._T = lfs::core::Tensor::from_vector(
            std::vector<float>(transform.translation_world_to_camera.begin(),
                               transform.translation_world_to_camera.end()),
            {3}, lfs::core::Device::CPU);
        data._focal_x = static_cast<float>(camera.focal_length);
        data._focal_y = static_cast<float>(camera.focal_length);
        data._center_x = static_cast<float>(camera.principal_point[0]);
        data._center_y = static_cast<float>(camera.principal_point[1]);
        data._image_name = camera.uri;
        data._image_path = camera.resolved_image_path.empty() ? std::filesystem::path(camera.uri)
                                                              : camera.resolved_image_path;
        data._camera_model_type = lfs::core::CameraModelType::PINHOLE;
        data._width = static_cast<int>(camera.width);
        data._height = static_cast<int>(camera.height);
        data._radial_distortion = lfs::core::Tensor::from_vector(
            std::vector<float>(camera.radial_distortion.begin(), camera.radial_distortion.end()),
            {3}, lfs::core::Device::CPU);
        data._tangential_distortion = lfs::core::Tensor::from_vector(
            std::vector<float>(camera.tangential_distortion.begin(), camera.tangential_distortion.end()),
            {2}, lfs::core::Device::CPU);
        return data;
    }

    Result<std::shared_ptr<lfs::core::Camera>> make_camera(const ImportedCamera& camera) {
        auto data = to_camera_data(camera);
        if (!data)
            return std::unexpected(data.error());
        auto result = std::make_shared<lfs::core::Camera>(
            data->_R,
            data->_T,
            data->_focal_x,
            data->_focal_y,
            data->_center_x,
            data->_center_y,
            data->_radial_distortion,
            data->_tangential_distortion,
            data->_camera_model_type,
            data->_image_name,
            data->_image_path,
            std::filesystem::path{},
            data->_width,
            data->_height,
            static_cast<int>(camera.id),
            static_cast<int>(camera.id));
        return result;
    }
} // namespace lfs::io::opf
