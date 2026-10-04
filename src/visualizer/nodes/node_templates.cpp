/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/nodes/modifier_manager.hpp"

#include "core/event_bridge/localization_manager.hpp"
#include "core/executable_path.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"

#include <algorithm>
#include <format>
#include <fstream>
#include <ranges>

namespace lfs::vis {
    namespace {
        std::filesystem::path builtinDirectory() {
#if defined(LFS_DEV_NODE_TEMPLATE_SOURCE_DIR) && !defined(LFS_MACOS_PORTABLE_APP)
            const auto source = std::filesystem::path(LFS_DEV_NODE_TEMPLATE_SOURCE_DIR);
            if (std::filesystem::is_directory(source))
                return source;
#endif
            return core::getResourceBaseDir() / "node_templates";
        }

        std::expected<std::filesystem::path, ModifierError> userDirectory() {
            const auto paths = core::UserPaths::resolve();
            if (!paths)
                return std::unexpected(ModifierError{std::string(paths.error().user_message())});
            const auto directory = paths->configDir() / "node_templates";
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error)
                return std::unexpected(ModifierError{std::format(
                    "Unable to create template directory '{}': {}",
                    core::path_to_utf8(directory), error.message())});
            return directory;
        }

        std::expected<nlohmann::json, ModifierError> readJson(const std::filesystem::path& path) {
            try {
                std::ifstream stream(path, std::ios::binary);
                if (!stream)
                    return std::unexpected(ModifierError{"Unable to read template file"});
                return nlohmann::json::parse(stream);
            } catch (const std::exception& error) {
                return std::unexpected(ModifierError{std::string("Invalid template JSON: ") + error.what()});
            }
        }

        std::expected<NodeGraphTemplate, ModifierError>
        decode(const nlohmann::json& json, const bool builtin) try {
            if (!json.is_object() || json.value("schema_version", 0) != 1 ||
                !json.contains("tree") || !json["tree"].is_object())
                return std::unexpected(ModifierError{"Template must contain schema_version 1 and a tree object"});
            NodeGraphTemplate result;
            result.id = json.value("id", "");
            result.name = json.value("name", "");
            result.description = json.value("description", "");
            result.category = json.value("category", "Other");
            result.adjust = json.value("adjust", "");
            result.scene_kinds = json.value("scene_kinds", std::vector<std::string>{});
            result.tree = json.at("tree");
            result.builtin = builtin;
            if (result.id.empty() || result.name.empty() || result.description.empty() ||
                result.adjust.empty() || result.scene_kinds.empty())
                return std::unexpected(ModifierError{"Template metadata is incomplete"});
            if (builtin) {
                auto& localization = event::LocalizationManager::getInstance();
                const auto localize = [&](const char* suffix, const std::string& fallback) {
                    const std::string key = "node_template." + result.id + "." + suffix;
                    const std::string translated = localization.get(key);
                    return translated == key ? fallback : translated;
                };
                result.name = localize("name", result.name);
                result.description = localize("description", result.description);
                result.adjust = localize("adjust", result.adjust);
                result.tree["name"] = result.name;
                for (auto& node : result.tree["nodes"])
                    if (node.value("type_id", "") == "lfs.frame") {
                        node["properties"]["label"] = localization.get("node_editor.template_adjust");
                        node["properties"]["note"] = result.adjust;
                    }
            }
            return result;
        } catch (const nlohmann::json::exception& error) {
            return std::unexpected(ModifierError{std::string("Invalid template metadata: ") + error.what()});
        }

        bool userId(const std::string_view id) {
            return id.starts_with("user.") && id.size() == 41 &&
                   core::Uuid::from_string(std::string(id.substr(5))).has_value();
        }

        ModifierResult validateTemplate(const nlohmann::json& json, const nodes::NodeTypeRegistry& registry) try {
            std::unordered_map<std::string, nodes::NodeTree> graphs;
            auto root = nodes::NodeTree::from_json(json, registry);
            graphs.emplace(root.uuid, std::move(root));
            if (json.contains("template_graphs"))
                for (const auto& dependency : json.at("template_graphs")) {
                    auto graph = nodes::NodeTree::from_json(dependency, registry);
                    graphs.emplace(graph.uuid, std::move(graph));
                }
            const nodes::TreeResolver resolver = [&](const std::string_view uuid) -> const nodes::NodeTree* {
                const auto found = graphs.find(std::string(uuid));
                return found == graphs.end() ? nullptr : &found->second;
            };
            for (const auto& [_, graph] : graphs) {
                const auto issues = graph.validate(resolver);
                if (!issues.empty())
                    return std::unexpected(ModifierError{"Invalid template graph: " + issues.front().message});
            }
            return {};
        } catch (const std::exception& error) {
            return std::unexpected(ModifierError{std::string("Invalid template graph: ") + error.what()});
        }

        nlohmann::json encode(const NodeGraphTemplate& value) {
            return {{"schema_version", 1}, {"id", value.id}, {"name", value.name}, {"description", value.description}, {"category", value.category}, {"scene_kinds", value.scene_kinds}, {"adjust", value.adjust}, {"tree", value.tree}};
        }

        std::vector<NodeGraphTemplate> readDirectory(const std::filesystem::path& directory,
                                                     const bool builtin) {
            std::vector<NodeGraphTemplate> result;
            std::error_code error;
            if (!std::filesystem::is_directory(directory, error))
                return result;
            for (std::filesystem::directory_iterator iterator(directory, error), end;
                 !error && iterator != end; iterator.increment(error)) {
                if (iterator->path().extension() != ".json")
                    continue;
                const auto json = readJson(iterator->path());
                if (!json)
                    continue;
                const auto value = decode(*json, builtin);
                if (value)
                    result.push_back(*value);
            }
            return result;
        }
    } // namespace

    std::vector<NodeGraphTemplate> ModifierManager::templates() const {
        auto result = readDirectory(builtinDirectory(), true);
        if (const auto user = userDirectory()) {
            auto values = readDirectory(*user, false);
            result.insert(result.end(), std::make_move_iterator(values.begin()),
                          std::make_move_iterator(values.end()));
        }
        std::ranges::sort(result, [](const auto& left, const auto& right) {
            if (left.builtin != right.builtin)
                return left.builtin > right.builtin;
            if (left.category != right.category)
                return left.category < right.category;
            return left.name < right.name;
        });
        return result;
    }

    std::expected<Modifier*, ModifierError>
    ModifierManager::applyTemplate(const core::Uuid& node_uuid,
                                   const std::string_view template_id, std::string name) {
        const auto values = templates();
        const auto found = std::ranges::find(values, template_id, &NodeGraphTemplate::id);
        if (found == values.end())
            return std::unexpected(ModifierError{"Node graph template does not exist"});
        if (const auto status = validateTemplate(found->tree, registry_); !status)
            return std::unexpected(status.error());
        auto tree_json = found->tree;
        auto dependencies = tree_json.value("template_graphs", nlohmann::json::array());
        tree_json.erase("template_graphs");
        std::unordered_map<std::string, std::string> identities;
        identities[tree_json.at("uuid").get<std::string>()] = core::generate_uuid_v4().to_string();
        for (const auto& dependency : dependencies)
            identities[dependency.at("uuid").get<std::string>()] = core::generate_uuid_v4().to_string();
        const auto freshen = [&](nlohmann::json& graph) {
            graph["uuid"] = identities.at(graph.at("uuid").get<std::string>());
            for (auto& node : graph["nodes"])
                if (node.value("type_id", "") == "lfs.group") {
                    auto& reference = node["properties"]["tree"];
                    reference = identities.at(reference.get<std::string>());
                }
        };
        freshen(tree_json);
        for (auto& dependency : dependencies) {
            freshen(dependency);
            loadTree(dependency);
        }
        tree_json["name"] = uniqueTreeName(found->name);
        auto& graph = loadTree(tree_json);
        auto& modifier = addModifier(node_uuid, graph.uuid,
                                     name.empty() ? found->name : std::move(name));
        return &modifier;
    }

    std::expected<NodeGraphTemplate, ModifierError>
    ModifierManager::saveTemplate(const std::string_view tree_uuid, std::string name,
                                  std::string description, std::string category) {
        const auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        if (name.empty() || description.empty() || category.empty())
            return std::unexpected(ModifierError{"Name, description and category are required"});
        NodeGraphTemplate value{.id = "user." + core::generate_uuid_v4().to_string(),
                                .name = std::move(name),
                                .description = std::move(description),
                                .category = std::move(category),
                                .scene_kinds = {"splat", "mesh", "points"},
                                .adjust = "Adjust the exposed inputs in the modifier panel.",
                                .tree = graph->to_json(),
                                .builtin = false};
        nlohmann::json dependencies = nlohmann::json::array();
        std::unordered_set<std::string> visited{graph->uuid};
        const std::function<ModifierResult(const nodes::NodeTree&)> collect = [&](const nodes::NodeTree& graph) -> ModifierResult {
            for (const auto& node : graph.nodes) {
                if (node.type_id != "lfs.group")
                    continue;
                const auto id = node.properties.value("tree", std::string{});
                if (!visited.insert(id).second)
                    continue;
                const auto* dependency = tree(id);
                if (!dependency)
                    return std::unexpected(ModifierError{"Template references a missing graph"});
                dependencies.push_back(dependency->to_json());
                if (const auto status = collect(*dependency); !status)
                    return status;
            }
            return {};
        };
        if (const auto status = collect(*graph); !status)
            return std::unexpected(status.error());
        if (!dependencies.empty())
            value.tree["template_graphs"] = std::move(dependencies);
        const auto directory = userDirectory();
        if (!directory)
            return std::unexpected(directory.error());
        const auto path = *directory / (value.id + ".json");
        if (const auto status = core::writeTextFileAtomically(path, encode(value).dump(2) + "\n"); !status)
            return std::unexpected(ModifierError{std::string(status.error().user_message())});
        return value;
    }

    ModifierResult ModifierManager::deleteTemplate(const std::string_view template_id) {
        if (!userId(template_id))
            return std::unexpected(ModifierError{"Built-in templates cannot be deleted"});
        const auto directory = userDirectory();
        if (!directory)
            return std::unexpected(directory.error());
        std::error_code error;
        if (!std::filesystem::remove(*directory / (std::string(template_id) + ".json"), error))
            return std::unexpected(ModifierError{error ? error.message() : "Template does not exist"});
        return {};
    }

    ModifierResult ModifierManager::renameTemplate(const std::string_view template_id,
                                                   std::string name) {
        if (!userId(template_id))
            return std::unexpected(ModifierError{"Built-in templates cannot be renamed"});
        if (name.empty())
            return std::unexpected(ModifierError{"Template name is required"});
        const auto directory = userDirectory();
        if (!directory)
            return std::unexpected(directory.error());
        const auto path = *directory / (std::string(template_id) + ".json");
        const auto json = readJson(path);
        if (!json)
            return std::unexpected(json.error());
        auto value = decode(*json, false);
        if (!value)
            return std::unexpected(value.error());
        value->name = std::move(name);
        if (const auto status = core::writeTextFileAtomically(path, encode(*value).dump(2) + "\n"); !status)
            return std::unexpected(ModifierError{std::string(status.error().user_message())});
        return {};
    }

    std::expected<NodeGraphTemplate, ModifierError>
    ModifierManager::importTemplate(const std::filesystem::path& path) {
        const auto json = readJson(path);
        if (!json)
            return std::unexpected(json.error());
        auto value = decode(*json, false);
        if (!value)
            return std::unexpected(value.error());
        value->id = "user." + core::generate_uuid_v4().to_string();
        if (const auto status = validateTemplate(value->tree, registry_); !status)
            return std::unexpected(status.error());
        const auto directory = userDirectory();
        if (!directory)
            return std::unexpected(directory.error());
        if (const auto status = core::writeTextFileAtomically(
                *directory / (value->id + ".json"), encode(*value).dump(2) + "\n");
            !status)
            return std::unexpected(ModifierError{std::string(status.error().user_message())});
        return *value;
    }

    ModifierResult ModifierManager::exportTemplate(const std::string_view template_id,
                                                   const std::filesystem::path& path) const {
        const auto values = templates();
        const auto found = std::ranges::find(values, template_id, &NodeGraphTemplate::id);
        if (found == values.end())
            return std::unexpected(ModifierError{"Node graph template does not exist"});
        if (const auto status = core::writeTextFileAtomically(path, encode(*found).dump(2) + "\n"); !status)
            return std::unexpected(ModifierError{std::string(status.error().user_message())});
        return {};
    }
} // namespace lfs::vis
