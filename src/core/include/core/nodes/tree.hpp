/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/registry.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lfs::nodes {

    class NodeTree;
    using TreeResolver = std::function<const NodeTree*(std::string_view uuid)>;

    struct Node {
        std::string name;
        std::string type_id;
        std::array<float, 2> location{0, 0};
        bool muted = false;
        int version = 1;
        std::unordered_map<std::string, Value> input_values;
        nlohmann::json properties = nlohmann::json::object();
        nlohmann::json ui = nlohmann::json::object();
        nlohmann::json preserved = nlohmann::json::object();
    };

    struct Link {
        std::string from_node;
        std::string from_socket;
        std::string to_node;
        std::string to_socket;
        bool operator==(const Link&) const = default;
    };

    struct InterfaceSocket {
        std::string identifier;
        std::string label;
        std::string type;
        Value default_value;
        std::optional<double> min;
        std::optional<double> max;
        std::optional<double> step;
    };

    struct TreeInterface {
        std::vector<InterfaceSocket> inputs;
        std::vector<InterfaceSocket> outputs;
    };

    struct ValidationIssue {
        std::string node;
        std::string message;
    };

    class LFS_CORE_API NodeTree {
    public:
        explicit NodeTree(const NodeTypeRegistry& registry, std::string name = "Node Graph",
                          std::string tree_type = "lfs.geometry");

        [[nodiscard]] Node* find_node(std::string_view name);
        [[nodiscard]] const Node* find_node(std::string_view name) const;
        Node& add_node(std::string type_id, std::string name = {});
        bool remove_node(std::string_view name);
        bool rename_node(std::string_view name, std::string new_name);
        bool add_link(Link link, std::string* error = nullptr, const TreeResolver& resolver = {});
        bool remove_link(const Link& link);
        [[nodiscard]] std::vector<ValidationIssue> validate(const TreeResolver& resolver = {}) const;

        [[nodiscard]] nlohmann::json to_json() const;
        static NodeTree from_json(const nlohmann::json& json, const NodeTypeRegistry& registry);

        [[nodiscard]] const NodeTypeRegistry& registry() const {
            return *registry_;
        }
        [[nodiscard]] const Node& input_node() const;
        [[nodiscard]] const Node& output_node() const;

        std::string uuid;
        std::string name;
        std::string tree_type;
        std::vector<Node> nodes;
        std::vector<Link> links;
        TreeInterface group_interface;

    private:
        enum class EmptyTag {};
        NodeTree(const NodeTypeRegistry& registry, EmptyTag);
        const NodeTypeRegistry* registry_ = nullptr;
    };

    // The only socket descriptor path for dynamic nodes. Returned declarations own
    // their strings and remain valid independently of the registry or referenced tree.
    LFS_CORE_API std::vector<SocketDecl> effective_inputs(const NodeTree& tree, const Node& node,
                                                          const TreeResolver& resolver = {});
    LFS_CORE_API std::vector<SocketDecl> effective_outputs(const NodeTree& tree, const Node& node,
                                                           const TreeResolver& resolver = {});
    LFS_CORE_API bool group_reference_would_cycle(const NodeTree& owner, std::string_view referenced_uuid,
                                                  const TreeResolver& resolver,
                                                  std::string* cycle = nullptr);
    LFS_CORE_API bool group_selection_would_cycle(
        const NodeTree& tree, const std::unordered_set<std::string>& selected);

    LFS_CORE_API void to_json(nlohmann::json& json, const Value& value);
    LFS_CORE_API void from_json(const nlohmann::json& json, Value& value);

} // namespace lfs::nodes
