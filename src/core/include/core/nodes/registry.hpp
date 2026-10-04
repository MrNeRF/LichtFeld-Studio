/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/nodes/types.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::nodes {

    class NodeContext;

    struct TreeTypeInfo {
        std::string id;
        std::string label;
        std::string description;
    };

    struct SocketTypeInfo {
        std::string id;
        std::string label;
        std::array<float, 4> color{1, 1, 1, 1};
    };

    struct SocketDecl {
        std::string identifier;
        std::string label;
        std::string type;
        Value default_value;
        std::optional<double> min;
        std::optional<double> max;
        std::optional<double> step;
        bool field = false;
        bool multi_input = false;
        bool hide_value = false;
        std::optional<double> soft_min;
        std::optional<double> soft_max;
        std::string description;

        SocketDecl&& range(double lower, double upper) && {
            min = lower;
            max = upper;
            return std::move(*this);
        }

        SocketDecl&& minimum(double lower) && {
            min = lower;
            return std::move(*this);
        }

        SocketDecl&& soft_range(double lower, double upper) && {
            soft_min = lower;
            soft_max = upper;
            return std::move(*this);
        }

        SocketDecl&& step_size(double value) && {
            step = value;
            return std::move(*this);
        }
    };

    enum class PropertyKind { Enum,
                              String,
                              Int,
                              Float,
                              Bool,
                              Data };

    struct PropertyDecl {
        std::string identifier;
        std::string label;
        PropertyKind kind = PropertyKind::String;
        nlohmann::json default_value;
        std::vector<std::string> items;
        std::optional<double> min;
        std::optional<double> max;
        std::string description;
    };

    struct NodeTypeInfo {
        std::string id;
        std::string label;
        std::string category;
        std::string description;
        std::string help;
        // Empty for plug-ins that supply their own display strings.
        std::string localization_key;
        int version = 1;
        std::vector<std::string> tree_types{"lfs.geometry"};
        std::vector<SocketDecl> inputs;
        std::vector<SocketDecl> outputs;
        std::vector<PropertyDecl> properties;
        std::function<void(NodeContext&)> evaluate;
        std::function<nlohmann::json(nlohmann::json, int)> upgrade;
        bool uses_host = false;
        // Geometry outputs keep the input's elements in their order, so rows still match the source.
        bool keeps_elements = false;
    };

    class LFS_CORE_API TreeTypeRegistry {
    public:
        TreeTypeRegistry();
        bool register_type(TreeTypeInfo info);
        bool unregister_type(std::string_view id);
        [[nodiscard]] std::optional<TreeTypeInfo> find(std::string_view id) const;
        [[nodiscard]] std::vector<TreeTypeInfo> list() const;

    private:
        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, TreeTypeInfo> types_;
    };

    class LFS_CORE_API SocketTypeRegistry {
    public:
        SocketTypeRegistry();
        bool register_type(SocketTypeInfo info);
        bool unregister_type(std::string_view id);
        [[nodiscard]] std::optional<SocketTypeInfo> find(std::string_view id) const;
        [[nodiscard]] std::vector<SocketTypeInfo> list() const;

    private:
        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, SocketTypeInfo> types_;
    };

    class LFS_CORE_API NodeTypeRegistry {
    public:
        bool register_type(NodeTypeInfo info);
        bool unregister_type(std::string_view id);
        [[nodiscard]] std::shared_ptr<const NodeTypeInfo> find(std::string_view id) const;
        [[nodiscard]] std::vector<std::shared_ptr<const NodeTypeInfo>> list() const;
        [[nodiscard]] std::shared_ptr<const NodeTypeInfo> find_localized(std::string_view id) const;
        [[nodiscard]] std::vector<std::shared_ptr<const NodeTypeInfo>> list_localized() const;

    private:
        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, std::shared_ptr<const NodeTypeInfo>> types_;
        mutable std::uint64_t language_generation_ = 0;
        mutable std::unordered_map<std::string, std::shared_ptr<const NodeTypeInfo>> localized_types_;
    };

} // namespace lfs::nodes
