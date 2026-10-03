/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/registry.hpp"
#include "core/event_bridge/localization_manager.hpp"

#include <algorithm>
#include <mutex>

namespace lfs::nodes {

    TreeTypeRegistry::TreeTypeRegistry() {
        register_type({"lfs.geometry", "Node Graph", "Non-destructive geometry modifiers"});
    }

    bool TreeTypeRegistry::register_type(TreeTypeInfo info) {
        std::unique_lock lock(mutex_);
        return types_.emplace(info.id, std::move(info)).second;
    }

    bool TreeTypeRegistry::unregister_type(std::string_view id) {
        std::unique_lock lock(mutex_);
        return types_.erase(std::string(id)) != 0;
    }

    std::optional<TreeTypeInfo> TreeTypeRegistry::find(std::string_view id) const {
        std::shared_lock lock(mutex_);
        if (const auto found = types_.find(std::string(id)); found != types_.end())
            return found->second;
        return std::nullopt;
    }

    std::vector<TreeTypeInfo> TreeTypeRegistry::list() const {
        std::shared_lock lock(mutex_);
        std::vector<TreeTypeInfo> result;
        result.reserve(types_.size());
        for (const auto& [_, info] : types_)
            result.push_back(info);
        std::ranges::sort(result, {}, &TreeTypeInfo::id);
        return result;
    }

    SocketTypeRegistry::SocketTypeRegistry() {
        register_type({std::string(GEOMETRY_SOCKET), "Geometry", {0.34f, 0.64f, 0.93f, 1.0f}});
        register_type({std::string(FLOAT_SOCKET), "Float", {0.90f, 0.58f, 0.29f, 1.0f}});
        register_type({std::string(INT_SOCKET), "Integer", {0.38f, 0.62f, 0.80f, 1.0f}});
        register_type({std::string(BOOL_SOCKET), "Boolean", {0.87f, 0.44f, 0.65f, 1.0f}});
        register_type({std::string(VECTOR_SOCKET), "Vector", {0.66f, 0.51f, 0.88f, 1.0f}});
        register_type({std::string(COLOUR_SOCKET), "Colour", {0.91f, 0.76f, 0.31f, 1.0f}});
        register_type({std::string(STRING_SOCKET), "String", {0.45f, 0.70f, 0.45f, 1.0f}});
    }

    bool SocketTypeRegistry::register_type(SocketTypeInfo info) {
        std::unique_lock lock(mutex_);
        return types_.emplace(info.id, std::move(info)).second;
    }

    bool SocketTypeRegistry::unregister_type(std::string_view id) {
        std::unique_lock lock(mutex_);
        return types_.erase(std::string(id)) != 0;
    }

    std::optional<SocketTypeInfo> SocketTypeRegistry::find(std::string_view id) const {
        std::shared_lock lock(mutex_);
        if (const auto found = types_.find(std::string(id)); found != types_.end())
            return found->second;
        return std::nullopt;
    }

    std::vector<SocketTypeInfo> SocketTypeRegistry::list() const {
        std::shared_lock lock(mutex_);
        std::vector<SocketTypeInfo> result;
        result.reserve(types_.size());
        for (const auto& [_, info] : types_)
            result.push_back(info);
        std::ranges::sort(result, {}, &SocketTypeInfo::id);
        return result;
    }

    bool NodeTypeRegistry::register_type(NodeTypeInfo info) {
        if (info.id.empty())
            return false;
        auto value = std::make_shared<const NodeTypeInfo>(std::move(info));
        std::unique_lock lock(mutex_);
        return types_.emplace(value->id, std::move(value)).second;
    }

    bool NodeTypeRegistry::unregister_type(std::string_view id) {
        std::unique_lock lock(mutex_);
        localized_types_.erase(std::string(id));
        return types_.erase(std::string(id)) != 0;
    }

    std::shared_ptr<const NodeTypeInfo> NodeTypeRegistry::find(std::string_view id) const {
        std::shared_lock lock(mutex_);
        if (const auto found = types_.find(std::string(id)); found != types_.end())
            return found->second;
        return {};
    }

    std::vector<std::shared_ptr<const NodeTypeInfo>> NodeTypeRegistry::list() const {
        std::shared_lock lock(mutex_);
        std::vector<std::shared_ptr<const NodeTypeInfo>> result;
        result.reserve(types_.size());
        for (const auto& [_, info] : types_)
            result.push_back(info);
        std::ranges::sort(result, {}, [](const auto& info) -> const std::string& {
            return info->id;
        });
        return result;
    }

    std::shared_ptr<const NodeTypeInfo> NodeTypeRegistry::find_localized(std::string_view id) const {
        auto& locale = event::LocalizationManager::getInstance();
        std::unique_lock lock(mutex_);
        const auto source = types_.find(std::string(id));
        if (source == types_.end())
            return {};
        if (source->second->localization_key.empty())
            return source->second;
        const auto generation = locale.getCurrentLanguageGeneration();
        if (generation != language_generation_) {
            localized_types_.clear();
            language_generation_ = generation;
        }
        if (const auto found = localized_types_.find(source->first); found != localized_types_.end())
            return found->second;
        auto info = std::make_shared<NodeTypeInfo>(*source->second);
        const auto translate = [&](std::string& value, const std::string& suffix) {
            const auto key = info->localization_key + "." + suffix;
            if (locale.hasKey(key))
                value = LOC(key);
        };
        translate(info->label, "label");
        translate(info->description, "description");
        translate(info->help, "help");
        for (auto& socket : info->inputs) {
            translate(socket.label, "input_labels." + socket.identifier);
            translate(socket.description, "inputs." + socket.identifier);
        }
        for (auto& socket : info->outputs) {
            translate(socket.label, "output_labels." + socket.identifier);
            translate(socket.description, "outputs." + socket.identifier);
        }
        for (auto& property : info->properties) {
            translate(property.label, "property_labels." + property.identifier);
            translate(property.description, "properties." + property.identifier);
        }
        localized_types_[source->first] = info;
        return info;
    }

    std::vector<std::shared_ptr<const NodeTypeInfo>> NodeTypeRegistry::list_localized() const {
        auto result = list();
        for (auto& type : result)
            type = find_localized(type->id);
        std::erase(result, nullptr);
        return result;
    }

} // namespace lfs::nodes
