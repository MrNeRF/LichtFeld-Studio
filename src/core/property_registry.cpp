/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/property_registry.hpp"

#include "core/logger.hpp"

#include <stdexcept>

namespace lfs::core::prop {

    PropertyRegistry& PropertyRegistry::instance() {
        static PropertyRegistry registry;
        return registry;
    }

    void PropertyRegistry::register_group(PropertyGroup group) {
        if (group.id == "optimization") {
            for (const auto& property : group.properties) {
                if (!property.strategy_applicability_explicit) {
                    throw std::logic_error("optimization property missing strategy applicability: " + property.id);
                }
            }
        }
        std::lock_guard lock(mutex_);
        groups_[group.id] = std::move(group);
    }

    void PropertyRegistry::unregister_group(const std::string& group_id) {
        std::lock_guard lock(mutex_);

        auto it = groups_.find(group_id);
        if (it == groups_.end()) {
            return;
        }

        for (const auto& prop : it->second.properties) {
            prop_subscribers_.erase({group_id, prop.id});
        }

        groups_.erase(it);
    }

    const PropertyGroup* PropertyRegistry::get_group(const std::string& group_id) const {
        std::lock_guard lock(mutex_);

        auto it = groups_.find(group_id);
        if (it != groups_.end()) {
            return &it->second;
        }
        return nullptr;
    }

    std::optional<PropertyGroup> PropertyRegistry::get_group_snapshot(const std::string& group_id) const {
        std::lock_guard lock(mutex_);

        const auto it = groups_.find(group_id);
        if (it == groups_.end()) {
            return std::nullopt;
        }
        return it->second;
    }

    std::optional<PropertyMeta> PropertyRegistry::get_property(const std::string& group_id,
                                                               const std::string& prop_id) const {
        std::lock_guard lock(mutex_);

        auto git = groups_.find(group_id);
        if (git == groups_.end()) {
            return std::nullopt;
        }

        const auto* meta = git->second.find(prop_id);
        if (!meta) {
            return std::nullopt;
        }

        return *meta;
    }

    void PropertyRegistry::register_operator_args(const std::string& operator_id,
                                                  std::vector<PropertyMeta> args) {
        for (auto& arg : args) {
            arg.flags |= PROP_OPERATOR_ARG;
            arg.default_value = std::nullopt;
        }
        register_group(PropertyGroup{
            .id = "operator." + operator_id,
            .name = operator_id,
            .properties = std::move(args),
        });
    }

    void PropertyRegistry::unregister_operator_args(const std::string& operator_id) {
        unregister_group("operator." + operator_id);
    }

    size_t PropertyRegistry::subscribe(PropertyCallback callback) {
        std::lock_guard lock(mutex_);

        size_t id = next_id_++;
        global_subscribers_[id] = std::move(callback);
        return id;
    }

    size_t PropertyRegistry::subscribe(const std::string& group_id,
                                       const std::string& prop_id,
                                       PropertyCallback callback) {
        std::lock_guard lock(mutex_);

        size_t id = next_id_++;
        prop_subscribers_[{group_id, prop_id}][id] = std::move(callback);
        return id;
    }

    void PropertyRegistry::unsubscribe(size_t id) {
        std::lock_guard lock(mutex_);

        global_subscribers_.erase(id);
        for (auto& [_, subs] : prop_subscribers_) {
            subs.erase(id);
        }
    }

    void PropertyRegistry::notify(const std::string& group_id,
                                  const std::string& prop_id,
                                  const std::any& old_value,
                                  const std::any& new_value) {
        std::vector<PropertyCallback> callbacks;

        {
            std::lock_guard lock(mutex_);

            for (const auto& [_, cb] : global_subscribers_) {
                callbacks.push_back(cb);
            }

            auto it = prop_subscribers_.find({group_id, prop_id});
            if (it != prop_subscribers_.end()) {
                for (const auto& [_, cb] : it->second) {
                    callbacks.push_back(cb);
                }
            }
        }

        for (const auto& cb : callbacks) {
            try {
                cb(group_id, prop_id, old_value, new_value);
            } catch (const std::exception& e) {
                LOG_ERROR("Property callback error for {}.{}: {}", group_id, prop_id, e.what());
            }
        }
    }

} // namespace lfs::core::prop
