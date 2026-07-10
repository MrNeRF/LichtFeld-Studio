/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "method_registry.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <format>
#include <mutex>
#include <ranges>

namespace lfs::vis {

    namespace {

        [[nodiscard]] bool is_valid_method_id(const std::string& id) {
            return !id.empty() && std::ranges::all_of(id, [](const unsigned char character) {
                return (character >= 'a' && character <= 'z') ||
                       (character >= '0' && character <= '9') || character == '-';
            });
        }

        template <typename Registry>
        [[nodiscard]] std::string available_methods(const Registry& registry) {
            std::vector<std::string> ids{"3dgs"};
            ids.reserve(registry.size() + 1);
            for (const auto& [id, entry] : registry) {
                (void)entry;
                ids.push_back(id);
            }
            std::ranges::sort(ids);
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

            std::string result;
            for (const auto& id : ids) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += id;
            }
            return result;
        }

    } // namespace

    bool MethodRegistry::register_method(MethodDescriptor descriptor, Factory factory) {
        if (!is_valid_method_id(descriptor.id)) {
            LOG_WARN("Method id '{}' is invalid; expected lowercase [a-z0-9-]", descriptor.id);
            return false;
        }
        if (descriptor.id == "3dgs") {
            LOG_WARN("Method id '3dgs' is reserved for the built-in trainer");
            return false;
        }
        if (!factory) {
            LOG_WARN("Method '{}' has no factory", descriptor.id);
            return false;
        }

        const std::string id = descriptor.id;
        std::unique_lock lock(mutex_);
        if (registry_.contains(id)) {
            LOG_WARN("Method '{}' already registered", id);
            return false;
        }
        registry_.emplace(id, Entry{
                                  .descriptor = std::move(descriptor),
                                  .factory = std::move(factory),
                              });
        LOG_DEBUG("Registered training method: {}", id);
        return true;
    }

    std::expected<std::unique_ptr<IMethodSession>, std::string> MethodRegistry::create(
        const std::string& id) const {
        Factory factory;
        {
            std::shared_lock lock(mutex_);
            const auto entry = registry_.find(id);
            if (entry == registry_.end()) {
                return std::unexpected(std::format(
                    "Unknown training method '{}'. Available methods: {}",
                    id,
                    available_methods(registry_)));
            }
            factory = entry->second.factory;
        }

        auto session = factory();
        if (!session) {
            return std::unexpected(std::format("Factory for method '{}' returned null", id));
        }
        return session;
    }

    bool MethodRegistry::has(const std::string& id) const {
        std::shared_lock lock(mutex_);
        return registry_.contains(id);
    }

    std::vector<MethodDescriptor> MethodRegistry::list() const {
        std::shared_lock lock(mutex_);
        std::vector<MethodDescriptor> descriptors;
        descriptors.reserve(registry_.size());
        for (const auto& [id, entry] : registry_) {
            (void)id;
            descriptors.push_back(entry.descriptor);
        }
        std::ranges::sort(descriptors, {}, &MethodDescriptor::id);
        return descriptors;
    }

    std::optional<MethodDescriptor> MethodRegistry::descriptor(const std::string& id) const {
        std::shared_lock lock(mutex_);
        const auto entry = registry_.find(id);
        if (entry == registry_.end()) {
            return std::nullopt;
        }
        return entry->second.descriptor;
    }

} // namespace lfs::vis
