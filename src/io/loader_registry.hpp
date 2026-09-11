/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader_interface.hpp"
#include <algorithm>
#include <memory>
#include <mutex>
#include <ranges>
#include <vector>

namespace lfs::io {

    /**
     * @brief Registry for data loaders
     *
     * Thread-safe registry that manages available loaders and finds
     * the appropriate loader for a given file path.
     */
    class DataLoaderRegistry {
    public:
        DataLoaderRegistry() = default;
        ~DataLoaderRegistry() = default;

        // Delete copy operations
        DataLoaderRegistry(const DataLoaderRegistry&) = delete;
        DataLoaderRegistry& operator=(const DataLoaderRegistry&) = delete;

        // Allow move operations
        DataLoaderRegistry(DataLoaderRegistry&&) = default;
        DataLoaderRegistry& operator=(DataLoaderRegistry&&) = default;

        /**
         * @brief Register a new loader
         * @param loader Unique pointer to loader implementation
         */
        void registerLoader(std::unique_ptr<IDataLoader> loader) {
            if (!loader)
                return;

            std::lock_guard lock(mutex_);
            loaders_.push_back(std::move(loader));

            // Sort by priority (highest first)
            std::ranges::sort(loaders_, [](const auto& a, const auto& b) {
                return a->priority() > b->priority();
            });
        }

        /**
         * @brief Find a loader that can handle the given path
         * @param path File or directory path
         * @return Pointer to loader or nullptr if none found
         */
        IDataLoader* findLoader(const std::filesystem::path& path) const {
            std::lock_guard lock(mutex_);

            // Find first loader that can handle this path
            auto it = std::ranges::find_if(loaders_, [&path](const auto& loader) {
                return loader->canLoad(path);
            });

            return it != loaders_.end() ? it->get() : nullptr;
        }

        /**
         * @brief Get all supported file extensions
         * @return Sorted vector of unique extensions
         */
        std::vector<std::string> getAllSupportedExtensions() const {
            std::lock_guard lock(mutex_);

            std::vector<std::string> extensions;
            for (const auto& loader : loaders_) {
                auto loader_exts = loader->supportedExtensions();
                extensions.insert(extensions.end(),
                                  loader_exts.begin(),
                                  loader_exts.end());
            }

            // Remove duplicates and sort
            std::ranges::sort(extensions);
            auto [first, last] = std::ranges::unique(extensions);
            extensions.erase(first, last);

            return extensions;
        }

        /**
         * @brief Get information about all registered loaders
         */
        struct LoaderInfo {
            std::string name;
            std::vector<std::string> extensions;
            int priority;
        };

        std::vector<LoaderInfo> getLoaderInfo() const {
            std::lock_guard lock(mutex_);

            std::vector<LoaderInfo> info;
            info.reserve(loaders_.size());

            for (const auto& loader : loaders_) {
                info.push_back({.name = loader->name(),
                                .extensions = loader->supportedExtensions(),
                                .priority = loader->priority()});
            }

            return info;
        }

        /**
         * @brief Get number of registered loaders
         */
        size_t size() const {
            std::lock_guard lock(mutex_);
            return loaders_.size();
        }

    private:
        mutable std::mutex mutex_;
        std::vector<std::unique_ptr<IDataLoader>> loaders_;
    };

} // namespace lfs::io