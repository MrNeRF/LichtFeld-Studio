/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "method_session.hpp"

#include <core/export.hpp>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {

    class LFS_VIS_API MethodRegistry {
    public:
        using Factory = std::function<std::unique_ptr<IMethodSession>()>;

        bool register_method(MethodDescriptor descriptor, Factory factory);

        [[nodiscard]] std::expected<std::unique_ptr<IMethodSession>, std::string> create(
            const std::string& id) const;
        [[nodiscard]] bool has(const std::string& id) const;
        [[nodiscard]] std::vector<MethodDescriptor> list() const;
        [[nodiscard]] std::optional<MethodDescriptor> descriptor(const std::string& id) const;

    private:
        struct Entry {
            MethodDescriptor descriptor;
            Factory factory;
        };

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, Entry> registry_;
    };

    LFS_VIS_API void registerConfiguredMethods(MethodRegistry& registry);

} // namespace lfs::vis
