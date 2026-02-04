/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "tool_descriptor.hpp"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::vis {

    class LFS_VIS_API UnifiedToolRegistry {
    public:
        static UnifiedToolRegistry& instance();

        void registerTool(ToolDescriptor desc);
        void unregisterTool(const std::string& id);
        void unregisterAllPython();

        [[nodiscard]] std::vector<const ToolDescriptor*> getAllTools() const;

        [[nodiscard]] bool poll(const std::string& id) const;
        void invoke(const std::string& id);

        void setActiveTool(const std::string& id);
        [[nodiscard]] const std::string& getActiveTool() const;
        void clearActiveTool();

        void setActiveSubmode(const std::string& submode_id);
        [[nodiscard]] const std::string& getActiveSubmode() const;
        void clearActiveSubmode();

    private:
        UnifiedToolRegistry() = default;
        UnifiedToolRegistry(const UnifiedToolRegistry&) = delete;
        UnifiedToolRegistry& operator=(const UnifiedToolRegistry&) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, ToolDescriptor> tools_;
        std::vector<std::string> group_order_;
        std::string active_tool_id_;
        std::string active_submode_id_;
    };

} // namespace lfs::vis
