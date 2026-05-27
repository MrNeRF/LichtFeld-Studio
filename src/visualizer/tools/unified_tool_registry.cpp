/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "unified_tool_registry.hpp"
#include "visualizer/app_store.hpp"
#include <algorithm>
#include <cassert>

namespace lfs::vis {

    UnifiedToolRegistry& UnifiedToolRegistry::instance() {
        static UnifiedToolRegistry registry;
        return registry;
    }

    void UnifiedToolRegistry::registerTool(ToolDescriptor desc) {
        assert(!desc.id.empty());
        std::lock_guard lock(mutex_);

        if (std::find(group_order_.begin(), group_order_.end(), desc.group) == group_order_.end()) {
            group_order_.push_back(desc.group);
        }

        tools_[desc.id] = std::move(desc);
    }

    void UnifiedToolRegistry::unregisterTool(const std::string& id) {
        std::lock_guard lock(mutex_);
        tools_.erase(id);

        if (active_tool_id_ == id) {
            active_tool_id_.clear();
        }
    }

    void UnifiedToolRegistry::unregisterAllPython() {
        std::lock_guard lock(mutex_);

        for (auto it = tools_.begin(); it != tools_.end();) {
            if (it->second.source == ToolSource::PYTHON) {
                if (active_tool_id_ == it->first) {
                    active_tool_id_.clear();
                }
                it = tools_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<const ToolDescriptor*> UnifiedToolRegistry::getAllTools() const {
        std::lock_guard lock(mutex_);
        std::vector<const ToolDescriptor*> result;
        result.reserve(tools_.size());

        for (const auto& [id, tool] : tools_) {
            result.push_back(&tool);
        }

        std::sort(result.begin(), result.end(),
                  [](const ToolDescriptor* a, const ToolDescriptor* b) { return a->order < b->order; });

        return result;
    }

    bool UnifiedToolRegistry::poll(const std::string& id) const {
        std::lock_guard lock(mutex_);
        const auto it = tools_.find(id);
        if (it == tools_.end()) {
            return false;
        }

        const auto& tool = it->second;
        return !tool.poll_fn || tool.poll_fn();
    }

    void UnifiedToolRegistry::invoke(const std::string& id) {
        std::function<void()> fn;
        {
            std::lock_guard lock(mutex_);
            const auto it = tools_.find(id);
            if (it != tools_.end() && it->second.invoke_fn) {
                fn = it->second.invoke_fn;
            }
        }
        if (fn) {
            fn();
        }
    }

    void UnifiedToolRegistry::setActiveTool(const std::string& id) {
        std::lock_guard lock(mutex_);
        if (id.empty() || tools_.contains(id)) {
            active_tool_id_ = id;
        }
    }

    const std::string& UnifiedToolRegistry::getActiveTool() const {
        std::lock_guard lock(mutex_);
        return active_tool_id_;
    }

    void UnifiedToolRegistry::clearActiveTool() {
        std::lock_guard lock(mutex_);
        active_tool_id_.clear();
    }

    void UnifiedToolRegistry::setActiveSubmode(const std::string& submode_id) {
        std::lock_guard lock(mutex_);
        active_submode_id_ = submode_id;
        app_store().active_submode.set(active_submode_id_);
    }

    const std::string& UnifiedToolRegistry::getActiveSubmode() const {
        std::lock_guard lock(mutex_);
        return active_submode_id_;
    }

    void UnifiedToolRegistry::clearActiveSubmode() {
        std::lock_guard lock(mutex_);
        active_submode_id_.clear();
        app_store().active_submode.set(std::string{});
    }

} // namespace lfs::vis
