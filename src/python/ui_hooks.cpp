/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ui_hooks.hpp"
#include "core/logger.hpp"
#include "python_runtime.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace lfs::python {

    namespace {
        struct HookEntry {
            UIHookCallback callback;
            HookPosition position;
            size_t id;
        };

        struct HookRegistry {
            std::mutex mutex;
            std::unordered_map<std::string, std::vector<HookEntry>> hooks;
            size_t next_id = 0;

            static HookRegistry& instance() {
                static HookRegistry registry;
                return registry;
            }

            static std::string make_key(const std::string& panel, const std::string& section) {
                return panel + ":" + section;
            }
        };

        std::atomic<PythonHookInvoker> g_hook_invoker{nullptr};
        std::atomic<PythonHookChecker> g_hook_checker{nullptr};
    } // namespace

    void register_ui_hook(const std::string& panel,
                          const std::string& section,
                          UIHookCallback callback,
                          const HookPosition position) {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);
        const auto key = HookRegistry::make_key(panel, section);
        reg.hooks[key].push_back({std::move(callback), position, reg.next_id++});
    }

    void remove_ui_hook(const std::string& panel,
                        const std::string& section,
                        UIHookCallback /*callback*/) {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);
        const auto key = HookRegistry::make_key(panel, section);
        if (const auto it = reg.hooks.find(key); it != reg.hooks.end() && !it->second.empty()) {
            it->second.pop_back();
        }
    }

    void clear_ui_hooks(const std::string& panel, const std::string& section) {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);

        if (section.empty()) {
            const auto prefix = panel + ":";
            std::erase_if(reg.hooks, [&](const auto& pair) { return pair.first.starts_with(prefix); });
        } else {
            reg.hooks.erase(HookRegistry::make_key(panel, section));
        }
    }

    void clear_all_ui_hooks() {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);
        reg.hooks.clear();
    }

    bool has_ui_hooks(const std::string& panel, const std::string& section) {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);
        const auto it = reg.hooks.find(HookRegistry::make_key(panel, section));
        return it != reg.hooks.end() && !it->second.empty();
    }

    void invoke_ui_hooks(const std::string& panel,
                         const std::string& section,
                         const HookPosition position) {
        std::vector<UIHookCallback> to_invoke;

        {
            auto& reg = HookRegistry::instance();
            const std::lock_guard lock(reg.mutex);
            const auto it = reg.hooks.find(HookRegistry::make_key(panel, section));
            if (it == reg.hooks.end())
                return;

            for (const auto& entry : it->second) {
                if (entry.position == position)
                    to_invoke.push_back(entry.callback);
            }
        }

        for (const auto& cb : to_invoke) {
            try {
                cb(nullptr);
            } catch (const std::exception& e) {
                LOG_ERROR("UI hook {}:{} failed: {}", panel, section, e.what());
            }
        }
    }

    std::vector<std::string> get_registered_hook_points() {
        auto& reg = HookRegistry::instance();
        const std::lock_guard lock(reg.mutex);

        std::vector<std::string> points;
        points.reserve(reg.hooks.size());
        for (const auto& [key, entries] : reg.hooks) {
            if (!entries.empty())
                points.push_back(key);
        }
        return points;
    }

    void set_python_hook_invoker(const PythonHookInvoker invoker) {
        g_hook_invoker.store(invoker, std::memory_order_release);
    }

    void set_python_hook_checker(const PythonHookChecker checker) {
        g_hook_checker.store(checker, std::memory_order_release);
    }

    void clear_python_hook_invoker() {
        g_hook_invoker.store(nullptr, std::memory_order_release);
        g_hook_checker.store(nullptr, std::memory_order_release);
    }

    void invoke_python_hooks(const std::string& panel, const std::string& section, const bool prepend) {
        const auto invoker = g_hook_invoker.load(std::memory_order_acquire);
        if (!invoker)
            return;
        if (bridge().prepare_ui)
            bridge().prepare_ui();
        invoker(panel.c_str(), section.c_str(), prepend);
    }

    bool has_python_hooks(const std::string& panel, const std::string& section) {
        const auto checker = g_hook_checker.load(std::memory_order_acquire);
        return checker && checker(panel.c_str(), section.c_str());
    }

} // namespace lfs::python
