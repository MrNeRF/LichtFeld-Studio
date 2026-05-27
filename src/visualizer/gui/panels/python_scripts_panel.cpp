/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_scripts_panel.hpp"

#include "visualizer/app_store.hpp"

namespace lfs::vis::gui::panels {

    namespace {
        void publishScriptsGeneration(const std::uint64_t generation) {
            if (generation != 0) {
                lfs::vis::app_store().scripts_generation.set(generation);
            }
        }
    } // namespace

    PythonScriptManagerState& PythonScriptManagerState::getInstance() {
        static PythonScriptManagerState instance;
        return instance;
    }

    void PythonScriptManagerState::addScript(const std::filesystem::path& path) {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            for (const auto& s : scripts_) {
                if (s.path == path)
                    return;
            }
            scripts_.push_back({path, true, false, ""});
            generation = ++generation_;
        }
        publishScriptsGeneration(generation);
    }

    void PythonScriptManagerState::setScripts(const std::vector<std::filesystem::path>& paths) {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            scripts_.clear();
            for (const auto& p : paths) {
                scripts_.push_back({p, true, false, ""});
            }
            generation = ++generation_;
        }
        publishScriptsGeneration(generation);
    }

    void PythonScriptManagerState::setScriptEnabled(size_t index, bool enabled) {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            if (index < scripts_.size() && scripts_[index].enabled != enabled) {
                scripts_[index].enabled = enabled;
                generation = ++generation_;
            }
        }
        publishScriptsGeneration(generation);
    }

    void PythonScriptManagerState::setScriptError(size_t index, const std::string& error) {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            if (index < scripts_.size()) {
                const bool has_error = !error.empty();
                if (scripts_[index].has_error != has_error || scripts_[index].error_message != error) {
                    scripts_[index].has_error = has_error;
                    scripts_[index].error_message = error;
                    generation = ++generation_;
                }
            }
        }
        publishScriptsGeneration(generation);
    }

    void PythonScriptManagerState::clearErrors() {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            bool changed = false;
            for (auto& s : scripts_) {
                if (s.has_error || !s.error_message.empty()) {
                    s.has_error = false;
                    s.error_message.clear();
                    changed = true;
                }
            }
            if (changed) {
                generation = ++generation_;
            }
        }
        publishScriptsGeneration(generation);
    }

    void PythonScriptManagerState::clear() {
        std::uint64_t generation = 0;
        {
            std::lock_guard lock(mutex_);
            if (!scripts_.empty() || needs_reload_) {
                generation = ++generation_;
            }
            scripts_.clear();
            needs_reload_ = false;
        }
        publishScriptsGeneration(generation);
    }

    std::vector<ScriptInfo> PythonScriptManagerState::scriptsSnapshot() const {
        std::lock_guard lock(mutex_);
        return scripts_;
    }

    std::vector<std::filesystem::path> PythonScriptManagerState::enabledScripts() const {
        std::lock_guard lock(mutex_);
        std::vector<std::filesystem::path> result;
        for (const auto& s : scripts_) {
            if (s.enabled) {
                result.push_back(s.path);
            }
        }
        return result;
    }

} // namespace lfs::vis::gui::panels
