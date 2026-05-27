/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::vis::gui::panels {

    struct ScriptInfo {
        std::filesystem::path path;
        bool enabled = true;
        bool has_error = false;
        std::string error_message;
    };

    // Script manager state (singleton)
    class LFS_VIS_API PythonScriptManagerState {
    public:
        static PythonScriptManagerState& getInstance();

        void addScript(const std::filesystem::path& path);
        void setScripts(const std::vector<std::filesystem::path>& paths);
        void setScriptEnabled(size_t index, bool enabled);
        void setScriptError(size_t index, const std::string& error);
        void clearErrors();
        void clear();

        std::vector<ScriptInfo> scriptsSnapshot() const;
        std::vector<std::filesystem::path> enabledScripts() const;
        bool needsReload() const { return needs_reload_; }
        void setNeedsReload(bool val) { needs_reload_ = val; }

    private:
        PythonScriptManagerState() = default;

        std::vector<ScriptInfo> scripts_;
        std::uint64_t generation_ = 0;
        bool needs_reload_ = false;
        mutable std::mutex mutex_;
    };

} // namespace lfs::vis::gui::panels
