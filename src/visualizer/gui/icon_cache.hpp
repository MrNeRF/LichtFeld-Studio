/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "gui/ui_texture.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace lfs::vis::gui {

    class LFS_VIS_API IconCache {
    public:
        static IconCache& instance();

        std::uintptr_t getIcon(const std::string& name);
        void clear();

    private:
        IconCache() = default;
        ~IconCache();
        IconCache(const IconCache&) = delete;
        IconCache& operator=(const IconCache&) = delete;

        std::unique_ptr<UiTexture> loadTexture(const std::string& icon_name);

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::unique_ptr<UiTexture>> cache_;
    };

} // namespace lfs::vis::gui
