/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "icon_cache.hpp"
#include "core/image_io.hpp"
#include "internal/resource_paths.hpp"

#include <cstdint>

namespace lfs::vis::gui {

    namespace {
        constexpr const char* ICON_PREFIX = "icon/";
        constexpr const char* ICON_SUFFIX = ".png";
        constexpr const char* DEFAULT_ICON = "default";
    } // namespace

    IconCache& IconCache::instance() {
        static IconCache cache;
        return cache;
    }

    IconCache::~IconCache() { clear(); }

    std::uintptr_t IconCache::getIcon(const std::string& name) {
        if (name.empty()) {
            return 0;
        }

        {
            std::lock_guard lock(mutex_);
            const auto it = cache_.find(name);
            if (it != cache_.end()) {
                return it->second ? it->second->textureId() : 0;
            }
        }

        auto texture = loadTexture(name);
        if ((!texture || !texture->valid()) && name != DEFAULT_ICON) {
            texture = loadTexture(DEFAULT_ICON);
        }
        const std::uintptr_t texture_id = texture ? texture->textureId() : 0;

        {
            std::lock_guard lock(mutex_);
            cache_[name] = std::move(texture);
        }

        return texture_id;
    }

    void IconCache::clear() {
        std::lock_guard lock(mutex_);
        cache_.clear();
    }

    std::unique_ptr<VulkanUiTexture> IconCache::loadTexture(const std::string& icon_name) {
        std::string path_str = icon_name;
        if (icon_name.find('/') == std::string::npos && icon_name.find('.') == std::string::npos) {
            path_str = std::string(ICON_PREFIX) + icon_name + ICON_SUFFIX;
        }

        const auto path = lfs::vis::getAssetPath(path_str);
        const auto [data, width, height, channels] = lfs::core::load_image_with_alpha(path);
        if (!data) {
            return nullptr;
        }

        auto texture = std::make_unique<VulkanUiTexture>();
        const bool uploaded = texture->upload(
            static_cast<const std::uint8_t*>(data),
            width,
            height,
            channels);
        lfs::core::free_image(data);
        return uploaded ? std::move(texture) : nullptr;
    }

} // namespace lfs::vis::gui
