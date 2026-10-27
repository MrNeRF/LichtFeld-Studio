/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/error.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/export.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
// Tensor-free collision volume types; the .licht reader and format tests build them without the tensor library.
namespace lfs::core {
    struct LFS_CORE_API CollisionSettings {
        float voxel_size = .05f, opacity = .1f, max_gaussian_area = 0;
        bool auto_resolution = false, cleanup = true;
        void validate() const;
    };
    struct LFS_CORE_API CollisionVolume {
        static constexpr uint32_t brick_tag = 0x80000000u;
        static constexpr size_t max_words = brick_tag; // The high pointer bit identifies a brick.
        CollisionSettings settings;
        glm::vec3 origin{0};
        glm::ivec3 dimensions{0};
        int edge = 1;
        uint64_t occupied_count = 0;
        // 0/1 uniform; branch = child base + 2; tagged leaf = two mask words.
        std::vector<uint32_t> words;
        bool occupied(glm::ivec3 cell) const;
        void validate() const;
    };
    inline const char* collisionStage(std::string_view key, const char* diagnostic) {
        return event::LocalizationManager::getInstance().hasKey(key) ? LOC(key) : diagnostic;
    }

    inline Exception collisionError(std::string_view key, std::string diagnostic,
                                    ErrorCode code = ErrorCode::FailedPrecondition, std::string details = {},
                                    SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
        return Exception(make_error({.code = code,
                                     .domain = ErrorDomain::Core,
                                     .user_message = std::move(diagnostic),
                                     .detection = site,
                                     .fields = SmallFields{}.add("message_key", key).add("message_details", std::move(details))}));
    }
    inline std::string_view collisionErrorField(const Error& error, std::string_view name) {
        for (const auto& frame : error.frames())
            for (const auto& field : frame.fields.entries())
                if (field.key == name)
                    if (const auto* value = std::get_if<std::string>(&field.value))
                        return *value;
        return {};
    }
    inline std::string localizedCollisionError(const std::exception& exception) {
        const auto* typed = dynamic_cast<const Exception*>(&exception);
        const auto key = typed ? collisionErrorField(typed->error(), "message_key") : std::string_view{};
        if (key.empty())
            return LOC("selection.operation_failed_generic");
        const auto details = collisionErrorField(typed->error(), "message_details");
        return std::string(LOC(key)) + (details.empty() ? "" : "\n" + std::string(details));
    }
} // namespace lfs::core
