/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/error.hpp"
#include "core/event_bridge/localization_manager.hpp"

namespace lfs::core {
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
            return exception.what();
        std::string message(LOC(key));
        const auto details = collisionErrorField(typed->error(), "message_details");
        if (!details.empty())
            message += "\n" + std::string(details);
        return message;
    }
} // namespace lfs::core
