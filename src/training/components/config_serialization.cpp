/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "config_serialization.hpp"

#include "core/error.hpp"

#include <string>
#include <utility>

namespace lfs::training::config_serialization_detail {

    [[noreturn]] void throw_unsupported_component_version(
        const std::string_view component,
        const uint32_t version,
        const uint32_t minimum_version,
        const uint32_t current_version) {
        lfs::SmallFields fields;
        fields.add("component", component)
            .add("version", static_cast<uint64_t>(version))
            .add("minimum_version", static_cast<uint64_t>(minimum_version))
            .add("current_version", static_cast<uint64_t>(current_version));
        throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Unsupported,
            .domain = lfs::ErrorDomain::Training,
            .user_message = "This checkpoint component was written by an unsupported LichtFeld version.",
            .detail = "Unsupported " + std::string(component) + " checkpoint version " +
                      std::to_string(version) + "; supported range is " +
                      std::to_string(minimum_version) + ".." + std::to_string(current_version),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = std::move(fields),
        }));
    }

    [[noreturn]] void throw_config_data_loss(
        const std::string_view field,
        const std::string_view problem) {
        lfs::SmallFields fields;
        fields.add("field", field).add("problem", problem);
        throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::DataLoss,
            .domain = lfs::ErrorDomain::Training,
            .user_message = "The checkpoint contains invalid component configuration data.",
            .detail = std::string(problem) + " serialized " + std::string(field),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = std::move(fields),
        }));
    }

    [[noreturn]] void throw_config_write_failure(const std::string_view field) {
        lfs::SmallFields fields;
        fields.add("field", field);
        throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Internal,
            .domain = lfs::ErrorDomain::Training,
            .user_message = "LichtFeld could not serialize component configuration data.",
            .detail = "Failed to write serialized " + std::string(field),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = std::move(fields),
        }));
    }

} // namespace lfs::training::config_serialization_detail
