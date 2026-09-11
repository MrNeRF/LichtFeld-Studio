/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include <array>
#include <map>
#include <string>
#include <vector>

namespace lfs::io::usd_flat {

    inline lfs::Error make_flat_error(std::string message) {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::DataLoss,
            .domain = lfs::ErrorDomain::IO,
            .severity = lfs::Severity::Error,
            .retryability = lfs::Retryability::NotRetryable,
            .operation_id = {},
            .user_message = message,
            .detail = std::move(message),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = {},
            .native = std::nullopt,
        });
    }

    struct FlatAttribute {
        std::string type_name;
        std::vector<float> values;
        int components = 1;
        bool authored = false;
    };

    struct FlatPrim {
        std::string path;
        std::string type_name;
        std::map<std::string, FlatAttribute> attributes;
        std::array<double, 16> local_transform{};
        bool reset_xform_stack = false;
    };

    struct FlatStage {
        std::string default_prim;
        std::string up_axis = "Y";
        double meters_per_unit = 1.0;
        std::map<std::string, std::string> custom_layer_data;
        std::vector<FlatPrim> prims;
    };

    inline std::array<double, 16> identity_matrix() {
        return {1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0};
    }

} // namespace lfs::io::usd_flat
