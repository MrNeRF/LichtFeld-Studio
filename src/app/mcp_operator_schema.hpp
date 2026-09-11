/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/property_system.hpp"
#include "mcp/mcp_tools.hpp"

#include <expected>
#include <string>
#include <vector>

namespace lfs::vis::op {
    class OperatorProperties;
}

namespace lfs::app {

    using json = nlohmann::json;

    json property_meta_to_json(const core::prop::PropertyMeta& property);
    mcp::McpToolInputSchema build_operator_input_schema(
        const std::string& operator_key,
        const std::vector<std::string>& required);
    std::expected<void, std::string> assign_operator_property_from_json(
        const json& args,
        const core::prop::PropertyMeta& property,
        vis::op::OperatorProperties& props);
    std::expected<void, std::string> populate_operator_props(
        const json& args,
        const std::string& operator_key,
        const std::vector<std::string>& required,
        vis::op::OperatorProperties& props);

} // namespace lfs::app
