/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "shaders/shader_service.hpp"
#include <nlohmann/json.hpp>

namespace lfs::vis::shaders {
    struct ShaderTool {
        std::string name, description;
        nlohmann::json properties;
        std::vector<std::string> required;
        bool query = false;
    };
    LFS_VIS_API std::vector<ShaderTool> shaderTools();
    LFS_VIS_API nlohmann::json shaderTarget(const ShaderSlot& target);
    LFS_VIS_API Result<ShaderSlot> parseShaderTarget(const nlohmann::json& target);
    LFS_VIS_API nlohmann::json shaderCatalog();
    LFS_VIS_API nlohmann::json shaderState(const ShaderInspection& state);
    LFS_VIS_API nlohmann::json shaderStatus(const core::Uuid& id, const ShaderStatus& status);
    LFS_VIS_API nlohmann::json shaderEvent(const ShaderEvent& event);
    LFS_VIS_API std::string_view shaderEventName(ShaderEvent::Kind kind);
    LFS_VIS_API Result<nlohmann::json> shaderOperation(ShaderService& service, std::string_view operation,
                                                       const nlohmann::json& args);
    LFS_VIS_API nlohmann::json shaderJob(const ShaderInspection& state);
} // namespace lfs::vis::shaders
