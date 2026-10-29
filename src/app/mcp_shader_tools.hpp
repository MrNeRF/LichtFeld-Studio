/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <functional>
namespace lfs::mcp {
    class ToolRegistry;
    class ResourceRegistry;
} // namespace lfs::mcp
namespace lfs::vis {
    class Visualizer;
    namespace shaders {
        class ShaderService;
    }
} // namespace lfs::vis
namespace lfs::app {
    using ShaderBackend = std::function<vis::shaders::ShaderService*()>;
    void register_shader_tools(mcp::ToolRegistry&, vis::Visualizer*, ShaderBackend);
    void register_shader_resources(mcp::ResourceRegistry&, vis::Visualizer*, ShaderBackend);
} // namespace lfs::app
