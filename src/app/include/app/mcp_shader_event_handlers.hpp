/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/event_bridge/scoped_handler.hpp"
#include "shaders/shader_api.hpp"

#include <functional>
#include <string>
#include <utility>

namespace lfs::app {

    // Shader events come from the visualizer, so hosts without it (the media
    // contract runner) register only mcp_event_handlers.hpp.
    template <typename PublishFn>
    void register_mcp_shader_event_handlers(event::ScopedHandler& handlers, PublishFn publish) {
        handlers.subscribe<vis::shaders::ShaderEvent>([publish = std::move(publish)](const auto& event) {
            std::invoke(publish, std::string(vis::shaders::shaderEventName(event.kind)),
                        vis::shaders::shaderEvent(event));
        });
    }

} // namespace lfs::app
