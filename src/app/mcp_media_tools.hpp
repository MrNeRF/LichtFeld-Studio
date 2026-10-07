/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/event_bridge/event_bridge.hpp"
#include "mcp/mcp_tools.hpp"

#include <string>

namespace lfs::app {
    struct MediaExtractEvent {
        using event_id = MediaExtractEvent;
        std::string phase;
        nlohmann::json data;
        void emit() const { event::emit(*this); }
    };
    void register_media_tools(mcp::ToolRegistry& registry);
    nlohmann::json media_extract_job_snapshot();
    Result<void> cancel_media_extract_job();
    // Requests cancellation and joins the worker before host teardown.
    void shutdown_media_extract_job();
} // namespace lfs::app
