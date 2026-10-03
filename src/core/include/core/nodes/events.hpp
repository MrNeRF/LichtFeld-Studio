/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/event_bridge/event_bridge.hpp"
#include <cstdint>
#include <string>

namespace lfs::nodes {
    // Emitted on the viewer thread, after submission/progress/publication.
    struct EvaluationEvent {
        using event_id = EvaluationEvent;
        std::string phase;
        std::uint64_t generation = 0;
        std::string target;
        std::string node;
        std::string label;
        std::size_t completed = 0;
        std::size_t total = 0;
        bool ok = true;
        void emit() const { lfs::event::emit(*this); }
    };
} // namespace lfs::nodes
