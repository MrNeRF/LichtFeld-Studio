/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <nlohmann/json_fwd.hpp>

#include <cstdint>
#include <string>

// The Phase 10 wire error schema (Section 5.10 of
// .codex_tmp/error-architecture-analysis.md): the single, bounded JSON shape
// MCP tool errors, MCP JSON-RPC error.data, TCP response errors, TCP/MCP event
// error_info, and the last_training_error surfaces all cross the wire as. This
// header is declaration-only on purpose — it pulls in nlohmann's forward
// declarations and forward-declares lfs::Error rather than core/error.hpp, so
// broadly-included headers (core/events.hpp) can carry a WireError without
// dragging the host-only rich Error API into any translation unit that reaches
// them.
namespace lfs {
    class Error;
}

namespace lfs::core {

    // Pre-sanitized, POD-string wire form of one lfs::Error. Deliberately NOT
    // lfs::Error: cheap to copy into events/queues, never holds the payload,
    // and safe in broadly-included headers that a CUDA TU may reach.
    struct WireError {
        std::string code;
        std::string domain;
        std::string message;
        std::uint64_t operation_id = 0;
        bool retryable = false;
    };

    LFS_CORE_API void to_json(nlohmann::json& j, const WireError& e);

    // The one sanitization point (Section 1.3/1.4 of the Phase 10 spec). Never
    // emits detail(), format_for_developer() text, frame sources, or any
    // non-allowlisted field: only the stable code/domain token, a safe
    // message, retryability, and the operation id.
    [[nodiscard]] LFS_CORE_API WireError to_wire_error(const Error& error);

    // Full envelope: the five WireError fields plus the allowlisted "details"
    // object. Total serialized size never exceeds kMaxSerializedErrorBytes and
    // the result is always valid JSON (degradation ladder in error_envelope.cpp).
    [[nodiscard]] LFS_CORE_API nlohmann::json to_wire_envelope(const Error& error);

} // namespace lfs::core
