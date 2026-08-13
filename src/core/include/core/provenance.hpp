/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <string>

namespace lfs::core {

    struct LFS_CORE_API ProvenanceStamp {
        std::string export_id;    // fresh UUIDv4 per export; empty on a minimal stamp
        int iteration = -1;       // training iteration; -1 = unknown
        std::string strategy;     // canonical optimizer strategy name; empty = unknown
        std::string app_version;  // GIT_TAGGED_VERSION; always set by the factory functions
        std::string build_commit; // GIT_COMMIT_HASH_SHORT; always set by the factory functions
        std::string exported_at;  // ISO-8601 UTC; empty on a minimal stamp
        // Project identity (.licht project format, #1525 / #1507). Empty until the
        // project session wiring lands; serialized only when non-empty.
        std::string project_id;
        std::string commit_id;
        std::string node_id;
        std::string dataset_id;
    };

    // Builds a full stamp with export_id/app_version/build_commit/exported_at
    // filled; iteration, strategy, and identity fields are the caller's to set
    // when known.
    [[nodiscard]] LFS_CORE_API ProvenanceStamp make_provenance_stamp();

    // Software identification only: app_version + build_commit. No export_id,
    // timestamp, iteration, strategy, or identity fields. Used when the user
    // opts out of identifying metadata; formats with a native slot still get
    // this minimal stamp.
    [[nodiscard]] LFS_CORE_API ProvenanceStamp make_minimal_provenance_stamp();

    // Single-line JSON: {"lichtfeld_provenance":1,...}
    // Omits iteration when < 0 and every string field that is empty, except the
    // schema version. A stamp from either factory always yields app_version and
    // build_commit. Field order: lichtfeld_provenance, export_id, project,
    // commit, node, dataset, iteration, strategy, app_version, build_commit,
    // exported_at. No newlines.
    [[nodiscard]] LFS_CORE_API std::string provenance_to_json(const ProvenanceStamp& stamp);

} // namespace lfs::core
